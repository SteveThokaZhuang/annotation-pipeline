"""Pipeline V3 adapter for the vendored Qwen FA + MFA implementation.

The upstream implementation stays under ``models/qwen3-fa-mfa/source``.  This
adapter only translates V3 Rover tasks to its row format and translates the
repaired rows back to the result shape consumed by V3 finalization.
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import soundfile as sf

from audio_pipeline.common import ROOT


_SOURCE = ROOT / 'models' / 'qwen3-fa-mfa' / 'source'
sys.path.insert(0, str(_SOURCE / 'qwen3-fa'))
sys.path.insert(1, str(_SOURCE / 'qwen3-fa+mfa'))

from core import QwenAligner  # noqa: E402
from process import audit  # noqa: E402
from repair import SentenceRepair  # noqa: E402
from sentences import sentence_spans  # noqa: E402


class Adapter:
    """One long-lived Qwen aligner with the upstream MFA sentence repair."""

    def __init__(self, job):
        self.job = job
        self.cfg = dict(job.get('settings', {}))
        self.root = ROOT
        self.backend = job.get('backend', 'public')
        self.device = 'cuda:0' if job.get('device') is not None else 'cpu'
        self.aligner = QwenAligner(job['checkpoint'], device=self.device,
                                   backend=self.backend,
                                   language=self.cfg.get('language', 'English'),
                                   tolerance=self.cfg.get('timestamp_tolerance', .08))
        self.mfa_python = self._configured_path('mfa_python', 'models/mfa/bin/python')
        self.dictionary = self._configured_path(
            'dictionary', 'models/mfa-english/dictionary/english_us_arpa.dict')
        self.acoustic_model = self._configured_path(
            'acoustic_model', 'models/mfa-english/english_us_arpa.zip')
        self.mfa_jobs = int(job.get('mfa_jobs', 8))
        self.mfa_timeout = int(job.get('mfa_timeout', 1800))
        self.padding = float(job.get('padding', .35))
        for path in (self.mfa_python, self.dictionary, self.acoustic_model):
            if not path.exists():
                raise FileNotFoundError('Missing MFA dependency: ' + str(path))

    def _configured_path(self, name, default):
        value = self.job.get(name, default)
        path = Path(value)
        return (path if path.is_absolute() else self.root / path).resolve()

    @staticmethod
    def _row(item):
        chunk = item['chunk']
        audio = chunk.get('audio_path') or chunk.get('path')
        if not audio:
            raise ValueError('Alignment chunk has no audio path')
        start = float(chunk.get('start', 0.0))
        end = float(chunk.get('end', start + chunk.get('duration', 0.0)))
        return dict(id=item['id'], audio=str(Path(audio).resolve()), text=item['text'],
                    duration=float(chunk.get('duration', end - start)),
                    chunk_start=start, chunk_end=end,
                    audio_start_sample=chunk.get('audio_start_sample'),
                    audio_end_sample=chunk.get('audio_end_sample'),
                    recording_id=chunk.get('recording_id'), channel=chunk.get('channel'),
                    language=item.get('language'))

    @staticmethod
    def _chunk_audio(row, directory):
        """Return an audio path whose duration matches the alignment chunk.

        V3 manifests may point at a full enhanced recording while carrying the
        chunk's sample bounds separately. Qwen alignment operates on the audio
        path it receives, so pass it an actual chunk-sized file.
        """
        source = Path(row['audio'])
        info = sf.info(source)
        expected = float(row['duration'])
        if abs(float(info.duration) - expected) <= 0.08:
            return row

        start = row.get('audio_start_sample')
        end = row.get('audio_end_sample')
        if start is None:
            start = round(float(row['chunk_start']) * info.samplerate)
        if end is None:
            end = round(float(row['chunk_end']) * info.samplerate)
        start, end = int(start), int(end)
        if not (0 <= start < end <= info.frames):
            raise ValueError(
                f"Invalid audio bounds for {row['id']}: {start}:{end} "
                f"of {info.frames} frames")

        audio, rate = sf.read(source, start=start, stop=end,
                              dtype='float32', always_2d=True)
        actual = len(audio) / rate
        if abs(actual - expected) > 0.08:
            raise ValueError(
                f"Audio duration mismatch for {row['id']}: "
                f"slice={actual:.3f}s metadata={expected:.3f}s")
        target = Path(directory) / f"{row['id']}.wav"
        sf.write(target, audio, rate, subtype='PCM_16')
        prepared = dict(row, audio=str(target), duration=actual)
        return prepared

    def _prediction(self, item, row, words, language):
        # Qwen3-ASR releases have returned both structured mapping objects and
        # tuple-like ``(text, start, end)`` items from the public align call.
        # Normalize both forms at the adapter boundary so the V3 worker always
        # receives the documented word dictionaries.
        normalized = []
        for word in words:
            if isinstance(word, dict):
                normalized.append(self._word_mapping(word))
            elif hasattr(word, 'text') and hasattr(word, 'start_time'):
                normalized.append(dict(text=word.text,
                                       start=float(word.start_time),
                                       end=float(getattr(word, 'end_time'))))
            elif isinstance(word, (tuple, list)):
                # Some qwen-asr builds expose a one-element list or a list of
                # small mapping fragments instead of one mapping object.
                if len(word) == 1:
                    normalized.append(self._word_mapping(word[0]))
                elif all(isinstance(part, dict) for part in word):
                    merged = {}
                    for part in word:
                        merged.update(part)
                    normalized.append(self._word_mapping(merged))
                elif len(word) >= 3:
                    text = next((part for part in word if isinstance(part, str)), '')
                    numbers = [part for part in word if isinstance(part, (int, float))]
                    mappings = [part for part in word if isinstance(part, dict)]
                    merged = {}
                    for part in mappings:
                        merged.update(part)
                    if text and 'text' not in merged:
                        merged['text'] = text
                    if len(numbers) >= 2:
                        merged.setdefault('start', numbers[-2]); merged.setdefault('end', numbers[-1])
                    normalized.append(self._word_mapping(merged))
                else:
                    raise TypeError(f'Unsupported Qwen alignment word type: {type(word).__name__}')
            else:
                raise TypeError(f'Unsupported Qwen alignment word type: {type(word).__name__}')
        value = dict(row, words=normalized, language=language,
                     sentences=sentence_spans(row['text'], language,
                                              self.aligner.model.aligner_processor))
        value['audio'] = row['audio']
        value['source_audio'] = row['audio']
        return value

    @staticmethod
    def _word_mapping(word):
        if not isinstance(word, dict):
            raise TypeError(f'Unsupported Qwen alignment word value: {type(word).__name__}')
        text = word.get('text', word.get('word', ''))
        start = word.get('start', word.get('start_time'))
        end = word.get('end', word.get('end_time'))
        if start is None or end is None:
            raise TypeError('Qwen alignment word has no start/end timestamps')
        return dict(text=str(text), start=float(start), end=float(end))

    def _repair(self, predictions):
        audit(predictions)
        if not any(row.get('issues') or row.get('error') for row in predictions):
            return
        with tempfile.TemporaryDirectory(prefix='qwen3-fa-mfa-',
                                          dir=tempfile.gettempdir()) as directory:
            args = SimpleNamespace(output=Path(directory), padding=self.padding,
                                   mfa_python=self.mfa_python,
                                   dictionary=self.dictionary,
                                   acoustic_model=self.acoustic_model,
                                   mfa_jobs=self.mfa_jobs,
                                   mfa_timeout=self.mfa_timeout)
            SentenceRepair(args, self.aligner).repair(predictions)

    @staticmethod
    def _v3_result(item, prediction):
        words = []
        for word in prediction.get('words', []):
            words.append({key: word[key] for key in ('text', 'start', 'end')})
        value = dict(text=item['text'], chunk=item['chunk'], words=words)
        if prediction.get('language'):
            value['language'] = prediction['language']
        if not item['text'].strip():
            value['empty_transcript'] = True
        return value

    def process(self, item):
        return self.process_batch([item])[0]

    def process_batch(self, items):
        rows = [self._row(item) for item in items]
        # Keep chunk files alive through MFA repair, which re-reads row['audio'].
        with tempfile.TemporaryDirectory(prefix='qwen3-fa-chunks-') as directory:
            rows = [self._chunk_audio(row, directory) for row in rows]
            predictions = [None] * len(items)
            active = [(index, item, row) for index, (item, row) in enumerate(zip(items, rows))
                      if item['text'].strip()]
            if active:
                values = self.aligner.batch([row for _, _, row in active], strict=False)
                for ((words, language), _aligned_row), (index, item, row) in zip(values, active):
                    predictions[index] = self._prediction(item, row, words, language)
            for index, (item, row) in enumerate(zip(items, rows)):
                if predictions[index] is None:
                    predictions[index] = dict(row, words=[], language=item.get('language'),
                                              sentences=[])
            self._repair(predictions)
            return [self._v3_result(item, prediction)
                    for item, prediction in zip(items, predictions)]

    def close(self):
        """Release references before the worker exits."""
        self.aligner = None

    def close_job(self):
        """Keep the aligner resident between persistent-worker bundles."""
        return None
