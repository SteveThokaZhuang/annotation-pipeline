"""Small, composable detectors. Optional neural models are local and lazy."""
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

from .evidence import RiskEvent, number, union_seconds


def package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return 'unavailable'


def sentence_bounds(row, span):
    words = row['words'][span['start_index']:span['end_index']]
    values = [(w['start'], w['end']) for w in words
              if math.isfinite(w['start']) and math.isfinite(w['end'])]
    if not values:
        return row['chunk_start'], row['chunk_end']
    a = max(row['chunk_start'], min(x[0] for x in values))
    b = min(row['chunk_end'], max(x[1] for x in values))
    return (a, b) if b > a else (row['chunk_start'], row['chunk_end'])


def text_units(text):
    return re.findall(r'[\u3400-\u9fff]|[^\W_]+(?:[\x27’][^\W_]+)*', text.casefold())


def edit_ratio(left, right):
    """Symmetric token edit distance normalized by the longer hypothesis."""
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1] / max(1, len(left), len(right))


class EvidenceDetector:
    name = 'evidence'

    def detect(self, row, evidence, result, audio):
        events = evidence.get('events', [])
        allowed = {'laughter', 'cough', 'breath', 'noise', 'overlap', 'transcript_uncertain'}
        for event in events:
            if event['tag'] not in allowed:
                raise ValueError('Unsupported evidence event: ' + event['tag'])
            score = event.get('score')
            if score is not None and not 0 <= number(score, 'event score') <= 1:
                raise ValueError('Event score must be in [0, 1]')
            details = dict(event.get('details') or {})
            details['detector_version'] = event.get('detector_version', 'unspecified')
            result.events.append(RiskEvent(event['start'], event['end'], event['tag'],
                                           event.get('source', 'sidecar'), score, details))
        for key in ('speakers', 'speech', 'dnsmos', 'hypotheses', 'target_speaker', 'target_isolated'):
            if key in evidence:
                result.observations[key] = evidence[key]
        if 'target_isolated' in evidence and type(evidence['target_isolated']) is not bool:
            raise ValueError('target_isolated must be a boolean')
        if evidence.get('transcript_uncertain'):
            result.events.append(RiskEvent(row['chunk_start'], row['chunk_end'],
                                           'transcript_uncertain', self.name))
        result.availability[self.name] = 'provided' if evidence else 'not_provided'


class TextRiskDetector:
    name = 'text'

    def __init__(self, config):
        self.config = config

    def detect(self, row, evidence, result, audio):
        for span in row.get('sentences', []):
            text = span['text']
            units = text_units(text)
            tags = []
            if any(t in {'um', 'uh', 'erm', '嗯', '呃', '额'} for t in units):
                tags.append('filler')
            if any(a == b for a, b in zip(units, units[1:])):
                tags.append('repetition')
            if re.search(r'\b[^\W_]+-(?=\s|$)', text):
                tags.append('cutoff')
            a, b = sentence_bounds(row, span)
            for tag in tags:
                result.events.append(RiskEvent(a, b, tag, self.name, details={
                    'start_index': span['start_index'], 'end_index': span['end_index'],
                    'heuristic': True}))
        hypotheses = result.observations.get('hypotheses', [])
        texts = [h if isinstance(h, str) else h.get('text', '')
                 for h in hypotheses if isinstance(h, (str, dict))]
        units = [text_units(t) for t in texts]
        if len(units) >= 2:
            # Compare each source with the final transcript too, to detect ROVER cleanup.
            reference = text_units(row['text'])
            pairs = [(units[0], other) for other in units[1:]] + [(reference, u) for u in units]
            score = max(edit_ratio(a, b) for a, b in pairs)
            result.observations['asr_disagreement'] = score
            if score >= self.config.disagreement_threshold:
                result.events.append(RiskEvent(row['chunk_start'], row['chunk_end'],
                    'transcript_uncertain', self.name, score, {'metric': 'normalized_edit_distance'}))
            result.availability['asr_disagreement'] = 'available'
        else:
            result.availability['asr_disagreement'] = 'insufficient_hypotheses'
        result.availability[self.name] = 'available'


class QualityDetector:
    name = 'quality'

    def __init__(self, config):
        self.config = config

    def detect(self, row, evidence, result, audio):
        mos = result.observations.get('dnsmos') or {}
        value = mos.get('BAK', mos.get('bak', mos.get('background')))
        if value is None:
            result.availability[self.name] = 'not_provided'
            return
        value = number(value, 'DNSMOS BAK')
        if not 1 <= value <= 5:
            raise ValueError('DNSMOS BAK must be in [1, 5]')
        result.observations['background_mos'] = value
        if value < self.config.background_mos_threshold:
            result.events.append(RiskEvent(row['chunk_start'], row['chunk_end'], 'noise', self.name,
                details={'background_mos': value, 'not_an_snr_estimate': True}))
        result.availability[self.name] = 'available'


class OverlapDetector:
    name = 'speaker_overlap'

    def __init__(self, config):
        self.config = config

    def detect(self, row, evidence, result, audio):
        speakers = result.observations.get('speakers')
        if speakers is None:
            result.availability[self.name] = 'not_provided'
            return
        changes = {}
        for item in speakers:
            speaker = item.get('speaker')
            if not isinstance(speaker, str) or not speaker:
                raise ValueError('Speaker intervals need nonempty speaker ids')
            changes.setdefault(item['start'], []).append((speaker, 1))
            changes.setdefault(item['end'], []).append((speaker, -1))
        active, previous, segments = {}, None, []
        for time, updates in sorted(changes.items()):
            if previous is not None and len(active) >= 2 and time > previous:
                if segments and abs(segments[-1][1] - previous) < 1e-7:
                    segments[-1] = (segments[-1][0], time)
                else:
                    segments.append((previous, time))
            for speaker, delta in updates:
                active[speaker] = active.get(speaker, 0) + delta
                if not active[speaker]:
                    del active[speaker]
            previous = time
        for a, b in segments:
            if b - a >= self.config.overlap_min_seconds:
                result.events.append(RiskEvent(a, b, 'overlap', self.name))
        result.availability[self.name] = 'available'


class AlignmentRiskDetector:
    name = 'alignment'

    def __init__(self, config):
        self.config = config

    def detect(self, row, evidence, result, audio):
        previous_start = previous_end = row['chunk_start']
        speech = result.observations.get('speech')
        for index, word in enumerate(row['words']):
            a, b = word['start'], word['end']
            reasons = []
            if not (math.isfinite(a) and math.isfinite(b)):
                reasons.append('nonfinite')
            else:
                if b <= a + 1e-7:
                    reasons.append('zero_duration' if abs(b-a) <= 1e-7 else 'negative_duration')
                if a < row['chunk_start'] - 1e-7 or b > row['chunk_end'] + 1e-7:
                    reasons.append('out_of_bounds')
                if a < previous_start - 1e-7 or (index and a < previous_end - 1e-7):
                    reasons.append('nonmonotonic_or_overlap')
                if not reasons and b - a > self.config.long_word_seconds:
                    result.events.append(RiskEvent(a, b, 'alignment_suspicious', self.name,
                        details={'index': index, 'reason': 'long_word'}))
                if not reasons and speech is not None and b - a >= self.config.vad_min_word_seconds:
                    coverage = union_seconds(speech, a, b) / (b - a)
                    if coverage < self.config.vad_min_coverage:
                        result.events.append(RiskEvent(a, b, 'alignment_suspicious', self.name,
                            details={'index': index, 'reason': 'low_vad_coverage', 'coverage': coverage}))
            if reasons:
                result.events.append(RiskEvent(row['chunk_start'], row['chunk_end'], 'structural_error',
                    self.name, details={'index': index, 'reasons': reasons}))
            if math.isfinite(a) and math.isfinite(b):
                previous_start, previous_end = a, b
        if row.get('error') or (row['text'].strip() and not row['words']):
            result.events.append(RiskEvent(row['chunk_start'], row['chunk_end'], 'structural_error',
                self.name, details={'reason': 'inference_failed_or_no_words'}))
        result.availability[self.name] = 'available'


def frame_events(scores, step, offset, tag, source, config):
    """Hysteresis with a trailing moving mean; scores are uncalibrated."""
    step = number(step, 'frame step')
    if step <= 0:
        raise ValueError('Frame step must be positive')
    values = [number(x, 'frame score') for x in scores]
    if any(not 0 <= x <= 1 for x in values):
        raise ValueError('Frame scores must be in [0, 1]')
    width = max(1, round(config.smoothing_seconds / step))
    smooth, running = [], 0.0
    for i, value in enumerate(values):
        running += value
        if i >= width:
            running -= values[i-width]
        smooth.append(running / min(width, i+1))
    spans, start, peak = [], None, 0.0
    for i, value in enumerate(smooth + [0.0]):
        if start is None and value >= config.event_on:
            start, peak = i, value
        elif start is not None:
            peak = max(peak, value)
            if value < config.event_off or i == len(smooth):
                spans.append((start * step, i * step, peak))
                start = None
    merged = []
    for a, b, score in spans:
        if merged and a - merged[-1][1] <= config.event_merge_seconds + 1e-7:
            merged[-1] = (merged[-1][0], b, max(score, merged[-1][2]))
        else:
            merged.append((a, b, score))
    return [RiskEvent(a + offset, b + offset, tag, source, score,
                      {'score_is_calibrated': False}) for a, b, score in merged
            if b - a >= config.event_min_seconds]


class PannsEventDetector:
    """Local PANNs decision-level model; never invoke its auto-download wrapper."""
    name = 'panns'

    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_error = None

    def _load(self):
        if self.load_error:
            raise RuntimeError(self.load_error)
        if self.model is None:
            try:
                import torch
                from panns_inference.config import labels
                from panns_inference.models import Cnn14_DecisionLevelMax
                path = Path(self.config.panns_checkpoint)
                if not path.is_file():
                    raise ValueError('PANNs requires a local decision-level checkpoint')
                model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320,
                    mel_bins=64, fmin=50, fmax=14000, classes_num=len(labels), interpolate_mode='nearest')
                state = torch.load(str(path), map_location='cpu')
                model.load_state_dict(state['model'])
                self.model = model.to(self.config.detector_device).eval()
                self.labels, self.torch = labels, torch
            except Exception as exc:
                self.load_error = repr(exc)
                raise

    def detect(self, row, evidence, result, audio):
        import numpy as np
        from scipy.signal import resample_poly
        self._load()
        waveform, sr = audio()
        divisor = math.gcd(sr, 32000)
        waveform = resample_poly(waveform, 32000//divisor, sr//divisor) if sr != 32000 else waveform
        bins = max(1, math.ceil(len(waveform) / 320))
        mapping = {'Laughter': 'laughter', 'Cough': 'cough', 'Breathing': 'breath',
                   'Music': 'noise', 'White noise': 'noise'}
        columns = [(self.labels.index(label), tag) for label, tag in mapping.items() if label in self.labels]
        scores = np.zeros((bins, len(columns)), dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int32)
        for start in range(0, max(1, len(waveform)), 5 * 32000):
            segment = waveform[start:start+10*32000]
            valid = min(math.ceil(len(segment)/320), bins-start//320)
            padded = np.pad(segment, (0, max(0, 10*32000-len(segment))))
            tensor = self.torch.as_tensor(padded[None, :], dtype=self.torch.float32,
                                          device=self.config.detector_device)
            with self.torch.inference_mode():
                framewise = self.model(input=tensor, mixup_lambda=None)['framewise_output'][0].cpu().numpy()
            if len(framewise) < valid:
                raise ValueError('PANNs returned fewer frames than the audio')
            lo = start // 320
            scores[lo:lo+valid] += framewise[:valid, [i for i, _ in columns]]
            counts[lo:lo+valid] += 1
        scores /= np.maximum(counts[:, None], 1)
        for i, (_, tag) in enumerate(columns):
            for event in frame_events(scores[:, i], .01, row['chunk_start'], tag, self.name, self.config):
                event.end = min(event.end, row['chunk_end'])
                if event.end > event.start:
                    result.events.append(event)
        result.availability[self.name] = 'available'
        result.observations['panns_audio'] = audio.source
        result.observations['panns_model'] = dict(checkpoint=self.config.panns_checkpoint,
                                                 package_version=package_version('panns-inference'))


class PyannoteSpeakerDetector:
    name = 'pyannote'

    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.load_error = None

    def detect(self, row, evidence, result, audio):
        import numpy as np
        import torch
        if self.load_error:
            raise RuntimeError(self.load_error)
        if self.pipeline is None:
            try:
                from pyannote.audio import Pipeline
                path = Path(self.config.pyannote_model)
                if not path.exists():
                    raise ValueError('pyannote_model must be a local pipeline directory/config')
                self.pipeline = Pipeline.from_pretrained(str(path))
                self.pipeline.to(torch.device(self.config.detector_device))
            except Exception as exc:
                self.load_error = repr(exc)
                raise
        waveform, sr = audio()
        value = self.pipeline({'waveform': torch.from_numpy(np.asarray(waveform, dtype=np.float32)[None, :]),
                               'sample_rate': sr})
        annotation = value.speaker_diarization if hasattr(value, 'speaker_diarization') else value
        speakers = []
        # Deliberately never use exclusive_speaker_diarization: it removes overlap.
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            a = max(row['chunk_start'], row['chunk_start'] + segment.start)
            b = min(row['chunk_end'], row['chunk_start'] + segment.end)
            if b > a:
                speakers.append(dict(start=a, end=b, speaker=str(speaker)))
        if 'speakers' not in result.observations:
            result.observations['speakers'] = speakers
        if 'speech' not in result.observations:
            result.observations['speech'] = [dict(start=s['start'], end=s['end']) for s in speakers]
        result.observations['pyannote_audio'] = audio.source
        result.observations['pyannote_model'] = dict(checkpoint=self.config.pyannote_model,
                                                    package_version=package_version('pyannote.audio'))
        result.availability[self.name] = 'available'
