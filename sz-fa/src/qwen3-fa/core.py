import json
import math
import time
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def expand_rows(rows):
    """Normalize ASR/ROVER JSONL or recording JSON into chunk-level FA rows."""
    out = []
    for row_index, recording in enumerate(rows):
        if 'audio' in recording and 'text' in recording:
            out.append(dict(recording, id=recording.get('id') or f'row:{row_index}'))
            continue
        chunks = recording.get('chunks')
        if chunks is None:
            raise ValueError('Each input row needs audio+text or chunks: ' + str(recording.get('id', recording.get('recording_channel'))))
        for index, item in enumerate(chunks):
            chunk = item.get('chunk', {})
            text = item.get('text', '').strip()
            audio = chunk.get('audio_path') or chunk.get('path')
            if not audio:
                raise ValueError('Chunk has no audio path')
            start = float(chunk.get('start', 0.0))
            end = float(chunk.get('end', start + float(chunk.get('duration', 0.0))))
            row = dict(id=item.get('id') or chunk.get('id') or f"{recording.get('recording_channel', recording.get('id', 'recording'))}:{index}",
                       audio=audio, text=text, duration=max(0.0, end - start),
                       chunk_start=start, chunk_end=end, recording_id=chunk.get('recording_id', recording.get('recording_channel')),
                       channel=chunk.get('channel'), source_recording=recording.get('recording_channel', recording.get('id')))
            for name in ('language', 'input_language', 'language_verified', 'hypotheses', 'dnsmos',
                         'complex_evidence', 'risk_audio', 'target_isolated'):
                if name in item:
                    row[name] = item[name]
                elif name in chunk:
                    row[name] = chunk[name]
                elif name in ('language', 'input_language', 'language_verified') and name in recording:
                    row[name] = recording[name]
            out.append(row)
    if len({row['id'] for row in out}) != len(out):
        raise ValueError('Duplicate input ids')
    return out


def validate_words(words, duration, tolerance=0.08):
    previous = 0.0
    issues = []
    for i, word in enumerate(words):
        start, end = float(word['start']), float(word['end'])
        if not (math.isfinite(start) and math.isfinite(end)):
            issues.append((i, 'nonfinite'))
        elif not 0 <= start <= end <= duration + tolerance:
            issues.append((i, 'out_of_bounds'))
        elif start < previous - tolerance:
            issues.append((i, 'nonmonotonic'))
        previous = start
    return issues


class QwenAligner:
    def __init__(self, checkpoint, device='cuda:0', backend='public', language='English', dtype='bfloat16', tolerance=0.08):
        import torch
        from qwen_asr import Qwen3ForcedAligner
        self.torch = torch
        self.backend = backend
        self.language_name = language
        self.tolerance = tolerance
        self.model = Qwen3ForcedAligner.from_pretrained(
            str(checkpoint), dtype=getattr(torch, dtype), device_map=device,
            attn_implementation='sdpa', local_files_only=True)
        self.device = device

    def language(self, row):
        language = row.get('language') or self.language_name
        aliases = {'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'}
        language = aliases.get(language, language)
        if language in (None, 'auto'):
            raise ValueError('Alignment requires an explicit language')
        supported = self.model.get_supported_languages()
        if supported and language.casefold() not in {x.casefold() for x in supported}:
            raise ValueError('Unsupported alignment language: ' + language)
        return language

    def _public(self, rows, active):
        return self.model.align(audio=[row['audio'] for row in active], text=[row['text'] for row in active],
                                language=[self.language(row) for row in active])

    def _native(self, rows, active):
        # Mirrors the optimized colleague path. These are private qwen-asr APIs;
        # the public backend remains the correctness reference.
        processor = self.model.aligner_processor
        library = __import__(type(self.model).__module__, fromlist=['normalize_audios'])
        words, texts, audios, languages = [], [], [], []
        for row in active:
            language = self.language(row)
            word_list, text = processor.encode_timestamp(row['text'], language)
            words.append(word_list); texts.append(text); languages.append(language)
            audios.append(row['audio'])
        normalized = library.normalize_audios(audios)
        inputs = self.model.processor(text=texts, audio=normalized, return_tensors='pt', padding=True)
        inputs = inputs.to(self.model.model.device).to(self.model.model.dtype)
        with self.torch.inference_mode():
            output_ids = self.model.model.thinker(**inputs).logits.argmax(dim=-1)
        results = []
        for input_ids, output_id, word_list in zip(inputs['input_ids'], output_ids, words):
            masked = output_id[input_ids == self.model.timestamp_token_id]
            timestamps = (masked * self.model.timestamp_segment_time).to('cpu').numpy()
            parsed = processor.parse_timestamp(word_list, timestamps)
            # qwen-asr's low-level parser returns milliseconds; the public
            # model.align result exposes seconds.
            for item in parsed:
                item['start_time'] = float(item['start_time']) / 1000.0
                item['end_time'] = float(item['end_time']) / 1000.0
            results.append(self.model._to_structured_items(parsed))
        return results

    def batch(self, rows, strict=True):
        active = [row for row in rows if row['text'].strip()]
        if not active:
            return [(([], self.language(row)), row) for row in rows]
        values = self._public(rows, active) if self.backend == 'public' else self._native(rows, active)
        if len(values) != len(active):
            raise RuntimeError('Incomplete alignment batch')
        by_id = {}
        for row, value in zip(active, values):
            words = [dict(text=item.text, start=float(item.start_time), end=float(item.end_time)) for item in value]
            issues = validate_words(words, row['duration'], self.tolerance)
            if issues and strict:
                raise ValueError(f'Invalid timestamps for {row["id"]}: {issues[:3]}')
            if row.get('chunk_start'):
                words = [dict(word, start=word['start'] + row['chunk_start'], end=word['end'] + row['chunk_start']) for word in words]
            by_id[row['id']] = (words, self.language(row))
        return [(by_id.get(row['id'], ([], self.language(row))), row) for row in rows]


def run_component(input_path, output_dir, checkpoint, backend='public', device='cuda:0', batch_size=8, language='English', tolerance=0.08):
    started = time.perf_counter()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    rows = expand_rows(read_jsonl(input_path))
    for row in rows:
        audio = Path(row['audio'])
        if not audio.is_absolute():
            row['audio'] = str((Path(input_path).resolve().parent / audio).resolve())
        if 'duration' not in row:
            import soundfile as sf
            row['duration'] = float(sf.info(row['audio']).duration)
    aligner = QwenAligner(checkpoint, device=device, backend=backend, language=language, tolerance=tolerance)
    results, failures = [], []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        try:
            values = aligner.batch(batch)
            for (words, lang), row in values:
                results.append(dict(id=row['id'], backend=backend, words=words, language=lang,
                                    audio=row['audio'], text=row['text'], chunk_start=row.get('chunk_start'),
                                    chunk_end=row.get('chunk_end'), recording_id=row.get('recording_id'),
                                    channel=row.get('channel')))
        except Exception as exc:
            for row in batch:
                failures.append(dict(id=row['id'], error=repr(exc)))
                results.append(dict(id=row['id'], backend=backend, words=[], error=repr(exc), audio=row['audio'], text=row['text']))
    from prediction_format import write_predictions
    for result, row in zip(results, rows):
        for key, value in row.items():
            result.setdefault(key, value)
        result.update(duration=row['duration'], chunk_start=row.get('chunk_start', 0.0),
                      chunk_end=row.get('chunk_end', row.get('chunk_start', 0.0) + row['duration']))
    serialization = write_predictions(output_dir, results)
    write_jsonl(output_dir / 'failures.jsonl', failures)
    runtime = dict(backend=backend, device=device, batch_size=batch_size, samples=len(rows), failures=len(failures),
                   seconds=time.perf_counter() - started, process_scope='input read + audio preprocessing + FA + validation + output IO')
    write_json(output_dir / 'runtime.json', runtime)
    write_json(output_dir / 'metadata.json', dict(checkpoint=str(checkpoint), backend=backend, language=language,
        timestamp_tolerance=tolerance, input=str(input_path), prediction_format=serialization['format'],
        timestamp_origin='selected audio segment start', audit_timestamp_origin='source recording'))
    return runtime
