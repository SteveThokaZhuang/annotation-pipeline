"""HLT-FD alignment arrays, with an explicit chunk variant for unsegmented ASR."""
import math
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from core import write_json, write_jsonl

FORMAT = 'hlt-fd-chunk-v1'
SCHEMA = dict(format=FORMAT, annotation_unit='chunk', timestamp_unit='seconds',
    timestamp_origin='start of audio_path[audio_start_sec:audio_end_sec]',
    force_align=dict(timestamps='one [start, end] pair per word', words='model text units, unchanged'),
    turns_status='not_available: no speaker/turn metadata; [] does not mean no speech',
    audit_file='alignment_predictions.jsonl', audit_timestamp_origin='source recording')


def force_align(words, origin=0.0):
    timestamps, texts = [], []
    for word in words:
        start, end = float(word['start']) - origin, float(word['end']) - origin
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError('Nonfinite alignment timestamp')
        # Keep zero/out-of-bounds intervals visible; serialization never repairs data.
        timestamps.append([start, end])
        texts.append(word['text'])
    return dict(timestamps=timestamps, words=texts)


def chunk_prediction(row):
    origin = float(row.get('chunk_start') or 0.0)
    duration = float(row['duration']) if 'duration' in row else float(row['chunk_end']) - origin
    # Production ROVER references full recordings; Candor references physical crops.
    audio_start = origin if row.get('audio_scope') == 'recording' else 0.0
    value = dict(conversation_id=row.get('conversation_id') or row.get('recording_id') or row['id'],
        chunk_id=row['id'], annotation_unit='chunk', audio_path=row['audio'],
        audio_start_sec=audio_start, audio_end_sec=audio_start + duration,
        audio_duration_sec=duration, text=row['text'],
        force_align=force_align(row['words'], origin),
        turns=[], turns_status='not_available',
        source_audio_path=row.get('source_audio', row['audio']),
        recording_start_sec=origin, recording_end_sec=origin + duration,
        alignment_status='failed' if row.get('error') else 'needs_review' if row.get('issues') or
                         (row['text'].strip() and not row['words'])
                         else 'aligned' if 'issues' in row else 'not_audited')
    for field in ['language', 'channel', 'source_sample_rate', 'start_sample', 'end_sample']:
        if field in row:
            value[field] = row[field]
    return value


def write_predictions(directory, rows):
    """Save normalized predictions and lossless internal audit records."""
    directory = Path(directory)
    write_jsonl(directory / 'alignment_predictions.jsonl', rows)
    write_jsonl(directory / 'predictions.jsonl', (chunk_prediction(row) for row in rows))
    write_json(directory / 'annotation_schema.json', SCHEMA)
    return dict(format=FORMAT, samples=len(rows), words=sum(len(row['words']) for row in rows))


def export_filename(label, identity):
    """Stable, readable filenames; IDs and paths cannot escape the export directory."""
    label = re.sub(r'[^\w.-]+', '_', label)[:100].strip('.') or 'annotation'
    return label + '_' + hashlib.sha256(identity.encode()).hexdigest()[:12] + '.json'


def audio_prediction(chunks, duration):
    """Merge normalized chunks into the coordinates of their actual audio file."""
    chunks = sorted(chunks, key=lambda row: (row['audio_start_sec'], row.get('channel') or 0, row['chunk_id']))
    if not chunks or len({row['audio_path'] for row in chunks}) != 1:
        raise ValueError('Expected chunks from exactly one audio file')
    timestamps, words, segments = [], [], []
    for row in chunks:
        offset = row['audio_start_sec']
        alignment = row['force_align']
        if len(alignment['timestamps']) != len(alignment['words']):
            raise ValueError('Timestamp/word count mismatch: ' + row['chunk_id'])
        # Keep every original interval, including zero durations and overlaps.
        timestamps.extend([[a + offset, b + offset] for a, b in alignment['timestamps']])
        words.extend(alignment['words'])
        segment = dict(chunk_id=row['chunk_id'], start=offset, end=row['audio_end_sec'],
                       text=row['text'], word_start_index=len(words) - len(alignment['words']),
                       word_end_index=len(words))
        if 'channel' in row:
            segment['channel'] = row['channel']
        segments.append(segment)
    states = {row['alignment_status'] for row in chunks}
    bad = any(not (0 <= a < b <= duration) for a, b in timestamps)
    bad |= any(timestamps[i][0] < timestamps[i - 1][1] for i in range(1, len(timestamps)))
    bad |= any(row['text'].strip() and not row['force_align']['words'] for row in chunks)
    status = ('failed' if states == {'failed'} else 'needs_review' if bad or states & {'failed', 'needs_review'}
              else 'aligned' if states == {'aligned'} else 'not_audited')
    conversations = list(dict.fromkeys(row['conversation_id'] for row in chunks))
    return dict(conversation_id=conversations[0], conversation_ids=conversations,
        annotation_unit='audio_file', audio_path=chunks[0]['audio_path'],
        audio_start_sec=0., audio_end_sec=duration, audio_duration_sec=duration,
        text=' '.join(row['text'] for row in chunks),
        force_align=dict(timestamps=timestamps, words=words), annotated_segments=segments,
        turns=[], turns_status='not_available', alignment_status=status)


def export_json(directory, normalized_rows, unit, duration_reader=None):
    """Export chunk JSONs, audio-file JSONs, or both without any model inference.

    Input is the normalized predictions.jsonl schema. An audio file is grouped
    by its absolute path, never by conversation ID or the original parent MP3.
    """
    if unit not in ('chunk', 'audio', 'both'):
        raise ValueError('unit must be chunk, audio or both')
    rows = list(normalized_rows)
    if len({row['chunk_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate chunk IDs')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    units = ['chunk', 'audio'] if unit == 'both' else [unit]
    if any((directory / (kind + '_json')).exists() for kind in units):
        raise FileExistsError('Export directory already exists; choose a new output directory')
    index = dict(format=FORMAT, export_unit=unit, chunks=len(rows),
                 words=sum(len(row['force_align']['words']) for row in rows), files=[])
    for kind in units:
        (directory / (kind + '_json')).mkdir()
    if 'chunk' in units:
        for row in rows:
            filename = export_filename(row['chunk_id'], row['chunk_id'])
            relative = 'chunk_json/' + filename
            write_json(directory / relative, row)
            index['files'].append(dict(unit='chunk', path=relative, chunk_id=row['chunk_id'],
                                       audio_path=row['audio_path']))
    if 'audio' in units:
        if duration_reader is None:
            import soundfile as sf
            duration_reader = lambda path: float(sf.info(path).duration)
        groups = defaultdict(list)
        for row in rows:
            groups[str(Path(row['audio_path']).resolve())].append(row)
        for path, chunks in sorted(groups.items()):
            value = audio_prediction(chunks, duration_reader(path))
            relative = 'audio_json/' + export_filename(Path(path).stem, path)
            write_json(directory / relative, value)
            index['files'].append(dict(unit='audio', path=relative, audio_path=path,
                                       chunks=len(chunks), words=len(value['force_align']['words'])))
    index['file_counts'] = {kind: sum(item['unit'] == kind for item in index['files']) for kind in units}
    write_json(directory / 'export_index.json', index)
    return index
