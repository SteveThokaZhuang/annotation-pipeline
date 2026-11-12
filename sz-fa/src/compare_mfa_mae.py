"""Matched-word onset/offset MAE relative to a pure MFA pseudo-reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import unicodedata


def canonical(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def align_words(predicted, reference):
    """Unit-cost Levenshtein; traceback prefers diagonal, deletion, insertion.

    Word identity alone determines correspondence; timestamps never affect it.
    Only exact, nonempty canonical matches are returned as scored pairs.
    """
    a = [canonical(w['text']) for w in predicted]
    b = [canonical(w['text']) for w in reference]
    d = [list(range(len(b) + 1))]
    for i, word in enumerate(a, 1):
        row = [i]
        for j, ref in enumerate(b, 1):
            row.append(min(d[i-1][j-1] + (word != ref), d[i-1][j] + 1, row[j-1] + 1))
        d.append(row)
    i, j = len(a), len(b)
    matches, excluded = [], []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (a[i-1] != b[j-1]):
            exact = bool(a[i-1]) and a[i-1] == b[j-1] and reference[j-1]['text'] not in {'<unk>', 'spn'}
            if exact:
                matches.append((i-1, j-1))
            else:
                excluded.append(dict(predicted_index=i-1, reference_index=j-1,
                                     predicted=predicted[i-1]['text'], reference=reference[j-1]['text']))
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            excluded.append(dict(predicted_index=i-1, predicted=predicted[i-1]['text']))
            i -= 1
        else:
            excluded.append(dict(reference_index=j-1, reference=reference[j-1]['text']))
            j -= 1
    return list(reversed(matches)), list(reversed(excluded)), d[-1][-1]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen', type=Path, default=Path('results/qwen-fa/predictions.jsonl'))
    p.add_argument('--hybrid', type=Path, default=Path('results/qwen3-fa+mfa/predictions.jsonl'))
    p.add_argument('--reference', type=Path, default=Path('results/mfa/alignment_predictions.jsonl'))
    p.add_argument('--output', type=Path, default=Path('results/mfa/mae_comparison'))
    p.add_argument('--skip-zero-chunks', action='store_true',
                   help='Exclude chunks containing any zero-duration Qwen word from all methods')
    args = p.parse_args()
    paths = dict(qwen=args.qwen, hybrid=args.hybrid, mfa=args.reference)
    data = {name: read(path) for name, path in paths.items()}
    ids = [r['id'] for r in data['qwen']]
    assert len(set(ids)) == len(ids)
    for rows in data.values():
        assert [r['id'] for r in rows] == ids, 'Input coverage/order differs'
    excluded_chunks = []
    if args.skip_zero_chunks:
        excluded_chunks = [dict(chunk_id=r['id'], zero_duration_words=sum(
            abs(w['end']-w['start']) <= 1e-7 for w in r['words']))
            for r in data['qwen'] if any(abs(w['end']-w['start']) <= 1e-7 for w in r['words'])]
        excluded_ids = {r['chunk_id'] for r in excluded_chunks}
        data = {k: [r for r in rows if r['id'] not in excluded_ids] for k, rows in data.items()}
        ids = [r['id'] for r in data['qwen']]
    errors = {name: [] for name in data}
    records, exclusions, chunks = [], [], []
    for q, h, m in zip(data['qwen'], data['hybrid'], data['mfa']):
        assert q['text'] == h['text'] == m['text']
        assert q['audio'] == h['audio'] == m['audio']
        assert q['chunk_start'] == h['chunk_start'] == m['chunk_start']
        assert [w['text'] for w in q['words']] == [w['text'] for w in h['words']]
        assert not any(r.get('error') for r in (q, h, m))
        pairs, skipped, distance = align_words(q['words'], m['words'])
        exclusions.extend(dict(chunk_id=q['id'], **item) for item in skipped)
        chunk_errors = {name: [] for name in data}
        for i, j in pairs:
            ref = m['words'][j]
            for name, row, index in [('qwen', q, i), ('hybrid', h, i), ('mfa', m, j)]:
                w = row['words'][index]
                onset = abs(w['start'] - ref['start']) * 1000
                offset = abs(w['end'] - ref['end']) * 1000
                delta = (onset + offset) / 2
                errors[name].append((onset, offset, delta))
                chunk_errors[name].append(delta)
                records.append(dict(method=name, chunk_id=q['id'], predicted_index=index,
                    reference_index=j, predicted_word=w['text'], reference_word=ref['text'],
                    predicted_start=w['start'], predicted_end=w['end'],
                    reference_start=ref['start'], reference_end=ref['end'],
                    onset_abs_error_ms=onset, offset_abs_error_ms=offset, boundary_mae_ms=delta,
                    predicted_source=w.get('source', name)))
        chunks.append(dict(chunk_id=q['id'], matched_words=len(pairs), edit_distance=distance,
                           qwen_words=len(q['words']), mfa_words=len(m['words']),
                           mae_ms={k: statistics.mean(v) if v else None for k, v in chunk_errors.items()}))
    metrics = {}
    for name, values in errors.items():
        assert values, 'No matched words'
        metrics[name] = dict(matched_words=len(values),
            onset_mae_ms=statistics.mean(v[0] for v in values),
            offset_mae_ms=statistics.mean(v[1] for v in values),
            boundary_mae_ms=statistics.mean(v[2] for v in values),
            offset_accuracy_percent={str(t): 100 * sum(v[1] <= t for v in values) / len(values)
                                     for t in [25, 50, 100, 200]})
    count = len(errors['qwen'])
    summary = dict(reference='pure MFA pseudo-reference, not manual ground truth',
        paper='https://arxiv.org/html/2610.22583v1#S4',
        metric='1000/(2N) * sum(abs(pred_start-ref_start)+abs(pred_end-ref_end))',
        aggregation='micro average over a common set of matched words',
        chunk_filter='exclude any Qwen zero-duration chunk (epsilon 1e-7 seconds) from all methods'
                     if args.skip_zero_chunks else 'none; zero-duration words included',
        excluded_zero_chunks=len(excluded_chunks),
        matching='NFKC + casefold + alphanumeric canonicalization; unit-cost Levenshtein; exact matches only; MFA OOV excluded; no timestamp-based matching or clipping',
        chunks=len(ids), nonempty_chunks=sum(bool(r['text'].strip()) for r in data['mfa']),
        matched_words=count,
        coverage={k: dict(total_words=sum(len(r['words']) for r in rows),
                         scored_words=count, excluded_words=sum(len(r['words']) for r in rows)-count,
                         scored_percent=100*count/sum(len(r['words']) for r in rows))
                  for k, rows in data.items()},
        inputs={k: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for k, path in paths.items()}, metrics=metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    for name, rows in [('excluded_words', exclusions), ('per_chunk', chunks),
                       ('excluded_zero_chunks', excluded_chunks)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    with (args.output/'per_word.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
