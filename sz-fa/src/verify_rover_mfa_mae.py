"""Independently recompute the completed streaming MAE from word timestamps."""
import argparse
import array
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/open_yap_1k_mfa_mae'))
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    while True:
        summary = json.loads((args.root/'summary.json').read_text())
        if summary['complete']:
            break
        if not args.wait:
            raise SystemExit('Experiment is incomplete')
        time.sleep(15)
    inventory = json.loads((args.root/'inventory.json').read_text())
    sums = [0., 0.]
    errors = array.array('d')
    count = 0
    chunk_count = 0
    threshold_counts = {str(t):0 for t in [25,50,100,200]}
    files = []
    for task in inventory['batches']:
        batch = Path(task['path'])
        result = json.loads((batch/'summary.json').read_text())
        assert result['mfa_exit_code'] == 0
        n, onset_sum, offset_sum = 0, 0., 0.
        with gzip.open(batch/'per_word.jsonl.gz','rt') as f:
            for line in f:
                row = json.loads(line)
                onset = abs(row['qwen'][0]-row['mfa'][0])*1000
                offset = abs(row['qwen'][1]-row['mfa'][1])*1000
                delta = (onset+offset)/2
                assert math.isclose(delta,row['boundary_mae_ms'],abs_tol=1e-8)
                onset_sum += onset; offset_sum += offset; n += 1
                errors.append(delta)
                for t in threshold_counts:
                    threshold_counts[t] += offset <= int(t)
        assert n == result['totals']['matched_words']
        assert math.isclose(onset_sum,result['totals']['onset_abs_sum_ms'],rel_tol=1e-10)
        assert math.isclose(offset_sum,result['totals']['offset_abs_sum_ms'],rel_tol=1e-10)
        assert (batch/'mfa_predictions.jsonl.gz').is_file()
        chunk_count += result['totals']['input_chunks']
        count += n; sums[0] += onset_sum; sums[1] += offset_sum
        files.append(dict(batch=batch.name,matched_words=n))
    assert chunk_count == inventory['counts']['eligible_chunks']
    assert count == summary['totals']['matched_words']
    expected = summary['metrics']['qwen']
    mae = sum(sums)/(2*count)
    assert math.isclose(mae,expected['boundary_mae_ms'],rel_tol=1e-10)
    assert summary['metrics']['qwen'] == summary['metrics']['qwen+mfa']
    assert summary['metrics']['mfa']['boundary_mae_ms'] == 0
    for t,n in threshold_counts.items():
        assert math.isclose(100*n/count,expected['offset_accuracy_percent'][t],rel_tol=1e-10)
    values = np.frombuffer(errors, dtype=np.float64)
    validation = dict(passed=True,chunks=chunk_count,matched_words=count,
        independently_recomputed_mae_ms=mae,
        boundary_error_percentiles_ms={str(q):float(np.percentile(values,q)) for q in [50,90,95,99]},
        boundary_error_over_200ms_percent=float(100*np.mean(values>200)),
        qwen_word_matching_coverage_percent=100*count/inventory['counts']['eligible_words'],
        mfa_word_matching_coverage_percent=100*count/summary['totals']['mfa_words'],
        batches=files,
        notes=['MFA is a pseudo-reference, not manually annotated ground truth.',
               'All eligible hybrid word dictionaries were verified identical to Qwen before scoring.',
               'No input alignment failure was hidden; MFA failures and unmatched words are reported.',
               'MFA unknown-label count counts literal <unk>/spn labels, not all dictionary OOV pronunciations.',
               'Only zero-duration Qwen chunks are filtered; other timestamp anomalies remain.'])
    (args.root/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    persistent = Path('results/open_yap_1k_summary/mfa_pseudo_reference_mae.json')
    persistent.write_text(json.dumps(dict(summary=summary,validation=validation,
        detailed_results=str(args.root.resolve())),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
