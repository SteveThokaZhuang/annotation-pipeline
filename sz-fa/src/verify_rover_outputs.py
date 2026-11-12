"""Check completed ROVER exports, source coverage receipts and code snapshots."""
import argparse
import collections
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {'qwen+mfa+nyra': 'qwen3-fa+mfa+nyra',
           'qwen+mfa': 'qwen3-fa+mfa', 'qwen+nyra': 'qwen3-fa+nyra'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/tmp/fa_open_yap_results'))
    parser.add_argument('--persistent', type=Path, default=ROOT / 'results/open_yap_1k_summary')
    args = parser.parse_args()
    preparation = json.loads((args.persistent / 'input_preparation.json').read_text())
    suite = json.loads((args.root / 'suite_summary.json').read_text())
    archives = json.loads((args.persistent / 'predictions/archives.json').read_text())
    results = {result['method']: result for result in suite['experiments']}
    assert set(results) == set(FOLDERS)
    assert len(archives) == 3 and {entry['method'] for entry in archives} == set(results)
    checks = 0
    verified = []
    for entry in archives:
        result = results[entry['method']]
        assert result['samples'] == preparation['unique_chunks'] == entry['samples']
        assert abs(result['audio_seconds'] - preparation['audio_seconds']) < .01
        assert result['qwen_matches_first_experiment']
        assert result['errors']['final']['failed_chunks'] == 0
        assert not result['routes'].get('mfa_failed_blocks')
        receipt = entry['validation']
        assert receipt['status'] == 'passed'
        assert receipt['words'] == result['errors']['final']['words']
        assert receipt['zero_duration_words'] == result['errors']['final']['zero_duration_words']
        validations = list((args.root / FOLDERS[entry['method']]).glob('gpu_*/blocks/*/validation.json'))
        assert len(validations) == receipt['blocks']
        stages = {'mfa', 'nyra'} if entry['method'] == 'qwen+mfa+nyra' else {entry['method'].split('+')[1]}
        for path in validations:
            validation = json.loads(path.read_text())
            assert set(validation) == stages, str(path)
            for check in validation.values():
                assert check['status'] == 'passed', str(path)
                checks += 1
        digest = sha256(Path(entry['path']))
        assert digest == entry['sha256']
        counts = collections.Counter()
        with gzip.open(entry['unresolved_path'], 'rt') as stream:
            for line in stream:
                counts[json.loads(line)['kind']] += 1
        for kind in ['rejected_sentences', 'residual_chunks']:
            assert counts[kind] == entry['unresolved_counts'][kind]
        verified.append(dict(method=entry['method'], samples=entry['samples'], words=receipt['words'],
            zero_words=receipt['zero_duration_words'], sha256=digest, unresolved_counts=dict(counts),
            stored_on_shared_storage=entry['shared_copy_error'] is None and entry['unresolved_shared_copy_error'] is None))
    design = json.loads((args.root / 'experiment_design.json').read_text())
    current_matches = True
    for name, digest in design['code_sha256'].items():
        assert sha256(args.root / 'code_snapshot' / name) == digest
        current_matches = current_matches and (ROOT / name).exists() and sha256(ROOT / name) == digest
    audit = dict(status='passed', completed_at=datetime.now(timezone.utc).isoformat(), experiments=3,
        route_validation_checks=checks, code_snapshot_matches=True,
        current_code_matches_snapshot=current_matches, persistent_archives=verified)
    (args.persistent / 'final_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
