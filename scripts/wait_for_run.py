#!/usr/bin/env python3
"""Wait for one registered pipeline controller to exit and finish successfully."""
import argparse
import json
from pathlib import Path
import sys
import time


def process_identity(pid):
    try:
        # Fields after the parenthesized comm start with state (field 3).
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19], fields[0]
    except (OSError, IndexError):
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    try:
        original = json.loads(args.run.read_text())
        pid, birth = original['pid'], str(original['proc_start'])
        print(f"Waiting for {original['dataset']}: {args.run.stem}, PID {pid}", flush=True)
        last_message = time.monotonic()
        while True:
            current_birth, state = process_identity(pid)
            if current_birth != birth or state == 'Z':
                break
            if time.monotonic() - last_message >= 60:
                print(f"Still waiting for {original['dataset']} (PID {pid}).", flush=True)
                last_message = time.monotonic()
            time.sleep(args.poll_seconds)
        result = json.loads(args.run.read_text())
        if (result['pid'], str(result['proc_start'])) != (pid, birth):
            raise ValueError('Run registration identity changed')
        status = result['status']
        if status not in {'passed', 'completed_with_errors'}:
            raise ValueError(f"Previous run ended with status {status!r}; dataset queue will not start")
        summary = json.loads((Path(result['run_dir']) / 'summary.json').read_text())
        if summary['status'] != status:
            raise ValueError('Run summary and registration status disagree')
        print(f"Previous run finished: {status}. Starting the dataset queue.", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Cannot start dataset queue: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
