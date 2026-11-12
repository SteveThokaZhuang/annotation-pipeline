"""Export existing FA results as one JSON per chunk or per audio file."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'qwen3-fa'))
from core import read_jsonl
from prediction_format import export_json


def predictions(path):
    if path.is_file():
        yield from read_jsonl(path)
    elif (path / 'predictions.jsonl').exists():
        yield from read_jsonl(path / 'predictions.jsonl')
    else:
        blocks = sorted((path / 'blocks').glob('b*/predictions.jsonl'))
        if not blocks:
            raise FileNotFoundError('No normalized predictions found in ' + str(path))
        for block in blocks:
            yield from read_jsonl(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Normalized JSONL or FA result directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', choices=['chunk', 'audio', 'both'], required=True)
    args = parser.parse_args()
    result = export_json(args.output, predictions(args.input), args.unit)
    print(json.dumps(dict(output=str(args.output.resolve()), file_counts=result['file_counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
