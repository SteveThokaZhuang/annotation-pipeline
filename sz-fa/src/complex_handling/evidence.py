"""Normalize externally computed evidence into recording-time intervals."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class RiskEvent:
    start: float
    end: float
    tag: str
    source: str
    score: object = None
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionResult:
    events: list = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be finite numeric seconds/scores')
    return float(value)


def intervals(items, origin, row):
    if origin not in ('chunk', 'recording'):
        raise ValueError('Temporal evidence needs timestamp_origin=chunk|recording')
    shift = row['chunk_start'] if origin == 'chunk' else 0.0
    output = []
    for item in items:
        a, b = number(item['start'], 'start') + shift, number(item['end'], 'end') + shift
        if b <= a:
            raise ValueError('Evidence interval must have positive duration')
        item = dict(item, start=max(a, row['chunk_start']), end=min(b, row['chunk_end']))
        if item['start'] < item['end']:
            output.append(item)
    return output


def union_seconds(items, start, end):
    total, previous = 0.0, start
    for a, b in sorted((max(start, x['start']), min(end, x['end'])) for x in items):
        total += max(0.0, b - max(a, previous))
        previous = max(previous, b)
    return total


class EvidenceStore:
    """Optional JSONL sidecar, keyed by exact FA row id; duplicate ids are errors."""
    def __init__(self, path=None):
        self.rows = {}
        self.path = str(Path(path).resolve()) if path else None
        if path:
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                    raise ValueError('Each evidence row needs a string id')
                if row['id'] in self.rows:
                    raise ValueError('Duplicate evidence id: ' + row['id'])
                self.rows[row['id']] = row

    def get(self, row):
        embedded = row.get('complex_evidence') or {}
        if not isinstance(embedded, dict):
            raise ValueError('complex_evidence must be an object')
        external = self.rows.get(row['id'])
        if external and embedded:
            raise ValueError('Choose either embedded or sidecar evidence for ' + row['id'])
        evidence = dict(external or embedded)
        temporal = any(key in evidence for key in ('events', 'speakers', 'speech'))
        origin = evidence.get('timestamp_origin')
        if temporal and origin not in ('chunk', 'recording'):
            raise ValueError('Temporal evidence needs explicit timestamp_origin')
        for key in ('events', 'speakers', 'speech'):
            if key in evidence:
                evidence[key] = intervals(evidence[key], origin, row)
        # Native ASR hypotheses may be retained on the canonical manifest.
        for key in ('hypotheses', 'dnsmos', 'target_isolated'):
            if key in row:
                evidence.setdefault(key, row[key])
        return evidence
