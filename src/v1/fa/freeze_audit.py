"""Freeze external audit BEFORE inference; prohibit meetings used for development."""
import hashlib
import json
from pathlib import Path
from common import ROOT,DATASET,load_manifest,write_jsonl

def main():
    out=ROOT/'results/fa'
    source=Path('/mnt/data/L202500293/obed/mfa/datasets/ami-extended-517/manifest.jsonl')
    base=load_manifest();rows=load_manifest(source)
    audit=[r for r in rows if r['meeting'] in {'ES2014a','TS3003a'}]
    assert not ({r['meeting'] for r in audit}&{r['meeting'] for r in base})
    assert not (out/'audit/qwen.jsonl').exists(), 'audit already observed; cannot refreeze'
    write_jsonl(out/'audit_manifest.jsonl',audit)
    metadata=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((out/'policy.json').read_bytes()).hexdigest(),held_out_meetings=sorted({r['meeting'] for r in audit}),samples=len(audit),audio_seconds=sum(r['duration'] for r in audit),reference_words=sum(len(r['words']) for r in audit),note='new meetings; silver reference; no retuning after audit')
    (out/'audit_freeze.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
