"""Artifact completeness and reference-free inference-contract checks."""
import argparse
import json
from pathlib import Path
from common import ROOT,DATASET,load_manifest,read_jsonl

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=DATASET/'manifest.jsonl');p.add_argument('--results',type=Path,default=ROOT/'results/fa');a=p.parse_args()
    rows=load_manifest(a.manifest);ids={r['id'] for r in rows};assert len(ids)==len(rows)
    summary={}
    for b in ['qwen','mfa','nyra','hybrid']:
        path=a.results/(b+'.jsonl');pp=read_jsonl(path)
        assert len(pp)==len(ids),(b,'wrong prediction count')
        assert {x['id'] for x in pp}==ids,(b,'ID mismatch')
        errors=[x for x in pp if x.get('error')]
        summary[b]=dict(samples=len(pp),execution_errors=len(errors),words=sum(len(x.get('words',[])) for x in pp))
    assert not any(summary[b]['execution_errors'] for b in summary),summary
    (a.results/'validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
