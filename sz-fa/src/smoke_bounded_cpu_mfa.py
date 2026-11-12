"""Run a small fresh CPU MFA smoke from a bounded replay's recorded Qwen input."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa+mfa'))
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from repair import SentenceRepair
from process import audit
from core import write_jsonl, read_jsonl
from compare_results import validate_repairs
from rover_worker import AudioReader, resample
from run_rover_mfa_mae import lines, dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':p.error("Set CUDA_VISIBLE_DEVICES=''")
    selected=json.loads((args.replay/'selection.json').read_text())
    inputs=list(lines(args.replay/'inputs/qwen.jsonl.gz'))
    difficult=[r for r in inputs if 'known_intra_midpoint_case' in selected[r['id']]['reasons']]
    ordinary=[r for r in inputs if 'first_zero_chunks' in selected[r['id']]['reasons']][:5]
    base=ordinary+difficult
    if not base:p.error('No recorded test inputs')
    reader=AudioReader()
    def loader(row):return resample(reader.read(row))[:,None],16000
    results={}
    try:
        for scope in ('chunk','sentence'):
            out=args.replay/('live_cpu_'+scope);out.mkdir()
            rows=copy.deepcopy(base);audit(rows)
            write_jsonl(out/'qwen_predictions.jsonl',rows)
            config=SimpleNamespace(output=out,mfa_python=ROOT/'models/mfa/bin/python',
                dictionary=ROOT/'models/mfa-english/dictionary/english_us_arpa.dict',
                acoustic_model=ROOT/'models/mfa-english/english_us_arpa.zip',
                mfa_jobs=2,mfa_timeout=180,padding=.35,mfa_seam_policy='bounded',
                mfa_repair_scope=scope,mfa_sentence_fallback=True,audio_loader=loader)
            repair=SentenceRepair(config,None).repair(rows)
            after=audit(rows);write_jsonl(out/'predictions.jsonl',rows)
            checked=validate_repairs(base,rows,read_jsonl(out/'mfa/repairs.jsonl'),
                json.loads((out/'unresolved.json').read_text()),
                cross_chunk_seams=read_jsonl(out/'mfa/cross_chunk_seams.jsonl'))
            assert repair['mfa_exit_code']==0, repair
            results[scope]=dict(before=audit(base),after=after,repair=repair,validation=checked)
            dump(out/'summary.json',results[scope])
    finally:reader.close()
    resources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (config.dictionary,config.acoustic_model)}
    dump(args.replay/'live_cpu_summary.json',dict(device='cpu',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        command=sys.argv,models_sha256=resources,chunks=len(base),results=results))
    print(json.dumps({k:dict(zero_words=v['after']['zero_duration_words'],validation=v['validation']['status'])
                      for k,v in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
