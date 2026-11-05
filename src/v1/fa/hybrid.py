"""Run Qwen first; only invoke the chosen fallback on flagged utterances.

Usage: python hybrid.py --manifest ... --output ... --policy results/fa/policy.json
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,load_manifest,read_jsonl,write_jsonl
from router import route,choose

def main():
    process_start=time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'results/fa/policy.json')
    p.add_argument('--qwen-python',default=os.environ.get('QWEN_PYTHON','/mnt/data/L202500293/obed/envs/qwenasr/bin/python'))
    p.add_argument('--mfa-python',default=str(ROOT/'models/fa-runtime/mfa/bin/python'))
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(a.manifest);policy=json.loads(a.policy.read_text());here=Path(__file__).parent
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    env['PYTHONPATH']=str(ROOT/'models/fa-runtime/nyra')+os.pathsep+env.get('PYTHONPATH','')
    stages=[]
    def run(cmd):
        start=time.perf_counter()
        result=subprocess.run(cmd,check=False,env=env)
        stages.append(dict(command=cmd,seconds=time.perf_counter()-start,returncode=result.returncode))
        result.check_returncode()
    run([a.qwen_python,str(here/'run_aligners.py'),'--backend','qwen','--manifest',str(a.manifest),'--output',str(a.output/'qwen.jsonl'),'--device',a.device])
    preds={'qwen':{p['id']:p for p in read_jsonl(a.output/'qwen.jsonl')}}
    for b in ['nyra','mfa']:
        # Strip reference words/times/condition/meeting from fallback input.
        selected=[dict(id=r['id'],audio=r['audio'],duration=r['duration'],text=r['text'],speaker=r.get('speaker',r['id']),language=r.get('language','English')) for r in rows if route(r['text'],preds['qwen'][r['id']].get('words',[]),r['duration'],policy)[0]==b]
        if not selected: preds[b]={};continue
        manifest=a.output/(b+'-requests.jsonl');write_jsonl(manifest,selected)
        try:
            if b=='nyra':
                run([a.qwen_python,str(here/'run_aligners.py'),'--backend','nyra','--manifest',str(manifest),'--output',str(a.output/'nyra.jsonl'),'--device',a.device])
                path=a.output/'nyra.jsonl'
            else:
                run([a.mfa_python,str(here/'run_mfa.py'),'--manifest',str(manifest),'--output',str(a.output/'mfa')])
                path=a.output/'mfa/predictions.jsonl'
        except (subprocess.CalledProcessError, OSError) as exc:
            preds[b]={r['id']:dict(id=r['id'],words=[],error=str(exc)) for r in selected}
            continue
        preds[b]={p['id']:p for p in read_jsonl(path)}
    output=[]
    for r in rows:
        cs={b:v[r['id']] for b,v in preds.items() if r['id'] in v}
        result=choose(r['text'],r['duration'],cs,policy)
        requested,reason=route(r['text'],preds['qwen'][r['id']].get('words',[]),r['duration'],policy)
        result.update(id=r['id'],requested_backend=requested);output.append(result)
    write_jsonl(a.output/'hybrid.jsonl',output)
    import collections
    (a.output/'runtime.json').write_text(json.dumps(dict(
        worker_seconds=time.perf_counter()-process_start,stages=stages,
        requested=dict(collections.Counter(p['requested_backend'] for p in output)),
        selected=dict(collections.Counter(p['selected_backend'] for p in output)),
        needs_review=sum(p['needs_review'] for p in output),
        cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
        scope='actual sequential Qwen -> Nyra subset -> MFA subset; child startup, routing, validation and output included; no cached predictions'),indent=2))
if __name__=='__main__':main()
