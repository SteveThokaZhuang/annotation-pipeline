"""Input transcript integrity, paired held-out rescue, and auditable source hashes."""
import collections
import hashlib
import json
from pathlib import Path
from common import ROOT,DATASET,load_manifest,read_jsonl,match_words,structural

def analyze(folder,manifest):
    rows=load_manifest(manifest);diag=read_jsonl(folder/'word_diagnostics.jsonl');out={}
    for backend in ['qwen','mfa','nyra','hybrid']:
        preds=read_jsonl(folder/(backend+'.jsonl'));byid={p['id']:p for p in preds}
        counts=collections.Counter();examples=[]
        for r in rows:
            p=byid[r['id']];words=p.get('words',[])
            tokens=[{'text':t} for t in r['text'].split()]
            m=match_words(tokens,words)
            counts.update(samples=1,input_tokens=len(tokens),predicted_tokens=len(words),input_tokens_matched=len(m['pairs']),input_tokens_deleted=len(m['deleted']),input_tokens_substituted=len(m['substituted']),extra_output_tokens=len(m['inserted']),execution_errors=bool(p.get('error')))
            for j,c in structural(words,r['duration']):
                counts[c]+=1
                if c=='zero_duration':examples.append(dict(id=r['id'],prediction_word_index=j,text=words[j]['text'],start=words[j]['start'],end=words[j]['end'],input_text=r['text']))
        out[backend]=dict(counts=counts,zero_examples=examples)
    q=[d for d in diag if d['backend']=='qwen']
    out['qwen_slices']={}
    for label,predicate in [('filler',lambda w:w['filler']),('nonfiller',lambda w:not w['filler']),('reference_flagged',lambda w:bool(w['reference_flags'])),('reference_unflagged',lambda w:w['filtered'])]:
        ww=[w for w in q if predicate(w)]
        out['qwen_slices'][label]=dict(words=len(ww),zero_duration=sum('zero_duration' in w['categories'] for w in ww),mae_gt100ms=sum(w['mae'] is not None and w['mae']>.1 for w in ww))
    out['held_out_paired']={}
    for b in ['mfa','nyra','hybrid']:
        other={(w['id'],w['word_index']):w for w in diag if w['backend']==b}
        qq=[w for w in q if w['meeting']=='IS1009a' and w['filtered']]
        bad=lambda w:w['mae'] is None or w['mae']>.1
        out['held_out_paired'][b]=dict(words=len(qq),rescued=sum(bad(w) and not bad(other[(w['id'],w['word_index'])]) for w in qq),regressed=sum(not bad(w) and bad(other[(w['id'],w['word_index'])]) for w in qq),zero_cases=sum('zero_duration' in w['categories'] for w in qq),zero_removed=sum('zero_duration' in w['categories'] and 'zero_duration' not in other[(w['id'],w['word_index'])]['categories'] and other[(w['id'],w['word_index'])]['mae'] is not None for w in qq))
    (folder/'insights.json').write_text(json.dumps(out,indent=2))
    return out

def provenance():
    files=[DATASET/'manifest.jsonl',DATASET/'source-provenance.json']
    files+=list((ROOT/'models/qwen3-forced-aligner/checkpoints').glob('*'))
    files+=list((ROOT/'models/nyra-forced-aligner/checkpoints').rglob('*.safetensors'))
    files+=list((ROOT/'models/nyra-forced-aligner/checkpoints').glob('*.npz'))
    files+=list((ROOT/'models/Montreal-Forced-Aligner/checkpoints').glob('*'))
    data={}
    for p in files:
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            data[str(p)]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
    (ROOT/'results/fa/provenance.json').write_text(json.dumps(data,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results/fa');p.add_argument('--manifest',type=Path,default=DATASET/'manifest.jsonl');p.add_argument('--hash',action='store_true');a=p.parse_args()
    print(json.dumps(analyze(a.results,a.manifest),indent=2))
    if a.hash:provenance()
