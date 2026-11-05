"""Reference-relative diagnostics and development-only routing selection."""
import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from common import ROOT,DATASET,load_manifest,read_jsonl,write_jsonl,match_words,structural,norm
from router import features,choose,route

EXCLUDE_FLAGS=('reference_duration_suspicious','truncated','overlap')
def eligible(w,duration):
    return all(not w.get(k,False) for k in EXCLUDE_FLAGS) and 0<=w['start']<w['end']<=duration+.02

def word_rows(r,p):
    pred=p.get('words',[]); matched=match_words(r['words'],pred)
    mapping=dict(matched['pairs']); issues=collections.defaultdict(list)
    for j,kind in structural(pred,r['duration']): issues[j].append(kind)
    out=[]
    for i,w in enumerate(r['words']):
        j=mapping.get(i); v=pred[j] if j is not None else None
        valid=v is not None and all(isinstance(v.get(k),(float,int)) and math.isfinite(v[k]) for k in ('start','end'))
        err=(abs(v['start']-w['start'])+abs(v['end']-w['end']))/2 if valid else None
        se=abs(v['start']-w['start']) if valid else None
        ee=abs(v['end']-w['end']) if valid else None
        cats=list(issues[j]) if j is not None else ['missing_or_substituted']
        if err is not None:
            if err>.1: cats.append('boundary_mae_gt100ms')
            if err>.2: cats.append('boundary_mae_gt200ms')
            if max(se,ee)>.1: cats.append('either_boundary_gt100ms')
            inter=max(0,min(v['end'],w['end'])-max(v['start'],w['start']))
            union=max(v['end'],w['end'])-min(v['start'],w['start'])
            iou=inter/union if union>0 else 0
            if iou<.5: cats.append('iou_lt0.5')
        else: iou=None
        out.append(dict(id=r['id'],base_id=r['base_id'],meeting=r['meeting'],condition=r['condition'],word_index=i,text=w['text'],reference_id=w['id'],ref_start=w['start'],ref_end=w['end'],pred_start=v['start'] if v else None,pred_end=v['end'] if v else None,mae=err,start_error=se,end_error=ee,iou=iou,filtered=eligible(w,r['duration']),categories=cats,reference_flags=[k for k in EXCLUDE_FLAGS if w.get(k)],filler=norm(w['text']) in {'um','uh','er','erm','hmm','mm'}))
    return out

def percentile(v,q):
    if not v: return None
    v=sorted(v); return v[min(len(v)-1,int((len(v)-1)*q))]

def summarize(items):
    good=[w for w in items if w['mae'] is not None]
    es=[w['mae'] for w in good]
    return dict(reference_words=len(items),matched_words=len(good),coverage=len(good)/len(items) if items else 0,mae_ms=statistics.mean(es)*1000 if es else None,p50_ms=percentile(es,.5)*1000 if es else None,p95_ms=percentile(es,.95)*1000 if es else None,mean_iou=statistics.mean(w['iou'] for w in good) if good else None,bad_or_missing=sum(w['mae'] is None or w['mae']>.1 for w in items),categories=dict(collections.Counter(c for w in items for c in w['categories'])))

def loss(items):
    items=[w for w in items if w['filtered']]
    # Fixed lexicographic-like cost: missing/large error dominates fine boundary gain.
    return sum((w['mae'] if w['mae'] is not None else 1) + .1*(w['mae'] is None or w['mae']>.1) for w in items)

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results/fa');p.add_argument('--manifest',type=Path,default=DATASET/'manifest.jsonl');p.add_argument('--policy',type=Path);p.add_argument('--hybrid-predictions',type=Path);a=p.parse_args()
    rows=load_manifest(a.manifest); base=a.results
    preds={b:{v['id']:v for v in read_jsonl(base/(b+'.jsonl'))} for b in ['qwen','nyra','mfa']}
    for b in preds:
        if set(preds[b])!={r['id'] for r in rows}: raise ValueError(f'{b}: incomplete/extra predictions')
    diagnostics={b:{r['id']:word_rows(r,preds[b][r['id']]) for r in rows} for b in preds}
    dev=[r for r in rows if r['meeting']=='ES2004a']; test=[r for r in rows if r['meeting']=='IS1009a']
    # Conditions from the same base utterance cannot cross meetings.
    candidates=[]
    for f,thresholds in [('hard',[1]),('short_ratio',[.05,.1,.2,.3]),('long',[1]),('fillers',[.05,.1,.2]),('repeat',[1]),('gap',[.5,1,2])]:
        for t,b in itertools.product(thresholds,['nyra','mfa']): candidates.append(dict(feature=f,threshold=t,backend=b))
    cache={}
    def eval_policy(rules):
        policy={'rules':rules,'guarded':True}; cost=0; count=0
        for r in dev:
            cs={b:preds[b][r['id']] for b in preds}
            pick=choose(r['text'],r['duration'],cs,policy); b=pick['selected_backend']
            count+=b!='qwen'; cost+=loss(diagnostics[b][r['id']])
        return cost,count
    baseline,_=eval_policy([])
    best=(baseline,0,[])
    # At most two ordered, explainable rules. Qwen retains >=60% of dev utterances.
    trials=[[]]+[[r] for r in candidates]+[[x,y] for x in candidates for y in candidates if x['feature']!=y['feature']]
    for rules in ([] if a.policy else trials):
        cost,count=eval_policy(rules)
        if count<=.4*len(dev) and (cost,count)<best[:2]: best=cost,count,rules
    policy=dict(version=2,guarded=True,development_meeting='ES2004a',held_out_meeting='IS1009a',reference_status='AMI forced-alignment silver; not independent human gold',max_development_routing_fraction=.4,objective='filtered uncapped MAE seconds (missing=1s) + 0.1*bad_or_missing count',development_baseline_loss=baseline,development_selected_loss=best[0],rules=best[2])
    if a.policy: policy=json.loads(a.policy.read_text())
    (base/'policy.json').write_text(json.dumps(policy,indent=2))
    hybrid=[]
    for r in rows:
        pick=choose(r['text'],r['duration'],{b:preds[b][r['id']] for b in preds},policy)
        requested,reason=route(r['text'],preds['qwen'][r['id']]['words'],r['duration'],policy)
        pick.update(id=r['id'],requested_backend=requested,route_features=features(r['text'],preds['qwen'][r['id']]['words'],r['duration']))
        hybrid.append(pick)
    if a.hybrid_predictions:
        hp={p['id']:p for p in read_jsonl(a.hybrid_predictions)}
        assert set(hp)=={r['id'] for r in rows}
        hybrid=[hp[r['id']] for r in rows]
        for r,p in zip(rows,hybrid):
            p.setdefault('requested_backend',route(r['text'],preds['qwen'][r['id']]['words'],r['duration'],policy)[0])
    write_jsonl(base/'hybrid.jsonl',hybrid)
    diagnostics['hybrid']={r['id']:word_rows(r,p) for r,p in zip(rows,hybrid)}
    report={'reference_status':policy['reference_status'],'dataset':dict(samples=len(rows),audio_seconds=sum(r['duration'] for r in rows),words=sum(len(r['words']) for r in rows),unique_events=len({r['base_id'] for r in rows})),'systems':{},'routing':{}}
    allrows=[]
    for b,rs in diagnostics.items():
        allitems=[w for vs in rs.values() for w in vs]
        allrows.extend(dict(w,backend=b) for w in allitems)
        report['systems'][b]={}
        for split,selected in [('all',rows),('development',dev),('held_out',test)]:
            items=[w for r in selected for w in rs[r['id']]]
            report['systems'][b][split]={'all_reference':summarize(items),'filtered_reference':summarize([w for w in items if w['filtered']]),'conditions':{c:summarize([w for w in items if w['condition']==c and w['filtered']]) for c in sorted({r['condition'] for r in rows})}}
    for split,selected in [('all',rows),('development',dev),('held_out',test)]:
        ids={r['id'] for r in selected}; hp=[p for p in hybrid if p['id'] in ids]
        report['routing'][split]=dict(selected=dict(collections.Counter(p['selected_backend'] for p in hp)),requested=dict(collections.Counter(p['requested_backend'] for p in hp)),needs_review=sum(p['needs_review'] for p in hp))
    # Paired same-word rescue/regression for each anomaly; keep unresolved cases explicit.
    report['paired_rescue']={}
    q=[w for vs in diagnostics['qwen'].values() for w in vs]
    for b in ['mfa','nyra','hybrid']:
        other={(w['id'],w['word_index']):w for vs in diagnostics[b].values() for w in vs}
        stats={}
        for category in ['zero_duration','short_duration','long_duration','missing_or_substituted','boundary_mae_gt100ms','boundary_mae_gt200ms','iou_lt0.5']:
            items=[w for w in q if category in w['categories']]
            alt=[other[(w['id'],w['word_index'])] for w in items]
            stats[category]=dict(qwen_cases=len(items),category_removed=sum(category not in v['categories'] and v['mae'] is not None for v in alt),timing_rescued_le100ms=sum(v['mae'] is not None and v['mae']<=.1 for v in alt),mae_improved=sum(v['mae'] is not None and (w['mae'] is None or v['mae']<w['mae']-1e-6) for w,v in zip(items,alt)))
        stats['regression_on_qwen_good_filtered']=sum(w['filtered'] and w['mae'] is not None and w['mae']<=.1 and (other[(w['id'],w['word_index'])]['mae'] is None or other[(w['id'],w['word_index'])]['mae']>.1) for w in q)
        report['paired_rescue'][b]=stats
    write_jsonl(base/'word_diagnostics.jsonl',allrows)
    (base/'evaluation.json').write_text(json.dumps(report,indent=2))
    with (base/'review.csv').open('w') as f:
        fields=['backend','id','word_index','text','ref_start','ref_end','pred_start','pred_end','mae','categories','reference_flags','human_start','human_end','reviewer','review_status']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for w in allrows:
            if w['backend']=='qwen' and (w['categories'] or w['reference_flags']): writer.writerow({**{k:w.get(k,'') for k in fields},'review_status':'pending'})
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
