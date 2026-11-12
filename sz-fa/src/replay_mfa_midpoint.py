"""Replay saved MFA sentence candidates with midpoint seams; no inference."""
import argparse
import copy
import gzip
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa+mfa'))
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from repair import map_mfa_words,splice_sentence
from process import audit
from compare_results import validate_repairs
from run_rover_mfa_mae import dump,lines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing',type=Path,default=ROOT/'results/open_yap_1k_rover_4gpu/qwen3-fa+mfa')
    p.add_argument('--output',type=Path,default=ROOT/'results/open_yap_midpoint_chunk/midpoint')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();summaries=[]
    for source in sorted(args.existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl')):
        name=source.parents[2].name+'_'+source.parent.name
        out=args.output/name;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():
            summaries.append(json.loads((out/'summary.json').read_text()));continue
        base=list(lines(source));old=list(lines(source.with_name('predictions.jsonl')))
        assert [r['id'] for r in base]==[r['id'] for r in old]
        rows=copy.deepcopy(base)
        raw={r['task_id']:r['alignment'] for r in lines(source.parent/'mfa/raw_alignments.jsonl.gz')}
        tasks=sorted(lines(source.parent/'mfa/repairs.jsonl'),key=lambda t:t['task_id'])
        accepted=[];rejected=[];changes=[];seams=[]
        for task in tasks:
            row=rows[task['row_index']];assert row['id']==task['id']
            record={k:v for k,v in task.items() if k not in ['status','reason','replacement_words','boundary_limits']}
            try:
                if task.get('preparation_error'):raise ValueError(task['preparation_error'])
                data=raw.get(task['task_id'])
                if data is None:raise ValueError('MFA has no saved raw output')
                tiers=[v for k,v in data['tiers'].items() if k.casefold().endswith('words')]
                if len(tiers)!=1:raise ValueError('MFA has no unique word tier')
                candidate=map_mfa_words(task['original_words'],tiers[0]['entries'],task['crop_start'])
                updated,bounds=splice_sentence(row['words'],task['start_index'],task['end_index'],
                    candidate,row['chunk_start'],row['chunk_end'],'midpoint')
                row['words']=updated
                record.update(status='repaired',replacement_words=updated[task['start_index']:task['end_index']],boundary_limits=bounds)
                accepted.append(record)
                seams.extend(dict(id=row['id'],task_id=task['task_id'],**item) for item in bounds['adjustments'])
            except ValueError as exc:
                record.update(status='unresolved',reason=str(exc));rejected.append(record)
            if record['status']!=task['status']:
                changes.append(dict(id=row['id'],task_id=task['task_id'],old_status=task['status'],
                                    new_status=record['status'],old_reason=task.get('reason'),new_reason=record.get('reason')))
        before=audit(base);old_stats=audit(old);after=audit(rows)
        unresolved=dict(rejected_sentences=rejected,residual_chunks=[
            dict(id=r['id'],issues=r['issues'],error=r.get('error')) for r in rows if r['issues'] or r.get('error')])
        validation=validate_repairs(base,rows,accepted+rejected,unresolved)
        with gzip.open(out/'predictions.jsonl.gz','wt',compresslevel=1) as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        for filename,records in [('repairs',accepted+rejected),('seams',seams),('status_changes',changes)]:
            with gzip.open(out/(filename+'.jsonl.gz'),'wt',compresslevel=1) as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        dump(out/'unresolved.json',unresolved)
        summary=dict(block=name,qwen=before,old_sentence=old_stats,midpoint_sentence=after,
            groups=len(tasks),accepted=len(accepted),rejected=len(rejected),midpoint_seams=len(seams),
            changed_statuses=len(changes),validation=validation)
        dump(out/'summary.json',summary);summaries.append(summary)
        print(f'Midpoint {len(summaries)}/30 block={name} seams={len(seams)} rejected={len(rejected)}',flush=True)
    totals={}
    for mode in ['qwen','old_sentence','midpoint_sentence']:
        keys=['chunks','words','zero_duration_words','invalid_words','bad_chunks','failed_chunks']
        totals[mode]={k:sum(s[mode][k] for s in summaries) for k in keys}
    summary=dict(complete=True,blocks=len(summaries),totals=totals,
        groups=sum(s['groups'] for s in summaries),accepted=sum(s['accepted'] for s in summaries),
        rejected=sum(s['rejected'] for s in summaries),midpoint_seams=sum(s['midpoint_seams'] for s in summaries),
        changed_statuses=sum(s['changed_statuses'] for s in summaries),seconds=time.perf_counter()-started,
        inference='none; saved raw MFA outputs replayed on original Qwen words',device='cpu')
    dump(args.output/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
