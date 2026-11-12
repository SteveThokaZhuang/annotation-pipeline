"""CPU-only smoke replay: saved real MFA, unchanged Qwen, audited bounded seams.

Purposive stress sample, NOT a representative accuracy evaluation. Includes the
first N zero chunks and every known intra/cross midpoint case plus neighbor rows.
No Qwen inference or GPU initialization; no audio or old results are overwritten.
"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa+mfa'))
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from repair import map_mfa_words, splice_sentence, SentenceRepair
from process import audit
from compare_results import validate_repairs
from seams import reconcile_rows
from run_rover_mfa_mae import lines, dump


def save(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def zero(row):
    return any(abs(w['end']-w['start'])<=1e-7 for w in row['words'])


def run_mode(base,tasks,raw,refs,out,policy,whole):
    out.mkdir(parents=True,exist_ok=True);(out/'mfa').mkdir()
    rows=copy.deepcopy(base);indices={r['id']:i for i,r in enumerate(rows)}
    records=[];whole_attempts=[];whole_ok=set()
    if whole:
        for row in rows:
            if not zero(row):continue
            i=indices[row['id']];original=base[i]
            record=dict(id=row['id'],task_id='chunk_'+str(i),row_index=i,start_index=0,
                end_index=len(row['words']),sentences=row['sentences'],original_words=copy.deepcopy(row['words']),
                zero_indices=[j for j,w in enumerate(row['words']) if abs(w['end']-w['start'])<=1e-7])
            try:
                reference=refs.get(row['id'])
                if reference is None:raise ValueError('No saved whole-chunk MFA output')
                candidate=map_mfa_words(row['words'],[[w['start'],w['end'],w['text']] for w in reference['words']],0)
                updated,bounds=splice_sentence(row['words'],0,len(row['words']),candidate,
                    row['chunk_start'],row['chunk_end'],policy,original['words'])
                row['words']=updated;whole_ok.add(row['id'])
                record.update(status='repaired',replacement_words=copy.deepcopy(updated),boundary_limits=bounds)
                records.append(record)
            except ValueError as exc:record.update(status='unresolved',reason=str(exc))
            whole_attempts.append(record)
    for task in tasks:
        if task['id'] in whole_ok:continue
        row=rows[indices[task['id']]];original=base[indices[task['id']]]
        record={k:copy.deepcopy(v) for k,v in task.items() if k not in ('status','reason','replacement_words','boundary_limits')}
        try:
            if task.get('preparation_error'):raise ValueError(task['preparation_error'])
            data=raw.get(task['task_id'])
            if data is None:raise ValueError('MFA has no saved raw output')
            tiers=[v for k,v in data['tiers'].items() if k.casefold().endswith('words')]
            if len(tiers)!=1:raise ValueError('MFA has no unique word tier')
            candidate=map_mfa_words(task['original_words'],tiers[0]['entries'],task['crop_start'])
            lo,hi=task['start_index'],task['end_index']
            updated,bounds=splice_sentence(row['words'],lo,hi,candidate,
                row['chunk_start'],row['chunk_end'],policy,original['words'])
            row['words']=updated
            record.update(status='repaired',replacement_words=copy.deepcopy(updated[lo:hi]),boundary_limits=bounds)
        except ValueError as exc:record.update(status='unresolved',reason=str(exc))
        records.append(record)
    from core import write_jsonl
    write_jsonl(out/'mfa/repairs.jsonl',records)
    audit(rows)
    dump(out/'unresolved.json',dict(rejected_sentences=[r for r in records if r['status']!='repaired'],
        residual_chunks=[dict(id=r['id'],issues=r['issues']) for r in rows if r['issues'] or r.get('error')]))
    accepted=[r for r in records if r['status']=='repaired']
    stats=dict(triggered_sentences=sum(len(r['sentences']) for r in records),
        seconds=0,accepted_groups=len(accepted),rejected_groups=len(records)-len(accepted))
    seams=[]
    if policy=='bounded':
        engine=SentenceRepair.__new__(SentenceRepair);engine.root=out/'mfa';engine.args=SimpleNamespace(output=out)
        stats=engine._finish_bounded(rows,base,stats)
        records=list(lines(out/'mfa/repairs.jsonl'));seams=list(lines(out/'mfa/cross_chunk_seams.jsonl'))
    else:
        rejected=[];reconcile_rows(rows,seams,rejected)
        write_jsonl(out/'mfa/cross_chunk_seams.jsonl',seams)
        write_jsonl(out/'mfa/cross_chunk_rejected.jsonl',rejected)
    result=audit(rows)
    unresolved=json.loads((out/'unresolved.json').read_text())
    unresolved['residual_chunks']=[dict(id=r['id'],issues=r['issues']) for r in rows if r['issues'] or r.get('error')]
    dump(out/'unresolved.json',unresolved)
    checked=validate_repairs(base,rows,records,unresolved,cross_chunk_seams=seams)
    edits=[dict(id=r['id'],task_id=r['task_id'],**e) for r in records if r['status']=='repaired'
           for e in r['boundary_limits'].get('adjustments',[])]
    if policy=='bounded':
        edits += [dict(id=s['qwen_id'],cross_chunk=True,**e) for s in seams for e in s['adjustments']]
    else:
        for seam in seams:
            side='left' if seam['before_left']['source']=='qwen' else 'right'
            edits.append(dict(id=seam[side+'_id'],neighbor_index=seam[side+'_index'],cross_chunk=True,
                before_neighbor=seam['before_'+side],after_neighbor=seam['after_'+side]))
    shifts=[abs(w[k]-o[k]) for original,row in zip(base,rows) for o,w in zip(original['words'],row['words'])
            if w['source']=='qwen' for k in ('start','end')]
    shrink=[1-(w['end']-w['start'])/(o['end']-o['start'])
            for original,row in zip(base,rows) for o,w in zip(original['words'],row['words'])
            if w['source']=='qwen' and o['end']-o['start']>1e-7]
    if policy=='bounded':
        assert max(shifts,default=0)<=.020+1e-12
        assert max(shrink,default=0)<=.10+1e-10
    for original,row in zip(base,rows):
        assert [w['text'] for w in row['words']]==[w['text'] for w in original['words']]
        for a,b in zip(original['words'],row['words']):
            assert not (b['end']-b['start']<=1e-7 and a['end']-a['start']>1e-7)
    save(out/'predictions.jsonl.gz',rows);save(out/'neighbor_edits.jsonl.gz',edits)
    save(out/'whole_attempts.jsonl.gz',whole_attempts)
    summary=dict(stats=result,validation=checked,whole_accepted_before_seams=len(whole_ok),
        repair_accepted=sum(r['status']=='repaired' for r in records),repair_rejected=sum(r['status']!='repaired' for r in records),
        rejection_reasons=dict(collections.Counter(r['reason'] for r in records if r['status']!='repaired')),
        normal_neighbor_edits=len(edits),max_original_endpoint_shift_ms=max(shifts,default=0)*1000,
        max_original_duration_shrink_fraction=max(shrink,default=0),
        cross_chunk=stats if policy=='bounded' else dict(applied=len(seams),rejected=len(rejected)))
    dump(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing',type=Path,default=ROOT/'results/open_yap_1k_rover_4gpu/qwen3-fa+mfa')
    p.add_argument('--previous',type=Path,default=ROOT/'results/open_yap_midpoint_chunk')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ordinary',type=int,default=256)
    p.add_argument('--recorded-inputs',type=Path,help='Replay an immutable prior smoke input snapshot instead of rescanning the dataset')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':p.error("Set CUDA_VISIBLE_DEVICES='' for this CPU-only replay")
    if args.output.exists():p.error('Use a fresh output directory to preserve experiment records')
    args.output.mkdir(parents=True)
    started=time.perf_counter();targets=set();selected={};origins={};selection=collections.defaultdict(set)
    if args.recorded_inputs:
        recorded=args.recorded_inputs
        base=list(lines(recorded/'inputs/qwen.jsonl.gz'))
        tasks=list(lines(recorded/'inputs/tasks.jsonl.gz'))
        raw={r['task_id']:r['alignment'] for r in lines(recorded/'inputs/raw_sentence_mfa.jsonl.gz')}
        refs={r['id']:r for r in lines(recorded/'inputs/whole_mfa.jsonl.gz')}
        selection_manifest=json.loads((recorded/'selection.json').read_text())
    else:
        for path in sorted((args.previous/'midpoint').glob('*/seams.jsonl.gz')):
            for r in lines(path):targets.add(r['id']);selection[r['id']].add('known_intra_midpoint_case')
        for seam in lines(args.previous/'cross_chunk_seams.jsonl'):
            for side in ('left_id','right_id'):
                targets.add(seam[side]);selection[seam[side]].add('known_cross_chunk_midpoint_case')
        ordinary=0;blocks=[]
        for path in sorted(args.existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl')):
            rows=list(lines(path));keep=set()
            for i,r in enumerate(rows):
                if ordinary<args.ordinary and zero(r):
                    targets.add(r['id']);selection[r['id']].add('first_zero_chunks');ordinary+=1
                if r['id'] in targets:
                    keep.add(i)
                    # Preserve adjacent same-recording context for actual seam checks.
                    if i:keep.add(i-1)
                    if i+1<len(rows):keep.add(i+1)
            if keep:
                name=path.parents[2].name+'_'+path.parent.name;blocks.append((name,path.parent))
                for i in sorted(keep):
                    r=rows[i];selected[r['id']]=r;origins[r['id']]=name
                    if not selection[r['id']]:selection[r['id']].add('neighbor_context')
        assert targets<=selected.keys(), 'Some stress cases are missing'
        base=list(selected.values());audit(base);ids=set(selected);tasks=[];raw={}
        for name,directory in blocks:
            local=[]
            for task in lines(directory/'mfa/repairs.jsonl'):
                if task['id'] in ids:
                    task['task_id']=name+'_'+task['task_id'];local.append(task)
            wanted={t['task_id'] for t in local}
            for item in lines(directory/'mfa/raw_alignments.jsonl.gz'):
                ident=name+'_'+item['task_id']
                if ident in wanted:raw[ident]=item['alignment']
            tasks.extend(sorted(local,key=lambda t:t['task_id']))
        refs={}
        for path in sorted((args.previous/'whole_mfa/batches').glob('*/mfa_predictions.jsonl.gz')):
            for row in lines(path):
                if row['id'] in ids:refs[row['id']]=row
        selection_manifest={k:dict(block=origins[k],reasons=sorted(selection[k])) for k in ids}
    save(args.output/'inputs/qwen.jsonl.gz',base)
    save(args.output/'inputs/tasks.jsonl.gz',tasks)
    save(args.output/'inputs/raw_sentence_mfa.jsonl.gz',[dict(task_id=k,alignment=v) for k,v in raw.items()])
    save(args.output/'inputs/whole_mfa.jsonl.gz',list(refs.values()))
    dump(args.output/'selection.json',selection_manifest)
    results={}
    for name,policy,whole in [('sentence_midpoint','midpoint',False),('sentence_bounded','bounded',False),
                              ('chunk_then_midpoint','midpoint',True),
                              ('chunk_then_bounded','bounded',True)]:
        results[name]=run_mode(base,tasks,raw,refs,args.output/name,policy,whole)
        print(name,json.dumps(results[name],ensure_ascii=False),flush=True)
    summary=dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        device='cpu',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],inference='none; replay of saved real MFA',
        command=sys.argv,parameters=dict(max_endpoint_shift_ms=20,max_duration_shrink_fraction=.10),
        selection=f'First {args.ordinary} zero chunks plus ALL prior 5 intra/28 cross seam cases and adjacent rows; purposive stress sample, not representative.',
        chunks=len(base),triggered_chunks=sum(zero(r) for r in base),qwen=audit(base),results=results,
        seconds=time.perf_counter()-started,
        limitations=['No human listening evaluation.','All modes include within-sample cross-chunk stitching.',
                     'Batch context only: supply all adjacent chunks when stitching globally.'])
    snapshot=args.output/'code_snapshot';snapshot.mkdir()
    files=['src/qwen3-fa/bounded_seams.py','src/qwen3-fa+mfa/repair.py','src/compare_results.py',
           'src/smoke_bounded_replacement.py','src/smoke_bounded_cpu_mfa.py','tests/test_bounded_seams.py']
    provenance={}
    for name in files:
        data=(ROOT/name).read_bytes();(snapshot/name.replace('/','__')).write_bytes(data)
        provenance[name]=hashlib.sha256(data).hexdigest()
    for path in sorted((args.output/'inputs').glob('*')):
        provenance[str(path.relative_to(args.output))]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(args.output/'provenance.json',provenance);dump(args.output/'summary.json',summary)
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
