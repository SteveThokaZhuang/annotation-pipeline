"""Compare legacy sentence repair, midpoint seams and whole-chunk MFA.

Whole-chunk replacements preserve Qwen word units and transcript. The final
policy retains validated midpoint sentence repair when a whole chunk is rejected.
The pure whole-chunk-only policy is also measured separately for comparison.
"""
import argparse
import collections
import copy
import gzip
import heapq
import itertools
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa+mfa'))
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from repair import map_mfa_words,splice_sentence
from process import word_issues,audit
from prediction_format import chunk_prediction,SCHEMA
from seams import reconcile_stream
from compare_mfa_mae import align_words
from run_rover_mfa_mae import dump,lines

MODES=['qwen','old_sentence','midpoint_sentence','whole_chunk','chunk_then_midpoint']


def zero(words):return sum(abs(w['end']-w['start'])<=1e-7 for w in words)


def replace_chunk(row,reference):
    if reference is None:return row['words'],False,'missing_mfa_output'
    try:
        candidate=map_mfa_words(row['words'],[[w['start'],w['end'],w['text']] for w in reference['words']],0)
        updated,_=splice_sentence(row['words'],0,len(row['words']),candidate,
                                   row['chunk_start'],row['chunk_end'],'midpoint')
        return updated,True,None
    except ValueError as exc:return row['words'],False,str(exc)


def choose_final(row,reference,midpoint):
    whole,ok,reason=replace_chunk(row,reference)
    final=whole if ok else midpoint['words']
    assert [w['text'] for w in final]==[w['text'] for w in row['words']]
    return whole,final,ok,reason


def midpoint_zero_rows(existing,midpoint):
    for path in sorted(existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl')):
        name=path.parents[2].name+'_'+path.parent.name
        for q,m in itertools.zip_longest(lines(path),lines(midpoint/name/'predictions.jsonl.gz')):
            assert q and m and q['id']==m['id']
            if zero(q['words']):yield m


def stats(words,lower,upper):
    issues=word_issues(words,lower,upper)
    return dict(words=len(words),zero_words=zero(words),zero_chunks=int(bool(zero(words))),
                invalid_words=len({r['index'] for r in issues}),bad_chunks=int(bool(issues)))


def process_batch(batch,midpoint_rows,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        for q in lines(batch/'input.jsonl.gz'):assert next(midpoint_rows)['id']==q['id']
        return json.loads((out/'summary.json').read_text())
    refs={r['id']:r for r in lines(batch/'mfa_predictions.jsonl.gz')}
    totals={k:collections.Counter() for k in MODES};metrics={k:collections.Counter() for k in MODES}
    reasons=collections.Counter();accepted=rejected=count=matched=0;worst=[];reservoir=[];rng=random.Random(20261110)
    with gzip.open(out/'predictions.jsonl.gz','wt',compresslevel=1) as pred, \
         gzip.open(out/'comparisons.jsonl.gz','wt',compresslevel=1) as comparisons, \
         gzip.open(out/'rejected.jsonl.gz','wt',compresslevel=1) as failures:
        for q in lines(batch/'input.jsonl.gz'):
            m=next(midpoint_rows);assert m['id']==q['id']
            reference=refs.get(q['id']);updated,final,ok,reason=choose_final(q,reference,m)
            variants=dict(qwen=q['words'],old_sentence=q['old_words'],midpoint_sentence=m['words'],
                          whole_chunk=updated,chunk_then_midpoint=final)
            assert all([w['text'] for w in ws]==[w['text'] for w in q['words']] for ws in variants.values())
            local={k:stats(ws,q['chunk_start'],q['chunk_end']) for k,ws in variants.items()}
            for k,s in local.items():totals[k].update(s)
            if ok:
                accepted+=1
                assert not local['whole_chunk']['zero_words'] and not local['whole_chunk']['invalid_words']
            else:
                rejected+=1;reasons[reason]+=1
                assert updated==q['words']
                failures.write(json.dumps(dict(id=q['id'],reason=reason),ensure_ascii=False)+'\n')
            row={k:v for k,v in q.items() if k!='old_words'}
            row.update(words=final,issues=word_issues(final,q['chunk_start'],q['chunk_end']),
                       repair_scope='chunk_then_sentence',whole_chunk_accepted=ok,whole_chunk_rejection=reason)
            pred.write(json.dumps(row,ensure_ascii=False)+'\n')
            local_error={k:0. for k in MODES};max_old_error=0.;pairs=[]
            if reference:
                pairs,_,_=align_words(q['words'],reference['words'])
                matched+=len(pairs)
                for i,j in pairs:
                    ref=reference['words'][j]
                    for mode,ws in variants.items():
                        onset=abs(ws[i]['start']-ref['start'])*1000
                        offset=abs(ws[i]['end']-ref['end'])*1000
                        delta=(onset+offset)/2
                        metrics[mode].update(dict(words=1,onset_sum_ms=onset,offset_sum_ms=offset,delta_sum_ms=delta))
                        local_error[mode]+=delta
                        if mode=='old_sentence':max_old_error=max(max_old_error,delta)
            maes={k:v/len(pairs) if pairs else None for k,v in local_error.items()}
            comparisons.write(json.dumps(dict(id=q['id'],whole_chunk_accepted=ok,rejection=reason,
                stats=local,matched_words=len(pairs),mae_ms=maes),ensure_ascii=False)+'\n')
            review=dict(id=q['id'],audio=q['audio'],channel=q['channel'],source_sample_rate=q['source_sample_rate'],
                start_sample=q['start_sample'],end_sample=q['end_sample'],duration=q['duration'],
                chunk_start=q['chunk_start'],text=q['text'],words=variants,
                mfa_reference=reference['words'] if reference else [],whole_chunk_accepted=ok,rejection=reason,
                mae_ms=maes,max_old_boundary_error_ms=max_old_error)
            key=(max_old_error,q['id'],review)
            if len(worst)<12:heapq.heappush(worst,key)
            elif key[:2]>worst[0][:2]:heapq.heapreplace(worst,key)
            count+=1
            if len(reservoir)<6:reservoir.append(review)
            else:
                index=rng.randrange(count)
                if index<6:reservoir[index]=review
    dump(out/'review_candidates.json',dict(worst=[x[2] for x in sorted(worst,reverse=True)],random=reservoir))
    summary=dict(batch=batch.name,chunks=count,accepted=accepted,rejected=rejected,reasons=dict(reasons),
        totals={k:dict(v) for k,v in totals.items()},metrics={k:dict(v) for k,v in metrics.items()},matched_words=matched,
        validation=dict(text_and_word_units_unchanged=True,accepted_chunks_positive_bounded_nonoverlapping=True,
                        rejected_chunks_retain_original_qwen=True))
    dump(out/'summary.json',summary)
    print(f'Compared {batch.name}: chunks={count} accepted={accepted} rejected={rejected}',flush=True)
    return summary


def aggregate(root,summaries):
    metrics={};totals={}
    for mode in MODES:
        total=collections.Counter();metric=collections.Counter()
        for s in summaries:total.update(s['totals'][mode]);metric.update(s['metrics'][mode])
        n=metric['words'];totals[mode]=dict(total)
        metrics[mode]=dict(matched_words=n,onset_mae_ms=metric['onset_sum_ms']/n if n else None,
            offset_mae_ms=metric['offset_sum_ms']/n if n else None,boundary_mae_ms=metric['delta_sum_ms']/n if n else None,
            sums=dict(metric))
    reasons=collections.Counter()
    for s in summaries:reasons.update(s['reasons'])
    result=dict(complete=False,compared_batches=len(summaries),chunks=sum(s['chunks'] for s in summaries),
        whole_chunk_accepted=sum(s['accepted'] for s in summaries),whole_chunk_rejected=sum(s['rejected'] for s in summaries),
        reasons=dict(reasons),zero_chunk_totals=totals,zero_chunk_mae_against_whole_mfa=metrics,
        note='MAE measures agreement with MFA pseudo-reference; copying MFA is not evidence of greater human accuracy.',
        mae_stage='Per-chunk replacement before cross-chunk midpoint stitching; common matched-token set for all variants.',
        policy='Any original Qwen zero word triggers full-chunk MFA. Accept only positive, bounded, nonoverlapping, text/word-unit-preserving output. Otherwise use validated sentence-midpoint repair; retain Qwen only for sentences rejected by both stages.',
        device='cpu')
    dump(root/'comparison.json',result);return result


def merge_all(existing,root,inventory):
    destination=root/'final';destination.mkdir(exist_ok=True)
    summaries=[]
    new_rows=(row for task in inventory['batches'] for row in lines(root/'zero_replacement'/Path(task['path']).name/'predictions.jsonl.gz'))
    def source_rows():
        for path in sorted(existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl')):
            name=path.parents[2].name+'_'+path.parent.name
            for q in lines(path):
                row=copy.deepcopy(q)
                if zero(q['words']):
                    updated=next(new_rows);assert updated['id']==q['id']
                    row.update(words=updated['words'],repair_scope='chunk_then_sentence',whole_chunk_accepted=updated['whole_chunk_accepted'],
                               whole_chunk_rejection=updated['whole_chunk_rejection'])
                else:row['repair_scope']='not_triggered'
                row['method']='qwen+mfa-chunk';row['_output_block']=name;yield row
    adjustments=[];rejected=[]
    for name,group in itertools.groupby(reconcile_stream(source_rows(),adjustments,rejected),key=lambda r:r['_output_block']):
        out=destination/name;out.mkdir(exist_ok=True);rows=list(group)
        triggered=sum(r['repair_scope']!='not_triggered' for r in rows)
        for r in rows:r.pop('_output_block')
        result=audit(rows)
        with gzip.open(out/'predictions.jsonl.gz','wt',compresslevel=1) as native, \
             gzip.open(out/'annotations.jsonl.gz','wt',compresslevel=1) as normalized:
            for row in rows:
                native.write(json.dumps(row,ensure_ascii=False)+'\n')
                normalized.write(json.dumps(chunk_prediction(row),ensure_ascii=False)+'\n')
        dump(out/'error_stats.json',result);summaries.append(result)
        print(f'Merged final {len(summaries)}/30 block={name} triggered={triggered}',flush=True)
    assert next(new_rows,None) is None
    keys=['chunks','words','zero_duration_words','invalid_words','bad_chunks','failed_chunks']
    total={k:sum(s[k] for s in summaries) for k in keys}
    assert total['chunks']==919459 and total['words']==10301121
    dump(destination/'annotation_schema.json',dict(SCHEMA,audit_file='predictions.jsonl.gz',
         prediction_file='annotations.jsonl.gz',files_relative_to='each shard directory'))
    dump(destination/'summary.json',total)
    for name,records in [('cross_chunk_seams',adjustments),('cross_chunk_rejected',rejected)]:
        with (root/(name+'.jsonl')).open('w') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    return total,dict(applied=len(adjustments),rejected=len(rejected),
        note='Only touching endpoints of different Qwen/MFA sources are modified; upstream overlapping audio windows are reported separately.')


def review_pack(root,inventory):
    worst=[];randoms=[]
    for t in inventory['batches']:
        p=root/'zero_replacement'/Path(t['path']).name/'review_candidates.json'
        d=json.loads(p.read_text());worst.extend(d['worst']);randoms.extend(d['random'])
    worst=sorted(worst,key=lambda r:(r['max_old_boundary_error_ms'],r['id']),reverse=True)[:30]
    rng=random.Random(20261110);randoms=rng.sample(randoms,min(30,len(randoms)))
    chosen={r['id']:dict(r,selection='random') for r in randoms}
    for r in worst:chosen[r['id']]=dict(r,selection='largest_old_sentence_vs_whole_mfa_disagreement')
    import soundfile as sf
    import soxr
    out=root/'review';out.mkdir(exist_ok=True);records=[]
    global_seams=list(lines(root/'cross_chunk_seams.jsonl'))
    for i,r in enumerate(chosen.values()):
        for seam in global_seams:
            for side in ['left','right']:
                if r['id']==seam[side+'_id']:
                    r['words']['chunk_then_midpoint'][seam[side+'_index']]=dict(seam['after_'+side])
        with sf.SoundFile(r['audio']) as f:
            f.seek(r['start_sample']);audio=f.read(r['end_sample']-r['start_sample'],dtype='float32',always_2d=True)[:,r['channel']]
            if f.samplerate!=16000:audio=soxr.resample(audio,f.samplerate,16000)
        name=f'{i:03d}.wav';sf.write(out/name,audio,16000,subtype='PCM_16')
        clip_words={mode:[dict(w,start=w['start']-r['chunk_start'],end=w['end']-r['chunk_start']) for w in ws]
                    for mode,ws in r['words'].items()}
        records.append(dict(r,review_audio=name,clip_words=clip_words,mae_stage='before cross-chunk stitching',
            review_timestamp_origin='words use source recording time; clip_words use this review audio start'))
    dump(out/'samples.json',dict(status='prepared_for_manual_listening; not human-adjudicated',samples=records))
    seam_reviews=[]
    for i,seam in enumerate(global_seams):
        with sf.SoundFile(seam['source_audio']) as f:
            start=max(0,seam['midpoint']-.5);end=min(f.frames/f.samplerate,seam['midpoint']+.5)
            a,b=round(start*f.samplerate),round(end*f.samplerate);f.seek(a)
            waveform=f.read(b-a,dtype='float32',always_2d=True)[:,seam['channel']]
            if f.samplerate!=16000:waveform=soxr.resample(waveform,f.samplerate,16000)
        name=f'seam_{i:03d}.wav';sf.write(out/name,waveform,16000,subtype='PCM_16')
        seam_reviews.append(dict(seam,review_audio=name,clip_start_sec=start,midpoint_in_clip_sec=seam['midpoint']-start))
    dump(out/'cross_chunk_seams.json',dict(status='prepared_for_manual_listening; not human-adjudicated',samples=seam_reviews))
    return len(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/open_yap_midpoint_chunk')
    p.add_argument('--existing',type=Path,default=ROOT/'results/open_yap_1k_rover_4gpu/qwen3-fa+mfa')
    p.add_argument('--wait',action='store_true');args=p.parse_args();root=args.root.resolve()
    def wait(path):
        while not path.exists():
            if not args.wait:raise SystemExit('Not ready: '+str(path))
            time.sleep(15)
    wait(root/'whole_mfa/inventory.json');wait(root/'midpoint/summary.json')
    assert json.loads((root/'midpoint/summary.json').read_text())['complete']
    inventory=json.loads((root/'whole_mfa/inventory.json').read_text())
    midpoint_rows=midpoint_zero_rows(args.existing,root/'midpoint');summaries=[]
    for task in inventory['batches']:
        batch=Path(task['path']);wait(batch/'summary.json')
        summaries.append(process_batch(batch,midpoint_rows,root/'zero_replacement'/batch.name))
        aggregate(root,summaries)
    assert next(midpoint_rows,None) is None
    result=aggregate(root,summaries);assert result['chunks']==303054
    result['full_dataset_totals'],result['cross_chunk_seams']=merge_all(args.existing,root,inventory)
    result['review_samples']=review_pack(root,inventory)
    result['midpoint_replay']=json.loads((root/'midpoint/summary.json').read_text())
    result['complete']=True
    dump(root/'comparison.json',result)
    persistent=ROOT/'results/open_yap_1k_summary/midpoint_chunk_comparison.json'
    dump(persistent,dict(result=result,detailed_results=str(root),storage='Large artifacts on local disk; summary on shared project storage.'))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
