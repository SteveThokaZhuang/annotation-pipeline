"""Validate completed replacement artifacts against original Qwen and arrays."""
import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from process import word_issues
from run_rover_mfa_mae import dump,lines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/open_yap_midpoint_chunk')
    p.add_argument('--existing',type=Path,default=ROOT/'results/open_yap_1k_rover_4gpu/qwen3-fa+mfa')
    p.add_argument('--wait',action='store_true');args=p.parse_args();root=args.root.resolve()
    while True:
        path=root/'comparison.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['complete']:break
        if not args.wait:raise SystemExit('Comparison incomplete')
        time.sleep(15)
    seams=list(lines(root/'cross_chunk_seams.jsonl'));rejected=list(lines(root/'cross_chunk_rejected.jsonl'))
    seam_by_id=collections.defaultdict(list)
    for seam in seams:
        midpoint=(seam['before_left']['end']+seam['before_right']['start'])/2
        assert math.isclose(midpoint,seam['midpoint'],abs_tol=1e-12)
        assert seam['after_left']==dict(seam['before_left'],end=midpoint)
        assert seam['after_right']==dict(seam['before_right'],start=midpoint)
        assert seam['after_left']['end']-seam['after_left']['start']>1e-7
        assert seam['after_right']['end']-seam['after_right']['start']>1e-7
        for side in ['left','right']:seam_by_id[seam[side+'_id']].append((seam,side))
    rejected_pairs={(s['left_id'],s['right_id']) for s in rejected}
    counts=collections.Counter();previous_ids=set();previous_by_channel={};audio=None
    for source in sorted(args.existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl')):
        name=source.parents[2].name+'_'+source.parent.name;out=root/'final'/name
        for q,r,n in itertools.zip_longest(lines(source),lines(out/'predictions.jsonl.gz'),lines(out/'annotations.jsonl.gz')):
            assert q and r and n and q['id']==r['id']==n['chunk_id']
            assert q['id'] not in previous_ids;previous_ids.add(q['id'])
            assert q['text']==r['text']==n['text']
            assert all(q[k]==r[k] for k in ['audio','channel','start_sample','end_sample','source_sample_rate'])
            assert [w['text'] for w in q['words']]==[w['text'] for w in r['words']]==n['force_align']['words']
            assert len(r['words'])==len(n['force_align']['timestamps'])
            origin=q['chunk_start']
            for w,span in zip(r['words'],n['force_align']['timestamps']):
                assert span==[w['start']-origin,w['end']-origin]
            zero=lambda ws:sum(abs(w['end']-w['start'])<=1e-7 for w in ws)
            if zero(q['words']):
                counts['triggered_chunks']+=1
                if r['whole_chunk_accepted']:
                    assert all(w['source']=='mfa' for w in r['words'])
                    assert not word_issues(r['words'],q['chunk_start'],q['chunk_end'])
                    counts['whole_chunk_accepted']+=1
                else:counts['sentence_fallback_chunks']+=1
            else:
                expected=[dict(w) for w in q['words']]
                for seam,side in seam_by_id[q['id']]:
                    index=seam[side+'_index']
                    assert expected[index]==seam['before_'+side]
                    expected[index]=dict(seam['after_'+side])
                assert r['words']==expected
                counts['zero_free_chunks_with_audited_neighbor_endpoint_change' if seam_by_id[q['id']] else 'unmodified_zero_free_chunks']+=1
            key=(q.get('source_audio') or q['audio'],q.get('channel') or 0)
            if key[0]!=audio:previous_by_channel={};audio=key[0]
            if r['words']:
                previous=previous_by_channel.get(key)
                if previous and previous['chunk_end']<=r['chunk_start']+1e-7:
                    a,b=previous['words'][-1],r['words'][0]
                    if {a.get('source'),b.get('source')}=={'qwen','mfa'} and a['end']>b['start']+1e-7:
                        assert (previous['id'],r['id']) in rejected_pairs
                        counts['reported_remaining_mixed_source_overlaps']+=1
                previous_by_channel[key]=r
            assert zero(r['words'])<=zero(q['words'])
            counts['chunks']+=1;counts['words']+=len(r['words']);counts['zero_words']+=zero(r['words'])
        print(f'Validated {name} chunks={counts["chunks"]}',flush=True)
    assert counts['chunks']==919459 and counts['words']==10301121 and counts['triggered_chunks']==303054
    assert counts['whole_chunk_accepted']==result['whole_chunk_accepted']
    assert counts['zero_words']==result['full_dataset_totals']['zero_duration_words']
    assert counts['zero_words']<=result['midpoint_replay']['totals']['midpoint_sentence']['zero_duration_words']
    sums={};words=0
    for path in sorted((root/'zero_replacement').glob('b*/comparisons.jsonl.gz')):
        for row in lines(path):
            n=row['matched_words'];words+=n
            for mode,mae in row['mae_ms'].items():
                if n:sums[mode]=sums.get(mode,0.)+n*mae
    for mode,total in sums.items():
        assert words==result['zero_chunk_mae_against_whole_mfa'][mode]['matched_words']
        assert math.isclose(total/words,result['zero_chunk_mae_against_whole_mfa'][mode]['boundary_mae_ms'],rel_tol=1e-10)
    samples=json.loads((root/'review/samples.json').read_text())
    assert all((root/'review'/r['review_audio']).exists() for r in samples['samples'])
    validation=dict(passed=True,counts=dict(counts),matched_words=words,
        text_and_token_units_unchanged=True,zero_free_chunks_unchanged_except_audited_touching_endpoints=True,
        accepted_whole_chunks_positive_bounded_nonoverlapping=True,
        normalized_timestamps_exact=True,no_new_zero_duration_words=True,
        independently_weighted_mae_recomputed=True,review_audio_samples=len(samples['samples']),
        cross_chunk_midpoints=len(seams),rejected_cross_chunk_seams=len(rejected),
        listening_status='Prepared for human review, not human-adjudicated')
    dump(root/'validation.json',validation)
    persistent=ROOT/'results/open_yap_1k_summary/midpoint_chunk_comparison.json'
    saved=json.loads(persistent.read_text());saved['validation']=validation;dump(persistent,saved)
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
