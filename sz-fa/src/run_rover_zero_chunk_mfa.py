"""CPU whole-chunk MFA references for all Qwen zero-duration ROVER chunks."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import itertools
import json
from pathlib import Path
import time

from run_rover_mfa_mae import ROOT,dump,lines,run_batch


def prepare(args):
    if (args.output/'inventory.json').exists():return json.loads((args.output/'inventory.json').read_text())
    tasks=[];batch=[];count=0;zero=0;zero_words=0
    def flush():
        if not batch:return
        directory=args.output/'batches'/f'b{len(tasks):05d}';directory.mkdir(parents=True,exist_ok=True)
        with gzip.open(directory/'input.jsonl.gz','wt',compresslevel=1) as f:
            for r in batch:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        tasks.append(dict(path=str(directory),chunks=len(batch),audio_seconds=sum(r['duration'] for r in batch)))
        batch.clear()
    for path in sorted(args.existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl')):
        for q,h in itertools.zip_longest(lines(path),lines(path.with_name('predictions.jsonl'))):
            assert q and h and q['id']==h['id'] and q['text']==h['text']
            count+=1
            nz=sum(abs(w['end']-w['start'])<=1e-7 for w in q['words'])
            if not nz:continue
            assert not q.get('error')
            assert [w['text'] for w in q['words']]==[w['text'] for w in h['words']]
            zero+=1;zero_words+=nz
            keep=['id','audio','audio_scope','text','channel','source_sample_rate','start_sample','end_sample',
                  'chunk_start','chunk_end','duration','source_json','words','language','recording_id','sentences']
            row={k:q[k] for k in keep if k in q};row['old_words']=h['words']
            batch.append(row)
            if len(batch)==args.batch_size:flush()
        print(f'Zero inventory input={count} selected={zero}',flush=True)
    flush()
    assert count==919459 and zero==303054 and zero_words==535259
    inventory=dict(input_chunks=count,selected_zero_chunks=zero,original_zero_words=zero_words,batches=tasks,
                   selected_audio_seconds=sum(t['audio_seconds'] for t in tasks),
                   source=str(args.existing),device='cpu',workers=args.workers,jobs=args.jobs,
                   description='Exact full ROVER chunk audio and text; no Qwen-based audio crops; GPU hidden.')
    dump(args.output/'inventory.json',inventory);return inventory


def aggregate(args,inventory):
    summaries=[]
    for task in inventory['batches']:
        path=Path(task['path'])/'summary.json'
        if path.exists():summaries.append(json.loads(path.read_text()))
    totals={}
    for s in summaries:
        for k,v in s['totals'].items():totals[k]=totals.get(k,0)+v
    value=dict(complete=len(summaries)==len(inventory['batches']),completed_batches=len(summaries),
               total_batches=len(inventory['batches']),selected_chunks=inventory['selected_zero_chunks'],
               totals=totals,failed_chunks=sum(s['failed_chunks'] for s in summaries),device='cpu',
               scope='Pure MFA references on Qwen zero-duration chunks. Replacement validation is separate.')
    dump(args.output/'summary.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing',type=Path,default=ROOT/'results/open_yap_1k_rover_4gpu/qwen3-fa+mfa')
    p.add_argument('--output',type=Path,default=ROOT/'results/open_yap_midpoint_chunk/whole_mfa')
    p.add_argument('--scratch',type=Path,default=Path('/tmp/fa_rover_zero_whole_mfa_work'))
    p.add_argument('--batch-size',type=int,default=16384);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--jobs',type=int,default=32)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();inventory=prepare(args);aggregate(args,inventory)
    todo=[dict(t,scratch=str(args.scratch),jobs=args.jobs) for t in inventory['batches']
          if not (Path(t['path'])/'summary.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_batch,t) for t in todo]
        for future in as_completed(futures):
            future.result();result=aggregate(args,inventory)
            print(f'Whole MFA complete {result["completed_batches"]}/{result["total_batches"]}',flush=True)
    result=aggregate(args,inventory);result['wall_seconds']=time.perf_counter()-started
    dump(args.output/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
