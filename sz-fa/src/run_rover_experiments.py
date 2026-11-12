"""Run four-GPU streaming cascade, then the two dual-backend ablations."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from rover_worker import Totals

ROOT=Path(__file__).resolve().parents[1]
MODES=[('qwen+mfa+nyra','qwen3-fa+mfa+nyra'),('qwen+mfa','qwen3-fa+mfa'),('qwen+nyra','qwen3-fa+nyra')]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def aggregate(directory,shards,seconds):
    stages={};phases=collections.Counter();routes=collections.Counter();workers=[]
    for shard in shards:
        worker=directory/f'gpu_{shard["gpu"]}'
        runtime=json.loads((worker/'runtime.json').read_text())
        if runtime['samples']!=shard['chunks']:
            raise RuntimeError('Incomplete shard '+str(worker))
        if abs(runtime['audio_seconds']-shard['audio_seconds'])>0.01:
            raise RuntimeError('Shard audio duration mismatch '+str(worker))
        workers.append(runtime)
        for name,stats in json.loads((worker/'error_stats.json').read_text()).items():
            stages.setdefault(name,Totals()).add(stats)
        phases.update(runtime['phases']);routes.update(runtime['routes'])
    total_audio=sum(w['audio_seconds'] for w in workers)
    result=dict(method=workers[0]['method'],gpu_count=len(shards),samples=sum(w['samples'] for w in workers),
        audio_seconds=total_audio,audio_hours=total_audio/3600,wall_seconds=seconds,wall_rtf=seconds/total_audio,
        sum_worker_seconds=sum(w['seconds'] for w in workers),
        worker_work_rtf=sum(w['seconds'] for w in workers)/total_audio,
        phases_sum_worker_seconds=dict(phases),routes=dict(routes),workers=workers)
    result['errors']={name:total.value() for name,total in stages.items()}
    write(directory/'summary.json',result)
    write(directory/'results.json',dict(layout='four GPU workers, each with streaming blocks',
        prediction_shards=[str(path.relative_to(directory)) for path in sorted(directory.glob('gpu_*/blocks/*/predictions.jsonl'))],
        audit_shards=[str(path.relative_to(directory)) for path in sorted(directory.glob('gpu_*/blocks/*/alignment_predictions.jsonl'))]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--python',default='/mnt/data/L202500293/obed/envs/qwenasr/bin/python')
    p.add_argument('--block-size',type=int,default=32768)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--mfa-jobs',type=int,default=32)
    a=p.parse_args()
    preparation=json.loads((a.input/'preparation.json').read_text())
    if preparation['invalid_rows'] or preparation['missing_audio']:
        raise ValueError('Input inventory has unresolved errors')
    a.output.mkdir(parents=True,exist_ok=False)
    a.scratch.mkdir(parents=True,exist_ok=True)
    source_names=['src/prepare_rover.py','src/rover_worker.py','src/run_rover_experiments.py',
                  'src/qwen3-fa/core.py','src/qwen3-fa/process.py','src/qwen3-fa/sentences.py',
                  'src/qwen3-fa/prediction_format.py',
                  'src/qwen3-fa/nyra_repair.py','src/qwen3-fa+mfa/repair.py','src/compare_results.py']
    code_hashes={}
    for name in source_names:
        source=ROOT/name;destination=a.output/'code_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        code_hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(a.output/'experiment_design.json',dict(command=sys.argv,preparation=str((a.input/'preparation.json').resolve()),
        methods=[m for m,_ in MODES],model_residency='one Qwen and lazy Nyra per GPU per experiment; reused between blocks',
        assignment=preparation['shards'],block_size=a.block_size,batch_size=a.batch_size,mfa_jobs_per_worker=a.mfa_jobs,
        code_sha256=code_hashes,
        batching='read source audio in recording order, infer in duration buckets, restore manifest order',
        io_threads_per_worker=8,
        transcript='unchanged fused ROVER text; whole VAD chunk fallback when punctuation is absent',
        scratch_policy='archive JSONL decisions and gzip raw MFA outputs; delete reconstructable crops and Kaldi intermediates',
        rtf_scope='parent elapsed from launching all workers through completion and summary IO / sum unique annotated segment seconds'))
    results=[]
    for method,name in MODES:
        started=time.perf_counter()
        directory=a.output/name;directory.mkdir()
        processes=[]
        logs=[]
        for shard in preparation['shards']:
            gpu=shard['gpu'];env=dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            command=[a.python,'-u',str(ROOT/'src/rover_worker.py'),'--input',str((a.input/f'gpu_{gpu}.jsonl').resolve()),
                '--output',str((directory/f'gpu_{gpu}').resolve()),'--scratch',str((a.scratch/name/f'gpu_{gpu}').resolve()),
                '--method',method,'--batch-size',str(a.batch_size),'--block-size',str(a.block_size),'--mfa-jobs',str(a.mfa_jobs)]
            log=(directory/f'gpu_{gpu}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env))
        print(f'START {method}: 4 GPUs, {preparation["unique_chunks"]} chunks',flush=True)
        while any(process.poll() is None for process in processes):
            progress=[]
            for shard,process in zip(preparation['shards'],processes):
                path=directory/f'gpu_{shard["gpu"]}'/'progress.json'
                try:value=json.loads(path.read_text())
                except (FileNotFoundError,json.JSONDecodeError):value={}
                progress.append(dict(gpu=shard['gpu'],exit_code=process.poll(),**value))
            write(a.output/'status.json',dict(method=method,elapsed_seconds=time.perf_counter()-started,
                completed_experiments=len(results),workers=progress))
            if any(process.poll() not in (None,0) for process in processes):
                for process in processes:
                    if process.poll() is None:process.terminate()
                raise RuntimeError('A worker failed; inspect GPU logs before continuing ablations')
            time.sleep(5)
        for log in logs:log.close()
        if any(process.returncode for process in processes):raise RuntimeError('Worker failure')
        result=aggregate(directory,preparation['shards'],time.perf_counter()-started)
        result['qwen_matches_first_experiment']=(not results or
            [w['qwen_output_sha256'] for w in result['workers']]==[w['qwen_output_sha256'] for w in results[0]['workers']])
        # Include summary-writing overhead in wall RTF.
        result['wall_seconds']=time.perf_counter()-started
        result['wall_rtf']=result['wall_seconds']/result['audio_seconds']
        write(directory/'summary.json',result)
        results.append(result)
        write(a.output/'suite_summary.json',dict(preparation_seconds=preparation['seconds'],experiments=results))
        persistent=ROOT/'results/open_yap_1k_summary';persistent.mkdir(exist_ok=True)
        shutil.copyfile(a.output/'suite_summary.json',persistent/'suite_summary.json')
        shutil.copyfile(a.output/'experiment_design.json',persistent/'experiment_design.json')
        print(json.dumps(dict(method=method,wall_seconds=result['wall_seconds'],rtf=result['wall_rtf'],errors=result['errors'])),flush=True)
    write(a.output/'status.json',dict(status='complete',experiments=len(results)))
    print('All streaming experiments complete',flush=True)


if __name__=='__main__':main()
