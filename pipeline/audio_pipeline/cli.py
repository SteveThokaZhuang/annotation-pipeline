from .relocation import transform, relocated_path
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
import yaml
from .common import ROOT,STAGES,atomic,digest,fingerprint,read,validate
from .planning import descriptor,head_results,items
from .state import State
from .writer import writer_service
from .resume import checked_tasks, ScanStopped
from .finalization import StopCheck,collect_recording,export_recordings
from .failure_policy import ASR_STAGES,SUCCESS,export_failures,failure_record,hypothesis_error,persist,is_infrastructure_error,InfrastructureError


def parser():
    p=argparse.ArgumentParser(description='Independent, resumable audio stages. Tests never start full execution.')
    p.add_argument('stages',nargs='*',help='test STAGE | STAGE... | all | status | stop | sync | workers-stop')
    p.add_argument('--datasets','--dataset-yaml',dest='datasets',type=Path,help='Dataset catalog YAML (paths relative to the catalog)')
    p.add_argument('--dataset');p.add_argument('--config',type=Path,default=ROOT/'config/pipeline.yaml')
    p.add_argument('--resources',type=Path,default=ROOT/'config/resources.yaml')
    p.add_argument('--output-root',type=Path);p.add_argument('--model')
    p.add_argument('--limit',type=int);p.add_argument('--test-id',default='smoke')
    p.add_argument('--gpus',help='Physical IDs, e.g. 0,1; cpu for explicit CPU execution')
    p.add_argument('--workers',type=int);p.add_argument('--cpu-workers',type=int)
    p.add_argument('--batch-size',type=int,help='Maximum inference batch count per worker (DNSMOS: windows)')
    p.add_argument('--adaptive',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--prepare-workers',type=int,help='Preparation workers per inference worker; VAD and ASR2/3 use CPU processes (0 disables VAD preparation)')
    p.add_argument('--prefetch-batches',type=int)
    p.add_argument('--cache',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--cache-gib',type=float)
    p.add_argument('--resume-scan-workers',type=int,help='Concurrent saved-result readers (default 16; independent of GPU workers)')
    p.add_argument('--resume-validation',choices=['trusted','full'],help='Trust committed checkpoints, or fully validate saved results and sources')
    p.add_argument('--resume-from-stage',choices=STAGES,help='Start all at this stage using saved earlier outputs; skip retrying earlier stage errors')
    p.add_argument('--max-padded-seconds',type=float)
    p.add_argument('--background',action='store_true',help='Run this command under nohup, printing a log path')
    p.add_argument('--result-batch-size',type=int,help='Results per checkpoint: disagreement default 4096; ROVER fusion/checkpoint default 128')
    p.add_argument('--io-workers',type=int);p.add_argument('--threads-per-worker',type=int)
    p.add_argument('--max-concurrent-stages',type=int)
    p.add_argument('--language');p.add_argument('--threshold',type=float)
    policy=p.add_mutually_exclusive_group();policy.add_argument('--enhance',action='store_true',default=None)
    policy.add_argument('--no-enhance',dest='enhance',action='store_false')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true',help='Completed valid results are always reused')
    p.add_argument('--dataset-list',nargs='+',help='Process datasets stage by stage, reusing resident models')
    p.add_argument('--explain-config',action='store_true',help='Print resolved configuration and exit')
    return p


def allocation(stage,resources,args,spec):
    entry=resources['stages'].get(stage,{})
    workers=args.workers or entry.get('workers',entry.get('cpu_workers',1))
    budget=args.cpu_workers or resources['cpu']['workers']
    io_workers=args.io_workers or resources['cpu']['io_workers']
    threads=args.threads_per_worker or resources['cpu']['threads_per_worker']
    gpus=list(entry.get('gpu_devices',[]))
    if args.gpus is not None:
        requested=[] if args.gpus.lower() in ['cpu','none',''] else [int(g) for g in args.gpus.split(',')]
        # --gpus is a global allowed set for all/sequence, a direct assignment for single stages.
        if len(args.stages)==1 and args.stages[0]!='all' and not getattr(args,'_global_gpu_set',False):gpus=requested
        else:
            gpus=[g for g in gpus if g in requested]
            if not gpus and spec.get('requires_gpu') and requested:gpus=[requested[0]]
    if len(gpus)!=len(set(gpus)) or any(g<0 for g in gpus):raise ValueError('GPU IDs must be unique nonnegative integers')
    if spec.get('requires_gpu') and not gpus:
        raise ValueError(stage+' / '+spec['model']+' requires a GPU in its verified implementation; select --gpus ID')
    if stage in ['ingest','disagreement','rover'] and gpus:raise ValueError(stage+' is CPU-only; use --gpus cpu')
    if args.gpus is None and any(g not in resources['gpu']['devices'] for g in gpus):raise ValueError('Stage GPU assignment is outside gpu.devices')
    slots=resources['scheduler']['max_concurrent_gpu_jobs_per_device']
    if min(workers,budget,io_workers,threads,slots)<1:raise ValueError('Resource counts must be positive')
    core_budget=resources['cpu'].get('cores',budget*threads)
    if workers>budget or workers*threads>core_budget:raise ValueError('Stage workers exceed CPU core budget')
    if gpus and workers>len(gpus)*slots:raise ValueError('Workers exceed configured GPU slots')
    defaults={'ingest':128,'vad':64,'dnsmos':32,'enhance':8,'disagreement':4096,'rover':256,'align':64}
    override=getattr(args,'result_batch_size',None)
    result_batch_size=override if override is not None else entry.get('result_batch_size',defaults.get(stage,64))
    if not isinstance(result_batch_size,int) or isinstance(result_batch_size,bool) or result_batch_size<1:
        raise ValueError('result_batch_size must be a positive integer')
    optimization=dict(resources.get('optimization',{}))
    optimization['resume']=dict(optimization.get('resume',{}))
    if getattr(args,'resume_validation',None) is not None:
        optimization['resume']['validation']=args.resume_validation
    optimization['cache']=dict(optimization.get('cache',{}))
    # Stage defaults are applied before explicit command-line overrides.
    optimization['cache'].update(entry.get('cache',{}))
    optimization['adaptive']=dict(optimization.get('adaptive',{}))
    optimization['adaptive'].update(entry.get('batch_policy',{}))
    if entry.get('task_dispatch'):optimization['task_dispatch']=dict(optimization.get('task_dispatch',{}),**entry['task_dispatch'])
    if 'feature_transport' in entry:optimization['feature_transport']=dict(optimization.get('feature_transport',{}),**entry['feature_transport'])
    if getattr(args,'resume_scan_workers',None) is not None:
        optimization['resume_scan_workers']=args.resume_scan_workers
    if getattr(args,'cache',None) is not None:optimization['cache']['enabled']=args.cache
    if getattr(args,'cache_gib',None) is not None:optimization['cache']['max_gib']=args.cache_gib
    if getattr(args,'max_padded_seconds',None) is not None:
        optimization['adaptive']['max_padded_seconds']=args.max_padded_seconds
        optimization['adaptive']['models']={name:dict(value,max_padded_seconds=args.max_padded_seconds)
            for name,value in optimization['adaptive'].get('models',{}).items()}
    prepare=getattr(args,'prepare_workers',None)
    if prepare is None:prepare=entry.get('prepare_workers',2)
    prefetch=getattr(args,'prefetch_batches',None) or entry.get('prefetch_batches',2)
    if not isinstance(prepare,int) or isinstance(prepare,bool) or prepare<(0 if stage=='vad' else 1) or prefetch<1:
        raise ValueError('Invalid preparation counts')
    vad_prepare=prepare if stage=='vad' and ('prepare_workers' in entry or getattr(args,'prepare_workers',None) is not None) else 0
    helpers=vad_prepare if stage=='vad' else spec.get('settings',{}).get('frontend_workers',prepare if resources.get('version')==3 and stage in ('asr1','align') else 0)
    if stage in ('dnsmos','enhance'):helpers=spec.get('settings',{}).get('prepare_workers',2)
    cost=workers*(threads+helpers)
    if cost>core_budget:raise ValueError('CPU preparation exceeds the CPU budget')
    if stage=='vad' and (workers*vad_prepare>budget or cost>core_budget):
        raise ValueError('VAD CPU preparation exceeds the CPU budget; reduce --prepare-workers or raise --cpu-workers')
    lookahead=entry.get('prepare_lookahead',max(1,vad_prepare*2))
    max_gib=entry.get('prepare_max_gib',2)
    if stage=='vad' and (not isinstance(lookahead,int) or isinstance(lookahead,bool) or lookahead<1 or max_gib<=0):
        raise ValueError('VAD preparation lookahead and prepared-buffer budget must be positive')
    decoded=optimization.get('decoded_ram_gib_per_process',0)
    queue_cost=workers*(max_gib+decoded*(1+helpers)+optimization.get('transfers',{}).get('max_pinned_gib_per_worker',0)+
        optimization.get('stage_pipelines',{}).get(spec['model'],{}).get('max_gib',0)+
        (optimization.get('funasr_pipeline',{}).get('max_gib',0) if spec['model']=='funasr' else 0))
    memory=optimization.get('memory',{})
    if queue_cost>memory.get('queue_budget_gib',float('inf')):raise ValueError(stage+' estimated queues/caches exceed memory.queue_budget_gib')
    return dict(workers=workers,threads_per_worker=threads,cpu_workers=budget,io_workers=io_workers,
                gpu_devices=gpus,gpu_slots=slots,
                result_batch_size=result_batch_size,prepare_workers=prepare,prefetch_batches=prefetch,
                inference_batch_size=(spec['settings'].get('batch_size',1) if stage.startswith('asr')
                                      else entry.get('inference_batch_size',1)),optimization=optimization,
                prefetch_recordings=entry.get('prefetch_recordings',True),cpu_cost=cost,
                prepare_max_gib=max_gib,vad_prepare_processes=vad_prepare,vad_prepare_lookahead=lookahead,vad_prepare_max_gib=max_gib,
                prepare_lookahead=entry.get('prepare_lookahead'),prepare_order=entry.get('prepare_order','input'),ready_wait_seconds=entry.get('ready_wait_seconds',0),
                checkpoint=entry.get('checkpoint',{}),memory_cost_gib=queue_cost,cpu_core_budget=core_budget)


@contextmanager
def stage_lock(root,stage):
    from .working_cache import canonical_path, _active
    lock_root=root if _active is not None and not _active.sync_enabled else canonical_path(root)
    directory=Path(lock_root)/'locks';directory.mkdir(parents=True,exist_ok=True)
    with open(directory/(stage+'.lock'),'a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process owns this dataset/stage: '+stage)
        yield


def stage_progress(state,stage,model,started_at,total,cached):
    counts=dict(state.db.execute(
        'SELECT status, COUNT(*) FROM tasks WHERE stage=? AND model=? AND updated>=? GROUP BY status',
        (stage,model,started_at)).fetchall())
    completed=counts.get('complete',0)
    failed=counts.get('failed',0)
    running=counts.get('running',0)
    remaining=max(0,total-cached-completed-failed)
    return dict(stage=stage,event='progress',percent=round(100*(cached+completed+failed)/total,2) if total else 100.0,
                completed=cached+completed,total=total,remaining=remaining,running=running,
                waiting=max(0,remaining-running),failed=failed,cached=cached)


def run_stage(stage,cfg,resources,args,root,run_dir,stop_file,enhanced,tolerate_failures=False,task_items_override=None,spec_override=None):
    scratch=resources.get('optimization',{}).get('scratch_root','/tmp/audio-pipeline-optimized')
    with stage_lock(root,stage),writer_service(root,scratch) as writer_socket:
        tick=time.perf_counter();state=State(root)
        spec=spec_override or descriptor(stage,cfg,args);alloc=allocation('enhance' if stage=='enhance_parts' else stage,resources,args,spec)
        if spec['settings'].get('adaptive'):
            from .batching import resolved_policy
            spec=dict(spec,settings=dict(spec['settings'],batch_policy=resolved_policy(alloc,spec['model'],ROOT)))
        scan_workers=alloc.get('optimization',{}).get('resume_scan_workers',16)
        state.manifest_cache_bytes=int(alloc.get('optimization',{}).get('manifest_cache_gib',0)*2**30)
        state.scan_workers=scan_workers;state.stop_file=stop_file;state.scan_scratch=scratch
        state.resume_validation=alloc.get('optimization',{}).get('resume',{}).get('validation','full')
        trusted=state.resume_validation=='trusted'
        cfg=dict(cfg,_discovery=resources.get('optimization',{}).get('discovery',{}))
        from . import resume_checkpoint
        resume_context=resume_checkpoint.context(state,stage,spec,cfg,args,enhanced) if task_items_override is None else None
        tasks=[];pending=[];jobs=[];cached_count=0
        def stopped_report():
            workers=[read(job['report']) for job in jobs if Path(job['report']).exists()]
            report=dict(stage=stage,model=spec['model'],status='stopped',fatal_error=False,
                        infrastructure_failure=False,ready_for_full_execution=False,
                        runtime_seconds=time.perf_counter()-tick,tasks=len(tasks),
                        completed=cached_count+sum(w.get('completed',0) for w in workers),
                        cached=cached_count,failed=sum(w.get('failed',0) for w in workers),
                        allocation=alloc,workers=workers,worker_errors=[],manifest=None,
                        note='Stopped without publishing a manifest; durable checkpoints and the previous manifest are retained.')
            atomic(run_dir/(stage+'-report.json'),report)
            state.db.close()
            print(json.dumps({k:v for k,v in report.items() if k!='workers'}),flush=True)
            return report
        if stop_file.exists():return stopped_report()
        previous=resume_checkpoint.load(state,stage,spec['model'],resume_context) if trusted else None
        if previous is not None:
            report=dict(previous,runtime_seconds=time.perf_counter()-tick,cached=previous['tasks'],
                allocation=alloc,workers=[],worker_errors=[],manifest_reused=True,resume_validation='trusted',
                resume_checkpoint_reused=True)
            report['failure_files']=export_failures(run_dir/(stage+'-failures'),[])
            atomic(run_dir/(stage+'-report.json'),report)
            state.db.close()
            print(json.dumps(dict(stage=stage,event='progress',phase='resume_checkpoint',
                cached=report['cached'],total=report['tasks'],result_files_read=0,
                elapsed_seconds=round(time.perf_counter()-tick,1))),flush=True)
            print(json.dumps(report),flush=True)
            return report
        from . import stage_plan
        try:stored_plan=stage_plan.load(state,stage,spec['model'],resume_context,stop_file) if trusted else None
        except ScanStopped:return stopped_report()
        settings=dict(spec['settings'])
        from .cache_version import inference_descriptor
        signature_base=digest(inference_descriptor(spec))
        tasks=[];next_stop_poll=0.
        if stored_plan is not None:
            tasks=stored_plan
            print(json.dumps(dict(stage=stage,event='progress',phase='resume_plan',tasks=len(tasks),
                result_files_read=0,elapsed_seconds=round(time.perf_counter()-tick,1))),flush=True)
        else:
            try:task_items=task_items_override if task_items_override is not None else items(stage,cfg,args,state,enhanced)
            except ScanStopped:return stopped_report()
            if args.limit is not None and task_items_override is None:task_items=task_items[:args.limit]
            for item in task_items:
                now=time.monotonic()
                if stage!='disagreement' or now>=next_stop_poll:
                    next_stop_poll=now+.1
                    if stop_file.exists():return stopped_report()
                identity=item['id']
                signature=digest(dict(adapter=signature_base,input=item))
                directory=Path(root)/stage/spec['model']/identity/signature[:20]
                output=directory/'result.json'
                item=dict(item,_artifact_dir=str(directory)) if stage=='enhance' else item
                tasks.append(dict(item=item,signature=signature,output=str(output)))
            if stop_file.exists():return stopped_report()
            stage_plan.save(state,stage,spec['model'],resume_context,tasks)
        # Stage adapters bound their preparation; ASR/alignment decode requested slices.
        finalization=alloc['optimization'].get('finalization',{})
        reuse=None
        if trusted or (stage=='ingest' and finalization.get('reuse_unchanged_ingest_manifest',False)):
            from .ingest_reuse import ManifestReuse
            reuse=ManifestReuse(state,tasks,settings,args.dataset,args.limit,enhanced,stage,spec['model'])
        incremental=None
        if trusted and stage in ASR_STAGES:
            from .incremental_resume import ASRPublication
            incremental=ASRPublication(reuse.manifest,stage,spec['model'],settings,tasks,args.dataset,args.limit,enhanced)
        pending=[]
        try:
            if trusted:
                from .resume import committed_tasks
                for task,recorded in committed_tasks(state,stage,spec['model'],tasks,stop_file):
                    if reuse is not None:reuse.observe(task,recorded,None,trusted=True)
                    if incremental is not None:incremental.observe(task,recorded)
                    if recorded and recorded[0]=='complete' and recorded[1]:cached_count+=1
                    else:pending.append(task)
            else:
                for task,recorded,value in checked_tasks(state,stage,spec['model'],tasks,
                        workers=scan_workers,stop_file=stop_file):
                    if reuse is not None:reuse.observe(task,recorded,value)
                    if value is None or (stage in ASR_STAGES and hypothesis_error(value)):
                        pending.append(task)
                    else:cached_count+=1
        except ScanStopped:return stopped_report()
        if reuse is not None and reuse.reusable and not pending and not state.skipped and not state.dependency_failures:
            check=StopCheck(stop_file)
            try:
                check(force=True)
                print(json.dumps(dict(stage=stage,event='progress',phase='finalizing',step='reuse_manifest',total=len(tasks),checked=len(tasks),manifest=str(reuse.path))),flush=True)
                from .accounting import write_ledger
                print(json.dumps(dict(stage=stage,event='progress',phase='finalizing',step='update_ledger',manifest_reused=True)),flush=True)
                write_ledger(root,state,stage,reuse.entries,state.skipped,
                             incremental=finalization.get('incremental_recording_ledger',False))
                check(force=True)
                failures=export_failures(run_dir/(stage+'-failures'),[])
                latest=export_failures(Path(root)/stage/spec['model']/'failures',[])
                report=dict(stage=stage,model=spec['model'],status='passed',fatal_error=False,infrastructure_failure=False,
                    ready_for_full_execution=bool(tasks),runtime_seconds=time.perf_counter()-tick,tasks=len(tasks),
                    completed=len(tasks),cached=len(tasks),failed=0,no_transcript=0,skipped=0,upstream_failures=0,
                    allocation=alloc,workers=[],worker_errors=[],manifest=str(reuse.path),manifest_reused=True,
                    resume_validation=state.resume_validation,failure_files=failures,latest_failure_files=latest)
                atomic(run_dir/(stage+'-report.json'),report)
                resume_checkpoint.save(state,stage,spec,cfg,args,enhanced,resume_context,report)
                state.db.close()
                print(json.dumps({k:v for k,v in report.items() if k!='workers'}),flush=True)
                return report
            except ScanStopped:return stopped_report()
        if stage=='enhance' and spec['settings'].get('parallel_segments') and pending:
            from .enhancement_dispatch import parts_for
            parts=parts_for(pending,spec)
            if parts:run_stage('enhance_parts',cfg,resources,args,root,run_dir,stop_file,enhanced,True,task_items_override=parts,spec_override=spec)
            if stop_file.exists():return stopped_report()
        worker_count=min(alloc['workers'],len(pending))
        from .scheduling import balanced_shards
        shards=balanced_shards(pending,worker_count,by_recording=stage.startswith('asr') or stage=='align')
        dispatch=alloc['optimization'].get('task_dispatch',{})
        queue_path=None
        if dispatch.get('enabled') and alloc['gpu_devices'] and pending:
            if not alloc['optimization'].get('persistent_workers',{}).get('enabled'):
                raise ValueError('task_dispatch requires persistent_workers.enabled')
            from .task_queue import create_queue
            queue_path=run_dir/(stage+'-queue.sqlite')
            create_queue(queue_path,pending,dispatch,workers=worker_count)
        jobs=[];processes=[]
        started_at=time.time()
        for index in range(worker_count):
            if stop_file.exists():break
            device=alloc['gpu_devices'][index%len(alloc['gpu_devices'])] if alloc['gpu_devices'] else None
            job=dict(spec,writer_socket=writer_socket,root=str(root),stage=stage,device=device,threads=alloc['threads_per_worker'],
                allocation=alloc,worker_index=index,tasks=shards[index],
                stop_file=str(stop_file),lock_root=resources['scheduler'].get('gpu_lock_root',str(Path(cfg['output_root'])/'control/gpu_locks')),
                gpu_slots=alloc['gpu_slots'],report=str(run_dir/f'{stage}-worker-{index}.json'))
            if queue_path:job['task_queue']=str(queue_path)
            from .task_manifest import write_tasks
            task_path=run_dir/f'{stage}-tasks-{index}.jsonl'
            write_tasks(task_path,job.pop('tasks'))
            job['tasks_file']=str(task_path)
            job_path=run_dir/f'{stage}-job-{index}.json';atomic(job_path,job);jobs.append(job)
            env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),
                     OMP_NUM_THREADS=str(alloc['threads_per_worker']),MKL_NUM_THREADS=str(alloc['threads_per_worker']),
                     OPENBLAS_NUM_THREADS=str(alloc['threads_per_worker']),
                     CUDA_VISIBLE_DEVICES='' if device is None else str(device))
            log=open(run_dir/f'{stage}-worker-{index}.log','w')
            resident=alloc['optimization'].get('persistent_workers',{}).get('enabled') and device is not None
            command=([sys.executable,'-m','audio_pipeline.resident','client',str(job_path)] if resident else
                     [spec['python'],'-m','audio_pipeline.worker',str(job_path)])
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((proc,log))
        stopping_since=None;last_log=time.monotonic()
        print(json.dumps(dict(stage_progress(state,stage,spec['model'],started_at,len(tasks),len(tasks)-len(pending)),
                              active_workers=sum(proc.poll() is None for proc,_ in processes),
                              elapsed_seconds=round(time.perf_counter()-tick,1))),flush=True)
        try:
            while any(proc.poll() is None for proc,_ in processes):
                from .working_cache import check_health
                check_health()
                for job,(proc,_) in zip(jobs,processes):
                    if proc.poll() not in (None,0,130) and Path(job['report']).exists():
                        if read(job['report']).get('infrastructure_failure'):stop_file.touch()
                if stop_file.exists():
                    if stopping_since is None:
                        stopping_since=time.monotonic()
                        for proc,_ in processes:
                            if proc.poll() is None:proc.terminate()
                    elif time.monotonic()-stopping_since>resources['scheduler']['shutdown_timeout_seconds']:
                        for proc,_ in processes:
                            if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
                if time.monotonic()-last_log>alloc.get('optimization',{}).get('monitoring',{}).get('report_seconds',15):
                    print(json.dumps(dict(stage_progress(state,stage,spec['model'],started_at,len(tasks),len(tasks)-len(pending)),
                                          active_workers=sum(proc.poll() is None for proc,_ in processes),
                                          elapsed_seconds=round(time.perf_counter()-tick,1))),flush=True)
                    last_log=time.monotonic()
                time.sleep(.25)
        finally:
            for proc,log in processes:
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()
        print(json.dumps(dict(stage_progress(state,stage,spec['model'],started_at,len(tasks),len(tasks)-len(pending)),
                              phase='finalizing',active_workers=sum(proc.poll() is None for proc,_ in processes),
                              elapsed_seconds=round(time.perf_counter()-tick,1))),flush=True)
        early_workers=[read(job['report']) for job in jobs if Path(job['report']).exists()]
        if any(worker.get('infrastructure_failure') for worker in early_workers):
            completed=cached_count+sum(worker.get('completed',0)+worker.get('cached',0) for worker in early_workers)
            report=dict(stage=stage,model=spec['model'],status='failed',fatal_error=True,
                infrastructure_failure=True,ready_for_full_execution=False,
                runtime_seconds=time.perf_counter()-tick,tasks=len(tasks),completed=completed,
                cached=cached_count,failed=sum(worker.get('failed',0) for worker in early_workers),
                pending=max(0,len(tasks)-completed),allocation=alloc,workers=early_workers,manifest=None,
                note='Worker infrastructure failure: final scan/publication skipped; resume retains committed checkpoints.')
            atomic(run_dir/(stage+'-report.json'),report)
            state.db.close()
            raise InfrastructureError(stage+' worker infrastructure failed; inspect '+str(run_dir/(stage+'-report.json')))
        if stop_file.exists():return stopped_report()
        incremental=incremental if incremental is not None and incremental.reusable else None
        entries=list(incremental.entries) if incremental else []
        finalize_tasks=incremental.changed if incremental else tasks
        if incremental:
            print(json.dumps(dict(stage=stage,event='progress',phase='incremental_finalization',
                total=len(tasks),reused_entries=len(tasks)-len(finalize_tasks),checking=len(finalize_tasks))),flush=True)
        failure_rows=[];no_transcript=0;route_values=[];assembled={}
        final_stop=StopCheck(stop_file)
        item_index={task['item']['id']:task['item'] for task in tasks}
        try:
            for task,recorded,value in checked_tasks(state,stage,spec['model'],finalize_tasks,
                    workers=scan_workers,stop_file=stop_file,phase='finalizing'):
                item=task['item'];identity=item['id']
                error=recorded[2] if recorded else 'Worker exited before task completion'
                if value is not None and stage in ASR_STAGES and hypothesis_error(value):
                    error=hypothesis_error(value);value=None
                if value is None:
                    error=error or 'Worker exited without a valid completed result'
                    failure_rows.append(failure_record(args.dataset,stage,spec['model'],item,error,signature=task['signature']))
                elif value.get('transcript_status')=='no_transcript':
                    no_transcript+=1
                    failure_rows.append(failure_record(args.dataset,stage,spec['model'],item,value.get('error'),status='no_transcript',signature=task['signature']))
                if value is not None and stage=='disagreement':route_values.append(value)
                if value is not None and stage in ['rover','align']:collect_recording(assembled,value)
                metadata=failure_record(args.dataset,stage,spec['model'],item,'')
                entry=dict(id=identity,error=error if value is None else None,
                    input={k:metadata[k] for k in ['recording_id','source_path','channel','start','end','start_sample','end_sample','duration']},
                    signature=task['signature'],output=recorded[1] if value is not None else task['output'],
                    status='complete' if value is not None else 'failed',
                    result_status=value.get('transcript_status') if value is not None else 'failed',
                    digest=digest(value) if value is not None else None)
                if incremental:entries[incremental.positions[identity]]=entry
                else:entries.append(entry)
        except ScanStopped:return stopped_report()
        workers=[read(job['report']) for job in jobs if Path(job['report']).exists()]
        worker_errors=[dict(worker=index,returncode=proc.returncode,log=str(run_dir/f'{stage}-worker-{index}.log'))
                       for index,(proc,_) in enumerate(processes) if proc.returncode not in [0,None]]
        fatal=any(w.get('fatal_error') or w['status']=='failed' for w in workers)
        infrastructure_failure=any(w.get('infrastructure_failure') for w in workers)
        if len(workers)!=len(jobs) and stage not in ASR_STAGES:fatal=True
        failed=sum(e['status']!='complete' for e in entries)
        upstream=list(state.dependency_failures.values())
        for issue in upstream:
            item=item_index.get(issue['id']) or issue.get('source') or {'id':issue['id']}
            if 'id' not in item:item=dict(item,id=issue['id'])
            failure_rows.append(failure_record(args.dataset,issue['stage'],issue['model'],item,issue['error'],status='upstream_failure'))
        for skipped in state.skipped:
            failure_rows.append(failure_record(args.dataset,stage,spec['model'],skipped,skipped['reason'],status='skipped'))
        degraded=bool(failed or no_transcript or upstream or state.skipped or worker_errors or
                      any(w['status']=='completed_with_errors' for w in workers))
        status='stopped' if stop_file.exists() else 'failed' if fatal else 'completed_with_errors' if degraded else 'passed'
        try:
            final_stop(force=True)
            print(json.dumps(dict(stage=stage,event='progress',phase='finalizing',step='write_manifest',total=len(tasks),checked=len(tasks))),flush=True)
            manifest=dict(stage=stage,model=spec['model'],entries=entries,settings=settings,enhanced=enhanced,
                          dataset=args.dataset,limited=args.limit,created=time.time(),status=status,
                          failed=failed,no_transcript=no_transcript,skipped=len(state.skipped))
            manifest_path=Path(root)/'manifests'/f'{stage}-{digest(manifest)[:20]}.json'
            final_stop(force=True)
            atomic(manifest_path,manifest)
            final_stop(force=True)
            failure_files=export_failures(run_dir/(stage+'-failures'),failure_rows)
            final_stop(force=True)
            latest_failure_files=export_failures(Path(root)/stage/spec['model']/'failures',failure_rows)
            final_stop(force=True)
            if stage=='disagreement':
                for label,high in [('low',False),('high',True)]:
                    selected=dict(manifest=str(manifest_path),chunks=[])
                    for value in route_values:
                        final_stop()
                        if value['high']==high:
                            selected['chunks'].append(dict(value['chunk'],routing_version=digest(value)))
                    final_stop(force=True)
                    atomic(manifest_path.with_suffix('.'+label+'.json'),selected)
                    final_stop(force=True)
                    atomic(Path(root)/'manifests'/f'{label}_disagreement.json',selected)
                    final_stop(force=True)
            if stage in ['rover','align']:
                try:vad_entries=state.head('vad')['entries']
                except RuntimeError:vad_entries=None
                expected_chunks={} if vad_entries is not None else None
                for entry in vad_entries or []:
                    if entry['status']!='complete':continue
                    for chunk in read(entry['output'])['chunks']:
                        key=chunk['recording_id']+f"_c{chunk['channel']}"
                        expected_chunks.setdefault(key,set()).add(chunk['id'])
                export_recordings(assembled,Path(root)/stage/'recordings',final_stop,write=atomic,
                                  workers=alloc.get('optimization',{}).get('export_workers',1),expected_chunks=expected_chunks)
            final_stop(force=True)
            if not fatal:
                state.set_head(stage,manifest_path)
                from .accounting import write_ledger
                print(json.dumps(dict(stage=stage,event='progress',phase='finalizing',step='update_ledger',total=len(tasks),checked=len(tasks))),flush=True)
                write_ledger(root,state,stage,entries,state.skipped,
                             incremental=finalization.get('incremental_recording_ledger',False))
        except ScanStopped:return stopped_report()
        report=dict(stage=stage,model=spec['model'],status=status,fatal_error=fatal,infrastructure_failure=infrastructure_failure,
                    ready_for_full_execution=status=='passed' and bool(entries),runtime_seconds=time.perf_counter()-tick,
                    tasks=len(tasks),completed=len(tasks)-failed,failed=failed,cached=len(tasks)-len(pending),
                    allocation=alloc,workers=workers,worker_errors=worker_errors,manifest=str(manifest_path),
                    no_transcript=no_transcript,skipped=len(state.skipped),upstream_failures=len(upstream),
                    failure_files=failure_files,latest_failure_files=latest_failure_files,
                    note='A smoke pass verifies execution and structural checks; transcription quality needs labeled validation.')
        atomic(run_dir/(stage+'-report.json'),report)
        resume_checkpoint.save(state,stage,spec,cfg,args,enhanced,resume_context,report)
        print(json.dumps({k:v for k,v in report.items() if k!='workers'}),flush=True)
        if fatal and not stop_file.exists():
            error_type=InfrastructureError if infrastructure_failure else RuntimeError
            raise error_type(stage+' failed; inspect '+str(run_dir/(stage+'-report.json')))
        return report


def control(args,cfg,resources=None):
    output_root=str(Path(cfg['output_root']).absolute())
    paths=list((Path(output_root)/'control/runs').glob('*.json'))
    settings=(resources or {}).get('optimization',{}).get('working_cache',{})
    if settings.get('root'):
        paths.extend(Path(settings['root']).glob('*/control/runs/*.json'))
    rows=[]
    for p in sorted(paths):
        row=read(p);pid=row['pid']
        if row.get('output_root',output_root)!=output_root:continue
        try:
            current=Path(f'/proc/{pid}/stat').read_text().split()[21]
            live=current==row['proc_start'] and row['status']=='running'
        except (OSError,IndexError):live=False
        if args.dataset and row['dataset']!=args.dataset:continue
        row['live']=live
        if args.stages==['stop'] and live:
            Path(row['stop_file']).touch()
            row['stop_requested']=True
        rows.append(row)
    print(json.dumps(rows,indent=2))


def main(argv=None,_dataset_list=False):
    p=parser();args=p.parse_args(argv)
    args._global_gpu_set=_dataset_list
    if not args.stages and not args.explain_config:p.error('Specify at least one stage, or --explain-config')
    if args.dataset_list:
        if args.dataset:p.error('Use --dataset OR --dataset-list')
        original=list(sys.argv[1:] if argv is None else argv)
        pos=original.index('--dataset-list');del original[pos:pos+1+len(args.dataset_list)]
        if args.stages!=['all']:p.error('--dataset-list currently requires all')
        original.remove('all')
        tail=original
        from .configuration import load_config
        catalog,_=load_config(args.config,args.resources,args.datasets)
        for dataset in args.dataset_list:
            if dataset not in catalog['datasets']:p.error('Unknown dataset '+dataset)
        for stage in STAGES:
            for dataset in args.dataset_list:
                enhance=args.enhance if args.enhance is not None else catalog['datasets'][dataset].get('enhance',False)
                if stage=='enhance' and not enhance:continue
                main([stage,'--dataset',dataset,*tail],_dataset_list=True)
        return
    for key in ['limit','workers','batch_size','result_batch_size','cpu_workers','io_workers','threads_per_worker','max_concurrent_stages','prepare_workers','prefetch_batches','resume_scan_workers']:
        value=getattr(args,key)
        if key=='prepare_workers' and value==0:continue
        if value is not None and value<1:p.error(key+' must be positive')
    if args.cache_gib is not None and args.cache_gib<=0:p.error('cache_gib must be positive')
    if args.max_padded_seconds is not None and args.max_padded_seconds<=0:p.error('max_padded_seconds must be positive')
    args.config=Path(relocated_path(args.config));args.resources=Path(relocated_path(args.resources))
    from .configuration import load_config
    cfg,resources=load_config(args.config,args.resources,args.datasets)
    if args.explain_config:
        resolved=[]
        for stage in STAGES:
            spec=descriptor(stage,cfg,args)
            resolved.append(dict(stage=stage,settings=spec['settings'],sources=spec.get('configuration_sources',{}),allocation=allocation(stage,resources,args,spec)))
        print(json.dumps(dict(pipeline=cfg,resources=resources,effective_stages=resolved),indent=2));return
    if args.stages==['workers-stop']:
        from .resident import shutdown_all
        shutdown_all(resources['optimization']['ipc_root']);return
    if args.output_root:cfg['output_root']=str(args.output_root.resolve())
    if args.background:
        directory=(Path(resources['optimization']['scratch_root'])/'background' if resources.get('version')==3 else ROOT/'reports/background')
        directory.mkdir(parents=True,exist_ok=True)
        logfile=directory/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:6]+'.log')
        forwarded=[x for x in (sys.argv[1:] if argv is None else argv) if x!='--background']
        with logfile.open('a') as stream:
            process=subprocess.Popen([sys.executable,str(ROOT/'run.py'),*forwarded],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        print(json.dumps(dict(pid=process.pid,log=str(logfile))));return
    if args.stages in [['status'],['stop']]:return control(args,cfg,resources)
    test=args.stages[0]=='test'
    if test:args.stages=args.stages[1:]
    if not args.dataset or not re.fullmatch(r'[A-Za-z0-9_-]+',args.dataset):p.error('An explicit safe --dataset name is required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.test_id):p.error('Invalid --test-id')
    if not args.stages:p.error('Select at least one stage')
    args.stages=['align' if x=='alignment' else x for x in args.stages]
    all_stages=args.stages==['all']
    if args.resume_from_stage and not all_stages:p.error('--resume-from-stage requires all')
    if 'all' in args.stages and not all_stages:p.error('all must be used alone')
    if args.model and (all_stages or len(args.stages)!=1 or not args.stages[0].startswith('asr')):
        p.error('--model applies to a single ASR stage; configure assignments for all')
    root=Path(cfg['output_root'])/('tests/'+args.test_id if test else 'production')/args.dataset
    if args.stages==['sync']:
        from .working_cache import WorkingCache
        working_settings=resources.get('optimization',{}).get('working_cache',{})
        if not working_settings.get('enabled'):p.error('sync requires optimization.working_cache.enabled')
        if args.dry_run:
            print('Would synchronize retained working cache to '+str(root));return
        # An explicit sync command transfers retained results even in cache-only mode.
        with WorkingCache(root,dict(working_settings,sync_enabled=True)) as working:working.finish()
        return
    enhanced=args.enhance
    if enhanced is None:
        enhanced=bool(cfg['datasets'].get(args.dataset,{}).get('enhance',False) or 'enhance' in args.stages)
    stages=list(STAGES) if all_stages else args.stages
    if all_stages and not enhanced:stages.remove('enhance')
    if args.resume_from_stage:
        if args.resume_from_stage not in stages:p.error('Resume stage is not enabled in this run')
        stages=stages[stages.index(args.resume_from_stage):]
    if len(stages)!=len(set(stages)):p.error('Duplicate stages')
    if any(stage not in STAGES for stage in stages):p.error('Unknown stage; choose '+', '.join(STAGES))
    if test and args.limit is None and any(s not in ['disagreement','rover'] for s in stages):args.limit=5
    concurrent=args.max_concurrent_stages or resources['scheduler']['max_concurrent_stages']
    if concurrent<1 or resources['scheduler']['shutdown_timeout_seconds']<1:p.error('Scheduler limits must be positive')
    plan=[dict(stage=s,model=(spec:=descriptor(s,cfg,args))['model'],python=spec['python'],settings=spec['settings'],
               configuration_sources=spec.get('configuration_sources',{}),allocation=allocation(s,resources,args,spec)) for s in stages]
    print(json.dumps(dict(mode='test' if test else 'production',dataset=args.dataset,output=str(root),
                         enhanced_audio=enhanced,limit=args.limit,max_concurrent_stages=concurrent,plan=plan),indent=2),flush=True)
    if args.dry_run:return
    from .working_cache import WorkingCache
    working_settings=resources.get('optimization',{}).get('working_cache',{})
    if working_settings.get('enabled'):
        with WorkingCache(root,working_settings) as working:
            return execute_run(args,cfg,resources,working.cache,stages,test,all_stages,enhanced,concurrent,plan,working)
    return execute_run(args,cfg,resources,root,stages,test,all_stages,enhanced,concurrent,plan)


def execute_run(args,cfg,resources,root,stages,test,all_stages,enhanced,concurrent,plan,working=None):
    run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    run_dir=root/'runs'/run_id;run_dir.mkdir(parents=True)
    atomic(run_dir/'effective-configuration.json',dict(pipeline=cfg,resources=resources,plan=plan))
    stop_file=run_dir/'STOP'
    def stop(*_):stop_file.touch()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    control_root=working.home if working is not None and not working.sync_enabled else Path(cfg['output_root'])
    registration=control_root/'control/runs'/(run_id+'.json')
    registration_value=dict(pid=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        dataset=args.dataset,mode='test' if test else 'production',stages=stages,status='running',
        output_root=str(Path(cfg['output_root']).absolute()),
        stop_file=str(stop_file),run_dir=str(run_dir),started=time.time())
    atomic(registration,registration_value)
    error=None;infrastructure_failure=False;reports=[];tick=time.perf_counter()
    try:
        if not all_stages or concurrent==1:
            for stage in stages:
                if stop_file.exists():break
                reports.append(run_stage(stage,cfg,resources,args,root,run_dir,stop_file,enhanced,tolerate_failures=all_stages))
        else:
            dependencies={s:set() for s in stages}
            for s in ['dnsmos','vad','enhance']:dependencies[s]={'ingest'} if s in stages else set()
            if 'enhance' in stages and cfg['settings'].get('enhance',{}).get('method_version') in ('vad_chunked_v1','auto_v3'):
                dependencies['enhance'].add('vad')
            for s in ['asr1','asr2']:dependencies[s]={'vad'}|({'enhance'} if enhanced else set())
            dependencies.update(disagreement={'asr1','asr2'},asr3={'disagreement'},rover={'asr3'},align={'rover'})
            finished=set(STAGES)-set(stages);pending=list(stages);running={};used_cpu=0
            budget=resources['cpu'].get('cores',(args.cpu_workers or resources['cpu']['workers'])*(args.threads_per_worker or resources['cpu']['threads_per_worker']))
            memory_budget=resources.get('optimization',{}).get('memory',{}).get('queue_budget_gib',float('inf'))
            allocations={x['stage']:x['allocation'] for x in plan}
            with ThreadPoolExecutor(max_workers=concurrent) as pool:
                while pending or running:
                    if stop_file.exists():pending=[]
                    for stage in list(pending):
                        need=allocations[stage]['cpu_cost']
                        used_memory=sum(allocations[s]['memory_cost_gib'] for s,_ in running.values())
                        if dependencies.get(stage,set())<=finished and len(running)<concurrent and used_cpu+need<=budget and used_memory+allocations[stage]['memory_cost_gib']<=memory_budget:
                            future=pool.submit(run_stage,stage,cfg,resources,args,root,run_dir,stop_file,enhanced,True)
                            running[future]=(stage,need);used_cpu+=need;pending.remove(stage)
                    if not running:
                        if pending:raise RuntimeError('Scheduler cannot allocate pending stages within CPU budget')
                        break
                    done,_=wait(running,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        stage,need=running.pop(future);used_cpu-=need
                        try:reports.append(future.result());finished.add(stage)
                        except Exception:stop_file.touch();raise
    except Exception as exc:
        error=str(exc);infrastructure_failure=is_infrastructure_error(exc);print('Error: '+error,file=sys.stderr)
    finally:
        status='failed' if error or any(r['status']=='failed' for r in reports) else 'stopped' if stop_file.exists() else 'completed_with_errors' if any(r['status']=='completed_with_errors' for r in reports) else 'passed'
        summary=dict(status=status,test=test,dataset=args.dataset,runtime_seconds=time.perf_counter()-tick,
                     error=error,infrastructure_failure=infrastructure_failure,reports=reports,run_dir=str(run_dir),
                     failed_tasks=sum(r['failed'] for r in reports),no_transcript=sum(r.get('no_transcript',0) for r in reports),
                     failure_reports=[r['failure_files'] for r in reports if r.get('failure_files')])
        atomic(run_dir/'summary.json',summary)
        if working is not None:
            try:
                working.check_health()
                if status in ('passed','completed_with_errors') and 'align' in stages:
                    if not working.finish(stop_file.exists):
                        status='stopped';summary.update(status=status)
                        atomic(run_dir/'summary.json',summary)
                else:
                    message=('Working cache retained; pending synchronization will resume on the next launch: '
                             if working.sync_enabled else 'Cache-only output retained; synchronization is disabled: ')
                    print(message+str(working.cache),flush=True)
            except Exception as exc:
                error=str(exc);status='failed'
                summary.update(status=status,error=error,infrastructure_failure=True)
                atomic(run_dir/'summary.json',summary)
        registration_value.update(status=status,finished=time.time());atomic(registration,registration_value)
        print('Run report: '+str(run_dir/'summary.json'),flush=True)
        print('Pipeline status: '+status,flush=True)
    if error or status=='failed':sys.exit(1)
    if stop_file.exists():sys.exit(130)
