"""One long-lived adapter per worker; batched ASR results retain per-item state."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import hashlib
import tempfile
import subprocess
import sys
import threading
import time
import traceback
import uuid
from .common import ROOT, atomic, fingerprint, load_module, read, validate
from .state import State
from .writer import StateWriter,WriterStopped
from .batching import BatchPolicy,is_cuda_oom,resolved_policy
from .gpu_memory import observe_cuda_memory
from .prefetch import AudioPrefetch
from .failure_policy import ASR_STAGES,InfrastructureError,is_infrastructure_error,persist,hypothesis_error,SUCCESS

STOP=False
_INTEROP_SET=False

def stop(*_):
    global STOP
    STOP=True


def gpu_query(device):
    try:
        result=subprocess.run(['nvidia-smi','--id='+str(device),'--query-gpu=memory.free,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        return list(map(float,result.stdout.strip().split(',')))
    except (OSError,ValueError,subprocess.SubprocessError):return None


def run_job(job, resident=None):
    global STOP,_INTEROP_SET
    STOP=False
    if 'tasks_file' in job:
        from .task_manifest import read_tasks
        job['tasks']=list(read_tasks(job['tasks_file']))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]=str(job['threads'])
    os.environ['CUDA_VISIBLE_DEVICES']='' if job['device'] is None else str(job['device'])
    os.environ['AUDIO_PIPELINE_IO_WORKERS']=str(job['allocation']['io_workers'])
    optimization=job['allocation'].get('optimization',{})
    from .limits import check_host
    scratch=Path(optimization.get('scratch_root','/tmp/audio-pipeline-optimized'))
    scratch.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(scratch)
    ipc=Path(optimization.get('ipc_root',scratch))
    ipc.mkdir(parents=True,exist_ok=True)
    if len(os.fsencode(str(ipc.absolute())))+37>107 and job['settings'].get('engine')=='vllm':
        raise ValueError('IPC path exceeds the Linux socket limit; configure a short optimization.ipc_root')
    os.environ['VLLM_RPC_BASE_PATH']=str(ipc)
    os.environ['AUDIO_PIPELINE_TRANSFERS']=json.dumps(optimization.get('transfers',{}))
    os.environ['AUDIO_PIPELINE_FEATURE_TRANSPORT']=json.dumps(optimization.get('feature_transport',{}))
    # Persistent helper processes retain their initializer environment. Keep the
    # I/O semaphore domain stable across their jobs and across GPU owners.
    domain=str(ipc) if optimization.get('persistent_workers',{}).get('enabled') else job['stop_file']
    run_key=hashlib.sha256(domain.encode()).hexdigest()[:20]
    os.environ['AUDIO_PIPELINE_IO_LOCKS']=str(scratch/('io-locks-'+run_key))
    os.environ['AUDIO_PIPELINE_CACHE']=json.dumps(optimization.get('cache',{}))
    os.environ['AUDIO_PIPELINE_DECODED_RAM_GIB']=str(optimization.get('decoded_ram_gib_per_process',0))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    state=State(job['root']);lock=None
    writer=StateWriter(job['writer_socket'],stop_requested=lambda: STOP or Path(job['stop_file']).exists()) if job.get('writer_socket') else state
    prefetch=None;save_pool=None;save_future=None;adapter=None;pending_active=[];pending_values=[]
    save_lock=threading.RLock();flush_stop=threading.Event();flush_thread=None;flush_errors=[]
    phase=dict(read_prepare_wait_seconds=0.,persistence_seconds=0.,persistence_wait_seconds=0.)
    report=dict(phase='initializing',status='running',pid=os.getpid(),stage=job['stage'],model=job['model'],device=job['device'],
        fatal_error=False,infrastructure_failure=False,no_transcript=0,threads=job['threads'],worker_index=job['worker_index'],completed=0,cached=0,failed=0,audio_seconds=0.,
        allocated=job['allocation'],python=sys.executable,versions={})
    packages={'ingest':['soundfile','numpy'],'disagreement':[],'rover':[], 'dnsmos':['onnxruntime','soundfile','numpy']}.get(job['stage'],['torch','transformers','qwen-asr','funasr','soundfile','deepfilterlib'])
    for package in packages:
        try:report['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    from .cpu_accounting import live_child_seconds
    started=time.perf_counter();cpu_start=time.process_time();child_start=resource.getrusage(resource.RUSAGE_CHILDREN);live_child_start=live_child_seconds();done=threading.Event();samples=[]
    report['model_reused']=False
    monitor_cfg=optimization.get('monitoring',{})
    sample_seconds=monitor_cfg.get('sample_seconds',1)
    report_seconds=monitor_cfg.get('report_seconds',20)
    def monitor():
        while not done.is_set():
            usage=resource.getrusage(resource.RUSAGE_SELF)
            value=dict(time=time.time(),peak_ram_mib=usage.ru_maxrss/1024,cpu_seconds=time.process_time()-cpu_start)
            if job['device'] is not None:
                gpu=gpu_query(job['device'])
                if gpu:value.update(gpu_used_mib=gpu[1],gpu_utilization=gpu[2])
            samples.append(value)
            if len(samples)%max(1,round(report_seconds/sample_seconds))==0:
                timings=dict(phase,read_prepare_wait_seconds=prefetch.seconds if prefetch else 0.)
                progress=dict(report)
                progress.update(event='progress',phases=timings,adapter_metrics=dict(getattr(adapter,'metrics',{})),
                                elapsed=time.perf_counter()-started,resources=value)
                print(json.dumps(progress),flush=True)
            done.wait(sample_seconds)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    try:
        report['phase']='waiting_for_memory'
        check_host(optimization.get('memory',{}),stop_requested=lambda:STOP or Path(job['stop_file']).exists())
        report['phase']='initializing'
        if job['device'] is not None and resident is None:
            # OS advisory locks are automatically released after a crash.
            directory=Path(job['lock_root']);directory.mkdir(parents=True,exist_ok=True)
            report['phase']='waiting_for_gpu'
            while not STOP and not Path(job['stop_file']).exists():
                for slot in range(job['gpu_slots']):
                    candidate=open(directory/f"gpu-{job['device']}-{slot}.lock",'a+')
                    try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:candidate.close();continue
                    lock=candidate;break
                if lock:break
                done.wait(.5)
            if not lock:raise InterruptedError('Stopped while waiting for GPU')
            gpu=gpu_query(job['device'])
            if gpu is None:raise RuntimeError('Unable to verify GPU availability')
            if gpu[0]/1024<job['min_free_vram_gib']:
                raise RuntimeError(f"GPU {job['device']} has {gpu[0]/1024:.2f} GiB free; requires {job['min_free_vram_gib']} GiB headroom")
        if job['stage'] not in ['ingest','disagreement','rover','dnsmos']:
            import torch
            torch.set_num_threads(job['threads'])
            if not _INTEROP_SET:torch.set_num_interop_threads(1);_INTEROP_SET=True
        adapter=None
        failed_this_run=set()
        inference_size=(job['settings'].get('batch_size',1) if job['stage'].startswith('asr') else
                        job['settings'].get('batch_size',job['allocation'].get('inference_batch_size',1)) if job['stage']=='align' else
                        job['allocation'].get('inference_batch_size',1) if job['stage'] in ('vad','dnsmos','enhance') else
                        job['allocation'].get('result_batch_size',1))
        batch_size=inference_size
        save_target=job['allocation'].get('result_batch_size',batch_size)
        checkpoint=job['allocation'].get('checkpoint',{})
        last_flush=time.monotonic();pending_bytes=0
        pending_active=[];pending_values=[]
        adaptive=job['settings'].get('adaptive',False) and (job['stage'].startswith('asr') or job['stage']=='align')
        settings=job['settings'].get('batch_policy') or resolved_policy(job['allocation'],job['model'],ROOT)
        calibration=settings['calibration']
        policy=BatchPolicy(maximum=batch_size,adaptive=adaptive or (job['stage'] in ('vad','dnsmos') and job['settings'].get('adaptive',False)),max_padded_seconds=settings.get('max_padded_seconds',160),
            min_free_gib=settings.get('min_free_gib',4),target_memory_fraction=settings.get('target_memory_fraction',.8),calibration=calibration)
        # The controller groups recording-local tasks with similar durations.
        # Global duration sorting destroys cache locality across recordings.
        ordered_tasks=job['tasks']

        from .stage_pipelines import enabled as stage_pipeline_enabled,owns_audio_prefetch
        if ((job['stage'].startswith('asr') or job['stage']=='align') and
                not job['settings'].get('frontend_workers',0) and not owns_audio_prefetch(job)):
            from . import audio as prepared_audio
            prefetch=AudioPrefetch(prepared_audio.load_chunk,workers=job['allocation'].get('prepare_workers',2),
                lookahead=min(512 if job['model']=='qwen3_asr' else 256,
                              batch_size*job['allocation'].get('prefetch_batches',2)),
                max_bytes=int(job['allocation'].get('prepare_max_gib',1)*2**30),estimate=prepared_audio.estimate_chunk_bytes)
            prefetch.start(ordered_tasks)
        if optimization.get('async_write',False) and job.get('writer_socket'):save_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='durable-write')

        def key_for(task):
            return job['stage'],job['model'],task['item']['id'],task['signature']

        def log_item(task):
            if (report['completed']+report['failed'])%100 and task['item']['id'] not in failed_this_run:return
            print(json.dumps(dict(event='item',item=task['item']['id'],completed=report['completed'],
                                  cached=report['cached'],failed=report['failed'])),flush=True)

        def fail(task,tick,error):
            report['failed']+=1
            failed_this_run.add(task['item']['id'])
            persist(writer.put,*key_for(task),'failed',error=error,seconds=time.perf_counter()-tick)
            print(json.dumps(dict(event='error',item=task['item']['id'],traceback=error)),flush=True)
            log_item(task)

        def prepare(task,value):
            if adapter is not None and hasattr(adapter,'finalize_result'):persist(adapter.finalize_result,task['item'],value)
            item=task['item'];source=item.get('chunk',item)
            if value.get('_item_error'):raise ValueError(value['_item_error'])
            recorded=source.get('source_fingerprint')
            if recorded and fingerprint(recorded['path'])!=recorded:
                raise RuntimeError('Source changed during inference')
            value['_provenance']=dict(signature=task['signature'],model=job['model'],stage=job['stage'],input={k:v for k,v in item.items() if k!='hypotheses'},
                                     checkpoint=job.get('checkpoint'),adapter_version=job['adapter_version'])
            if job['stage'] in ASR_STAGES and hypothesis_error(value):
                raise RuntimeError('ASR hypothesis rejected: '+hypothesis_error(value))
            validate(value)
            return source

        def complete(task,tick,value):
            source=prepare(task,value)
            persist(atomic,task['output'],value)
            persist(writer.put,*key_for(task),'complete',task['output'],seconds=time.perf_counter()-tick)
            report['completed']+=1
            report['no_transcript']+=value.get('transcript_status')=='no_transcript'
            report['audio_seconds']+=float(value.get('duration',source.get('duration',0)))
            log_item(task)

        def complete_batch(active,values):
            valid=[]
            for (task,tick),value in zip(active,values):
                try:valid.append((task,tick,value,prepare(task,value)))
                except Exception as error:
                    if is_infrastructure_error(error):raise
                    fail(task,tick,traceback.format_exc())
            if not valid:return
            token=uuid.uuid4().hex
            path=Path(job['root'])/job['stage']/job['model']/'batch_results'/token[:2]/(token+'.json')
            try:persist(atomic,path,dict(results=[value for _,_,value,_ in valid]))
            except InfrastructureError:raise
            except Exception:
                print(json.dumps(dict(event='batch_write_retry',traceback=traceback.format_exc())),flush=True)
                for task,tick,value,_ in valid:
                    try:complete(task,tick,value)
                    except Exception as error:
                        if is_infrastructure_error(error):raise
                        fail(task,tick,traceback.format_exc())
                return
            persist(writer.put_many,[(*key_for(task),'complete',f'{path}#result={index}',None,time.perf_counter()-tick)
                            for index,(task,tick,_,_) in enumerate(valid)])
            for task,_,value,source in valid:
                report['completed']+=1
                report['no_transcript']+=value.get('transcript_status')=='no_transcript'
                report['audio_seconds']+=float(value.get('duration',source.get('duration',0)))
                log_item(task)

        def _finish_save():
            nonlocal save_future
            if save_future is not None:
                tick=time.perf_counter();save_future.result();phase['persistence_wait_seconds']+=time.perf_counter()-tick
                save_future=None

        def finish_save():
            with save_lock:_finish_save()

        def save_batch(active,values):
            tick=time.perf_counter()
            try:complete_batch(active,values)
            finally:phase['persistence_seconds']+=time.perf_counter()-tick

        def _queue_save(active,values,force=False):
            nonlocal last_flush,pending_bytes
            # Keep at most one durable group in flight and one group being filled.
            pending_active.extend(active);pending_values.extend(values)
            if checkpoint.get('max_mib'):
                pending_bytes+=sum(len(json.dumps(value,ensure_ascii=False).encode()) for value in values)
            due=(bool(pending_active) and (time.monotonic()-last_flush>=checkpoint.get('max_seconds',float('inf')) or pending_bytes>=checkpoint.get('max_mib',float('inf'))*2**20))
            force=force or due
            while len(pending_active)>=save_target or (force and pending_active):
                count=min(save_target,len(pending_active))
                batch_active=pending_active[:count];batch_values=pending_values[:count]
                del pending_active[:count];del pending_values[:count]
                finish_save()
                if save_pool:
                    nonlocal_save(batch_active,batch_values)
                else:save_batch(batch_active,batch_values)
                last_flush=time.monotonic();pending_bytes=0

        def queue_save(active,values,force=False):
            with save_lock:_queue_save(active,values,force)

        def flush_completed():
            while not flush_stop.wait(checkpoint.get('poll_seconds',min(1,checkpoint['max_seconds']/4))):
                try:
                    with save_lock:
                        if pending_active and time.monotonic()-last_flush>=checkpoint['max_seconds']:queue_save([],[],force=True)
                except BaseException as error:
                    flush_errors.append(error);flush_stop.set();return

        def nonlocal_save(active,values):
            nonlocal save_future
            save_future=save_pool.submit(save_batch,active,values)

        def infer(items):
            try:
                return (adapter.process_batch(items) if inference_size>1 and hasattr(adapter,'process_batch')
                        else [adapter.process(item) for item in items])
            except Exception as error:
                if not is_cuda_oom(error) or len(items)==1:raise
            # Leave the except block first so its traceback releases failed tensors.
            policy.oom()
            if 'torch' in sys.modules:sys.modules['torch'].cuda.empty_cache()
            middle=len(items)//2
            return infer(items[:middle])+infer(items[middle:])

        def recover_batch(group):
            """Keep healthy items batched when one input causes a model error."""
            if STOP or Path(job['stop_file']).exists():return
            retry_start=time.perf_counter()
            try:
                values=infer([task['item'] for task,_ in group])
                if len(values)!=len(group):raise RuntimeError('Model returned an incomplete batch')
            except Exception as error:
                if is_infrastructure_error(error):raise
                report['inference_seconds']+=time.perf_counter()-retry_start
                if len(group)==1:
                    fail(group[0][0],group[0][1],traceback.format_exc())
                    return
            else:
                report['inference_seconds']+=time.perf_counter()-retry_start
                queue_save(group,values)
                print(json.dumps(dict(event='batch_recovery',items=len(group))),flush=True)
                return
            # Release the failed inference traceback before retrying smaller groups.
            middle=len(group)//2
            recover_batch(group[:middle]);recover_batch(group[middle:])

        if checkpoint.get('max_seconds') and job.get('writer_socket'):
            flush_thread=threading.Thread(target=flush_completed,daemon=True);flush_thread.start()
        groups=policy.groups(ordered_tasks)
        from .funasr_pipeline import enabled as funasr_pipeline_enabled
        if funasr_pipeline_enabled(job) or stage_pipeline_enabled(job):
            # Keep scheduling outside the adapter/inference version so existing
            # durable task identities remain valid after enabling overlap.
            report['phase']='loading_model'
            load_start=time.perf_counter()
            adapter=(resident.acquire(job) if resident is not None else load_module(job['module'],'stage_adapter').Adapter(job))
            report['model_reused']=bool(resident is not None and resident.reused)
            stop_requested=lambda:STOP or Path(job['stop_file']).exists()
            if funasr_pipeline_enabled(job):
                from .funasr_pipeline import FunASRPipeline
                adapter=FunASRPipeline(adapter,job,optimization['funasr_pipeline'],stop_requested=stop_requested)
            else:
                from .stage_pipelines import wrap
                adapter=wrap(adapter,job,stop_requested)
            report['model_load_seconds']=time.perf_counter()-load_start
            preparation_start=time.perf_counter()
            adapter.start([task['item'] for task in ordered_tasks])
            phase['preparation_start_seconds']=time.perf_counter()-preparation_start
            if job['stage']=='vad' and job['allocation'].get('prepare_order')=='ready':
                base=getattr(adapter,'base',adapter)
                groups=base.preparation.ready_groups(ordered_tasks,batch_size,job['allocation'].get('ready_wait_seconds',0))
            if hasattr(adapter,'groups'):groups=adapter.groups(groups)
        report['phase']='processing'
        for group in groups:
            if flush_errors:raise InfrastructureError('Timed checkpoint flush failed') from flush_errors[0]
            if STOP or Path(job['stop_file']).exists():break
            active=[]
            if prefetch:
                prepared_audio.set_prepared({})
            # Only one pending durable batch per worker. Inference can overlap it.
            if save_future is not None and save_future.done():finish_save()
            cached_results=state.cached_many(job['stage'],job['model'],
                                              [(task['item']['id'],task['signature']) for task in group])
            for task in group:
                cached=cached_results.get((task['item']['id'],task['signature']))
                if cached is not None and not (job['stage'] in ASR_STAGES and hypothesis_error(cached)):
                    if prefetch:prefetch.discard(task['item'])
                    report['cached']+=1;continue
                tick=time.perf_counter()
                try:
                    if task['item'].get('_planning_error'):raise ValueError(task['item']['_planning_error'])
                    source=task['item'].get('chunk',task['item'])
                    recorded=source.get('source_fingerprint')
                    if recorded and fingerprint(recorded['path'])!=recorded:
                        raise RuntimeError('Input changed since ingestion; rerun ingest and upstream stages')
                    if prefetch:
                        value=prefetch.take(task['item'])
                        prepared_audio.add_prepared(task['item'].get('chunk',task['item']),value)
                    active.append((task,tick))
                except Exception as error:
                    if is_infrastructure_error(error):raise
                    fail(task,tick,traceback.format_exc())
            if not active:continue
            # Text-only disagreement is fast; its durable completed checkpoint is
            # sufficient for resume. Avoid a second full database rewrite just
            # to mark these same rows running immediately before completion.
            if job['stage']!='disagreement' and optimization.get('record_running_state',True):
                persist(writer.put_many,[(*key_for(task),'running',None,None,0) for task,_ in active])
            if adapter is None:
                report['phase']='loading_model'
                load_start=time.perf_counter()
                try:
                    adapter=(resident.acquire(job) if resident is not None else load_module(job['module'],'stage_adapter').Adapter(job))
                    report['model_reused']=bool(resident is not None and resident.reused)
                except Exception as error:
                    if is_infrastructure_error(error):raise
                    if job['stage'] not in ASR_STAGES:
                        for task,tick in active:fail(task,tick,traceback.format_exc())
                        raise RuntimeError('Model initialization failed; remaining tasks were not scheduled') from error
                    raise RuntimeError('Model unavailable: '+traceback.format_exc()) from error
                report['model_load_seconds']=time.perf_counter()-load_start
                if hasattr(adapter,'set_stop_requested'):
                    adapter.set_stop_requested(lambda: STOP or Path(job['stop_file']).exists())
                if hasattr(adapter,'start'):
                    preparation_start=time.perf_counter()
                    adapter.start([task['item'] for task in ordered_tasks])
                    phase['preparation_start_seconds']=time.perf_counter()-preparation_start
                report['phase']='processing'
            inference_start=time.perf_counter()
            if adaptive:
                import torch
                torch.cuda.reset_peak_memory_stats()
            retry=False
            try:
                values=infer([task['item'] for task,_ in active])
                if len(values)!=len(active):raise RuntimeError('Model returned an incomplete batch')
            except Exception as error:
                if is_infrastructure_error(error):raise
                if STOP or Path(job['stop_file']).exists():raise InterruptedError('Stopped during inference') from error
                report['inference_seconds']=report.get('inference_seconds',0)+time.perf_counter()-inference_start
                if is_cuda_oom(error):policy.oom()
                if len(active)==1:
                    fail(active[0][0],active[0][1],traceback.format_exc())
                    continue
                print(json.dumps(dict(event='batch_retry',items=[task['item']['id'] for task,_ in active],
                                      traceback=traceback.format_exc())),flush=True)
                retry=True
            else:
                report['inference_seconds']=report.get('inference_seconds',0)+time.perf_counter()-inference_start
                from .batching import duration
                memory=observe_cuda_memory(policy,torch.cuda,external=(settings.get('memory_feedback')=='driver_free' or (settings.get('memory_feedback')=='auto' and job['settings'].get('engine')=='vllm'))) if adaptive else dict(next_batch_limit=policy.limit)
                print(json.dumps(dict(event='batch',items=len(active),
                    padded_audio_seconds=len(active)*max(duration(task) for task,_ in active),
                    adapter_seconds=time.perf_counter()-inference_start,
                    **memory)),flush=True)
                queue_save(active,values)
            if retry:
                # Outside the exception handler: the failed forward's tensors
                # must be released before any smaller inference is attempted.
                finish_save()
                if 'torch' in sys.modules:
                    import torch
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                middle=len(active)//2
                recover_batch(active[:middle]);recover_batch(active[middle:])
        queue_save([],[],force=True)
        finish_save()
        report['status']='stopped' if STOP or Path(job['stop_file']).exists() else 'completed_with_errors' if report['failed'] or report['no_transcript'] else 'passed'
    except BaseException as error:
        message=traceback.format_exc()
        if STOP or Path(job['stop_file']).exists():
            report['status']='stopped'
        elif is_infrastructure_error(error):
            report.update(status='failed',fatal_error=True,infrastructure_failure=True,error=message)
        elif report.get('phase') in ('initializing','loading_model') and optimization.get('failure_policy',{}).get('model_initialization')=='stop':
            report.update(status='failed',fatal_error=True,model_unavailable=True,error=message)
        elif job['stage'] in ASR_STAGES and not is_infrastructure_error(error):
            # A failed ASR process/model is optional to the other recognizers.
            # Keep completed results, mark every remaining assigned task explicitly.
            try:
                remaining=[]
                for task in job['tasks']:
                    cached=state.cached(*key_for(task))
                    if cached is not None and not hypothesis_error(cached):continue
                    if task['item']['id'] in failed_this_run:continue
                    failed_this_run.add(task['item']['id'])
                    remaining.append((*key_for(task),'failed',None,message,0))
                    if len(remaining)>=512:
                        persist(writer.put_many,remaining);report['failed']+=len(remaining);remaining=[]
                if remaining:persist(writer.put_many,remaining);report['failed']+=len(remaining)
                report.update(status='completed_with_errors',model_error=message,model_unavailable=True)
            except BaseException:
                report.update(status='failed',fatal_error=True,infrastructure_failure=True,error=traceback.format_exc())
        else:
            report.update(status='failed',fatal_error=True,infrastructure_failure=is_infrastructure_error(error),error=message)
        print(message,flush=True)
    finally:
        flush_stop.set()
        if flush_thread:flush_thread.join()
        try:
            if flush_errors:raise InfrastructureError('Timed checkpoint flush failed') from flush_errors[0]
            if pending_active:queue_save([],[],force=True)
            if save_future is not None:save_future.result()
        except (InterruptedError,WriterStopped):
            if STOP or Path(job['stop_file']).exists():report['status']='stopped'
            else:report.update(status='failed',fatal_error=True,infrastructure_failure=True,persistence_error=traceback.format_exc())
        except Exception:
            report.update(status='failed',fatal_error=True,infrastructure_failure=True,persistence_error=traceback.format_exc())
        if save_pool:save_pool.shutdown(wait=True)
        if prefetch:
            prefetch.close();phase['read_prepare_wait_seconds']=prefetch.seconds
        if adapter is not None and hasattr(adapter,'close'):
            adapter.close()
        if adapter is not None and hasattr(adapter,'metrics'):
            report['adapter_metrics']=dict(adapter.metrics)
        from .transfers import metrics as transfer_metrics
        report['transfer_metrics']=transfer_metrics(reset=True)
        from .ram_audio import metrics as decoded_metrics
        report['decoded_audio_cache']=decoded_metrics()
        report['phase']='finished';report['phases']=phase
        state.db.close()
        done.set();monitor_thread.join(timeout=6)
        report['runtime_seconds']=time.perf_counter()-started
        report['cpu_seconds']=time.process_time()-cpu_start
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        report['cpu_children_seconds']=max(0.,children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime+live_child_seconds()-live_child_start)
        report['cpu_total_seconds']=report['cpu_seconds']+report['cpu_children_seconds']
        report['cpu_percent_of_one_core']=100*report['cpu_seconds']/max(.001,report['runtime_seconds'])
        report['peak_ram_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        report['audio_seconds_per_wall_second']=report['audio_seconds']/max(.001,report['runtime_seconds'])
        report['gpu_samples']=len([s for s in samples if 'gpu_used_mib' in s])
        if report['gpu_samples']:
            report['peak_device_used_mib']=max(s.get('gpu_used_mib',0) for s in samples)
            report['mean_device_utilization']=sum(s.get('gpu_utilization',0) for s in samples)/report['gpu_samples']
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_available():report['peak_process_vram_mib']=torch.cuda.max_memory_allocated()/2**20
        try:persist(atomic,job['report'],report)
        finally:
            if lock:lock.close()
    # Break recursive worker closures' reference to a completed adapter/session.
    adapter=None
    return 0 if report['status'] in SUCCESS else 130 if report['status']=='stopped' else 1

def main():return run_job(read(sys.argv[1]))

if __name__=='__main__':sys.exit(main())
