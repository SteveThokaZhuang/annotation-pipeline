"""One exclusive GPU owner per IPC endpoint; compatible jobs reuse the loaded model.

Only local job-file references cross the socket. Durable results still use the
normal state writer. Incompatible environments restart the owner, and model
changes evict the previous model before loading another one.
"""
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import threading
from functools import lru_cache
from .common import atomic,digest,load_module,read
from .writer import send,receive


@lru_cache(maxsize=1)
def runtime_version():
    value=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        value.update(path.name.encode());value.update(path.read_bytes())
    return value.hexdigest()


class JobStream:
    """Logging handlers may retain this stream across multiple job log files."""
    def __init__(self,target):self.target=target
    def write(self,value):return self.target.write(value)
    def flush(self):return self.target.flush()
    def __getattr__(self,name):return getattr(self.target,name)


class Session:
    def __init__(self,base):object.__setattr__(self,'base',base)
    def __getattr__(self,name):return getattr(object.__getattribute__(self,'base'),name)
    def __setattr__(self,name,value):setattr(self.base,name,value)
    def __copy__(self):
        # Pipeline postprocessing needs an independent shallow adapter view.
        return copy.copy(object.__getattribute__(self,'base'))
    def close(self):
        closer=getattr(self.base,'close_job',None)
        if closer:closer()
        elif hasattr(self.base,'close') and not hasattr(self.base,'model'):self.base.close()


class PreparedSession(Session):
    def __init__(self,base,preparation,job):
        super().__init__(base)
        object.__setattr__(self,'preparation',preparation)
        object.__setattr__(self,'preparation_job',job)
    def start(self,items):self.preparation.start(items,self.preparation_job)
    def close(self):self.preparation.finish_job()


class Models:
    def __init__(self):self.key=None;self.base=None;self.reused=False;self.preparation=None
    def prepare(self,job):
        settings={k:v for k,v in job['settings'].items() if k not in ('language','batch_policy')}
        key=digest([job['module'],job.get('checkpoint_version'),job.get('adapter_version'),settings,job['threads'],job['allocation'].get('optimization',{}).get('feature_transport'),job['allocation'].get('prepare_workers')])
        if key!=self.key:self.close();self.key=key
    def acquire(self,job):
        self.reused=self.base is not None
        if self.base is None:
            if job.get('device') is not None:
                from .worker import gpu_query
                gpu=gpu_query(job['device'])
                if gpu is None:raise RuntimeError('Unable to verify GPU availability')
                if gpu[0]/1024<job['min_free_vram_gib']:
                    raise RuntimeError(f"GPU {job['device']} has insufficient free VRAM for this model")
            self.base=load_module(job['module'],'resident_adapter').Adapter(job)
        else:
            base=self.base
            if hasattr(base,'cfg'):base.cfg=job['settings']
            if hasattr(base,'job'):base.job=job
            if hasattr(base,'allocation'):base.allocation=job['allocation']
            if hasattr(base,'stop_file'):base.stop_file=job['stop_file']
            for k,v in getattr(base,'metrics',{}).items():
                if isinstance(v,(int,float)) and not k.endswith(('_limit','_processes')):base.metrics[k]=0
        dispatch=job['allocation'].get('optimization',{}).get('task_dispatch',{})
        if job.get('stage')=='dnsmos' and dispatch.get('prefetch_bundles',0) and getattr(self.base,'prefetch_enabled',False):
            from .continuous_preparation import RecordingPreparation
            if self.preparation is None:self.preparation=RecordingPreparation(self.base)
            return PreparedSession(self.base,self.preparation,job)
        if self.preparation is not None:
            self.preparation.close();self.preparation=None
        return Session(self.base)
    def reset_dataset(self):
        """DNSMOS-only: drain dataset preparation while retaining the ONNX model."""
        if self.preparation is not None:
            self.preparation.close();self.preparation=None
        if self.base is not None:
            closer=getattr(self.base,'close_job',None)
            if closer:closer()
            self.base.items=[];self.base.positions={};self.base.next_index=0
        import gc
        gc.collect()

    def close(self):
        if self.preparation is not None:
            self.preparation.close();self.preparation=None
        if self.base is not None:
            try:
                if hasattr(self.base,'close'):self.base.close()
            finally:
                self.base=None
                import gc
                gc.collect()
                if 'torch' in sys.modules:
                    import torch
                    if torch.cuda.is_available():torch.cuda.empty_cache()
        self.key=None


def endpoint(job):
    root=Path(job['allocation']['optimization']['ipc_root']);root.mkdir(parents=True,exist_ok=True)
    domain=digest(str(Path(job['lock_root']).absolute()))[:8]
    return root/f'g{job["device"]}-{domain}.sock'


def serve(path,job_path):
    first=read(job_path);settings=first['allocation']['optimization']['persistent_workers']
    lock_dir=Path(first['lock_root']);lock_dir.mkdir(parents=True,exist_ok=True)
    models=Models()
    original_out,original_err=sys.stdout,sys.stderr
    output,error_output=JobStream(original_out),JobStream(original_err)
    with (lock_dir/f'g{first["device"]}-resident.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        path=Path(path);path.unlink(missing_ok=True)
        with (lock_dir/f'gpu-{first["device"]}-0.lock').open('a+') as gpu_lock:
            # The same lock protocol as ordinary workers, including v2.
            deadline=time.monotonic()+settings['startup_timeout_seconds']
            while True:
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if Path(first['stop_file']).exists() or time.monotonic()>deadline:return
                    time.sleep(settings['poll_seconds'])
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
                server.bind(str(path));os.chmod(path,0o600);server.listen(16)
                server.settimeout(settings['idle_seconds'])
                atomic(path.with_suffix('.json'),dict(pid=os.getpid(),proc_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],python=sys.executable,device=first['device']))
                try:
                    while True:
                        try:stream,_=server.accept()
                        except socket.timeout:break
                        with stream:
                            request=receive(stream)
                            if request.get('shutdown'):send(stream,{'ok':True});break
                            # A GPU owner can outlive its dataset/controller.
                            # Helpers inherit this mapping at initialization. Drain
                            # DNSMOS preparation on a dataset switch, keeping its
                            # compatible ONNX session. Other models still evict.
                            from .working_cache import ENV
                            job=read(request['job'])
                            working_tree=request.get('working_tree')
                            if working_tree!=os.environ.get(ENV):
                                if job.get('model')=='dnsmos':models.reset_dataset()
                                else:models.close()
                                from .common import _read_batch
                                _read_batch.cache_clear()
                                if working_tree is None:os.environ.pop(ENV,None)
                                else:os.environ[ENV]=working_tree
                            if request.get('runtime_version')!=runtime_version() or os.path.abspath(job['python'])!=os.path.abspath(sys.executable) or job['threads']!=first['threads']:
                                send(stream,{'restart':True});break
                            models.prepare(job)
                            finished=threading.Event();send_lock=threading.Lock()
                            def heartbeat():
                                while not finished.wait(settings['poll_seconds']):
                                    try:
                                        with send_lock:send(stream,{'waiting':True})
                                    except OSError:return
                            pulse=threading.Thread(target=heartbeat,daemon=True);pulse.start()
                            try:
                                from .worker import run_job
                                with open(request['log'],'a',buffering=1) as log:
                                    output.target=error_output.target=log
                                    try:
                                        with contextlib.redirect_stdout(output),contextlib.redirect_stderr(error_output):
                                            code=run_job(job,resident=models)
                                    finally:output.target=original_out;error_output.target=original_err
                                result=read(job['report'])
                                if result.get('model_unavailable') or result.get('fatal_error') or code==130:models.close()
                                response={'ok':True,'returncode':code}
                            except BaseException:
                                models.close();response={'ok':False,'error':traceback.format_exc()}
                            finally:finished.set();pulse.join(timeout=2)
                            with send_lock:send(stream,response)
                finally:
                    models.close();path.unlink(missing_ok=True);path.with_suffix('.json').unlink(missing_ok=True)


def request(job,path,log):
    settings=job['allocation']['optimization']['persistent_workers'];address=endpoint(job)
    deadline=time.monotonic()+settings['startup_timeout_seconds'];stopped_at=None
    with (address.with_suffix('.start.lock')).open('a+') as admission:
        while True:
            if Path(job['stop_file']).exists():return 130
            if time.monotonic()>deadline:raise TimeoutError('Resident worker unavailable; inspect '+str(address.with_suffix('.log')))
            fcntl.flock(admission,fcntl.LOCK_EX)
            stream=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            try:
                stream.connect(str(address))
            except OSError:
                stream.close()
                starting=address.with_suffix('.starting.json');alive=False
                if starting.exists():
                    try:
                        pid=read(starting)['pid'];alive=Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
                    except (OSError,ValueError,KeyError):pass
                if not alive:
                    with open(address.with_suffix('.log'),'a') as daemon_log:
                        proc=subprocess.Popen([job['python'],'-m','audio_pipeline.resident','serve',str(address),str(path)],stdin=subprocess.DEVNULL,stdout=daemon_log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic(starting,{'pid':proc.pid})
                fcntl.flock(admission,fcntl.LOCK_UN)
                time.sleep(settings['poll_seconds']);continue
            fcntl.flock(admission,fcntl.LOCK_UN)
            try:
                from .working_cache import ENV
                send(stream,{'job':str(path),'log':str(log),'runtime_version':runtime_version(),'working_tree':os.environ.get(ENV)})
                while True:
                    result=receive(stream)
                    if not result.get('waiting'):break
                    if Path(job['stop_file']).exists():
                        if stopped_at is None:stopped_at=time.monotonic()
                        if time.monotonic()-stopped_at>settings['shutdown_timeout_seconds']:
                            owner=read(address.with_suffix('.json'));pid=owner['pid']
                            if Path(f'/proc/{pid}/stat').read_text().split()[21]==owner['proc_start']:os.killpg(pid,signal.SIGKILL)
                            return 130
            except (OSError,EOFError):
                # A job may already have committed results. Let normal resume
                # decide what remains; never report transport failure as success.
                raise RuntimeError('Resident worker disconnected; durable checkpoints retained')
            finally:stream.close()
            if result.get('restart'):
                while address.exists() and time.monotonic()<deadline:time.sleep(settings['poll_seconds'])
                continue
            if not result.get('ok'):raise RuntimeError(result.get('error','Resident worker failed'))
            return result['returncode']


def aggregate(reports,job):
    if not reports:
        return dict(stage=job['stage'],model=job['model'],status='passed',fatal_error=False,infrastructure_failure=False,completed=0,cached=0,failed=0,worker_index=job['worker_index'],device=job['device'],runtime_seconds=0.)
    result=dict(reports[-1]);result['bundle_reports']=[r['report_path'] for r in reports]
    for key in ['completed','cached','failed','no_transcript','audio_seconds','runtime_seconds','model_load_seconds','inference_seconds','cpu_total_seconds','cpu_seconds','cpu_children_seconds']:
        result[key]=sum(r.get(key,0) for r in reports)
    for key in ['peak_ram_mib','peak_device_used_mib','peak_process_vram_mib']:
        result[key]=max(r.get(key,0) for r in reports)
    result['model_reuse_jobs']=sum(bool(r.get('model_reused')) for r in reports)
    result['fatal_error']=any(r.get('fatal_error') for r in reports)
    result['infrastructure_failure']=any(r.get('infrastructure_failure') for r in reports)
    result['status']='failed' if result['fatal_error'] else 'stopped' if any(r['status']=='stopped' for r in reports) else 'completed_with_errors' if result['failed'] else 'passed'
    for section in ['phases','adapter_metrics','transfer_metrics']:
        totals={}
        for r in reports:
            for k,v in r.get(section,{}).items():
                if isinstance(v,(int,float)):
                    totals[k]=max(totals.get(k,0),v) if ('peak' in k or '_max_' in k or k.startswith('largest_') or k.endswith(('_limit','_processes'))) else totals.get(k,0)+v
        result[section]=totals
    result['gpu_samples']=sum(r.get('gpu_samples',0) for r in reports)
    if result['gpu_samples']:result['mean_device_utilization']=sum(r.get('mean_device_utilization',0)*r.get('gpu_samples',0) for r in reports)/result['gpu_samples']
    return result


def client(path):
    job=read(path);log=Path(job['report']).with_suffix('.log');reports=[]
    # Controller signals terminate the client only after its normal stop grace.
    # The server observes this same durable stop file at batch boundaries.
    signal.signal(signal.SIGTERM,lambda *_:Path(job['stop_file']).touch())
    signal.signal(signal.SIGINT,lambda *_:Path(job['stop_file']).touch())
    if not job.get('task_queue'):return request(job,path,log)
    from .task_queue import claim,finish
    from .task_manifest import write_tasks
    from collections import deque
    dispatch=job['allocation'].get('optimization',{}).get('task_dispatch',{})
    ahead=dispatch.get('prefetch_bundles',0) if job['stage']=='dnsmos' else 0
    if ahead not in (0,1):raise ValueError('prefetch_bundles must be 0 or 1')
    folder=Path(job['report']).parent/'bundles';folder.mkdir(exist_ok=True)
    pending=deque();exhausted=False
    while not Path(job['stop_file']).exists():
        while len(pending)<1+ahead and not exhausted:
            value=claim(job['task_queue'],os.getpid())
            if value is None:exhausted=True;break
            identity,tasks=value;task_path=folder/f'{job["stage"]}-{identity}.jsonl'
            write_tasks(task_path,tasks);pending.append((identity,task_path))
        if not pending:break
        identity,task_path=pending.popleft();sub=dict(job);sub.pop('task_queue',None)
        sub['tasks_file']=str(task_path);sub['report']=str(folder/f'{job["stage"]}-{identity}-report.json')
        if pending:sub['prefetch_tasks_file']=str(pending[0][1])
        subpath=folder/f'{job["stage"]}-{identity}-job.json';atomic(subpath,sub)
        code=request(sub,subpath,log)
        if Path(sub['report']).exists():reports.append(dict(read(sub['report']),report_path=sub['report']))
        finish(job['task_queue'],identity,'complete' if code==0 else 'failed')
        if code!=0:break
    result=aggregate(reports,job)
    if Path(job['stop_file']).exists():result['status']='stopped'
    atomic(job['report'],result)
    return 0 if result['status'] in ('passed','completed_with_errors') else 130 if result['status']=='stopped' else 1


def shutdown_all(root):
    for path in Path(root).glob('g*.sock'):
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
                stream.settimeout(5);stream.connect(str(path));send(stream,{'shutdown':True});receive(stream)
        except (OSError,EOFError):pass


def main():
    if sys.argv[1]=='serve':serve(sys.argv[2],sys.argv[3]);return 0
    if sys.argv[1]=='client':return client(sys.argv[2])
    raise ValueError('Expected serve or client')

if __name__=='__main__':sys.exit(main())
