"""Bounded CPU VAD preparation, with disposable arrays on local scratch storage."""
from concurrent.futures import ProcessPoolExecutor,TimeoutError
from concurrent.futures.process import BrokenProcessPool
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import uuid

from .failure_policy import InfrastructureError,is_infrastructure_error

_EXTRACTOR=None


def initialize(cmvn,environment):
    # Spawned helpers never initialize CUDA or inherit its runtime state.
    os.environ.update(environment)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name]='1'
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    import torch
    from fireredvad.core.audio_feat import AudioFeat
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    global _EXTRACTOR
    _EXTRACTOR=AudioFeat(cmvn)


def prepare_recording(item,directory):
    import numpy as np
    from .audio import read_audio
    from .common import fingerprint
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    metrics=dict(read_seconds=0.,feature_seconds=0.,spool_seconds=0.)
    def check():
        expected=item.get('source_fingerprint')
        if expected and fingerprint(expected['path'])!=expected:
            raise ValueError('Source changed during VAD preparation; rerun ingest')
    try:
        check();channels=[]
        for channel in range(item['channels']):
            tick=time.perf_counter()
            audio,rate=read_audio(item['path'],channel=channel,rate=16000)
            pcm=np.rint(np.clip(audio,-1,32767/32768)*32768).astype(np.int16)
            if len(pcm)<3200:pcm=np.pad(pcm,(0,3200-len(pcm)))
            metrics['read_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter();features,detector_duration=_EXTRACTOR.extract(pcm)
            metrics['feature_seconds']+=time.perf_counter()-tick
            if features.ndim!=2 or features.shape[1]!=80 or not np.isfinite(features.numpy()).all():
                raise ValueError('Invalid VAD features')
            tick=time.perf_counter()
            audio_path=path/f'{channel}-audio.npy';feature_path=path/f'{channel}-features.npy'
            np.save(audio_path,audio,allow_pickle=False)
            np.save(feature_path,features.numpy(),allow_pickle=False)
            metrics['spool_seconds']+=time.perf_counter()-tick
            channels.append(dict(audio=str(audio_path),features=str(feature_path),detector_duration=detector_duration))
        check()
        return dict(channels=channels,metrics=metrics)
    except BaseException:
        shutil.rmtree(path,ignore_errors=True)
        raise


def prepared_bytes(item):
    samples=math.ceil(float(item.get('duration',0))*16000)
    frames=max(1,(max(samples,3200)-400)//160+1)
    # Float32 waveform + normalized float32 features, with headers/margin.
    return math.ceil((samples*4+frames*80*4)*max(1,item.get('channels',1))*1.05)+65536


class PreparationPool:
    def __init__(self,workers,lookahead,max_bytes,scratch,cmvn,stop_file=None):
        if min(workers,lookahead,max_bytes)<1:raise ValueError('VAD preparation limits must be positive')
        Path(scratch).mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='vad-prepare-',dir=scratch)
        self.root=Path(self.temporary.name);self.lookahead=lookahead;self.max_bytes=max_bytes
        self.stop_file=Path(stop_file) if stop_file else None
        self.tasks=[];self.cursor=0;self.entries={};self.reserved=0;self.closed=False
        self.metrics=dict(peak_reserved_bytes=0,peak_pending_recordings=0)
        environment={key:os.environ[key] for key in ['AUDIO_PIPELINE_CACHE','AUDIO_PIPELINE_IO_LOCKS','AUDIO_PIPELINE_IO_WORKERS'] if key in os.environ}
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(str(cmvn),environment))

    def start(self,items):
        if self.closed or self.entries:raise RuntimeError('VAD preparation still has pending recordings')
        self.tasks=items;self.cursor=0;self._fill()

    def _check_stop(self):
        if self.stop_file and self.stop_file.exists():
            self.close()
            raise InterruptedError('Stopped during VAD preparation')

    def _fill(self):
        if self.closed:return
        self._check_stop()
        while self.cursor<len(self.tasks) and len(self.entries)<self.lookahead:
            item=self.tasks[self.cursor]
            if item.get('_planning_error'):
                self.cursor+=1;continue
            size=prepared_bytes(item)
            # A recording larger than the budget runs alone, rather than stalling.
            if self.entries and self.reserved+size>self.max_bytes:break
            directory=self.root/uuid.uuid4().hex
            try:future=self.pool.submit(prepare_recording,item,str(directory))
            except BrokenProcessPool as error:
                raise InfrastructureError('A VAD CPU preparation process exited unexpectedly') from error
            self.entries[item['id']]=dict(future=future,directory=directory,size=size)
            self.reserved+=size;self.cursor+=1
            self.metrics['peak_reserved_bytes']=max(self.metrics['peak_reserved_bytes'],self.reserved)
            self.metrics['peak_pending_recordings']=max(self.metrics['peak_pending_recordings'],len(self.entries))

    def take(self,item):
        self._check_stop()
        entry=self.entries.get(item['id'])
        if entry is None:raise RuntimeError('VAD preparation order is inconsistent')
        while True:
            try:return entry['future'].result(timeout=.2)
            except TimeoutError:
                if entry['future'].done():raise
                self._check_stop()
            except BrokenProcessPool as error:
                raise InfrastructureError('A VAD CPU preparation process exited unexpectedly') from error

    def release(self,item):
        entry=self.entries.pop(item['id'],None)
        if entry:
            shutil.rmtree(entry['directory'],ignore_errors=True)
            self.reserved-=entry['size']
        self._fill()

    def discard(self,item):
        entry=self.entries.get(item['id'])
        if entry:
            if not entry['future'].cancel():
                try:self.take(item)
                except Exception as error:
                    if isinstance(error,InterruptedError) or is_infrastructure_error(error):raise
            self.release(item)

    def reset(self):
        self.tasks=[];self.cursor=0
        for entry in self.entries.values():entry['future'].cancel()
        for entry in self.entries.values():
            try:entry['future'].result()
            except Exception:pass
            shutil.rmtree(entry['directory'],ignore_errors=True)
        self.entries.clear();self.reserved=0
        self.metrics=dict(peak_reserved_bytes=0,peak_pending_recordings=0)

    def ready_groups(self,tasks,maximum,wait_seconds=0):
        # Completion order affects scheduling only. Item IDs and sample ranges
        # remain unchanged, and every item is yielded exactly once.
        from concurrent.futures import wait,FIRST_COMPLETED
        remaining={t['item']['id']:t for t in tasks}
        while remaining:
            self._check_stop()
            available=[key for key,entry in self.entries.items() if key in remaining and entry['future'].done()]
            if not available:
                futures=[e['future'] for k,e in self.entries.items() if k in remaining]
                if not futures:raise RuntimeError('VAD readiness queue lost pending work')
                wait(futures,timeout=.2,return_when=FIRST_COMPLETED);continue
            if wait_seconds:
                wait([e['future'] for k,e in self.entries.items() if k in remaining],timeout=wait_seconds)
                available=[key for key,entry in self.entries.items() if key in remaining and entry['future'].done()]
            selected=available[:maximum]
            yield [remaining.pop(key) for key in selected]

    def close(self):
        if self.closed:return
        self.closed=True
        for entry in self.entries.values():entry['future'].cancel()
        # Python 3.10 has no public terminate_workers(). Only terminate this
        # executor's owned CPU helpers, then let its manager clean up queues.
        processes=list((getattr(self.pool,'_processes',None) or {}).values())
        for process in processes:
            if process.is_alive():process.terminate()
        self.pool.shutdown(wait=True,cancel_futures=True)
        for process in processes:process.join(timeout=5)
        # Forced cancellation may interrupt an explicitly enabled source copy.
        import json
        cache=json.loads(os.environ.get('AUDIO_PIPELINE_CACHE','{}'))
        if cache.get('enabled') and cache.get('root'):
            root=Path(cache['root'])
            if (root/'.audio-pipeline-cache-v1').is_file():
                for process in processes:
                    for path in root.glob(f'*.partial-{process.pid}'):path.unlink(missing_ok=True)
        self.entries.clear();self.reserved=0;self.temporary.cleanup()
