"""CPU-only ASR feature workers. No model or CUDA context enters these helpers."""
from concurrent.futures import ProcessPoolExecutor, wait
from functools import partial
import json
import tempfile
import multiprocessing
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from .prefetch import AudioPrefetch
from .audio import estimate_chunk_bytes

_KIND=None
_EXTRACTOR=None
_TOKENIZER=None
_STUB=None
_SETTINGS=None


def initialize(kind,checkpoint,settings,frontend,tokenizer,low_frame_rate,environment):
    os.environ.update(environment);os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)
    global _KIND,_EXTRACTOR,_TOKENIZER,_STUB,_SETTINGS
    _KIND=kind;_SETTINGS=settings;_TOKENIZER=tokenizer
    if kind=='funasr':
        _EXTRACTOR=frontend
        _STUB=SimpleNamespace(use_low_frame_rate=low_frame_rate)
    elif kind=='firered':
        import sys
        from .common import ROOT
        sys.path.insert(0,str(ROOT/'models/firered-asr2/scripts/FireRedASR2S'))
        from fireredasr2s.fireredasr2.data.asr_feat import ASRFeatExtractor
        _EXTRACTOR=ASRFeatExtractor(str(Path(checkpoint)/'cmvn.ark'))
    else:raise ValueError('Unknown CPU frontend: '+kind)


def _prepare(item,language=None):
    import numpy as np
    import torch
    from .audio import load_chunk
    audio,rate=load_chunk(item)
    if _KIND=='firered':
        pcm=np.rint(audio*32768).astype(np.int16)
        features,lengths,durations,_,ids=_EXTRACTOR([[rate,pcm]],[item['id']])
        if features is None or ids!=[item['id']]:raise ValueError('No FireRed features')
        return dict(features=features[0,:int(lengths[0])].numpy(),duration=durations[0])
    from funasr.models.fun_asr_nano.model import FunASRNano
    language=language or _SETTINGS.get('language','auto')
    language={'en':'英文','zh':'中文','English':'英文','Chinese':'中文','auto':None}.get(language,language)
    prompt=FunASRNano.get_prompt(_STUB,[],language,True)
    chat=FunASRNano.generate_chatml(_STUB,prompt,torch.from_numpy(audio))
    contents=FunASRNano.data_template(_STUB,chat)
    if _SETTINGS.get('frontend_seed') is not None:
        import hashlib
        seed=int.from_bytes(hashlib.sha256(item['id'].encode()).digest()[:4],'little')
        torch.manual_seed((seed+int(_SETTINGS['frontend_seed']))%2**32)
    values=FunASRNano.data_load_speech(_STUB,contents,_TOKENIZER,_EXTRACTOR,meta_data={},device='cpu')
    return {key:value.numpy() if isinstance(value,torch.Tensor) else value for key,value in values.items()}


def prepare(item,language=None):
    value=_prepare(item,language)
    root=os.environ.get('AUDIO_PIPELINE_FEATURE_ROOT')
    if root:
        from .feature_transport import pack
        return pack(value,root)
    return value


class FeaturePrefetch(AudioPrefetch):
    def __init__(self,kind,job,frontend=None,tokenizer=None,low_frame_rate=False):
        settings=job['settings'];allocation=job.get('allocation',{})
        workers=settings.get('frontend_workers',allocation.get('prepare_workers',2))
        lookahead=settings.get('frontend_lookahead',max(1,settings.get('batch_size',32)*allocation.get('prefetch_batches',2)))
        budget=int(settings.get('frontend_max_gib',allocation.get('prepare_max_gib',1))*2**30)
        # Feature/prompt arrays, audio decode and IPC copies get extra headroom.
        super().__init__(partial(prepare,language=settings.get('language')),workers=workers,lookahead=lookahead,max_bytes=budget,
                         estimate=lambda item:estimate_chunk_bytes(item)*2,stop_file=job.get('stop_file'))
        self.pool.shutdown()
        self.transport=job.get('allocation',{}).get('optimization',{}).get('feature_transport',{})
        self.spool=None
        environment={k:v for k,v in os.environ.items() if k.startswith('AUDIO_PIPELINE_')}
        if self.transport.get('mode','pickle')=='memmap':
            root=self.transport['root'];Path(root).mkdir(parents=True,exist_ok=True)
            self.spool=tempfile.TemporaryDirectory(prefix='features-',dir=root)
            environment['AUDIO_PIPELINE_FEATURE_ROOT']=self.spool.name
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(kind,job['checkpoint'],settings,frontend,tokenizer,low_frame_rate,environment))

    def _fill(self):
        from concurrent.futures.process import BrokenProcessPool
        from .failure_policy import InfrastructureError
        try:return super()._fill()
        except BrokenProcessPool as error:raise InfrastructureError('An ASR CPU feature worker exited unexpectedly') from error

    def start(self,tasks):
        self.positions={task['item']['id']:i for i,task in enumerate(tasks)}
        self.next_index=0
        super().start(tasks)

    def take(self,item):
        index=self.positions[item['id']]
        for skipped in self.tasks[self.next_index:index]:self.discard(skipped['item'])
        self.next_index=index+1
        # Even an oversize input must run in a helper, whose initialized CPU
        # frontend lives outside the GPU process. Wait for earlier work first.
        if item['id'] not in self.futures:
            self.futures[item['id']]=self.pool.submit(self.loader,item.get('chunk',item))
            self.handled.add(item['id'])
        from concurrent.futures.process import BrokenProcessPool
        from .failure_policy import InfrastructureError
        try:
            value=super().take(item)
            if self.spool:
                from .feature_transport import unpack
                value=unpack(value)
            return value
        except BrokenProcessPool as error:raise InfrastructureError('An ASR CPU feature worker exited unexpectedly') from error

    def reset(self):
        # Retain initialized CPU processes; drain only this job's bounded work.
        pending=list(self.futures.values())+[f for f,_ in self.retired]
        for future in pending:future.cancel()
        if pending:wait(pending)
        self.futures.clear();self.costs.clear();self.retired.clear();self.handled.clear()
        self.tasks=[];self.cursor=0;self.seconds=0.;self.peak_reserved_bytes=0
        if self.spool:
            for p in Path(self.spool.name).glob('*.npy'):p.unlink(missing_ok=True)

    def reset_job(self,job):
        self.stop_file=Path(job['stop_file']) if job.get('stop_file') else None
        self.loader=partial(prepare,language=job['settings'].get('language'))

    def close(self):
        if self.closed:return
        self.closed=True
        processes=list((getattr(self.pool,'_processes',None) or {}).values())
        for future in self.futures.values():future.cancel()
        for process in processes:
            if process.is_alive():process.terminate()
        self.pool.shutdown(wait=True,cancel_futures=True)
        for process in processes:process.join(timeout=5)
        cleanup_cache_partials([process.pid for process in processes])
        self.futures.clear();self.costs.clear();self.retired.clear()
        if self.spool:self.spool.cleanup()


def cleanup_cache_partials(pids):
    """Remove only interrupted copies belonging to this pool's exited helpers."""
    import json
    cache=json.loads(os.environ.get('AUDIO_PIPELINE_CACHE','{}'))
    if not cache.get('enabled') or not cache.get('root'):return
    root=Path(cache['root'])
    if not (root/'.audio-pipeline-cache-v1').is_file():return
    for pid in pids:
        for path in root.glob(f'*.partial-{pid}'):path.unlink(missing_ok=True)
