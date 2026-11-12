"""Bounded FunASR encoding ahead of vLLM decoding, without changing checkpoints.

Only the producer touches the native FunASR model and its mutable frontend state.
The consumer owns vLLM generation and EOS statistics. The native inference path
builds the same padded prompts; a private control-flow exception hands those
prompts to the consumer before generation. This module changes scheduling only:
the worker retains ownership of validation, durable saving, and task signatures.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,field
import copy
import math
import re
import threading
import time

from .asr_completion import recover_unfinished


@dataclass
class Prepared:
    items: list
    arguments: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    error: Exception = None
    consumed: bool = False


class _Captured(BaseException):
    """Leave native inference after prompt assembly, before any generation."""
    def __init__(self,arguments):
        self.arguments=arguments


class FunASRPipeline:
    def __init__(self,adapter,job,settings,stop_requested=lambda:False):
        self.base=adapter
        self.job=job
        self.depth=settings.get('lookahead_batches',2)
        gib=settings.get('max_gib',4)
        if not isinstance(self.depth,int) or isinstance(self.depth,bool) or not 1<=self.depth<=4:
            raise ValueError('FunASR pipeline lookahead_batches must be an integer in [1,4]')
        if not isinstance(gib,(int,float)) or isinstance(gib,bool) or not math.isfinite(gib) or gib<=0:
            raise ValueError('FunASR pipeline max_gib must be positive')
        self.budget=int(gib*2**30)
        self.stop_requested=stop_requested
        self.closing=threading.Event()
        self.current=None
        self.futures=set()
        self.metrics=adapter.metrics
        self.metrics.update(pipeline_prepare_seconds=0.,pipeline_capture_seconds=0.,
                            pipeline_decode_seconds=0.,pipeline_wait_seconds=0.,
                            pipeline_peak_reserved_bytes=0,pipeline_batches=0)
        self.decoder=self.decoder_for(adapter)
        self.generate=self.decoder.generate
        self.tokenizer=self.tokenizer_for(adapter)
        # The producer captures generation instead of entering vLLM. The saved
        # generation wrapper is invoked exclusively by the consumer.
        self.decoder.generate=self._capture
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='funasr-encode')

    def __getattr__(self,name):
        return getattr(self.base,name)

    def decoder_for(self,adapter):return adapter.model.model.llm

    def tokenizer_for(self,adapter):return copy.deepcopy(adapter.model.kwargs['tokenizer'])

    def start(self,items):
        self.base.start(items)

    def _stopped(self):
        return self.closing.is_set() or self.stop_requested()

    def _capture(self,*args,**kwargs):
        import torch
        if args:raise ValueError('Expected native FunASR keyword generation arguments')
        started=time.perf_counter()
        # Batch the device-to-host handoff. The decoder bridge receives the
        # identical padded values/mask, then removes padding exactly as before.
        arguments={key:value.detach().to(device='cpu').contiguous()
                   if isinstance(value,torch.Tensor) else value for key,value in kwargs.items()}
        self.metrics['pipeline_capture_seconds']+=time.perf_counter()-started
        raise _Captured(arguments)

    def _prepare(self,items,features=None):
        import torch
        started=time.perf_counter()
        current={}
        try:
            if self._stopped():raise InterruptedError('Stopped before FunASR preparation')
            if self.base.features:
                for item in items:
                    identity=item['id']
                    current[identity]=(features[identity] if features is not None and identity in features
                                       else self.base.features.take(item))
            self.base.current=current
            try:
                # Reuse prompt creation, encoder microbatching, padding, dtype
                # and generation options from the unchanged native adapter.
                with torch.inference_mode():self.base._predict(items)
            except _Captured as captured:
                return Prepared(items,arguments=captured.arguments,features=current)
            raise RuntimeError('FunASR did not produce a generation request')
        except Exception as error:
            # Do not retain failed encoder tensors through an exception stack.
            return Prepared(items,features=current,error=error.with_traceback(None))
        finally:
            self.base.current={}
            self.metrics['pipeline_prepare_seconds']+=time.perf_counter()-started

    def _estimate(self,group):
        # Conservative scheduling reservation for padded embeddings plus CPU
        # features. As with audio prefetch, this is an estimate, not total RSS.
        weight=self.decoder.get_input_embeddings().weight
        hidden=int(weight.shape[-1]);width=weight.element_size()
        duration=max((float(t['item'].get('duration',t['item'].get('chunk',{}).get('duration',0)))
                      for t in group),default=0)
        tokens=max(4096,1024+math.ceil(duration*100))
        return len(group)*(tokens*hidden*width+math.ceil(duration*16000*4)*2)

    def groups(self,groups):
        """Yield the worker's actual task groups with one batch prepared ahead."""
        iterator=iter(groups);pending=deque();deferred=None;reserved=0;exhausted=False
        try:
            while not self._stopped():
                while not exhausted and len(pending)<self.depth:
                    if deferred is None:
                        group=next(iterator,None)
                        if group is None:exhausted=True;break
                        deferred=(group,self._estimate(group))
                    group,cost=deferred
                    if pending and reserved+cost>self.budget:break
                    # An oversized batch runs alone, rather than deadlocking or
                    # dropping work. At most depth batches are retained.
                    future=self.pool.submit(self._prepare,[t['item'] for t in group])
                    self.futures.add(future)
                    pending.append((group,future,cost));reserved+=cost;deferred=None
                    self.metrics['pipeline_peak_reserved_bytes']=max(self.metrics['pipeline_peak_reserved_bytes'],reserved)
                if not pending:break
                group,future,cost=pending.popleft()
                started=time.perf_counter()
                self.current=future.result()
                self.metrics['pipeline_wait_seconds']+=time.perf_counter()-started
                self.futures.discard(future)
                try:yield group
                finally:
                    self.current=None
                    reserved-=cost
        finally:
            self.current=None
            for _,future,_ in pending:future.cancel()

    def _decode(self,prepared):
        if prepared.error is not None:raise prepared.error
        started=time.perf_counter()
        try:
            output=self.generate(**prepared.arguments)
            stats=[dict(stat) for stat in self.base.stats]
            texts=self.tokenizer.batch_decode(output,skip_special_tokens=True)
            if len(texts)!=len(prepared.items) or len(stats)!=len(prepared.items):
                raise RuntimeError('Incomplete pipelined FunASR output/generation metadata')
            # Match native _inference_llm_batch normalization, then the adapter's
            # existing result schema. EOS handling remains in the saved bridge.
            results=[dict(text=re.sub(r'\s+',' ',text.replace('/sil',' ')),
                          generation=dict(stat),chunk=item)
                     for item,text,stat in zip(prepared.items,texts,stats)]
            return results,stats
        finally:
            self.metrics['pipeline_decode_seconds']+=time.perf_counter()-started
            # Recovery re-encodes smaller subsets with their original frontend
            # features, so it does not retain an extra full embedding batch.
            prepared.arguments={}

    def process_batch(self,items):
        current=self.current
        if current is None:raise RuntimeError('FunASR pipeline requires its prepared task iterator')
        def predict(subset):
            if (not current.consumed and
                    [item['id'] for item in subset]==[item['id'] for item in current.items]):
                current.consumed=True
                prepared=current
            else:
                # One producer also owns retries; it cannot mutate native state
                # concurrently with preparation of the following batch.
                future=self.pool.submit(self._prepare,subset,current.features)
                prepared=future.result()
            return self._decode(prepared)
        values=recover_unfinished(items,predict,self.metrics)
        self.metrics['pipeline_batches']+=1
        return values

    def process(self,item):
        return self.process_batch([item])[0]

    def close(self):
        self.closing.set()
        for future in self.futures:future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.futures.clear();self.current=None
        self.decoder.generate=self.generate
        self.base.close()


def enabled(job):
    settings=job.get('allocation',{}).get('optimization',{}).get('funasr_pipeline',{})
    return (settings.get('enabled',False) and job.get('model')=='funasr' and
            job.get('stage','').startswith('asr') and job['settings'].get('engine')=='vllm')
