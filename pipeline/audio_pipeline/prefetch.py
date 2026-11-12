"""Count- and byte-bounded preparation ahead of the active inference batch.

The byte budget covers queued/running preparations, using caller estimates that
include decode/resample working space. Active inputs and model allocations have
separate budgets. An item larger than this queue budget is loaded synchronously.
"""
from concurrent.futures import ThreadPoolExecutor,TimeoutError
from pathlib import Path
import math
import time


class AudioPrefetch:
    def __init__(self,loader,workers=2,lookahead=32,max_bytes=None,estimate=None,stop_file=None):
        if workers<1 or lookahead<1:raise ValueError('Prefetch limits must be positive')
        if max_bytes is not None and (not math.isfinite(max_bytes) or max_bytes<1 or estimate is None):
            raise ValueError('Byte-bounded prefetch requires a positive budget and an estimator')
        self.stop_file=Path(stop_file) if stop_file else None
        self.loader=loader;self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='audio-prepare')
        self.lookahead=lookahead;self.max_bytes=max_bytes;self.estimate=estimate
        self.futures={};self.costs={};self.retired=[];self.handled=set()
        self.tasks=[];self.cursor=0;self.seconds=0;self.peak_reserved_bytes=0;self.closed=False

    @property
    def reserved_bytes(self):return sum(self.costs.values())+sum(cost for _,cost in self.retired)

    def start(self,tasks):
        if self.tasks:raise RuntimeError('Prefetch has already started')
        self.tasks=tasks;self._fill()

    def _fill(self):
        if self.closed:return
        # Cancellation cannot stop a loader that already started. Keep charging
        # its reservation until it finishes, including its returned array.
        self.retired=[(f,c) for f,c in self.retired if not f.done()]
        while self.cursor<len(self.tasks) and len(self.futures)+len(self.retired)<self.lookahead:
            item=self.tasks[self.cursor]['item'];identity=item['id'];chunk=item.get('chunk',item)
            if item.get('_planning_error') or identity in self.handled:
                self.handled.discard(identity);self.cursor+=1;continue
            cost=int(self.estimate(chunk)) if self.estimate else 0
            if cost<0:raise ValueError('Preparation estimate cannot be negative')
            if self.max_bytes is not None and self.reserved_bytes+cost>self.max_bytes:break
            self.cursor+=1;self.costs[identity]=cost
            self.futures[identity]=self.pool.submit(self.loader,chunk)
            self.peak_reserved_bytes=max(self.peak_reserved_bytes,self.reserved_bytes)

    def take(self,item):
        tick=time.perf_counter();identity=item['id'];future=self.futures.pop(identity,None)
        cost=self.costs.pop(identity,0)
        if future is None:self.handled.add(identity)
        try:
            if future is None:return self.loader(item.get('chunk',item))
            while True:
                if self.stop_file and self.stop_file.exists():raise InterruptedError('Stopped during audio preparation')
                try:return future.result(timeout=.2)
                except TimeoutError:
                    if future.done():raise
        finally:
            self.seconds+=time.perf_counter()-tick
            self._fill()

    def discard(self,item):
        identity=item['id'];future=self.futures.pop(identity,None);cost=self.costs.pop(identity,0)
        if future is None:self.handled.add(identity)
        elif not future.cancel() and not future.done():self.retired.append((future,cost))
        self._fill()

    def close(self):
        self.closed=True
        for future in self.futures.values():future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.futures.clear();self.costs.clear();self.retired.clear()
