"""Bounded recording preparation across resident DNSMOS bundle boundaries.

This changes scheduling only. Native decode, window placement, inference and
result validation remain in the existing adapter. Future bundles are prepared
speculatively but never committed before their normal worker job executes.
"""
from pathlib import Path
from .audio import estimate_recording_bytes
from .prefetch import AudioPrefetch
from .task_manifest import read_tasks


class RecordingPreparation:
    def __init__(self,base):
        self.base=base;self.prefetch=None;self.domain=None;self.current=[]

    def start(self,items,job):
        cfg=job['settings'];domain=(job['root'],job['stop_file'])
        if self.domain!=domain:
            self.close();self.domain=domain
        if self.prefetch is None:
            self.prefetch=AudioPrefetch(self.base._prepare,
                workers=cfg.get('prepare_workers',2),lookahead=cfg.get('prepare_lookahead',2),
                max_bytes=int(cfg.get('prepare_max_gib',1)*2**30),
                estimate=estimate_recording_bytes,stop_file=job['stop_file'])
        upcoming=[]
        if job.get('prefetch_tasks_file'):
            upcoming=[task['item'] for task in read_tasks(job['prefetch_tasks_file'])]
        combined=list(items)+upcoming
        identities=[item['id'] for item in combined]
        if len(identities)!=len(set(identities)):
            raise ValueError('Duplicate recording IDs across preparation bundles')
        pool=self.prefetch;allowed=set(identities)
        # Drop obsolete work before resetting the bounded iterator. Retired
        # running loaders keep their reservations until completion.
        pool.tasks=[];pool.cursor=0
        for identity in list(pool.futures):
            if identity not in allowed:pool.discard({'id':identity})
        ready=sum(pool.futures[i['id']].done() for i in items if i['id'] in pool.futures)
        pool.handled.clear();pool.seconds=0.
        pool.tasks=[{'item':item} for item in combined if item['id'] not in pool.futures]
        pool._fill()
        self.current=list(items)
        self.base.items=combined;self.base.positions={item['id']:i for i,item in enumerate(combined)}
        self.base.next_index=0;self.base.prefetch=pool
        self.base.metrics.update(prefetch_ready_at_bundle_start=ready,
                                 prefetch_upcoming_recordings=len(upcoming))

    def finish_job(self):
        if self.prefetch is None:return
        # Cancel skipped/failed current items, but keep next-bundle futures and
        # the same preparation threads alive. Nothing here publishes results.
        for item in self.current:self.prefetch.discard(item)
        self.base.metrics['prefetch_peak_reserved_bytes']=self.prefetch.peak_reserved_bytes
        self.current=[]

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()
            if self.base.prefetch is self.prefetch:self.base.prefetch=None
            self.prefetch=None
        self.current=[];self.domain=None
