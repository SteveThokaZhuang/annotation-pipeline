"""Overlap DNSMOS window construction/calibration with one ONNX session owner."""
from concurrent.futures import ThreadPoolExecutor
import math
import time


class DNSMOSPipeline:
    def __init__(self,adapter,job,settings,stop_requested=lambda:False):
        self.base=adapter;self.stop_requested=stop_requested
        gib=settings.get('max_gib',1)
        if not isinstance(gib,(int,float)) or isinstance(gib,bool) or not math.isfinite(gib) or gib<=0:
            raise ValueError('DNSMOS window pipeline max_gib must be positive')
        self.budget=int(gib*2**30)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dnsmos-onnx')
        self.metrics=adapter.metrics
        self.metrics.update(pipeline_window_prepare_seconds=0.,pipeline_window_wait_seconds=0.,
                            pipeline_peak_window_bytes=0)

    def __getattr__(self,name):return getattr(self.base,name)
    def start(self,items):self.base.start(items)
    def process_batch(self,items):return [self.process(item) for item in items]

    def process(self,item):
        import numpy as np
        base=self.base;tick=time.perf_counter()
        try:
            if base.prefetch:
                index=base.positions[item['id']]
                for skipped in base.items[base.next_index:index]:base.prefetch.discard(skipped)
                base.next_index=index+1;audios=base.prefetch.take(item)
            else:audios=base._prepare(item)
        finally:self.metrics['prepare_wait_seconds']+=time.perf_counter()-tick
        channels=[];rate=16000;length=144160;hop=round(base.cfg['hop_seconds']*rate)
        coefficients=[[-.08397278,1.22083953,.0052439],[-.13166888,1.60915514,-.39604546],[-.06766283,1.11546468,.04602535]]
        for channel,audio in enumerate(audios):
            original=len(audio)
            while len(audio)<length:audio=np.concatenate([audio,audio])
            starts=[s for s in range(0,max(0,int(np.floor(len(audio)/rate)-9.01)+1)*rate,hop) if s+length<=len(audio)]
            windows=[];offset=0;future=None;selected=None;reserved=0
            def prepare_next():
                nonlocal offset
                selected=starts[offset:offset+base.batch_limit]
                tick=time.perf_counter()
                features=np.stack([audio[s:s+length] for s in selected]).astype(np.float32,copy=False)
                self.metrics['pipeline_window_prepare_seconds']+=time.perf_counter()-tick
                offset+=len(selected)
                return selected,features
            try:
                if starts:
                    selected,features=prepare_next();reserved=features.nbytes
                    future=self.pool.submit(base._infer,features)
                    del features
                    self.metrics['pipeline_peak_window_bytes']=max(self.metrics['pipeline_peak_window_bytes'],reserved)
                while future is not None:
                    if self.stop_requested():raise InterruptedError('Stopped during DNSMOS window pipeline')
                    following=None
                    next_bytes=min(base.batch_limit,len(starts)-offset)*length*4
                    if offset<len(starts) and reserved+next_bytes<=self.budget:
                        following=prepare_next()
                        self.metrics['pipeline_peak_window_bytes']=max(self.metrics['pipeline_peak_window_bytes'],reserved+following[1].nbytes)
                    tick=time.perf_counter();raw=future.result();future=None
                    self.metrics['pipeline_window_wait_seconds']+=time.perf_counter()-tick
                    completed=selected
                    if following is None and offset<len(starts):following=prepare_next()
                    if following is not None:
                        selected,features=following;reserved=features.nbytes
                        future=self.pool.submit(base._infer,features)
                        del following,features
                    # Identical float64 calibration and window order to the
                    # native adapter, while the next inference is in flight.
                    calibrated=np.stack([np.polyval(coef,raw[:,i].astype(np.float64)) for i,coef in enumerate(coefficients)],axis=1)
                    if not np.isfinite(calibrated).all() or np.any(calibrated<0) or np.any(calibrated>5.5):
                        raise ValueError('Implausible DNSMOS scores')
                    for start,values,scores in zip(completed,raw,calibrated):
                        windows.append(dict(start=start/rate,end=(start+length)/rate,
                            repeated_short_audio=original<length,source_duration=original/rate,
                            scores=dict(zip(['SIG','BAK','OVRL'],map(float,scores))),raw=list(map(float,values))))
                if not windows:raise ValueError('No DNSMOS windows')
                channels.append(dict(channel=channel,segments=windows,
                    scores={key:float(np.mean([w['scores'][key] for w in windows])) for key in ['SIG','BAK','OVRL']}))
            finally:
                # Drain the sole session call before another recording or a
                # retry uses the adapter. Preserve the original raised error.
                if future is not None:
                    try:future.result()
                    except Exception:pass
        return dict(recording_id=item['id'],channels=channels,
                    scores={key:float(np.mean([c['scores'][key] for c in channels])) for key in ['SIG','BAK','OVRL']})

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.base.close()
