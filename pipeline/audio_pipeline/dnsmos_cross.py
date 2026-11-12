"""Fixed-window DNSMOS batching across independent recordings and channels."""
from concurrent.futures import ThreadPoolExecutor
import time


class CrossRecordingDNSMOS:
    def __init__(self,base,job,settings,stop_requested=lambda:False):
        self.base=base;self.metrics=base.metrics;self.stop_requested=stop_requested
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='dnsmos-cross')
        self.metrics.update(cross_recording_batches=0,window_copy_seconds=0.)
    def __getattr__(self,name):return getattr(self.base,name)
    def start(self,items):self.base.start(items)
    def process(self,item):return self.process_batch([item])[0]
    def process_batch(self,items):
        import numpy as np
        base=self.base;rate=16000;length=144160;hop=round(base.cfg['hop_seconds']*rate)
        limit=base.batch_limit
        buffers=[np.empty((limit,length),np.float32),np.empty((limit,length),np.float32)]
        current=0;count=0;metadata=[];future=None;previous=[];results={};errors={}
        coefficients=[[-.08397278,1.22083953,.0052439],[-.13166888,1.60915514,-.39604546],[-.06766283,1.11546468,.04602535]]
        def collect():
            nonlocal future
            if future is None:return
            raw=future.result();future=None
            calibrated=np.stack([np.polyval(c,raw[:,i].astype(np.float64)) for i,c in enumerate(coefficients)],axis=1)
            if not np.isfinite(calibrated).all() or np.any(calibrated<0) or np.any(calibrated>5.5):raise ValueError('Implausible DNSMOS scores')
            for (target,start,original),values,scores in zip(previous,raw,calibrated):
                target.append(dict(start=start/rate,end=(start+length)/rate,repeated_short_audio=original<length,source_duration=original/rate,scores=dict(zip(['SIG','BAK','OVRL'],map(float,scores))),raw=list(map(float,values))))
        def flush():
            nonlocal current,count,metadata,future,previous
            if not count:return
            collect();previous=metadata;future=self.pool.submit(base._infer,buffers[current][:count])
            self.metrics['cross_recording_batches']+=1
            current=1-current;count=0;metadata=[]
        try:
            for item in items:
                if self.stop_requested():raise InterruptedError('Stopped during DNSMOS window batching')
                tick=time.perf_counter()
                try:
                    if base.prefetch:
                        index=base.positions[item['id']]
                        for skipped in base.items[base.next_index:index]:base.prefetch.discard(skipped)
                        base.next_index=index+1;audios=base.prefetch.take(item)
                    else:audios=base._prepare(item)
                except Exception as error:
                    from .failure_policy import is_infrastructure_error
                    if is_infrastructure_error(error) or isinstance(error,InterruptedError):raise
                    errors[item['id']]=str(error);continue
                finally:self.metrics['prepare_wait_seconds']+=time.perf_counter()-tick
                channels=[];results[item['id']]=channels
                for channel,audio in enumerate(audios):
                    original=len(audio)
                    if not original:raise ValueError('Empty DNSMOS audio')
                    while len(audio)<length:audio=np.concatenate([audio,audio])
                    windows=[];channels.append(dict(channel=channel,segments=windows))
                    for start in range(0,max(0,int(np.floor(len(audio)/rate)-9.01)+1)*rate,hop):
                        if start+length>len(audio):continue
                        tick=time.perf_counter();buffers[current][count]=audio[start:start+length]
                        self.metrics['window_copy_seconds']+=time.perf_counter()-tick
                        metadata.append((windows,start,original));count+=1
                        if count==limit:flush()
            flush();collect()
            output=[]
            for item in items:
                if item['id'] in errors:output.append({'_item_error':errors[item['id']]});continue
                channels=results[item['id']]
                for c in channels:
                    if not c['segments']:raise ValueError('No DNSMOS windows')
                    c['scores']={key:float(np.mean([w['scores'][key] for w in c['segments']])) for key in ['SIG','BAK','OVRL']}
                output.append(dict(recording_id=item['id'],channels=channels,scores={key:float(np.mean([c['scores'][key] for c in channels])) for key in ['SIG','BAK','OVRL']}))
            return output
        finally:
            if future is not None:
                try:future.result()
                except Exception:pass
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True);self.base.close()
