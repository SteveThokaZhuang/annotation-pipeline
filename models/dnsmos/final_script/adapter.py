"""DNSMOS window batches with bounded preparation of the next recording."""
import time
from audio_pipeline.common import BASE, load_module
from audio_pipeline.audio import read_audio,read_recording,estimate_recording_bytes


class Adapter:
    def __init__(self,job):
        self.cfg=job['settings']
        self.batch_limit=self.cfg.get('window_batch_size',1)
        if not isinstance(self.batch_limit,int) or isinstance(self.batch_limit,bool) or self.batch_limit<1:
            raise ValueError('DNSMOS window_batch_size must be a positive integer')
        if round(self.cfg['window_seconds']*16000)!=144160:
            raise ValueError('DNSMOS requires 9.01 second windows')
        if round(self.cfg['hop_seconds']*16000)<1:
            raise ValueError('DNSMOS hop must be positive')
        self.prefetch_enabled=job.get('allocation',{}).get('prefetch_recordings',True)
        self.prefetch=None;self.items=[];self.positions={};self.next_index=0
        self.metrics=dict(prepare_wait_seconds=0.,inference_seconds=0.,model_calls=0,
                          windows=0,largest_window_batch=0,window_batch_limit=self.batch_limit)
        module=load_module(BASE/'models/dnsmos/scripts/test_dnsmos.py','existing_dnsmos')
        self.model=module.DNSMOS(job.get('checkpoint') or BASE/'models/dnsmos/checkpoint','cuda' if job['device'] is not None else 'cpu',job['threads'])
        self.input_name=self.model.primary.get_inputs()[0].name

    def _prepare(self,item):
        audio=read_recording(item,16000)
        return [audio[:,channel] for channel in range(audio.shape[1])]

    def start(self,items):
        if self.prefetch_enabled:
            from audio_pipeline.prefetch import AudioPrefetch
            self.items=items;self.positions={item['id']:i for i,item in enumerate(items)}
            # One current recording plus at most one prepared recording ahead.
            self.prefetch=AudioPrefetch(self._prepare,workers=self.cfg.get('prepare_workers',2),
                lookahead=self.cfg.get('prepare_lookahead',2),
                max_bytes=int(self.cfg.get('prepare_max_gib',1)*2**30),estimate=estimate_recording_bytes)
            self.prefetch.start([dict(item=item) for item in items])

    def close(self):
        self.close_job()

    def close_job(self):
        if self.prefetch:
            self.prefetch.close();self.prefetch=None

    def _infer(self,features):
        import numpy as np
        start=time.perf_counter()
        try:
            values=self.model.primary.run(None,{self.input_name:features})[0]
        except Exception as error:
            # Release failed tensors before retrying a CUDA memory failure.
            if len(features)==1 or 'out of memory' not in str(error).lower():raise
        else:
            if values.shape!=(len(features),3):raise ValueError('Incomplete DNSMOS model output')
            self.metrics['model_calls']+=1
            self.metrics['windows']+=len(features)
            self.metrics['largest_window_batch']=max(self.metrics['largest_window_batch'],len(features))
            return values
        finally:
            self.metrics['inference_seconds']+=time.perf_counter()-start
        self.batch_limit=max(1,min(self.batch_limit,len(features)//2))
        self.metrics['window_batch_limit']=self.batch_limit
        limit=self.batch_limit
        return np.concatenate([self._infer(features[i:i+limit])
                               for i in range(0,len(features),limit)])

    def process(self,item):
        import numpy as np
        start=time.perf_counter()
        try:
            if self.prefetch:
                # Some records may have been skipped as already cached by the worker.
                index=self.positions[item['id']]
                for skipped in self.items[self.next_index:index]:self.prefetch.discard(skipped)
                self.next_index=index+1
                audios=self.prefetch.take(item)
            else:
                audios=self._prepare(item)
        finally:self.metrics['prepare_wait_seconds']+=time.perf_counter()-start
        channels=[];rate=16000;length=144160;hop=round(self.cfg['hop_seconds']*rate)
        coefficients=[[-.08397278,1.22083953,.0052439],[-.13166888,1.60915514,-.39604546],[-.06766283,1.11546468,.04602535]]
        for channel,audio in enumerate(audios):
            original=len(audio)
            while len(audio)<length:audio=np.concatenate([audio,audio])
            starts=[s for s in range(0,max(0,int(np.floor(len(audio)/rate)-9.01)+1)*rate,hop)
                    if s+length<=len(audio)]
            windows=[];offset=0
            while offset<len(starts):
                selected=starts[offset:offset+self.batch_limit]
                features=np.stack([audio[s:s+length] for s in selected]).astype(np.float32,copy=False)
                raw=self._infer(features)
                # Preserve the previous scalar calibration's float64 arithmetic.
                calibrated=np.stack([np.polyval(coef,raw[:,i].astype(np.float64))
                                     for i,coef in enumerate(coefficients)],axis=1)
                if not np.isfinite(calibrated).all() or np.any(calibrated<0) or np.any(calibrated>5.5):
                    raise ValueError('Implausible DNSMOS scores')
                for s,values,scores in zip(selected,raw,calibrated):
                    windows.append(dict(start=s/rate,end=(s+length)/rate,
                        repeated_short_audio=original<length,source_duration=original/rate,
                        scores=dict(zip(['SIG','BAK','OVRL'],map(float,scores))),raw=list(map(float,values))))
                offset+=len(selected)
            if not windows:raise ValueError('No DNSMOS windows')
            channels.append(dict(channel=channel,segments=windows,
                scores={key:float(np.mean([w['scores'][key] for w in windows])) for key in ['SIG','BAK','OVRL']}))
        return dict(recording_id=item['id'],channels=channels,
                    scores={key:float(np.mean([c['scores'][key] for c in channels])) for key in ['SIG','BAK','OVRL']})
