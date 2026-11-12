from pathlib import Path
import tempfile
from audio_pipeline.common import BASE, load_module
from audio_pipeline.audio import chunk_audio
from audio_pipeline.asr_completion import monitor_generation,recover_unfinished

class Adapter:
    def __init__(self,job):
        import torch
        backend=load_module(BASE/'models/firered-asr2/final_script/backend.py','existing_firered_backend')
        self.job=job;self.features=None;self.current={}
        self.model=backend.FireRedBackend(checkpoint=job['checkpoint'],beam_size=job['settings'].get('beam_size',3),
                                         max_tokens=job['settings']['max_new_tokens'],threads=job['threads'])
        torch.set_num_threads(job['threads'])
        if job['settings'].get('engine','transformers')=='vllm':
            from audio_pipeline.vllm_decoder import replace_decoder
            replace_decoder(self.model.asr.model,job,Path(job['checkpoint'])/'Qwen2-7B-Instruct')
        elif job['settings'].get('engine','transformers')!='transformers':raise ValueError('Unsupported FireRed engine')
        self.stats=monitor_generation(self.model.asr.model.llm)
        self.metrics={}

    def start(self,items):
        if self.job['settings'].get('frontend_workers',0):
            from audio_pipeline.frontends import FeaturePrefetch
            if self.features is None:self.features=FeaturePrefetch('firered',self.job)
            else:self.features.reset_job(self.job)
            self.features.start([dict(item=item) for item in items])
            self.model.asr.feat_extractor=self.prepared_features

    def close_job(self):
        if self.features:self.features.reset()
        self.current={}

    def close(self):
        if self.features:
            self.metrics['frontend_wait_seconds']=self.features.seconds
            self.metrics['frontend_peak_queued_bytes']=self.features.peak_reserved_bytes
            self.features.close();self.features=None
        closer=getattr(self.model.asr.model.llm,'close',None)
        if closer is not None:closer()

    def prepared_features(self,paths,ids):
        import torch
        values=[self.current[identity] for identity in ids]
        arrays=[torch.from_numpy(v['features']) for v in values]
        return (torch.nn.utils.rnn.pad_sequence(arrays,batch_first=True),
                torch.tensor([len(a) for a in arrays]),[v['duration'] for v in values],paths,ids)

    def process(self,item):
        return self.process_batch([item])[0]

    def process_batch(self,items):
        if self.features:self.current={item['id']:self.features.take(item) for item in items}
        try:values=recover_unfinished(items,self._predict,self.metrics)
        finally:self.current={}
        self.stats[:]=[dict(value['generation'][0]) for value in values]
        return values

    def _predict(self,items):
        import numpy as np
        audios=[]
        for item in items:
            if self.features:
                audios.append('prepared:'+item['id']);continue
            audio,rate=chunk_audio(item)
            # load_chunk has already quantized to PCM16; Kaldi expects PCM scale.
            audios.append([rate,np.rint(audio*32768).astype(np.int16)])
        values=self.model.transcribe([item['id'] for item in items],audios)
        if len(values)!=len(items) or len(self.stats)!=len(items):
            raise RuntimeError('FireRed returned incomplete predictions or generation metadata')
        stats=[dict(stat) for stat in self.stats]
        return ([dict(text=value['text'],chunk=item,metadata=self.model.metadata,generation=[dict(stat)])
                for item,value,stat in zip(items,values,stats)],stats)
