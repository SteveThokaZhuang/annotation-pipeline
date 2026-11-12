"""Conservative per-channel speech regions; no original audio is discarded."""
import math
import time
from audio_pipeline.common import BASE, load_module
from pathlib import Path


class PreparedFeatures:
    def __init__(self,features,duration):self.features=features;self.duration=duration
    def extract(self,_):return self.features,self.duration


class Adapter:
    def __init__(self,job):
        import torch
        from fireredvad import FireRedVad,FireRedVadConfig
        self.cfg=job['settings']
        config=FireRedVadConfig(use_gpu=job['device'] is not None,
            speech_threshold=self.cfg['speech_threshold'],min_speech_frame=max(1,math.ceil(self.cfg['min_speech_ms']/10)),
            min_silence_frame=15,max_speech_frame=3000,merge_silence_frame=0,extend_speech_frame=0)
        self.checkpoint=Path(job.get('checkpoint') or BASE/'models/VAD/FireRedVAD/checkpoints/VAD')
        self.model=FireRedVad.from_pretrained(str(self.checkpoint),config)
        self.device=str(next(self.model.vad_model.parameters()).device)
        if (job['device'] is not None)!=self.device.startswith('cuda'):raise RuntimeError('VAD model device does not match allocation')
        self.regions=load_module(BASE/'vendor/vad_common.py','existing_regions').make_regions
        self.allocation=job.get('allocation',{});self.stop_file=job.get('stop_file')
        self._ready_prepared={}
        self.preparation=None;self.items=[];self.positions={};self.next_index=0
        self.metrics=dict(prepare_wait_seconds=0.,read_seconds=0.,feature_seconds=0.,spool_seconds=0.,
            model_seconds=0.,prepared_recordings=0,prepare_processes=self.allocation.get('vad_prepare_processes',0))

    def start(self,items):
        workers=self.allocation.get('vad_prepare_processes',0)
        if not workers:return
        from audio_pipeline.vad_preparation import PreparationPool
        self.items=items;self.positions={item['id']:i for i,item in enumerate(items)};self.next_index=0
        if self.preparation is None:
            self.preparation=PreparationPool(workers,self.allocation.get('vad_prepare_lookahead',workers*2),
                int(self.allocation.get('vad_prepare_max_gib',2)*2**30),
                self.allocation.get('optimization',{}).get('scratch_root','/tmp/audio-pipeline-optimized'),
                self.checkpoint/'cmvn.ark',self.stop_file)
        self.preparation.stop_file=Path(self.stop_file) if self.stop_file else None
        self.preparation.start(items)
        self.metrics.update(self.preparation.metrics)

    def close_job(self):
        self._ready_prepared.clear()
        if self.preparation:self.preparation.reset()

    def close(self):
        self._ready_prepared.clear()
        if self.preparation:
            self.metrics.update(self.preparation.metrics)
            self.preparation.close();self.preparation=None

    def process(self,item):
        if item['id'] in self._ready_prepared:
            try:return self._process(item,self._ready_prepared[item['id']])
            finally:self._ready_prepared.pop(item['id'],None)
        prepared=None
        if self.preparation:
            tick=time.perf_counter()
            index=self.positions[item['id']]
            for skipped in ([] if self.allocation.get('prepare_order')=='ready' else self.items[self.next_index:index]):self.preparation.discard(skipped)
            self.next_index=index+1
            try:
                prepared=self.preparation.take(item)
                self.metrics['prepare_wait_seconds']+=time.perf_counter()-tick
                for key,value in prepared['metrics'].items():self.metrics[key]+=value
                self.metrics['prepared_recordings']+=1
                self.metrics.update(self.preparation.metrics)
                return self._process(item,prepared)
            finally:
                if prepared is None:self.metrics['prepare_wait_seconds']+=time.perf_counter()-tick
                self.preparation.release(item)
        return self._process(item,None)

    def process_batch(self,items):
        """Batch independent feature sequences with explicit masks at every FSMN layer."""
        if not self.preparation or (len(items)<2 and self.allocation.get('prepare_order')!='ready'):return [self.process(item) for item in items]
        import numpy as np
        import torch
        prepared_items=[];jobs=[];parts={}
        for item in items:
            prepared=self._ready_prepared.get(item['id'])
            if prepared is None:
                index=self.positions[item['id']]
                for skipped in ([] if self.allocation.get('prepare_order')=='ready' else self.items[self.next_index:index]):self.preparation.discard(skipped)
                self.next_index=index+1;tick=time.perf_counter()
                prepared=self.preparation.take(item)
                self.metrics['prepare_wait_seconds']+=time.perf_counter()-tick
                for key,value in prepared['metrics'].items():self.metrics[key]+=value
                self.metrics['prepared_recordings']+=1
                for data in prepared['channels']:
                    # Keep mappings alive for OOM/single-item recovery after unlink.
                    data['audio_array']=np.load(data['audio'],mmap_mode='r',allow_pickle=False)
                    data['feature_array']=np.load(data['features'],mmap_mode='c',allow_pickle=False)
                self._ready_prepared[item['id']]=prepared
                self.preparation.release(item)
            for channel,data in enumerate(prepared['channels']):
                features=torch.from_numpy(data['feature_array'])
                key=(item['id'],channel);parts[key]=[]
                for piece in features.split(self.model.config.chunk_max_frame,dim=0):jobs.append((key,piece))
            prepared_items.append(prepared)
        self.metrics.update(self.preparation.metrics)
        limit=self.cfg.get('gpu_batch_size',len(items));tick=time.perf_counter()
        try:
            with torch.inference_mode():
                for offset in range(0,len(jobs),limit):
                    if self.stop_file and Path(self.stop_file).exists():raise InterruptedError('Stopped during VAD batch')
                    group=jobs[offset:offset+limit];lengths=torch.tensor([len(f) for _,f in group],device=self.device)
                    from audio_pipeline.transfers import to_device
                    features=to_device(torch.nn.utils.rnn.pad_sequence([f for _,f in group],batch_first=True),self.device)
                    hidden,_=self.model.vad_model.dfsmn(features,input_lengths=lengths)
                    probabilities=torch.sigmoid(self.model.vad_model.out(hidden)).cpu()
                    for row,(key,feature) in enumerate(group):parts[key].append(probabilities[row,:len(feature)].squeeze(-1))
                    self.metrics['gpu_batches']=self.metrics.get('gpu_batches',0)+1
            self.metrics['model_seconds']+=time.perf_counter()-tick
            self._batched_probs={key:torch.cat(value) for key,value in parts.items()}
            results=[self._process(item,prepared) for item,prepared in zip(items,prepared_items)]
            for item in items:self._ready_prepared.pop(item['id'],None)
            return results
        finally:self._batched_probs={}

    def _process(self,item,prepared):
        import numpy as np
        import torch
        from audio_pipeline.audio import read_audio
        chunks=[];channels=[]
        for channel in range(item['channels']):
            if prepared:
                data=prepared['channels'][channel]
                audio=data['audio_array'] if 'audio_array' in data else np.load(data['audio'],mmap_mode='r',allow_pickle=False);rate=16000
                batched=getattr(self,'_batched_probs',{}).get((item['id'],channel))
                if batched is not None:
                    probs=batched
                    decisions=self.model.vad_postprocessor.process(probs.tolist())
                    result=dict(timestamps=self.model.vad_postprocessor.decision_to_segment(decisions,data['detector_duration']))
                else:
                    features=data['feature_array'] if 'feature_array' in data else np.load(data['features'],mmap_mode='c',allow_pickle=False)
                    original=self.model.audio_feat
                    self.model.audio_feat=PreparedFeatures(torch.from_numpy(features),data['detector_duration'])
                    tick=time.perf_counter()
                    try:
                        with torch.inference_mode():result,probs=self.model.detect(None)
                    finally:
                        self.model.audio_feat=original
                        self.metrics['model_seconds']+=time.perf_counter()-tick
            else:
                audio,rate=read_audio(item['path'],channel=channel,rate=16000)
                pcm=np.rint(np.clip(audio,-1,32767/32768)*32768).astype(np.int16)
                # Offline implementation requires enough frames for feature extraction.
                if len(pcm)<3200:pcm=np.pad(pcm,(0,3200-len(pcm)))
                with torch.inference_mode():result,probs=self.model.detect(pcm)
            if not torch.isfinite(probs).all():
                raise ValueError('Nonfinite VAD probabilities')
            duration=len(audio)/rate
            intervals=[(s,min(e,duration)) for s,e in result['timestamps'] if s<duration]
            from vad_common import quiet_energy_cut
            energy_window_ms=self.cfg.get('energy_window_ms',100.)
            def quiet_cut(lower,upper):
                return quiet_energy_cut(audio,lower,upper,energy_window_ms)
            raw,regions,boundaries=self.regions(intervals,duration,self.cfg['padding'],self.cfg['merge_gap'],
                self.cfg['chunk_seconds'],quiet_cut,min_chunk_seconds=self.cfg.get('min_chunk_seconds',10.),
                min_silence_seconds=self.cfg.get('min_silence_seconds',.1),
                fallback_search_seconds=self.cfg.get('fallback_search_seconds',2.),return_boundaries=True)
            channels.append(dict(channel=channel,intervals=raw))
            for (start,end),boundary in zip(regions,boundaries):
                for entry in boundary.values():
                    if entry['type']=='forced_low_energy':entry['energy_window_ms']=energy_window_ms
                # Store indices in original sample-rate coordinates.
                start=round(start*item['sample_rate']/16000)
                end=min(item['total_samples'],round(end*item['sample_rate']/16000))
                if end<=start:continue
                identity=f"{item['id']}_c{channel}_{start}_{end}"
                chunks.append(dict(id=identity,recording_id=item['id'],path=item['path'],channel=channel,
                    sample_rate=item['sample_rate'],start_sample=start,end_sample=end,
                    total_samples=item['total_samples'],start=start/item['sample_rate'],end=end/item['sample_rate'],
                    duration=(end-start)/item['sample_rate'],source_fingerprint=item['source_fingerprint'],
                    **boundary))
        return dict(recording_id=item['id'],device=self.device,channels=channels,chunks=chunks,
                    preservation_note='40ms minimum speech, 250ms context, channels independent; audit missed vocal events manually')
