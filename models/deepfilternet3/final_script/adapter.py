"""Exact offline whole-recording enhancement, one channel at a time.

The installed offline model has sequence-level GRUs/convolutions. A single forward
per channel preserves all recurrent and normalization context. CPU is the bounded
VRAM default; refuse oversized jobs before allocation. No approximate block resets.
"""
import os
from pathlib import Path
import sys
from audio_pipeline.common import BASE, fingerprint, load_module

class Adapter:
    def __init__(self,job):
        self.job=job
        os.environ['DEVICE']='cuda:0' if job['device'] is not None else 'cpu'
        sys.path.insert(0,str(BASE/'models/deepfilternet3/DeepFilterNet'))
        from df.enhance import init_df,enhance
        self.model,self.state,*_=init_df(str(job.get('checkpoint') or BASE/'models/deepfilternet3/checkpoint/DeepFilterNet3'),log_file=None)
        self.enhance=enhance
        self.metrics=dict(prepare_wait_seconds=0.,feature_seconds=0.,model_and_transfer_seconds=0.,synthesis_seconds=0.,native_enhance_seconds=0.)
        self.feature_preparation=load_module(Path(__file__).with_name('feature_preparation.py'),'enhancement_feature_preparation')
        from df.model import ModelParams
        parameters=ModelParams()
        self.frontend_parameters=dict(sr=parameters.sr,fft_size=parameters.fft_size,hop_size=parameters.hop_size,
                                      nb_bands=parameters.nb_erb,min_nb_erb_freqs=parameters.min_nb_freqs)
        self.nb_df=getattr(self.model,'nb_df',getattr(self.model,'df_bins',parameters.nb_df))
        self.gru_chunk_frames=None
        if job['device'] is not None:
            self.gru_chunk_frames=int(job['settings'].get('gru_chunk_frames',30000))
            recurrent=load_module(Path(__file__).with_name('recurrent.py'),'enhancement_recurrent')
            recurrent.bound_gru_sequences(self.model,self.gru_chunk_frames)
        self.prepare=load_module(Path(__file__).with_name('preparation.py'),'enhancement_preparation').prepare
        self.prefetch=None
        self.prepare_gib=float(job['settings'].get('prepare_max_gib',1))
        self.channel_batch_size=int(job['settings'].get('channel_batch_size',1))
        if not 0<self.prepare_gib<=16 or self.channel_batch_size<1:
            raise ValueError('prepare_max_gib must be in (0,16]; channel_batch_size must be positive')

    def start(self,items):
        """One future recording overlaps current inference; at most two inputs resident."""
        enabled=self.job['settings'].get('prefetch_recordings')
        if enabled is None:enabled=self.job['device'] is not None
        if enabled:
            from audio_pipeline.prefetch import AudioPrefetch
            from audio_pipeline.audio import estimate_recording_bytes
            cfg=self.job['settings']
            self.prefetch=AudioPrefetch(self.prepare_recording,workers=cfg.get('prepare_workers',2),
                lookahead=cfg.get('prepare_lookahead',2),max_bytes=int(self.prepare_gib*2**30),
                estimate=lambda item:estimate_recording_bytes(item,48000)*(2 if cfg.get('prepare_spectral_features',False) else 1))
            self.prefetch.start([dict(item=item) for item in items])

    def prepare_recording(self,item):
        if self.job['settings'].get('prepare_spectral_features',False):
            return self.feature_preparation.prepare(item,self.prepare,self.prepare_gib,self.frontend_parameters,self.nb_df)
        return self.prepare(item,self.prepare_gib)

    def close(self):
        self.close_job()

    def close_job(self):
        if self.prefetch:self.prefetch.close();self.prefetch=None
        if getattr(self,'publisher',None):
            self.publisher.shutdown(wait=True)
            for future in self.pending_publish.values():future.result()
            self.publisher=None;self.pending_publish={}

    def finalize_result(self,item,value):
        future=getattr(self,'pending_publish',{}).pop(value['path'],None)
        if future is not None:value['artifacts']=[future.result()]


    def process(self,item):
        import numpy as np
        import soundfile as sf
        import torch
        from audio_pipeline.audio import read_audio
        cfg=self.job['settings']
        if self.job['device'] is not None and item['duration']>cfg.get('max_gpu_seconds',1800):
            raise RuntimeError('Long whole-sequence CUDA enhancement exceeds the verified compatibility policy: installed PyTorch/cuDNN has a large-convolution indexing limit. Select --gpus cpu; CPU has passed a one-hour stress test.')
        batch_size=min(item['channels'],self.channel_batch_size)
        ram_gib=item['duration']*cfg.get('estimated_ram_gib_per_second',.04)*batch_size+2+2*self.prepare_gib
        gpu_gib=item['duration']*cfg.get('estimated_gpu_gib_per_second',.04)*batch_size+2
        from audio_pipeline.memory import available_host_bytes
        available_gib=available_host_bytes()/2**30
        # A preferred channel batch must not make an otherwise eligible file
        # fail. Reduce it before allocating, without changing device placement.
        while batch_size>1:
            fits_cpu=ram_gib<=min(cfg['max_estimated_ram_gib'],available_gib*cfg.get('host_headroom_fraction',.75))
            fits_gpu=(self.job['device'] is None or (
                gpu_gib<=torch.cuda.mem_get_info()[0]/2**30*cfg.get('gpu_headroom_fraction',.75) and
                batch_size*item['duration']<=cfg.get('max_gpu_seconds',1800)))
            if fits_cpu and fits_gpu:break
            batch_size-=1
            ram_gib=item['duration']*cfg.get('estimated_ram_gib_per_second',.04)*batch_size+2+2*self.prepare_gib
            gpu_gib=item['duration']*cfg.get('estimated_gpu_gib_per_second',.04)*batch_size+2
        estimated_gib=ram_gib
        if item['duration']>cfg['max_seconds'] or estimated_gib>cfg['max_estimated_ram_gib']:
            raise RuntimeError(f'Whole-sequence estimate {estimated_gib:.1f} GiB exceeds configured enhancement limit; increase limits after measuring')
        if estimated_gib>available_gib*cfg.get('host_headroom_fraction',.75):raise RuntimeError('Insufficient available host RAM for continuous enhancement')
        if self.job['device'] is not None and gpu_gib>torch.cuda.mem_get_info()[0]/2**30*cfg.get('gpu_headroom_fraction',.75):
            raise RuntimeError(f'Estimated enhancement memory {gpu_gib:.1f} GiB exceeds GPU headroom; select CPU explicitly using --gpus cpu')
        target=Path(item['_artifact_dir'])/'enhanced.wav'
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.partial.wav')
        frames=round(item['total_samples']*48000/item['sample_rate'])
        import time
        if not hasattr(self,'metrics'):self.metrics=dict(prepare_wait_seconds=0.,feature_seconds=0.,model_and_transfer_seconds=0.,synthesis_seconds=0.,native_enhance_seconds=0.)
        tick=time.perf_counter()
        prepared=self.prefetch.take(item) if self.prefetch else self.prepare_recording(item)
        self.metrics['prepare_wait_seconds']+=time.perf_counter()-tick
        spectral=isinstance(prepared,dict) and prepared.get('spectral_features',False)
        if spectral:self.metrics['feature_seconds']+=prepared['feature_seconds']
        if prepared is None:batch_size=1
        # Mono (and a complete native channel batch) writes directly. Only
        # separately enhanced multichannel output needs disk-backed interleaving.
        import tempfile
        async_audio=cfg.get('async_audio_write',False)
        scratch=Path(self.job.get('allocation',{}).get('optimization',{}).get('scratch_root',tempfile.gettempdir()))
        scratch.mkdir(parents=True,exist_ok=True)
        mmap_path=scratch/(item['id']+'-'+__import__('uuid').uuid4().hex+'.buffer') if async_audio else target.with_suffix('.buffer')
        output=None
        try:
            if batch_size<item['channels']:
                output=np.memmap(mmap_path,dtype='float32',mode='w+',shape=(frames,item['channels']))
            for channel in range(0,item['channels'],batch_size):
                if spectral:
                    with torch.inference_mode():enhanced,timings=self.feature_preparation.enhance(self.model,prepared,channel,batch_size)
                    for key,value in timings.items():self.metrics[key]+=value
                    actual_channels=min(batch_size,item['channels']-channel);audio=None
                else:
                    if prepared is None:
                        audio,_=read_audio(item['path'],channel=channel,rate=48000)
                        audio=audio[None]
                    else:audio=np.ascontiguousarray(prepared[:,channel:channel+batch_size].T)
                    tick=time.perf_counter()
                    with torch.inference_mode():enhanced=self.enhance(self.model,self.state,torch.from_numpy(audio),pad=True).numpy()
                    self.metrics['native_enhance_seconds']+=time.perf_counter()-tick
                    actual_channels=audio.shape[0]
                if enhanced.shape[0]!=actual_channels or enhanced.shape[1]<frames or not np.isfinite(enhanced).all():
                    raise ValueError('Invalid enhanced audio shape or values')
                enhanced=enhanced[:,:frames]
                if output is not None:output[:,channel:channel+actual_channels]=enhanced.T
                elif async_audio:output=enhanced.T
                else:
                    with sf.SoundFile(temporary,'w',samplerate=48000,channels=item['channels'],subtype='FLOAT') as stream:
                        for start in range(0,frames,max(1,round(48000*cfg.get('assembly_block_seconds',10)))):stream.write(enhanced[:,start:start+max(1,round(48000*cfg.get('assembly_block_seconds',10)))].T)
                del enhanced,audio
            if output is not None and not async_audio:
                with sf.SoundFile(temporary,'w',samplerate=48000,channels=item['channels'],subtype='FLOAT') as stream:
                    for start in range(0,frames,max(1,round(48000*cfg.get('assembly_block_seconds',10)))):stream.write(output[start:start+max(1,round(48000*cfg.get('assembly_block_seconds',10)))])
            if async_audio:
                from concurrent.futures import ThreadPoolExecutor
                from audio_pipeline.staging import publish_audio
                if not getattr(self,'publisher',None):
                    self.publisher=ThreadPoolExecutor(max_workers=1,thread_name_prefix='enhanced-audio-publish')
                    self.pending_publish={}
                # One publishing buffer may overlap the current model call.
                for future in list(self.pending_publish.values()):
                    if not future.done():future.result()
                self.pending_publish[str(target)]=self.publisher.submit(publish_audio,output,target,scratch)
            else:os.replace(temporary,target)
        finally:
            del output
            mmap_path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
        return dict(recording_id=item['id'],path=str(target),sample_rate=48000,total_samples=frames,
                    channels=item['channels'],duration=frames/48000,artifacts=[] if async_audio else [fingerprint(target)],
                    mode='whole_sequence_per_channel',channel_batch_size=batch_size,
                    single_decode=prepared is not None,prefetch_enabled=self.prefetch is not None,
                    gru_chunk_frames=getattr(self,'gru_chunk_frames',None),
                    estimated_ram_gib=ram_gib,estimated_gpu_gib=gpu_gib if self.job['device'] is not None else None)
