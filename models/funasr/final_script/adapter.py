import sys
from pathlib import Path
import tempfile
from audio_pipeline.common import BASE, load_module
from audio_pipeline.audio import chunk_audio
from audio_pipeline.asr_completion import recover_unfinished

class Adapter:
    def __init__(self,job):
        from funasr import AutoModel
        backend=load_module(BASE/'models/funasr/scripts/backend.py','funasr_generation_fix')
        checkpoint=job['checkpoint'];self.cfg=job['settings']
        self.job=job;self.features=None;self.current={}
        self.model=AutoModel(model=checkpoint,hub='hf',device='cuda:0',disable_update=True,
            disable_pbar=True,ncpu=job['threads'],ctc_decoder=None,
            llm_conf={'llm_dtype':'bf16','init_param_path':checkpoint+'/Qwen3-0.6B',
                      'load_kwargs':{'attn_implementation':'sdpa'}})
        # The installed frontend defaults to random PCM-level dithering, even
        # during inference. Preserve that default unless explicitly configured.
        if 'frontend_dither' in self.cfg:
            value=float(self.cfg['frontend_dither'])
            if value<0:raise ValueError('frontend_dither must be nonnegative')
            self.model.kwargs['frontend'].dither=value
        if self.cfg.get('engine','transformers')=='vllm':
            from audio_pipeline.vllm_decoder import replace_decoder
            replace_decoder(self.model.model,job,Path(checkpoint)/'Qwen3-0.6B')
        elif self.cfg.get('engine','transformers')!='transformers':raise ValueError('Unsupported FunASR engine')
        self.stats=backend.install_generation_fix(self.model.model.llm)
        self.metrics={}
        encoder_batch=self.cfg.get('encoder_batch_size',32)
        if encoder_batch>1:
            batching=load_module(BASE/'models/funasr/final_script/encoder_batching.py','funasr_encoder_batching')
            batching.install(self.model.model,encoder_batch,self.cfg.get('encoder_max_padded_frames',16000),self.metrics)
        if self.cfg.get('engine')=='vllm':
            # Native single-item inference keeps two training-target mask slots.
            # The native batch path rebuilds a mask for source embeddings only,
            # and also handles a singleton during token-limit recovery correctly.
            self.model.model.inference_llm=self.model.model._inference_llm_batch

    def start(self,items):
        if self.cfg.get('frontend_workers',0):
            from audio_pipeline.frontends import FeaturePrefetch
            if self.features is None:
                self.features=FeaturePrefetch('funasr',self.job,frontend=self.model.kwargs['frontend'],
                    tokenizer=self.model.kwargs['tokenizer'],low_frame_rate=self.model.model.use_low_frame_rate)
            else:self.features.reset_job(self.job)
            self.features.start([dict(item=item) for item in items])
            self.original_data_load=self.model.model.data_load_speech
            self.model.model.data_load_speech=self.prepared_speech

    def prepared_speech(self,contents,*args,**kwargs):
        import re
        import torch
        match=re.search(r'pipeline-prepared:([^<]+)',str(contents['user']))
        if not match: return self.original_data_load(contents,*args,**kwargs)
        row=self.current[match.group(1)]
        return {key:torch.from_numpy(value) if hasattr(value,'dtype') else value for key,value in row.items()}

    def close_job(self):
        if self.features:
            self.model.model.data_load_speech=self.original_data_load
            self.current={}
            self.features.reset()

    def close(self):
        if self.features:
            self.metrics['frontend_wait_seconds']=self.features.seconds
            self.metrics['frontend_peak_queued_bytes']=self.features.peak_reserved_bytes
            self.features.close();self.features=None
        closer=getattr(self.model.model.llm,'close',None)
        if closer is not None:closer()

    def process(self,item):
        return self.process_batch([item])[0]

    def process_batch(self,items):
        if self.features:self.current={item['id']:self.features.take(item) for item in items}
        try:values=recover_unfinished(items,self._predict,self.metrics)
        finally:self.current={}
        self.stats[:]=[dict(value['generation']) for value in values]
        return values

    def _predict(self,items):
        options=dict(batch_size=len(items),cache={},max_length=self.cfg['max_new_tokens'],llm_dtype='bf16',
                     itn=True,llm_kwargs={'do_sample':False})
        language=self.cfg['language']
        if language!='auto':options['language']={'en':'英文','zh':'中文','English':'英文','Chinese':'中文'}.get(language,language)
        if self.features:
            import torch
            arguments=dict(self.model.kwargs);arguments.update(options)
            with torch.inference_mode():
                values,_=self.model.model.inference(data_in=['pipeline-prepared:'+item['id'] for item in items],
                    key=[item['id'] for item in items],**arguments)
        else:
            values=self._predict_files(items,options)
        if len(values)!=len(items) or len(self.stats)!=len(items):
            raise RuntimeError('Incomplete FunASR output/generation metadata')
        stats=[dict(stat) for stat in self.stats]
        return ([dict(text=value['text'],generation=dict(stat),chunk=item)
                for item,value,stat in zip(items,values,stats)],stats)

    def _predict_files(self,items,options):
        # Installed Fun-ASR-Nano routes arrays through a different chat-template path.
        # Reuse its verified filename interface, materializing just the active chunk.
        import soundfile as sf
        with tempfile.TemporaryDirectory(prefix='audio-pipeline-funasr-') as directory:
            paths=[]
            for index,item in enumerate(items):
                audio,rate=chunk_audio(item)
                path=Path(directory)/f'chunk-{index}.wav'
                sf.write(path,audio,rate,subtype='PCM_16')
                paths.append(str(path))
            values=self.model.generate(input=paths,**options)
        return values
