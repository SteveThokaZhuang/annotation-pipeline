from audio_pipeline.audio import chunk_audio

class Adapter:
    def __init__(self,job):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.cfg=job['settings']
        self.engine=self.cfg.get('engine','transformers')
        self.stats=[]
        if self.engine=='vllm':
            options=dict(self.cfg.get('vllm',{}))
            if options.get('tensor_parallel_size',1)!=1:
                raise ValueError('Pipeline workers own one GPU each; tensor_parallel_size must be 1')
            options.setdefault('gpu_memory_utilization',.8)
            options.setdefault('enable_prefix_caching',False)
            options.setdefault('mm_processor_cache_gb',0)
            self.model=Qwen3ASRModel.LLM(model=job['checkpoint'],
                max_inference_batch_size=self.cfg['batch_size'],max_new_tokens=self.cfg['max_new_tokens'],**options)
            self._native_generate=self.model.model.generate
            self.model.model.generate=self.generate_vllm
        elif self.engine=='transformers':
            self.model=Qwen3ASRModel.from_pretrained(job['checkpoint'],dtype=torch.bfloat16,
                device_map='cuda:0',attn_implementation='sdpa',max_inference_batch_size=self.cfg['batch_size'],
                max_new_tokens=self.cfg['max_new_tokens'])
            self._native_generate=self.model.model.generate
            self.model.model.generate=self.generate
        else:raise ValueError('Unsupported Qwen engine: '+self.engine)

    def close(self):
        if self.engine=='vllm':
            from audio_pipeline.engines import shutdown_vllm
            shutdown_vllm(self.model.model)

    def generate_vllm(self,*args,**kwargs):
        outputs=self._native_generate(*args,**kwargs)
        for result in outputs:
            if len(result.outputs)!=1:raise RuntimeError('Expected one vLLM hypothesis per audio')
            value=result.outputs[0]
            if value.finish_reason not in ('stop','length'):
                raise RuntimeError('vLLM did not finish successfully: '+str(value.finish_reason))
            self.stats.append(dict(generated_tokens=len(value.token_ids),
                                   finish_reason='eos' if value.finish_reason=='stop' else 'length'))
        return outputs

    def generate(self,*args,**kwargs):
        # Qwen returns the decoder cache in addition to sequences. No later
        # pipeline operation needs that cache, including token-limit recovery.
        self.stats.clear()
        output=self._native_generate(*args,**kwargs)
        if getattr(output,'past_key_values',None) is not None:
            output.past_key_values=None
        sequences=output.sequences
        prompt=kwargs.get('input_ids',args[0] if args else None)
        offset=prompt.shape[1] if prompt is not None else 0
        eos=kwargs.get('eos_token_id',self.model.model.generation_config.eos_token_id)
        eos={eos} if isinstance(eos,int) else set(eos or [])
        if not eos:raise RuntimeError('Cannot verify completion: model has no EOS token ID')
        for row in sequences[:,offset:].tolist():
            end=next((i for i,token in enumerate(row) if token in eos),None)
            self.stats.append(dict(generated_tokens=len(row) if end is None else end+1,
                                   finish_reason='length' if end is None else 'eos'))
        return output

    def process(self,item):
        return self.process_batch([item])[0]

    def process_batch(self,items):
        self.stats.clear()
        audios=[chunk_audio(item) for item in items]
        language={'en':'English','zh':'Chinese','auto':None}.get(self.cfg['language'],self.cfg['language'])
        values=self.model.transcribe(audio=audios,language=language,return_time_stamps=False)
        if len(values)!=len(items) or len(self.stats)!=len(items):
            raise RuntimeError('Qwen returned incomplete predictions or generation metadata')
        results=[]
        for item,value,stat in zip(items,values,self.stats):
            if stat['finish_reason']=='eos':
                results.append(dict(text=value.text,language=value.language,chunk=item,generation=[dict(stat)]))
            else:
                # Never publish truncated text. The worker rejects only this
                # hypothesis; completed peers are saved without another decode.
                results.append(dict(text='',language=None,chunk=item,generation=[dict(stat)],
                    transcription_error='token_limit_without_eos',fallback_requested=True))
        return results
