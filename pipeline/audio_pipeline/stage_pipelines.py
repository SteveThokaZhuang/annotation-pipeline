"""Stage-specific scheduling; model code and checkpoint identities stay intact.

One producer owns mutable preparation/encoder state. The consumer owns decoding
or CPU postprocessing. No two threads call the same GPU forward concurrently.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import math
import threading
import time

from .funasr_pipeline import FunASRPipeline,Prepared,_Captured


class StagePipeline(FunASRPipeline):
    """Reuse the bounded task iterator without the FunASR-specific decoder hook."""
    def __init__(self,adapter,job,settings,stop_requested=lambda:False):
        self.base=adapter;self.job=job;self.stop_requested=stop_requested
        self.depth=settings.get('lookahead_batches',2);gib=settings.get('max_gib',4)
        if not isinstance(self.depth,int) or isinstance(self.depth,bool) or not 1<=self.depth<=4:
            raise ValueError('Stage pipeline lookahead_batches must be an integer in [1,4]')
        if not isinstance(gib,(int,float)) or isinstance(gib,bool) or not math.isfinite(gib) or gib<=0:
            raise ValueError('Stage pipeline max_gib must be positive')
        self.budget=int(gib*2**30);self.closing=threading.Event();self.current=None;self.futures=set()
        if not hasattr(adapter,'metrics'):adapter.metrics={}
        self.metrics=adapter.metrics
        self.metrics.update(pipeline_prepare_seconds=0.,pipeline_decode_seconds=0.,pipeline_wait_seconds=0.,
                            pipeline_peak_reserved_bytes=0,pipeline_batches=0)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix=job['stage']+'-prepare')
        self.audio_prefetch=None

    def start(self,items):
        if hasattr(self.base,'start'):self.base.start(items)

    def start_audio(self,items):
        from .prefetch import AudioPrefetch
        from .audio import load_chunk,estimate_chunk_bytes
        alloc=self.job['allocation'];batch=self.job['settings'].get('batch_size',alloc.get('inference_batch_size',1))
        self.audio_prefetch=AudioPrefetch(load_chunk,workers=alloc.get('prepare_workers',2),
            lookahead=alloc.get('prepare_lookahead') or max(1,batch*alloc.get('prefetch_batches',2)),
            max_bytes=int(alloc.get('prepare_max_gib',1)*2**30),estimate=estimate_chunk_bytes,
            stop_file=self.job.get('stop_file'))
        self.audio_prefetch.start([dict(item=item) for item in items])

    def audio(self,item,features):
        saved=(features or {}).get('audios',{})
        if item['id'] in saved:return saved[item['id']]
        return self.audio_prefetch.take(item)

    def _estimate(self,group):
        return sum(max(1.,float(t['item'].get('duration',t['item'].get('chunk',{}).get('duration',0))))*
                   max(1,t['item'].get('channels',1))*16000*4*6 for t in group)

    def _prepare(self,items,features=None):
        started=time.perf_counter()
        try:
            if self._stopped():raise InterruptedError('Stopped before stage preparation')
            return self.prepare_batch(items,features)
        except Exception as error:return Prepared(items,features=features or {},error=error.with_traceback(None))
        finally:self.metrics['pipeline_prepare_seconds']+=time.perf_counter()-started

    def process_batch(self,items):
        current=self.current
        if current is None:raise RuntimeError('Stage pipeline requires its prepared task iterator')
        if not current.consumed and [i['id'] for i in items]==[i['id'] for i in current.items]:
            current.consumed=True;prepared=current
        else:prepared=self.pool.submit(self._prepare,items,current.features).result()
        started=time.perf_counter()
        try:
            if prepared.error is not None:raise prepared.error
            values=self.consume(prepared)
            self.metrics['pipeline_batches']+=1
            return values
        finally:
            prepared.arguments={}
            self.metrics['pipeline_decode_seconds']+=time.perf_counter()-started

    def restore(self):pass

    def close(self):
        self.closing.set()
        for future in self.futures:future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.futures.clear();self.current=None
        if self.audio_prefetch:self.audio_prefetch.close();self.audio_prefetch=None
        self.restore()
        if hasattr(self.base,'close'):self.base.close()


class FireRedPipeline(FunASRPipeline):
    def decoder_for(self,adapter):return adapter.model.asr.model.llm
    def tokenizer_for(self,adapter):return copy.deepcopy(adapter.model.asr.tokenizer)

    def _decode(self,prepared):
        if prepared.error is not None:raise prepared.error
        started=time.perf_counter()
        try:
            output=self.generate(**prepared.arguments)
            stats=[dict(stat) for stat in self.base.stats]
            texts=self.tokenizer.batch_decode(output,skip_special_tokens=True)
            if len(texts)!=len(prepared.items) or len(stats)!=len(prepared.items):
                raise RuntimeError('Incomplete pipelined FireRed output/generation metadata')
            return ([dict(text=text.lower(),chunk=item,metadata=self.base.model.metadata,generation=[dict(stat)])
                     for text,item,stat in zip(texts,prepared.items,stats)],stats)
        finally:
            prepared.arguments={}
            self.metrics['pipeline_decode_seconds']+=time.perf_counter()-started


class QwenPipeline(StagePipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.native=self.base.model
        self.library=inspect.getmodule(type(self.native))
        self.infer=self.native._infer_asr
        self.generate=self.native.model.generate
        self.native._infer_asr=self.capture_requests

    def start(self,items):super().start(items);self.start_audio(items)

    def capture_requests(self,contexts,wavs,languages):
        requests=[dict(prompt=self.native._build_text_prompt(context=context,force_language=language),
                       multi_modal_data={'audio':[wav]}) for context,wav,language in zip(contexts,wavs,languages)]
        raise _Captured(dict(requests=requests,languages=languages))

    def prepare_batch(self,items,features):
        audios={item['id']:self.audio(item,features) for item in items}
        language={'en':'English','zh':'Chinese','auto':None}.get(self.base.cfg['language'],self.base.cfg['language'])
        try:self.native.transcribe(audio=list(audios.values()),language=language,return_time_stamps=False)
        except _Captured as captured:
            # The existing adapter requires one generation per VAD chunk too.
            if len(captured.arguments['requests'])!=len(items):
                raise RuntimeError('Qwen returned incomplete predictions or generation metadata')
            return Prepared(items,arguments=captured.arguments,features={'audios':audios})
        raise RuntimeError('Qwen did not produce an inference request')

    def consume(self,prepared):
        self.base.stats.clear();outputs=[]
        requests=prepared.arguments['requests'];limit=self.native.max_inference_batch_size
        if limit is None or limit<1:limit=len(requests)
        for start in range(0,len(requests),limit):
            outputs.extend(self.generate(requests[start:start+limit],sampling_params=self.native.sampling_params,use_tqdm=False))
        if len(outputs)!=len(prepared.items) or len(self.base.stats)!=len(prepared.items):
            raise RuntimeError('Qwen returned incomplete predictions or generation metadata')
        results=[]
        for item,output,language,stat in zip(prepared.items,outputs,prepared.arguments['languages'],self.base.stats):
            lang,text=self.library.parse_asr_output(output.outputs[0].text,user_language=language)
            lang=self.library.merge_languages([lang])
            if stat['finish_reason']=='eos':
                results.append(dict(text=text,language=lang,chunk=item,generation=[dict(stat)]))
            else:
                results.append(dict(text='',language=None,chunk=item,generation=[dict(stat)],
                                    transcription_error='token_limit_without_eos',fallback_requested=True))
        return results

    def restore(self):self.native._infer_asr=self.infer


class AlignmentPipeline(StagePipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.native=self.base.model
        self.library=inspect.getmodule(type(self.native))
        # Linguistic preparation and timestamp parsing have separate owners.
        self.text_processor=copy.deepcopy(self.native.aligner_processor)

    def start(self,items):super().start(items);self.start_audio(items)

    def prepare_batch(self,items,features):
        active=[];audios={};words=[];texts=[]
        for index,item in enumerate(items):
            if not item['text'].strip():continue
            active.append(index);audios[item['id']]=self.audio(item,features)
            word_list,text=self.text_processor.encode_timestamp(item['text'],self.base.language(item))
            words.append(word_list);texts.append(text)
        inputs=None
        if active:
            normalized=self.library.normalize_audios(list(audios.values()))
            inputs=self.native.processor(text=texts,audio=normalized,return_tensors='pt',padding=True)
        return Prepared(items,arguments=dict(active=active,inputs=inputs,words=words),features={'audios':audios})

    def consume(self,prepared):
        import torch
        results=[dict(text=item['text'],chunk=item['chunk'],words=[],empty_transcript=True)
                 if not item['text'].strip() else None for item in prepared.items]
        active=prepared.arguments['active']
        if active:
            with torch.inference_mode():
                from .transfers import to_device
                inputs={k:to_device(v,self.native.model.device,self.native.model.dtype if v.is_floating_point() else v.dtype) for k,v in prepared.arguments['inputs'].items()}
                output_ids=self.native.model.thinker(**inputs).logits.argmax(dim=-1)
                if len(output_ids)!=len(active):raise RuntimeError('Incomplete alignment batch')
                for index,input_id,output_id,words in zip(active,inputs['input_ids'],output_ids,prepared.arguments['words']):
                    masked=output_id[input_id==self.native.timestamp_token_id]
                    timestamps=(masked*self.native.timestamp_segment_time).to('cpu').numpy()
                    parsed=self.native.aligner_processor.parse_timestamp(words,timestamps)
                    for row in parsed:
                        row['start_time']=round(row['start_time']/1000.,3)
                        row['end_time']=round(row['end_time']/1000.,3)
                    value=self.native._to_structured_items(parsed)
                    item=prepared.items[index]
                    results[index]=self.base.format(item,value,self.base.language(item))
        return results


class VADPipeline(StagePipeline):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.post=copy.copy(self.base)
        self.original_process=self.base._process
        self.base._process=self.capture_postprocess
        self.post_lock=threading.Lock()

    def capture_postprocess(self,item,prepared):
        raise _Captured(dict(probabilities=dict(self.base._batched_probs)))

    def prepare_batch(self,items,features):
        prior=(features or {}).get('prepared',{})
        self.base._ready_prepared.update({i['id']:prior[i['id']] for i in items if i['id'] in prior})
        if not self.base.preparation or (len(items)<2 and self.job['allocation'].get('prepare_order')!='ready'):
            # Native singleton/OOM recovery may include the postprocessor. Its
            # lock prevents it sharing mutable state with consumer postprocessing.
            with self.post_lock:
                self.base._process=self.original_process
                try:values=[self.base.process(item) for item in items]
                finally:self.base._process=self.capture_postprocess
            return Prepared(items,arguments={'direct':values})
        try:self.base.process_batch(items)
        except _Captured as captured:
            records={item['id']:self.base._ready_prepared.pop(item['id']) for item in items}
            captured.arguments['records']=records
            return Prepared(items,arguments=captured.arguments,features={'prepared':records})
        raise RuntimeError('VAD did not produce a postprocessing request')

    def consume(self,prepared):
        if 'direct' in prepared.arguments:return prepared.arguments['direct']
        with self.post_lock:
            self.post._batched_probs=prepared.arguments['probabilities']
            try:return [self.post._process(item,prepared.arguments['records'][item['id']]) for item in prepared.items]
            finally:self.post._batched_probs={}

    def restore(self):self.base._process=self.original_process


def settings_for(job):
    return job.get('allocation',{}).get('optimization',{}).get('stage_pipelines',{}).get(job.get('model'),{})


def enabled(job):
    if not settings_for(job).get('enabled',False):return False
    model=job.get('model')
    if model in ('qwen3_asr','fireredasr'):return job['settings'].get('engine')=='vllm'
    return model in ('dnsmos','vad','align')


def owns_audio_prefetch(job):
    return enabled(job) and job.get('model') in ('qwen3_asr','align')


def wrap(adapter,job,stop_requested):
    settings=settings_for(job)
    classes={'fireredasr':FireRedPipeline,'qwen3_asr':QwenPipeline,'align':AlignmentPipeline,'vad':VADPipeline}
    if job['model']=='dnsmos' and settings.get('cross_recording_batching',False):
        from .dnsmos_cross import CrossRecordingDNSMOS
        return CrossRecordingDNSMOS(adapter,job,settings,stop_requested)
    if job['model']=='dnsmos':
        from .dnsmos_pipeline import DNSMOSPipeline
        return DNSMOSPipeline(adapter,job,settings,stop_requested)
    return classes[job['model']](adapter,job,settings,stop_requested=stop_requested)
