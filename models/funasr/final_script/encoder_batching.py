"""Bounded encoder microbatches; retain FunASR's frontend and decoder paths."""
from types import MethodType


def frame_groups(lengths, batch_size, max_frames):
    if batch_size < 1 or max_frames < 1:
        raise ValueError('Encoder batch size and frame budget must be positive')
    group=[]
    for index in sorted(range(len(lengths)),key=lengths.__getitem__):
        if group and (len(group)>=batch_size or (len(group)+1)*lengths[index]>max_frames):
            yield group
            group=[]
        group.append(index)
    if group:yield group


def install(model, batch_size, max_frames, metrics):
    import torch
    from audio_pipeline.batching import is_cuda_oom
    # Validate at startup, including when the first request contains one item.
    list(frame_groups([],batch_size,max_frames))
    original=model._inference_llm_batch

    def batched(self,data_in,data_lengths,key,tokenizer,frontend,**kwargs):
        # Preserve the native path for unsupported training/multiturn requests.
        if kwargs.get('teacherforcing',False) or any(
                sum(turn.get('role')=='user' for turn in row)!=1 for row in data_in):
            return original(data_in,data_lengths,key,tokenizer,frontend,**kwargs)
        prepared=[]
        for row in data_in:
            contents=self.data_template(row)
            prepared.append(self.data_load_speech(contents,tokenizer,frontend,meta_data={},**kwargs))
        if any(not isinstance(row['speech'],torch.Tensor) or row['speech'].shape[0]!=1
               or row['fbank_beg'].numel()!=1 for row in prepared):
            raise ValueError('Batched encoder requires one audio segment per request')
        lengths=[int(row['speech_lengths'].flatten()[0]) for row in prepared]
        embeddings=[None]*len(prepared)
        device=kwargs['device']

        def encode_group(indices):
            failed=False
            try:
                speech=torch.nn.utils.rnn.pad_sequence(
                    [prepared[i]['speech'][0,:lengths[i]] for i in indices],batch_first=True).to(device)
                sizes=torch.tensor([lengths[i] for i in indices],device=device,dtype=prepared[indices[0]]['speech_lengths'].dtype)
                encoded,encoded_lengths=self.encode(speech,sizes)
                adapted,adapted_lengths=self.audio_adaptor(encoded,encoded_lengths)
                for position,index in enumerate(indices):
                    row=prepared[index]
                    ids=row['source_ids'].to(device).clamp_min(0)
                    embedding=self.llm.model.get_input_embeddings()(ids)
                    start=max(0,int(row['fbank_beg'].flatten()[0]))
                    count=max(0,int(row['fake_token_len'].flatten()[0]))
                    if start>0:
                        if count>int(adapted_lengths[position]) or start+count>embedding.shape[1]:
                            raise ValueError('Audio embedding length disagrees with native prompt')
                        embedding[0,start:start+count]=adapted[position,:count].to(embedding.dtype)
                    embeddings[index]=embedding
                metrics['encoder_calls']=metrics.get('encoder_calls',0)+1
                metrics['encoder_items']=metrics.get('encoder_items',0)+len(indices)
                metrics['encoder_max_batch']=max(metrics.get('encoder_max_batch',0),len(indices))
            except Exception as error:
                if not is_cuda_oom(error) or len(indices)==1:raise
                failed=True
            if failed:
                # Release the failed call's tensors and traceback before retry.
                speech=sizes=encoded=encoded_lengths=adapted=adapted_lengths=None
                torch.cuda.empty_cache()
                metrics['encoder_oom_splits']=metrics.get('encoder_oom_splits',0)+1
                midpoint=len(indices)//2
                encode_group(indices[:midpoint])
                encode_group(indices[midpoint:])

        for indices in frame_groups(lengths,batch_size,max_frames):encode_group(indices)
        prepare=self.inference_prepare
        queued=iter(embeddings)
        def prepared_embedding(*args,**kw):
            return next(queued),None,None,None,None
        self.inference_prepare=prepared_embedding
        try:
            return original(data_in,data_lengths,key,tokenizer,frontend,**kwargs)
        finally:
            self.inference_prepare=prepare

    model._inference_llm_batch=MethodType(batched,model)
