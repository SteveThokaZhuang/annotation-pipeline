"""Experimental vLLM decoder bridge for the trained FunASR/FireRed decoders.

Keep the native audio encoder, projector, prompt assembly and text embeddings.
Export the loaded decoder (including merged LoRA), never the untrained base LLM.
Embedding prompts support greedy decoding; beam search is explicitly rejected.
"""
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
from .common import ROOT,atomic,digest


class Decoder(torch.nn.Module):
    def __init__(self,embedding,config,generation_config,engine):
        super().__init__()
        self.embedding=embedding;self.config=config;self.generation_config=generation_config
        self.engine=engine

    @property
    def model(self):return self

    def get_input_embeddings(self):return self.embedding

    def close(self):
        from .engines import shutdown_vllm
        shutdown_vllm(self.engine)

    @torch.inference_mode()
    def generate(self,inputs_embeds,attention_mask=None,max_new_tokens=512,**kwargs):
        from vllm import SamplingParams
        if kwargs.get('num_beams',1)!=1:raise ValueError('vLLM embedding prompts do not support beam search; explicitly select beam_size: 1')
        if kwargs.get('do_sample',False):raise ValueError('This ASR decoder bridge supports deterministic greedy decoding')
        if kwargs.get('min_length',0)>0 or kwargs.get('repetition_penalty',1.)!=1.:
            raise ValueError('Unsupported decoding constraints for vLLM embedding prompts')
        eos=kwargs.get('eos_token_id',self.generation_config.eos_token_id)
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        if not eos:raise ValueError('Decoder must define EOS token IDs')
        pad=kwargs.get('pad_token_id',self.config.pad_token_id)
        if pad is None:pad=eos[0]
        prompts=[]
        for index,row in enumerate(inputs_embeds):
            active=row if attention_mask is None else row[attention_mask[index].bool()]
            if not len(active):raise ValueError('Empty embedding prompt')
            prompts.append({'prompt_embeds':active.detach().to(device='cpu').contiguous()})
        outputs=self.engine.generate(prompts,SamplingParams(temperature=0,max_tokens=max_new_tokens,stop_token_ids=eos),use_tqdm=False)
        if len(outputs)!=len(prompts):raise RuntimeError('Incomplete vLLM output')
        rows=[]
        for result in outputs:
            if len(result.outputs)!=1:raise RuntimeError('Expected one vLLM output per prompt')
            output=result.outputs[0];tokens=list(output.token_ids)
            if output.finish_reason=='stop':
                # vLLM may omit the stop token. Native EOS monitors require it.
                if not tokens or tokens[-1] not in eos:tokens.append(eos[0])
            elif output.finish_reason!='length':raise RuntimeError('vLLM generation failed: '+str(output.finish_reason))
            rows.append(torch.tensor(tokens,device=inputs_embeds.device,dtype=torch.long))
        return torch.nn.utils.rnn.pad_sequence(rows,batch_first=True,padding_value=pad)


def replace_decoder(owner,job,tokenizer_path):
    settings=job['settings']
    if settings.get('beam_size',1)!=1:raise ValueError('vLLM requires explicit beam_size: 1 for FireRed embedding prompts')
    options=dict(settings.get('vllm',{}))
    if options.get('tensor_parallel_size',1)!=1:raise ValueError('Each pipeline worker owns one GPU; tensor_parallel_size must be 1')
    options.setdefault('gpu_memory_utilization',.6)
    options.setdefault('enable_prefix_caching',False)
    if options.get('enable_prompt_embeds',True) is not True:raise ValueError('enable_prompt_embeds must be true')
    options['enable_prompt_embeds']=True
    old=owner.llm
    config=copy.deepcopy(old.config);generation=copy.deepcopy(old.generation_config)
    # Move the full decoder to host RAM before exporting or creating vLLM.
    if hasattr(old,'merge_and_unload'):old=old.merge_and_unload(safe_merge=True)
    old.cpu()
    embedding=copy.deepcopy(old.get_input_embeddings())
    embedding.requires_grad_(False)
    parent=Path(settings.get('decoder_export_root',ROOT/'outputs/model_exports'))
    if not parent.is_absolute():parent=ROOT/parent
    parent.mkdir(parents=True,exist_ok=True)
    provenance=dict(checkpoint=job['checkpoint'],version=job.get('checkpoint_version'),
                    converter='native-trained-decoder-peft-merge-v1',dtype=str(old.get_input_embeddings().weight.dtype),model=job.get('model'))
    # A standalone adapter caller must still invalidate exports after model edits.
    if provenance['version'] is None:
        from .common import version_tree
        provenance['version']=version_tree(job['checkpoint'])
    identity=digest(provenance);target=parent/identity
    with (parent/(identity+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (target/'export.json').is_file():
            temporary=Path(tempfile.mkdtemp(prefix='.decoder-',dir=parent))
            try:
                old.save_pretrained(temporary,safe_serialization=True)
                for p in Path(tokenizer_path).iterdir():
                    if p.is_file() and (p.name.startswith(('tokenizer','vocab.','merges.','chat_template','added_tokens','special_tokens_map'))):
                        shutil.copy2(p,temporary/p.name)
                atomic(temporary/'export.json',provenance)
                os.replace(temporary,target)
            finally:
                if temporary.exists():shutil.rmtree(temporary)
    # Drop the model before vLLM allocates weights and KV cache. Keep only its
    # trained embedding table alongside the native encoder/projector.
    owner.llm=torch.nn.Identity()
    del old
    gc.collect();torch.cuda.empty_cache()
    from vllm import LLM
    engine=LLM(model=str(target),**options)
    device=next(owner.parameters()).device
    owner.llm=Decoder(embedding.to(device),config,generation,engine)
    return target
