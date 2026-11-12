"""Reused verified scripts/backend.py loader, with configurable CPU threads.

Model loading, checkpoint checks, precision and decoding behavior are preserved.
The original script is unchanged; this final module adds only the thread argument.
"""
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'scripts/FireRedASR2S'


class FireRedBackend:
    def __init__(self,checkpoint=None,beam_size=3,max_tokens=512,attention='sdpa',use_half=True,threads=4):
        import torch
        sys.path.insert(0,str(CODE))
        from fireredasr2s.fireredasr2 import FireRedAsr2,FireRedAsr2Config
        from fireredasr2s.fireredasr2.data.asr_feat import ASRFeatExtractor
        from fireredasr2s.fireredasr2.models.fireredasr_llm import FireRedAsrLlm
        from fireredasr2s.fireredasr2.tokenizer.llm_tokenizer import LlmTokenizerWrapper
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable')
        torch.set_num_threads(threads)
        checkpoint=Path(checkpoint or ROOT/'checkpoints/FireRedASR2-LLM').resolve()
        for required in ('model.pth.tar','asr_encoder.pth.tar','cmvn.ark','Qwen2-7B-Instruct/config.json'):
            if not (checkpoint/required).is_file():raise ValueError('Missing FireRed checkpoint component: '+required)
        manifest={'repo_id':'FireRedTeam/FireRedASR2-LLM','revision':'user-supplied checkpoint'}
        # Equivalent to the official loader, with explicit inference attention
        # and precision overrides so flash-attn compilation is not required.
        package=torch.load(checkpoint/'model.pth.tar',map_location='cpu',weights_only=False)
        args=package['args']
        original_flash=getattr(args,'use_flash_attn',None)
        args.encoder_path=str(checkpoint/'asr_encoder.pth.tar')
        args.llm_dir=str(checkpoint/'Qwen2-7B-Instruct')
        args.use_flash_attn=False
        args.use_fp16=use_half
        model=FireRedAsrLlm.from_args(args)
        keys=model.load_state_dict(package['model_state_dict'],strict=False)
        model.eval()
        # Qwen2 4.51 dispatches its attention function from this shared config.
        model.llm.config._attn_implementation=attention
        del package
        tokenizer=LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        config=FireRedAsr2Config(use_gpu=True,use_half=use_half,beam_size=beam_size,
                               decode_max_len=max_tokens,decode_min_len=0,repetition_penalty=1.0,
                               llm_length_penalty=0.,temperature=1.)
        self.asr=FireRedAsr2('llm',ASRFeatExtractor(str(checkpoint/'cmvn.ark')),model,tokenizer,None,config)
        self.last_decode_error=None
        original=model.transcribe
        def tracked(*args,**kwargs):
            try:
                return original(*args,**kwargs)
            except Exception:
                self.last_decode_error=traceback.format_exc()
                raise
        model.transcribe=tracked
        self.metadata={'model':manifest['repo_id'],'revision':manifest['revision'],'checkpoint':str(checkpoint),
                       'code_commit':(CODE/'SOURCE_COMMIT').read_text().strip(),
                       'attention':attention,'checkpoint_flash_attention_flag':original_flash,
                       'dtype':str(next(model.parameters()).dtype),'beam_size':beam_size,'max_tokens':max_tokens,
                       'missing_state_keys':len(keys.missing_keys),'unexpected_state_keys':len(keys.unexpected_keys),
                       'cpu_threads':torch.get_num_threads(),
                       'timestamps':'chunk boundaries only; LLM variant does not return word alignments'}

    def transcribe(self,ids,paths):
        self.last_decode_error=None
        results=self.asr.transcribe(ids,paths)
        if self.last_decode_error:
            raise RuntimeError('FireRed decoding failed:\n'+self.last_decode_error)
        if len(results)!=len(ids) or [r['uttid'] for r in results]!=ids:
            raise RuntimeError(f'FireRed returned incomplete/misordered results: expected {ids}, got {results}')
        if any('rtf' not in r for r in results):
            raise RuntimeError('FireRed feature extraction failed; see the traceback above')
        return results
