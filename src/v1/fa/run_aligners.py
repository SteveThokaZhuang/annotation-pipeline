"""Resumable Qwen / Nyra inference. Inputs are transcript + audio only."""
import argparse
import json
import os
import time
from pathlib import Path
from common import ROOT, DATASET, load_manifest, read_jsonl

def main():
    process_start=time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['qwen','nyra','asr'],required=True)
    p.add_argument('--manifest',type=Path,default=DATASET/'manifest.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int)
    p.add_argument('--retry-errors',action='store_true')
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    if os.environ.get('FA_REQUIRE_SINGLE_GPU')=='1':
        if torch.cuda.device_count()!=1:
            raise RuntimeError('Single-GPU benchmark requires exactly one visible GPU')
        torch.cuda.set_device(a.device)
    gpu_info={'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'visible_gpu_count':torch.cuda.device_count(),
              'gpu_name':torch.cuda.get_device_name(a.device) if a.device.startswith('cuda') else None}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    existing=read_jsonl(a.output) if a.output.exists() else []
    if a.retry_errors:
        from common import write_jsonl
        existing=[r for r in existing if not r.get('error')]
        write_jsonl(a.output,existing)
    done={r['id'] for r in existing}
    rows=load_manifest(a.manifest)
    if a.limit: rows=rows[:a.limit]
    start=time.perf_counter()
    if a.backend=='qwen':
        from qwen_asr import Qwen3ForcedAligner
        model=Qwen3ForcedAligner.from_pretrained(str(a.model or ROOT/'models/qwen3-forced-aligner/checkpoints'),dtype=torch.bfloat16,device_map=a.device,attn_implementation='sdpa',local_files_only=True)
    elif a.backend=='nyra':
        from nyra_align import Aligner
        model=Aligner(model=a.model or ROOT/'models/nyra-forced-aligner/checkpoints',device=a.device,precision='fp32')
        # WavLM is lazy in Nyra. Account for its load in model startup, not inference.
        _=model.wavlm
    else:
        from qwen_asr import Qwen3ASRModel
        model=Qwen3ASRModel.from_pretrained(str(a.model),dtype=torch.bfloat16,device_map=a.device,max_inference_batch_size=1,max_new_tokens=512)
    if a.device.startswith('cuda'): torch.cuda.synchronize()
    model_load_seconds=time.perf_counter()-start
    processing_start=time.perf_counter()
    processed=0
    with a.output.open('a') as out:
        for i,r in enumerate(rows):
            if r['id'] in done: continue
            if a.device.startswith('cuda'): torch.cuda.synchronize()
            t=time.perf_counter()
            try:
                if a.backend=='qwen':
                    res=model.align(audio=r['audio'],text=r['text'],language='English')[0]
                    result={'words':[dict(text=w.text,start=float(w.start_time),end=float(w.end_time)) for w in res]}
                elif a.backend=='nyra':
                    # Text-only convention mapping. Do not use reference truncation flags.
                    text=' '.join({'um':'[UM]','uh':'[UH]'}.get(t.lower(),t) for t in r['text'].split())
                    res=model.align(r['audio'],text)
                    result={'words':[dict(text={'[UM]':'um','[UH]':'uh'}.get(w.word,w.word),start=w.start,end=w.end,type=w.type) for w in res.words], 'skipped':res.skipped}
                else:
                    res=model.transcribe(audio=r['audio'],language='English')[0]
                    result={'text':res.text}
            except Exception as e:
                import traceback
                traceback.print_exc()
                result={'words':[],'error':repr(e)}
            if torch.cuda.is_available(): torch.cuda.synchronize()
            result.update(id=r['id'],seconds=time.perf_counter()-t,backend=a.backend)
            out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush()
            print(f'{a.backend} {i+1}/{len(rows)} {r["id"]} {result["seconds"]:.3f}s {result.get("error", "ok")}',flush=True)
            processed+=1
    if a.device.startswith('cuda'): torch.cuda.synchronize()
    processing_seconds=time.perf_counter()-processing_start
    (a.output.parent/(a.output.stem+'.runtime.json')).write_text(json.dumps({
        'backend':a.backend,'device':a.device,'seconds_with_load':time.perf_counter()-start,
        'worker_seconds':time.perf_counter()-process_start,'model_load_seconds':model_load_seconds,
        'processing_seconds':processing_seconds,'processed_samples':processed,'resumed_samples':len(done),
        'processing_scope':'audio reading, preprocessing, alignment, CUDA synchronization, JSONL writing; includes first inference; no warm-up',
        'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,
        'torch':torch.__version__,'cuda_runtime':torch.version.cuda,**gpu_info},indent=2))
if __name__=='__main__': main()
