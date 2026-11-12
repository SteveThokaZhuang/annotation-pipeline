"""Bounded pinned staging buffers; buffers are reused only after their CUDA event."""
import json
import os
import threading
import time
_POOL=[]
_LOCK=threading.Lock()
_STATS=dict(host_to_device_bytes=0,pinned_bytes_peak=0,pinned_reuses=0,blocking_fallbacks=0,transfer_submit_seconds=0.)


def to_device(value,device,dtype=None):
    import torch
    device=torch.device(device)
    dtype=dtype or value.dtype
    cfg=json.loads(os.environ.get('AUDIO_PIPELINE_TRANSFERS','{}'))
    tick=time.perf_counter()
    if value.device.type!='cpu' or device.type!='cuda' or not cfg.get('pinned',False):
        return value.to(device=device,dtype=dtype)
    size=value.numel()*value.element_size();limit=int(cfg.get('max_pinned_gib_per_worker',0)*2**30)
    _STATS['host_to_device_bytes']+=size
    if size>limit:
        _STATS['blocking_fallbacks']+=1
        return value.to(device=device,dtype=dtype)
    with _LOCK:
        chosen=None
        for entry in _POOL:
            if entry['event'].query() and tuple(entry['buffer'].shape)==tuple(value.shape) and entry['buffer'].dtype==value.dtype:
                chosen=entry;_STATS['pinned_reuses']+=1;break
        if chosen is None:
            used=sum(e['buffer'].numel()*e['buffer'].element_size() for e in _POOL)
            for entry in list(_POOL):
                if used+size<=limit:break
                if entry['event'].query():
                    _POOL[:]=[e for e in _POOL if e is not entry];used-=entry['buffer'].numel()*entry['buffer'].element_size()
            if used+size>limit:
                _STATS['blocking_fallbacks']+=1
                return value.to(device=device,dtype=dtype)
            chosen={'buffer':torch.empty_like(value,device='cpu',pin_memory=True),'event':torch.cuda.Event()}
            _POOL.append(chosen)
        chosen['buffer'].copy_(value)
        output=chosen['buffer'].to(device=device,dtype=dtype,non_blocking=cfg.get('non_blocking',True))
        chosen['event'].record(torch.cuda.current_stream(device))
        _STATS['pinned_bytes_peak']=max(_STATS['pinned_bytes_peak'],sum(e['buffer'].numel()*e['buffer'].element_size() for e in _POOL))
    _STATS['transfer_submit_seconds']+=time.perf_counter()-tick
    return output


def metrics(reset=False):
    result=dict(_STATS)
    if reset:
        for k in _STATS:_STATS[k]=0
    return result
