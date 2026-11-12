"""Pass large frontend arrays through local memory mappings instead of pickling."""
import json
import os
from pathlib import Path
import tempfile
import uuid


def pack(value,root):
    import numpy as np
    if isinstance(value,np.ndarray):
        path=Path(root)/(uuid.uuid4().hex+'.npy');np.save(path,value,allow_pickle=False)
        return {'__audio_pipeline_array__':str(path)}
    if isinstance(value,dict):return {k:pack(v,root) for k,v in value.items()}
    if isinstance(value,list):return [pack(v,root) for v in value]
    if isinstance(value,tuple):return {'__audio_pipeline_tuple__':[pack(v,root) for v in value]}
    return value


def unpack(value):
    import numpy as np
    if isinstance(value,dict) and set(value)=={'__audio_pipeline_array__'}:
        path=Path(value['__audio_pipeline_array__'])
        # COW mappings support torch.from_numpy without granting writes to the
        # shared file. Linux keeps the inode alive until all views are released.
        result=np.load(path,mmap_mode='c',allow_pickle=False);path.unlink();return result
    if isinstance(value,dict) and set(value)=={'__audio_pipeline_tuple__'}:return tuple(unpack(v) for v in value['__audio_pipeline_tuple__'])
    if isinstance(value,dict):return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):return [unpack(v) for v in value]
    return value
