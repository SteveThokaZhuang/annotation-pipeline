"""Bounded native-rate recording cache; resampling still happens after slicing.

Read-only arrays are shared by preparation threads within a process. File identity
uses the original source metadata, never the staging cache's changing LRU mtime.
"""
from collections import OrderedDict
import os
from pathlib import Path
import threading


class RecordingCache:
    def __init__(self,max_bytes):
        self.limit=max(0,int(max_bytes));self.used=0;self.reserved=0
        self.entries=OrderedDict();self.loading=set();self.condition=threading.Condition()
        self.stats=dict(hits=0,misses=0,loads=0,bypasses=0,evictions=0,peak_bytes=0)

    def get(self,key,loader):
        with self.condition:
            while key in self.loading:self.condition.wait()
            if key in self.entries:
                self.stats['hits']+=1;self.entries.move_to_end(key);return self.entries[key]
            self.loading.add(key);self.stats['misses']+=1
        reservation=0
        def reserve(size):
            nonlocal reservation
            with self.condition:
                if size>self.limit or size<=0:return False
                while self.entries and self.used+self.reserved+size>self.limit:
                    _,old=self.entries.popitem(last=False);self.used-=old[0].nbytes;self.stats['evictions']+=1
                if self.used+self.reserved+size>self.limit:return False
                self.reserved+=size;reservation=size
                self.stats['peak_bytes']=max(self.stats['peak_bytes'],self.used+self.reserved)
                return True
        try:
            value=loader(reserve)
            with self.condition:
                if value is None:self.stats['bypasses']+=1
                else:
                    if value[0].nbytes>reservation:raise ValueError('Decoded cache loader exceeded its reservation')
                    value[0].flags.writeable=False
                    self.entries[key]=value;self.used+=value[0].nbytes;self.stats['loads']+=1
            return value
        finally:
            with self.condition:
                self.reserved-=reservation;self.loading.discard(key);self.condition.notify_all()


_CACHE=None
_CACHE_LOCK=threading.Lock()

def get_recording(path):
    global _CACHE
    gib=float(os.environ.get('AUDIO_PIPELINE_DECODED_RAM_GIB','0'))
    if gib<=0:return None
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE.limit!=int(gib*2**30):_CACHE=RecordingCache(gib*2**30)
    source=Path(path);before=source.stat();key=(str(source.resolve()),before.st_size,before.st_mtime_ns)
    def load(reserve):
        import numpy as np
        import soundfile as sf
        from .audio import audio_source,io_slot
        with audio_source(source) as local,io_slot(),sf.SoundFile(local) as stream:
            if not reserve(len(stream)*stream.channels*4):return None
            rate=stream.samplerate;frames=len(stream)
            audio=stream.read(dtype='float32',always_2d=True)
        after=source.stat()
        if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Source changed during decoded-cache preparation')
        if len(audio)!=frames or audio.size==0:raise ValueError('Incomplete or invalid decoded recording')
        return audio,rate
    return _CACHE.get(key,load)


def metrics():
    if _CACHE is None:return dict(enabled=False)
    with _CACHE.condition:return dict(enabled=True,limit_bytes=_CACHE.limit,retained_bytes=_CACHE.used,**_CACHE.stats)
