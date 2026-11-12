import json
import math
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time

@contextmanager
def io_slot():
    directory=os.environ.get('AUDIO_PIPELINE_IO_LOCKS')
    if not directory:
        yield
        return
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    handle=None
    while handle is None:
        for slot in range(int(os.environ.get('AUDIO_PIPELINE_IO_WORKERS','1'))):
            candidate=open(root/f'{slot}.lock','a+')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate.close();continue
            handle=candidate;break
        if handle is None:time.sleep(.002)
    try:yield
    finally:handle.close()



@contextmanager
def audio_source(path):
    """Lease a local recording for any audio-consuming stage."""
    from .local_cache import cached_source
    config=json.loads(os.environ.get("AUDIO_PIPELINE_CACHE","{}"))
    if config.get("enabled"):
        with cached_source(path,config["root"],int(config.get("max_gib",20)*2**30)) as local:
            yield local
    else:
        yield path


def read_audio(path,channel=0,rate=None,start_sample=0,end_sample=None):
    from .ram_audio import get_recording
    native=get_recording(path)
    if native is not None:return _read_audio(path,channel,rate,start_sample,end_sample,native=native)
    with audio_source(path) as local:
        return _read_audio(local,channel,rate,start_sample,end_sample)


def _read_audio(path,channel=0,rate=None,start_sample=0,end_sample=None,native=None):
    import numpy as np
    import soundfile as sf
    if native is not None:
        recording,source_rate=native
        end=len(recording) if end_sample is None else end_sample
        if start_sample<0 or end<=start_sample or end>len(recording):raise ValueError('Audio interval is empty or outside the recording')
        audio=recording[start_sample:end].copy()
    else:
        with io_slot(), sf.SoundFile(path) as stream:
            source_rate=stream.samplerate
            end=len(stream) if end_sample is None else end_sample
            if start_sample<0 or end<=start_sample or end>len(stream):
                raise ValueError('Audio interval is empty or outside the recording')
            stream.seek(start_sample)
            audio=stream.read(end-start_sample,dtype='float32',always_2d=True)
    if len(audio)!=end-start_sample:
        raise ValueError('Incomplete audio decode: fewer samples than requested')
    if audio.size==0 or not np.isfinite(audio).all():
        raise ValueError('Empty or nonfinite audio')
    if channel>=audio.shape[1]:raise ValueError('Invalid channel')
    audio=audio[:,channel]
    if rate and rate!=source_rate:
        from scipy.signal import resample_poly
        factor=math.gcd(rate,source_rate)
        audio=resample_poly(audio,rate//factor,source_rate//factor).astype(np.float32)
    return audio,rate or source_rate


def load_chunk(item):
    from .common import fingerprint
    if item.get('audio_fingerprint') and fingerprint(item['audio_path'])!=item['audio_fingerprint']:
        raise RuntimeError('Enhanced audio changed; rerun enhance and ASR stages')
    audio,rate=read_audio(item['audio_path'],channel=item['channel'],rate=16000,
                      start_sample=item.get('audio_start_sample',item['start_sample']),
                      end_sample=item.get('audio_end_sample',item['end_sample']))
    import numpy as np
    # Match the PCM16 filename-only interfaces used by FunASR and FireRed exactly.
    audio=np.rint(np.clip(audio,-1,32767/32768)*32768).astype(np.int16).astype(np.float32)/32768
    if len(audio)<3200:audio=np.pad(audio,(0,3200-len(audio)))
    return audio,rate


_PREPARED={}

def _key(item):
    return (item['id'],item.get('audio_path'),item.get('channel'),item.get('start_sample'),item.get('end_sample'))

def set_prepared(values):
    global _PREPARED
    _PREPARED=values

def add_prepared(item,value):_PREPARED[_key(item)]=value

def chunk_audio(item):
    value=_PREPARED.get(_key(item))
    return value if value is not None else load_chunk(item)


def estimate_chunk_bytes(item):
    # Include input channel matrix, resampling buffers and quantization copies.
    duration=item.get('duration',max(0,item.get('end',0)-item.get('start',0)))
    rate=item.get('sample_rate',48000);channels=item.get('channels',2)
    return max(1,int(max(.2,duration)*(rate*channels+16000)*4*4))


def estimate_recording_bytes(item,rate=16000):
    return max(1,int(item.get('duration',0)*item.get('channels',1)*(item.get('sample_rate',rate)+rate)*4*2))


def read_recording(item,rate=16000):
    """Decode all channels once, then resample together using the native filter."""
    import numpy as np
    import soundfile as sf
    with audio_source(item['path']) as local, io_slot(), sf.SoundFile(local) as stream:
        source_rate=stream.samplerate
        if 'sample_rate' in item and (source_rate,stream.channels,len(stream))!=(item['sample_rate'],item['channels'],item['total_samples']):
            raise ValueError('Recording format changed after ingestion')
        audio=stream.read(dtype='float32',always_2d=True)
    if audio.size==0 or not np.isfinite(audio).all():raise ValueError('Empty or nonfinite audio')
    if rate!=source_rate:
        from scipy.signal import resample_poly
        factor=math.gcd(rate,source_rate)
        audio=resample_poly(audio,rate//factor,source_rate//factor,axis=0).astype('float32')
    return audio
