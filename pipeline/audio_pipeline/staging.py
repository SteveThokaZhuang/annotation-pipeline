"""Publish complete audio from local scratch; acknowledge only after durable copy."""
import os
import json
from pathlib import Path
import shutil
import tempfile
from .common import fingerprint


def publish_audio(audio,target,scratch):
    import soundfile as sf
    target=Path(target);scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    local_fd,local=tempfile.mkstemp(suffix='.wav',prefix='enhance-',dir=scratch);os.close(local_fd)
    temporary=None
    try:
        with sf.SoundFile(local,'w',samplerate=48000,channels=audio.shape[1],subtype='FLOAT') as stream:
            for start in range(0,len(audio),48000*10):stream.write(audio[start:start+48000*10])
        target.parent.mkdir(parents=True,exist_ok=True)
        remote_fd,temporary=tempfile.mkstemp(prefix='.enhanced-',suffix='.partial.wav',dir=target.parent)
        with open(local,'rb') as source,os.fdopen(remote_fd,'wb') as dest:
            shutil.copyfileobj(source,dest,length=8*1024*1024);dest.flush();os.fsync(dest.fileno())
        os.replace(temporary,target)
        from .local_cache import seed_source
        seed_source(target,local,json.loads(os.environ.get('AUDIO_PIPELINE_CACHE','{}')))
        return fingerprint(target)
    except Exception as error:
        from .failure_policy import InfrastructureError
        raise InfrastructureError('Enhanced audio publication failed: '+str(error)) from error
    finally:
        Path(local).unlink(missing_ok=True)
        if temporary:Path(temporary).unlink(missing_ok=True)
