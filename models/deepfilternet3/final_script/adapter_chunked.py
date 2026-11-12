"""Opt-in vad_chunked_v1: context, complementary crossfades, resumable parts.

The legacy whole-recording adapter is unchanged. Recurrent/STFT state is reset
for every bounded channel segment; this method is intentionally a new identity.
"""
import os
from pathlib import Path
import time

from audio_pipeline.common import atomic,digest,fingerprint,load_module,read
from audio_pipeline.cache_version import inference_descriptor
from audio_pipeline.enhancement_chunks_v1 import METHOD,policy,read_segment,join_weights
from audio_pipeline.working_cache import canonical_path,preferred_path

_Legacy = load_module(Path(__file__).with_name('adapter.py'),'legacy_whole_enhancement').Adapter


class Adapter(_Legacy):
    def __init__(self,job):
        policy(job['settings'])
        super().__init__(job)
        if self.frontend_parameters['sr']!=48000:
            raise ValueError('vad_chunked_v1 requires the 48 kHz DeepFilterNet model')
        self.metrics.update(segment_calls=0,reused_segments=0,segment_read_seconds=0.,
                            segment_inference_seconds=0.,assembly_seconds=0.)

    def start(self,items):
        # Never start legacy whole-recording preparation for a segmented run.
        pass

    def set_stop_requested(self,callback):
        self._stop_requested=callback

    def _check_stop(self):
        if (getattr(self,'_stop_requested',lambda:False)() or
            (self.job.get('stop_file') and Path(self.job['stop_file']).exists())):
            raise InterruptedError('Stopped during chunked enhancement; segment checkpoints retained')

    def _native(self,audio):
        import torch
        from libdf import DF
        # Fresh transform state avoids leaking context between channels/parts.
        state=DF(**self.frontend_parameters)
        with torch.inference_mode():
            value=self.enhance(self.model,state,torch.from_numpy(audio[None].copy()),pad=True)
        return value.numpy()[0]

    def _part(self,item,plan,piece,channel,key,directory):
        import numpy as np
        import soundfile as sf
        self._check_stop()
        stem=f"s{piece['index']:06d}-c{channel:04d}"
        path=directory/(stem+'.wav');checkpoint=directory/(stem+'.json')
        expected=piece['output_end']-piece['output_start']
        try:
            saved=read(canonical_path(checkpoint))
            candidate=Path(preferred_path(canonical_path(path)))
            info=sf.info(candidate)
            if (saved['key']==key and digest(saved['artifact'])==digest(fingerprint(candidate)) and
                saved['start']==piece['output_start'] and saved['end']==piece['output_end'] and
                info.frames==expected and info.channels==1 and info.samplerate==plan['rate']):
                if candidate!=path:
                    import shutil
                    temporary=path.with_suffix('.partial.wav')
                    try:
                        shutil.copy2(candidate,temporary)
                        with temporary.open('rb') as stream:os.fsync(stream.fileno())
                        os.replace(temporary,path)
                        atomic(checkpoint,saved)
                    finally:temporary.unlink(missing_ok=True)
                self.metrics['reused_segments']+=1
                return path
        except (OSError,ValueError,KeyError,RuntimeError):
            pass
        duration=(piece['read_end']-piece['read_start'])/plan['rate'];cfg=self.job['settings']
        from audio_pipeline.memory import available_host_bytes
        ram=duration*cfg.get('estimated_ram_gib_per_second',.04)+2
        ram+=duration*(item['sample_rate']*item['channels']+plan['rate'])*4*3/2**30
        if ram>min(cfg.get('max_estimated_ram_gib',32),available_host_bytes()/2**30*cfg.get('host_headroom_fraction',.75)):
            raise RuntimeError('Segment enhancement exceeds host RAM headroom')
        if self.job['device'] is not None:
            import torch
            gpu=duration*cfg.get('estimated_gpu_gib_per_second',.02)+2
            if duration>cfg.get('max_gpu_seconds',1800) or gpu>torch.cuda.mem_get_info()[0]/2**30*cfg.get('gpu_headroom_fraction',.75):
                raise RuntimeError('Segment enhancement exceeds configured GPU headroom; lower segment duration')
        tick=time.perf_counter();audio=read_segment(item,piece,channel,plan['rate'])
        self.metrics['segment_read_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();enhanced=self._native(audio)
        self.metrics['segment_inference_seconds']+=time.perf_counter()-tick
        if enhanced.shape!=audio.shape or not np.isfinite(enhanced).all():
            raise ValueError('Native segment output must be finite and preserve exact length')
        offset=piece['output_start']-piece['read_start']
        enhanced=enhanced[offset:offset+expected]
        self._check_stop()
        temporary=path.with_suffix('.partial.wav')
        try:
            sf.write(temporary,enhanced,plan['rate'],subtype='FLOAT',format='WAV')
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(temporary,path)
            atomic(checkpoint,dict(key=key,artifact=fingerprint(path),start=piece['output_start'],end=piece['output_end']))
        finally:temporary.unlink(missing_ok=True)
        self.metrics['segment_calls']+=1
        return path

    def _assemble(self,plan,parts,target):
        import numpy as np
        import soundfile as sf
        self._check_stop();tick=time.perf_counter()
        temporary=target.with_suffix('.partial.wav')
        frames=plan['total_samples'];channels=plan['channels'];rate=plan['rate']
        # RF64 handles recordings whose float WAV data exceed the RIFF 4 GiB limit.
        format='RF64' if frames*channels*4+4096>=2**32 else 'WAV'
        try:
            first=0
            with sf.SoundFile(temporary,'w',samplerate=rate,channels=channels,subtype='FLOAT',format=format) as output:
                for start in range(0,frames,max(1,round(rate*self.job['settings'].get('assembly_block_seconds',10)))):
                    self._check_stop();end=min(frames,start+max(1,round(rate*self.job['settings'].get('assembly_block_seconds',10))))
                    values=np.zeros((end-start,channels),dtype=np.float64)
                    weights=np.zeros(end-start,dtype=np.float64)
                    pieces=plan['pieces']
                    while first<len(pieces) and pieces[first]['output_end']<=start:first+=1
                    index=first
                    while index<len(pieces) and pieces[index]['output_start']<end:
                        piece=pieces[index];a=max(start,piece['output_start']);b=min(end,piece['output_end'])
                        if a<b:
                            w=join_weights(piece,plan,a,b);weights[a-start:b-start]+=w
                            for channel in range(channels):
                                with sf.SoundFile(parts[index][channel]) as source:
                                    source.seek(a-piece['output_start']);audio=source.read(b-a,dtype='float32')
                                if len(audio)!=b-a or not np.isfinite(audio).all():
                                    raise ValueError('Incomplete/nonfinite segment during assembly')
                                values[a-start:b-start,channel]+=audio*w
                        index+=1
                    if np.any(weights<=0):raise ValueError('Uncovered samples in segment assembly')
                    values/=weights[:,None]
                    if not np.isfinite(values).all():raise ValueError('Nonfinite assembled enhancement')
                    output.write(values.astype(np.float32))
            info=sf.info(temporary)
            if (info.frames,info.channels,info.samplerate)!=(frames,channels,rate):
                raise ValueError('Assembled recording changed timeline/channel layout')
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            self._check_stop();os.replace(temporary,target)
        finally:
            temporary.unlink(missing_ok=True)
            self.metrics['assembly_seconds']+=time.perf_counter()-tick

    def process(self,item):
        self._check_stop()
        if fingerprint(item['path'])!=item['source_fingerprint']:
            raise ValueError('Source changed before chunked enhancement')
        plan=item.get('enhancement_plan')
        if not plan or plan.get('method_version')!=METHOD:
            raise ValueError('Missing versioned VAD enhancement plan')
        if plan['source_fingerprint']!=item['source_fingerprint'] or plan['channels']!=item['channels']:
            raise ValueError('Enhancement plan input mismatch')
        directory=Path(item['_artifact_dir']);directory.mkdir(parents=True,exist_ok=True)
        plan_path=directory/'segment-plan.json'
        atomic(plan_path,plan)
        part_dir=directory/'segments';part_dir.mkdir(exist_ok=True)
        key=digest(dict(plan=plan,adapter=self.job.get('adapter_version'),
                        checkpoint=self.job.get('checkpoint_version'),
                        settings=inference_descriptor(dict(settings=self.job['settings']))['settings']))
        parts=[]
        for piece in plan['pieces']:
            parts.append([self._part(item,plan,piece,channel,key,part_dir) for channel in range(item['channels'])])
        target=directory/'enhanced.wav'
        self._assemble(plan,parts,target)
        if fingerprint(item['path'])!=item['source_fingerprint']:
            raise ValueError('Source changed during chunked enhancement')
        return dict(recording_id=item['id'],path=str(target),sample_rate=plan['rate'],
                    total_samples=plan['total_samples'],channels=plan['channels'],
                    duration=plan['total_samples']/plan['rate'],artifacts=[fingerprint(target),fingerprint(plan_path)],
                    mode=METHOD,method_version=METHOD,segment_count=len(plan['pieces']),
                    plan_version=digest(plan),context_seconds=plan['policy']['context_seconds'],
                    crossfade_seconds=plan['policy']['crossfade_seconds'])
