"""Read source metadata; processing stages validate decoded audio on use."""
from audio_pipeline.audio import io_slot
from audio_pipeline.common import fingerprint

class Adapter:
    def __init__(self, job):
        from concurrent.futures import ThreadPoolExecutor
        self.pool=ThreadPoolExecutor(max_workers=min(4,job.get('allocation',{}).get('io_workers',4)))

    def process(self, item):
        import soundfile as sf
        path=item['path']
        with io_slot(), sf.SoundFile(path) as stream:
            info=dict(sample_rate=stream.samplerate,channels=stream.channels,total_samples=len(stream),
                      duration=len(stream)/stream.samplerate,format=stream.format,subtype=stream.subtype)
        return {**item,**info,'file_status':'metadata_read','source_fingerprint':fingerprint(path)}

    def process_batch(self,items):
        return list(self.pool.map(self.process,items))

    def close(self):self.pool.shutdown(wait=True)
