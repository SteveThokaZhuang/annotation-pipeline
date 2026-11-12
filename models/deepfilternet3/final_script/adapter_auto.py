"""Per-recording selection of whole or VAD-guided enhancement, with explicit identity."""
from pathlib import Path
from audio_pipeline.common import load_module,read,fingerprint
_chunked=load_module(Path(__file__).with_name('adapter_chunked.py'),'auto_chunked').Adapter
_whole=load_module(Path(__file__).with_name('adapter.py'),'auto_whole').Adapter


def choose(item,settings):
    policy=settings['auto_policy']
    channels=max(1,item['channels']);batch=min(channels,settings['channel_batch_size'])
    ram=item['duration']*settings['estimated_ram_gib_per_second']*channels+2+2*settings.get('prepare_max_gib',0)
    gpu=item['duration']*settings['estimated_gpu_gib_per_second']*batch+2
    return 'whole_sequence' if (item['duration']<=min(policy['whole_max_seconds'],settings['max_gpu_seconds'],settings['max_seconds']) and ram<=policy['whole_max_ram_gib'] and gpu<=policy['whole_max_gpu_gib']) else 'vad_chunked_v1'


class Adapter(_chunked):
    def start(self,items):
        short=[item for item in items if item['_enhance_method']=='whole_sequence']
        if short:_whole.start(self,short)

    def process(self,item):
        if item.get('_segment_only'):
            plan=read(item['_plan_path']);recording=item['_recording']
            piece=plan['pieces'][item['_part_index']]
            directory=Path(recording['_artifact_dir'])/'segments';directory.mkdir(parents=True,exist_ok=True)
            path=self._part(recording,plan,piece,item['_channel'],item['_parts_key'],directory)
            return dict(recording_id=recording['id'],path=str(path),duration=item['duration'],segment_index=item['_part_index'],channel=item['_channel'],artifacts=[fingerprint(path)])
        if item['_enhance_method']=='whole_sequence':return _whole.process(self,item)
        return super().process(item)
