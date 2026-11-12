"""Expose independently resumable enhancement segments to all allocated GPUs."""
from pathlib import Path
from .common import atomic,digest
from .cache_version import inference_descriptor


def parts_for(tasks,spec):
    result=[]
    for task in tasks:
        item=task['item'];plan=item.get('enhancement_plan')
        if not plan or item.get('_planning_error'):continue
        directory=Path(item['_artifact_dir']);directory.mkdir(parents=True,exist_ok=True)
        plan_path=directory/'segment-plan.json';atomic(plan_path,plan)
        key=digest(dict(plan=plan,adapter=spec.get('adapter_version'),checkpoint=spec.get('checkpoint_version'),settings=inference_descriptor(dict(settings=spec['settings']))['settings']))
        recording={k:v for k,v in item.items() if k!='enhancement_plan'}
        for piece in plan['pieces']:
            for channel in range(item['channels']):
                identity=f"{item['id']}_s{piece['index']:06d}_c{channel}"
                result.append(dict(id=identity,recording_id=item['id'],path=item['path'],source_fingerprint=item['source_fingerprint'],duration=(piece['read_end']-piece['read_start'])/plan['rate'],
                    _segment_only=True,_enhance_method='vad_chunked_v1',_recording=recording,_plan_path=str(plan_path),_part_index=piece['index'],_channel=channel,_parts_key=key))
    return result
