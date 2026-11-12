"""Controller-only bounded reuse of validated immutable stage results.

Recheck manifest, result-file and source metadata before reuse. Workers retain
independent source checks; invalid or changed entries are reloaded normally.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from .common import fingerprint

_ENTRIES=OrderedDict()
_BYTES=0

def get_results(state,stage,required,loader):
    global _BYTES
    budget=getattr(state,'manifest_cache_bytes',0)
    trusted=getattr(state,'resume_validation','full')=='trusted'
    if not budget:return loader(state,stage,required)
    try:path=state.head_path(stage)
    except RuntimeError:
        if required:raise
        return {},None
    key=(str(path),'trusted' if trusted else 'full');entry=_ENTRIES.get(key)
    def snapshot(name):
        stop=getattr(state,'stop_file',None)
        if stop and stop.exists():
            from .resume import ScanStopped
            raise ScanStopped()
        try:return fingerprint(name)
        except OSError:return None
    if entry:
        names=[str(path)] if trusted else list(entry['snapshots'])
        with ThreadPoolExecutor(max_workers=min(16,getattr(state,'scan_workers',8))) as pool:
            valid=all(actual==entry['snapshots'][name] for name,actual in zip(names,pool.map(snapshot,names)))
        if valid:
            _ENTRIES.move_to_end(key)
            state.dependency_failures.update(entry['issues'])
            state.manifest_cache_hits=getattr(state,'manifest_cache_hits',0)+1
            return entry['value']
        _BYTES-=entry['cost'];del _ENTRIES[key]
    before=set(state.dependency_failures)
    value=loader(state,stage,required);values,head=value
    if head is None:return value
    output_files={str(path)}
    for row in head['entries']:
        output=row.get('output')
        if output:output_files.add(output.rpartition('#result=')[0] if '#result=' in output else output)
    names=set(output_files)
    for row in ([] if trusted else values.values()):
        source=row.get('_provenance',{}).get('input',{});source=source.get('chunk',source)
        for name in ['source_fingerprint','audio_fingerprint']:
            if source.get(name):names.add(source[name]['path'])
        for artifact in row.get('artifacts',[]):names.add(artifact['path'])
    with ThreadPoolExecutor(max_workers=min(16,getattr(state,'scan_workers',8))) as pool:
        snapshots=dict(zip(names,pool.map(snapshot,names)))
    if any(v is None for v in snapshots.values()):return value
    cost=6*sum(snapshots[name]['size'] for name in output_files)
    if cost>budget:return value
    while _ENTRIES and _BYTES+cost>budget:
        _,old=_ENTRIES.popitem(last=False);_BYTES-=old['cost']
    _ENTRIES[key]=dict(value=value,cost=cost,snapshots=snapshots,issues={k:v for k,v in state.dependency_failures.items() if k not in before})
    _BYTES+=cost
    return value
