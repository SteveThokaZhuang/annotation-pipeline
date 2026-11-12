"""Durable immutable task plans for trusted recovery of unfinished stages."""
from .common import atomic,digest,read,transform
from .resume_checkpoint import stamp
from .task_manifest import read_tasks,write_tasks
from .resume import ScanStopped


def paths(state,stage,model,token):
    root=state.root/'control/resume/plans'
    name=stage+'-'+digest(model)[:16]+'-'+token[:24]
    return root/(name+'.json'),root/(name+'.jsonl')


def load(state,stage,model,token,stop_file):
    if token is None:return None
    header,path=paths(state,stage,model,token)
    try:
        metadata=read(header)
        if metadata['schema']!=1 or metadata['context']!=token or metadata['snapshot']!=stamp(path):return None
        tasks=[];mapped=str(state.root)!=metadata['root']
        for index,task in enumerate(read_tasks(path)):
            if index%8192==0 and stop_file.exists():raise ScanStopped()
            tasks.append(transform(task) if mapped else task)
        if len(tasks)!=metadata['tasks']:return None
        state.skipped=metadata.get('skipped',[])
        for row in metadata.get('issues',[]):
            state.dependency_failures[(row['stage'],row['model'],row['id'])]=row
        return tasks
    except (OSError,ValueError,KeyError,TypeError):return None


def save(state,stage,model,token,tasks):
    if token is None:return
    header,path=paths(state,stage,model,token)
    write_tasks(path,tasks)
    atomic(header,dict(schema=1,context=token,root=str(state.root),snapshot=stamp(path),tasks=len(tasks),
        skipped=state.skipped,issues=list(state.dependency_failures.values())))
