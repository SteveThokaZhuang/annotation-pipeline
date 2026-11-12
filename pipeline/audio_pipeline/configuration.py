"""Resolve portable YAML paths without inheriting the containing installation."""
import os
from pathlib import Path
import sys
import yaml


def load_yaml(path, stack=()):
    path=Path(path).resolve()
    if path in stack:raise ValueError("Configuration inheritance cycle: "+str(path))
    value=yaml.safe_load(path.read_text())
    if not isinstance(value,dict):raise ValueError("Configuration must be a mapping: "+str(path))
    bases=value.pop("extends",[])
    if isinstance(bases,str):bases=[bases]
    def merge(a,b):
        out=dict(a)
        for k,v in b.items():out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
        return out
    result={}
    for base in bases:result=merge(result,load_yaml(path.parent/base,(*stack,path)))
    return merge(result,value)
from .common import ROOT


def expand(value):
    if isinstance(value,str):
        expanded=os.path.expandvars(os.path.expanduser(value))
        if '${' in expanded:raise ValueError('Unset configuration environment variable: '+value)
        return expanded
    if isinstance(value,list):return [expand(v) for v in value]
    if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
    return value


def absolute(value,base=ROOT):
    path=Path(value)
    return str((path if path.is_absolute() else Path(base)/path).resolve())


def load_config(config_path,resources_path,datasets_path=None):
    # Pipeline/model paths use the repository root. Dataset paths use their own
    # catalog, so a dataset YAML can live alongside the dataset on any machine.
    cfg=expand(load_yaml(config_path))
    resources=expand(load_yaml(resources_path))
    for key in ('dataset_root','output_root'):cfg[key]=absolute(cfg[key])
    for spec in cfg.get('models',{}).values():
        for key in ('module','checkpoint'):
            if spec.get(key):spec[key]=absolute(spec[key])
        if spec.get('python'):
            # Preserve the venv's executable symlink: resolving it can select
            # the base interpreter and silently drop its installed dependencies.
            value=Path(spec['python'])
            spec['python']=os.path.abspath(value if value.is_absolute() else ROOT/value)
    cfg.setdefault('datasets',{})
    def resolve_entries(entries,base):
        for name,policy in entries.items():
            if not isinstance(policy,dict):raise ValueError('Dataset entry must be a mapping: '+str(name))
            for key in ('path','manifest'):
                if policy.get(key):policy[key]=absolute(policy[key],base)
    resolve_entries(cfg['datasets'],ROOT)
    if datasets_path:
        catalog=Path(datasets_path)
        document=expand(yaml.safe_load(catalog.read_text()))
        entries=document.get('datasets') if isinstance(document,dict) else None
        if not isinstance(entries,dict):raise ValueError('Dataset YAML must contain a datasets mapping')
        resolve_entries(entries,catalog.resolve().parent)
        cfg['datasets'].update(entries)
    optimization=resources.setdefault('optimization',{})
    working=optimization.get('working_cache',{})
    for key in ('enabled','sync_enabled'):
        if key in working and not isinstance(working[key],bool):
            raise ValueError('working_cache.'+key+' must be boolean')
    if working.get('enabled'):
        if not working.get('root'):raise ValueError('working_cache.root is required when enabled')
        working['root']=absolute(working['root'])
    optimization['scratch_root']=absolute(optimization.get('scratch_root','/tmp/audio-pipeline/scratch'))
    cache=optimization.setdefault('cache',{})
    cache['root']=absolute(cache.get('root','/tmp/audio-pipeline/cache'))
    scheduler=resources.setdefault('scheduler',{})
    scheduler['gpu_lock_root']=absolute(os.environ.get('PIPELINE_GPU_LOCK_ROOT',scheduler.get('gpu_lock_root','/tmp/audio-pipeline-gpu-locks')))
    for key in ('ipc_root',):
        if optimization.get(key):optimization[key]=absolute(optimization[key])
    for name in ('discovery','feature_transport'):
        if optimization.get(name,{}).get('root'):optimization[name]['root']=absolute(optimization[name]['root'])
    for entry in resources.get('stages',{}).values():
        for section in ('cache','feature_transport'):
            if entry.get(section,{}).get('root'):entry[section]['root']=absolute(entry[section]['root'])
    from .config_v3 import normalize,validate_nested
    normalize(cfg,resources)
    validate_nested(cfg,resources)
    if resources.get('version')==3:cfg['_resources_v3']=resources
    cfg.setdefault('_configuration',{}).update(config_path=str(Path(config_path).resolve()),resources_path=str(Path(resources_path).resolve()))
    return cfg,resources


def python_for(cfg,name):
    key='PIPELINE_PYTHON_'+name.upper().replace('-','_')
    return os.environ.get(key) or cfg.get('environments',{}).get(name) or sys.executable
