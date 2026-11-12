#!/usr/bin/env python3
"""Read-only deployment checks; does not initialize inference models or CUDA."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))
from audio_pipeline.configuration import load_config,python_for
from audio_pipeline.planning import ADAPTERS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/pipeline.yaml')
    parser.add_argument('--resources',type=Path,default=ROOT/'config/resources.yaml')
    parser.add_argument('--datasets',type=Path)
    args=parser.parse_args()
    cfg,res=load_config(args.config,args.resources,args.datasets)
    entries=[]
    for name,spec in cfg['models'].items():
        entries.append(dict(name=name,python=spec.get('python') or python_for(cfg,spec.get('environment',name)),checkpoint=spec['checkpoint']))
    for name,(env,module,checkpoint) in ADAPTERS.items():
        override=cfg.get('stage_models',{}).get(name,{})
        location=override.get('checkpoint',str(ROOT/'models'/checkpoint) if checkpoint else None)
        if location and not Path(location).is_absolute():location=str(ROOT/location)
        entries.append(dict(name=name,python=override.get('python',python_for(cfg,env)),checkpoint=location))
    for entry in entries:
        entry['python_found']=bool(shutil.which(entry['python']))
        entry['checkpoint_found']=bool(entry['checkpoint'] and Path(entry['checkpoint']).exists()) if entry['checkpoint'] else True
    result=dict(models=entries,cache=res['optimization']['cache'],scratch=res['optimization']['scratch_root'],
                gpu_lock_root=res['scheduler']['gpu_lock_root'],datasets=list(cfg['datasets']),gcc=shutil.which('gcc'))
    result['ready']=all(e['python_found'] and e['checkpoint_found'] for e in entries)
    print(json.dumps(result,indent=2))
    return 0 if result['ready'] else 1

if __name__=='__main__':sys.exit(main())
