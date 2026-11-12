"""Dependency-free bootstrap; environment paths and defaults live in configuration."""
import json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
settings=json.loads((root/'config/launch.v3.json').read_text())
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join([str(root/p) for p in settings['python_paths']]+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
python=env.get('PIPELINE_PYTHON',settings['python'])
args=[python,str(root/'run.py'),'--config',str(root/settings['pipeline']),'--resources',str(root/settings['resources']),'--datasets',str(root/settings['datasets']),*sys.argv[1:]]
os.execve(python,args,env)
