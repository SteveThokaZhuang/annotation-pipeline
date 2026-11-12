import json
from pathlib import Path
cache=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
for log in cache.glob('*-baseline.log'):
 lines=log.read_text(errors='replace').splitlines(); last=next((s for s in reversed(lines) if s.startswith('{') and '"stage"' in s),'{}')
 try:r=json.loads(last)
 except ValueError:r={}
 stage=r.get('stage');root=cache/'baseline/production'/log.name.removesuffix('-baseline.log')/'runs'
 print(log.name, {k:r.get(k) for k in ['stage','phase','status','elapsed_seconds','completed','total','failed'] if k in r})
 if root.exists():
  run=sorted(root.iterdir())[-1]
  for f in sorted(run.glob(str(stage)+'-worker-*.log')):
   if f.with_suffix('.json').exists():continue
   progress=[]
   for s in f.read_text(errors='replace').splitlines():
    try:v=json.loads(s)
    except ValueError:continue
    if v.get('event')=='progress':progress.append(v)
   if progress:
    w=progress[-1];m=w.get('adapter_metrics',{})
    print(f.name,{k:w.get(k) for k in ['phase','elapsed','completed','failed']},'gpu',w.get('resources',{}).get('gpu_utilization'),'metrics',{k:m[k] for k in ['prepare_wait_seconds','model_seconds','windows','pipeline_prepare_seconds','pipeline_decode_seconds','pipeline_batches'] if k in m})
