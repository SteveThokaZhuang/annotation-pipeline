import json
from pathlib import Path
cache=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
for d in sorted((cache/'baseline/production').glob('*')):
 print('\nDATASET',d.name)
 for run in sorted((d/'runs').glob('*')):
  print('RUN',run.name)
  for p in sorted(run.glob('*-report.json'),key=lambda p:p.stat().st_mtime):
   r=json.loads(p.read_text());print(r['stage'],r['status'],'wall',round(r['runtime_seconds'],2),'tasks',r['tasks'],'done',r['completed'],'failed',r['failed'])
  for f in sorted(run.glob('*-worker-*.log')):
   if f.with_suffix('.json').exists() or f.name.startswith('ingest'):continue
   lines=f.read_text(errors='replace').splitlines()
   
   tail=next((line for line in reversed(lines) if line.startswith('{')), lines[-1] if lines else '')
   try:
    v=json.loads(tail);tail=json.dumps({k:v[k] for k in ['event','phase','completed','failed','elapsed','resources','phases','adapter_metrics','items','adapter_seconds','next_batch_limit'] if k in v})
   except (ValueError,TypeError):pass
   print(f.name,tail[:2200])
 log=cache/(d.name+'-baseline.log')
 if log.exists():print('CONTROLLER',log.read_text(errors='replace').splitlines()[-1][:1400])
