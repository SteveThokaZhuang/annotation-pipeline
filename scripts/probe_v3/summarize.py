"""Summarize measured probe artifacts without changing runtime state."""
import json,statistics
from pathlib import Path
CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
rows=[]
for report in sorted(CACHE.glob('*/production/*/runs/*/*-report.json')):
 r=json.loads(report.read_text());workers=r.get('workers',[])
 if not workers:continue
 batches=[];progress=[];errors=[]
 for log in report.parent.glob(r['stage']+'-worker-*.log'):
  for line in log.read_text(errors='replace').splitlines():
   try:v=json.loads(line)
   except ValueError:continue
   if v.get('event')=='batch':batches.append(v)
   if v.get('event')=='progress':progress.append(v)
   if v.get('event') in {'error','batch_retry','batch_recovery'}:errors.append(v)
 result=dict(mode=report.parents[4].name,dataset=report.parents[2].name,stage=r['stage'],status=r['status'],report=str(report),wall_seconds=r['runtime_seconds'],tasks=r['tasks'],completed=r['completed'],failed=r['failed'],skipped=r.get('skipped',0),upstream_failures=r.get('upstream_failures',0),cached=r.get('cached',0),workers=[{k:w.get(k) for k in ['worker_index','device','completed','failed','model_load_seconds','runtime_seconds','inference_seconds','cpu_total_seconds','mean_device_utilization','peak_ram_mib','peak_device_used_mib','phases','adapter_metrics','decoded_audio_cache','error']} for w in workers],batches=batches,retries_or_errors=errors)
 rows.append(result)
(CACHE/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 w=r['workers'];util=[x['mean_device_utilization'] for x in w if x.get('mean_device_utilization') is not None];b=r['batches']
 print(r['dataset'],r['stage'],'wall',round(r['wall_seconds'],1),'done/fail',r['completed'],r['failed'],'gpu%',round(statistics.mean(util),1) if util else None,'batch median',statistics.median([x['items'] for x in b]) if b else None)
