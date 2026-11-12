"""Assemble measured findings and explicit limitations for the v3 probe."""
import json,hashlib,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
def read(p):return json.loads(Path(p).read_text())
def relative_artifact(p):return str(p)
if (CACHE/'code-integrity.json').exists():
 integrity=read(CACHE/'code-integrity.json')
else:
 manifest=read(ROOT/'V3_COPY_MANIFEST.json');changed=[]
 for entry in manifest['files']:
  p=ROOT/entry['path']
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:changed.append(entry['path'])
 integrity={'verified_original_files':len(manifest['files']),'changed_original_files':changed,'original_code_unchanged':not changed}
 (CACHE/'code-integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
records=[]
for f in sorted(CACHE.glob('*/production/*/runs/*/*-report.json')):
 r=read(f);workers=r.get('workers',[])
 if not workers:continue
 setup_issue=any('ipc path' in (w.get('model_error','')+w.get('error','')) and '107' in (w.get('model_error','')+w.get('error','')) for w in workers)
 row=dict(mode=f.parents[4].name,dataset=f.parents[2].name,stage=r['stage'],run=f.parent.name,report=str(f),status=r['status'],tasks=r['tasks'],completed=r['completed'],failed=r['failed'],cached=r.get('cached',0),upstream_failures=r.get('upstream_failures',0),wall_seconds=r['runtime_seconds'],probe_socket_setup_failure=setup_issue,worker_load_seconds_max=max(w.get('model_load_seconds',0) for w in workers),worker_inference_seconds_max=max(w.get('inference_seconds',0) for w in workers),worker_cpu_seconds_sum=sum(w.get('cpu_total_seconds',0) for w in workers),gpu_utilization_mean=statistics.mean(w['mean_device_utilization'] for w in workers if 'mean_device_utilization' in w) if any('mean_device_utilization' in w for w in workers) else None)
 records.append(row)
selection=[]
for name in ['duplexconv','basis-conversations-1500-en']:
 d=read(CACHE/(name+'-sample-info.json'));v=sorted(x['duration'] for x in d['samples'])
 selection.append(dict(dataset=name,files=len(v),audio_hours=sum(v)/3600,minimum_seconds=min(v),median_seconds=statistics.median(v),maximum_seconds=max(v),over_30_minutes=sum(x>1800 for x in v),over_60_minutes=sum(x>3600 for x in v),selection=d['selection'],candidate_scope=d['candidate_scope'],candidates=d['candidates'],details=str(CACHE/(name+'-sample-info.json'))))
key_measurements=[]
for name in ['duplexconv','basis-conversations-1500-en']:
 for stage in ['ingest','dnsmos','vad','enhance','asr1','asr2','disagreement','asr3','rover','align']:
  candidates=[r for r in records if r['mode']=='baseline' and r['dataset']==name and r['stage']==stage and not r['probe_socket_setup_failure'] and r['status']!='stopped' and r['cached']==0]
  if candidates:key_measurements.append(candidates[0])
conclusions=[
 {'finding':'Large file counts introduce measurable metadata-only startup work.', 'evidence':'Full DuplexConv discovery examined209060 audio files in95.7s with no audio decoding or model work.', 'action':'Maintain a persistent dataset catalog and perform incremental discovery/validation; stream bounded task planning rather than rebuilding the full catalog on every launch.'},
 {'finding':'Alignment rejected short chunks for word timestamps outside their allowed bounds.', 'evidence':'11/2953 Basis alignment chunks failed; examples are0.81-6.87s, not hour-long model inputs.', 'action':'Keep timestamp validation, add explicit alignment failure reporting, and assess transcript/audio mismatch or boundary-context retries before changing tolerance.'},
 {'finding':'Whole-recording GPU enhancement rejects duration >1800 seconds.','evidence':'65/100 Basis files rejected by the same explicit compatibility guard; all100 DuplexConv files enhanced.','action':'Use bounded enhancement segments for long inputs and preserve sample/timestamp mapping.'},
 {'finding':'ASR initialization dominates small-file sample wall time.','evidence':'DuplexConv Qwen load ~150s vs ~8s worker decode; FunASR load ~210s plus helper startup vs ~2s decoder time; FireRed load ~311s.','action':'Reuse model workers across workloads, and measure runtime/worker startup separately from steady inference.'},
 {'finding':'VAD waits for whole-recording CPU preparation before its GPU batches.','evidence':'Basis workers wait ~68-71s for preparation vs ~6s model time. Short-file workers wait ~22s vs <1s model time.','action':'Schedule bounded prepared work by readiness and duration; avoid requiring a group of long recordings to finish preparation before the GPU can start.'},
 {'finding':'Increasing preparation helper counts alone did not improve the short-file sample.','evidence':'One-GPU VAD: 12 helpers 36.4s/48.2 CPU seconds; 2 helpers 35.4s/26.7 CPU seconds; exact same outputs.','action':'Choose helpers from measured throughput and real CPU quota; repeat controlled trials before claiming a speedup.'},
 {'finding':'The inherited Basis scratch paths exceed the installed vLLM IPC path length limit without an override.','evidence':str(CACHE/'ipc-path-audit.json'),'action':'Use a short cache-backed scratch/IPC directory and validate path length before model startup.'},
 {'finding':'File-count checkpoint groups can obscure progress on long inputs.','evidence':'Basis DNSMOS processed ~75000 windows per worker while completed-recording counters remained zero.','action':'Expose window/chunk progress and use elapsed-time/byte limits for checkpoint flushing.'},
 {'finding':'The inherited GPU profiles select only GPUs0/1/2, while four H100s are available.','evidence':str(CACHE/'hardware.json'),'action':'Measure stage throughput with a hardware-aware allocation; reserve capacity for other stages or additional replicas as appropriate.'},
 {'finding':'The installed vLLM engine owns GPU memory in a subprocess.','evidence':'Qwen controller peak allocation reported 0GiB while driver-reported GPU usage was ~66GB.','action':'Include engine memory and batch execution metrics in adaptive policy feedback.'}
]
report={'updated_utc_epoch':time.time(),'code_root':str(ROOT),'runtime_root':str(CACHE),'final_destination_sync':False,'code_integrity':integrity,'hardware':read(CACHE/'hardware.json'),'datasets':selection,'selected_stage_measurements':key_measurements,'all_stage_runs':records,'conclusions':conclusions,'supplementary':{name:read(CACHE/name) for name in ['vad-ablation-comparison.json','chunked-longest-validation.json','raw-qwen-validation.json','no-speech-audio-check.json','ipc-path-audit.json','probe-setup-correction.json','failure-audit.json','full-discovery-benchmark.json'] if (CACHE/name).exists()},'limitations':['Duration-stratified 100-file samples, not whole-dataset or million-file scale validation. DuplexConv survey covers a bounded3000-file candidate pool.','No labeled transcription accuracy or subjective enhancement quality evaluation.','GPU utilization is sampled; short bursts can be missed.','Supplementary tests used GPU3 concurrently with Basis inference on GPUs0/1/2; elapsed-time comparisons are diagnostic, not statistically established speedups.','The first Basis ASR attempt hit a probe scratch-path IPC error and was stopped; those startup failures are excluded from dataset-rejection findings.','Three sampled Basis files produced no VAD chunks; they are mostly zero-valued, not proven to contain no speech.']}
(CACHE/'probe-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(CACHE/'probe-report.json'),'integrity':integrity,'recorded_stage_runs':len(records),'primary_measurements':len(key_measurements)},indent=2))
