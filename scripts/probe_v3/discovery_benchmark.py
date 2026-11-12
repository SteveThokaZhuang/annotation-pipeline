"""Read-only full-catalog metadata scan; no model inference or audio decoding."""
import json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
sys.path.insert(0,str(ROOT/'pipeline'))
from audio_pipeline.configuration import load_config
from audio_pipeline.datasets import discover
cfg,res=load_config(ROOT/'config/pipeline.vllm.duplexconv.v1-calibrated.local.yaml',ROOT/'config/resources.duplexconv.v1-calibrated.local.yaml')
rows=[]
for name in ['duplexconv']:
 start=time.perf_counter();cpu=time.process_time();items=discover(cfg,name)
 row=dict(dataset=name,files=len(items),wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,total_source_bytes=sum(i['source_fingerprint']['size'] for i in items),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,scope='Full original dataset discovery: directory traversal, supported-extension filtering, source resolution/stat, duplicate validation and sorting. No audio decoding or GPU inference. Concurrent Basis pipeline finishing on GPUs0/1/2; no repeated trial.',sample_manifest_files=100)
 rows.append(row);print(json.dumps(row),flush=True)
(CACHE/'full-discovery-benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
