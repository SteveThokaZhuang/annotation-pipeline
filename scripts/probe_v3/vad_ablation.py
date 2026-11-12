"""Compare VAD helper counts on spare GPU 3 with the same 100 short files."""
import json,os,subprocess,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
PYTHON='/hbt_share02/202610/ob/ob_ppl_env_data/envs/firered-vad/bin/python'
env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['pipeline','vendor','models/funasr/source','models/deepfilternet3/DeepFilterNet']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
rows=[]
for helpers in [12,2]:
 res=yaml.safe_load((ROOT/'config/resources.probe-v3-duplexconv.local.yaml').read_text())
 res['gpu']['devices']=[3];res['stages']['ingest']['workers']=4
 res['stages']['vad'].update(gpu_devices=[3],workers=1,prepare_workers=helpers)
 res['optimization']['scratch_root']=str(CACHE/f'vad-ablation-{helpers}-scratch')
 resource=ROOT/f'config/resources.probe-v3-vad-{helpers}.local.yaml';resource.write_text(yaml.safe_dump(res,sort_keys=False))
 command=[PYTHON,str(ROOT/'run.py'),'ingest','vad','--dataset','duplexconv','--config',str(ROOT/'config/pipeline.probe-v3-duplexconv.local.yaml'),'--resources',str(resource),'--output-root',str(CACHE/f'vad-ablation-{helpers}'),'--no-enhance']
 start=time.time()
 with (CACHE/f'vad-ablation-{helpers}.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(helpers=helpers,gpu=3,start=start,end=time.time(),returncode=result.returncode,command=command,concurrent_work='Basis DNSMOS on GPUs 0/1/2'))
 (CACHE/'vad-ablation-driver.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows[-1]),flush=True)
