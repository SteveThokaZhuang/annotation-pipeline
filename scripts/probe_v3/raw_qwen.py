"""Run Qwen on all 100 long recordings' VAD chunks, bypassing enhancement diagnostically."""
import json,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
sys.path.insert(0,str(ROOT/'pipeline'))
from audio_pipeline.common import read,atomic
from audio_pipeline.state import State
while not (CACHE/'chunked-longest-driver.json').exists():time.sleep(2)
name='basis-conversations-1500-en';destination=CACHE/'raw-qwen/production'/name
state=State(destination)
try:
 for stage in ['ingest','vad']:
  reports=list((CACHE/'baseline/production'/name/'runs').glob('*/'+stage+'-report.json'))
  source=next(read(p) for p in reports if read(p).get('completed')==100 and read(p).get('cached')==0)
  head=read(source['manifest']);path=destination/'manifests'/('reused-'+stage+'.json');atomic(path,head);state.set_head(stage,path)
finally:state.db.close()
cfg=yaml.safe_load((ROOT/f'config/pipeline.probe-v3-{name}.local.yaml').read_text());cfg['output_root']=str(CACHE/'raw-qwen')
res=yaml.safe_load((ROOT/f'config/resources.probe-v3-{name}.local.yaml').read_text());res['gpu']['devices']=[3];res['stages']['asr1'].update(workers=1,gpu_devices=[3]);res['optimization']['scratch_root']=str(CACHE/'s/r')
cp=ROOT/'config/pipeline.probe-v3-raw-qwen.local.yaml';rp=ROOT/'config/resources.probe-v3-raw-qwen.local.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));rp.write_text(yaml.safe_dump(res,sort_keys=False))
env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['pipeline','vendor','models/funasr/source','models/deepfilternet3/DeepFilterNet']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(CACHE/'hf'),VLLM_CACHE_ROOT=str(CACHE/'vllm_cache'),TORCHINDUCTOR_CACHE_DIR=str(CACHE/'torchinductor'),TRITON_CACHE_DIR=str(CACHE/'triton'))
command=[sys.executable,str(ROOT/'run.py'),'asr1','--dataset',name,'--config',str(cp),'--resources',str(rp),'--no-enhance']
start=time.time()
with (CACHE/'raw-qwen.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(CACHE/'raw-qwen-driver.json').write_text(json.dumps(dict(start=start,end=time.time(),returncode=result.returncode,command=command,note='Qwen covers VAD chunks from all100 original recordings, independent of whole-recording enhancement rejection. GPU3, concurrently with baseline on GPUs0/1/2; no quality comparison intended.'),indent=2)+'\n')
print(result.returncode,flush=True)
