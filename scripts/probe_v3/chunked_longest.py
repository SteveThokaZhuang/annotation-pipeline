"""Exercise the existing opt-in segmented enhancer on the rejected longest input."""
import json,os,subprocess,sys,time,copy
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
sys.path.insert(0,str(ROOT/'pipeline'))
from audio_pipeline.common import atomic,read,digest
from audio_pipeline.state import State
name='basis-conversations-1500-en';info=json.loads((CACHE/(name+'-sample-info.json')).read_text());sample=max(info['samples'],key=lambda r:r['duration']);identity=f"sample_{info['samples'].index(sample):03d}"
manifest=CACHE/'longest-only.jsonl';manifest.write_text(json.dumps(dict(path=sample['path'],id=identity))+'\n')
cfg=yaml.safe_load((ROOT/'config/pipeline.vllm.basis-conversations-1500.vad-chunked-v1.local.yaml').read_text())
cfg['datasets']={name:dict(manifest=str(manifest),language='English',enhance=True)};cfg['output_root']=str(CACHE/'chunked-longest')
res=yaml.safe_load((ROOT/f'config/resources.probe-v3-{name}.local.yaml').read_text());res['gpu']['devices']=[3];res['stages']['enhance'].update(workers=1,gpu_devices=[3]);res['optimization']['scratch_root']=str(CACHE/'chunked-longest-scratch')
cp=ROOT/'config/pipeline.probe-v3-chunked-longest.local.yaml';rp=ROOT/'config/resources.probe-v3-chunked-longest.local.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));rp.write_text(yaml.safe_dump(res,sort_keys=False))
destination=CACHE/'chunked-longest/production'/name;state=State(destination)
try:
 for stage in ['ingest','vad']:
  report=read(next((CACHE/'baseline/production'/name/'runs').glob('*/'+stage+'-report.json')))
  head=read(report['manifest']);entry=copy.deepcopy(next(e for e in head['entries'] if e['id']==identity));value=read(entry['output'])
  output=destination/stage/stage/identity/'result.json';atomic(output,value);entry['output']=str(output)
  head['entries']=[entry];head['limited']=1
  headpath=destination/'manifests'/(stage+'-probe-source.json');atomic(headpath,head)
  state.put_many([(stage,head['model'],identity,entry['signature'],'complete',str(output),None,0)])
  state.set_head(stage,headpath)
finally:state.db.close()
env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['pipeline','vendor','models/funasr/source','models/deepfilternet3/DeepFilterNet']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
command=[sys.executable,str(ROOT/'run.py'),'enhance','--dataset',name,'--config',str(cp),'--resources',str(rp),'--enhance']
start=time.time()
with (CACHE/'chunked-longest.log').open('w') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(CACHE/'chunked-longest-driver.json').write_text(json.dumps(dict(source=sample,identity=identity,start=start,end=time.time(),returncode=result.returncode,command=command,note='Existing vad_chunked_v1 adapter, no inference code changes; segment context reset differs from whole-recording processing. Concurrent Basis ASR on GPUs 0/1/2.'),indent=2)+'\n')
print('returncode',result.returncode,flush=True)
