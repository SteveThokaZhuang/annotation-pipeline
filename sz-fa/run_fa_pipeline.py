#!/usr/bin/env python3
"""Run V3 upstream stages followed by the pure MFA CPU baseline."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
PIPELINE_PYTHON=Path(os.environ.get('PIPELINE_PYTHON','python3'))
MFA_PYTHON=Path(os.environ.get('MFA_PYTHON',str(ROOT/'models/mfa/bin/python')))
STAGES=('ingest','dnsmos','vad','enhance','asr1','asr2','disagreement','asr3','rover')
def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def logged(cmd,log,env):
 t=time.perf_counter()
 with log.open('w') as s:
  s.write('$ '+' '.join(map(str,cmd))+'\n\n'); s.flush(); r=subprocess.run([str(x) for x in cmd],stdout=s,stderr=subprocess.STDOUT,env=env)
 return r.returncode,time.perf_counter()-t
def reports(root,dataset):
 runroot=root/'production'/dataset/'runs'; out={}
 if not runroot.exists(): return []
 for run in sorted((p for p in runroot.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime):
  for p in run.glob('*-report.json'):
   try: x=json.loads(p.read_text())
   except Exception: continue
   out[x.get('stage',p.stem.replace('-report',''))]=x
 return [out[x] for x in STAGES if x in out]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',default='oto_141'); p.add_argument('--output-root',type=Path,default=ROOT/'runs'/'oto141_pure_mfa'); p.add_argument('--skip-upstream',action='store_true'); p.add_argument('--mfa-jobs',type=int,default=32); p.add_argument('--mfa-timeout',type=int,default=14400); a=p.parse_args()
 output=a.output_root.resolve(); output.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false'); commands=[]; timings={}
 upstream=[PIPELINE_PYTHON,PARENT/'run.py',*STAGES,'--dataset',a.dataset,'--config',PARENT/'config/pipeline.v3.oto104_magiczh.yaml','--resources',PARENT/'config/resources.v3.oto104_magiczh.yaml','--datasets',PARENT/'config/datasets.v3.oto104_magiczh.local.yaml','--output-root',output/'upstream_output','--gpus','0,1,2,3','--no-cache','--resume-validation','full']; commands.append([str(x) for x in upstream])
 if not a.skip_upstream:
  code,sec=logged(upstream,output/'upstream_pipeline.log',env); timings['upstream_seconds']=sec
  if code: raise SystemExit(f'upstream pipeline failed with exit code {code}')
  upstream_state={'returncode':code,'seconds':sec,'command':commands[-1]}
 else: upstream_state={'skipped':True,'command':commands[-1]}
 rover=output/'upstream_output'/'production'/a.dataset/'rover'/'recordings'
 if not rover.exists(): raise FileNotFoundError(f'ROVER recording JSON directory missing: {rover}')
 pure=output/'pure_mfa'; cmd=[PIPELINE_PYTHON,ROOT/'src/run_recordings_mfa.py','--recordings',rover,'--output',pure,'--mfa-jobs',str(a.mfa_jobs),'--mfa-python',MFA_PYTHON,'--mfa-timeout',str(a.mfa_timeout)]; commands.append([str(x) for x in cmd]); code,sec=logged(cmd,output/'pure_mfa.log',env); timings['pure_mfa_seconds']=sec
 if code: raise SystemExit(f'pure MFA failed with exit code {code}; see {output}/pure_mfa.log')
 mfa=json.loads((pure/'summary.json').read_text()); report={'status':'complete','method':'pure_mfa_cpu','dataset':a.dataset,'source_root':str(PARENT),'output_root':str(output),'gpu_count':0,'mfa_jobs':a.mfa_jobs,'timings_seconds':timings,'total_seconds':time.perf_counter()-started,'upstream':upstream_state,'upstream_stage_reports':reports(output/'upstream_output',a.dataset),'pure_mfa':mfa,'commands':commands,'code_sha256':hashlib.sha256((ROOT/'src/mfa/process.py').read_bytes()).hexdigest()}; dump(output/'fa_report.json',report); dump(output/'call_stack.json',{'commands':commands,'stages':['run_fa_pipeline.py','parent/run.py -> upstream nine stages','ROVER recordings -> src/run_recordings_mfa.py','src/mfa/process.py -> MFA CPU subprocess','pure_mfa/summary.json -> fa_report.json'],'python':{'pipeline':str(PIPELINE_PYTHON),'mfa':str(MFA_PYTHON),'version':sys.version}}); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
