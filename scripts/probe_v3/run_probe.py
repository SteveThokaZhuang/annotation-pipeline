"""Cache-only real inference probe using v3 code and inherited stage policies."""
import json,os,subprocess,time,threading,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
PYTHON='/hbt_share02/202610/ob/ob_ppl_env_data/envs/firered-vad/bin/python'
stop=threading.Event();context={}
def monitor():
    with (CACHE/'telemetry.jsonl').open('a') as out:
        while not stop.is_set():
            row=dict(time=time.time(),**context)
            try:
                row['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                row['cpu_stat']=Path('/sys/fs/cgroup/cpu.stat').read_text()
                row['memory_current']=int(Path('/sys/fs/cgroup/memory.current').read_text())
            except Exception as e:row['monitor_error']=str(e)
            out.write(json.dumps(row)+'\n');out.flush();stop.wait(2)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
results=[]
try:
    for dataset in sys.argv[1:] or ['duplexconv','basis-conversations-1500-en']:
        context.update(dataset=dataset,mode='baseline-enhanced')
        command=[PYTHON,str(ROOT/'run.py'),'all','--dataset',dataset,'--config',str(ROOT/f'config/pipeline.probe-v3-{dataset}.local.yaml'),'--resources',str(ROOT/f'config/resources.probe-v3-{dataset}.local.yaml'),'--enhance']
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['pipeline','vendor','models/funasr/source','models/deepfilternet3/DeepFilterNet']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(CACHE/'scratch'),HF_HOME=str(CACHE/'hf'),VLLM_CACHE_ROOT=str(CACHE/'vllm_cache'),TORCHINDUCTOR_CACHE_DIR=str(CACHE/'torchinductor'),TRITON_CACHE_DIR=str(CACHE/'triton'))
        (CACHE/'scratch').mkdir(exist_ok=True)
        start=time.time()
        with (CACHE/(dataset+'-baseline.log')).open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            context.update(pid=proc.pid)
            print(json.dumps(dict(event='start',dataset=dataset,pid=proc.pid,command=command)),flush=True)
            result=proc.wait()
        results.append(dict(dataset=dataset,returncode=result,wall_seconds=time.time()-start,command=command))
        (CACHE/'driver-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
finally:
    stop.set();thread.join(6)
