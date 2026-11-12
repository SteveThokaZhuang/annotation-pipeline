import os,json,random,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
base=Path('/hbt_share02/202610/ob/ob_ppl_env_data/dataset/data_processing')
out=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
rng=random.Random(20261106)
for name,sub,cap in [('duplexconv','duplexconv',3000),('basis-conversations-1500-en','basis-conversations-1500/en/audio',10000)]:
    start=time.time(); paths=[]
    for directory,dirs,files in os.walk(base/sub):
        dirs.sort()
        for f in sorted(files):
            if Path(f).suffix.lower() in {'.wav','.flac','.ogg'}:paths.append(str(Path(directory)/f))
        if len(paths)>=cap:break
    rng.shuffle(paths);paths=paths[:cap]
    def header(p):
        try:
            t=time.perf_counter();info=sf.info(p)
            return dict(path=p,duration=info.duration,sample_rate=info.samplerate,channels=info.channels,frames=info.frames,bytes=Path(p).stat().st_size,header_seconds=time.perf_counter()-t)
        except Exception as e:return dict(path=p,error=str(e))
    with ThreadPoolExecutor(max_workers=16) as pool: rows=list(pool.map(header,paths))
    valid=sorted([r for r in rows if 'error' not in r],key=lambda r:r['duration'])
    if len(valid)<100:raise RuntimeError(f'{name}: only {len(valid)} files')
    # Systematic quantiles, including observed shortest and longest.
    samples=[valid[round(i*(len(valid)-1)/99)] for i in range(100)]
    (out/(name+'.jsonl')).write_text(''.join(json.dumps(dict(path=r['path'],id=f'sample_{i:03d}'))+'\n' for i,r in enumerate(samples)))
    result=dict(dataset=name,candidate_scope=sub,candidate_cap=cap,candidates=len(rows),header_errors=[r for r in rows if 'error' in r],selection='100 evenly spaced duration ranks including both extremes; fixed seed 20261106',samples=samples,survey=rows,total_audio_seconds=sum(r['duration'] for r in samples),over_1800=sum(r['duration']>1800 for r in samples),over_3600=sum(r['duration']>3600 for r in samples),survey_seconds=time.time()-start)
    (out/(name+'-sample-info.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'samples','survey','header_errors'}},indent=2),flush=True)
