"""Stage a private corpus and run MFA; never modify the source dataset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,DATASET,load_manifest,write_jsonl

def main():
    process_start=time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=DATASET/'manifest.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--beam',type=int,default=400)
    p.add_argument('--retry-beam',type=int,default=4000)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--export-only',action='store_true',help='Export existing TextGrids without rerunning alignment')
    a=p.parse_args(); rows=load_manifest(a.manifest)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    corpus=out/'corpus'
    for r in ([] if a.export_only else rows):
        folder=corpus/r['id']; folder.mkdir(parents=True,exist_ok=True)
        wav=folder/(r['id']+'.wav')
        if not wav.exists(): wav.symlink_to(r['audio'])
        (folder/(r['id']+'.lab')).write_text(r['text']+'\n')
    models=ROOT/'models/Montreal-Forced-Aligner/checkpoints'
    acoustic=models/'english_mfa.zip'; dictionary=models/'english_us_mfa.dict'
    cmd=[str(Path(sys.executable).parent/'mfa'),'align',str(corpus),str(dictionary),str(acoustic),str(out/'aligned'),'--beam',str(a.beam),'--retry_beam',str(a.retry_beam),'--num_jobs',str(a.jobs),'--temporary_directory',str(out/'work'),'--clean','--overwrite','--use_threading','--single_speaker']
    start=time.perf_counter()
    env=dict(os.environ)
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    env['MFA_ROOT_DIR']=str(ROOT/'models/Montreal-Forced-Aligner/cache')
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    log_path=out/('align-'+str(time.time_ns())+'.log')
    alignment_start=time.perf_counter()
    if not a.export_only:
        with log_path.open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
    alignment_subprocess_seconds=time.perf_counter()-alignment_start
    export_start=time.perf_counter()
    from praatio import textgrid
    preds=[]
    grid_index={p.stem:p for p in (out/'aligned').rglob('*.TextGrid')}
    for r in rows:
        grid_path=grid_index.get(r['id'])
        if grid_path is None:
            preds.append(dict(id=r['id'],words=[],error='missing TextGrid')); continue
        tg=textgrid.openTextgrid(str(grid_path),includeEmptyIntervals=False)
        name=next(n for n in tg.tierNames if n=='words' or n.endswith(' - words'))
        words=[dict(text=e.label,start=e.start,end=e.end) for e in tg.getTier(name).entries if e.label not in {'','sil','sp','<eps>'}]
        preds.append(dict(id=r['id'],words=words,backend='mfa'))
    write_jsonl(out/'predictions.jsonl',preds)
    runtime_path=out/('export-runtime.json' if a.export_only else 'runtime.json')
    runtime_path.write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        worker_seconds=time.perf_counter()-process_start,
        alignment_subprocess_seconds=alignment_subprocess_seconds,
        export_seconds=time.perf_counter()-export_start,
        staging_seconds=start-process_start,
        alignment_scope='fresh MFA CLI process including imports, model loading, corpus processing and TextGrid export; NOT warm inference',
        threads=dict(jobs=a.jobs,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1),
        cuda_visible_devices='',command=cmd,command_executed=not a.export_only,log=str(log_path) if not a.export_only else None,speaker_normalization='one utterance per speaker directory; adaptation disabled'),indent=2))
if __name__=='__main__': main()
