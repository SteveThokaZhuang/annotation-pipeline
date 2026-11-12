import os,sys,json,time
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parents[2];CACHE=Path('/tmp/shared-storage/audio-pipeline-github-v3-probe')
sys.path.insert(0,str(ROOT/'pipeline'))
from audio_pipeline.common import read
from audio_pipeline.audio import audio_source
os.environ['AUDIO_PIPELINE_CACHE']=json.dumps(dict(enabled=True,root=str(CACHE/'source_cache'),max_gib=6144))
report=next(r for r in (read(p) for p in (CACHE/'baseline/production/basis-conversations-1500-en/runs').glob('*/vad-report.json')) if r.get('completed')==100 and r.get('cached')==0)
rows=[]
for entry in read(report['manifest'])['entries']:
 value=read(entry['output'])
 if value['chunks']:continue
 source=value['_provenance']['input'];count=0;squares=0.;peak=0.;nonzero=0
 start=time.time()
 with audio_source(source['path']) as local,sf.SoundFile(local) as stream:
  for block in stream.blocks(blocksize=48000*10,dtype='float64',always_2d=True):
   count+=block.size;squares+=float(np.sum(block*block));peak=max(peak,float(np.max(np.abs(block))));nonzero+=int(np.count_nonzero(block))
 row=dict(id=entry['id'],path=source['path'],duration=source['duration'],samples=count,nonzero_samples=nonzero,peak=peak,rms=(squares/count)**.5,scan_seconds=time.time()-start)
 rows.append(row);print(json.dumps(row),flush=True)
(CACHE/'no-speech-audio-check.json').write_text(json.dumps(rows,indent=2)+'\n')
