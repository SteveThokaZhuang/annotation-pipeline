"""Inventory production ROVER batches and partition exact source segments."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time


def canonical(item, source):
    c=item['chunk']
    sr=int(c['sample_rate'])
    start=int(c.get('start_sample',round(c['start']*sr)))
    end=int(c.get('end_sample',round(c['end']*sr)))
    if sr<=0 or start<0 or end<=start:
        raise ValueError('Invalid source sample bounds: '+str(c.get('id')))
    input_language=next((h.get('language') for h in (item.get('hypotheses') or [])
                         if isinstance(h,dict) and h.get('language')), None)
    language=input_language or 'English'
    return dict(id=c['id'],audio=c.get('audio_path') or c['path'],source_audio=c.get('audio_path') or c['path'],
                audio_scope='recording',text=item.get('text',''),language=language,
                recording_id=c['recording_id'],channel=int(c.get('channel',0)),
                source_sample_rate=sr,start_sample=start,end_sample=end,
                chunk_start=start/sr,chunk_end=end/sr,duration=(end-start)/sr,
                source_json=str(source),rover_signature=(item.get('_provenance') or {}).get('signature'),
                input_language=input_language,
                risk_audio=dict(path=c['path'], sample_rate=sr, start_sample=start, end_sample=end,
                                channel=int(c.get('channel',0))) if c.get('path') else None,
                hypotheses=[{k:h[k] for k in ('text','language','source_result') if k in h}
                            for h in (item.get('hypotheses') or []) if isinstance(h,dict)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--index',type=Path,default=Path('/tmp')/f'fa_rover_inventory_{os.getpid()}.sqlite')
    p.add_argument('--reuse-index',action='store_true',help='Re-export an already complete, unique inventory after an output IO failure')
    a=p.parse_args()
    started=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False)
    paths=sorted(a.input.glob('*/*.json'))
    if not paths: raise ValueError('No ROVER batch JSON files')
    # Random SQLite writes belong on local scratch, not the shared data mount.
    database=sqlite3.connect(a.index)
    database.executescript('PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS rows (id TEXT PRIMARY KEY, audio TEXT, start INTEGER, duration REAL, data TEXT, mtime INTEGER);')
    existing_sources=(dict(database.execute("SELECT json_extract(data,'$.source_json'),COUNT(*) FROM rows GROUP BY json_extract(data,'$.source_json')"))
                      if a.reuse_index else {})
    if a.reuse_index and set(existing_sources)!={str(path) for path in paths}:
        raise ValueError('Reused index does not cover exactly the requested source files')
    total=duplicates=conflicts=bad=0
    sources=[]
    with (a.output/'invalid_inputs.jsonl').open('w') as invalid, (a.output/'duplicate_conflicts.jsonl').open('w') as conflict_log:
        for file_index,path in enumerate(paths):
            raw=path.read_bytes(); values=[] if a.reuse_index else json.loads(raw)['results']; stamp=path.stat().st_mtime_ns
            items=existing_sources[str(path)] if a.reuse_index else len(values)
            sources.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),items=items))
            if a.reuse_index:total+=items
            for item in values:
                total+=1
                try:
                    row=canonical(item,path)
                    payload=json.dumps(row,ensure_ascii=False,separators=(',',':'))
                    try:
                        database.execute('INSERT INTO rows VALUES (?,?,?,?,?,?)',(row['id'],row['audio'],row['start_sample'],row['duration'],payload,stamp))
                    except sqlite3.IntegrityError:
                        duplicates+=1
                        previous,previous_stamp=database.execute('SELECT data,mtime FROM rows WHERE id=?',(row['id'],)).fetchone()
                        before=json.loads(previous)
                        fields=['text','audio','start_sample','end_sample','source_sample_rate','channel']
                        if any(before[k]!=row[k] for k in fields):
                            conflicts+=1
                            conflict_log.write(json.dumps(dict(id=row['id'],previous=before,new=row,selection='latest source file mtime'),ensure_ascii=False)+'\n')
                            if stamp>previous_stamp:
                                database.execute('UPDATE rows SET audio=?,start=?,duration=?,data=?,mtime=? WHERE id=?',
                                    (row['audio'],row['start_sample'],row['duration'],payload,stamp,row['id']))
                except Exception as exc:
                    bad+=1
                    invalid.write(json.dumps(dict(source_json=str(path),id=item.get('chunk',{}).get('id'),error=repr(exc)),ensure_ascii=False)+'\n')
            database.commit()
            if (file_index+1)%100==0 or file_index+1==len(paths):
                print(f'Inventory {file_index+1}/{len(paths)} files; rows={total}; duplicates={duplicates}',flush=True)
    audio_groups=list(database.execute('SELECT audio,SUM(duration),COUNT(*) FROM rows GROUP BY audio'))
    gpu_ids=a.gpus.split(',')
    loads=[0.0]*len(gpu_ids)
    assignments={}
    missing=[]
    for audio,duration,count in sorted(audio_groups,key=lambda x:-x[1]):
        if not Path(audio).is_file(): missing.append(dict(audio=audio,chunks=count,seconds=duration))
        slot=min(range(len(loads)),key=loads.__getitem__)
        assignments[audio]=slot;loads[slot]+=duration
    streams=[(a.output/f'gpu_{gpu}.jsonl').open('w') for gpu in gpu_ids]
    counts=[0]*len(gpu_ids);languages=collections.Counter();empty=0;maximum=0
    try:
        for audio,payload in database.execute('SELECT audio,data FROM rows ORDER BY audio,start,id'):
            row=json.loads(payload);slot=assignments[audio]
            streams[slot].write(payload+'\n');counts[slot]+=1
            languages[row['language']]+=1
            empty+=not row['text'].strip();maximum=max(maximum,row['duration'])
    finally:
        for stream in streams:stream.close()
    manifest_hashes={}
    for gpu in gpu_ids:
        path=a.output/f'gpu_{gpu}.jsonl'
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        manifest_hashes[gpu]=digest.hexdigest()
    summary=dict(input=str(a.input.resolve()),files=len(paths),raw_rows=total,
        unique_chunks=sum(counts),duplicate_rows=duplicates,conflicting_duplicates=conflicts,
        invalid_rows=bad,missing_audio=missing,recording_channels=len(audio_groups),
        audio_seconds=sum(loads),max_chunk_seconds=maximum,empty_text_chunks=empty,languages=dict(languages),
        shards=[dict(gpu=gpu,chunks=count,audio_seconds=duration,manifest_sha256=manifest_hashes[gpu])
                for gpu,count,duration in zip(gpu_ids,counts,loads)],
        seconds=time.perf_counter()-started,denominator='sum of unique annotated source segment durations; no silence gaps or channel deduplication',
        timing_scope='re-export of a complete existing index' if a.reuse_index else 'complete source scan, index and shard export',
        source_batches=sources)
    database.close()
    shutil.copyfile(a.index,a.output/'index.sqlite')
    summary['seconds']=time.perf_counter()-started
    (a.output/'preparation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_batches'},ensure_ascii=False,indent=2),flush=True)
    if missing or bad:raise RuntimeError('Input issues recorded; resolve before claiming a complete run')


if __name__=='__main__':main()
