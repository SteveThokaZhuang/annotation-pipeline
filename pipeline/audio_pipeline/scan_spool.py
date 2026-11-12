"""Read each shared result file once; restore task order from disposable local I/O."""
from array import array
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
from pathlib import Path
import tempfile
import time
from .failure_policy import ASR_STAGES,hypothesis_error


def grouped_tasks(state,stage,model,tasks,workers,stop_file,phase,window,scratch,validate_results=True):
    from .resume import _check_file,ScanStopped
    started=time.monotonic();last_log=started;next_stop=0.
    checked=cached=files_read=delivered=0
    def stop(force=False):
        nonlocal next_stop
        now=time.monotonic()
        if force or now>=next_stop:
            next_stop=now+.1
            if stop_file is not None and stop_file.exists():raise ScanStopped()
    def progress(step,force=False):
        nonlocal last_log
        now=time.monotonic()
        if force or now-last_log>=10:
            elapsed=now-started
            print(json.dumps(dict(stage=stage,event='progress',phase=phase,step=step,
                checked=checked,total=len(tasks),cached=cached,scan_workers=workers,
                result_files_read=files_read,results_delivered=delivered,
                percent=round(100*(checked+delivered)/(2*len(tasks)),2) if tasks else 100.,
                validation_percent=round(100*checked/len(tasks),2) if tasks else 100.,
                elapsed_seconds=round(elapsed,1),items_per_second=round(checked/max(elapsed,.001),1))),flush=True)
            last_log=now
    stop(True);progress('group_files',True)
    rows=[];files={}
    for offset in range(0,len(tasks),window):
        stop()
        group=tasks[offset:offset+window]
        keys=[(task['item']['id'],task['signature']) for task in group]
        found=state.task_rows(stage,model,keys)
        for position,key in enumerate(keys,offset):
            row=found.get(key);rows.append(row)
            if row is None or row[0]!='complete' or not row[1]:continue
            base,separator,index=row[1].rpartition('#result=')
            if separator and not index.isdecimal():continue
            path=base if separator else row[1]
            files.setdefault(path,[]).append((position,int(index) if separator else None))
        progress('group_files')
    checked=len(tasks)-sum(map(len,files.values()))
    offsets=array('q',[-1])*len(tasks);sizes=array('Q',[0])*len(tasks)
    Path(scratch).mkdir(parents=True,exist_ok=True)
    active={};pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='batch-scan')
    try:
        # No fsync: this is disposable validation scratch, never a checkpoint.
        with tempfile.TemporaryFile(prefix='result-scan-',dir=scratch) as spool:
            pending=iter(files.items());exhausted=False
            while active or not exhausted:
                stop()
                while not exhausted and len(active)<workers*2:
                    try:path,entries=next(pending)
                    except StopIteration:exhausted=True;break
                    active[pool.submit(_check_file,path,entries,validate_results)]=len(entries)
                if active:
                    done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        stop()
                        values=future.result();checked+=active.pop(future);files_read+=1
                        for position,value in values.items():
                            stop()
                            data=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                            offsets[position]=spool.tell();sizes[position]=len(data);spool.write(data)
                            cached+=not(stage in ASR_STAGES and hypothesis_error(value))
                        del values
                progress('read_batches')
            spool.flush();stop(True)
            # No shared-file reread or parsing of unrelated rows in this pass.
            for offset in range(0,len(tasks),window):
                for position,task in enumerate(tasks[offset:offset+window],offset):
                    stop();value=None
                    if offsets[position]>=0:
                        spool.seek(offsets[position]);data=spool.read(sizes[position])
                        if len(data)!=sizes[position]:raise IOError('Incomplete local validation spool')
                        value=json.loads(data)
                    delivered+=1
                    yield task,rows[position],value
                    progress('ordered_results')
            stop(True);progress('complete',True)
    finally:
        for future in active:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
