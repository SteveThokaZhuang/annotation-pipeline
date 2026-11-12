"""Bounded concurrent validation of saved results; SQLite stays on the controller."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import time
import tempfile

from .common import read, validate
from .failure_policy import ASR_STAGES, hypothesis_error


class ScanStopped(Exception):
    pass


def _check_file(path, entries, validate_results=True):
    values = {}
    try:
        saved = read(path)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return values
    for key, index in entries:
        try:
            value = saved if index is None else saved['results'][index]
            if validate_results:validate(value)
            elif not isinstance(value,dict):continue
            values[key] = value
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            pass
    return values


# One compact entry per requested task; large scans avoid random SQLite reads.
RESULT_INDEX_MIN_TASKS = 16384
FILE_GROUP_MIN_TASKS = 16384


def stage_row_index(state, stage, model, tasks, stop_file=None):
    """Snapshot committed rows in one sequential scan before/after workers run.

    Only requested IDs/signatures are retained. Never reused across scan calls,
    so newly committed results are visible during finalization and later resumes.
    """
    wanted={task['item']['id']:(task['signature'],None) for task in tasks}
    start=time.monotonic();last=start;scanned=0
    for item,signature,status,output,error in state.db.execute(
            'SELECT item,signature,status,output,error FROM tasks WHERE stage=? AND model=?',
            (stage,model)):
        expected=wanted.get(item)
        if expected is not None and expected[0]==signature:
            wanted[item]=(signature,(status,output,error))
        scanned+=1
        if scanned%8192==0:
            if stop_file is not None and stop_file.exists():raise ScanStopped()
            now=time.monotonic()
            if now-last>=10:
                print(json.dumps(dict(stage=stage,event='progress',phase='result_index',
                    scanned=scanned,elapsed_seconds=round(now-start,1))),flush=True)
                last=now
    if stop_file is not None and stop_file.exists():raise ScanStopped()
    print(json.dumps(dict(stage=stage,event='progress',phase='result_index',
        scanned=scanned,elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
    class Index:
        def task_rows(self,_stage,_model,keys):
            result={}
            for key in keys:
                expected=wanted.get(key[0])
                if expected is not None and expected[0]==key[1] and expected[1] is not None:
                    result[key]=expected[1]
            return result
    return Index()


def checked_tasks(state, stage, model, tasks, workers=16, stop_file=None,
                  phase='resume_scan', window=8192, validate_results=True):
    """Yield (task, database row, validated result) in task order.

    Large scans group files across the full scan and restore task order from
    disposable local scratch. Small scans use a bounded in-memory window.
    Reader threads never touch SQLite; queued reads are bounded.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('resume_scan_workers must be a positive integer')
    started = time.monotonic()
    checked = cached = 0
    last_log = started

    def progress(force=False):
        nonlocal last_log
        now = time.monotonic()
        if force or now - last_log >= 10:
            elapsed = now - started
            print(json.dumps(dict(stage=stage, event='progress', phase=phase,
                checked=checked, total=len(tasks), cached=cached, scan_workers=workers,
                percent=round(100 * checked / len(tasks), 2) if tasks else 100.,
                elapsed_seconds=round(elapsed, 1),
                items_per_second=round(checked / max(elapsed, .001), 1))), flush=True)
            last_log = now

    next_stop_poll=0.
    def check_stop(force=False):
        nonlocal next_stop_poll
        now=time.monotonic()
        if not force and now<next_stop_poll:return
        next_stop_poll=now+.1
        if stop_file is not None and stop_file.exists():
            raise ScanStopped()

    check_stop()
    scratch=getattr(state,'scan_scratch',tempfile.gettempdir())
    if len(tasks)>=RESULT_INDEX_MIN_TASKS and hasattr(state,'db'):
        state=stage_row_index(state,stage,model,tasks,stop_file)
    if len(tasks)>=FILE_GROUP_MIN_TASKS:
        from .scan_spool import grouped_tasks
        yield from grouped_tasks(state,stage,model,tasks,workers,stop_file,phase,window,scratch,validate_results)
        return
    progress(True)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='result-scan')
    active = {}
    try:
        for offset in range(0, len(tasks), window):
            check_stop()
            group = tasks[offset:offset + window]
            keys = [(task['item']['id'], task['signature']) for task in group]
            rows = state.task_rows(stage, model, keys)
            files = {}
            for key, row in rows.items():
                if row[0] != 'complete' or not row[1]:
                    continue
                base, separator, index = row[1].rpartition('#result=')
                if separator and not index.isdecimal():
                    continue
                path = base if separator else row[1]
                files.setdefault(path, []).append((key, int(index) if separator else None))
            checked += len(group) - sum(len(entries) for entries in files.values())
            remaining = iter(files.items())
            values = {}
            exhausted = False
            while active or not exhausted:
                check_stop()
                while not exhausted and len(active) < workers * 2:
                    try:
                        path, entries = next(remaining)
                    except StopIteration:
                        exhausted = True
                        break
                    active[pool.submit(_check_file, path, entries, validate_results)] = len(entries)
                if active:
                    done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        found = future.result()
                        checked += active.pop(future)
                        cached += sum(not (stage in ASR_STAGES and hypothesis_error(value))
                                      for value in found.values())
                        values.update(found)
                progress()
            for task, key in zip(group, keys):
                check_stop()
                yield task, rows.get(key), values.get(key)
        check_stop(force=True)
        progress(True)
    finally:
        for future in active:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def checked_entries(entries, stage, workers=16, stop_file=None, validate_results=True):
    """Use the same bounded reader for upstream manifests without copying them."""
    class Tasks:
        def __len__(self):return len(entries)
        def __getitem__(self, selection):
            return [dict(item=dict(id=index),signature='',entry=entries[index])
                    for index in range(*selection.indices(len(entries)))]

    class Rows:
        def task_rows(self, _stage, _model, keys):
            return {key:(entries[key[0]]['status'],entries[key[0]].get('output'),
                         entries[key[0]].get('error')) for key in keys}

    for task, _, value in checked_tasks(Rows(),stage,'manifest',Tasks(),workers=workers,
                                        stop_file=stop_file,phase='upstream_scan',validate_results=validate_results):
        yield task['entry'],value


def committed_tasks(state,stage,model,tasks,stop_file=None,window=8192):
    """Trust durable SQLite completion records without opening result files."""
    started=time.monotonic();cached=0
    index=stage_row_index(state,stage,model,tasks,stop_file)
    for offset in range(0,len(tasks),window):
        if stop_file is not None and stop_file.exists():raise ScanStopped()
        group=tasks[offset:offset+window]
        keys=[(task['item']['id'],task['signature']) for task in group]
        rows=index.task_rows(stage,model,keys)
        for task,key in zip(group,keys):
            row=rows.get(key)
            cached+=bool(row and row[0]=='complete' and row[1])
            yield task,row
    if stop_file is not None and stop_file.exists():raise ScanStopped()
    print(json.dumps(dict(stage=stage,event='progress',phase='resume_index',validation='trusted',
        checked=len(tasks),total=len(tasks),cached=cached,result_files_read=0,
        elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
