"""Parallel file discovery and metadata-only source snapshots."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import re
from .common import digest,fingerprint
from .resume import ScanStopped


def _discover_live(cfg,name,stop_file=None):
    policy=cfg.get('datasets',{}).get(name,{})
    workers=policy.get('scan_workers',16)
    if not isinstance(workers,int) or isinstance(workers,bool) or not 1<=workers<=256:
        raise ValueError('scan_workers must be an integer in [1,256]')
    output=Path(cfg['output_root']).resolve()
    source=Path(policy.get('path',str(Path(cfg['dataset_root'])/name))).resolve()
    manifest=Path(policy['manifest']) if policy.get('manifest') else None
    if manifest and policy.get('path'):raise ValueError('Choose dataset path OR manifest')
    def check_stop():
        if stop_file is not None and Path(stop_file).exists():raise ScanStopped()
    def allowed(path):
        return path!=output and output not in path.parents
    if not manifest and (not source.is_dir() or not allowed(source)):
        raise ValueError('Dataset directory missing or overlaps output directory: '+str(source))
    extensions={e.lower() if e.startswith('.') else '.'+e.lower() for e in policy.get('extensions',['.wav','.flac','.ogg'])}
    exclude=policy.get('exclude',[])
    include=policy.get('include',['*'])
    if isinstance(exclude,str) or isinstance(include,str):raise ValueError('include and exclude must be lists of globs')
    def selected(relative):return any(fnmatch(relative,p) for p in include) and not any(fnmatch(relative,p) for p in exclude)
    def walk(root):
        # Do not follow directory symlinks: no cycles or accidentally scanned mounts.
        rows=[]
        for directory,dirs,files in os.walk(root,followlinks=False):
            check_stop()
            if '_directory_snapshots' in cfg:cfg['_directory_snapshots'].append(dict(path=directory,mtime_ns=Path(directory).stat().st_mtime_ns))
            dirs[:]=[d for d in dirs if allowed((Path(directory)/d).resolve())]
            for name in files:
                p=Path(directory)/name;relative=p.relative_to(source).as_posix()
                if p.suffix.lower() in extensions and selected(relative) and allowed(p.resolve()):rows.append((p,relative,None))
        return rows
    def rows(pool):
        if manifest:
            with manifest.open() as stream:
                for line in stream:
                    check_stop()
                    if not line.strip():continue
                    row=json.loads(line)
                    if not isinstance(row,dict) or not isinstance(row.get('path'),str):raise ValueError('Manifest rows require a path string')
                    p=Path(row['path']);p=p if p.is_absolute() else manifest.parent/p
                    p=p.resolve()
                    if not allowed(p):raise ValueError('Manifest references pipeline outputs')
                    yield p,row.get('relative_path',row['path']),row.get('id')
        else:
            roots=[]
            if '_directory_snapshots' in cfg:cfg['_directory_snapshots'].append(dict(path=str(source),mtime_ns=source.stat().st_mtime_ns))
            with os.scandir(source) as entries:
                for entry in entries:
                    check_stop();p=Path(entry.path)
                    if entry.is_dir(follow_symlinks=False) and allowed(p.resolve()):roots.append(p)
                    elif entry.is_file() and p.suffix.lower() in extensions and selected(p.name) and allowed(p.resolve()):yield p,p.name,None
            # Bounded submissions; directory traversal overlaps network metadata I/O.
            pending=deque()
            iterator=iter(sorted(roots))
            while True:
                while len(pending)<workers:
                    root=next(iterator,None)
                    if root is None:break
                    pending.append(pool.submit(walk,root))
                if not pending:break
                yield from pending.popleft().result()
    def snapshot(row):
        check_stop();p,relative,identity=row
        if identity is not None and (not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',identity)):
            raise ValueError('Manifest id must contain only letters, digits, underscores or hyphens')
        return dict(id=identity or digest(relative)[:24],path=str(p.resolve()),relative_path=relative,source_fingerprint=fingerprint(p))
    result=[];identities=set();paths=set()
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='dataset-scan') as scan_pool, ThreadPoolExecutor(max_workers=workers,thread_name_prefix='dataset-stat') as stat_pool:
        pending=deque()
        for row in rows(scan_pool):
            pending.append(stat_pool.submit(snapshot,row))
            if len(pending)>=workers*2:
                result.append(pending.popleft().result())
        result.extend(f.result() for f in pending)
    for item in result:
        if item['id'] in identities or item['path'] in paths:raise ValueError('Duplicate dataset id or source path: '+item['path'])
        identities.add(item['id']);paths.add(item['path'])
    if not result:raise ValueError('Dataset contains no selected files')
    return sorted(result,key=lambda item:item['relative_path'])


def discover(cfg,name,stop_file=None):
    from .catalog import discover_cached
    return discover_cached(cfg,name,stop_file,_discover_live)
