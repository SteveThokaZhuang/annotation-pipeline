"""Copy immutable completed upstream checkpoints into an isolated profile.

Only ingest/DNSMOS/VAD are eligible. Source state is opened read-only, and cached
artifacts are preferred. Never borrow enhancement/ASR heads or modify the source.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import tempfile
import time
import uuid

from .common import atomic,digest
from .state import State
from .working_cache import canonical_path

STAGES=('ingest','dnsmos','vad')


def seed(source_root,destination_root,stop_requested=lambda:False):
    source_root=Path(source_root).absolute();destination_root=Path(destination_root).absolute()
    if str(source_root)==canonical_path(destination_root):
        raise ValueError('Upstream reuse requires a separate destination output')
    source_cache=None
    marker=source_root/'.working-cache.lock'
    if marker.is_file():
        home=marker.read_text().strip()
        if home:
            candidate=Path(home)/'files'
            if candidate.is_dir():source_cache=candidate
    def relative(value):
        p=Path(value)
        for base in (source_cache,source_root):
            if base is not None:
                try:
                    result=p.relative_to(base)
                    if '..' in result.parts:raise ValueError('Unsafe upstream output path')
                    return result
                except ValueError:pass
        raise ValueError('Upstream artifact is outside the selected source dataset: '+str(p))
    def locate(value):
        rel=relative(value)
        if source_cache is not None and (source_cache/rel).is_file():return source_cache/rel
        return source_root/rel
    database=(source_cache/'state.sqlite') if source_cache is not None and (source_cache/'state.sqlite').is_file() else source_root/'state.sqlite'
    if not database.is_file():return []
    destination=State(destination_root);imported=[];copied=set()
    try:
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as source:
            for stage in STAGES:
                if stop_requested():raise InterruptedError('Stopped during upstream reuse')
                if destination.db.execute('SELECT 1 FROM heads WHERE stage=?',(stage,)).fetchone():continue
                row=source.execute('SELECT manifest FROM heads WHERE stage=?',(stage,)).fetchone()
                if row is None:continue
                manifest=json.loads(locate(row[0]).read_text())
                if manifest['stage']!=stage:raise ValueError('Upstream manifest stage mismatch')
                if manifest.get('limited') is not None:
                    # Do not seed limited test heads into an unrestricted run.
                    continue
                manifest=copy.deepcopy(manifest);rows=[]
                for entry in manifest['entries']:
                    if stop_requested():raise InterruptedError('Stopped during upstream reuse')
                    output=entry.get('output')
                    if output:
                        base,separator,index=output.rpartition('#result=')
                        if not separator:base=output
                        elif not index.isdecimal():raise ValueError('Invalid upstream batch reference')
                        rel=relative(base);target=destination_root/rel
                        if rel not in copied:
                            original=locate(base);target.parent.mkdir(parents=True,exist_ok=True)
                            fd,name=tempfile.mkstemp(prefix='.reuse-',dir=target.parent)
                            try:
                                with os.fdopen(fd,'wb') as out,original.open('rb') as inp:
                                    while True:
                                        if stop_requested():raise InterruptedError('Stopped during upstream copy')
                                        block=inp.read(8*1024*1024)
                                        if not block:break
                                        out.write(block)
                                    out.flush();os.fsync(out.fileno())
                                shutil.copystat(original,name)
                                os.replace(name,target)
                            finally:Path(name).unlink(missing_ok=True)
                            copied.add(rel)
                        output=canonical_path(target)+('#result='+index if separator else '')
                        entry['output']=output
                    rows.append((stage,manifest['model'],entry['id'],entry['signature'],entry['status'],output,entry.get('error'),0))
                target=destination_root/'manifests'/f'{stage}-{digest(manifest)[:20]}.json'
                atomic(target,manifest)
                destination.put_many(rows)
                destination.set_head(stage,target)
                imported.append(stage)
                print(json.dumps(dict(event='upstream_reused',stage=stage,entries=len(rows),source=str(source_root),destination=canonical_path(destination_root))),flush=True)
    finally:destination.db.close()
    return imported


def main():
    from .configuration import load_config
    from .working_cache import WorkingCache
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output-root',type=Path,required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resources',type=Path,required=True)
    args=p.parse_args()
    cfg,resources=load_config(args.config,args.resources)
    if args.dataset not in cfg['datasets']:p.error('Dataset is not in the selected profile')
    source=args.source_output_root/'production'/args.dataset
    destination=Path(cfg['output_root'])/'production'/args.dataset
    settings=resources.get('optimization',{}).get('working_cache',{})
    def transfer(root,working=None):
        run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-reuse-'+uuid.uuid4().hex[:8]
        run_dir=Path(root)/'runs'/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        stop_file=run_dir/'STOP'
        def stop(*_):stop_file.touch()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        control_root=working.home if working is not None and not working.sync_enabled else Path(cfg['output_root'])
        registration=control_root/'control/runs'/(run_id+'.json')
        value=dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
                   dataset=args.dataset,mode='production',stages=['reuse_upstream'],status='running',
                   output_root=str(Path(cfg['output_root']).absolute()),
                   stop_file=str(stop_file),run_dir=str(run_dir),started=time.time())
        atomic(registration,value);status='failed'
        try:
            seed(source,root,stop_file.exists)
            if working is not None:working.check_health()
            if stop_file.exists():raise InterruptedError('Stopped after upstream reuse')
            status='passed'
        except InterruptedError:
            status='stopped';raise SystemExit(130)
        finally:
            value.update(status=status,finished=time.time());atomic(registration,value)
    if settings.get('enabled'):
        with WorkingCache(destination,settings) as working:transfer(working.cache,working)
    else:transfer(destination)


if __name__=='__main__':main()
