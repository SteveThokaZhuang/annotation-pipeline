"""Persistent membership catalog with directory-change detection and fresh file stats."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from .common import atomic,digest
from .resume import ScanStopped


def discover_cached(cfg,name,stop_file,live):
    options=cfg.get('_discovery',{})
    if not options.get('enabled'):return live(cfg,name,stop_file)
    policy=cfg.get('datasets',{}).get(name,{})
    key=digest([name,policy,cfg['dataset_root'],cfg['output_root']])
    root=Path(options['root']);root.mkdir(parents=True,exist_ok=True);path=root/(key+'.json')
    workers=policy.get('scan_workers',16)
    def stop():
        if stop_file is not None and Path(stop_file).exists():raise ScanStopped()
    def stat_directory(entry):
        stop()
        try:return Path(entry['path']).stat().st_mtime_ns==entry['mtime_ns']
        except FileNotFoundError:return False
    def manifest_identity():
        if not policy.get('manifest'):return None
        p=Path(policy['manifest']);s=p.stat();return [str(p),s.st_size,s.st_mtime_ns]
    document=None
    if path.exists():
        try:document=json.loads(path.read_text())
        except (OSError,ValueError):pass
    if document and document.get('manifest')==manifest_identity():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            unchanged=all(pool.map(stat_directory,document['directories']))
            if unchanged:
                def refresh(item):
                    stop();s=Path(item['path']).stat()
                    return dict(item,source_fingerprint=dict(path=item['path'],size=s.st_size,mtime_ns=s.st_mtime_ns))
                try:items=list(pool.map(refresh,document['items']))
                except FileNotFoundError:items=None
                if items is not None:
                    print(json.dumps(dict(event='catalog',dataset=name,status='validated',files=len(items))),flush=True)
                    return items
    for _ in range(options.get('max_refresh_attempts',1)):
        directories=[];local=dict(cfg,_directory_snapshots=directories)
        before=manifest_identity();items=live(local,name,stop_file)
        with ThreadPoolExecutor(max_workers=workers) as pool:consistent=all(pool.map(stat_directory,directories))
        if consistent and before==manifest_identity():
            atomic(path,dict(schema=1,manifest=before,directories=directories,items=items))
            print(json.dumps(dict(event='catalog',dataset=name,status='refreshed',files=len(items))),flush=True)
            return items
    raise RuntimeError('Dataset membership changed during discovery; no catalog was published')
