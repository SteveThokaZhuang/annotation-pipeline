"""Optional bounded local source-file cache, with leases and atomic publication.

Leases prevent eviction of files in use by another worker. Originals are never
modified. Oversized files or a cache full of leased files use the source directly.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from .cache_index import inventory,reserve,remove,touch,forget_missing


@contextmanager
def cached_source(path,root,max_bytes):
    from .working_cache import mapping,preferred_path
    source=Path(preferred_path(path))
    working=mapping()
    if working and Path(working['cache']) in source.parents:
        yield source;return
    info=source.stat()
    if info.st_size>max_bytes:
        yield source;return
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    marker=root/'.audio-pipeline-cache-v1'
    if not marker.exists():
        if any(p.name!=marker.name for p in root.iterdir()) and not marker.exists():
            raise ValueError('Cache root must be empty or an existing managed cache')
        try:
            with marker.open('x') as stream:stream.write('Managed local source cache v1\n')
        except FileExistsError:pass
    identity=json.dumps([str(source.resolve()),info.st_size,info.st_mtime_ns])
    key=hashlib.sha256(identity.encode()).hexdigest()
    target=root/(key+source.suffix);lock=root/(key+'.lease')
    # Lock order: file lease first, then a short metadata lock. Eviction only
    # attempts nonblocking leases, so it cannot deadlock with readers/copiers.
    with (root/'cache.lock').open('a+') as global_lock,lock.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_SH)
        with _locked(global_lock):
            hit=target.exists()
            if hit:
                os.utime(target,None);touch(root,target,info.st_size)
            else:forget_missing(root,target)
        if hit:
            yield target;return
        fcntl.flock(lease,fcntl.LOCK_UN)
        # Only requests for this source wait for its copy. Unrelated cache hits
        # and different source copies continue while shared-storage I/O runs.
        fcntl.flock(lease,fcntl.LOCK_EX)
        with _locked(global_lock):
            hit=target.exists()
            if hit:
                os.utime(target,None);touch(root,target,info.st_size)
            else:forget_missing(root,target)
        if not hit:
            if shutil.disk_usage(root).free<info.st_size*1.1:
                yield source;return
            temporary=root/(key+f'.partial-{os.getpid()}')
            try:
                shutil.copyfile(source,temporary)
                after=source.stat()
                if (info.st_size,info.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise RuntimeError('Source changed while staging')
                if temporary.stat().st_size!=info.st_size:
                    raise RuntimeError('Incomplete local cache copy')
                with _locked(global_lock):
                    # Recheck the shared budget at publication, after concurrent
                    # copies finish. Partial files are temporary staging space.
                    hit=_make_room(root,max_bytes,info.st_size)
                    if hit:
                        # Commit a reservation before publication. A killed copy
                        # may leave a stale reservation, which is repaired safely.
                        with inventory(root) as connection:reserve(connection,target.name,info.st_size)
                        os.replace(temporary,target)
            finally:temporary.unlink(missing_ok=True)
        if hit:
            fcntl.flock(lease,fcntl.LOCK_SH)
            yield target
        else:
            yield source


@contextmanager
def _locked(handle):
    fcntl.flock(handle,fcntl.LOCK_EX)
    try:yield
    finally:fcntl.flock(handle,fcntl.LOCK_UN)


def _make_room(root,max_bytes,needed):
    with inventory(root) as connection:
        used=connection.execute('SELECT bytes FROM usage WHERE id=1').fetchone()[0]
        if used+needed<=max_bytes:return True
        # Fetch candidates in small batches. Never scan the entire cache on a
        # hit or a miss when it is below budget, even for million-file datasets.
        cursor=None
        while used+needed>max_bytes:
            rows=connection.execute('SELECT name,size,accessed FROM files '+
                ('WHERE (accessed,name)>(?,?) ' if cursor else '')+
                'ORDER BY accessed,name LIMIT 128',cursor or ()).fetchall()
            if not rows:break
            for name,size,accessed in rows:
                cursor=(accessed,name)
                old=root/name
                with (root/(old.stem+'.lease')).open('a+') as lease:
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    old.unlink(missing_ok=True)
                    remove(connection,name);used-=size
                if used+needed<=max_bytes:break
        return used+needed<=max_bytes


def seed_source(source,local_file,config):
    """Seed a just-published source from its existing NVMe WAV, without rereading it.

    Uses the same identity, lease order and inventory as cached_source. Failure
    leaves the durable source usable; cache entries remain disposable.
    """
    if not config.get('enabled'):return False
    source=Path(source);local_file=Path(local_file);info=source.stat()
    maximum=int(config.get('max_gib',20)*2**30)
    if info.st_size>maximum or local_file.stat().st_size!=info.st_size:return False
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    marker=root/'.audio-pipeline-cache-v1'
    if not marker.exists():
        if any(root.iterdir()):return False
        try:marker.write_text('Managed local source cache v1\n')
        except OSError:return False
    key=hashlib.sha256(json.dumps([str(source.resolve()),info.st_size,info.st_mtime_ns]).encode()).hexdigest()
    target=root/(key+source.suffix);temporary=root/(key+f'.partial-seed-{os.getpid()}')
    try:
        with (root/(key+'.lease')).open('a+') as lease,(root/'cache.lock').open('a+') as global_lock:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if target.exists():return True
            try:os.link(local_file,temporary)
            except OSError:shutil.copyfile(local_file,temporary)
            after=source.stat()
            if (after.st_size,after.st_mtime_ns)!=(info.st_size,info.st_mtime_ns):return False
            with _locked(global_lock):
                if not _make_room(root,maximum,info.st_size):return False
                with inventory(root) as connection:reserve(connection,target.name,info.st_size)
                os.replace(temporary,target)
        return True
    except OSError:return False
    finally:temporary.unlink(missing_ok=True)
