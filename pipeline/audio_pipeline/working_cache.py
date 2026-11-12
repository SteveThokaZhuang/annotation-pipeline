"""Persistent working tree with ordered, recoverable output replication.

The local database is authoritative while this working tree exists. Files are
replicated before a SQLite backup is published; live SQLite files are never copied.
"""
import contextlib
from functools import lru_cache
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import threading
import time

from .failure_policy import InfrastructureError

ENV = 'AUDIO_PIPELINE_WORKING_TREE'
_active = None


@lru_cache(maxsize=8)
def _parse_mapping(raw):
    return json.loads(raw) if raw else None


def mapping():
    return _parse_mapping(os.environ.get(ENV))


def canonical_path(value, pair=None):
    pair = pair or mapping()
    value = str(value)
    if pair:
        cache, final = pair['cache'], pair['final']
        if value == cache or value.startswith(cache + '/'):
            return final + value[len(cache):]
    return value


def preferred_path(value, pair=None):
    pair = pair or mapping()
    value = str(value)
    if pair:
        final, cache = pair['final'], pair['cache']
        if value == final or value.startswith(final + '/'):
            candidate = cache + value[len(final):]
            base = candidate.rpartition('#result=')[0] if '#result=' in candidate else candidate
            if Path(base).exists():
                return candidate
    return value


def check_health():
    if _active is not None:
        _active.check_health()


def _backup(source, destination, cancelled=lambda: False):
    started = time.monotonic()
    def progress(*_):
        if cancelled() or time.monotonic() - started > 120:
            raise InterruptedError('SQLite snapshot interrupted; local state is retained')
    with contextlib.closing(sqlite3.connect(Path(source).as_uri() + '?mode=ro', uri=True)) as src:
        with contextlib.closing(sqlite3.connect(str(destination))) as dst:
            src.backup(dst, pages=256, progress=progress, sleep=.01)
    with open(destination, 'rb') as stream:
        os.fsync(stream.fileno())


class WorkingCache:
    def __init__(self, final, settings):
        self.final = Path(final).absolute()
        self.settings = settings
        self.sync_enabled = settings.get('sync_enabled', True)
        if not isinstance(self.sync_enabled, bool):
            raise ValueError('working_cache.sync_enabled must be boolean')
        self.interval = float(settings.get('sync_interval_seconds', 30))
        self.bandwidth = float(settings.get('sync_mib_per_second', 64)) * 2**20
        self.min_free = float(settings.get('min_free_gib', 8)) * 2**30
        if not all(math.isfinite(x) for x in (self.interval, self.bandwidth, self.min_free)) or self.interval <= 0 or self.bandwidth <= 0 or self.min_free < 0:
            raise ValueError('Invalid working-cache synchronization limits')
        identity = hashlib.sha256(str(self.final).encode()).hexdigest()[:24]
        self.home = Path(settings['root']).absolute() / identity
        self.cache = self.home / 'files'
        if self.final == self.cache or self.final in self.cache.parents or self.cache in self.final.parents:
            raise ValueError('Working cache and final output must be separate trees')
        self.pair = dict(final=str(self.final), cache=str(self.cache))
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.guard = None
        self.local_guard = None
        self.previous_env = None
        self.mapping_installed = False
        self.finished = False
        self.copied_files = 0
        self.copied_bytes = 0
        self.last_snapshot = None
        self.sync_lock = threading.Lock()
        self.cancel_requested = lambda: False

    def _cancelled(self):
        return self.stop_event.is_set() or self.cancel_requested()

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError('A working cache is already active in this controller')
        # Older controllers do not take the working-tree guard. Never import or
        # replace their database while a registered run still owns this output.
        registrations = list((self.final.parent.parent / 'control' / 'runs').glob('*.json'))
        registrations.extend((self.home / 'control' / 'runs').glob('*.json'))
        for path in registrations:
            try:
                row = json.loads(path.read_text())
                run_dir = canonical_path(row.get('run_dir',''), self.pair)
                if row.get('status') != 'running' or not run_dir.startswith(str(self.final) + '/runs/'):
                    continue
                pid = int(row['pid'])
                if pid != os.getpid() and Path(f'/proc/{pid}/stat').read_text().split()[21] == row['proc_start']:
                    raise RuntimeError('Another registered pipeline is active for this output; stop it before enabling the working cache')
            except (OSError,ValueError,KeyError,TypeError,IndexError):
                continue
        try:
            self.cache.mkdir(parents=True, exist_ok=True)
            # All new controllers, including explicit sync, share this local lock.
            self.local_guard = (self.home / '.working-cache.lock').open('a+')
            fcntl.flock(self.local_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            final_guard = self.final / '.working-cache.lock'
            if self.sync_enabled:
                self.final.mkdir(parents=True, exist_ok=True)
                self.guard = final_guard.open('a+')
            else:
                # Honor an older controller's lock/owner without writing final storage.
                try:
                    self.guard = final_guard.open('r')
                except FileNotFoundError:
                    pass
            if self.guard is not None:
                fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.guard.seek(0)
                owner = self.guard.read().strip()
                if owner and owner != str(self.home):
                    raise RuntimeError('This output has a retained working cache at ' + owner + '; use that cache root to preserve unsynchronized results')
                if self.sync_enabled:
                    self.guard.seek(0); self.guard.truncate(); self.guard.write(str(self.home)); self.guard.flush(); os.fsync(self.guard.fileno())
            db = self.cache / 'state.sqlite'
            if not db.exists() and (self.final / 'state.sqlite').exists():
                temporary = self.home / 'bootstrap.sqlite'
                temporary.unlink(missing_ok=True)
                _backup(self.final / 'state.sqlite', temporary)
                os.replace(temporary, db)
            self.previous_env = os.environ.get(ENV)
            os.environ[ENV] = json.dumps(self.pair)
            self.mapping_installed = True
            _active = self
            self.check_health()
            if self.sync_enabled:
                self.thread = threading.Thread(target=self._run, name='output-sync', daemon=True)
                self.thread.start()
            print(json.dumps(dict(event='working_cache', **self.pair, retained=True, sync_enabled=self.sync_enabled)), flush=True)
            return self
        except BaseException:
            self.close()
            raise

    def check_health(self):
        if self.error is not None:
            raise InfrastructureError('Output synchronization failed; cached work is retained: ' + str(self.error)) from self.error
        if shutil.disk_usage(self.cache).free < self.min_free:
            raise InfrastructureError('Working-cache free space is below min_free_gib; cache retained, no files evicted')

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self.sync_once()
                self.stop_event.wait(self.interval)
        except InterruptedError:
            if not self.stop_event.is_set():
                self.error = RuntimeError('Output synchronization was interrupted')
        except BaseException as error:
            self.error = error

    def _copy(self, source, destination, json_paths=False):
        before = source.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.sync-', dir=destination.parent)
        started = time.monotonic(); written = 0
        prefix = json.dumps(str(self.cache), ensure_ascii=False)[1:-1].encode()
        replacement = ('"' + json.dumps(str(self.final), ensure_ascii=False)[1:-1]).encode()
        pattern = re.compile(rb'(?<!\\)"' + re.escape(prefix) + rb'(?=/|")')
        carry = b''
        try:
            with os.fdopen(fd, 'wb') as out, source.open('rb') as inp:
                while True:
                    if self._cancelled():
                        raise InterruptedError('Synchronization paused')
                    block = inp.read(1024 * 1024)
                    if not block:
                        if carry: out.write(pattern.sub(lambda _: replacement, carry))
                        break
                    if json_paths:
                        data = carry + block; safe = max(0, len(data) - len(prefix) - 3); pos = 0
                        for match in pattern.finditer(data):
                            if match.start() >= safe: break
                            out.write(data[pos:match.start()]); out.write(replacement); pos = match.end()
                        end = max(pos, safe); out.write(data[pos:end]); carry = data[end:]
                    else:
                        out.write(block)
                    written += len(block)
                    delay = written / self.bandwidth - (time.monotonic() - started)
                    while delay > 0:
                        if self._cancelled() or self.stop_event.wait(min(delay,.1)):
                            raise InterruptedError('Synchronization paused')
                        delay = written / self.bandwidth - (time.monotonic() - started)
                out.flush(); os.fsync(out.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                return None
            os.utime(name, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(name, destination)
            self.copied_files += 1; self.copied_bytes += written
            return (before.st_size, before.st_mtime_ns)
        finally:
            Path(name).unlink(missing_ok=True)

    def sync_once(self):
        if not self.sync_enabled:
            raise RuntimeError('Working-cache synchronization is disabled; use the sync command to transfer retained results')
        with self.sync_lock:
            return self._sync_once()

    def _sync_once(self):
        self.check_health()
        index_path = self.home / 'sync-index.sqlite'
        index = sqlite3.connect(index_path)
        snapshot = self.home / 'snapshot.sqlite'
        database = self.cache / 'state.sqlite'
        db_version = None
        try:
            index.execute('CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,size INTEGER,mtime INTEGER)')
            if database.exists():
                stat = database.stat(); db_version = (stat.st_size, stat.st_mtime_ns)
                if db_version != self.last_snapshot:
                    snapshot.unlink(missing_ok=True)
                    _backup(database, snapshot, self._cancelled)
            complete = True
            for directory, dirs, files in os.walk(self.cache):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'locks']
                for name in files:
                    if self._cancelled(): raise InterruptedError('Synchronization paused')
                    if name.startswith('.') or name in ('state.sqlite', 'state.sqlite-journal', 'state.sqlite-wal', 'state.sqlite-shm') or name.endswith(('.lock','.buffer','.tmp')) or '.partial' in name:
                        continue
                    source = Path(directory) / name
                    if source.is_symlink(): continue
                    relative = source.relative_to(self.cache).as_posix()
                    stat = source.stat(); version = (stat.st_size, stat.st_mtime_ns)
                    destination = self.final / relative
                    previous = index.execute('SELECT size,mtime FROM files WHERE path=?', (relative,)).fetchone()
                    if previous == version and destination.exists(): continue
                    copied = self._copy(source, destination, source.suffix in ('.json', '.jsonl'))
                    if copied is None:
                        # Active logs may change without invalidating a completed task snapshot.
                        if source.suffix != '.log': complete = False
                        continue
                    index.execute('INSERT OR REPLACE INTO files VALUES(?,?,?)', (relative, *copied))
                index.commit()
            if complete and db_version is not None and snapshot.exists():
                if self._copy(snapshot, self.final / 'state.sqlite') is not None:
                    self.last_snapshot = db_version
                    snapshot.unlink(missing_ok=True)
            return complete
        finally:
            index.close()

    def finish(self, stop_requested=lambda: False):
        """Drain only after workers stop; keep the working tree after completion."""
        if not self.sync_enabled:
            self.check_health()
            self.finished = True
            print(json.dumps(dict(event='working_cache_complete', sync_enabled=False, cache_retained=str(self.cache))), flush=True)
            return True
        self.stop_event.set()
        if self.thread is not None: self.thread.join()
        self.check_health()
        self.stop_event.clear()
        self.cancel_requested = stop_requested
        try:
            if not self.sync_once():
                raise InfrastructureError('Output changed during final synchronization; cached work is retained')
        except InterruptedError:
            if stop_requested(): return False
            raise
        finally:
            self.cancel_requested = lambda: False
        self.finished = True
        print(json.dumps(dict(event='output_sync_complete', files=self.copied_files, bytes=self.copied_bytes, cache_retained=str(self.cache))), flush=True)
        return True

    def close(self):
        global _active
        self.stop_event.set()
        if self.thread is not None: self.thread.join()
        if _active is self: _active = None
        if self.mapping_installed:
            if self.previous_env is not None: os.environ[ENV] = self.previous_env
            else: os.environ.pop(ENV, None)
            self.mapping_installed = False
        if self.guard is not None: self.guard.close(); self.guard = None
        if self.local_guard is not None: self.local_guard.close(); self.local_guard = None

    def __exit__(self, *_):
        self.close()
