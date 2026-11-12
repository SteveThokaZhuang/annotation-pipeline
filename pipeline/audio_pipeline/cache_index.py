"""Local cache inventory: constant-time accounting, indexed least-recent use.

Call only while holding cache.lock. Publication reserves bytes before rename;
unlink happens before removing a reservation. A process crash can overcount, not silently
under-count a newly published file. Missing reservations are repaired on access
or eviction. Existing managed caches are inventoried once during migration.
"""
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import time

_PATTERN=re.compile(r'[0-9a-f]{64}\.[^./]+')

@contextmanager
def inventory(root):
    database=Path(root)/'cache-index.sqlite'
    connection=sqlite3.connect(database,timeout=60)
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=NORMAL')
        connection.execute('CREATE TABLE IF NOT EXISTS files(name TEXT PRIMARY KEY,size INTEGER NOT NULL,accessed INTEGER NOT NULL)')
        connection.execute('CREATE INDEX IF NOT EXISTS oldest ON files(accessed,name)')
        connection.execute('CREATE TABLE IF NOT EXISTS usage(id INTEGER PRIMARY KEY,bytes INTEGER NOT NULL)')
        if connection.execute('SELECT bytes FROM usage WHERE id=1').fetchone() is None:
            total=0
            for path in Path(root).iterdir():
                if not _PATTERN.fullmatch(path.name) or path.suffix in ('.lease','.lock') or path.suffix.startswith('.partial-'):continue
                try:info=path.stat()
                except FileNotFoundError:continue
                if not path.is_file():continue
                connection.execute('INSERT OR REPLACE INTO files VALUES(?,?,?)',(path.name,info.st_size,info.st_mtime_ns));total+=info.st_size
            connection.execute('INSERT INTO usage VALUES(1,?)',(total,))
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback();raise
    finally:connection.close()


def remove(connection,name):
    row=connection.execute('SELECT size FROM files WHERE name=?',(name,)).fetchone()
    if row:
        connection.execute('DELETE FROM files WHERE name=?',(name,))
        connection.execute('UPDATE usage SET bytes=bytes-? WHERE id=1',(row[0],))


def reserve(connection,name,size):
    remove(connection,name)
    connection.execute('INSERT INTO files VALUES(?,?,?)',(name,size,time.time_ns()))
    connection.execute('UPDATE usage SET bytes=bytes+? WHERE id=1',(size,))


def touch(root,target,size):
    with inventory(root) as connection:
        row=connection.execute('SELECT size FROM files WHERE name=?',(target.name,)).fetchone()
        if row is None or row[0]!=size:reserve(connection,target.name,size)
        else:connection.execute('UPDATE files SET accessed=? WHERE name=?',(time.time_ns(),target.name))


def forget_missing(root,target):
    with inventory(root) as connection:remove(connection,target.name)
