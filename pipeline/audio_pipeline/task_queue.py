"""Atomic claims of bounded work bundles. Resume uses authoritative task checkpoints."""
import json
import math
import sqlite3
import time
from pathlib import Path


def create_queue(path,tasks,settings,workers=1):
    from .batching import duration
    maximum=int(settings['max_items']);seconds=float(settings['max_audio_seconds'])
    if maximum<1 or seconds<=0:raise ValueError('Task dispatch limits must be positive')
    target=settings.get('target_bundles_per_worker',1)
    maximum=min(maximum,max(1,math.ceil(len(tasks)/(workers*target))))
    groups=[];group=[];cost=0.
    units=[[t] for t in tasks]
    if settings.get('preserve_recordings',False):
        by_recording={}
        for task in tasks:
            item=task['item'].get('chunk',task['item']);key=item.get('recording_id',item['id'])
            by_recording.setdefault(key,[]).append(task)
        units=list(by_recording.values())
    if settings.get('order','input')=='longest_first':
        units.sort(key=lambda unit:sum(max(0.,duration(task)) for task in unit),reverse=True)
    for unit in units:
        value=sum(max(0.,duration(task)) for task in unit)
        if group and (len(group)+len(unit)>maximum or cost+value>seconds):groups.append(group);group=[];cost=0.
        group.extend(unit);cost+=value
    if group:groups.append(group)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE bundles(id INTEGER PRIMARY KEY,payload TEXT,status TEXT,owner INTEGER,updated REAL)')
        db.executemany('INSERT INTO bundles(payload,status,updated) VALUES(?,?,?)',[(json.dumps(g),'pending',time.time()) for g in groups])


def claim(path,owner):
    with sqlite3.connect(path,timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT id,payload FROM bundles WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
        if row is None:return None
        db.execute("UPDATE bundles SET status='leased',owner=?,updated=? WHERE id=?",(owner,time.time(),row[0]))
        return row[0],json.loads(row[1])


def finish(path,identity,status):
    with sqlite3.connect(path,timeout=60) as db:db.execute('UPDATE bundles SET status=?,updated=? WHERE id=?',(status,time.time(),identity))
