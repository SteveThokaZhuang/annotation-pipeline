"""Per-recording coverage ledger. Partial transcripts never imply complete coverage."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
from .common import atomic,read,fingerprint,digest
from .working_cache import canonical_path


@contextmanager
def ledger_lock(root):
    path=Path(root)/'locks'/'recordings-ledger.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def write_ledger(root,state,stage,entries,skipped,incremental=False):
    if stage=='enhance_parts':return None
    with ledger_lock(root):return _write_ledger(root,state,stage,entries,skipped,incremental)


def _write_ledger(root,state,stage,entries,skipped,incremental):
    try:ingest_path=state.head_path('ingest')
    except RuntimeError:return None
    previous=Path(root)/'recordings-ledger.json'
    index_path=Path(root)/'recordings-ledger-index.json'
    ingest_head=canonical_path(ingest_path)
    stage_head=canonical_path(state.head_path(stage)) if incremental else None
    index={};trusted_snapshot=False
    if incremental:
        try:
            index=read(index_path)
            index['ingest_manifest']=canonical_path(index.get('ingest_manifest'))
            index['stage_manifests']={k:canonical_path(v) for k,v in index.get('stage_manifests',{}).items()}
            trusted_snapshot=index.get('ledger')==fingerprint(previous)
            if (index.get('ingest_manifest')==ingest_head and
                index.get('stage_manifests',{}).get(stage)==stage_head and
                trusted_snapshot):
                return str(previous)
        except (OSError,ValueError,KeyError,TypeError,AttributeError):index={}
    ingest=state.head('ingest')
    rebuilt=False
    try:
        ledger=read(previous) if previous.exists() else {'recordings':{}}
        if not isinstance(ledger,dict) or not isinstance(ledger.get('recordings'),dict) or any(not isinstance(r,dict) or not isinstance(r.get('stages'),dict) for r in ledger['recordings'].values()):
            raise ValueError('Invalid recording ledger structure')
    except (ValueError,FileNotFoundError):
        ledger={'recordings':{}};rebuilt=True;trusted_snapshot=False
    changed=rebuilt or not previous.exists();versions={};updates=[]
    if incremental:
        state.db.execute('CREATE TABLE IF NOT EXISTS ledger_sources(item TEXT PRIMARY KEY,result_digest TEXT NOT NULL)')
        if trusted_snapshot:versions=dict(state.db.execute('SELECT item,result_digest FROM ledger_sources'))
    expected={}
    if stage not in ('ingest','dnsmos','vad','enhance'):
        try:
            vad_path=state.head_path('vad');version=digest(fingerprint(vad_path));reuse_counts=False
            if incremental:
                state.db.execute('CREATE TABLE IF NOT EXISTS ledger_vad_counts(item TEXT PRIMARY KEY,chunks INTEGER NOT NULL)')
                state.db.execute('CREATE TABLE IF NOT EXISTS ledger_vad_version(version TEXT PRIMARY KEY)')
                saved=state.db.execute('SELECT version FROM ledger_vad_version').fetchone()
                reuse_counts=bool(saved and saved[0]==version and getattr(state,'resume_validation','full')=='trusted')
            if reuse_counts:
                expected=dict(state.db.execute('SELECT item,chunks FROM ledger_vad_counts'))
            else:
                for entry in state.head('vad')['entries']:
                    if entry['status']=='complete':
                        value=read(entry['output']);expected[entry['id']]=len(value['chunks'])
                if incremental:
                    state.db.execute('DELETE FROM ledger_vad_counts');state.db.execute('DELETE FROM ledger_vad_version')
                    state.db.executemany('INSERT INTO ledger_vad_counts VALUES(?,?)',expected.items())
                    state.db.execute('INSERT INTO ledger_vad_version VALUES(?)',(version,))
                    state.db.commit()
        except RuntimeError:pass
    for entry in ingest['entries']:
        identity=entry['id']
        if identity not in ledger['recordings']:
            ledger['recordings'][identity]={'id':identity,'stages':{}};changed=True
        row=ledger['recordings'][identity]
        if row.get('id')!=identity:row['id']=identity;changed=True
        if entry.get('output') and entry['status']=='complete':
            if not incremental or not entry.get('digest') or versions.get(identity)!=entry.get('digest') or any(k not in row for k in ('source_path','duration','channels')):
                value=read(entry['output']);metadata=dict(source_path=value['path'],duration=value['duration'],channels=value['channels'])
                if any(row.get(k)!=v for k,v in metadata.items()):row.update(metadata);changed=True
                if incremental and identity in versions and versions[identity]!=entry.get('digest'):
                    for name,counts in row['stages'].items():
                        if name!='ingest':counts['status']='stale_source';changed=True
                if incremental and entry.get('digest'):updates.append((identity,entry['digest']))
    active={e['id'] for e in ingest['entries']}
    routed={}
    if stage=='asr3':
        for e in entries:
            key=e.get('input',{}).get('recording_id') or e['id'];routed[key]=routed.get(key,0)+1
    totals={}
    for entry in entries:
        identity=entry.get('input',{}).get('recording_id') or entry['id']
        counts=totals.setdefault(identity,{'completed':0,'failed':0,'no_transcript':0,'no_speech':False})
        counts['completed' if entry['status']=='complete' else 'failed']+=1
        counts['no_transcript']+=entry.get('result_status')=='no_transcript'
        if stage=='vad' and entry['status']=='complete':counts['no_speech']=not read(entry['output'])['chunks']
    for identity,row in ledger['recordings'].items():
        if row.get('active')!=(identity in active):row['active']=identity in active;changed=True
        if not row['active']:continue
        counts=totals.get(identity,{'completed':0,'failed':0,'no_transcript':0,'no_speech':False})
        required=1 if stage in ('ingest','dnsmos','vad','enhance') else expected.get(identity,0)
        if stage=='asr3':required=routed.get(identity,0)
        counts['expected']=required
        if stage not in ('ingest','dnsmos','vad','enhance') and identity not in expected:status='blocked_by_vad'
        elif stage=='asr3' and required==0 and expected.get(identity,0):status='not_routed'
        elif counts['no_speech'] or (stage not in ('ingest','dnsmos','vad','enhance') and expected.get(identity)==0):status='no_detected_speech'
        elif counts['failed']:status='partial' if counts['completed'] else 'failed'
        elif counts['completed']<required:
            status='not_processed' if not counts['completed'] else 'partial'
            if row['stages'].get('enhance',{}).get('status')=='failed':status='blocked_by_enhancement'
        else:status='complete'
        counts['status']=status
        if row['stages'].get(stage)!=counts:row['stages'][stage]=counts;changed=True
    if changed or not incremental:
        ledger['stage']=stage;atomic(previous,ledger)
    if incremental:
        if updates:
            state.db.executemany('INSERT OR REPLACE INTO ledger_sources VALUES(?,?)',updates)
        state.db.commit()
        applied=dict(index.get('stage_manifests',{})) if index.get('ingest_manifest')==ingest_head else {}
        applied[stage]=stage_head
        atomic(index_path,dict(schema=1,ingest_manifest=ingest_head,stage_manifests=applied,ledger=fingerprint(previous)))
    return str(previous)
