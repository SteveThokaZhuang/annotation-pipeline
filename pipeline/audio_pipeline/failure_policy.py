"""Recoverable item failures, fatal persistence errors, and auditable exports."""
import csv
import errno
import json
import os
from pathlib import Path
import sqlite3
import tempfile

SUCCESS = {'passed', 'completed_with_errors'}
ASR_STAGES = {'asr1', 'asr2', 'asr3'}


class InfrastructureError(RuntimeError):
    """Progress/results cannot be persisted reliably; stop this run safely."""


def is_infrastructure_error(error):
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error,(InfrastructureError,sqlite3.Error)):
            return True
        if isinstance(error,OSError) and error.errno in {errno.ENOSPC,errno.EDQUOT,errno.EROFS}:
            return True
        error=error.__cause__ or error.__context__
    return False


def persist(call,*args,**kwargs):
    try:return call(*args,**kwargs)
    except (OSError,sqlite3.Error) as error:
        raise InfrastructureError(f'Unable to persist pipeline state/results: {error}') from error


def hypothesis_error(value):
    if value is None:return 'missing_hypothesis'
    if value.get('transcription_error'):return str(value['transcription_error'])
    if value.get('fallback_requested'):return 'fallback_requested'
    if not isinstance(value.get('text'),str):return 'invalid_transcript_type'
    return None


def failure_category(error):
    message=str(error).lower()
    if 'token_limit' in message:return 'decoder_token_limit'
    if 'aligner timestamp' in message:return 'alignment_timestamp'
    if 'ipc path' in message or 'socket' in message:return 'runtime_transport'
    if 'model unavailable' in message or 'initialization failed' in message:return 'model_initialization'
    if 'whole-sequence' in message or 'enhancement limit' in message:return 'enhancement_policy'
    if 'out of memory' in message or 'headroom' in message or 'host ram' in message:return 'memory_limit'
    if 'missing enhanced' in message:return 'missing_upstream_enhancement'
    return 'processing_error'


def failure_record(dataset,stage,model,item,error,status='failed',signature=None):
    source=item.get('chunk',item)
    identity=item.get('id',source.get('id'))
    message=str(error or status)
    return dict(dataset=dataset,stage=stage,model=model,chunk_id=identity,
        recording_id=source.get('recording_id',identity),source_path=source.get('path',source.get('source_path')),
        channel=source.get('channel'),start=source.get('start'),end=source.get('end'),
        start_sample=source.get('start_sample'),end_sample=source.get('end_sample'),
        duration=source.get('duration'),status=status,category=failure_category(message),error_message=message.strip().splitlines()[-1],
        error=message,signature=signature)


def export_failures(base,records):
    """Write JSONL plus CSV atomically; never append to a production manifest."""
    base=Path(base);base.parent.mkdir(parents=True,exist_ok=True)
    records=list(records)
    fields=list(failure_record(None,None,None,{},'').keys())
    paths={}
    for suffix in ['jsonl','csv']:
        destination=base.with_suffix('.'+suffix)
        fd,temporary=tempfile.mkstemp(dir=destination.parent,prefix='.'+destination.name)
        try:
            with os.fdopen(fd,'w',newline='') as stream:
                if suffix=='jsonl':
                    for row in records:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                else:
                    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
                    writer.writeheader();writer.writerows(records)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,destination)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        paths[suffix]=str(destination)
    return paths
