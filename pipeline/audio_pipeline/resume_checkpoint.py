"""Small completion receipts for explicitly trusted resumes.

Receipts bind successful publications to inference settings, upstream heads and
committed database rows. They do not certify external source/artifact edits;
full validation remains the recovery path for those changes.
"""
from pathlib import Path
from .common import atomic, digest, fingerprint, read
from .cache_version import inference_descriptor, source_version
from .working_cache import canonical_path


DEPENDENCIES={
    'ingest':[],
    'dnsmos':['ingest'],
    'vad':['ingest'],
    'enhance':['ingest','vad'],
    'asr1':['ingest','vad'],
    'asr2':['ingest','vad'],
    'disagreement':['ingest','vad','asr1','asr2'],
    'asr3':['ingest','vad','asr1','asr2','disagreement'],
    'rover':['ingest','vad','asr1','asr2','disagreement','asr3'],
    'align':['ingest','vad','asr1','asr2','disagreement','asr3','rover'],
}


def stamp(path):
    return digest(fingerprint(path))


def context(state,stage,spec,cfg,args,enhanced):
    if stage not in DEPENDENCIES:return None
    heads={}
    names=list(DEPENDENCIES[stage])
    if enhanced and stage in ('asr1','asr2','disagreement','asr3','rover','align'):
        names.append('enhance')
    for name in names:
        try:heads[name]=stamp(state.head_path(name))
        except (RuntimeError,OSError,ValueError):heads[name]=None
    planning={name:source_version(Path(__file__).with_name(name))
              for name in ('planning.py','enhancement_chunks_v1.py')}
    return digest(dict(schema=1,stage=stage,inference=inference_descriptor(spec),planning=planning,
        dataset=args.dataset,dataset_root=cfg.get('dataset_root'),
        policy=cfg.get('datasets',{}).get(args.dataset,{}),
        limit=args.limit,enhanced=enhanced,upstream=heads))


def database_version(state,stage,model):
    return list(state.db.execute('SELECT COUNT(*),MAX(updated) FROM tasks WHERE stage=? AND model=?',
                                (stage,model)).fetchone())


def receipt_path(state,stage,model):
    return state.root/'control/resume'/(stage+'-'+digest(model)[:16]+'.json')


def publication(state,stage):
    path=state.head_path(stage)
    files={'manifest':stamp(path)}
    if stage=='disagreement':
        for label in ('low','high'):
            files[label]=stamp(path.with_suffix('.'+label+'.json'))
    return canonical_path(path),files


def load(state,stage,model,token):
    if token is None:return None
    try:
        receipt=read(receipt_path(state,stage,model))
        if receipt['schema']!=1 or receipt['context']!=token:return None
        head,files=publication(state,stage)
        if canonical_path(receipt['head'])!=head or receipt['files']!=files:return None
        if receipt['database']!=database_version(state,stage,model):return None
        report=receipt['report']
        if not isinstance(report,dict):return None
        if (report.get('status')!='passed' or report.get('failed',0) or
                report.get('stage')!=stage or report.get('model')!=model or
                type(report.get('tasks')) is not int or report['tasks']<0 or
                report.get('completed')!=report['tasks']):return None
        return report
    except (OSError,ValueError,KeyError,TypeError,RuntimeError):
        return None


def save(state,stage,spec,cfg,args,enhanced,token,report):
    if (token is None or report.get('status')!='passed' or report.get('failed',0) or
            report.get('upstream_failures',0) or report.get('no_transcript',0) or report.get('skipped',0)):
        return
    # Never certify a newer upstream publication than the stage actually used.
    if context(state,stage,spec,cfg,args,enhanced)!=token:return
    try:head,files=publication(state,stage)
    except (RuntimeError,OSError,ValueError):return
    summary={k:v for k,v in report.items() if k not in ('workers','allocation','worker_errors')}
    atomic(receipt_path(state,stage,spec['model']),dict(schema=1,context=token,head=head,files=files,
        database=database_version(state,stage,spec['model']),report=summary))
