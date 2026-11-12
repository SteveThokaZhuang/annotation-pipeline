"""Cancellable controller exports; consume already validated result objects."""
import time
from .common import atomic
from .resume import ScanStopped


class StopCheck:
    def __init__(self,path,interval=.1):
        self.path=path;self.interval=interval;self.next_poll=0.

    def __call__(self,force=False):
        now=time.monotonic()
        if not force and now<self.next_poll:return
        self.next_poll=now+self.interval
        if self.path.exists():raise ScanStopped()


def collect_recording(assembled,value):
    chunk=value['chunk']
    key=chunk['recording_id']+f"_c{chunk['channel']}"
    assembled.setdefault(key,[]).append(value)


def export_recordings(assembled,directory,check,write=atomic,workers=1,expected_chunks=None):
    if not isinstance(workers,int) or isinstance(workers,bool) or workers<1:raise ValueError('export workers must be positive')
    check(force=True)
    def export(pair):
        key,values=pair
        check(force=True)
        values=sorted(values,key=lambda v:v['chunk']['start_sample'])
        output=dict(recording_channel=key,text=' '.join(v['text'] for v in values),
                    chunks=values,words=[w for v in values for w in v.get('words',[])])
        if expected_chunks is not None:
            expected=expected_chunks.get(key,set())
            actual={v['chunk']['id'] for v in values}
            missing=sorted(expected-actual)
            output['coverage']=dict(status='partial' if missing else 'complete',
                                    expected_chunks=len(expected),completed_chunks=len(actual),
                                    missing_chunk_ids=missing,
                                    no_transcript_chunks=sum(v.get('transcript_status')=='no_transcript' for v in values))
        check(force=True)
        write(directory/(key+'.json'),output)
        check(force=True)
    if workers==1:
        for pair in assembled.items():export(pair)
        return
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    iterator=iter(assembled.items());pending=set()
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='recording-export') as pool:
        try:
            while True:
                check(force=True)
                while len(pending)<workers*2:
                    pair=next(iterator,None)
                    if pair is None:break
                    pending.add(pool.submit(export,pair))
                if not pending:break
                done,pending=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
                for future in done:future.result()
        finally:
            for future in pending:future.cancel()
