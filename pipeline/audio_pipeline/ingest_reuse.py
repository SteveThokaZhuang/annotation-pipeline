"""Reuse matching publications after validation or explicit trust of committed rows."""
from .common import digest
from .cache_version import inference_descriptor
from .working_cache import canonical_path


class ManifestReuse:
    def __init__(self,state,tasks,settings,dataset,limit,enhanced,stage='ingest',model=None):
        model=model or stage
        self.path=None;self.entries=[];self.checked=0;self.valid=False;self.manifest=None
        try:
            path=state.head_path(stage);manifest=state.head(stage)
        except (RuntimeError,OSError,ValueError,KeyError):return
        if not isinstance(manifest,dict):return
        self.manifest=manifest
        semantic=lambda value:inference_descriptor({'settings':value})['settings']
        if (manifest.get('stage')!=stage or manifest.get('model')!=model or
            manifest.get('status')!='passed' or manifest.get('failed',0) or
            manifest.get('dataset')!=dataset or manifest.get('limited')!=limit or
            manifest.get('enhanced')!=enhanced or
            semantic(manifest.get('settings',{}))!=semantic(settings) or
            not isinstance(manifest.get('entries'),list) or len(manifest['entries'])!=len(tasks)):
            return
        self.path=path;self.entries=manifest['entries'];self.valid=True

    def observe(self,task,row,value,trusted=False):
        if not self.valid:return
        entry=self.entries[self.checked];self.checked+=1
        if not isinstance(entry,dict):self.valid=False;self.entries=[];return
        self.valid=bool((trusted or value is not None) and row and row[0]=='complete' and
            entry.get('id')==task['item']['id'] and entry.get('signature')==task['signature'] and
            entry.get('status')=='complete' and not entry.get('error') and
            canonical_path(entry.get('output',''))==canonical_path(row[1]) and
            (trusted or entry.get('digest')==digest(value)))
        if not self.valid:self.entries=[]

    @property
    def reusable(self):return self.valid and self.checked==len(self.entries)


IngestReuse=ManifestReuse
