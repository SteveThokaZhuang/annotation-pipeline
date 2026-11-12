"""Retain trusted ASR manifest entries; validate only changed or unfinished rows."""
from .cache_version import inference_descriptor
from .failure_policy import ASR_STAGES
from .working_cache import canonical_path


class ASRPublication:
    def __init__(self,manifest,stage,model,settings,tasks,dataset,limit,enhanced):
        self.valid=False;self.entries=[];self.changed=[];self.positions={};self.checked=0
        if stage not in ASR_STAGES or not isinstance(manifest,dict):return
        semantic=lambda value:inference_descriptor(dict(settings=value))['settings']
        if (manifest.get('stage')!=stage or manifest.get('model')!=model or
                manifest.get('dataset')!=dataset or manifest.get('limited')!=limit or
                manifest.get('enhanced')!=enhanced or semantic(manifest.get('settings',{}))!=semantic(settings) or
                not isinstance(manifest.get('entries'),list) or len(manifest['entries'])!=len(tasks)):
            return
        self.entries=manifest['entries'];self.valid=True

    def observe(self,task,row):
        if not self.valid:return
        index=self.checked;self.checked+=1;entry=self.entries[index]
        if (not isinstance(entry,dict) or entry.get('id')!=task['item']['id'] or
                entry.get('signature')!=task['signature']):
            self.valid=False;self.entries=[];self.changed=[];self.positions={};return
        reusable=bool(row and row[0]=='complete' and row[1] and entry.get('status')=='complete' and
            not entry.get('error') and entry.get('digest') and entry.get('result_status')!='no_transcript' and
            canonical_path(row[1])==canonical_path(entry.get('output','')))
        if not reusable:
            self.changed.append(task);self.positions[task['item']['id']]=index

    @property
    def reusable(self):return self.valid and self.checked==len(self.entries)
