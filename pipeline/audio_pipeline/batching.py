"""Small duration-aware batch planner with conservative memory feedback.

The initial duration budget is a guard, not a promise of memory fit. A model's
calibration file can impose measured limits. Only actual CUDA OOMs shrink batches.
"""
import json
from pathlib import Path


def resolved_policy(allocation,model,root):
    """Resolve model-specific policy by value for both execution and signatures."""
    settings=dict(allocation.get('optimization',{}).get('adaptive',{}))
    settings.update(settings.pop('models',{}).get(model,{}))
    calibration=[]
    if settings.get('calibration'):
        path=Path(settings['calibration'])
        if not path.is_absolute():path=Path(root)/path
        calibration=json.loads(path.read_text())
    return dict(memory_feedback=settings.get('memory_feedback','torch'),max_padded_seconds=settings.get('max_padded_seconds',160),
        min_free_gib=settings.get('min_free_gib',4),
        target_memory_fraction=settings.get('target_memory_fraction',.8),calibration=calibration)


def duration(task):
    item=task['item']
    return float(item.get('duration',item.get('chunk',{}).get('duration',0)))


class BatchPolicy:
    def __init__(self,maximum=16,adaptive=False,max_padded_seconds=160,min_free_gib=4,
                 target_memory_fraction=.8,calibration=None,memory_feedback='torch'):
        if maximum<1 or max_padded_seconds<=0:raise ValueError('Invalid batch budget')
        if not 0<target_memory_fraction<1:raise ValueError('Memory fraction must be between 0 and 1')
        self.maximum=maximum;self.limit=maximum;self.adaptive=adaptive
        self.seconds=max_padded_seconds;self.min_free_gib=min_free_gib
        self.target=target_memory_fraction;self.calibration=calibration or []
        self.memory_feedback=memory_feedback
        previous=0
        for row in self.calibration:
            if row['max_seconds']<=previous or row['max_items']<1:raise ValueError('Invalid calibration table')
            previous=row['max_seconds']

    def groups(self,tasks):
        index=0
        while index<len(tasks):
            count=min(self.limit,len(tasks)-index)
            if self.adaptive:
                longest=0.;selected=1
                # Build prefix maxima once instead of rescanning/decrementing
                # the candidate batch, which is quadratic in its maximum size.
                for size,task in enumerate(tasks[index:index+count],1):
                    longest=max(longest,duration(task))
                    cap=1 if self.calibration else self.maximum
                    for row in self.calibration:
                        if longest<=row['max_seconds']:
                            cap=min(self.maximum,int(row['max_items']));break
                    if size<=cap and size*longest<=self.seconds:selected=size
                count=selected
            yield tasks[index:index+count]
            index+=count

    def observe(self,peak_bytes,total_bytes,free_bytes):
        if not self.adaptive:return
        if free_bytes<self.min_free_gib*2**30 or peak_bytes>total_bytes*self.target:
            self.limit=max(1,self.limit//2)
        elif peak_bytes<total_bytes*self.target*.65:
            self.limit=min(self.maximum,self.limit+max(1,self.limit//8))

    def oom(self):self.limit=max(1,self.limit//2)


def is_cuda_oom(error):
    # Do not retry arbitrary RuntimeErrors as memory failures.
    return type(error).__name__=='OutOfMemoryError' or 'cuda out of memory' in str(error).lower()
