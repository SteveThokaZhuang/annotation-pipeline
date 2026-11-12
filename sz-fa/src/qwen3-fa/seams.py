"""Audited midpoint stitching of touching Qwen/MFA words across chunks."""
from collections import deque
import math


def audio_key(row):
    return row.get('source_audio') or row['audio'], row.get('channel') or 0


def join_pair(left,right,adjustments,rejected):
    if audio_key(left)!=audio_key(right):return
    if not left['words'] or not right['words']:return
    a,b=left['words'][-1],right['words'][0]
    if {a.get('source'),b.get('source')}!={'qwen','mfa'}:return
    if not (math.isfinite(a['end']) and math.isfinite(b['start'])):return
    if a['end']<=b['start']+1e-7:return
    record=dict(left_id=left['id'],right_id=right['id'],left_index=len(left['words'])-1,right_index=0,
        source_audio=audio_key(left)[0],channel=audio_key(left)[1],
        left_chunk_end=left['chunk_end'],right_chunk_start=right['chunk_start'],
        left_block=left.get('_output_block'),right_block=right.get('_output_block'),
        before_left=dict(a),before_right=dict(b),overlap_seconds=a['end']-b['start'])
    if left['chunk_end']>right['chunk_start']+1e-7:
        rejected.append(dict(record,reason='upstream_chunk_audio_intervals_overlap'));return
    midpoint=(a['end']+b['start'])/2
    if (not all(math.isfinite(v) for v in (a['start'],b['end'],midpoint)) or
            midpoint-a['start']<=1e-7 or b['end']-midpoint<=1e-7):
        rejected.append(dict(record,reason='midpoint_would_collapse_word'));return
    new_a,new_b=dict(a,end=midpoint),dict(b,start=midpoint)
    # Replace dictionaries so sentence-candidate audit snapshots remain immutable.
    left['words'][-1]=new_a;right['words'][0]=new_b
    adjustments.append(dict(record,midpoint=midpoint,after_left=dict(new_a),after_right=dict(new_b)))


def reconcile_stream(rows,adjustments,rejected):
    """Yield the original order once both seams are final; input is audio/time ordered.

    Buffer one pending nonempty chunk per channel. Empty chunks do not erase the
    previous lexical boundary. Upstream overlapping windows are never deduplicated.
    """
    queue=deque();pending={};current_audio=None
    for row in rows:
        key=audio_key(row)
        if key[0]!=current_audio:
            for item in pending.values():item['ready']=True
            pending={};current_audio=key[0]
            while queue:yield queue.popleft()['row']
        item=dict(row=row,ready=not bool(row['words']));queue.append(item)
        if row['words']:
            previous=pending.get(key)
            if previous:
                join_pair(previous['row'],row,adjustments,rejected)
                previous['ready']=True
            pending[key]=item
        while queue and queue[0]['ready']:yield queue.popleft()['row']
    while queue:yield queue.popleft()['row']


def reconcile_rows(rows,adjustments,rejected):
    """Stitch a complete in-memory batch without changing its original row order."""
    ordered=sorted(rows,key=lambda r:(audio_key(r),r['chunk_start'],r['id']))
    list(reconcile_stream(ordered,adjustments,rejected))
    return rows
