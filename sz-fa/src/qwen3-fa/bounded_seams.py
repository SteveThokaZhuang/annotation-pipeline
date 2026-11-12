"""Transactional Qwen adjustments with a fixed, original-timestamp budget.

These are annotation edits, never audio time stretching. MFA times stay fixed.
The 20 ms / 10% limits are engineering constraints, not perceptual guarantees.
"""
import copy
import math

MAX_SHIFT = 0.020
MAX_SHRINK = 0.10
EPS = 1e-7


def valid(words, lower, upper):
    previous = lower
    for w in words:
        a, b = w['start'], w['end']
        if (not all(math.isfinite(x) for x in (a, b)) or
                b-a <= EPS or a < previous-EPS or b > upper+EPS):
            return False
        previous = b
    return True


def adjust_run(output, originals, indices, side, target, lower, upper):
    """Propagate only as far as needed, using existing gaps before more edits.

    Left side: move the end left, then the start only if 90% duration needs it.
    Right side is symmetric. Each endpoint is checked against ORIGINAL Qwen,
    including edits from earlier repairs. Caller owns copies for atomic rollback.
    """
    # A gap after the chunk edge is not permission to keep Qwen outside its
    # own audio window. Choose the closest feasible endpoint inside BOTH limits.
    target = min(target, upper) if side == 'left' else max(target, lower)
    changes = []
    for index in indices:
        before, original = output[index], originals[index]
        endpoint = 'end' if side == 'left' else 'start'
        overlap = before['end']-target if side == 'left' else target-before['start']
        if overlap <= EPS:
            break
        if (original.get('source') != 'qwen' or before.get('source') != 'qwen' or
                original['text'] != before['text'] or not valid([original], -math.inf, math.inf) or
                not valid([before], -math.inf, math.inf)):
            raise ValueError('Bounded seam requires a normal original Qwen neighbor')
        duration = original['end']-original['start']
        minimum = max((1-MAX_SHRINK)*duration, 2*EPS)
        after = dict(before)
        after[endpoint] = target
        if side == 'left':
            after['start'] = min(before['start'], target-minimum)
        else:
            after['end'] = max(before['end'], target+minimum)
        shifts = {k:after[k]-original[k] for k in ('start', 'end')}
        if any(abs(v) > MAX_SHIFT+1e-12 for v in shifts.values()):
            raise ValueError('Bounded seam exceeds original Qwen endpoint budget (20 ms)')
        if not valid([after], lower, upper):
            raise ValueError('Bounded seam would leave chunk or collapse a word')
        if after['end']-after['start'] < minimum-1e-12:
            raise ValueError('Bounded seam exceeds original Qwen duration shrink budget (10%)')
        output[index] = after
        changes.append(dict(side=side, neighbor_index=index, before_neighbor=dict(before),
            after_neighbor=dict(after), original_neighbor=dict(original),
            endpoint_shifts=shifts, duration_shrink_seconds=duration-(after['end']-after['start']),
            max_shift_seconds=MAX_SHIFT, max_shrink_fraction=MAX_SHRINK))
        target = after['start'] if side == 'left' else after['end']
    return changes


def splice_bounded(words, start, end, replacement, lower, upper, originals=None):
    if not 0 <= start < end <= len(words):
        raise ValueError('Replacement span is outside the word sequence')
    if len(replacement) != end-start:
        raise ValueError('Replacement changes token count')
    if [w['text'] for w in replacement] != [w['text'] for w in words[start:end]]:
        raise ValueError('Replacement changes token sequence')
    originals = words if originals is None else originals
    if len(originals) != len(words):
        raise ValueError('Original Qwen baseline changes token count')
    if not valid(replacement, lower, upper):
        raise ValueError('Bounded seam requires unchanged valid MFA times within chunk')
    output = [dict(w) for w in words]
    changes = adjust_run(output, originals, range(start-1, -1, -1), 'left',
                         replacement[0]['start'], lower, upper)
    changes += adjust_run(output, originals, range(end, len(words)), 'right',
                          replacement[-1]['end'], lower, upper)
    output[start:end] = [dict(w) for w in replacement]
    touched = [start, end-1] + [c['neighbor_index'] for c in changes]
    lo, hi = max(0, min(touched)-1), min(len(output), max(touched)+2)
    if not valid(output[lo:hi], lower, upper):
        raise ValueError('Bounded seam leaves invalid neighboring intervals')
    return output, dict(policy='bounded', left=output[start-1]['end'] if start else lower,
        right=output[end]['start'] if end<len(output) else upper, adjustments=changes,
        max_shift_seconds=MAX_SHIFT, max_shrink_fraction=MAX_SHRINK,
        mfa_timestamps_unchanged=True)


def reconcile_bounded_rows(rows, baseline):
    """Stitch a complete batch; reject the MFA row if its seam cannot fit.

    Retry from immutable pre-stitch candidates after rollback, so no orphaned
    neighbor edits survive a rejected repair. The original audio windows remain.
    This handles current-batch seams only; callers must supply all neighbor rows.
    """
    from seams import audio_key
    original = {r['id']:r for r in baseline}
    candidates = copy.deepcopy(rows)
    rolled_back = set()
    rejected = []
    while True:
        trial = copy.deepcopy(candidates)
        for row in trial:
            if row['id'] in rolled_back:
                row['words'] = copy.deepcopy(original[row['id']]['words'])
        changes = []
        pending = {}
        failure = None
        for right in sorted(trial, key=lambda r:(audio_key(r), r['chunk_start'], r['id'])):
            if not right['words']:
                continue
            key = audio_key(right)
            left = pending.get(key)
            pending[key] = right
            if left is None:
                continue
            a, b = left['words'][-1], right['words'][0]
            if {a.get('source'), b.get('source')} != {'qwen', 'mfa'} or a['end'] <= b['start']+EPS:
                continue
            mfa_row = left if a.get('source') == 'mfa' else right
            qwen_row = right if mfa_row is left else left
            side = 'right' if mfa_row is left else 'left'
            record = dict(policy='bounded', left_id=left['id'], right_id=right['id'],
                          mfa_id=mfa_row['id'], qwen_id=qwen_row['id'])
            try:
                if left['chunk_end'] > right['chunk_start']+EPS:
                    raise ValueError('Upstream chunk audio intervals overlap')
                indices = range(len(qwen_row['words'])) if side=='right' else range(len(qwen_row['words'])-1,-1,-1)
                edits = adjust_run(qwen_row['words'], original[qwen_row['id']]['words'], indices,
                    side, a['end'] if side=='right' else b['start'],
                    qwen_row['chunk_start'], qwen_row['chunk_end'])
                touched=[e['neighbor_index'] for e in edits]
                lo,hi=max(0,min(touched)-1),min(len(qwen_row['words']),max(touched)+2)
                if not valid(qwen_row['words'][lo:hi], qwen_row['chunk_start'], qwen_row['chunk_end']):
                    raise ValueError('Bounded cross-chunk seam leaves invalid Qwen row')
                if left['words'][-1]['end'] > right['words'][0]['start']+EPS:
                    raise ValueError('Bounded cross-chunk seam still overlaps')
                changes.append(dict(record, adjustments=edits))
            except ValueError as exc:
                failure = mfa_row['id']
                if failure in rolled_back:
                    raise RuntimeError('Bounded rollback made no progress')
                rejected.append(dict(record, reason=str(exc), action='restore_original_qwen_chunk'))
                break
        if failure is None:
            by_id = {r['id']:r for r in trial}
            for row in rows:
                row['words'] = by_id[row['id']]['words']
            return changes, rejected, rolled_back
        rolled_back.add(failure)
