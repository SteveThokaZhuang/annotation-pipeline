"""Shared IDs, interval preparation and atomic reports; no ML dependencies."""
import json
import math
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
RATE = 16000


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_inputs(paths, manifest=None, conversation_id='conversation_000'):
    if manifest:
        location = Path(manifest).resolve()
        rows = json.loads(location.read_text())['conversations']
        relative_to = location.parent
    else:
        if len(paths) != 2:
            raise ValueError('Supply two channel files, or --manifest with conversations')
        rows = [dict(id=conversation_id, channels={'1':paths[0], '2':paths[1]})]
        relative_to = Path.cwd()
    if not rows:
        raise ValueError('No conversations supplied')
    seen = set()
    result = []
    for row in rows:
        identity = row['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identity) or identity in seen:
            raise ValueError(f'Conversation ID must be unique and contain letters/numbers/_/-: {identity}')
        seen.add(identity)
        if set(row['channels']) != {'1','2'}:
            raise ValueError(f'{identity}: channels must contain exactly "1" and "2"')
        channels = {}
        for channel, name in row['channels'].items():
            path = Path(name).expanduser()
            path = (relative_to / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise ValueError(f'Missing audio: {path}')
            channels[channel] = str(path)
        if channels['1'] == channels['2']:
            raise ValueError(f'{identity}: supply distinct channel files')
        result.append(dict(id=identity, channels=channels))
    return result


def merge_intervals(intervals, gap=0):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap + 1e-9:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def split_policy(max_seconds, min_chunk_seconds=10., min_silence_seconds=.1,
                 fallback_search_seconds=2., energy_window_ms=100.):
    """Validate settings shared by config loading, planning and the splitter."""
    values = (max_seconds, min_chunk_seconds, min_silence_seconds,
              fallback_search_seconds, energy_window_ms)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or
           not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('VAD split settings must be finite positive numbers')
    if not min_chunk_seconds <= max_seconds <= 30:
        raise ValueError('Require min_chunk_seconds <= chunk_seconds <= 30')
    if fallback_search_seconds > max_seconds or energy_window_ms/1000 > fallback_search_seconds:
        raise ValueError('VAD energy window must fit fallback search, which must fit chunk_seconds')


def quiet_energy_cut(audio, lower, upper, energy_window_ms=100.):
    """Choose the midpoint of the quietest full energy window (16 kHz)."""
    import numpy as np
    window = max(1, round(energy_window_ms*RATE/1000))
    x = np.asarray(audio[lower:upper],dtype=np.float64)
    if len(x) < window:return (lower+upper)//2
    # Search every 10 ms, averaging a sustained window rather than one frame.
    starts = np.arange(0,len(x)-window+1,max(1,RATE//100))
    sums = np.concatenate(([0.],np.cumsum(x*x)))
    energies = (sums[starts+window]-sums[starts])/window
    return lower+int(starts[int(np.argmin(energies))])+window//2


def make_regions(intervals, duration, padding, merge_gap, max_seconds, quiet_cut=None,
                 *, min_chunk_seconds=None, min_silence_seconds=.1,
                 fallback_search_seconds=2., return_boundaries=False):
    """Preserve raw intervals; prefer silence before bounded acoustic fallback.

    Omitting min_chunk_seconds preserves legacy splitting for standalone callers.
    Returned region coordinates remain integer 16 kHz samples. Boundary metadata
    uses global seconds and is optional for callers expecting the original pair.
    """
    if min_chunk_seconds is not None:
        split_policy(max_seconds,min_chunk_seconds,min_silence_seconds,fallback_search_seconds)
    clipped = []
    for start, end in intervals:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + .03):
            raise ValueError(f'Invalid VAD interval {(start,end)} for duration {duration}')
        end = min(end, duration)
        if start < end:clipped.append((start,end))
    raw = merge_intervals(clipped)
    gaps = [(round(a[1]*RATE),round(b[0]*RATE)) for a,b in zip(raw,raw[1:])
            if b[0]-a[1]+1e-9 >= min_silence_seconds]
    padded = [(max(0,s-padding), min(duration,e+padding))
              for s,e in merge_intervals(raw,merge_gap)]
    regions = []; boundaries = []
    for start,end in merge_intervals(padded):
        start_frame,end_frame = round(start*RATE),round(end*RATE)
        limit = int(max_seconds*RATE)
        start_boundary = dict(type='region_start',time=start_frame/RATE)
        while end_frame-start_frame > limit:
            upper = start_frame+limit
            cut = None
            if min_chunk_seconds is not None:
                earliest = start_frame+math.ceil(min_chunk_seconds*RATE)
                # Prefer the latest complete silence midpoint within this chunk.
                choices = [( (a+b)//2,a,b) for a,b in gaps
                           if a >= start_frame and b <= end_frame and earliest <= (a+b)//2 <= upper]
                if choices:
                    cut,a,b = max(choices)
                    boundary = dict(type='vad_silence',time=cut/RATE,
                                    silence_start=a/RATE,silence_end=b/RATE)
            if cut is None:
                lower = max(start_frame+limit//2,upper-round(fallback_search_seconds*RATE))
                cut = quiet_cut(lower,upper) if quiet_cut else upper
                if not lower <= cut <= upper or cut <= start_frame:
                    raise ValueError('Invalid quiet split point')
                boundary = dict(type='forced_low_energy' if quiet_cut else 'forced_limit',
                                time=cut/RATE,reason='no_suitable_vad_silence',
                                search_start=lower/RATE,search_end=upper/RATE)
            regions.append((start_frame,cut))
            boundaries.append(dict(start_boundary=start_boundary,end_boundary=boundary))
            start_frame = cut; start_boundary = dict(boundary)
        if end_frame > start_frame:
            regions.append((start_frame,end_frame))
            boundaries.append(dict(start_boundary=start_boundary,
                                   end_boundary=dict(type='region_end',time=end_frame/RATE)))
    return (raw,regions,boundaries) if return_boundaries else (raw,regions)
