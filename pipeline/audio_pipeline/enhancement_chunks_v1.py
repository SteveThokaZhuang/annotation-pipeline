"""VAD-guided bounded enhancement geometry, globally aligned resampling and joins.

All coordinates are integer samples at the model rate. Silence is a preferred
cut location, never permission to discard samples. This module is versioned only
by the opt-in chunked enhancement adapter.
"""
import math

from .common import digest

METHOD = 'vad_chunked_v1'
DEFAULTS = dict(target_seconds=120., max_segment_seconds=180., min_segment_seconds=60.,
                boundary_search_seconds=15., min_silence_seconds=.3,
                context_seconds=2., crossfade_seconds=.25)


def policy(settings):
    if settings.get('method_version') not in (METHOD,'auto_v3'):
        raise ValueError('Chunked enhancement requires method_version: ' + METHOD)
    result = {key:float(settings.get(key, value)) for key,value in DEFAULTS.items()}
    if not all(math.isfinite(v) and v >= 0 for v in result.values()):
        raise ValueError('Segment settings must be finite and nonnegative')
    if not (result['context_seconds'] > 0 and result['crossfade_seconds'] > 0 and
            result['min_silence_seconds'] > 0 and
            result['crossfade_seconds'] < result['min_segment_seconds'] <= result['target_seconds']):
        raise ValueError('Require positive context/fade/silence and fade < minimum <= target')
    if result['target_seconds'] + 2*result['context_seconds'] + result['crossfade_seconds'] >= result['max_segment_seconds']:
        raise ValueError('Maximum segment must include target, context and crossfade')
    return result


def plan_recording(item, vad, settings, rate=48000):
    p = policy(settings)
    if vad.get('recording_id') != item['id']:
        raise ValueError('VAD recording identity mismatch')
    source = vad.get('_provenance',{}).get('input',{}).get('source_fingerprint')
    if source != item['source_fingerprint']:
        raise ValueError('VAD source differs from ingest; rerun VAD')
    channel_ids = [row['channel'] for row in vad['channels']]
    if sorted(channel_ids) != list(range(item['channels'])):
        raise ValueError('Need raw VAD intervals for every channel')
    total = round(item['total_samples'] * rate / item['sample_rate'])
    if total < 1:
        raise ValueError('Empty recording')
    speech = []
    for channel in vad['channels']:
        for begin,end in channel['intervals']:
            begin,end = float(begin),float(end)
            if not (math.isfinite(begin) and math.isfinite(end) and 0 <= begin < end <= item['duration'] + .03):
                raise ValueError('Invalid raw VAD interval')
            a,b = max(0,round(begin*rate)),min(total,round(end*rate))
            if a < b:speech.append((a,b))
    merged = []
    for a,b in sorted(speech):
        if merged and a <= merged[-1][1]:merged[-1][1] = max(b,merged[-1][1])
        else:merged.append([a,b])
    gaps = [];cursor = 0
    for a,b in merged:
        if a > cursor:gaps.append((cursor,a))
        cursor = b
    if cursor < total:gaps.append((cursor,total))
    context = math.ceil(p['context_seconds']*rate)
    half = max(1,round(p['crossfade_seconds']*rate/2))
    minimum = math.ceil(p['min_segment_seconds']*rate)
    target = round(p['target_seconds']*rate)
    maximum = math.floor(p['max_segment_seconds']*rate)-2*context-2*half
    if maximum <= target or minimum <= 2*half:
        raise ValueError('Invalid segment geometry after sample rounding')
    search = round(p['boundary_search_seconds']*rate)
    quiet_margin = max(1,math.ceil(p['min_silence_seconds']*rate/2))
    cuts = [0];reasons = []
    while total-cuts[-1] > maximum:
        start = cuts[-1];ideal = start+target
        lower,upper = start+minimum,min(start+maximum,total-minimum)
        if lower > upper:
            raise ValueError('Cannot partition recording within configured bounds')
        def candidate(lo,hi):
            choices = []
            for a,b in gaps:
                left,right = max(lo,a+quiet_margin),min(hi,b-quiet_margin)
                if left <= right:
                    value = min(max(ideal,left),right)
                    choices.append((abs(value-ideal),value))
            return min(choices)[1] if choices else None
        cut = candidate(max(lower,ideal-search),min(upper,ideal+search))
        if cut is None:cut = candidate(lower,upper)
        reasons.append('all_channel_silence' if cut is not None else 'bounded_continuous_speech')
        cuts.append(cut if cut is not None else min(max(ideal,lower),upper))
    cuts.append(total)
    pieces = []
    for index,(a,b) in enumerate(zip(cuts,cuts[1:])):
        output_start = a if index == 0 else a-half
        output_end = b if index == len(cuts)-2 else b+half
        read_start,read_end = max(0,output_start-context),min(total,output_end+context)
        if read_end-read_start > math.floor(p['max_segment_seconds']*rate):
            raise ValueError('Segment exceeds bounded inference duration')
        pieces.append(dict(index=index,core_start=a,core_end=b,
                           output_start=output_start,output_end=output_end,
                           read_start=read_start,read_end=read_end))
    return dict(method_version=METHOD,rate=rate,total_samples=total,channels=item['channels'],
                source_fingerprint=item['source_fingerprint'],vad_version=digest(vad),
                policy=p,half_crossfade=half,boundaries=cuts,boundary_reasons=reasons,pieces=pieces)


def read_segment(item, piece, channel, rate=48000):
    """Resample on the global rational grid; FIR guard samples are discarded.

    Aligning the native start to the resampling denominator avoids a new phase
    origin on every segment. Extra filter support makes cropped interiors agree
    with a full-recording scipy resample_poly call, without loading that recording.
    """
    from .audio import audio_source,_read_audio
    factor = math.gcd(rate,item['sample_rate'])
    up,down = rate//factor,item['sample_rate']//factor
    start,end = piece['read_start'],piece['read_end']
    guard = math.ceil(10*max(up,down)/up)+2 if up != down else 0
    native_start = max(0,((start*down//up-guard)//down)*down)
    native_end = min(item['total_samples'],math.ceil(end*down/up)+guard)
    # Do not let the optional decoded-recording RAM cache read the whole source.
    with audio_source(item['path']) as local:
        audio,actual_rate = _read_audio(local,channel=channel,rate=rate,
                                       start_sample=native_start,end_sample=native_end)
    offset = start-native_start*up//down
    audio = audio[offset:offset+end-start]
    if actual_rate != rate or len(audio) != end-start:
        raise ValueError('Incomplete globally aligned segment decode')
    return audio


def join_weights(piece, plan, begin, end):
    import numpy as np
    positions = np.arange(begin,end,dtype=np.float64)
    weights = np.ones(end-begin,dtype=np.float64)
    half = plan['half_crossfade'];width = 2*half-1
    if piece['index'] > 0:
        weights *= np.clip((positions-(piece['core_start']-half))/width,0,1)
    if piece['index'] < len(plan['pieces'])-1:
        weights *= 1-np.clip((positions-(piece['core_end']-half))/width,0,1)
    return weights
