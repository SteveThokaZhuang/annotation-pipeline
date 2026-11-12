"""Bounded, single-decode preparation of complete recording channels."""
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from audio_pipeline.audio import io_slot
from audio_pipeline.local_cache import cached_source
import json
import os
from contextlib import nullcontext


def prepare(item, max_gib):
    # Include native and resampled arrays plus resampler working headroom.
    estimated = item['duration'] * item['channels'] * (item['sample_rate'] + 48000) * 4 * 2
    if estimated > max_gib * 2**30:
        return None  # Adapter retains its per-channel whole-sequence path.
    cache = json.loads(os.environ.get('AUDIO_PIPELINE_CACHE', '{}'))
    source = cached_source(item['path'], cache['root'], int(cache.get('max_gib',20)*2**30)) if cache.get('enabled') else nullcontext(item['path'])
    with source as path, io_slot(), sf.SoundFile(path) as stream:
        if (stream.samplerate, stream.channels, len(stream)) != (item['sample_rate'], item['channels'], item['total_samples']):
            raise ValueError('Recording format changed after ingestion')
        audio = stream.read(dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or nonfinite recording')
    if item['sample_rate'] != 48000:
        factor = math.gcd(48000, item['sample_rate'])
        audio = resample_poly(audio,48000//factor,item['sample_rate']//factor,axis=0).astype('float32')
    return audio
