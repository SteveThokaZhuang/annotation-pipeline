"""Render synchronized participant audio and score individual events with JEV."""

from math import gcd
from pathlib import Path


class MixedAudioJudge:
    def __init__(self, checkpoint, device):
        from duplexjev import Decider, Question

        self.engine = Decider.from_pretrained(str(checkpoint), device=device,
                                            attn_implementation="sdpa")
        self.question = Question(
            "event", "At the specified onset, is the target speaker giving supportive "
            "feedback without taking the floor (BACKCHANNEL), trying to claim/change "
            "the floor before the other finishes (INTERRUPT), or neither? "
            "Track who is speaking before onset. A later overlap alone is neither.",
            ["BACKCHANNEL", "INTERRUPT", "NEITHER"], audio="mix")

    def vote(self, case, candidate, waveform, lang):
        start, end = case["interval"]
        onset = candidate["start"]
        transcript = [
            f"{speaker}: " + " ".join(w["text"] for w in words if onset-3 <= w["start"] <= onset+3)
            for speaker, words in case["streams"].items()
        ]
        body = (f"Target: {candidate['initiator']} speaking to {candidate['recipient']}; "
                f"target starts at {onset-start:.3f}s relative to clip, "
                f"ends at {min(candidate['end'], end)-start:.3f}s; "
                f"target words: {candidate['text']}. Nearby transcript:\n" + "\n".join(transcript))
        return self.engine.decide_batch({"mix": waveform}, [self.question], context={"mix": body},
                                        lang=lang, mode="packed", max_items=1, n_perm=4,
                                        seed=0)["mix"]["event"]


def audio_duration(tracks):
    import soundfile as sf

    durations = []
    for path, channel in tracks.values():
        info = sf.info(path)
        if type(channel) is not int or not 0 <= channel < info.channels or info.duration <= 0:
            raise ValueError(f"invalid_source_audio_channel:{path}:{channel}")
        durations.append(info.duration)
    if abs(durations[0] - durations[1]) > .08 + 1e-9:
        raise ValueError("source_track_durations_differ")
    return min(durations)


def render_mix(tracks, interval, destination, rate=16000):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    start, stop = interval
    frames = round((stop-start)*rate)
    if len(tracks) != 2 or not np.isfinite([start, stop]).all() or start < 0 or frames <= 0:
        raise ValueError("invalid_mixed_audio_window")
    waves = []
    for path, channel in tracks.values():
        with sf.SoundFile(path) as track:
            lo, hi = round(start*track.samplerate), round(stop*track.samplerate)
            track.seek(lo)
            samples = track.read(hi-lo, dtype="float32", always_2d=True)[:, channel]
            if not np.isfinite(samples).all():
                raise ValueError(f"nonfinite_audio_samples:{path}")
            if len(samples) != hi-lo:
                raise ValueError(f"incomplete_audio_window:{path}")
            divisor = gcd(rate, track.samplerate)
            if rate != track.samplerate:
                samples = resample_poly(samples, rate//divisor, track.samplerate//divisor)
            if abs(len(samples)-frames) > 1:
                raise ValueError(f"resampled_length_mismatch:{path}")
            waves.append(np.pad(samples[:frames], (0, max(0, frames-len(samples)))))
    channels = np.column_stack(waves)
    gain = min(1.0, .999 / max(.999, float(np.max(np.abs(channels)))))
    sf.write(Path(destination), channels.mean(axis=1)*gain, rate)
    return str(destination)
