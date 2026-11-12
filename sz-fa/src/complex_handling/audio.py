"""Read exactly one selected channel/segment, without changing time coordinates."""
import math


class SelectedAudio:
    """Lazily read a row's segment; a separate original risk_audio is optional."""
    def __init__(self, row, loader=None, prefer_original=False):
        self.row, self.loader = row, loader
        self.override = row.get('risk_audio') if prefer_original else None
        self.value = None
        self.source = 'risk_audio' if self.override else 'fa_audio'

    def __call__(self):
        if self.value is not None:
            return self.value
        import numpy as np
        import soundfile as sf
        row = self.row
        if self.override:
            item = self.override
            # Explicit sample coordinates belong to this file's own sample rate.
            required = {'path', 'sample_rate', 'start_sample', 'end_sample', 'channel'}
            if not isinstance(item, dict) or not required <= item.keys():
                raise ValueError('risk_audio needs path, sample_rate, start_sample, end_sample, channel')
            path, start, end, sr, channel = (item[k] for k in
                ('path', 'start_sample', 'end_sample', 'sample_rate', 'channel'))
        elif self.loader:
            waveform, sr = self.loader(row)
            waveform = np.asarray(waveform)
            if waveform.ndim == 2:
                if waveform.shape[1] != 1:
                    raise ValueError('audio_loader must select one channel explicitly')
                waveform = waveform[:, 0]
            self.value = self._validate(waveform, sr)
            return self.value
        else:
            path = row['audio']
            channel = row.get('channel')
            if row.get('audio_scope') == 'recording':
                start, end, sr = (row[k] for k in ('start_sample', 'end_sample', 'source_sample_rate'))
            else:
                info = sf.info(path)
                start, end, sr = 0, info.frames, info.samplerate
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end, sr)):
            raise ValueError('Audio sample bounds/rate must be integers')
        if not 0 <= start < end or sr <= 0:
            raise ValueError('Invalid selected audio sample bounds')
        with sf.SoundFile(path) as stream:
            if stream.samplerate != sr or end > stream.frames:
                raise ValueError('Audio sample rate/bounds do not match the selected file')
            if channel is None:
                if stream.channels != 1:
                    raise ValueError('Multi-channel audio requires an explicit channel')
                channel = 0
            if not isinstance(channel, int) or isinstance(channel, bool) or not 0 <= channel < stream.channels:
                raise ValueError('Invalid audio channel')
            stream.seek(start)
            waveform = stream.read(end-start, dtype='float32', always_2d=True)[:, channel]
        self.value = self._validate(waveform, sr)
        return self.value

    def _validate(self, waveform, sr):
        import numpy as np
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError('Expected nonempty finite mono audio')
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError('Expected positive integer sample rate')
        if not math.isclose(len(waveform)/sr, self.row['duration'], abs_tol=max(2/sr, .002)):
            raise ValueError('Selected audio duration differs from the FA chunk')
        return waveform, sr
