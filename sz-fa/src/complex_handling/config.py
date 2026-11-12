"""Versioned, explicit configuration for complex-scene routing."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class ComplexConfig:
    version: int = 1
    event_on: float = 0.6
    event_off: float = 0.35
    event_min_seconds: float = 0.15
    event_merge_seconds: float = 0.2
    event_margin_seconds: float = 0.3
    smoothing_seconds: float = 0.05
    overlap_min_seconds: float = 0.15
    disagreement_threshold: float = 0.2
    background_mos_threshold: float = 2.5
    long_word_seconds: float = 2.0
    vad_min_word_seconds: float = 0.3
    vad_min_coverage: float = 0.2
    crop_padding_seconds: float = 0.35
    mfa_language: str = 'en'
    mfa_dictionary: str = ''
    mfa_acoustic_model: str = ''
    candidate_backend: str = 'nyra'
    panns_checkpoint: str = ''
    pyannote_model: str = ''
    detector_device: str = 'cpu'

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('Unsupported complex_handling config version')
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))
                                        or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid nonnegative finite setting: ' + field.name)
        for name in ('event_on', 'event_off', 'disagreement_threshold', 'vad_min_coverage'):
            if getattr(self, name) > 1:
                raise ValueError(name + ' must be in [0, 1]')
        if not 0 < self.event_on <= 1 or self.event_off > self.event_on:
            raise ValueError('Require 0 <= event_off <= event_on and event_on > 0')
        if self.background_mos_threshold > 5:
            raise ValueError('background_mos_threshold must be in [0, 5]')
        if self.candidate_backend not in ('nyra', 'none'):
            raise ValueError('candidate_backend must be nyra or none')
        if self.mfa_language not in ('en', 'zh', 'disabled'):
            raise ValueError('mfa_language must be en, zh or disabled')
        if bool(self.mfa_dictionary) != bool(self.mfa_acoustic_model):
            raise ValueError('Declare both mfa_dictionary and mfa_acoustic_model')
        if self.mfa_language == 'zh' and not self.mfa_dictionary:
            raise ValueError('Chinese MFA requires an explicit model/dictionary pair')
        for name in ('mfa_dictionary', 'mfa_acoustic_model', 'panns_checkpoint', 'pyannote_model', 'detector_device'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(name + ' must be a string')

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('Complex config must be a JSON object')
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Unknown complex config settings: ' + ', '.join(sorted(unknown)))
        for name in ('mfa_dictionary', 'mfa_acoustic_model', 'panns_checkpoint', 'pyannote_model'):
            if data.get(name):
                value = Path(data[name]).expanduser()
                data[name] = str((path.parent / value).resolve()) if not value.is_absolute() else str(value)
        return cls(**data)

    def to_dict(self):
        return asdict(self)


class LanguagePolicy:
    """Check declared input language, never a Qwen fallback language alone."""
    def __init__(self, mfa_language='en'):
        self.mfa_language = mfa_language

    def language(self, row):
        value = row.get('input_language') if 'input_language' in row else row.get('language')
        value = str(value or '').strip().casefold().replace('_', '-')
        aliases = {'english': 'en', 'en-us': 'en', 'en-gb': 'en',
                   'chinese': 'zh', 'mandarin': 'zh', 'zh-cn': 'zh', 'zh-hans': 'zh'}
        value = aliases.get(value, value)
        if row.get('language_verified') is False or value not in ('en', 'zh'):
            return 'unknown'
        # Reject an obvious mixed/non-English transcript even if upstream says en.
        if value == 'en' and any('\u3400' <= c <= '\u9fff' for c in row.get('text', '')):
            return 'mixed'
        if value == 'zh' and re.search(r'[A-Za-z]', row.get('text', '')):
            return 'mixed'
        return value

    def rejection(self, row, backend):
        if backend not in ('nyra', 'mfa'):
            raise ValueError('Unknown alignment backend: ' + backend)
        language = self.language(row)
        supported = ('en',) if backend == 'nyra' else (self.mfa_language,)
        if language not in supported or language in ('unknown', 'mixed', 'disabled'):
            return 'unsupported_language:' + backend + ':' + language
        return None
