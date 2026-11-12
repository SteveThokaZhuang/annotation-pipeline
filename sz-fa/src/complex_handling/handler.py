"""Class entry point for detectors, rule routing and audited shadow candidates."""
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import time

from .audio import SelectedAudio
from .backends import NyraCandidateBackend
from .config import ComplexConfig, LanguagePolicy
from .detectors import (AlignmentRiskDetector, EvidenceDetector, OverlapDetector,
                        PannsEventDetector, PyannoteSpeakerDetector, QualityDetector, TextRiskDetector)
from .evidence import DetectionResult, EvidenceStore
from .router import RuleRouter


def json_safe(data):
    if isinstance(data, float) and not math.isfinite(data):
        return None
    if isinstance(data, dict):
        return {k: json_safe(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_safe(v) for v in data]
    return data


def write_json(path, data):
    Path(path).write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, records):
    with Path(path).open('w') as stream:
        for record in records:
            stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + '\n')


def add_arguments(parser):
    parser.add_argument('--complex-handling', choices=['off', 'shadow'], default='off',
                        help='Enable complex scene detection, language gates and separate shadow candidates')
    parser.add_argument('--complex-config', type=Path, help='Versioned complex_handling JSON config')
    parser.add_argument('--complex-evidence', type=Path, help='Optional per-chunk evidence JSONL keyed by id')


class ComplexHandler:
    """A handler can persist across streaming blocks; model adapters load once."""
    def __init__(self, config=None, evidence_store=None, detectors=None, candidate_backend=None):
        self.config = config or ComplexConfig()
        self.evidence_store = evidence_store or EvidenceStore()
        self.language_policy = LanguagePolicy(self.config.mfa_language)
        self.router = RuleRouter(self.config, self.language_policy)
        self.candidate_backend = candidate_backend
        self._prepared = {}
        self._prepared_seconds = 0.0
        if detectors is None:
            detectors = [EvidenceDetector()]
            if self.config.panns_checkpoint:
                detectors.append(PannsEventDetector(self.config))
            if self.config.pyannote_model:
                detectors.append(PyannoteSpeakerDetector(self.config))
            detectors += [TextRiskDetector(self.config), QualityDetector(self.config),
                          OverlapDetector(self.config), AlignmentRiskDetector(self.config)]
        self.detectors = detectors

    @classmethod
    def from_args(cls, args):
        if getattr(args, 'complex_handling', 'off') == 'off':
            if getattr(args, 'complex_config', None) or getattr(args, 'complex_evidence', None):
                raise ValueError('--complex-config/evidence requires --complex-handling shadow')
            args.complex_handler = None
            return None
        config = ComplexConfig.load(getattr(args, 'complex_config', None))
        for name in ('mfa_dictionary', 'mfa_acoustic_model', 'panns_checkpoint', 'pyannote_model'):
            if getattr(config, name) and not Path(getattr(config, name)).exists():
                raise ValueError('Missing configured local detector model: ' + name)
        if config.mfa_dictionary:
            args.dictionary, args.acoustic_model = Path(config.mfa_dictionary), Path(config.mfa_acoustic_model)
        if config.pyannote_model:
            # Set before the main Qwen/transformers imports initialize hub clients.
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['PYANNOTE_METRICS_ENABLED'] = '0'
        handler = cls(config, EvidenceStore(getattr(args, 'complex_evidence', None)))
        args.complex_handler = handler
        return handler

    def baseline_rejection(self, row, backend):
        rejection = self.language_policy.rejection(row, backend)
        if rejection:
            return rejection
        detection = self._prepared.get(row['id'])
        if detection is not None:
            if detection.errors:
                return 'complex_detection_incomplete'
            if any(e.tag == 'transcript_uncertain' for e in detection.events):
                return 'transcript_uncertain'
            if any(e.tag == 'overlap' for e in detection.events) and detection.observations.get('target_isolated') is not True:
                return 'overlap_requires_target_audio'
        return None

    def prepare(self, predictions, audio_loader=None):
        """Compute evidence before baseline repair, so ambiguous audio can be gated."""
        started = time.perf_counter()
        self._prepared = {row['id']: self.detect(row, audio_loader) for row in predictions}
        self._prepared_seconds = time.perf_counter()-started

    def detect(self, row, audio_loader=None):
        result = DetectionResult()
        audio = SelectedAudio(row, audio_loader, prefer_original=True)
        try:
            evidence = self.evidence_store.get(row)
        except Exception as exc:
            result.errors.append(dict(detector='evidence', error=repr(exc)))
            result.availability['evidence'] = 'error'
            return result
        for detector in self.detectors:
            try:
                detector.detect(row, evidence, result, audio)
            except Exception as exc:
                result.availability[detector.name] = 'error'
                result.errors.append(dict(detector=detector.name, error=repr(exc)))
        return result

    def run(self, predictions, output, audio_loader=None, aligner=None, args=None, nyra_engine=None):
        """Never change the main words; all candidates and decisions are separate."""
        started = time.perf_counter()
        root = Path(output) / 'complex_handling'
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'config.json', self.config.to_dict())
        detection_records, decisions, tasks, candidates = [], [], [], []
        counts = Counter()
        if self.config.candidate_backend == 'nyra' and self.candidate_backend is None and args is not None:
            self.candidate_backend = NyraCandidateBackend(args, aligner, nyra_engine)
        for row_index, row in enumerate(predictions):
            detection = self._prepared.pop(row['id'], None)
            if detection is None:
                detection = self.detect(row, audio_loader)
            else:
                # Keep expensive acoustic results, but refresh word-level risks after repair.
                detection.events = [e for e in detection.events if e.source != 'alignment']
                AlignmentRiskDetector(self.config).detect(row, {}, detection, None)
            routes = self.router.route(row, detection)
            detection_records.append(dict(id=row['id'], timestamp_origin='recording',
                events=[event.to_dict() for event in detection.events],
                observations=detection.observations, availability=detection.availability, errors=detection.errors))
            decisions.extend(route.to_dict() for route in routes)
            counts.update(route.action for route in routes)
            # Merge adjacent selected full sentences; each token is aligned once.
            groups = []
            for route in routes:
                if route.action != 'shadow':
                    continue
                if groups and groups[-1][-1].end_index == route.start_index:
                    groups[-1].append(route)
                else:
                    groups.append([route])
            audio = SelectedAudio(row, audio_loader)
            for group in groups:
                ident = 'c{:06d}'.format(len(tasks))
                try:
                    task = self._task(row, row_index, group, ident, audio, root)
                    # Keep sample selectors rather than a transient crop path in the audit.
                    tasks.append({k: v for k, v in task.items() if k != 'crop_audio'})
                    if self.candidate_backend is None:
                        candidate = dict(task_id=ident, id=row['id'], status='not_run', reason='no_candidate_backend')
                    else:
                        candidate = self.candidate_backend.align(copy.deepcopy(task))
                        candidate.update(task_id=ident, id=row['id'], backend=group[0].backend)
                        try:
                            if candidate.get('skipped'):
                                raise ValueError('Shadow candidate skipped input tokens')
                            if candidate.get('alignment_error'):
                                raise ValueError(candidate['alignment_error'])
                            self._validate(row, task, candidate['candidate_words'])
                            candidate.update(status='shadow_accepted', applied=False)
                        except Exception as exc:
                            candidate.update(status='rejected', applied=False, reason=str(exc))
                except Exception as exc:
                    if not tasks or tasks[-1]['task_id'] != ident:
                        tasks.append(dict(task_id=ident, id=row['id'], start_index=group[0].start_index,
                            end_index=group[-1].end_index, preparation_error=repr(exc)))
                    candidate = dict(task_id=ident, id=row['id'], status='failed', applied=False, reason=repr(exc))
                candidate.setdefault('applied', False)
                candidates.append(candidate)
                counts[candidate['status']] += 1
        write_jsonl(root/'detections.jsonl', detection_records)
        write_jsonl(root/'routes.jsonl', decisions)
        write_jsonl(root/'tasks.jsonl', tasks)
        write_jsonl(root/'candidates.jsonl', candidates)
        stats = dict(mode='shadow', chunks=len(predictions), counts=dict(counts),
                     rule_version=1,
                     candidate_tasks=len(tasks), seconds=time.perf_counter()-started+self._prepared_seconds,
                     preparation_seconds=self._prepared_seconds,
                     applied_candidates=0, timestamp_origin='recording',
                     detector_errors=sum(len(d['errors']) for d in detection_records),
                     evidence_path=self.evidence_store.path,
                     nyra_model_load_seconds=sum(c.get('model_load_seconds', 0) for c in candidates))
        write_json(root/'runtime.json', stats)
        self._prepared.clear()
        self._prepared_seconds = 0.0
        return stats

    def _task(self, row, row_index, group, ident, audio, root):
        import soundfile as sf
        lo, hi = group[0].start_index, group[-1].end_index
        if not 0 <= lo < hi <= len(row['words']):
            raise ValueError('Sentence token span is outside the aligned word sequence')
        words = row['words'][lo:hi]
        if not words or any(not math.isfinite(w[k]) for w in words for k in ('start', 'end')):
            raise ValueError('Cannot crop a sentence without finite word boundaries')
        waveform, sr = audio()
        a = min(w['start'] for w in words) - row['chunk_start']
        b = max(w['end'] for w in words) - row['chunk_start']
        # Include observed events, but never borrow audio outside this ASR chunk.
        for route in group:
            for event in route.events:
                if event['tag'] != 'structural_error':
                    a = min(a, event['start']-row['chunk_start'])
                    b = max(b, event['end']-row['chunk_start'])
        padding = self.config.crop_padding_seconds
        start = max(0, math.floor((a-padding)*sr))
        end = min(len(waveform), math.ceil((b+padding)*sr))
        if end <= start:
            raise ValueError('Empty complex sentence crop')
        corpus = root/'corpus'
        corpus.mkdir(exist_ok=True)
        crop = corpus/(ident+'.wav')
        sf.write(crop, waveform[start:end], sr, subtype='PCM_16')
        spans = [s for s in row.get('sentences', []) if lo <= s['start_index'] and s['end_index'] <= hi]
        return dict(task_id=ident, id=row['id'], row_index=row_index, start_index=lo, end_index=hi,
            sentences=spans, original_words=copy.deepcopy(words), text=' '.join(w['text'] for w in words),
            risk_tags=sorted({tag for route in group for tag in route.risk_tags}),
            crop_audio=str(crop.resolve()), crop_start=row['chunk_start']+start/sr,
            crop_end=row['chunk_start']+end/sr, crop_start_sample=start, crop_end_sample=end, sample_rate=sr,
            source_audio=row['audio'], source_channel=row.get('channel'),
            source_start_sample=row.get('start_sample', 0), timestamp_origin='recording')

    @staticmethod
    def _validate(row, task, words):
        lo, hi = task['start_index'], task['end_index']
        if [w['text'] for w in words] != [w['text'] for w in task['original_words']]:
            raise ValueError('Shadow candidate changes text/token coverage')
        left = max(row['chunk_start'], row['words'][lo-1]['end'] if lo else row['chunk_start'])
        right = min(row['chunk_end'], row['words'][hi]['start'] if hi < len(row['words']) else row['chunk_end'])
        if not math.isfinite(left) or not math.isfinite(right) or right <= left:
            raise ValueError('Invalid unchanged-neighbor boundary')
        previous = max(left, task['crop_start'])
        upper = min(right, task['crop_end'])
        for word in words:
            a, b = word['start'], word['end']
            if not math.isfinite(a) or not math.isfinite(b) or b <= a or a < previous-1e-7 or b > upper+1e-7:
                raise ValueError('Invalid shadow candidate boundaries; no clamping is applied')
            previous = b
