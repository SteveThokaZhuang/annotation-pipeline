"""Direct Nyra fallback or Qwen -> MFA -> Nyra, with atomic sentence repair."""
import collections
import ctypes
import json
import os
from pathlib import Path
import sys
import time

from core import read_jsonl, write_json, write_jsonl
from process import audit, word_issues
from repair import SentenceRepair, map_mfa_words, prepare_sentence_tasks, splice_sentence


def nyra_text(words):
    return ' '.join({'um':'[UM]', 'uh':'[UH]'}.get(w['text'].casefold(), w['text']) for w in words)


def nyra_candidate(expected, result, offset):
    if result.skipped:
        raise ValueError('Nyra skipped input tokens: '+repr(result.skipped))
    entries = [(float(w.start), float(w.end), w.word) for w in result.words]
    try:
        return map_mfa_words(expected, entries, offset, source='nyra')
    except ValueError as exc:
        raise ValueError(str(exc).replace('MFA', 'Nyra')) from exc


def configure_nyra(args):
    sys.path.insert(0, str(args.nyra_runtime.resolve()))
    # Set paths for a local espeak installation without changing the host.
    prefix = args.nyra_espeak.resolve()
    library_dir = prefix/'usr/lib/x86_64-linux-gnu'
    pcaudio = library_dir/'libpcaudio.so.0'
    if pcaudio.exists():
        pulse = prefix.parent/'mfa/lib/libpulse-simple.so.0'
        if pulse.exists():
            ctypes.CDLL(str(pulse), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(pcaudio), mode=ctypes.RTLD_GLOBAL)
    sonic = library_dir/'libsonic.so.0'
    if sonic.exists():
        ctypes.CDLL(str(sonic), mode=ctypes.RTLD_GLOBAL)
    espeak = library_dir/'libespeak-ng.so.1'
    if espeak.exists():
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(espeak)
        os.environ['PHONEMIZER_ESPEAK_DATA_PATH'] = str(prefix/'usr/lib/x86_64-linux-gnu/espeak-ng-data')
    # Both models are local, including Nyra's bundled WavLM front-end.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


class NyraRepair:
    def __init__(self, args, aligner):
        self.args = args
        self.aligner = aligner
        self._nyra_model = None
        self._nyra_load_error = None

    def _load_model(self):
        if self._nyra_model is not None:
            return self._nyra_model, 0.0
        if self._nyra_load_error:
            raise RuntimeError('Nyra model loading previously failed: '+self._nyra_load_error)
        started = time.perf_counter()
        try:
            configure_nyra(self.args)
            from nyra_align import Aligner
            model = Aligner(model=self.args.nyra_checkpoint, device=self.args.device, precision='fp32')
            _ = model.wavlm
            if self.args.device.startswith('cuda'):
                self.aligner.torch.cuda.synchronize()
        except Exception as exc:
            self._nyra_load_error = repr(exc)
            raise
        self._nyra_model = model
        return model, time.perf_counter()-started

    def shadow_candidate(self, task):
        """Produce a checked lexical candidate without modifying any main row."""
        model, loaded = self._load_model()
        started = time.perf_counter()
        result = model.align(task['crop_audio'], nyra_text(task['original_words']))
        if self.args.device.startswith('cuda'):
            self.aligner.torch.cuda.synchronize()
        raw = dict(raw_words=[dict(text=w.word, start=float(w.start), end=float(w.end), type=w.type)
                              for w in result.words], skipped=result.skipped,
                   model_load_seconds=loaded, inference_seconds=time.perf_counter()-started)
        # Keep raw rejected results for diagnosis, including skipped input tokens.
        try:
            raw['candidate_words'] = nyra_candidate(task['original_words'], result, task['crop_start'])
        except ValueError as exc:
            raw['candidate_words'] = []
            raw['alignment_error'] = str(exc)
        return raw

    def repair(self, predictions):
        started = time.perf_counter()
        mfa_stats = None
        if self.args.method == 'qwen+mfa+nyra':
            mfa_stats = SentenceRepair(self.args, self.aligner).repair(predictions)
            write_json(self.args.output/'mfa/unresolved.json', json.loads((self.args.output/'unresolved.json').read_text()))
            after_mfa = audit(predictions)
            write_jsonl(self.args.output/'mfa_predictions.jsonl', predictions)
            write_json(self.args.output/'mfa_error_stats.json', after_mfa)
            # Reuse rejected groups/crops, auditing against current post-MFA times.
            tasks = [r for r in read_jsonl(self.args.output/'mfa/repairs.jsonl') if r['status']=='unresolved']
            current={r['id']:r for r in predictions}
            for task in tasks:
                task['mfa_original_words']=task['original_words']
                task['original_words']=[dict(w) for w in current[task['id']]['words'][task['start_index']:task['end_index']]]
                task['mfa_rejection_reason'] = task.pop('reason')
                task.pop('status')
                task['crop_audio'] = str((self.args.output/'mfa/corpus'/(task['task_id']+'.wav')).resolve())
            unresolved = []
            root = self.args.output/'nyra'
            root.mkdir()
            write_jsonl(root/'tasks.jsonl', tasks)
        else:
            root = self.args.output/'nyra'
            root.mkdir()
            tasks, unresolved = prepare_sentence_tasks(predictions, root, self.args.padding,
                getattr(self.args, 'audio_loader', None),
                task_guard=(self.args.complex_handler.baseline_rejection
                            if getattr(self.args, 'complex_handler', None) else None), backend='nyra')
            for task in tasks:
                task['crop_audio'] = str((root/'corpus'/(task['task_id']+'.wav')).resolve())
            write_jsonl(root/'tasks.jsonl', tasks)
        if getattr(self.args, 'complex_handler', None):
            for task in tasks:
                row = predictions[task['row_index']]
                rejection = self.args.complex_handler.baseline_rejection(row, 'nyra')
                if rejection:
                    task['preparation_error'] = rejection
                elif task.get('preparation_error', '').startswith('unsupported_language:mfa:'):
                    # MFA may have been deliberately disabled or use another language.
                    # Its guard creates no crop; make one only after the Nyra gate passes.
                    try:
                        import math
                        import soundfile as sf
                        from complex_handling.audio import SelectedAudio
                        audio, sr = SelectedAudio(row, getattr(self.args, 'audio_loader', None))()
                        words = task['original_words']
                        start = max(0, math.floor((min(w['start'] for w in words)-row['chunk_start']-self.args.padding)*sr))
                        end = min(len(audio), math.ceil((max(w['end'] for w in words)-row['chunk_start']+self.args.padding)*sr))
                        if end <= start:
                            raise ValueError('Empty sentence crop after MFA language skip')
                        (root/'corpus').mkdir(exist_ok=True)
                        crop = root/'corpus'/(task['task_id']+'.wav')
                        sf.write(crop, audio[start:end], sr, subtype='PCM_16')
                        task.update(crop_audio=str(crop.resolve()), crop_start=row['chunk_start']+start/sr,
                                    crop_end=row['chunk_start']+end/sr)
                        task.pop('preparation_error')
                    except Exception as exc:
                        task['preparation_error'] = repr(exc)
            write_jsonl(root/'tasks.jsonl', tasks)
        print(f'Nyra: {len(tasks)} complete sentence groups requested', flush=True)
        accepted, rejected, raw_results = [], [], []
        model = self._nyra_model
        load_seconds = inference_seconds = 0.0
        load_error = None
        if any(not t.get('preparation_error') for t in tasks) and model is None:
            load_started = time.perf_counter()
            try:
                model, _ = self._load_model()
            except Exception as exc:
                load_error = repr(exc)
            load_seconds = time.perf_counter()-load_started
        for index, task in enumerate(tasks):
            row = predictions[task['row_index']]
            record = dict(task)
            raw = dict(task_id=task['task_id'], id=task['id'], words=[], skipped=[])
            call_started = time.perf_counter()
            try:
                if load_error:
                    raise RuntimeError('Nyra model loading failed: '+load_error)
                if task.get('preparation_error'):
                    raise ValueError(task['preparation_error'])
                result = model.align(task['crop_audio'], nyra_text(task['original_words']))
                if self.args.device.startswith('cuda'):
                    self.aligner.torch.cuda.synchronize()
                raw.update(words=[dict(text=w.word, start=float(w.start), end=float(w.end), type=w.type)
                                  for w in result.words], skipped=result.skipped)
                replacement = nyra_candidate(task['original_words'], result, task['crop_start'])
                record['candidate_words'] = replacement
                updated, bounds = splice_sentence(row['words'], task['start_index'], task['end_index'],
                                                  replacement, row['chunk_start'], row['chunk_end'])
                row['words'] = updated
                record.update(status='repaired', source='nyra', replacement_words=updated[task['start_index']:task['end_index']],
                              boundary_limits=bounds)
                accepted.append(record)
            except Exception as exc:
                record.update(status='unresolved', source='nyra', reason=str(exc))
                raw['error'] = repr(exc)
                rejected.append(record)
                unresolved.append(record)
            elapsed = time.perf_counter()-call_started
            inference_seconds += elapsed
            raw['seconds'] = elapsed
            raw_results.append(raw)
            if (index+1)%10 == 0 or index+1 == len(tasks):
                print(f'Nyra {index+1}/{len(tasks)} groups; accepted={len(accepted)}, rejected={len(rejected)}', flush=True)
        residual = []
        for row in predictions:
            issues = word_issues(row['words'], row['chunk_start'], row['chunk_end'])
            if issues or row.get('error'):
                residual.append(dict(id=row['id'], audio=row['audio'], source_audio=row.get('source_audio'),
                                     text=row['text'], issues=issues, error=row.get('error')))
        write_jsonl(root/'predictions.jsonl', raw_results)
        write_jsonl(root/'repairs.jsonl', accepted+rejected)
        write_json(self.args.output/'unresolved.json', dict(rejected_sentences=unresolved, residual_chunks=residual))
        requested_sentences = sum(len(t['sentences']) for t in tasks)
        stats = dict(method=self.args.method, mfa=mfa_stats, sentence_groups=len(tasks),
            triggered_sentences=requested_sentences, accepted_groups=len(accepted), rejected_groups=len(rejected),
            repaired_sentences=sum(len(t['sentences']) for t in accepted),
            fixed_zero_words=sum(len(t['zero_indices']) for t in accepted),
            replaced_words=sum(t['end_index']-t['start_index'] for t in accepted),
            sentence_repair_success_rate=sum(len(t['sentences']) for t in accepted)/requested_sentences if tasks else None,
            rejection_reasons=dict(collections.Counter(t['reason'] for t in rejected)),
            nyra_checkpoint=str(self.args.nyra_checkpoint.resolve()), nyra_precision='fp32',
            nyra_model_load_seconds=load_seconds, nyra_inference_seconds=inference_seconds,
            nyra_load_error=load_error, seconds=time.perf_counter()-started)
        write_json(root/'runtime.json', stats)
        print(json.dumps({k:v for k,v in stats.items() if k!='mfa'},ensure_ascii=False), flush=True)
        return stats


def run_nyra_mode(method):
    from process import ROOT, parser, run
    p = parser()
    p.description = method + ': zero-duration trigger, complete-sentence fallback'
    p.add_argument('--nyra-checkpoint', type=Path, default=ROOT/'models/nyra-forced-aligner/checkpoints')
    p.add_argument('--nyra-runtime', type=Path, default=ROOT/'models/nyra')
    p.add_argument('--nyra-espeak', type=Path, default=ROOT/'models/nyra-espeak')
    p.add_argument('--padding', type=float, default=0.35)
    p.add_argument('--mfa-python', type=Path, default=ROOT/'models/mfa/bin/python')
    p.add_argument('--dictionary', type=Path, default=ROOT/'models/mfa-english/dictionary/english_us_arpa.dict')
    p.add_argument('--acoustic-model', type=Path, default=ROOT/'models/mfa-english/english_us_arpa.zip')
    p.add_argument('--mfa-jobs', type=int, default=8)
    p.add_argument('--mfa-timeout', type=int, default=1800)
    p.add_argument('--mfa-repair-scope', choices=['sentence','chunk'], default='chunk')
    p.add_argument('--mfa-seam-policy', choices=['midpoint','clamp','bounded'], default='midpoint')
    p.add_argument('--no-mfa-sentence-fallback', action='store_false',dest='mfa_sentence_fallback',default=True)
    args = p.parse_args()
    args.method = method
    required = [args.nyra_checkpoint/'model.npz', args.nyra_checkpoint/'wavlm/model.safetensors']
    if method == 'qwen+mfa+nyra':
        required.extend([args.mfa_python,args.dictionary,args.acoustic_model])
    # With language gates active, non-English jobs do not require Nyra weights.
    if getattr(args, 'complex_handling', 'off') != 'off':
        required = []
    for path in required:
        if not path.exists():
            p.error('Missing local model/dependency: '+str(path))
    if args.padding < 0 or args.mfa_jobs < 1:
        p.error('padding must be nonnegative; mfa-jobs must be positive')
    run(args, repair_factory=NyraRepair)
