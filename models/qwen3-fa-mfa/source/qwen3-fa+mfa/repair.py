"""Replace complete affected sentences, accepting only valid MFA intervals."""
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import time
import unicodedata

from core import write_json, write_jsonl
from process import word_issues


def key(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def map_mfa_words(expected, entries, offset, source='mfa'):
    """Allow MFA to split contractions; never invent times by splitting an interval."""
    observed = []
    for start, end, label in entries:
        if not label.strip() or label in {'<eps>', 'sil', 'sp', '<sil>'}:
            continue
        if label in {'<unk>', 'spn'}:
            raise ValueError('MFA returned an unknown word')
        observed.append(dict(text=label, start=float(start)+offset, end=float(end)+offset))
    if ''.join(key(w['text']) for w in expected) != ''.join(key(w['text']) for w in observed):
        raise ValueError('MFA transcript coverage mismatch (missing/OOV/changed words)')
    output, pos = [], 0
    for word in expected:
        target, assembled, used = key(word['text']), '', []
        if not target:
            raise ValueError('Output token has no lexical characters')
        while pos < len(observed) and len(assembled) < len(target):
            item = observed[pos]
            if item['end'] <= item['start'] or not all(math.isfinite(item[k]) for k in ['start','end']):
                raise ValueError('MFA returned a zero/negative/nonfinite duration')
            if used and item['start'] < used[-1]['end']-1e-7:
                raise ValueError('MFA returned overlapping subwords')
            used.append(item); assembled += key(item['text']); pos += 1
        if assembled != target or not used:
            raise ValueError('MFA/Qwen token boundary mismatch')
        output.append(dict(text=word['text'], start=used[0]['start'], end=used[-1]['end'], source=source))
    if pos != len(observed):
        raise ValueError('Unconsumed MFA words')
    return output


def splice_sentence(words, start_index, end_index, replacement, lower, upper):
    """Clamp replacement to unchanged neighbors, then validate both seams."""
    candidate = [dict(w) for w in replacement]
    left = max(lower, words[start_index-1]['end'] if start_index else lower)
    right = min(upper, words[end_index]['start'] if end_index < len(words) else upper)
    if right <= left:
        raise ValueError('No positive interval between unchanged neighbors')
    for w in candidate:
        w['start'] = max(left, w['start'])
        w['end'] = min(right, w['end'])
    if word_issues(candidate, left, right):
        raise ValueError('Sentence is invalid after boundary clamping')
    if len(candidate) != end_index-start_index:
        raise ValueError('Replacement changes token count')
    if [w['text'] for w in candidate] != [w['text'] for w in words[start_index:end_index]]:
        raise ValueError('Replacement changes token sequence')
    return words[:start_index] + candidate + words[end_index:], dict(left=left, right=right)


def prepare_sentence_tasks(predictions, root, padding, audio_loader=None):
    """Shared zero-trigger grouping and crops, identical for all fallbacks."""
    import soundfile as sf
    corpus = root / 'corpus'
    corpus.mkdir()
    tasks, unresolved = [], []
    for row_index, row in enumerate(predictions):
        zero = {i for i,w in enumerate(row['words']) if abs(w['end']-w['start']) <= 1e-7}
        if not zero:
            if row.get('error') or row.get('issues'):
                unresolved.append(dict(id=row['id'], reason='qwen_error_without_zero_trigger',
                                       error=row.get('error'), issues=row.get('issues')))
            continue
        affected = [s for s in row['sentences'] if any(s['start_index']<=i<s['end_index'] for i in zero)]
        # Adjacent affected sentences share one crop, avoiding artificial seams
        # between two independently repaired intervals. Each whole sentence is replaced.
        groups = []
        for sentence in affected:
            if groups and groups[-1][-1]['end_index'] == sentence['start_index']:
                groups[-1].append(sentence)
            else:
                groups.append([sentence])
        audio, sr = (audio_loader(row) if audio_loader else
                     sf.read(row['audio'], dtype='float32', always_2d=True))
        for group in groups:
            lo, hi = group[0]['start_index'], group[-1]['end_index']
            words = row['words'][lo:hi]
            task_id = f's{len(tasks):06d}'
            local_start = max(0, min(w['start'] for w in words)-row['chunk_start']-padding)
            local_end = min(row['duration'], max(w['end'] for w in words)-row['chunk_start']+padding)
            a, b = math.floor(local_start*sr), math.ceil(local_end*sr)
            task = dict(task_id=task_id, row_index=row_index, id=row['id'],
                        start_index=lo, end_index=hi, sentences=group,
                        text=' '.join(s['text'] for s in group),
                        zero_indices=sorted(zero.intersection(range(lo,hi))),
                        crop_start=row['chunk_start']+a/sr, crop_end=row['chunk_start']+b/sr,
                        original_words=words)
            tasks.append(task)
            if b<=a:
                task['preparation_error']='Empty sentence audio crop'
                continue
            sf.write(corpus/(task_id+'.wav'), audio[a:b].mean(axis=1), sr, subtype='PCM_16')
            # Input units match Qwen exactly; preserve complete sentence coverage.
            (corpus/(task_id+'.lab')).write_text(' '.join(w['text'] for w in words)+'\n')
    write_jsonl(root/'tasks.jsonl', tasks)
    return tasks, unresolved


class SentenceRepair:
    def __init__(self, args, aligner):
        self.args = args
        self.aligner = aligner
        self.root = args.output / 'mfa'
        self.root.mkdir()

    def repair(self, predictions):
        started = time.perf_counter()
        tasks, unresolved = prepare_sentence_tasks(predictions, self.root, self.args.padding,
                                                   getattr(self.args, 'audio_loader', None))
        corpus = self.root / 'corpus'
        print(f'MFA: {len(tasks)} sentence groups need repair', flush=True)
        status, command, command_seconds = None, [], 0.0
        if tasks and any(not t.get('preparation_error') for t in tasks):
            env = dict(os.environ)
            prefix = self.args.mfa_python.resolve().parent.parent
            env.update(PATH=str(prefix/'bin')+os.pathsep+env.get('PATH',''),
                       LD_LIBRARY_PATH=str(prefix/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
                       MFA_ROOT_DIR=str((self.root/'cache').resolve()), CUDA_VISIBLE_DEVICES='',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [str(self.args.mfa_python), '-c',
                       'from montreal_forced_aligner.command_line.mfa import mfa_cli; mfa_cli()',
                       'align', str(corpus.resolve()), str(self.args.dictionary.resolve()),
                       str(self.args.acoustic_model.resolve()), str((self.root/'aligned').resolve()),
                       '--output_format', 'json', '--num_jobs', str(self.args.mfa_jobs),
                       '--temporary_directory', str((self.root/'work').resolve()),
                       '--single_speaker', '--use_threading', '--no_use_postgres',
                       '--beam', '100', '--retry_beam', '400', '--clean', '--overwrite']
            command_started = time.perf_counter()
            with (self.root/'mfa.log').open('w') as log:
                try:
                    status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=self.args.mfa_timeout, check=False).returncode
                except subprocess.TimeoutExpired:
                    status = 'timeout'
            command_seconds = time.perf_counter()-command_started
        accepted, rejected = [], []
        for task in tasks:
            row = predictions[task['row_index']]
            record = dict(task)
            try:
                if task.get('preparation_error'):
                    raise ValueError(task['preparation_error'])
                path = self.root/'aligned'/(task['task_id']+'.json')
                if not path.exists():
                    matches = list((self.root/'aligned').rglob(task['task_id']+'.json'))
                    if len(matches) != 1:
                        raise ValueError(f'MFA has no unique output (exit={status})')
                    path = matches[0]
                data = json.loads(path.read_text())
                tiers = [tier for name,tier in data['tiers'].items() if name.casefold().endswith('words')]
                if len(tiers) != 1:
                    raise ValueError('MFA has no unique word tier')
                replacement = map_mfa_words(task['original_words'], tiers[0]['entries'], task['crop_start'])
                updated, bounds = splice_sentence(row['words'], task['start_index'], task['end_index'],
                                                  replacement, row['chunk_start'], row['chunk_end'])
                row['words'] = updated
                record.update(status='repaired', replacement_words=updated[task['start_index']:task['end_index']],
                              boundary_limits=bounds)
                accepted.append(record)
            except Exception as exc:
                record.update(status='unresolved', reason=str(exc))
                rejected.append(record)
                unresolved.append(record)
        # Include every residual zero and other structural issue, with original audio.
        residual = []
        for row in predictions:
            issues = word_issues(row['words'], row['chunk_start'], row['chunk_end'])
            if issues or row.get('error'):
                residual.append(dict(id=row['id'], audio=row['audio'], source_audio=row.get('source_audio'),
                                     text=row['text'], issues=issues, error=row.get('error')))
        write_jsonl(self.root/'repairs.jsonl', accepted+rejected)
        write_json(self.args.output/'unresolved.json', dict(rejected_sentences=unresolved, residual_chunks=residual))
        repaired_sentences = sum(len(t['sentences']) for t in accepted)
        total_sentences = sum(len(t['sentences']) for t in tasks)
        stats = dict(trigger='any zero-duration word -> replace entire sentence',
            sentence_groups=len(tasks), triggered_sentences=total_sentences,
            accepted_groups=len(accepted), rejected_groups=len(rejected),
            repaired_sentences=repaired_sentences,
            sentence_repair_success_rate=repaired_sentences/total_sentences if total_sentences else None,
            originally_zero_words=sum(len(t['zero_indices']) for t in tasks),
            fixed_zero_words=sum(len(t['zero_indices']) for t in accepted),
            replaced_words=sum(t['end_index']-t['start_index'] for t in accepted),
            rejection_reasons=dict(collections.Counter(t['reason'] for t in rejected)),
            mfa_exit_code=status, mfa_command=command, mfa_command_seconds=command_seconds,
            seconds=time.perf_counter()-started,
            acoustic_model=str(self.args.acoustic_model.resolve()), dictionary=str(self.args.dictionary.resolve()))
        write_json(self.root/'runtime.json', stats)
        print(json.dumps({k:v for k,v in stats.items() if k!='mfa_command'}, ensure_ascii=False), flush=True)
        return stats
