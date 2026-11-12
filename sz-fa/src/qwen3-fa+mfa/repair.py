"""Replace complete affected sentences, accepting only valid MFA intervals."""
import collections
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import time
import unicodedata

from core import read_jsonl, write_json, write_jsonl
from process import word_issues
from seams import reconcile_rows
from bounded_seams import splice_bounded, reconcile_bounded_rows


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


def splice_sentence(words, start_index, end_index, replacement, lower, upper, seam_policy='clamp', original_words=None):
    """Join a candidate using the chosen seam rule, preserving a complete audit."""
    if seam_policy == 'bounded':
        return splice_bounded(words, start_index, end_index, replacement, lower, upper, original_words)
    if seam_policy == 'midpoint':
        return splice_sentence_midpoint(words, start_index, end_index, replacement, lower, upper)
    if seam_policy != 'clamp':
        raise ValueError('Unknown seam policy: ' + seam_policy)
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


def splice_sentence_midpoint(words, start_index, end_index, replacement, lower, upper):
    """Share each overlapping seam equally between the two touching words.

    Only the immediate neighbor's touching endpoint can change. Gaps are kept.
    Work on copies and reject the entire splice if either word would collapse.
    """
    if not 0 <= start_index < end_index <= len(words):
        raise ValueError('Replacement span is outside the word sequence')
    if len(replacement) != end_index-start_index:
        raise ValueError('Replacement changes token count')
    if [w['text'] for w in replacement] != [w['text'] for w in words[start_index:end_index]]:
        raise ValueError('Replacement changes token sequence')
    output = [dict(w) for w in words]
    candidate = [dict(w) for w in replacement]
    for w in candidate:
        if not all(math.isfinite(w[k]) for k in ('start', 'end')):
            raise ValueError('Nonfinite replacement timestamp')
        w['start'] = max(lower, w['start'])
        w['end'] = min(upper, w['end'])
    if word_issues(candidate, lower, upper):
        raise ValueError('Sentence is invalid within chunk boundaries')
    adjustments = []
    for side, index, candidate_index, neighbor_endpoint, candidate_endpoint in [
        ('left', start_index-1, 0, 'end', 'start'),
        ('right', end_index, len(candidate)-1, 'start', 'end'),
    ]:
        if index < 0 or index >= len(output):
            continue
        neighbor = output[index]
        item = candidate[candidate_index]
        a, b = neighbor[neighbor_endpoint], item[candidate_endpoint]
        if not math.isfinite(a):
            raise ValueError('Nonfinite neighbor timestamp')
        overlaps = a > b if side == 'left' else b > a
        if overlaps:
            midpoint = (a+b)/2
            before = dict(neighbor)
            neighbor[neighbor_endpoint] = item[candidate_endpoint] = midpoint
            adjustments.append(dict(side=side, neighbor_index=index,
                replacement_index=start_index+candidate_index, endpoint=neighbor_endpoint,
                before_neighbor=before, after_neighbor=dict(neighbor),
                original_neighbor_endpoint=a, original_replacement_endpoint=b,
                midpoint=midpoint, overlap_seconds=abs(a-b)))
    output[start_index:end_index] = candidate
    lo, hi = max(0, start_index-1), min(len(output), end_index+1)
    if word_issues(output[lo:hi], lower, upper):
        raise ValueError('Midpoint seam would create invalid/zero-duration intervals')
    left = output[start_index-1]['end'] if start_index else lower
    right = output[end_index]['start'] if end_index < len(output) else upper
    return output, dict(left=max(lower,left), right=min(upper,right),
                        policy='midpoint', adjustments=adjustments)


def prepare_sentence_tasks(predictions, root, padding, audio_loader=None, task_guard=None, backend='mfa', scope='sentence'):
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
        if scope == 'chunk':
            groups = [row['sentences']]
        elif scope != 'sentence':
            raise ValueError('Unknown MFA repair scope: '+scope)
        audio = sr = None
        for group in groups:
            lo, hi = group[0]['start_index'], group[-1]['end_index']
            words = row['words'][lo:hi]
            task_id = f's{len(tasks):06d}'
            rejection = task_guard(row, backend) if task_guard else None
            if rejection:
                tasks.append(dict(task_id=task_id, row_index=row_index, id=row['id'],
                    start_index=lo, end_index=hi, sentences=group,
                    text=' '.join(s['text'] for s in group), original_words=words,
                    zero_indices=sorted(zero.intersection(range(lo, hi))),
                    crop_start=row['chunk_start'], crop_end=row['chunk_end'], preparation_error=rejection))
                continue
            if audio is None:
                audio, sr = (audio_loader(row) if audio_loader else
                             sf.read(row['audio'], dtype='float32', always_2d=True))
            local_start = max(0, min(w['start'] for w in words)-row['chunk_start']-padding)
            local_end = min(row['duration'], max(w['end'] for w in words)-row['chunk_start']+padding)
            if scope == 'chunk':
                local_start, local_end = 0, row['duration']
            a, b = math.floor(local_start*sr), math.ceil(local_end*sr)
            if scope == 'chunk':
                a, b = 0, len(audio)
            task = dict(task_id=task_id, row_index=row_index, id=row['id'],
                        start_index=lo, end_index=hi, sentences=group,
                        text=' '.join(s['text'] for s in group),
                        zero_indices=sorted(zero.intersection(range(lo,hi))),
                        crop_start=row['chunk_start']+a/sr, crop_end=row['chunk_start']+b/sr,
                        original_words=words)
            task['repair_scope'] = scope
            tasks.append(task)
            if b<=a:
                task['preparation_error']='Empty sentence audio crop'
                continue
            sf.write(corpus/(task_id+'.wav'), audio[a:b].mean(axis=1), sr, subtype='PCM_16')
            # Input units match Qwen exactly; preserve complete sentence coverage.
            transcript = row['text'] if scope == 'chunk' else ' '.join(w['text'] for w in words)
            (corpus/(task_id+'.lab')).write_text(transcript+'\n')
    write_jsonl(root/'tasks.jsonl', tasks)
    return tasks, unresolved


class SentenceRepair:
    def __init__(self, args, aligner):
        self.args = args
        self.aligner = aligner
        self.root = args.output / 'mfa'
        self.root.mkdir()

    def repair(self, predictions):
        baseline=copy.deepcopy(predictions) if getattr(self.args,'mfa_seam_policy','midpoint')=='bounded' else None
        stats=self._repair_priority(predictions)
        if baseline is not None and not getattr(self.args,'_mfa_skip_cross_chunk',False):
            return self._finish_bounded(predictions,baseline,stats)
        if (getattr(self.args,'_mfa_skip_cross_chunk',False) or
                getattr(self.args,'mfa_seam_policy','midpoint')!='midpoint'):
            return stats
        started=time.perf_counter();adjustments=[];rejected=[]
        reconcile_rows(predictions,adjustments,rejected)
        write_jsonl(self.root/'cross_chunk_seams.jsonl',adjustments)
        write_jsonl(self.root/'cross_chunk_rejected.jsonl',rejected)
        unresolved=json.loads((self.args.output/'unresolved.json').read_text())
        for row in predictions:row['issues']=word_issues(row['words'],row['chunk_start'],row['chunk_end'])
        unresolved['residual_chunks']=[dict(id=r['id'],audio=r.get('audio'),source_audio=r.get('source_audio'),
            text=r['text'],issues=r['issues'],error=r.get('error')) for r in predictions if r['issues'] or r.get('error')]
        write_json(self.args.output/'unresolved.json',unresolved)
        stats['sentence_midpoint_seams']=stats.get('midpoint_seams',0)
        stats['cross_chunk_midpoint_seams']=len(adjustments)
        stats['cross_chunk_unresolved_seams']=len(rejected)
        stats['midpoint_seams']=stats.get('midpoint_seams',0)+len(adjustments)
        stats['seconds']=stats.get('seconds',0)+time.perf_counter()-started
        write_json(self.root/'runtime.json',stats)
        return stats

    def _finish_bounded(self, predictions, baseline, stats):
        started=time.perf_counter()
        seams,rejected,rolled_back=reconcile_bounded_rows(predictions,baseline)
        write_jsonl(self.root/'cross_chunk_seams.jsonl',seams)
        write_jsonl(self.root/'cross_chunk_rejected.jsonl',rejected)
        records=read_jsonl(self.root/'repairs.jsonl')
        for record in records:
            if record['id'] in rolled_back and record['status']=='repaired':
                record['rolled_back_attempt']=dict(replacement_words=record.pop('replacement_words'),
                    boundary_limits=record.pop('boundary_limits'))
                record.update(status='unresolved',reason='Bounded cross-chunk seam rejected; restored original Qwen chunk')
        write_jsonl(self.root/'repairs.jsonl',records)
        accepted=[r for r in records if r['status']=='repaired']
        failed=[r for r in records if r['status']!='repaired']
        unresolved=json.loads((self.args.output/'unresolved.json').read_text())
        extras=[r for r in unresolved['rejected_sentences'] if 'task_id' not in r]
        for row in predictions:row['issues']=word_issues(row['words'],row['chunk_start'],row['chunk_end'])
        unresolved.update(rejected_sentences=extras+failed,residual_chunks=[
            dict(id=r['id'],audio=r.get('audio'),text=r['text'],issues=r['issues'],error=r.get('error'))
            for r in predictions if r['issues'] or r.get('error')])
        write_json(self.args.output/'unresolved.json',unresolved)
        repaired_sentences=sum(len(r['sentences']) for r in accepted)
        stats.update(accepted_groups=len(accepted),rejected_groups=len(failed),
            repaired_sentences=repaired_sentences,
            sentence_repair_success_rate=repaired_sentences/stats['triggered_sentences'] if stats['triggered_sentences'] else None,
            fixed_zero_words=sum(len(r['zero_indices']) for r in accepted),
            replaced_words=sum(r['end_index']-r['start_index'] for r in accepted),
            rejection_reasons=dict(collections.Counter(r['reason'] for r in failed)),
            midpoint_seams=0,bounded_neighbor_edits=sum(len(r['boundary_limits']['adjustments']) for r in accepted),
            cross_chunk_bounded_seams=len(seams),cross_chunk_unresolved_seams=len(rejected),
            rolled_back_chunks=len(rolled_back),max_endpoint_shift_seconds=0.020,max_duration_shrink_fraction=0.10)
        stats['seconds']+=time.perf_counter()-started
        write_json(self.root/'runtime.json',stats)
        return stats

    def _repair_priority(self, predictions):
        if (getattr(self.args, 'mfa_repair_scope', 'sentence') != 'chunk' or
                not getattr(self.args, 'mfa_sentence_fallback', True)):
            return self._repair_once(predictions)
        started=time.perf_counter()
        chunk_stats=self._repair_once(predictions)
        attempts=read_jsonl(self.root/'repairs.jsonl')
        write_jsonl(self.root/'chunk_attempts.jsonl',attempts)
        if not chunk_stats['rejected_groups']:
            chunk_stats['priority_policy']='chunk_then_sentence'
            write_json(self.root/'runtime.json',chunk_stats)
            return chunk_stats
        for row in predictions:
            row['issues']=word_issues(row['words'],row['chunk_start'],row['chunk_end'])
        fallback_args=copy.copy(self.args)
        fallback_args.output=self.root/'sentence_fallback'
        fallback_args.output.mkdir()
        fallback_args.mfa_repair_scope='sentence'
        fallback_args._mfa_skip_cross_chunk=True
        fallback=SentenceRepair(fallback_args,self.aligner)
        sentence_stats=fallback.repair(predictions)
        records=read_jsonl(fallback.root/'repairs.jsonl')
        for record in records:
            original_id=record['task_id']
            record.update(task_id='fallback_'+original_id,source_task_id=original_id,
                          attempt_source='sentence_fallback/mfa')
            # Nyra, if requested, must reuse this exact fallback sentence crop.
            audio=fallback.root/'corpus'/(original_id+'.wav')
            if audio.exists():
                (self.root/'corpus'/(record['task_id']+'.wav')).symlink_to(audio.resolve())
        final=[r for r in attempts if r['status']=='repaired']+records
        write_jsonl(self.root/'repairs.jsonl',final)
        unresolved=json.loads((fallback_args.output/'unresolved.json').read_text())
        for record in unresolved['rejected_sentences']:
            if 'task_id' in record:
                record['source_task_id']=record['task_id']
                record['task_id']='fallback_'+record['task_id']
                record['attempt_source']='sentence_fallback/mfa'
        write_json(self.args.output/'unresolved.json',unresolved)
        accepted=[r for r in final if r['status']=='repaired']
        rejected=[r for r in final if r['status']=='unresolved']
        total_sentences=sum(len(r['sentences']) for r in final)
        repaired_sentences=sum(len(r['sentences']) for r in accepted)
        combined=dict(chunk_stats,priority_policy='chunk_then_sentence',
            trigger='any zero-duration word -> whole chunk; rejected chunks -> sentence fallback',
            sentence_groups=len(final),accepted_groups=len(accepted),rejected_groups=len(rejected),
            triggered_sentences=total_sentences,repaired_sentences=repaired_sentences,
            sentence_repair_success_rate=repaired_sentences/total_sentences if total_sentences else None,
            originally_zero_words=sum(len(r['zero_indices']) for r in final),
            fixed_zero_words=sum(len(r['zero_indices']) for r in accepted),
            replaced_words=sum(r['end_index']-r['start_index'] for r in accepted),
            midpoint_seams=sum(len(r['boundary_limits'].get('adjustments',[])) for r in accepted
                if r['boundary_limits'].get('policy')=='midpoint'),
            bounded_neighbor_edits=sum(len(r['boundary_limits'].get('adjustments',[])) for r in accepted
                if r['boundary_limits'].get('policy')=='bounded'),
            rejection_reasons=dict(collections.Counter(r['reason'] for r in rejected)),
            mfa_command_seconds=chunk_stats['mfa_command_seconds']+sentence_stats['mfa_command_seconds'],
            seconds=time.perf_counter()-started,chunk_stage=chunk_stats,sentence_fallback=sentence_stats)
        statuses=[chunk_stats['mfa_exit_code'],sentence_stats['mfa_exit_code']]
        combined['mfa_exit_code']=next((s for s in statuses if s not in (0,None)),0 if 0 in statuses else None)
        write_json(self.root/'runtime.json',combined)
        return combined

    def _repair_once(self, predictions):
        started = time.perf_counter()
        original_words=([copy.deepcopy(r['words']) for r in predictions]
                        if getattr(self.args,'mfa_seam_policy','midpoint')=='bounded' else None)
        tasks, unresolved = prepare_sentence_tasks(predictions, self.root, self.args.padding,
            getattr(self.args, 'audio_loader', None),
            task_guard=(self.args.complex_handler.baseline_rejection
                        if getattr(self.args, 'complex_handler', None) else None),
            scope=getattr(self.args, 'mfa_repair_scope', 'sentence'))
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
                except OSError as exc:
                    status = 'dependency_error'
                    log.write(repr(exc)+'\n')
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
                                                  replacement, row['chunk_start'], row['chunk_end'],
                                                  seam_policy=getattr(self.args, 'mfa_seam_policy', 'midpoint'),
                                                  original_words=original_words[task['row_index']] if original_words is not None else None)
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
        stats = dict(trigger='any zero-duration word -> replace entire '+getattr(self.args, 'mfa_repair_scope', 'sentence'),
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
        stats['seam_policy'] = getattr(self.args, 'mfa_seam_policy', 'midpoint')
        stats['repair_scope'] = getattr(self.args, 'mfa_repair_scope', 'sentence')
        edits = sum(len(t['boundary_limits'].get('adjustments', [])) for t in accepted)
        stats['midpoint_seams'] = edits if stats['seam_policy']=='midpoint' else 0
        stats['bounded_neighbor_edits'] = edits if stats['seam_policy']=='bounded' else 0
        write_json(self.root/'runtime.json', stats)
        print(json.dumps({k:v for k,v in stats.items() if k!='mfa_command'}, ensure_ascii=False), flush=True)
        return stats
