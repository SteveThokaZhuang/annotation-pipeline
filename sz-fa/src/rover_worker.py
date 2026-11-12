"""Resident-model streaming worker for exact production ROVER segments."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa+mfa'))
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from core import QwenAligner, read_jsonl, write_json, write_jsonl
from process import audit
from sentences import sentence_spans
from repair import SentenceRepair
from nyra_repair import NyraRepair
from prediction_format import FORMAT, export_json, write_predictions
from compare_results import validate_repairs
from complex_handling import ComplexHandler, add_arguments as add_complex_arguments


class Totals:
    def __init__(self):
        self.counts=collections.Counter();self.issues=collections.Counter()
    def add(self,stats):
        for k,v in stats.items():
            if k=='longest_consecutive_zero_words':self.counts[k]=max(self.counts[k],v)
            elif isinstance(v,int):self.counts[k]+=v
        self.issues.update(stats['issues_by_type'])
    def value(self):
        d=dict(self.counts)
        for key,numerator,denominator in [('zero_duration_rate','zero_duration_words','words'),
            ('invalid_word_rate','invalid_words','words'),('chunk_error_rate','bad_chunks','chunks'),
            ('sentence_error_rate','bad_sentences','sentences'),('zero_chunk_rate','zero_chunks','chunks'),
            ('zero_sentence_rate','zero_sentences','sentences')]:
            d[key]=d.get(numerator,0)/d[denominator] if d.get(denominator) else 0
        d['issues_by_type']=dict(self.issues)
        return d


def statistics(rows):
    stats=audit(rows)
    zero_chunks=zero_sentences=longest=0
    for row in rows:
        zero={x['index'] for x in row['issues'] if x['reason']=='zero_duration'}
        zero_chunks+=bool(zero)
        zero_sentences+=sum(any(s['start_index']<=i<s['end_index'] for i in zero) for s in row['sentences'])
        run=0
        for i in range(len(row['words'])):
            run=run+1 if i in zero else 0;longest=max(longest,run)
    stats.update(zero_chunks=zero_chunks,zero_sentences=zero_sentences,longest_consecutive_zero_words=longest)
    return stats


class AudioReader:
    """Reuse the current source handle and read only requested samples."""
    def __init__(self):self.path=None;self.handle=None
    def read(self,row):
        import soundfile as sf
        if self.path!=row['audio']:
            self.close();self.path=row['audio'];self.handle=sf.SoundFile(self.path)
        handle=self.handle
        if handle.samplerate!=row['source_sample_rate']:raise ValueError('Source sample rate changed: '+row['id'])
        handle.seek(row['start_sample'])
        audio=handle.read(row['end_sample']-row['start_sample'],dtype='float32',always_2d=True)
        if len(audio)!=row['end_sample']-row['start_sample']:raise ValueError('Source segment truncated: '+row['id'])
        if not 0<=row['channel']<audio.shape[1]:raise ValueError('Source channel missing: '+row['id'])
        return audio[:,row['channel']].copy(),handle.samplerate
    def close(self):
        if self.handle is not None:self.handle.close()
        self.path=self.handle=None


def resample(value):
    import soxr
    waveform,sr=value
    return soxr.resample(waveform,sr,16000) if sr!=16000 else waveform


class ParallelAudioReader:
    """A separate reusable SoundFile handle per IO thread."""
    def __init__(self):
        self.local=threading.local();self.readers=[];self.lock=threading.Lock()
    def read(self,row):
        try:
            if not hasattr(self.local,'reader'):
                self.local.reader=AudioReader()
                with self.lock:self.readers.append(self.local.reader)
            return row,resample(self.local.reader.read(row)),None
        except Exception as exc:return row,None,repr(exc)
    def close(self):
        for reader in self.readers:reader.close()


def manifest_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def archive_block(scratch,destination,rows):
    destination.mkdir(parents=True)
    by_id={r['id']:r for r in rows}
    for name in ['qwen_predictions.jsonl','mfa_predictions.jsonl','predictions.jsonl','alignment_predictions.jsonl',
                 'annotation_schema.json','error_stats.json',
                 'mfa_error_stats.json','failures.jsonl','validation.json','runtime.json']:
        source=scratch/name
        if source.exists():shutil.copyfile(source,destination/name)
    for backend in ['mfa','nyra']:
        if not (scratch/backend).exists():continue
        out=destination/backend;out.mkdir()
        for name in ['tasks.jsonl','repairs.jsonl','predictions.jsonl']:
            source=scratch/backend/name
            if not source.exists():continue
            with source.open() as inp,(out/name).open('w') as output:
                for line in inp:
                    record=json.loads(line);row=by_id.get(record.get('id'))
                    if row:
                        record.update(source_audio=row['audio'],source_sample_rate=row['source_sample_rate'],
                                      channel=row['channel'],chunk_start_sample=row['start_sample'])
                    record.pop('crop_audio',None)
                    output.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        for name in ['runtime.json','mfa.log','chunk_attempts.jsonl','cross_chunk_seams.jsonl','cross_chunk_rejected.jsonl']:
            source=scratch/backend/name
            if source.exists():shutil.copyfile(source,out/name)
        aligned=scratch/backend/'aligned'
        if aligned.exists():
            with gzip.open(out/'raw_alignments.jsonl.gz','wt',compresslevel=1) as archive:
                for path in sorted(aligned.rglob('*.json')):
                    archive.write(json.dumps(dict(task_id=path.stem,alignment=json.loads(path.read_text())),ensure_ascii=False)+'\n')
        fallback=scratch/backend/'sentence_fallback'
        if fallback.exists():
            target=out/'sentence_fallback';target.mkdir()
            if (fallback/'unresolved.json').exists():shutil.copyfile(fallback/'unresolved.json',target/'unresolved.json')
            stage=fallback/'mfa';stage_out=target/'mfa';stage_out.mkdir()
            for name in ['tasks.jsonl','repairs.jsonl','runtime.json','mfa.log']:
                if (stage/name).exists():shutil.copyfile(stage/name,stage_out/name)
            with gzip.open(stage_out/'raw_alignments.jsonl.gz','wt',compresslevel=1) as archive:
                for path in sorted((stage/'aligned').rglob('*.json')):
                    archive.write(json.dumps(dict(task_id=path.stem,alignment=json.loads(path.read_text())),ensure_ascii=False)+'\n')
    complex_root=scratch/'complex_handling'
    if complex_root.exists():
        target=destination/'complex_handling';target.mkdir()
        for source in complex_root.iterdir():
            if source.is_file() and source.suffix in ('.json','.jsonl'):
                shutil.copyfile(source,target/source.name)
    unresolved=json.loads((scratch/'unresolved.json').read_text())
    for record in unresolved['rejected_sentences']:
        row=by_id.get(record['id'])
        if row:record['source_audio']=row['audio']
        record.pop('crop_audio',None)
    write_json(destination/'unresolved.json',unresolved)
    if (scratch/'mfa/unresolved.json').exists():shutil.copyfile(scratch/'mfa/unresolved.json',destination/'mfa/unresolved.json')


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--method',choices=['qwen+mfa+nyra','qwen+mfa','qwen+nyra'],required=True)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--block-size',type=int,default=32768)
    p.add_argument('--mfa-jobs',type=int,default=32);p.add_argument('--limit',type=int)
    p.add_argument('--mfa-seam-policy',choices=['midpoint','clamp','bounded'],default='midpoint')
    p.add_argument('--mfa-repair-scope',choices=['sentence','chunk'],default='chunk')
    p.add_argument('--no-mfa-sentence-fallback',action='store_false',dest='mfa_sentence_fallback',default=True)
    p.add_argument('--export-unit',choices=['chunk','audio','both'],help='Also export individual JSON files after all blocks')
    add_complex_arguments(p)
    a=p.parse_args()
    a.device='cuda:0';a.padding=.35;a.mfa_timeout=14400
    a.mfa_python=ROOT/'models/mfa/bin/python';a.acoustic_model=ROOT/'models/mfa-english/english_us_arpa.zip'
    a.dictionary=ROOT/'models/mfa-english/dictionary/english_us_arpa.dict'
    a.nyra_runtime=ROOT/'models/nyra';a.nyra_espeak=ROOT/'models/nyra-espeak'
    a.nyra_checkpoint=ROOT/'models/nyra-forced-aligner/checkpoints'
    complex_handler=ComplexHandler.from_args(a)
    output=a.output;output.mkdir(parents=True,exist_ok=False);a.scratch.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4)
    aligner=QwenAligner(ROOT/'models/qwen3-forced-aligner/checkpoints',device=a.device)
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-started
    engine=NyraRepair(a,aligner) if a.method!='qwen+mfa' else None
    totals={stage:Totals() for stage in ['qwen','mfa','final']}
    phase_seconds=collections.Counter();route_counts=collections.Counter();qwen_digest=hashlib.sha256()
    processed=blocks=failures=0;audio_seconds=0.0
    source=manifest_rows(a.input)
    if a.limit:source=itertools.islice(source,a.limit)
    reader=ParallelAudioReader()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            while True:
                rows=list(itertools.islice(source,a.block_size))
                if not rows:break
                block_started=time.perf_counter();scratch=a.scratch/f'b{blocks:05d}';scratch.mkdir()
                destination=output/'blocks'/scratch.name
                cache={};failed=[];predictions=[];qwen_started=time.perf_counter()
                # Read sequentially for file locality, then bucket inference by
                # duration to avoid padding 128 short inputs to a 25-second clip.
                for offset in range(0,len(rows),512):
                    for row,waveform,error in executor.map(reader.read,rows[offset:offset+512]):
                        if error:failed.append(dict(id=row['id'],error=error))
                        else:cache[row['id']]=waveform
                    if (offset//512+1)%8==0:
                        print(f'Audio block={blocks} read={min(offset+512,len(rows))}/{len(rows)}',flush=True)
                ordered=sorted(rows,key=lambda row:(row['duration'],row['id']))
                for offset in range(0,len(rows),a.batch_size):
                    batch=ordered[offset:offset+a.batch_size]
                    payload=[dict(row,audio=(cache[row['id']],16000)) for row in batch if row['id'] in cache]
                    values=[]
                    if payload:
                        try:values=aligner.batch(payload,strict=False)
                        except Exception:
                            for row in payload:
                                try:values.extend(aligner.batch([row],strict=False))
                                except Exception as exc:failed.append(dict(id=row['id'],error=repr(exc)))
                    aligned={row['id']:(words,lang) for (words,lang),row in values}
                    failed_by_id={r['id']:r['error'] for r in failed}
                    for row in batch:
                        words,lang=aligned.get(row['id'],([],row['language']))
                        value=dict(row,method=a.method,backend='public',language=lang,
                            words=[dict(w,source='qwen') for w in words],
                            sentences=sentence_spans(row['text'],lang,aligner.model.aligner_processor))
                        if complex_handler:value['input_language']=row.get('input_language',row.get('language'))
                        if row['id'] in failed_by_id:value['error']=failed_by_id[row['id']]
                        predictions.append(value)
                    if (offset//a.batch_size+1)%32==0:print(f'Qwen block={blocks} chunks={processed+offset+len(batch)}',flush=True)
                positions={row['id']:i for i,row in enumerate(rows)}
                predictions.sort(key=lambda row:positions[row['id']])
                torch.cuda.synchronize();qwen_seconds=time.perf_counter()-qwen_started
                before=statistics(predictions);totals['qwen'].add(before)
                for row in predictions:
                    qwen_digest.update((json.dumps(dict(id=row['id'],words=row['words']),ensure_ascii=False,
                                                  separators=(',',':'),allow_nan=False)+'\n').encode())
                original=copy.deepcopy(predictions);write_jsonl(scratch/'qwen_predictions.jsonl',predictions)
                a.output=scratch;a.audio_loader=lambda row:(cache[row['id']][:,None],16000)
                if complex_handler:complex_handler.prepare(predictions,a.audio_loader)
                repair_started=time.perf_counter()
                route=SentenceRepair(a,aligner).repair(predictions) if a.method=='qwen+mfa' else engine.repair(predictions)
                repair_seconds=time.perf_counter()-repair_started
                if route.get('nyra_load_error'):raise RuntimeError(route['nyra_load_error'])
                mfa_route=route if a.method=='qwen+mfa' else route.get('mfa')
                if mfa_route and mfa_route.get('mfa_exit_code') not in (0,None):route_counts['mfa_failed_blocks']+=1
                after=statistics(predictions);totals['final'].add(after);validation={}
                if a.method=='qwen+mfa':
                    totals['mfa'].add(after)
                    validation['mfa']=validate_repairs(original,predictions,read_jsonl(scratch/'mfa/repairs.jsonl'),
                        json.loads((scratch/'unresolved.json').read_text()),
                        cross_chunk_seams=read_jsonl(scratch/'mfa/cross_chunk_seams.jsonl') if (scratch/'mfa/cross_chunk_seams.jsonl').exists() else [])
                    route_counts['mfa_requested']+=route['sentence_groups'];route_counts['mfa_accepted']+=route['accepted_groups']
                    phase_seconds['mfa_command_seconds']+=route['mfa_command_seconds']
                elif a.method=='qwen+mfa+nyra':
                    intermediate=read_jsonl(scratch/'mfa_predictions.jsonl');totals['mfa'].add(statistics(intermediate))
                    validation['mfa']=validate_repairs(original,intermediate,read_jsonl(scratch/'mfa/repairs.jsonl'),
                        json.loads((scratch/'mfa/unresolved.json').read_text()),
                        cross_chunk_seams=read_jsonl(scratch/'mfa/cross_chunk_seams.jsonl') if (scratch/'mfa/cross_chunk_seams.jsonl').exists() else [])
                    validation['nyra']=validate_repairs(intermediate,predictions,read_jsonl(scratch/'nyra/repairs.jsonl'),
                        json.loads((scratch/'unresolved.json').read_text()),source='nyra')
                    route_counts['mfa_requested']+=route['mfa']['sentence_groups'];route_counts['mfa_accepted']+=route['mfa']['accepted_groups']
                    phase_seconds['mfa_command_seconds']+=route['mfa']['mfa_command_seconds']
                else:
                    validation['nyra']=validate_repairs(original,predictions,read_jsonl(scratch/'nyra/repairs.jsonl'),
                        json.loads((scratch/'unresolved.json').read_text()),source='nyra')
                if a.method!='qwen+mfa':
                    route_counts['nyra_requested']+=route['sentence_groups'];route_counts['nyra_accepted']+=route['accepted_groups']
                    phase_seconds['nyra_model_load_seconds']+=route['nyra_model_load_seconds']
                    phase_seconds['nyra_inference_seconds']+=route['nyra_inference_seconds']
                if complex_handler:
                    complex_stats=complex_handler.run(predictions,scratch,audio_loader=a.audio_loader,
                        aligner=aligner,args=a,nyra_engine=engine)
                    route['complex_handling']=complex_stats
                    phase_seconds['complex_handling_seconds']+=complex_stats['seconds']
                    phase_seconds['nyra_model_load_seconds']+=complex_stats['nyra_model_load_seconds']
                write_json(scratch/'validation.json',validation);write_predictions(scratch,predictions)
                write_jsonl(scratch/'failures.jsonl',failed);write_json(scratch/'error_stats.json',dict(before=before,after=after,repair=route))
                block_duration=sum(r['duration'] for r in rows)
                write_json(scratch/'runtime.json',dict(chunks=len(rows),audio_seconds=block_duration,qwen_seconds=qwen_seconds,
                    repair_seconds=repair_seconds,seconds_before_archive=time.perf_counter()-block_started))
                archive_block(scratch,destination,predictions);shutil.rmtree(scratch)
                phase_seconds['qwen_seconds']+=qwen_seconds;phase_seconds['repair_seconds']+=repair_seconds
                processed+=len(rows);audio_seconds+=block_duration;failures+=len(failed);blocks+=1
                progress=dict(method=a.method,processed_chunks=processed,blocks=blocks,audio_seconds=audio_seconds,
                    elapsed_seconds=time.perf_counter()-started,zero_qwen=totals['qwen'].value()['zero_duration_words'],
                    zero_final=totals['final'].value()['zero_duration_words'],failures=failures,
                    phases=dict(phase_seconds),routes=dict(route_counts),latest_block=str(destination))
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True);cache.clear()
    finally:reader.close()
    exported=None
    if a.export_unit:
        exported=export_json(output,(row for path in sorted((output/'blocks').glob('b*/predictions.jsonl'))
                                    for row in read_jsonl(path)),a.export_unit)
    runtime=dict(method=a.method,samples=processed,blocks=blocks,audio_seconds=audio_seconds,
        mfa_repair_scope=a.mfa_repair_scope,mfa_seam_policy=a.mfa_seam_policy,mfa_sentence_fallback=a.mfa_sentence_fallback,
        prediction_format=FORMAT,
        seconds=time.perf_counter()-started,setup_seconds=setup_seconds,phases=dict(phase_seconds),routes=dict(route_counts),
        failures=failures,block_size=a.block_size,batch_size=a.batch_size,input=str(a.input.resolve()),device=a.device,
        batching='duration buckets within block; outputs restored to input order',
        io_threads=8,
        qwen_output_sha256=qwen_digest.hexdigest(),visible_gpu_count=torch.cuda.device_count(),gpu_name=torch.cuda.get_device_name(),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='streamed source reads/resampling + resident Qwen/MFA/Nyra + all model loads + failed attempts + QA + archived output IO')
    runtime['rtf']=runtime['seconds']/audio_seconds
    if exported:runtime['json_exports']=exported['file_counts']
    write_json(output/'error_stats.json',{stage:counter.value() for stage,counter in totals.items() if counter.counts})
    write_json(output/'runtime.json',runtime);print(json.dumps(runtime),flush=True)


if __name__=='__main__':main()
