"""Persist compressed final predictions without duplicating large audit trees."""
import gzip
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    directory=Path('/tmp/fa_open_yap_results')
    suite=json.loads((directory/'suite_summary.json').read_text())
    if len(suite['experiments'])!=3:raise RuntimeError('Wait for all three experiments')
    target=ROOT/'results/open_yap_1k_summary/predictions';target.mkdir(exist_ok=True)
    archives=[]
    for experiment in suite['experiments']:
        started=time.perf_counter()
        name={'qwen+mfa+nyra':'qwen3-fa+mfa+nyra','qwen+mfa':'qwen3-fa+mfa','qwen+nyra':'qwen3-fa+nyra'}[experiment['method']]
        source=directory/name
        shard_index=json.loads((source/'results.json').read_text())['prediction_shards']
        packed=directory/(name+'.predictions.jsonl.gz')
        count=words=zeros=0
        with ExitStack() as stack:
            output=stack.enter_context(gzip.open(packed,'wb',compresslevel=5))
            manifests={str(gpu):stack.enter_context((ROOT/f'data/open_yap_1k_rover/gpu_{gpu}.jsonl').open()) for gpu in range(4)}
            for relative in shard_index:
                gpu=Path(relative).parts[0].removeprefix('gpu_')
                with (source/relative).open('rb') as stream, ExitStack() as block_stack:
                    audit_path=(source/relative).with_name('alignment_predictions.jsonl')
                    audit_stream=block_stack.enter_context(audit_path.open()) if audit_path.exists() else None
                    for line in stream:
                        if not line.strip():continue
                        payload=json.loads(line)
                        prediction=json.loads(next(audit_stream)) if audit_stream else payload
                        if audit_stream:
                            assert payload['chunk_id']==prediction['id'] and payload['text']==prediction['text']
                            assert payload['audio_path']==prediction['audio']
                            alignment=payload['force_align'];units=prediction['words'];origin=prediction['chunk_start']
                            assert alignment['words']==[word['text'] for word in units]
                            assert len(alignment['timestamps'])==len(units)
                            for pair,word in zip(alignment['timestamps'],units):
                                assert len(pair)==2
                                assert all(math.isclose(value+origin,word[key],abs_tol=1e-9,rel_tol=0)
                                    for value,key in zip(pair,['start','end']))
                        expected=json.loads(next(manifests[gpu]))
                        for field in ['id','text','source_audio','start_sample','end_sample','channel','source_sample_rate']:
                            if prediction[field]!=expected[field]:
                                raise RuntimeError(f'Input/prediction coverage mismatch: {relative} {field} {expected["id"]}')
                        words+=len(prediction['words'])
                        zeros+=sum(abs(word['end']-word['start'])<=1e-7 for word in prediction['words'])
                        output.write(line);count+=1
                    if audit_stream and any(line.strip() for line in audit_stream):
                        raise RuntimeError('Normalized prediction omits audit rows')
            if any(any(line.strip() for line in stream) for stream in manifests.values()):
                raise RuntimeError('Prediction archive omits manifest rows')
        if count!=experiment['samples']:raise RuntimeError('Prediction archive completeness mismatch')
        final=experiment['errors']['final']
        if (words,zeros)!=(final['words'],final['zero_duration_words']):
            raise RuntimeError('Prediction archive word/error totals differ from experiment summary')
        validations=[json.loads((source/relative).with_name('validation.json').read_text()) for relative in shard_index]
        if not validations or any(check['status']!='passed' for block in validations for check in block.values()):
            raise RuntimeError('A completed block has not passed route validation')
        digest=hashlib.sha256()
        with packed.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        destination=target/packed.name
        temporary=destination.with_name(destination.name+'.partial')
        error=None
        try:
            shutil.copyfile(packed,temporary);temporary.rename(destination)
            stored=str(destination)
        except OSError as exc:
            if temporary.exists():temporary.unlink()
            error=repr(exc);stored=str(packed)
        record=dict(method=experiment['method'],samples=count,bytes=packed.stat().st_size,
            prediction_format=experiment['workers'][0].get('prediction_format','legacy-audit-v1'),
            sha256=digest.hexdigest(),path=stored,shared_copy_error=error,seconds=time.perf_counter()-started,
            validation=dict(status='passed',manifest_coverage='all rows, order, IDs, source samples and unchanged text',
                            words=words,zero_duration_words=zeros,blocks=len(validations)),
            scope='post-benchmark compression and shared-storage export; not included in FA wall RTF')
        unresolved=directory/(name+'.unresolved.jsonl.gz')
        residual_counts={'rejected_sentences':0,'residual_chunks':0}
        with gzip.open(unresolved,'wt',compresslevel=5) as stream:
            for relative in shard_index:
                values=json.loads((source/relative).with_name('unresolved.json').read_text())
                for kind,items in values.items():
                    residual_counts[kind]+=len(items)
                    for item in items:
                        stream.write(json.dumps(dict(kind=kind,record=item),ensure_ascii=False)+'\n')
        residual_destination=target/unresolved.name
        residual_temporary=residual_destination.with_name(residual_destination.name+'.partial')
        residual_error=None
        try:
            shutil.copyfile(unresolved,residual_temporary);residual_temporary.rename(residual_destination)
            residual_path=str(residual_destination)
        except OSError as exc:
            if residual_temporary.exists():residual_temporary.unlink()
            residual_error=repr(exc);residual_path=str(unresolved)
        backend='nyra' if 'nyra' in experiment['method'] else 'mfa'
        routes=experiment['routes']
        rejected=routes[backend+'_requested']-routes[backend+'_accepted']
        residual_counts.update(backend_rejected_groups=rejected,
            nontriggered_issue_records=residual_counts['rejected_sentences']-rejected)
        record.update(unresolved_path=residual_path,unresolved_counts=residual_counts,
                      unresolved_shared_copy_error=residual_error,seconds=time.perf_counter()-started)
        archives.append(record)
        (target/'archives.json').write_text(json.dumps(archives,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
