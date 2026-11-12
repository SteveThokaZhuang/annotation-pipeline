"""Resolve explicit dependencies into immutable task descriptions."""
from pathlib import Path
import math
import json
import subprocess
from functools import lru_cache
from .common import BASE,ROOT,digest,fingerprint,read,validate,version_tree
from .cache_version import shared_inference_version,adapter_inference_version,source_version
from .failure_policy import hypothesis_error,is_infrastructure_error
from .resume import checked_entries
from .configuration import python_for,absolute

_DESCRIPTORS={}

@lru_cache(maxsize=16)
def inherited_sites(python):
    return json.loads(subprocess.check_output([python,'-c','import site,json;print(json.dumps(site.getsitepackages()))'],text=True,timeout=60))



def hypothesis_payload(value):
    """Keep text/model metadata; reference authoritative ASR provenance once.

    The caller validates the full saved chunk first. Copying that chunk and its
    nested provenance into every downstream hypothesis inflates task files.
    """
    if value is None:return None
    result={k:v for k,v in value.items() if k not in ['chunk','_provenance']}
    provenance=value.get('_provenance',{})
    result['source_result']={key:provenance.get(key) for key in ['stage','model','signature']}
    result['source_result']['item']=value.get('chunk',{}).get('id')
    return result

ADAPTERS={
 'ingest':('firered-vad','audio_ingestion/final_script/adapter.py',None),
 'dnsmos':('dnsmos','dnsmos/final_script/adapter.py','dnsmos/checkpoint'),
 'vad':('firered-vad','VAD/FireRedVAD/final_script/adapter.py','VAD/FireRedVAD/checkpoints/VAD'),
 'enhance':('denoising-comparison','deepfilternet3/final_script/adapter.py','deepfilternet3/checkpoint/DeepFilterNet3'),
 'disagreement':('firered-vad','disagreement/final_script/adapter.py',None),
 'rover':('firered-vad','roverfusion/final_script/adapter.py','roverfusion/bin/rover'),
 'align':('qwen3-asr','qwen3-fa-mfa/final_script/adapter.py','qwen3-asr/checkpoints/Qwen3-ForcedAligner-0.6B')}


def descriptor(stage,cfg,args):
    cache_key=digest([stage,cfg,getattr(args,'dataset',None),args.model,args.language,args.threshold,getattr(args,'batch_size',None),getattr(args,'adaptive',None),getattr(args,'prepare_workers',None)])
    if cache_key in _DESCRIPTORS:return _DESCRIPTORS[cache_key]
    if stage.startswith('asr'):
        name=args.model or cfg['assignments'][stage]
        if name not in cfg['models']:raise ValueError('Unknown model: '+name)
        spec=dict(cfg['models'][name]);spec['model']=name
        spec.setdefault('python',python_for(cfg,spec.pop('environment',name)))
        spec['settings']=dict(cfg['settings']['asr'])
        spec['settings'].update(spec.pop('inference',{}))
        if getattr(args,'prepare_workers',None) is not None and 'frontend_workers' in spec['settings']:
            spec['settings']['frontend_workers']=args.prepare_workers
        if getattr(args,'batch_size',None) is not None:spec['settings']['batch_size']=args.batch_size
        if getattr(args,'adaptive',None) is not None:spec['settings']['adaptive']=args.adaptive
    else:
        env,module,checkpoint=ADAPTERS[stage]
        spec=dict(model=stage,python=python_for(cfg,env),module=str(BASE/'models'/module),
                  checkpoint=str(BASE/'models'/checkpoint) if checkpoint else None,
                  min_free_vram_gib=4 if stage=='align' else 1,requires_gpu=stage=='align',
                  settings=dict(cfg['settings'].get(stage,{})))
    if stage in cfg.get('stage_models',{}):spec.update(cfg['stage_models'][stage])
    # MFA resources are language-specific.  Keep the routing explicit at the
    # dataset boundary so a Chinese corpus can never silently fall back to the
    # English dictionary/acoustic model.  The dataset catalog still controls
    # the Qwen/ASR language; this only selects the MFA repair resources.
    if stage == 'align':
        mfa_routes = spec.get('mfa_by_dataset', {})
        route = mfa_routes.get(getattr(args, 'dataset', None))
        if route:
            for key in ('mfa_python', 'dictionary', 'acoustic_model'):
                if key in route:
                    spec[key] = route[key]
            spec['mfa_language'] = route.get('language', 'Chinese')
            spec['settings'] = dict(spec.get('settings', {}), language=route.get('language', 'Chinese'))
    spec['module']=absolute(spec['module'])
    if spec.get('checkpoint'):spec['checkpoint']=absolute(spec['checkpoint'])
    if stage=='align' and getattr(args,'batch_size',None) is not None:spec['settings']['batch_size']=args.batch_size
    if stage=='dnsmos' and getattr(args,'batch_size',None) is not None:spec['settings']['window_batch_size']=args.batch_size
    if stage=='align' and getattr(args,'adaptive',None) is not None:spec['settings']['adaptive']=args.adaptive
    if args.language:
        spec['settings']['language']=args.language
    elif stage.startswith('asr') or stage=='align' or (stage=='disagreement' and spec['settings'].get('language','auto')=='auto'):
        spec['settings']['language']=cfg['datasets'].get(args.dataset,{}).get('language',spec['settings']['language'])
    if args.threshold is not None and stage=='disagreement':spec['settings']['threshold']=args.threshold
    if stage=='disagreement' and (not math.isfinite(spec['settings']['threshold']) or spec['settings']['threshold']<0):
        raise ValueError('Disagreement threshold must be finite and nonnegative')
    if stage=='vad':
        v=spec['settings']
        from .config_v3 import validate_vad_split
        validate_vad_split(v)
        if not 0<v['chunk_seconds']<=30 or min(v['padding'],v['merge_gap'])<0:raise ValueError('VAD chunk_seconds must be in (0,30]; padding/gap must be nonnegative')
    if stage.startswith('asr'):
        engine=spec['settings'].get('engine','transformers')
        if engine not in ('transformers','vllm'):raise ValueError('ASR engine must be transformers or vllm')
        helpers=spec['settings'].get('frontend_workers',0)
        if not isinstance(helpers,int) or isinstance(helpers,bool) or helpers<0:raise ValueError('frontend_workers must be a nonnegative integer')
        if spec['model']=='fireredasr' and engine=='vllm' and spec['settings'].get('beam_size',3)!=1:
            raise ValueError('FireRed vLLM embedding prompts require explicit beam_size: 1')
    if stage.startswith('asr') and spec['settings']['max_new_tokens']<1:raise ValueError('ASR token limit must be positive')
    if stage.startswith('asr') and spec['settings'].get('batch_size',1)<1:
        raise ValueError('ASR batch size must be positive')
    # Include reused source in invalidation, without hashing multi-GB weights each run.
    reused={
        'enhance':[BASE/'models/deepfilternet3/final_script/preparation.py',BASE/'models/deepfilternet3/final_script/feature_preparation.py',BASE/'models/deepfilternet3/final_script/recurrent.py',BASE/'models/deepfilternet3/DeepFilterNet/df',BASE/'pipeline/audio_pipeline/staging.py'],
        'vad':[BASE/'vendor/vad_common.py',BASE/'pipeline/audio_pipeline/vad_preparation.py',BASE/'vendor/fireredvad/core/detect_model.py'],
        'dnsmos':[BASE/'models/dnsmos/scripts/test_dnsmos.py'],
        'align':[BASE/'vendor/qwen_asr/inference/assets/korean_dict_jieba.dict'],
        'funasr':[BASE/'models/funasr/scripts/backend.py',BASE/'models/funasr/final_script/encoder_batching.py',BASE/'pipeline/audio_pipeline/asr_completion.py',BASE/'models/funasr/source/funasr'],
        'fireredasr':[BASE/'models/firered-asr2/final_script/backend.py',BASE/'pipeline/audio_pipeline/asr_completion.py',BASE/'models/firered-asr2/scripts/FireRedASR2S/fireredasr2s'],
        'rover':[BASE/'models/roverfusion/fuse.py']}.get(spec['model'],[])
    if stage=='enhance' and spec['settings'].get('method_version') in ('vad_chunked_v1','auto_v3'):
        reused=reused+[BASE/'pipeline/audio_pipeline/enhancement_chunks_v1.py',
                       BASE/'models/deepfilternet3/final_script/adapter.py',BASE/'models/deepfilternet3/final_script/adapter_chunked.py']
    if cfg.get('_resources_v3'):
        from .config_v3 import apply_stage
        spec=apply_stage(stage,cfg,cfg['_resources_v3'],spec,args)
    spec['adapter_version']=digest([adapter_inference_version(spec['module']),shared_inference_version()]+
                                    [source_version(p) for p in reused]+[source_version(BASE/'vendor')])
    environment=Path(spec['python']).parent.parent
    prefixes=('torch-','transformers-','qwen_asr-','funasr-','fireredvad-','numpy-','scipy-','soundfile-','onnxruntime-','onnxruntime_gpu-','deepfilterlib-')
    sites=list(environment.glob('lib/python*/site-packages'))
    venv=environment/'pyvenv.cfg'
    if venv.is_file() and any(line.strip().lower()=='include-system-site-packages = true' for line in venv.read_text().splitlines()):
        sites=[Path(site) for site in inherited_sites(spec['python']) if Path(site).is_dir()]
        prefixes+=('vllm-','peft-','torchaudio-','kaldi_native_fbank-','kaldiio-')
    metadata=[p/'METADATA' for site in sites for p in site.iterdir() if p.name.endswith('.dist-info') and p.name.startswith(prefixes)]
    spec['environment_version']=[fingerprint(p) for p in sorted(metadata) if p.is_file()]
    spec['checkpoint_version']=version_tree(spec['checkpoint']) if spec.get('checkpoint') else None
    _DESCRIPTORS[cache_key]=spec
    return spec


def head_results(state,stage,required=True):
    from .manifest_memory import get_results
    return get_results(state,stage,required,_head_results)


def _head_results(state,stage,required=True):
    try:head=state.head(stage)
    except RuntimeError:
        if required:raise
        return {},None
    values={};fingerprints={}
    trusted=getattr(state,'resume_validation','full')=='trusted'
    for entry,value in checked_entries(head['entries'],stage,
            workers=getattr(state,'scan_workers',16),stop_file=getattr(state,'stop_file',None),
            validate_results=not trusted):
        if entry['status']!='complete':
            state.note_issue(stage,head.get('model',stage),entry['id'],entry.get('error') or entry['status'],entry.get('input'))
            continue
        try:
            if value is None:raise ValueError('Saved result is missing or invalid')
            if not trusted and digest(value)!=entry['digest']:raise ValueError('Saved result checksum changed')
            source=value['_provenance']['input']
            source=source.get('chunk',source)
            for key in ['source_fingerprint','audio_fingerprint']:
                fp=source.get(key)
                if fp and not trusted:
                    path=fp['path']
                    # Snapshot each source once per upstream-manifest load.
                    # Workers still revalidate sources before inference/save.
                    if path not in fingerprints:fingerprints[path]=fingerprint(path)
                    if fingerprints[path]!=fp:raise ValueError('Source/audio changed; rerun affected upstream stages')
        except Exception as error:
            if is_infrastructure_error(error):raise
            state.note_issue(stage,head.get('model',stage),entry['id'],str(error),entry.get('input'))
            continue
        # Keep historical flagged placeholders unchanged: an in-flight disagreement
        # run may already have saved them in its routing snapshot. ROVER excludes
        # them from votes; future workers record such results as failed tasks.
        if stage.startswith('asr') and hypothesis_error(value):
            state.note_issue(stage,head.get('model',stage),entry['id'],hypothesis_error(value),value.get('chunk'))
        values[entry['id']]=value
    return values,head


def recordings(state):
    values,head=head_results(state,'ingest')
    return values


def chunks(state,enhanced):
    source=recordings(state)
    values,_=head_results(state,'vad')
    enhancements,_=head_results(state,'enhance',required=enhanced) if enhanced else ({},None)
    output=[];ingest_versions={};enhancement_versions={};enhancement_fingerprints={}
    for record in values.values():
        for original in record['chunks']:
            chunk=dict(original)
            ingest=source.get(chunk['recording_id'])
            if ingest is None:continue
            if ingest['source_fingerprint']!=chunk['source_fingerprint']:
                state.note_issue('vad','vad',chunk['id'],'VAD input differs from ingestion; rerun vad',chunk)
                continue
            chunk['audio_path']=chunk['path']
            identity=chunk['recording_id']
            if identity not in ingest_versions:ingest_versions[identity]=digest(ingest)
            chunk['ingest_version']=ingest_versions[identity]
            if enhanced:
                value=enhancements.get(chunk['recording_id'])
                if value is None:
                    state.note_issue('enhance','enhance',chunk['id'],'Missing enhanced recording; run enhance or --no-enhance',chunk)
                    continue
                fp=value['_provenance']['input']['source_fingerprint']
                if fp!=ingest['source_fingerprint']:
                    state.note_issue('enhance','enhance',chunk['id'],'Enhancement is stale; rerun enhance',chunk)
                    continue
                chunk['audio_path']=value['path']
                chunk['audio_start_sample']=round(chunk['start']*value['sample_rate'])
                chunk['audio_end_sample']=min(value['total_samples'],round(chunk['end']*value['sample_rate']))
                if identity not in enhancement_versions:
                    enhancement_versions[identity]=digest(value)
                    enhancement_fingerprints[identity]=fingerprint(value['path'])
                chunk['enhancement_version']=enhancement_versions[identity]
                chunk['audio_fingerprint']=enhancement_fingerprints[identity]
            output.append(chunk)
    return sorted(output,key=lambda c:(c['recording_id'],c['channel'],c['start_sample']))


def items(stage,cfg,args,state,enhanced):
    if stage=='ingest':
        from .datasets import discover
        return discover(cfg,args.dataset,getattr(state,'stop_file',None))
    if stage=='enhance' and cfg['settings'].get('enhance',{}).get('method_version') in ('vad_chunked_v1','auto_v3'):
        from .enhancement_chunks_v1 import plan_recording
        from .common import load_module
        settings=descriptor(stage,cfg,args)['settings'] if cfg.get('_resources_v3') else cfg['settings']['enhance']
        choose=None
        if settings['method_version']=='auto_v3':
            choose=load_module(BASE/'models/deepfilternet3/final_script/adapter_auto.py','auto_selection').choose
        vad,_=head_results(state,'vad',required=False)
        result=[]
        for record in recordings(state).values():
            item=dict(record)
            if choose:
                item['_enhance_method']=choose(item,settings)
                if item['_enhance_method']=='whole_sequence':
                    result.append(item);continue
            try:item['enhancement_plan']=plan_recording(item,vad[item['id']],settings)
            except (ValueError,KeyError) as error:item['_planning_error']='Cannot plan VAD-guided enhancement: '+str(error)
            result.append(item)
        return result
    if stage in ['vad','dnsmos','enhance']:
        return [dict(value) for value in recordings(state).values()]
    if stage in ['asr1','asr2']:
        return chunks(state,enhanced)
    if stage=='asr3':
        path=state.head_path('disagreement').with_suffix('.high.json')
        if not path.is_file():raise RuntimeError('Missing selective high-disagreement manifest; run disagreement for this dataset first')
        selected=read(path)
        routed=[]
        for original in selected['chunks']:
            chunk=dict(original)
            try:
                for key in ['source_fingerprint','audio_fingerprint']:
                    fp=chunk.get(key)
                    if fp and getattr(state,'resume_validation','full')!='trusted' and fingerprint(fp['path'])!=fp:
                        raise ValueError('Routed audio changed; rerun affected upstream stages')
            except Exception as error:
                if is_infrastructure_error(error):raise
                chunk['_planning_error']=str(error)
            routed.append(chunk)
        return routed
    if stage in ['disagreement','rover']:
        a,ha=head_results(state,'asr1',required=False);b,hb=head_results(state,'asr2',required=False)
        if stage=='disagreement' and (ha is None or hb is None):
            raise RuntimeError('Disagreement needs saved asr1 and asr2 manifests; run each independently first')
        if stage=='rover':
            routes,_=head_results(state,'disagreement');c,hc=head_results(state,'asr3',required=False)
        result=[]
        current=chunks(state,enhanced)
        for chunk in current:
            identity=chunk['id']
            def hypothesis(values):
                value=values.get(identity)
                if value and digest(value['chunk'])!=digest(chunk):
                    state.note_issue('asr_inputs','asr',identity,'ASR audio/chunk configuration changed; rerun affected ASR',chunk)
                    return None
                return hypothesis_payload(value)
            hypotheses=[hypothesis(a),hypothesis(b)]
            item=dict(id=identity,chunk=chunk,hypotheses=hypotheses)
            if stage=='rover':
                route=routes.get(identity)
                if route is None:
                    result.append(dict(item,_planning_error='Missing disagreement decision; rerun disagreement'))
                    continue
                if route['hypotheses']!=hypotheses:
                    result.append(dict(item,_planning_error='ASR hypotheses changed; rerun disagreement'))
                    continue
                item['routing']='high' if route['high'] else 'low'
                item['routing_version']=digest(route)
                if route['high']:
                    value=c.get(identity)
                    if value:
                        fallback_chunk=dict(value['chunk']);version=fallback_chunk.pop('routing_version',None)
                        if version!=digest(route) or fallback_chunk!=chunk:
                            state.note_issue('asr3','asr3',identity,'ASR3 routing changed; rerun asr3',chunk)
                            value=None
                    item['hypotheses'].append(hypothesis_payload(value))
            result.append(item)
        return result
    if stage=='align':
        values,_=head_results(state,'rover')
        result=[]
        for key,value in values.items():
            if value.get('transcript_status')=='no_transcript':
                state.skipped.append(dict(id=key,chunk=value['chunk'],reason='no_transcript'))
                continue
            languages={h['language'] for h in value['hypotheses'] if h and h.get('language')}
            result.append(dict(id=key,chunk=value['chunk'],text=value['text'],fusion_version=digest(value),
                               language=next(iter(languages)) if len(languages)==1 else None))
        return result
    raise ValueError(stage)
