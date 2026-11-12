"""Versioned resource policy. Every v3 option is validated before workers start."""
import math
import os
from pathlib import Path
import subprocess


def cpu_capacity():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,max(1,math.floor(int(quota)/int(period))))
    except (OSError,ValueError):pass
    return count


def number(value,name,minimum=0,integer=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<minimum or (integer and int(value)!=value):
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"} >= {minimum}')
    return value


def keys(value,allowed,name):
    if not isinstance(value,dict):raise ValueError(name+' must be a mapping')
    extra=set(value)-set(allowed)
    if extra:raise ValueError(f'Unknown {name} options: '+', '.join(sorted(extra)))


def normalize(cfg,res):
    if res.get('version')!=3:return
    keys(res,'version configuration cpu gpu stages scheduler optimization'.split(),'resources')
    keys(res.get('configuration',{}),['conflicts'],'configuration')
    if res.get('configuration',{}).get('conflicts','error') not in ('error','resources_win'):raise ValueError('configuration.conflicts must be error or resources_win')
    cpu=res['cpu'];keys(cpu,['cores','workers','io_workers','threads_per_worker'],'cpu')
    actual=cpu_capacity();requested=cpu.get('cores','auto')
    cpu['cores']=actual if requested=='auto' else number(requested,'cpu.cores',1,True)
    if cpu['cores']>actual:raise ValueError(f'cpu.cores exceeds the available quota ({actual})')
    cpu.setdefault('workers',cpu['cores']);cpu.setdefault('threads_per_worker',1);cpu.setdefault('io_workers',cpu['cores'])
    for k in ['workers','threads_per_worker','io_workers']:number(cpu[k],'cpu.'+k,1,True)
    keys(res['gpu'],['devices','allow_cpu_fallback'],'gpu')
    if res['gpu']['devices']=='auto':
        run=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        res['gpu']['devices']=[int(x) for x in run.stdout.split()]
    devices=res['gpu']['devices']
    if not isinstance(devices,list) or len(set(devices))!=len(devices) or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in devices):raise ValueError('gpu.devices must contain unique nonnegative GPU IDs')
    optimization=res.setdefault('optimization',{})
    keys(optimization,'working_cache adaptive async_write cache resume resume_scan_workers scratch_root ipc_root stage_pipelines funasr_pipeline decoded_ram_gib_per_process manifest_cache_gib export_workers record_running_state persistent_workers task_dispatch transfers memory discovery monitoring feature_transport failure_policy finalization'.split(),'optimization')
    sections={
      'persistent_workers':('enabled idle_seconds startup_timeout_seconds poll_seconds shutdown_timeout_seconds'.split(),{}),
      'task_dispatch':('enabled max_items max_audio_seconds preserve_recordings target_bundles_per_worker order prefetch_bundles'.split(),{}),
      'transfers':('pinned non_blocking max_pinned_gib_per_worker'.split(),{}),
      'memory':('max_host_gib host_fraction queue_budget_gib min_free_gib cache_reclaim_fraction wait_timeout_seconds wait_poll_seconds'.split(),{}),
      'discovery':('enabled root max_refresh_attempts'.split(),{}),
      'monitoring':('sample_seconds report_seconds'.split(),{}),
      'feature_transport':('mode root'.split(),{})}
    for name,(allowed,_) in sections.items():keys(optimization.get(name,{}),allowed,'optimization.'+name)
    for name in ['persistent_workers','task_dispatch','discovery']:
        value=optimization.get(name,{}).get('enabled',False)
        if not isinstance(value,bool):raise ValueError(name+'.enabled must be boolean')
    transport=optimization.get('feature_transport',{}).get('mode','pickle')
    if transport not in ('pickle','memmap'):raise ValueError('feature_transport.mode must be pickle or memmap')
    for section in ['persistent_workers','task_dispatch','monitoring','transfers','memory']:
        for k,v in optimization.get(section,{}).items():
            if k in ('enabled','pinned','non_blocking','preserve_recordings','order'):continue
            if v=='auto' and k=='max_host_gib':continue
            number(v,section+'.'+k,0)
    memory=optimization.get('memory',{})
    if not 0<=memory.get('cache_reclaim_fraction',.5)<=1:raise ValueError('memory.cache_reclaim_fraction must be in [0,1]')
    if memory.get('wait_poll_seconds',2)<=0:raise ValueError('memory.wait_poll_seconds must be positive')
    fraction=memory.get('host_fraction',.7)
    if not 0<fraction<=1:raise ValueError('memory.host_fraction must be in (0,1]')
    if memory.get('max_host_gib')=='auto':
        from .memory import available_host_bytes
        memory['max_host_gib']=available_host_bytes(memory.get('cache_reclaim_fraction',.5))/2**30*fraction
    if memory.get('queue_budget_gib',0)>memory.get('max_host_gib',float('inf')):raise ValueError('Queue memory budget exceeds host memory budget')
    ipc=optimization.get('ipc_root')
    if not ipc:raise ValueError('v3 requires optimization.ipc_root')
    # Installed Linux AF_UNIX limit, including a UUID and slash used by vLLM.
    if len(os.fsencode(os.path.abspath(ipc)))+37>107:raise ValueError('optimization.ipc_root is too long for vLLM Unix sockets (107 bytes including generated filename)')
    scheduler=res['scheduler']
    keys(scheduler,'max_concurrent_stages max_concurrent_gpu_jobs_per_device shutdown_timeout_seconds gpu_lock_root'.split(),'scheduler')
    if optimization.get('persistent_workers',{}).get('enabled') and scheduler.get('max_concurrent_gpu_jobs_per_device',1)!=1:
        raise ValueError('Persistent workers require one GPU owner per device')
    stage_keys='workers cpu_workers gpu_devices result_batch_size prepare_workers prefetch_batches prepare_max_gib prepare_lookahead inference_batch_size prefetch_recordings cache adaptive batch_policy prepare_order ready_wait_seconds checkpoint feature_transport task_dispatch'.split()
    for stage,entry in res['stages'].items():
        if stage not in ('ingest','dnsmos','vad','enhance','asr1','asr2','asr3','disagreement','rover','align'):raise ValueError('Unknown stage '+stage)
        keys(entry,stage_keys,'stages.'+stage)
        for key in ['workers','cpu_workers','result_batch_size','prefetch_batches','prepare_lookahead','inference_batch_size']:
            if key in entry:number(entry[key],stage+'.'+key,1,True)
        if 'prepare_workers' in entry:number(entry['prepare_workers'],stage+'.prepare_workers',0 if stage=='vad' else 1,True)
        if 'prepare_max_gib' in entry:number(entry['prepare_max_gib'],stage+'.prepare_max_gib',.001)
        if entry.get('prepare_order','input') not in ('input','ready'):raise ValueError('prepare_order must be input or ready')
        if entry.get('prepare_order')=='ready' and stage!='vad':raise ValueError('Readiness scheduling is currently supported for VAD')
        keys(entry.get('checkpoint',{}),['max_seconds','max_mib','poll_seconds'],stage+'.checkpoint')
        for k,v in entry.get('checkpoint',{}).items():number(v,stage+'.checkpoint.'+k,.001)
        keys(entry.get('batch_policy',{}),['max_padded_seconds','min_free_gib','target_memory_fraction','calibration'],stage+'.batch_policy')
        if entry.get('gpu_devices')=='all':entry['gpu_devices']=list(devices)
    cfg['_configuration']={'version':3,'available_cpu_cores':actual,'precedence':'CLI > stage resources > model inference > stage settings > common settings','conflicts':res.get('configuration',{}).get('conflicts','error')}


def apply_stage(stage,cfg,res,spec,args):
    """Resolve scheduling controls once, before planning or constructing an adapter."""
    if res.get('version')!=3:return spec
    entry=res['stages'].get(stage,{})
    settings=spec['settings'];origins={}
    def put(key,value,origin):settings[key]=value;origins[key]=origin
    if 'inference_batch_size' in entry:
        key='batch_size'
        if stage=='dnsmos':key=None  # Independent recording count, not ONNX window count.
        if key:
            explicit=(cfg['models'].get(spec['model'],{}).get('inference',{}) if stage.startswith('asr') else cfg['settings'].get(stage,{}))
            if key in explicit and explicit[key]!=entry['inference_batch_size'] and cfg['_configuration']['conflicts']=='error':
                raise ValueError(f'Conflicting batch sizes for {stage}: model/stage settings vs resources')
            put(key,entry['inference_batch_size'],'resources.stages.'+stage+'.inference_batch_size')
    if 'adaptive' in entry:put('adaptive',entry['adaptive'],'resources.stages.'+stage+'.adaptive')
    if 'prepare_workers' in entry:
        key='frontend_workers' if stage in ('asr2','asr3') else 'prepare_workers'
        put(key,entry['prepare_workers'],'resources.stages.'+stage+'.prepare_workers')
    for k in ('prepare_max_gib','prepare_lookahead','prefetch_recordings'):
        if k in entry and stage in ('enhance','dnsmos'):put(k,entry[k],'resources.stages.'+stage+'.'+k)
    if getattr(args,'batch_size',None) is not None:
        if stage.startswith('asr') or stage=='align':put('batch_size',args.batch_size,'CLI --batch-size')
        if stage=='dnsmos':put('window_batch_size',args.batch_size,'CLI --batch-size')
    if getattr(args,'prepare_workers',None) is not None:
        key='frontend_workers' if stage in ('asr2','asr3') else 'prepare_workers'
        put(key,args.prepare_workers,'CLI --prepare-workers')
    if stage in ('asr2','asr3'):
        for source,target in [('prepare_max_gib','frontend_max_gib'),('prepare_lookahead','frontend_lookahead')]:
            if source in entry:put(target,entry[source],'resources.stages.'+stage+'.'+source)
    spec['configuration_sources']=origins
    return spec


def validate_nested(cfg,res):
    if res.get('version')!=3:return
    opt=res['optimization']
    keys(opt.get('resume',{}),['validation'],'optimization.resume')
    if opt.get('resume',{}).get('validation','full') not in ('full','trusted'):
        raise ValueError('resume.validation must be full or trusted')
    keys(opt.get('finalization',{}),['reuse_unchanged_ingest_manifest','incremental_recording_ledger'],'optimization.finalization')
    for k,v in opt.get('finalization',{}).items():
        if not isinstance(v,bool):raise ValueError('finalization.'+k+' must be boolean')
    keys(cfg,['assignments','dataset_root','datasets','environments','models','output_root','settings','stage_models','_configuration'],'pipeline')
    keys(opt.get('failure_policy',{}),['model_initialization'],'failure_policy')
    if opt.get('failure_policy',{}).get('model_initialization','stop') not in ('stop','route_missing'):raise ValueError('Invalid model initialization failure policy')
    for name,section in opt.get('stage_pipelines',{}).items():
        if name not in ('qwen3_asr','fireredasr','align','vad','dnsmos'):raise ValueError('Unknown stage pipeline '+name)
        keys(section,['enabled','lookahead_batches','max_gib','cross_recording_batching'],'stage_pipelines.'+name)
        if section.get('lookahead_batches',2) not in (1,2,3,4):raise ValueError('Pipeline depth must be in [1,4]')
        number(section.get('max_gib',1),'pipeline.max_gib',.001)
        if section.get('cross_recording_batching') and name!='dnsmos':raise ValueError('Cross-recording window batching applies to DNSMOS')
    keys(opt.get('funasr_pipeline',{}),['enabled','lookahead_batches','max_gib'],'funasr_pipeline')
    for section,allowed in [('adaptive',['memory_feedback','max_padded_seconds','min_free_gib','target_memory_fraction','calibration','models']),('cache',['enabled','root','max_gib']),('working_cache',['enabled','sync_enabled','root','sync_interval_seconds','sync_mib_per_second','min_free_gib'])]:
        keys(opt.get(section,{}),allowed,'optimization.'+section)
    if opt.get('adaptive',{}).get('memory_feedback','auto') not in ('auto','torch','driver_free'):raise ValueError('Invalid memory_feedback')
    for name in ('persistent_workers','monitoring'):
        for k,v in opt.get(name,{}).items():
            if k!='enabled':number(v,name+'.'+k,.001)
    for name in ('discovery','feature_transport'):
        if (opt.get(name,{}).get('enabled') or opt.get(name,{}).get('mode')=='memmap') and not opt[name].get('root'):raise ValueError(name+'.root is required')
    dispatches=[opt.get('task_dispatch',{})]+[e.get('task_dispatch',{}) for e in res['stages'].values()]
    for dispatch in dispatches:
        if 'prefetch_bundles' in dispatch:
            number(dispatch['prefetch_bundles'],'task_dispatch.prefetch_bundles',0,True)
            if dispatch['prefetch_bundles']>1:raise ValueError('prefetch_bundles supports at most one upcoming bundle')
        if 'max_audio_seconds' in dispatch:number(dispatch['max_audio_seconds'],'task_dispatch.max_audio_seconds',.001)
        if dispatch.get('order','input') not in ('input','longest_first'):raise ValueError('Invalid task dispatch order')
        for k in ('max_items','target_bundles_per_worker'):
            if k in dispatch:number(dispatch[k],'task_dispatch.'+k,1,True)
    for name,entry in res['stages'].items():
        dispatch=dict(opt.get('task_dispatch',{}),**entry.get('task_dispatch',{}))
        if dispatch.get('prefetch_bundles',0):
            if name!='dnsmos':raise ValueError('Cross-bundle preparation currently supports DNSMOS only; configure its stage task_dispatch')
            if not dispatch.get('enabled') or not opt.get('persistent_workers',{}).get('enabled') or not entry.get('prefetch_recordings',True):raise ValueError('prefetch_bundles requires dispatch, persistent workers and recording prefetch')
        keys(entry.get('cache',{}),['enabled','root','max_gib'],name+'.cache')
        keys(entry.get('feature_transport',{}),['mode','root'],name+'.feature_transport')
        if entry.get('feature_transport',{}).get('mode','pickle') not in ('pickle','memmap'):raise ValueError('Invalid stage feature transport mode')
        keys(entry.get('task_dispatch',{}),['enabled','max_items','max_audio_seconds','preserve_recordings','target_bundles_per_worker','order','prefetch_bundles'],name+'.task_dispatch')
        if 'task_dispatch' in entry:
            for key in ('max_items','max_audio_seconds'):
                if key in entry['task_dispatch']:number(entry['task_dispatch'][key],name+'.task_dispatch.'+key,1,key=='max_items')
        for key in ('max_padded_seconds','min_free_gib'):
            if key in entry.get('batch_policy',{}):number(entry['batch_policy'][key],name+'.'+key,.001)
    supported={
      'vad':'speech_threshold min_speech_ms padding merge_gap chunk_seconds gpu_batch_size min_chunk_seconds min_silence_seconds fallback_search_seconds energy_window_ms',
      'dnsmos':'window_seconds hop_seconds window_batch_size prepare_workers prepare_lookahead prepare_max_gib prefetch_recordings',
      'enhance':'host_headroom_fraction gpu_headroom_fraction assembly_block_seconds method_version parallel_segments auto_policy target_seconds max_segment_seconds min_segment_seconds boundary_search_seconds min_silence_seconds context_seconds crossfade_seconds channel_batch_size estimated_gpu_gib_per_second estimated_ram_gib_per_second gru_chunk_frames max_estimated_ram_gib max_gpu_seconds max_seconds prepare_max_gib prepare_workers prepare_lookahead prefetch_recordings prepare_spectral_features async_audio_write',
      'asr':'language max_new_tokens batch_size adaptive',
      'disagreement':'language reference metric threshold',
      'rover':'unit alpha null_confidence',
      'align':'language batch_size adaptive timestamp_tolerance'}
    for name,values in cfg['settings'].items():
        if name not in supported:raise ValueError('Unknown stage settings '+name)
        keys(values,supported[name].split(),'settings.'+name)
    inference='adaptive batch_size engine beam_size encoder_batch_size encoder_max_padded_frames frontend_workers frontend_max_gib frontend_lookahead frontend_seed frontend_dither vllm decoder_export_root'.split()
    for name,model in cfg['models'].items():keys(model.get('inference',{}),inference,'models.'+name+'.inference')
    if 'vad' in cfg['settings']:validate_vad_split(cfg['settings']['vad'])
    enhance=cfg['settings'].get('enhance',{})
    for key in ('host_headroom_fraction','gpu_headroom_fraction'):
        if not 0<enhance.get(key,.75)<=1:raise ValueError(key+' must be in (0,1]')
    number(enhance.get('assembly_block_seconds',10),'enhance.assembly_block_seconds',.001)
    if enhance.get('method_version')=='auto_v3':
        keys(enhance['auto_policy'],['whole_max_seconds','whole_max_ram_gib','whole_max_gpu_gib'],'enhance.auto_policy')
        for key in ('whole_max_seconds','whole_max_ram_gib','whole_max_gpu_gib'):number(enhance['auto_policy'][key],'auto_policy.'+key,.001)
        module=cfg.get('stage_models',{}).get('enhance',{}).get('module','')
        if not module.endswith('adapter_auto.py'):raise ValueError('auto_v3 requires the auto enhancement adapter')
        if enhance.get('parallel_segments') and not (opt.get('persistent_workers',{}).get('enabled') and opt.get('task_dispatch',{}).get('enabled')):raise ValueError('parallel_segments requires persistent workers and task dispatch')


def validate_vad_split(settings):
    from .common import BASE,load_module
    helper=load_module(BASE/'vendor/vad_common.py','vad_split_policy')
    helper.split_policy(settings['chunk_seconds'],settings.get('min_chunk_seconds',10.),
        settings.get('min_silence_seconds',.1),settings.get('fallback_search_seconds',2.),
        settings.get('energy_window_ms',100.))
