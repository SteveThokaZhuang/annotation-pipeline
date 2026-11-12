"""Semantic cache version: resource settings never masquerade as model settings."""
import hashlib
from .relocation import semantic_hash
from pathlib import Path
from .common import BASE,ROOT

def source_version(path):
    path=Path(path)
    if path.is_file():return semantic_hash(path)
    return {str(p.relative_to(path)):semantic_hash(p) for p in sorted(path.rglob('*.py')) if '__pycache__' not in p.parts}

def shared_inference_version():
    return {name:source_version(BASE/'pipeline/audio_pipeline'/name) for name in ['common.py','audio.py','ram_audio.py','failure_policy.py','batching.py','frontends.py','vllm_decoder.py','engines.py','dnsmos_cross.py','transfers.py','feature_transport.py','enhancement_dispatch.py']}

def adapter_inference_version(module):return source_version(module)


def inference_descriptor(spec):
    """Resource queue sizes do not change the identity of model inference."""
    result={k:v for k,v in spec.items() if k not in ('python','configuration_sources')}
    ignored={'frontend_workers','frontend_max_gib','frontend_lookahead',
             'prepare_workers','prepare_max_gib','prepare_lookahead','prefetch_recordings','async_audio_write','feature_transport','parallel_segments'}
    result['settings']={k:v for k,v in spec['settings'].items() if k not in ignored}
    return result
