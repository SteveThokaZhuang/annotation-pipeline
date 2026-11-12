import hashlib
import importlib.util
import json
import os
import copy
from pathlib import Path
import re
import tempfile
import unicodedata
from functools import lru_cache
from .relocation import relocated_path, transform, logical_size

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE
STAGES = ['ingest','dnsmos','vad','enhance','asr1','asr2','disagreement','asr3','rover','align']


def digest(value):
    return hashlib.sha256(json.dumps(transform(value,canonical=True), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(relocated_path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read(path):
    raw=relocated_path(path)
    base,separator,index=raw.rpartition('#result=')
    if separator:
        if not index.isdecimal():raise ValueError('Invalid batched result reference')
        return copy.deepcopy(_read_batch(base)['results'][int(index)])
    return transform(json.loads(Path(relocated_path(path)).read_text()))


@lru_cache(maxsize=64)
def _read_batch(path):
    # Batch paths are unique and immutable. The cache avoids reopening a shared
    # filesystem object for every chunk during manifest assembly.
    return transform(json.loads(Path(relocated_path(path)).read_text()))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path):
    path = Path(relocated_path(path)).resolve()
    info = path.stat()
    return dict(path=str(path), size=logical_size(path,info), mtime_ns=info.st_mtime_ns)


def version_tree(path):
    path = Path(relocated_path(path))
    if path.is_file():
        return fingerprint(path)
    return [fingerprint(p) for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts]


def tokens(text, unit='mixed'):
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
    def meaningful_punctuation(index,c):
        if c in "'%$€£¥+":return True
        if c in '.,' and 0<index<len(text)-1 and text[index-1].isdigit() and text[index+1].isdigit():return True
        if c=='-' and index+1<len(text) and text[index+1].isdigit():return True
        return False
    text = ''.join(' ' if unicodedata.category(c)[0] in 'PS' and not meaningful_punctuation(i,c) else c for i,c in enumerate(text))
    text = re.sub(r"(?<!\w)'|'(?!\w)", ' ', text)
    if unit == 'char':
        return [c for c in text if not c.isspace()]
    if unit == 'word':
        return text.split()
    # Chinese/Japanese characters are lexical units; Latin words remain whole.
    return re.findall(r"[\u3400-\u9fff\u3040-\u30ff]|[-+]?\d+(?:[.,]\d+)*%?|[^\W_\u3400-\u9fff\u3040-\u30ff]+(?:'[^\W_\u3400-\u9fff\u3040-\u30ff]+)*|[$€£¥%+]", text)


def detokenize(sequence):
    value = ' '.join(sequence)
    return re.sub(r'(?<=[\u3400-\u9fff\u3040-\u30ff]) (?=[\u3400-\u9fff\u3040-\u30ff])', '', value)


def distance(a, b):
    """Exact unit-cost Levenshtein distance using bit-parallel integer operations.

    Normalized tokens remain unchanged. Python's arbitrary-width integers process
    many reference positions per operation instead of allocating every DP cell.
    """
    if len(a)>len(b):a,b=b,a
    width=len(a)
    if not width:return len(b)
    matches={}
    for i,token in enumerate(a):matches[token]=matches.get(token,0)|(1<<i)
    mask=(1<<width)-1;high=1<<(width-1);positive=mask;negative=0;score=width
    for token in b:
        equal=matches.get(token,0);vertical=equal|negative
        horizontal=(((equal&positive)+positive)^positive)|equal
        add=negative|~(horizontal|positive);subtract=positive&horizontal
        if add&high:score+=1
        if subtract&high:score-=1
        add=(add<<1)|1;subtract<<=1
        positive=(subtract|~(vertical|add))&mask
        negative=add&vertical&mask
    return score


def disagreement(a,b,unit='mixed'):
    a,b = tokens(a,unit),tokens(b,unit)
    return distance(a,b)/max(1,len(a),len(b))


def validate(result):
    if not isinstance(result,dict):
        raise ValueError('Stage output must be an object')
    json.dumps(result,allow_nan=False)
    if 'text' in result and not isinstance(result['text'],str):
        raise ValueError('Transcript must be a string')
    for c in result.get('chunks',[]):
        if not 0 <= c['start_sample'] < c['end_sample'] <= c['total_samples']:
            raise ValueError('Invalid chunk sample boundaries')
    for artifact in result.get('artifacts',[]):
        p = Path(artifact['path'])
        if not p.is_file() or fingerprint(p) != artifact:
            raise ValueError('Missing or modified artifact: '+str(p))


def guard_generation(model):
    """Reject truncated generations before they can become completed transcripts."""
    original=model.generate
    stats=[]
    def generate(*args,**kwargs):
        output=original(*args,**kwargs)
        sequences=output.sequences if hasattr(output,'sequences') else output
        prompt=kwargs.get('input_ids')
        offset=prompt.shape[1] if prompt is not None else 0
        eos=kwargs.get('eos_token_id',model.generation_config.eos_token_id)
        eos={eos} if isinstance(eos,int) else set(eos or [])
        if not eos:raise RuntimeError('Cannot verify completion: model has no EOS token ID')
        stats.clear()
        for row in sequences[:,offset:].tolist():
            end=next((i for i,t in enumerate(row) if t in eos),None)
            stats.append(dict(generated_tokens=len(row) if end is None else end+1,finish_reason='length' if end is None else 'eos'))
        if any(s['finish_reason']!='eos' for s in stats):raise RuntimeError('ASR generation reached its token limit without EOS')
        return output
    model.generate=generate
    return stats
