"""Portable path identity, with an optional persistent working tree."""
import hashlib
from pathlib import Path
from .working_cache import canonical_path, preferred_path, mapping

def relocated_path(value):return preferred_path(value)
def transform(value,canonical=False):
    pair=mapping()
    if not pair:return value
    convert=canonical_path if canonical else preferred_path
    # Resolve the environment once per object, rather than for every nested scalar.
    def visit(item):
        if isinstance(item,str):return convert(item,pair)
        if isinstance(item,list):return [visit(v) for v in item]
        if isinstance(item,tuple):return tuple(visit(v) for v in item)
        if isinstance(item,dict):return {k:visit(v) for k,v in item.items()}
        return item
    return visit(value)
def version_record(path):return None
def logical_size(path,info):return info.st_size
def semantic_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
