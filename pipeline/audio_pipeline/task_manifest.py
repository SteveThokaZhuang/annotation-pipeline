"""Compact line-oriented task transfer; job headers contain only a path."""
import json
import os
from pathlib import Path
import tempfile


def write_tasks(path,tasks):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            for task in tasks:stream.write(json.dumps(task,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def read_tasks(path):
    with open(path) as stream:
        for number,line in enumerate(stream,1):
            try:yield json.loads(line)
            except ValueError as error:raise ValueError(f'Invalid task manifest at line {number}: {path}') from error
