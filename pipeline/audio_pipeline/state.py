import json
from pathlib import Path
import sqlite3
import time
from .common import read, validate
from .relocation import relocated_path
from .working_cache import canonical_path


class State:
    def __init__(self, root):
        self.dependency_failures={}
        self.skipped=[]
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.db = sqlite3.connect(str(self.root/'state.sqlite'), timeout=60)
        # Rollback journal works on network filesystems; WAL requires shared-memory semantics.
        self.db.execute('PRAGMA busy_timeout=60000')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS tasks(stage TEXT, model TEXT, item TEXT, signature TEXT,
          status TEXT, output TEXT, error TEXT, seconds REAL, updated REAL,
          PRIMARY KEY(stage,model,item,signature));
        CREATE TABLE IF NOT EXISTS heads(stage TEXT PRIMARY KEY, manifest TEXT);
        CREATE INDEX IF NOT EXISTS task_progress ON tasks(stage,model,updated,status);
        ''')
        self.db.commit()

    def cached(self,stage,model,item,signature):
        row=self.db.execute('SELECT status,output FROM tasks WHERE stage=? AND model=? AND item=? AND signature=?',
                            (stage,model,item,signature)).fetchone()
        return self.cached_from_row(row)

    def cached_from_row(self,row):
        if row and row[0]=='complete':
            try:
                value=read(row[1]);validate(value)
                return value
            except (OSError,ValueError,KeyError,IndexError,TypeError):
                return None

    def task_rows(self,stage,model,keys):
        """Fetch task state for a group of IDs in one SQLite query."""
        keys=list(keys)
        if not keys:return {}
        wanted=set(keys)
        ids=list(dict.fromkeys(item for item,_ in keys))
        rows={}
        for offset in range(0,len(ids),400):
            group=ids[offset:offset+400]
            marks=','.join('?' for _ in group)
            query=f'SELECT item,signature,status,output,error FROM tasks WHERE stage=? AND model=? AND item IN ({marks})'
            for item,signature,status,output,error in self.db.execute(query,(stage,model,*group)):
                key=(item,signature)
                if key in wanted:rows[key]=(status,output,error)
        return rows

    def cached_many(self,stage,model,keys):
        rows=self.task_rows(stage,model,keys)
        values={}
        for key,row in rows.items():
            value=self.cached_from_row(row)
            if value is not None:values[key]=value
        return values

    def put(self,stage,model,item,signature,status,output=None,error=None,seconds=0):
        self.db.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?,?)',
                        (stage,model,item,signature,status,canonical_path(output) if output else output,error,seconds,time.time()))
        self.db.commit()

    def put_many(self,rows):
        """Commit a model batch together; an interrupted batch is retried."""
        now=time.time()
        self.db.executemany('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?,?)',
                            [(*row[:5],canonical_path(row[5]) if row[5] else row[5],*row[6:],now) for row in rows])
        self.db.commit()

    def head_path(self,stage):
        row=self.db.execute('SELECT manifest FROM heads WHERE stage=?',(stage,)).fetchone()
        if not row:
            raise RuntimeError(f'Missing {stage} manifest; run {stage} for this dataset/output namespace first')
        return Path(relocated_path(row[0]))

    def head(self,stage):
        return read(self.head_path(stage))

    def set_head(self,stage,path):
        self.db.execute('INSERT OR REPLACE INTO heads VALUES(?,?)',(stage,canonical_path(path)))
        self.db.commit()

    def note_issue(self,stage,model,item,error,source=None):
        self.dependency_failures[(stage,model,item)]=dict(stage=stage,model=model,id=item,error=error,source=source)
