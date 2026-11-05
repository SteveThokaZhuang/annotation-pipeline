"""Shared schema, lexical DP matching, and reference-free structural checks."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = Path('/mnt/data/L202500293/obed/mfa/datasets/ami-100')

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))

def norm(text):
    return re.sub(r"[^a-z0-9']", '', text.lower().replace('’', "'")).strip("'")

def match_words(ref, pred):
    """Levenshtein alignment; substitutions are NOT timing matches. Repeats retain order."""
    a, b = [norm(w['text']) for w in ref], [norm(w['text']) for w in pred]
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i][j] = min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]!=b[j-1]))
    i,j = len(a),len(b)
    pairs,deleted,inserted,substituted = [],[],[],[]
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1] == b[j-1] and a[i-1]: pairs.append((i-1,j-1))
            else: substituted.append((i-1,j-1))
            i-=1; j-=1
        elif i and dp[i][j] == dp[i-1][j]+1: deleted.append(i-1); i-=1
        else: inserted.append(j-1); j-=1
    return dict(pairs=pairs[::-1], deleted=deleted[::-1], inserted=inserted[::-1], substituted=substituted[::-1])

def structural(words, duration):
    issues=[]
    for i,w in enumerate(words):
        s,e=w.get('start'),w.get('end')
        if s is None or e is None or not math.isfinite(s) or not math.isfinite(e):
            issues.append((i,'nonfinite')); continue
        if e < s-1e-6: issues.append((i,'negative_duration'))
        elif e-s <= 1e-6: issues.append((i,'zero_duration'))
        elif e-s < .04-1e-6: issues.append((i,'short_duration'))
        if e-s > 1.5: issues.append((i,'long_duration'))
        if s < -.02 or e > duration+.02: issues.append((i,'out_of_bounds'))
        if i and s < words[i-1]['start']-1e-6: issues.append((i,'nonmonotonic'))
        if i and s < words[i-1]['end']-.02: issues.append((i,'overlap'))
    return issues

def load_manifest(path=DATASET/'manifest.jsonl'):
    rows=read_jsonl(path)
    for row in rows:
        row['audio']=str((Path(path).parent/row['audio']).resolve())
    return rows
