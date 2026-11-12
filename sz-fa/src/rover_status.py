"""Print compact live counts, without mistaking in-flight chunks for results."""
import json
from pathlib import Path
import re

directory=Path('/tmp/fa_open_yap_results')
status=json.loads((directory/'status.json').read_text())
if status.get('status')=='complete':
    print(json.dumps(status));raise SystemExit
method=status['method']
folder={'qwen+mfa+nyra':'qwen3-fa+mfa+nyra','qwen+mfa':'qwen3-fa+mfa','qwen+nyra':'qwen3-fa+nyra'}[method]
workers=[]
for item in status['workers']:
    gpu=item['gpu'];log=directory/folder/f'gpu_{gpu}.log'
    lines=log.read_text().splitlines() if log.exists() else []
    qwen=re.findall(r'Qwen block=(\d+) chunks=(\d+)','\n'.join(lines))
    block=int(qwen[-1][0]) if qwen else 0
    mfa=Path('/tmp/fa_open_yap_work')/folder/f'gpu_{gpu}'/f'b{block:05d}'/'mfa'
    progress=0
    for path in (mfa/'work/corpus/alignment/log').glob('align.*.log'):
        progress+=path.read_text(errors='replace').count('Processing ')
    workers.append(dict(gpu=gpu,routed=item.get('processed_chunks',0),
        qwen=int(qwen[-1][1]) if qwen else 0,zeros=item.get('zero_final',0),
        failures=item.get('failures',0),mfa_tasks_started=progress,last_line=lines[-1][:160] if lines else 'loading'))
print(json.dumps(dict(method=method,elapsed_seconds=round(status['elapsed_seconds']),
    completed_experiments=status['completed_experiments'],
    routed=sum(w['routed'] for w in workers),qwen=sum(max(w['qwen'],w['routed']) for w in workers),
    zeros_in_completed_blocks=sum(w['zeros'] for w in workers),workers=workers),ensure_ascii=False))
