"""After all GPU experiments, export compressed predictions and refresh docs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
directory=Path('/tmp/fa_open_yap_results')
while True:
    try:complete=json.loads((directory/'status.json').read_text()).get('status')=='complete'
    except (FileNotFoundError,json.JSONDecodeError):complete=False
    if complete:break
    time.sleep(10)
started=time.perf_counter()
for script in ['archive_rover_predictions.py','verify_rover_outputs.py','report_rover_experiments.py']:
    subprocess.run([sys.executable,str(ROOT/'src'/script)],check=True)
(ROOT/'results/open_yap_1k_summary/finalization.json').write_text(json.dumps(dict(
    status='complete',seconds=time.perf_counter()-started,
    scope='post-benchmark compressed prediction export and final report'),indent=2)+'\n')
print('Final prediction export and report complete',flush=True)
