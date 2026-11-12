"""Update the saved report whenever a full experiment finishes."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
directory=Path('/tmp/fa_open_yap_results')
previous=-1
while True:
    path=directory/'suite_summary.json'
    try:
        count=len(json.loads(path.read_text())['experiments'])
    except (FileNotFoundError,json.JSONDecodeError):count=0
    if count and count!=previous:
        subprocess.run([sys.executable,str(ROOT/'src/report_rover_experiments.py'),'--root',str(directory)],check=True)
        previous=count
        print('Saved report for completed experiments:',count,flush=True)
    try:complete=json.loads((directory/'status.json').read_text()).get('status')=='complete'
    except (FileNotFoundError,json.JSONDecodeError):complete=False
    if complete and count==3:break
    time.sleep(10)
