"""Independent asr2 entrypoint; --test selects isolated test mode."""
from pathlib import Path
import os
import sys
root=Path(__file__).resolve().parents[1]
args=sys.argv[1:]
test='--test' in args
args=[arg for arg in args if arg!='--test']
os.execv('/bin/bash',['bash',str(root/'run_pipeline.sh'),*(['test'] if test else []),'asr2',*args])
