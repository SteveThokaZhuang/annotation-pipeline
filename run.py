"""Relocatable Python controller; run with the recorded controller environment."""
from pathlib import Path
import os
import sys
root=Path(__file__).resolve().parent
paths=[str(root/'pipeline'),str(root/'vendor'),str(root/'models/funasr/source'),str(root/'models/deepfilternet3/DeepFilterNet')]
sys.path[:0]=paths
os.environ['PYTHONPATH']=os.pathsep.join(paths+([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
from audio_pipeline.cli import main
if __name__=='__main__':main()
