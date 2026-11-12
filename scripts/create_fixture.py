#!/usr/bin/env python3
"""Generate synthetic non-speech audio for metadata/I/O smoke runs."""
from pathlib import Path
import numpy as np
import soundfile as sf
root=Path(__file__).resolve().parents[1]/'tests/data'
root.mkdir(parents=True,exist_ok=True)
for index in (1,2):
    t=np.arange(16000*index,dtype=np.float32)/16000
    sf.write(root/f'synthetic-{index}.wav',.03*np.sin(2*np.pi*(200+index*50)*t),16000)
print(root)
