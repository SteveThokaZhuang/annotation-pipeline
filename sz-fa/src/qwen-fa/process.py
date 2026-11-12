"""Compatibility entry point for the requested qwen-fa directory."""
from pathlib import Path
import runpy
import sys

directory = Path(__file__).resolve().parents[1] / 'qwen3-fa'
sys.path.insert(0, str(directory))
if __name__ == '__main__':
    runpy.run_path(str(directory / 'process.py'), run_name='__main__')
