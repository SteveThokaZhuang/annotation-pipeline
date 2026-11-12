"""Qwen -> MFA -> Nyra whole-sentence fallback."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/qwen3-fa+mfa'))
sys.path.insert(0, str(ROOT/'src/qwen3-fa'))
from nyra_repair import run_nyra_mode

if __name__ == '__main__':
    run_nyra_mode('qwen+mfa+nyra')
