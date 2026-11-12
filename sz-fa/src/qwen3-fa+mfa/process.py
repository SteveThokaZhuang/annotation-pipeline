"""Run Qwen, then CPU MFA chunk repair with midpoint sentence fallback."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/qwen3-fa'))
from process import parser, run
from repair import SentenceRepair


def main():
    p = parser()
    p.add_argument('--mfa-python', type=Path, default=ROOT/'models/mfa/bin/python')
    p.add_argument('--dictionary', type=Path, default=ROOT/'models/mfa-english/dictionary/english_us_arpa.dict')
    p.add_argument('--acoustic-model', type=Path, default=ROOT/'models/mfa-english/english_us_arpa.zip')
    p.add_argument('--mfa-jobs', type=int, default=8)
    p.add_argument('--mfa-timeout', type=int, default=1800)
    p.add_argument('--padding', type=float, default=0.35)
    p.add_argument('--mfa-seam-policy', choices=['midpoint', 'clamp', 'bounded'], default='midpoint')
    p.add_argument('--mfa-repair-scope', choices=['sentence', 'chunk'], default='chunk')
    p.add_argument('--no-mfa-sentence-fallback', action='store_false', dest='mfa_sentence_fallback', default=True)
    args = p.parse_args()
    for path in [args.mfa_python, args.dictionary, args.acoustic_model]:
        if not path.exists() and args.complex_handling == 'off':
            p.error('Missing MFA dependency: '+str(path))
    if args.padding < 0 or args.mfa_jobs < 1:
        p.error('padding must be nonnegative; mfa-jobs must be positive')
    run(args, repair_factory=SentenceRepair)


if __name__ == '__main__':
    main()
