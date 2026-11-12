"""Lazy adapter to the main Nyra class, sharing its resident model when present."""
from pathlib import Path
import sys
from types import SimpleNamespace


class NyraCandidateBackend:
    name = 'nyra'

    def __init__(self, args, aligner=None, engine=None):
        self.args, self.aligner, self.engine = args, aligner, engine

    def align(self, task):
        if self.engine is None:
            root = Path(__file__).resolve().parents[2]
            for path in (root/'qwen3-fa+mfa', root/'qwen3-fa'):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            from nyra_repair import NyraRepair
            project = root.parent
            defaults = dict(device=getattr(self.args, 'device', 'cpu'),
                nyra_checkpoint=project/'models/nyra-forced-aligner/checkpoints',
                nyra_runtime=project/'models/nyra', nyra_espeak=project/'models/nyra-espeak')
            defaults.update(vars(self.args))
            self.engine = NyraRepair(SimpleNamespace(**defaults), self.aligner)
        return self.engine.shadow_candidate(task)
