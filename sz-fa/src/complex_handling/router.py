"""Deterministic sentence routing, independent from model loading."""
from dataclasses import asdict, dataclass, field

from .config import LanguagePolicy
from .detectors import sentence_bounds


@dataclass
class RouteDecision:
    id: str
    start_index: int
    end_index: int
    action: str
    route_reason: str
    risk_tags: list = field(default_factory=list)
    events: list = field(default_factory=list)
    backend: object = None

    def to_dict(self):
        return asdict(self)


class RuleRouter:
    def __init__(self, config, language_policy=None):
        self.config = config
        self.language_policy = language_policy or LanguagePolicy(config.mfa_language)

    def route(self, row, detection):
        decisions = []
        spans = row.get('sentences') or [dict(start_index=0, end_index=len(row['words']), text=row['text'])]
        for span in spans:
            lo, hi = span['start_index'], span['end_index']
            a, b = sentence_bounds(row, span)
            selected = []
            for event in detection.events:
                details = event.details
                if 'index' in details:
                    relevant = lo <= details['index'] < hi
                elif 'start_index' in details:
                    relevant = details['start_index'] < hi and details['end_index'] > lo
                else:
                    margin = self.config.event_margin_seconds
                    relevant = event.start < b + margin and event.end > a - margin
                if relevant:
                    selected.append(event.to_dict())
            tags = sorted({e['tag'] for e in selected})
            action, reason, backend = 'keep', 'no_observed_risk', None
            rejection = self.language_policy.rejection(row, 'nyra')
            isolated = detection.observations.get('target_isolated') is True
            if detection.errors:
                action, reason = 'review', 'detector_error'
            elif row.get('error') or not row['words']:
                action, reason = 'review', 'alignment_unavailable'
            elif 'transcript_uncertain' in tags:
                action, reason = 'review', 'transcript_uncertain'
            elif 'overlap' in tags and not isolated:
                # A target label alone does not separate its speech from a mixture.
                action, reason = 'review', 'overlap_requires_target_audio'
            elif tags:
                if rejection:
                    action, reason = 'review', rejection
                elif self.config.candidate_backend == 'none':
                    action, reason = 'review', 'candidate_backend_disabled'
                elif all(w.get('source') == 'nyra' for w in row['words'][lo:hi]):
                    reason = 'already_aligned_by_nyra'
                else:
                    action, reason, backend = 'shadow', 'risk_sentence_candidate', 'nyra'
            decisions.append(RouteDecision(row['id'], lo, hi, action, reason, tags, selected, backend))
        return decisions
