"""Directional WER/CER between primary ASRs for selective fallback routing."""
import math
import re
from audio_pipeline.common import distance,tokens


class Adapter:
    def __init__(self,job):
        self.cfg=job['settings']
        self.threshold=float(self.cfg.get('threshold',0.1))
        if not math.isfinite(self.threshold) or self.threshold<0:
            raise ValueError('Disagreement threshold must be finite and nonnegative')
        self.reference=self.cfg.get('reference','asr1')
        if self.reference not in ['asr1','asr2']:
            raise ValueError('Disagreement reference must be asr1 or asr2')
        self.metric=self.cfg.get('metric','auto').lower()
        if self.metric not in ['auto','wer','cer']:
            raise ValueError('Disagreement metric must be auto, wer or cer')
        self.language=self.cfg.get('language','auto').lower().replace('_','-')

    def metric_for(self,text):
        if self.metric!='auto':return self.metric
        if self.language=='chinese' or self.language.split('-')[0]=='zh':return 'cer'
        if self.language=='english' or self.language.split('-')[0]=='en':return 'wer'
        if self.language=='auto':
            return 'cer' if re.search(r'[\u3400-\u9fff]',text) else 'wer'
        raise ValueError('Disagreement needs English/Chinese language or an explicit wer/cer metric')

    def process(self,item):
        a,b=item['hypotheses']
        reference,hypothesis=(a,b) if self.reference=='asr1' else (b,a)
        missing=reference is None or hypothesis is None
        metric=self.metric_for(' '.join(v['text'] for v in [reference,hypothesis] if v is not None))
        unit='word' if metric=='wer' else 'char'
        ref=tokens(reference['text'],unit) if reference is not None else []
        hyp=tokens(hypothesis['text'],unit) if hypothesis is not None else []
        edits=None if missing else 0 if ref==hyp else distance(ref,hyp)
        # Empty-reference convention: each inserted word/character contributes 1.
        # Missing/empty hypotheses always retain the existing fallback behavior.
        score=1.0 if missing else edits/max(1,len(ref))
        empty=not missing and (not ref or not hyp)
        high=missing or empty or score>self.threshold
        return dict(chunk=item['chunk'],hypotheses=[a,b],score=score,high=high,
            threshold=self.threshold,unit=unit,missing_hypothesis=missing,
            reason='missing' if missing else 'empty' if empty else 'disagreement' if high else 'agreement',
            metric=metric,reference=self.reference,hypothesis='asr2' if self.reference=='asr1' else 'asr1',
            edit_distance=edits,reference_units=len(ref),hypothesis_units=len(hyp))
