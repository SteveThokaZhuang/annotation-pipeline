from pathlib import Path
from collections import defaultdict
import tempfile
from audio_pipeline.common import BASE,load_module,tokens,detokenize
from audio_pipeline.failure_policy import hypothesis_error

class Adapter:
    def __init__(self,job):
        self.cfg=job['settings']
        self.module=load_module(BASE/'models/roverfusion/fuse.py','existing_nist_rover')

    def process(self,item):
        hypotheses=item['hypotheses']
        available=[h for h in hypotheses if h is not None and not hypothesis_error(h)]
        if not available:
            return dict(text='',tokens=[],chunk=item['chunk'],hypotheses=hypotheses,
                routing=item['routing'],systems_used=0,expected_systems=3 if item['routing']=='high' else 2,
                missing_systems=len(hypotheses),transcript_status='no_transcript',
                error='No successful ASR hypothesis is available',confidence_available=False)
        sequences=[tokens(h['text'],self.cfg['unit']) for h in available]
        if len(sequences)==1:
            fused=sequences[0]
        else:
            prepared=[{'chunk':[(t,1.) for t in sequence]} for sequence in sequences]
            with tempfile.TemporaryDirectory(prefix='audio-pipeline-rover-') as directory:
                fused=self.module.rover_fuse(prepared,BASE/'models/roverfusion/bin/rover',
                    self.cfg['alpha'],self.cfg['null_confidence'],Path(directory))['chunk']
        return dict(text=available[0]['text'] if len(available)==1 else detokenize(fused),tokens=fused,chunk=item['chunk'],hypotheses=hypotheses,
            routing=item['routing'],transcript_status='single_model' if len(available)==1 else 'fused',systems_used=len(available),expected_systems=3 if item['routing']=='high' else 2,
            missing_systems=len(hypotheses)-len(available),tie_policy='NIST ROVER input order: ASR1 then ASR2 then ASR3',
            normalization=self.cfg['unit'],confidence_available=False)

    def process_batch(self,items):
        """Fuse independent chunks per system-count group, retaining input order.

        Separate utterance IDs prevent cross-chunk alignments. Empty successful
        hypotheses remain votes; missing/failed systems are excluded just as in
        process(). The worker retries a failed batch one chunk at a time.
        """
        groups=defaultdict(list)
        results={}
        for index,item in enumerate(items):
            available=[h for h in item['hypotheses'] if h is not None and not hypothesis_error(h)]
            if len(available)<2:
                results[index]=self.process(item)
            else:
                groups[len(available)].append((index,item,available))
        for count,group in groups.items():
            prepared=[{str(index):[(token,1.) for token in tokens(available[column]['text'],self.cfg['unit'])]
                       for index,item,available in group} for column in range(count)]
            with tempfile.TemporaryDirectory(prefix='audio-pipeline-rover-batch-') as directory:
                fused=self.module.rover_fuse(prepared,BASE/'models/roverfusion/bin/rover',
                    self.cfg['alpha'],self.cfg['null_confidence'],Path(directory))
            for index,item,available in group:
                sequence=fused[str(index)]
                results[index]=dict(text=detokenize(sequence),tokens=sequence,chunk=item['chunk'],
                    hypotheses=item['hypotheses'],routing=item['routing'],transcript_status='fused',
                    systems_used=count,expected_systems=3 if item['routing']=='high' else 2,
                    missing_systems=len(item['hypotheses'])-count,
                    tie_policy='NIST ROVER input order: ASR1 then ASR2 then ASR3',
                    normalization=self.cfg['unit'],confidence_available=False)
        return [results[index] for index in range(len(items))]
