import unittest
from risk_annotations import annotate, tag_pipeline_result


class RiskTests(unittest.TestCase):
    def test_disagreement_does_not_erase_binary_label(self):
        r=annotate('gap','PAUSE','TURN_TRANSITION',['PAUSE','PAUSE'],'TURN_TRANSITION',rejudged=True)
        self.assertEqual(r['label'],'TURN_TRANSITION')
        self.assertTrue(r['risk_unresolved'])
        self.assertFalse(r['preference_eligible'])

    def test_corrected_seed_is_not_automatically_backfilled(self):
        r=annotate('gap','PAUSE','TURN_TRANSITION',['TURN_TRANSITION']*2,'TURN_TRANSITION',rejudged=True)
        self.assertTrue(r['risk_flag'])
        self.assertFalse(r['risk_unresolved'])
        self.assertTrue(r['preference_eligible'])

    def test_alignment_error_overrides_model_consensus(self):
        r=annotate('overlap','INTERRUPT','INTERRUPT',['INTERRUPT']*2,'INTERRUPT',['target_alignment_mismatch'])
        self.assertFalse(r['preference_eligible'])

    def test_missing_audio_is_not_consensus(self):
        r=annotate('gap','PAUSE','PAUSE',['PAUSE']*2,None)
        self.assertIn('audio_judge_failure',r['risk_reasons'])
        self.assertTrue(r['risk_unresolved'])

    def test_reject_other_gap_label(self):
        with self.assertRaises(ValueError): annotate('gap','PAUSE','NEITHER',['PAUSE']*2,'PAUSE')

    def test_possible_omission_is_preserved_without_inventing_an_event(self):
        r=tag_pipeline_result(dict(events=[],gaps=[],windows=[dict(audio_votes=[dict(candidate_id='c',mix=dict(answer='INTERRUPT'))],decisions=[dict(candidate_id='c',action='DROP')])]))
        self.assertEqual(r['events'],[])
        self.assertEqual(r['risk_summary']['possible_omissions'],1)

    def test_risk_does_not_remove_real_annotation(self):
        r=tag_pipeline_result(dict(events=[dict(event_id='e',label='INTERRUPT',start=1,end=2,candidate_ids=['c'])],gaps=[],windows=[dict(audio_votes=[dict(candidate_id='c',mix=dict(answer='BACKCHANNEL',probs={'BACKCHANNEL':.6,'INTERRUPT':.4}))])]))
        self.assertEqual(r['events'][0]['label'],'INTERRUPT')
        self.assertTrue(r['events'][0]['quality']['risk_flag'])

    def test_context_only_candidate_is_not_counted_as_target_omission(self):
        r=tag_pipeline_result(dict(events=[],gaps=[],windows=[dict(target=[2,4],candidate_intervals={'c':[1,1.5]},audio_votes=[dict(candidate_id='c',mix=dict(answer='INTERRUPT'))],decisions=[dict(candidate_id='c',action='DROP')])]))
        self.assertEqual(r['risk_summary']['possible_omissions'],0)


if __name__=='__main__': unittest.main()
