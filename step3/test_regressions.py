"""Focused regressions and randomized timestamp checks for Step 3."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pipeline
import run_multi
import gap_judge
from audio_judge import render_mix, audio_duration
from event_extractor import extract_candidates, finalize
from input_seamless import load_published
from input_v3 import load_channel


def word(identity, start, end, text="word"):
    return dict(id=identity, start=start, end=end, text=text)


class RegressionTests(unittest.TestCase):
    def test_nested_word_is_retained(self):
        words = [word("a", 0, 10), word("b", 1, 2)]
        self.assertEqual(pipeline.WordTimeline({"A": words}).between(3, 4)["A"], [words[0]])

    def test_random_windows_match_exhaustive_intersections(self):
        rng = random.Random(20261115)
        for _ in range(1000):
            words = [word(str(i), a, a+rng.uniform(.01, 15))
                     for i, a in enumerate(sorted(rng.uniform(0, 100) for _ in range(30)))]
            lo, hi = sorted(rng.uniform(0, 115) for _ in range(2))
            found = pipeline.WordTimeline({"A": words}).between(lo, hi)["A"]
            self.assertEqual([w["id"] for w in found],
                             [w["id"] for w in words if w["end"] > lo and w["start"] < hi])

    def test_context_end_never_moves_backwards(self):
        self.assertEqual(pipeline.context({"A": [word("a", 0, 10), word("b", 1, 2)]})[0]["end"], 10)

    def test_gap_threshold_and_identity(self):
        streams = {"A": [word("a", 0, 1), word("b", 1.5, 2), word("c", 2.50001, 3)], "B": []}
        gaps = list(pipeline.gap_candidates("original", pipeline.WordTimeline(streams)))
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["target"]["left_word_id"], "b")
        self.assertEqual(gaps[0]["target"]["absolute_gap"], [2, 2.50001])

    def test_right_boundary_does_not_skip_to_a_later_action(self):
        streams={'A':[word('a',0,1,'a place,'),word('b',1.7,2,'and'),
                      word('c',2,2.4,"I..."),word('d',5,6,'postpone it.')],
                 'B':[word('other',3,4,'No.') ]}
        gap=next(pipeline.gap_candidates('original',pipeline.WordTimeline(streams)))
        self.assertEqual(gap['target_right_unit']['text'],"and I...")
        self.assertEqual(gap['other_before_right'],[])

    def test_tail_answer_filter_preserves_substantive_early_entry(self):
        from event_extractor import suppress_terminal_answers
        case=dict(conversation_id='c',interval=[0,4],streams={
            'A':[word('a',0,2,'Is that right?')],
            'B':[word('b',1.74,2.04,'Yes.')]})
        candidates=extract_candidates(case,refined=True,floor_checks=True)
        target=next(c for c in candidates if c['initiator']=='B')
        self.assertTrue(target['terminal_question_answer'])
        decisions=[dict(candidate_id=c['candidate_id'],action='KEEP',label='BACKCHANNEL',merge_into='',reason='test') for c in candidates]
        suppress_terminal_answers(candidates,dict(decisions=decisions))
        self.assertEqual(decisions[1]['action'],'DROP')
        case['streams']['B']=[word('b',1,3,'Yes, let me explain.')]
        target=next(c for c in extract_candidates(case,refined=True,floor_checks=True) if c['initiator']=='B')
        self.assertFalse(target['terminal_question_answer'])

    def test_continuation_filter_requires_all_three_evidence_types(self):
        from event_extractor import suppress_supported_floor_continuation
        import copy
        candidate=dict(candidate_id='current',previous_own_candidate='previous',
                       previous_own_clause=dict(text='according to your',end=2),start=2.7,
                       recipient_clause_at_onset=dict(text='What if I,'),alignment_flags=[],
                       audio_neither_probability=.98)
        decisions=[dict(candidate_id='previous',action='DROP',label='BACKCHANNEL'),
                   dict(candidate_id='current',action='KEEP',label='INTERRUPT',merge_into='',reason='test')]
        result=copy.deepcopy(decisions)
        suppress_supported_floor_continuation([candidate],dict(decisions=result))
        self.assertEqual(result[1]['action'],'DROP')
        for change in [dict(audio_neither_probability=.5),dict(previous_own_clause=dict(text='I finished.',end=2)),
                       dict(recipient_clause_at_onset=dict(text='I have a different question for you.'))]:
            result=copy.deepcopy(decisions)
            suppress_supported_floor_continuation([dict(candidate,**change)],dict(decisions=result))
            self.assertEqual(result[1]['action'],'KEEP')

    def test_cropped_continuation_does_not_become_a_new_onset(self):
        streams = {"A": [word(str(i), i, i+.8) for i in range(12)],
                   "B": [word("b", 2, 10)]}
        timeline = pipeline.WordTimeline(streams)
        case = dict(conversation_id="original", interval=[4, 8], streams=timeline.between(4, 8))
        self.assertEqual(extract_candidates(case), [])

    def test_chinese_sentence_boundary_is_a_candidate(self):
        words = [word("a", 1, 2, "好。"), word("b", 2.1, 3, "等等")]
        case = dict(conversation_id="original", interval=[0, 4], streams={"A": words, "B": []})
        self.assertEqual(len(extract_candidates(case)), 2)

    def test_candidate_end_includes_nested_long_word(self):
        case = dict(conversation_id="c", interval=[0, 12], streams={
            "A": [word("a", 0, 10), word("b", 1, 2)], "B": [word("c", .5, 11)]})
        candidate = next(c for c in extract_candidates(case) if c['initiator'] == 'A')
        self.assertEqual(candidate['end'], 10)

    def test_reverse_interrupt_and_nested_feedback_are_separate(self):
        case = dict(id="clip", conversation_id="c", interval=[0, 12], target=[0,12], streams={
            "A": [word("a0", 0, 3), word("a1", 4, 7), word("a2", 8, 9)],
            "B": [word("b0", 2, 11)]})
        candidates = extract_candidates(case)
        decisions = [dict(candidate_id=c['candidate_id'], action='KEEP', merge_into='',
                          label='BACKCHANNEL' if c['start']==8 else 'INTERRUPT', reason='test')
                     for c in candidates]
        result = finalize(case,candidates,dict(decisions=decisions),dict(status='PASS'))
        self.assertEqual([(e['initiator'],e['label'],e['start']) for e in result['events']],
                         [('B','INTERRUPT',2),('A','INTERRUPT',4),('A','BACKCHANNEL',8)])

    def test_failed_review_does_not_count_as_completed(self):
        gap = dict(label='PAUSE', first=dict(attempts=1), reviewed=True, review_complete=False,
                   review=dict(short=dict(attempts=2), verify=dict(attempts=2)))
        record = dict(status='partial', gaps=[gap], failures=[dict(stage='gap_review')])
        result = pipeline.summarize([record],1,1)
        self.assertEqual((result['reviewed_gaps'],result['completed_reviews'],result['completed']), (1,0,0))
        self.assertEqual(result['model_requests']['gap_review'],4)

    def test_cli_persists_progress_and_isolates_failed_input(self):
        streams={'A':[word('a',0,.4,'before'),word('b',1.2,1.6,'after')],
                 'B':[word('c',3,3.5,'reply')]}
        answer=dict(status='ok',label='PAUSE',relation='CONTINUATION',reason='same',
                    left_evidence='before',right_evidence='after',attempts=1)
        with TemporaryDirectory() as temp:
            root=Path(temp)
            pairs=root/'pairs.jsonl'
            pairs.write_text('{"conversation_id":"good"}\n{"conversation_id":"bad"}\n')
            args=['pipeline','--seamless-root',str(root),'--pairs',str(pairs),'--output',str(root/'out'),
                  '--jev-checkpoint',str(root/'unused')]
            inputs=[(dict(conversation_id='good',invalid_fa_words=0),streams,{},None),
                    (dict(conversation_id='bad'),None,None,'missing_alignment')]
            with patch.object(sys,'argv',args),patch.object(pipeline,'load_published',return_value=inputs),\
                 patch.object(pipeline,'audio_duration',return_value=4),patch.object(pipeline,'MixedAudioJudge'),\
                 patch.object(pipeline,'request',return_value=answer),patch('builtins.print'):
                with self.assertRaises(SystemExit) as caught:
                    pipeline.main()
            self.assertEqual(caught.exception.code,1)
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual((summary['processed_recordings'],summary['completed'],summary['run_state']),
                             (2,1,'finished'))
            self.assertEqual(len(list((root/'out/recordings').glob('*.json'))),2)
            self.assertTrue((root/'out/provenance.json').is_file())

    def test_malformed_model_fields_are_retried(self):
        bad = dict(label="PAUSE", relation="CONTINUATION", reason=[], left_evidence=None, right_evidence={})
        body = json.dumps(dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(bad)))]))
        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: io.StringIO(body)):
            response = gap_judge.request({}, "rules")
        self.assertEqual(response["status"], "error")
        self.assertEqual(response["attempts"], 2)

    def test_review_attempt_is_distinct_from_completion(self):
        row = dict(reviewed=False)
        payload = dict(target={"absolute_gap": [1, 2]}, context=[], other_near_target=[],
                       focus_before_target="before", focus_after_target="after")
        pipeline.gap_review(row, payload, lambda *a: dict(status="error"))
        self.assertTrue(row["reviewed"])
        self.assertFalse(row["review_complete"])

    def test_single_high_risk_gap_is_reviewed(self):
        streams = {"A": [word("a", 0, 1, "before"), word("b", 2, 3, "after")], "B": []}
        answer = dict(status="ok", label="PAUSE", relation="CONTINUATION", reason="same",
                      left_evidence="before", right_evidence="after", attempts=1)
        with TemporaryDirectory() as temp, ThreadPoolExecutor(1) as pool:
            args = SimpleNamespace(output=Path(temp), qwen_endpoint="test", qwen_model="test",
                                   language="en", review_fraction=.2)
            with patch.object(pipeline, "request", return_value=answer):
                result = pipeline.annotate(dict(conversation_id="c"), streams, {}, 4, None, args, pool)
        self.assertTrue(result["gaps"][0]["review_complete"])
        self.assertEqual(result["status"], "complete")

    def test_training_events_available_after_mixed_audio(self):
        case = dict(id="case", number="case", original_label=None, target=[0, 4], interval=[0, 4],
                    conversation_id="c", streams={"A": [word("a", 0, 3)], "B": [word("b", 1, 2)]},
                    context=[])
        candidates = extract_candidates(case)
        decisions = [dict(candidate_id=c["candidate_id"], action="KEEP", label="INTERRUPT",
                          merge_into="", reason="test") for c in candidates]
        body = json.dumps(dict(choices=[dict(finish_reason="stop",
                                           message=dict(content=json.dumps(dict(decisions=decisions))))]))
        with patch("urllib.request.urlopen", return_value=io.StringIO(body)):
            result = run_multi.classify(case, audio=dict(clip_id="case", events=[
                dict(candidate_id=c['candidate_id'],mix=dict(answer='INTERRUPT'))
                for c in candidates if c['other_active_run_at_onset']]))
        self.assertEqual(result["speaker_gate"]["status"], "PASS")
        self.assertEqual(len(result["training_events"]), 1)
        with self.assertRaisesRegex(ValueError,'mixed_audio_votes_missing_or_invalid'):
            run_multi.classify(case,audio=dict(clip_id='case',events=[]))

    def test_overlap_network_failures_have_time_and_one_entry_per_request(self):
        case = dict(id="case", number="case", original_label=None, target=[0,4], interval=[0,4],
                    conversation_id="c", streams={"A":[word("a",0,3)],"B":[word("b",1,2)]},context=[])
        with patch('urllib.request.urlopen',side_effect=OSError('connection reset')):
            result=run_multi.classify(case)
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(result['attempts']),3)
        self.assertTrue(all(a['request_sent'] and a['seconds']>=0 and 'error' in a for a in result['attempts']))

    def test_corrupt_published_record_is_isolated(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"training/index_v1").mkdir(parents=True)
            (root/"seamless_fd_data").mkdir()
            (root/"seamless_fd_data/metadata.jsonl").write_bytes(b"NOT JSON")
            row = dict(interaction_id="bad", metadata_byte_offset=0, metadata_bytes=8)
            (root/"training/index_v1/train.jsonl").write_text(json.dumps(row))
            result = list(load_published(root, [dict(conversation_id="bad"), dict(conversation_id="missing")]))
            self.assertEqual(len(result), 2)
            self.assertTrue(result[0][3])
            self.assertEqual(result[1][3], "not_in_published_fa_index")

    def test_v3_chunk_offset_and_punctuation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            record = dict(recording_channel="c_c0", coverage=dict(status="complete"),
                          chunks=[dict(chunk=dict(id="chunk", recording_id="c", channel=0,
                                                  start=100, end=105, path="audio.wav"),
                                       text="Hello?", words=[word("unused", .5, 1, "Hello")])])
            (root/"c_c0.json").write_text(json.dumps(record))
            words, _ = load_channel(root, dict(recording_id="c", channel=0), root)
            self.assertEqual((words[0]["start"], words[0]["text"]), (100.5, "Hello?"))

    def test_audio_nonfinite_and_channel_mix(self):
        import numpy as np
        import soundfile as sf
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"source.wav"
            sf.write(source, np.column_stack([np.full(1600, .125), np.full(1600, .375)]), 8000)
            tracks = {"A": (source, 0), "B": (source, 1)}
            render_mix(tracks, [0, .2], root/"mix.wav")
            mixed, rate = sf.read(root/"mix.wav")
            self.assertEqual((len(mixed), rate), (3200, 16000))
            self.assertAlmostEqual(float(np.mean(mixed)), .25, places=3)
            sf.write(source, np.full((1600, 2), np.nan), 8000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                render_mix(tracks, [0, .2], root/"mix.wav")

    def test_exact_duration_tolerance_is_accepted(self):
        infos = [SimpleNamespace(channels=1,duration=d) for d in (183.92,184.0)]
        with patch('soundfile.info',side_effect=infos):
            self.assertEqual(audio_duration({'A':('a.wav',0),'B':('b.wav',0)}),183.92)


if __name__ == "__main__":
    unittest.main()
