# Step 3 release — original full-duplex pipeline

This release connects existing Step 2 word alignment to directed interruption/backchannel events and same-speaker gaps greater than 0.5 seconds. The original Qwen + JEV + rule pipeline is the released annotation implementation. The separately preserved text-SFT experiment is not substituted for it.

## Preserved versions

| Version | Implementation | Evidence | Limit |
|---|---|---|---|
| Original pipeline | Qwen-compatible text service, JEV-32B mixed-audio votes, local event rules | 70/76 reference agreement on 30 historical regression clips | Known-error regression, not population accuracy or new blind evaluation |
| Best SFT A (preserved separately) | JEV-4B language-model LoRA, separate gap/overlap adapters | 404/500 on a different locked automatic-reference set | Ordinary-entry rejection failed 0/34; DPO was not selected |

Different evaluation sets prevent a direct ranking of these percentages. No SFT weights, raw recordings, transcripts, credentials, server connection files, or experimental training scripts are included in this release.

## Event and gap behavior

- Preserve original conversation/word IDs and absolute timestamps.
- Two opposing interruptions remain two directed events; feedback nested in an interruption can remain a separate event.
- An overlap alone does not establish an interruption; ordinary answers and continuations can be excluded.
- Gap labels remain `PAUSE` and `TURN_TRANSITION`. An independent new contribution can be initiated by the same speaker without changing the topic.
- Speaker attribution problems are quality failures, not evidence of an additional participant.
- Existing FA is read directly. The bundled upstream FA fix selects the intended physical audio channel before slicing.

## Risk sidecar

Each output event/gap receives `quality.risk_flag`, `risk_reasons`, and a rule version. The original semantic label is unchanged. `risk_subsets/*.json` preserves flagged annotations, possible dropped-event candidates, and technical failures separately.

Possible omissions require a candidate in the current core window. Older records without candidate intervals carry `candidate_scope_not_recorded`; they cannot establish a target omission by themselves. A risk marker is not a confirmed error or a calibrated probability.

The experimental SFT error detector is evaluated separately: a fixed detector fitted with conversation-group cross-validation on 250 validation targets detected 81/96 errors in the previously evaluated 500-target set, compared with 3/96 previously. It flagged 262 targets, including 181 correct ones. It requires an auxiliary adapter, is not calibrated for this released original pipeline, and is therefore not enabled here.

## Reproduce and inspect

See [README.md](README.md) for Step 2/V3 and published Seamless input commands. Install the requirements in an appropriate model environment; the Qwen-compatible service and model checkpoints are supplied separately.

```bash
python -m unittest discover -s step3 -p 'test_*.py' -v
```

Open [demo/index.html](demo/index.html) for the local annotation viewer. The repository viewer contains no recorded conversations. Import a local presentation bundle with `schema_version: 1`, `cases`, and optional `audio`. Case records contain `id`, `conversation_id`, `version` (`original` or `sft`), `task`, `label`, absolute `target` and `interval`, `context`, and optional risk metadata. Embedded audio accepts `data:audio/...` URLs only; otherwise choose a local clip matching the displayed interval.

The complete local HTML retains original recording IDs, model labels, risk flags, and timeline evidence. It is delivered separately to avoid publishing dataset audio/transcripts as application source.

## Remaining limitations

Historical 70/76 results reused overlap audio votes; they are not a new end-to-end speed benchmark. Candidate classification and full-recording event recall are distinct evaluations. Risk flags do not guarantee correctness of retained data. The default gap recheck records a proposal while preserving the original first-pass label; it does not silently replace labels. Model HTTP failures and FA anomalies are counted separately.
