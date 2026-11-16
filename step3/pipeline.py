"""Run the existing overlap and gap judges on V3 aligned recordings."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
from pathlib import Path
import time
from tempfile import TemporaryDirectory

from audio_judge import MixedAudioJudge, audio_duration, render_mix
from event_extractor import extract_candidates
from gap_judge import PROMPT, VERIFY, expected, grounded, request, risk, shorter
from input_v3 import load_conversation
from input_seamless import load_published
from run_multi import classify
from risk_annotations import tag_pipeline_result


def context(streams):
    spans = []
    for speaker, words in streams.items():
        current = None
        for word in words:
            if current is None or word["start"]-current["end"] > .5:
                current = dict(speaker=speaker, start=word["start"], end=word["end"], text=word["text"])
                spans.append(current)
            else:
                current["end"] = max(current["end"], word["end"])
                current["text"] += " " + word["text"]
    return sorted(spans, key=lambda span: (span["start"], span["speaker"]))


class WordTimeline:
    def __init__(self, streams):
        self.streams = {speaker: sorted(words, key=lambda w: (w["start"], w["end"]))
                        for speaker, words in streams.items()}
        self.starts = {speaker: [w["start"] for w in words] for speaker, words in self.streams.items()}
        self.max_ends = {}
        for speaker, words in self.streams.items():
            ends = []
            previous = None
            unit_start = None
            for word in words:
                ends.append(max(ends[-1] if ends else 0, word["end"]))
                if (previous is None or word["start"]-previous["end"] > .35 or
                        previous["text"].rstrip().endswith((".", "?", "!", "。", "？", "！"))):
                    unit_start = word["id"]
                word["lexical_unit_start"] = unit_start
                previous = word
            self.max_ends[speaker] = ends

    def between(self, lo, hi):
        selected = {}
        for speaker, words in self.streams.items():
            starts = self.starts[speaker]
            index = bisect_right(self.max_ends[speaker], lo)
            selected[speaker] = [w for w in words[index:bisect_left(starts, hi)] if w["end"] > lo]
        return selected


def windows(conversation, timeline, duration, width=8, padding=4, core=None, analysis=None):
    cores = [core] if core else [(lo, min(lo+width, duration)) for lo in range(0, int(duration)+1, width)
                                if lo < duration]
    for lo, hi in cores:
        interval = analysis or [max(0, lo-padding), min(duration, hi+padding)]
        streams = timeline.between(*interval)
        if not all(streams.values()):
            continue
        key = f"{conversation}:{lo}-{hi}"
        case = dict(id=key, number=key, conversation_id=conversation, interval=interval,
                    target=[lo, hi], original_label=None, streams=streams, context=context(streams))
        onsets = [
            c for c in extract_candidates(case)
            if c["other_active_run_at_onset"] and c["overlap_seconds"] > 0
        ]
        if any(lo <= c["start"] < hi for c in onsets):
            yield case, onsets


def gap_candidates(conversation, timeline, core=None, interval=None):
    for speaker, own in timeline.streams.items():
        other = next(p for p in timeline.streams if p != speaker)
        for index, (a, b) in enumerate(zip(own, own[1:])):
            lo, hi = a["end"], b["start"]
            if round(hi-lo, 6) <= .5:
                continue
            if core and not core[0] <= (lo+hi)/2 < core[1]:
                continue
            if interval and not interval[0] <= lo < hi <= interval[1]:
                continue
            # Bound model input while retaining both ends of a long gap.
            left = timeline.between(max(0, lo-12), min(hi, lo+12))
            right = timeline.between(max(lo, hi-12), hi+12)
            near = {
                token: sorted({w["id"]: w for w in left[token]+right[token]}.values(),
                              key=lambda w: w["start"])
                for token in timeline.streams
            }
            spans = context(near)
            uid = f"{conversation}:{a['id']}:{b['id']}"
            right_unit = [b]
            for following in own[index+2:]:
                if (following['start']-right_unit[-1]['end'] > .5 or
                        right_unit[-1]['text'].rstrip().endswith(('.', '?', '!', '。', '？', '！'))):
                    break
                right_unit.append(following)
            yield dict(
                id=uid, focus=speaker, other=other,
                target=dict(left_word=a["text"], right_word=b["text"], left_word_id=a["id"],
                            right_word_id=b["id"], absolute_gap=[lo, hi], gap_seconds=round(hi-lo, 6)),
                focus_before_target=" ".join(w["text"] for w in own[max(0, index-20):index+1]
                                             if w["start"] >= lo-12 or w is a),
                focus_after_target=" ".join(w["text"] for w in own[index+1:index+22]
                                            if w["start"] <= hi+12),
                target_right_unit=dict(start=b['start'],end=max(w['end'] for w in right_unit),
                                       text=' '.join(w['text'] for w in right_unit)),
                other_before_right=[s for s in context({other: [w for w in near[other] if w['start']<hi]})
                                    if s['end']>=lo-3],
                context=spans,
                other_near_target=[s for s in spans if s["speaker"] == other and
                                   s["end"] >= lo-3 and s["start"] <= hi+3],
                long_gap_middle_omitted=hi-lo > 24)


def gap_first(payload, call):
    first = call(payload, PROMPT)
    return dict(id=payload["id"], speaker=payload["focus"], target=payload["target"], first=first,
                label=first.get("label"), risk=risk(payload, first), reviewed=False)


def gap_review(row, payload, call):
    reduced = shorter(payload)
    short = call(reduced, PROMPT)
    verify = call(payload, VERIFY)
    label = verify.get("label")
    proposal = label if (label and short.get("label") == label and
                         expected(short.get("relation")) == expected(verify.get("relation")) == label and
                         grounded(reduced, short) and grounded(payload, verify)) else None
    complete = short.get("status") == verify.get("status") == "ok"
    row.update(reviewed=True, review_complete=complete,
               review=dict(short=short, verify=verify, proposal=proposal))
    # Existing full-run policy keeps the first decision and logs the review separately.


def annotate(row, streams, tracks, duration, judge, args, pool):
    clock = time.monotonic
    start = clock()
    conversation = row["conversation_id"]
    timeline = WordTimeline(streams)
    core, analysis = row.get("core"), row.get("interval")
    gaps = list(gap_candidates(conversation, timeline, core, analysis))
    calls = dict(endpoint=args.qwen_endpoint, model=args.qwen_model)

    def call(payload, prompt):
        return request(payload, prompt, **calls)

    events, windows_out, failures = [], [], []
    overlap_candidates = 0
    audio_requests = 0
    with TemporaryDirectory(dir=args.output.resolve(), prefix="mix-") as temp:
        for case, candidates in windows(conversation, timeline, duration, core=core, analysis=analysis):
            votes = dict(clip_id=case["id"], events=[])
            lo, hi = case["target"]
            overlap_candidates += sum(lo <= c["start"] < hi for c in candidates)
            try:
                mix = render_mix(tracks, case["interval"], Path(temp)/"mix.wav")
                for c in candidates:
                    audio_requests += 1
                    answer = judge.vote(case, c, mix, row.get("language", args.language))
                    votes["events"].append(dict(candidate_id=c["candidate_id"], mix=answer))
                result = classify(case, audio=votes, concise=True, refined=True, floor_checks=True,
                                  recover_invalid_merge=True, **calls)
                if result["status"] != "ok":
                    failures.append(dict(stage="overlap_qwen", id=case["id"]))
                else:
                    events.extend(e for e in result["events"] if lo <= e["start"] < hi)
                attempts = [{k: v for k, v in attempt.items() if k != "response"}
                            for attempt in result["attempts"]]
                windows_out.append(dict(id=case["id"], interval=case["interval"], target=case["target"],
                                        candidate_intervals={c['candidate_id']:[c['start'],c['end']] for c in candidates},
                                        status=result["status"], audio_votes=votes["events"], attempts=attempts,
                                        decisions=result.get("judgment", {}).get("decisions", [])))
            except Exception as exc:
                failures.append(dict(stage="overlap_audio", id=case["id"], reason=str(exc)))
    labels = list(pool.map(lambda payload: gap_first(payload, call), gaps))
    first_wall = clock()-start
    # The prior OB141 run used a 20% review budget. Here the unit is one recording.
    review_count = math.ceil(len(labels)*args.review_fraction)
    ranked = sorted((i for i, g in enumerate(labels) if g["risk"] and g["label"]),
                    key=lambda i: (-sum(labels[i]["risk"].values()), labels[i]["id"]))
    list(pool.map(lambda i: gap_review(labels[i], gaps[i], call), ranked[:review_count]))
    failures.extend(dict(stage="gap_review", id=g["id"], reason="review_request_failed")
                    for g in labels if g["reviewed"] and not g["review_complete"])
    failures.extend(dict(stage="gap_qwen", id=g["id"], reason=g["first"].get("errors"))
                    for g in labels if g["label"] is None)
    measured_duration = analysis[1]-analysis[0] if analysis else duration
    return tag_pipeline_result(dict(conversation_id=conversation, duration_seconds=measured_duration,
                source_duration_seconds=duration, source_interval=analysis or [0, duration],
                status="partial" if failures else "complete", windows=windows_out,
                events=sorted(events, key=lambda e: e["start"]), gaps=labels, failures=failures,
                overlap_candidates=overlap_candidates, audio_requests=audio_requests,
                first_wall_seconds=first_wall, with_review_wall_seconds=clock()-start))


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def summarize(results, total, wall):
    completed = sum(r["status"] == "complete" for r in results)
    events = Counter(e["label"] for r in results for e in r.get("events", []))
    labels = Counter(g["label"] for r in results for g in r.get("gaps", []) if g["label"])
    failure_types = Counter(e["stage"] for r in results for e in r.get("failures", []))
    gaps = sum(len(r.get("gaps", [])) for r in results)
    reviewed = sum(g["reviewed"] for r in results for g in r.get("gaps", []))
    review_completed = sum(g.get("review_complete", g["reviewed"]) for r in results for g in r.get("gaps", []))
    first_wall = sum(r.get("first_wall_seconds", 0) for r in results)
    valid = sum(events.values())+sum(labels.values())
    def calls(decision):
        return decision.get("attempts", len(decision.get("errors", [])))

    audio_requests = sum(r.get("audio_requests", 0) for r in results)
    overlap_requests = sum(sum(a.get("request_sent", False) for a in w["attempts"])
                           for r in results for w in r.get("windows", []))
    gap_requests = sum(calls(g["first"]) for r in results for g in r.get("gaps", []))
    review_requests = sum(calls(g["review"][k]) for r in results for g in r.get("gaps", [])
                          if g["reviewed"] for k in ("short", "verify"))
    source_seconds = sum(r.get("duration_seconds", 0) for r in results if r["status"] == "complete")
    return dict(recordings=total, completed=completed, completion=completed/total if total else None,
                risk_flagged_items=sum(r.get('risk_summary',{}).get('flagged_items',0) for r in results),
                possible_event_omissions=sum(r.get('risk_summary',{}).get('possible_omissions',0) for r in results),
                completed_audio_seconds=source_seconds,
                valid_labels=valid, event_counts=events, gap_counts=labels, gap_candidates=gaps,
                overlap_candidates=sum(r.get("overlap_candidates", 0) for r in results),
                overlap_windows=sum(len(r.get("windows", [])) for r in results),
                model_requests=dict(jev=audio_requests, overlap_qwen=overlap_requests,
                                    gap_first=gap_requests, gap_review=review_requests),
                reviewed_gaps=reviewed, completed_reviews=review_completed,
                review_completion=review_completed/reviewed if reviewed else None,
                review_rate=reviewed/gaps if gaps else None,
                failures_by_stage=failure_types, first_wall_seconds=first_wall, total_wall_seconds=wall,
                first_labels_per_minute=60*valid/first_wall if first_wall else None,
                valid_labels_per_minute=60*valid/wall if wall else None,
                rtf=wall/source_seconds if source_seconds else None,
                seconds_per_completed_recording=wall/completed if completed else None,
                recordings_per_minute=60*completed/wall if wall else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--align-recordings", type=Path)
    source.add_argument("--seamless-root", type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jev-checkpoint", type=Path)
    parser.add_argument("--jev-device", default="cuda:0")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:18770/v1/chat/completions")
    parser.add_argument("--qwen-model", default="qwen38-27b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--review-fraction", type=float, default=.2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.review_fraction <= 1:
        parser.error("Invalid worker count or review fraction")
    if not args.prepare_only and not args.jev_checkpoint:
        parser.error("Provide --jev-checkpoint for model inference")
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [r["conversation_id"] for r in pairs]
    if not ids or any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        parser.error("Provide a nonempty manifest with unique string conversation IDs")
    if any(r.get("language", args.language) not in ("en", "zh") for r in pairs):
        parser.error("Manifest language must be en or zh")
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output/"recordings"
    if not args.prepare_only:
        try:
            folder.mkdir()
        except FileExistsError:
            parser.error("Use a new --output directory for each model run")
    write_json(args.output/"provenance.json", dict(
        arguments={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
        manifest_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        code_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(Path(__file__).parent.iterdir()) if path.suffix in {'.py','.txt'}}))
    model_started = time.monotonic()
    judge = None if args.prepare_only else MixedAudioJudge(args.jev_checkpoint, args.jev_device)
    model_load_seconds = time.monotonic()-model_started
    reports = []
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        inputs = load_published(args.seamless_root, pairs) if args.seamless_root else (
            (row, None, None, None) for row in pairs
        )
        for index, (row, streams, tracks, problem) in enumerate(inputs, 1):
            try:
                if problem:
                    raise ValueError(problem)
                if args.align_recordings:
                    streams, tracks = load_conversation(args.align_recordings, row, args.pairs.resolve().parent)
                duration = audio_duration(tracks)
                if args.seamless_root:
                    for words in streams.values():
                        outside = [w["id"] for w in words if w["end"] > duration+.001]
                        row["invalid_fa_words"] += len(outside)
                        if outside:
                            excluded = set(outside)
                            words[:] = [w for w in words if w["id"] not in excluded]
                elif any(words and words[-1]["end"] > duration+.001 for words in streams.values()):
                    raise ValueError("alignment_outside_source_duration")
                if row.get("interval"):
                    a, b = row["interval"]
                    lo, hi = row.get("core", row["interval"])
                    if not 0 <= a <= lo < hi <= b <= duration+.0001:
                        raise ValueError("test_window_outside_shared_timeline")
                if args.prepare_only:
                    timeline = WordTimeline(streams)
                    core, analysis = row.get("core"), row.get("interval")
                    prepared_windows = list(windows(row["conversation_id"], timeline, duration,
                                                    core=core, analysis=analysis))
                    result = dict(conversation_id=row["conversation_id"], status="ready",
                                  duration_seconds=duration, words={k: len(v) for k, v in streams.items()},
                                  overlap_windows=len(prepared_windows),
                                  invalid_fa_words=row.get("invalid_fa_words", 0),
                                  gap_candidates=sum(1 for _ in gap_candidates(row["conversation_id"], timeline,
                                                                              core, analysis)))
                else:
                    result = annotate(row, streams, tracks, duration, judge, args, pool)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                result = dict(conversation_id=row["conversation_id"], status="failed",
                              failures=[dict(stage="input_or_runtime", reason=str(exc))])
            if args.prepare_only:
                reports.append(result)
            else:
                write_json(folder/f"{index:08d}.json", result)
                risk_folder=args.output/'risk_subsets'
                risk_folder.mkdir(exist_ok=True)
                write_json(risk_folder/f"{index:08d}.json",dict(conversation_id=result['conversation_id'],items=result.get('risk_items',[]),technical_failures=result.get('failures',[])))
                reports.append({
                    **result,
                    "events": [{"label": event["label"]} for event in result.get("events", [])],
                    "windows": [{"audio_votes": window["audio_votes"], "attempts": window["attempts"]}
                                for window in result.get("windows", [])],
                    "gaps": [{"label": gap["label"], "first": gap["first"], "reviewed": gap["reviewed"],
                              **({"review": gap["review"], "review_complete": gap["review_complete"]}
                                 if gap["reviewed"] else {})}
                             for gap in result.get("gaps", [])],
                })
                progress = summarize(reports, len(pairs), time.monotonic()-began)
                progress.update(processed_recordings=len(reports),run_state="running",
                                model_load_seconds=model_load_seconds)
                write_json(args.output/"summary.json", progress)
            print(row["conversation_id"], result["status"], flush=True)
    if args.prepare_only:
        write_json(args.output/"prepare_report.json", reports)
    else:
        summary = summarize(reports, len(pairs), time.monotonic()-began)
        summary["model_load_seconds"] = model_load_seconds
        summary.update(processed_recordings=len(reports),run_state="finished")
        write_json(args.output/"summary.json", summary)
    if any(r["status"] in {"failed", "partial"} for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
