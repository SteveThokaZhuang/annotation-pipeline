#!/usr/bin/env python3
"""Fuse matching ASR segments with official NIST ROVER, optionally using Hystoc."""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/hystoc"))


def tokenize(text, unit="char", strip_punctuation=False, lowercase=False):
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_punctuation:
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return [c for c in text if not c.isspace()] if unit == "char" else text.split()


def load_jsonl(path):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("utt_id")
            if not isinstance(key, str) or not key or any(c.isspace() for c in key):
                raise ValueError(f"{path}:{number}: utt_id must be a nonempty string without whitespace")
            if key in records:
                raise ValueError(f"{path}:{number}: duplicate utt_id {key!r}")
            records[key] = row
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Token confidence must be finite and between 0 and 1")
    return value


def prepare(row, mode, tokens, temperature):
    if mode == "hystoc":
        from hystoc.hystoc_confidences import get_token_confidences
        nbest = row.get("nbest")
        if not isinstance(nbest, list) or not nbest:
            raise ValueError("Hystoc needs a nonempty nbest list with text and log_score")
        hypotheses, scores = {}, {}
        for index, hyp in enumerate(nbest):
            if not isinstance(hyp.get("text"), str):
                raise ValueError("Every N-best hypothesis needs string text")
            score = float(hyp["log_score"])
            if not math.isfinite(score):
                raise ValueError("log_score must be finite; larger means more likely")
            hypotheses[str(index)] = " ".join(tokens(hyp["text"]))
            scores[str(index)] = score
        # Hystoc's initial empty network otherwise loses the mass of leading empty hypotheses.
        boundary = "hystocboundary"
        vocabulary = {t for h in hypotheses.values() for t in h.split()}
        while boundary in vocabulary:
            boundary += "x"
        hypotheses = {key: boundary + " " + text + " " + boundary for key, text in hypotheses.items()}
        return [(t, probability(p)) for t, p in get_token_confidences(scores, hypotheses, temperature) if t != boundary]
    if not isinstance(row.get("text"), str):
        raise ValueError("Each input record needs string text")
    sequence = tokens(row["text"])
    if mode == "vote":
        return [(t, 1.0) for t in sequence]
    confidences = row.get("confidences")
    if not isinstance(confidences, list) or len(confidences) != len(sequence):
        raise ValueError("Confidence mode requires one confidence per normalized token")
    return list(zip(sequence, map(probability, confidences)))


def rover_fuse(prepared, binary, alpha, null_confidence, workdir):
    """Use reversible ASCII token IDs to support UTF-8 and literal SCTK metacharacters."""
    keys = sorted(prepared[0])
    vocabulary = sorted({token for system in prepared for seq in system.values() for token, _ in seq})
    encode = {t: f"t{i:08d}" for i, t in enumerate(vocabulary)}
    decode = {value: key for key, value in encode.items()}
    # Shared boundary tokens also represent empty hypotheses without dropping a system's vote.
    begin, end = "boundarybegin", "boundaryend"
    ctm_ids = {key: f"u{i:08d}" for i, key in enumerate(keys)}
    reverse_ids = {value: key for key, value in ctm_ids.items()}
    output = {key: [] for key in keys}
    command = [str(binary)]
    for index, system in enumerate(prepared):
        path = workdir / f"system_{index}.ctm"
        with path.open("w", encoding="ascii") as stream:
            for key in keys:
                sequence = [(begin, 1.0)] + [(encode[t], p) for t, p in system[key]] + [(end, 1.0)]
                for position, (token, confidence) in enumerate(sequence):
                    stream.write(f"{ctm_ids[key]} 1 {position * 0.1:.2f} 0.10 {token} {confidence:.12g}\n")
        command += ["-h", str(path), "ctm"]
    fused = workdir / "fused.ctm"
    command += ["-o", str(fused), "-m", "meth1", "-a", str(alpha), "-c", str(null_confidence), "-f", "0", "-s"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (workdir / "rover.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("ROVER failed: " + result.stdout + result.stderr)
    for line in fused.read_text().splitlines():
        fields = line.split()
        token = fields[4]
        if token not in (begin, end):
            output[reverse_ids[fields[0]]].append(decode[token])
    return output


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate(references, hypotheses):
    errors = sum(edit_distance(references[key], hypotheses[key]) for key in references)
    count = sum(map(len, references.values()))
    return {"errors": errors, "reference_tokens": count, "error_rate": errors / count if count else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="One JSONL per ASR system; identical utt_id sets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="JSONL records with utt_id and text")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mode", choices=("vote", "confidence", "hystoc"), default="vote")
    parser.add_argument("--unit", choices=("char", "word"), default="char")
    parser.add_argument("--strip-punctuation", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--alpha", type=float, help="ROVER frequency weight: default 1 for vote, 0.7 otherwise")
    parser.add_argument("--null-confidence", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rover", type=Path, default=ROOT / "bin/rover")
    parser.add_argument("--work-dir", type=Path, help="Retain encoded CTMs and ROVER log in this directory")
    args = parser.parse_args(argv)
    try:
        if len(args.inputs) < 2 or len(args.inputs) > 50:
            raise ValueError("Provide between 2 and 50 ASR systems")
        alpha = probability(args.alpha if args.alpha is not None else (1.0 if args.mode == "vote" else 0.7))
        null_confidence = probability(args.null_confidence)
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        if not args.rover.is_file():
            raise ValueError("ROVER executable missing; run python3 build_rover.py first")
        systems = [load_jsonl(path) for path in args.inputs]
        report_path = args.report or args.output.with_suffix(".metrics.json")
        destinations = [args.output.resolve(), report_path.resolve()]
        sources = {path.resolve() for path in args.inputs}
        if args.reference:
            sources.add(args.reference.resolve())
        if destinations[0] == destinations[1] or any(path in sources for path in destinations):
            raise ValueError("Output and report must be distinct and must not overwrite input files")
        keys = set(systems[0])
        for path, system in zip(args.inputs, systems):
            if set(system) != keys:
                raise ValueError(f"{path}: utt_id sets differ; all systems must transcribe the same segments")
        tokens = lambda text: tokenize(text, args.unit, args.strip_punctuation, args.lowercase)
        references = None
        if args.reference:
            reference_rows = load_jsonl(args.reference)
            if set(reference_rows) != keys:
                raise ValueError("Reference utt_id set must match all input segments")
            if any(not isinstance(row.get("text"), str) for row in reference_rows.values()):
                raise ValueError("Each reference needs string text")
            references = {key: tokens(row["text"]) for key, row in reference_rows.items()}
        prepared = [{key: prepare(row, args.mode, tokens, args.temperature) for key, row in system.items()} for system in systems]
        if args.work_dir:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            fused = rover_fuse(prepared, args.rover.resolve(), alpha, null_confidence, args.work_dir.resolve())
        else:
            with tempfile.TemporaryDirectory(prefix="roverfusion-") as directory:
                fused = rover_fuse(prepared, args.rover.resolve(), alpha, null_confidence, Path(directory))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joiner = "" if args.unit == "char" else " "
        with args.output.open("w", encoding="utf-8") as stream:
            for key in sorted(keys):
                stream.write(json.dumps({"utt_id": key, "text": joiner.join(fused[key]), "tokens": fused[key]}, ensure_ascii=False) + "\n")
        report = {"mode": args.mode, "metric": "CER" if args.unit == "char" else "WER", "segments": len(keys), "alpha": alpha,
                  "null_confidence": null_confidence, "temperature": args.temperature,
                  "normalization": {"unicode": "NFC", "strip_punctuation": args.strip_punctuation, "lowercase": args.lowercase},
                  "input_files": [str(path.resolve()) for path in args.inputs]}
        if references is not None:
            report["fused"] = evaluate(references, fused)
            report["systems"] = []
            for path, rows, system in zip(args.inputs, systems, prepared):
                # Report original recognizer top-1, plus Hystoc consensus before cross-system fusion.
                top1 = {}
                for key, row in rows.items():
                    text = max(row["nbest"], key=lambda h: float(h["log_score"]))["text"] if args.mode == "hystoc" else row["text"]
                    top1[key] = tokens(text)
                metrics = {"input": str(path), "top1": evaluate(references, top1)}
                if args.mode == "hystoc":
                    metrics["hystoc_consensus"] = evaluate(references, {key: [t for t, _ in seq] for key, seq in system.items()})
                report["systems"].append(metrics)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("Transcripts: " + str(args.output))
    except (ValueError, KeyError, ImportError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
