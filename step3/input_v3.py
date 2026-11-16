"""Read completed V3 recording exports and bind them to known participant tracks."""

import json
import math
import re
from pathlib import Path


def restore_punctuation(words, text):
    # The aligner can strip punctuation. Restore sentence marks only when
    # the whole lexical sequence matches, without changing any word times.
    letters = re.compile(r"[^\W_]", re.UNICODE)
    matches = list(letters.finditer(text))
    lexical = [letters.findall(word["text"]) for word in words]
    if "".join(m.group() for m in matches).casefold() != "".join(
        char for token in lexical for char in token
    ).casefold():
        return
    index = 0
    for word, token in zip(words, lexical):
        index += len(token)
        if token:
            stop = matches[index].start() if index < len(matches) else len(text)
            tail = text[matches[index - 1].end():stop]
            mark = re.match(r"[.!?。！？]+", tail)
            if mark and not word["text"].endswith(tuple(".!?。！？")):
                word["text"] += mark.group().translate(str.maketrans("。！？", ".!?"))


def load_channel(align_dir, participant, path_base):
    recording_id = participant["recording_id"]
    channel = participant["channel"]
    if not re.fullmatch(r"[\w-]+", recording_id) or type(channel) is not int or channel < 0:
        raise ValueError("invalid_recording_channel")
    key = f"{recording_id}_c{channel}"
    export = json.loads((align_dir / f"{key}.json").read_bytes())
    if export["recording_channel"] != key:
        raise ValueError("recording_channel_mismatch")
    coverage = export.get("coverage")
    if coverage and coverage["status"] != "complete":
        raise ValueError(f"partial_fa:{key}:{coverage['missing_chunk_ids']}")
    chunks = export["chunks"]
    if not chunks:
        raise ValueError(f"missing_fa_chunks:{key}")
    result, paths, seen = [], set(), set()
    for row in sorted(chunks, key=lambda row: row["chunk"]["start"]):
        chunk = row["chunk"]
        if (chunk["recording_id"], chunk["channel"]) != (recording_id, channel):
            raise ValueError(f"foreign_chunk:{chunk['id']}")
        if chunk["id"] in seen:
            raise ValueError(f"duplicate_chunk:{chunk['id']}")
        seen.add(chunk["id"])
        lo, hi = float(chunk["start"]), float(chunk["end"])
        source = participant.get("audio_path") or chunk.get("source_path") or chunk.get("path")
        if not source or not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
            raise ValueError(f"invalid_chunk:{chunk['id']}")
        paths.add((path_base / source).resolve())
        if not row["words"] and row["text"].strip():
            raise ValueError(f"missing_word_alignment:{chunk['id']}")
        words = []
        for index, word in enumerate(row["words"]):
            start, end = float(word["start"]), float(word["end"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= hi-lo+0.001):
                raise ValueError(f"invalid_word_alignment:{chunk['id']}:{index}")
            words.append(dict(id=f"{key}:{chunk['id']}:{index}", text=word["text"],
                              start=round(lo+start, 6), end=round(lo+min(end, hi-lo), 6),
                              chunk_id=chunk["id"], word_index=index))
        restore_punctuation(words, row["text"])
        result.extend(words)
    result.sort(key=lambda w: (w["start"], w["end"], w["id"]))
    if len(paths) != 1:
        raise ValueError(f"inconsistent_source_audio:{key}")
    if any(b["start"] < a["end"] - .001 for a, b in zip(result, result[1:])):
        raise ValueError(f"overlapping_same_channel_fa:{key}")
    return result, (paths.pop(), participant.get("audio_channel", channel))


def load_conversation(align_dir, row, path_base):
    participants = row["participants"]
    if len(participants) != 2 or len({p["speaker"] for p in participants}) != 2:
        raise ValueError("expected_two_mapped_participants")
    if len({(p["recording_id"], p["channel"]) for p in participants}) != 2:
        raise ValueError("duplicate_fa_channel")
    if any(not isinstance(p["speaker"], str) or not p["speaker"].strip() for p in participants):
        raise ValueError("invalid_speaker_id")
    streams, tracks = {}, {}
    for p in participants:
        streams[p["speaker"]], tracks[p["speaker"]] = load_channel(align_dir, p, path_base)
        if not streams[p["speaker"]]:
            raise ValueError(f"missing_participant_words:{p['speaker']}")
    if len(set(tracks.values())) != 2:
        raise ValueError("duplicate_source_audio_channel")
    return streams, tracks
