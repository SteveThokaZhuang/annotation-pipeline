"""Read published Seamless word alignment without renumbering source word IDs."""

import json
import math
from pathlib import Path


def load_published(root, rows):
    ids = {row["conversation_id"] for row in rows}
    index = {}
    for file in (root/"training/index_v1").glob("*.jsonl"):
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["interaction_id"] in ids:
                    index[entry["interaction_id"]] = entry
        if ids <= index.keys():
            break
    with (root/"seamless_fd_data/metadata.jsonl").open("rb") as stream:
        for row in rows:
            key = row["conversation_id"]
            if key not in index:
                yield row, None, None, "not_in_published_fa_index"
                continue
            entry = index[key]
            try:
                stream.seek(entry["metadata_byte_offset"])
                meta = json.loads(stream.read(entry["metadata_bytes"]))
                if meta["interaction_id"] != key or meta["participant_count"] != 2:
                    raise ValueError("published_fa_conversation_mismatch")
                tracks = {}
                for p in entry["participants"]:
                    speaker = p["sample_id"].rsplit("_", 1)[1]
                    source = Path(p["audio_path"])
                    if not source.is_file():
                        marker = "/seamless_fd/"
                        source = root/source.as_posix().split(marker, 1)[1]
                    tracks[speaker] = (source, 0)
                streams = {}
                discarded = 0
                for p in meta["participants"]:
                    speaker = p["sample_id"].rsplit("_", 1)[1]
                    words = []
                    for i, w in enumerate(p["words"]):
                        a, b = float(w["start_sec"]), float(w["end_sec"])
                        if math.isfinite(a) and math.isfinite(b) and 0 <= a < b and w["word"].strip():
                            words.append(dict(id=f"{speaker}:w{i}", text=w["word"], start=a, end=b))
                        else:
                            discarded += 1
                    streams[speaker] = sorted(words, key=lambda w: (w["start"], w["end"]))
                if (len(tracks) != 2 or len(set(tracks.values())) != 2 or
                        set(streams) != set(tracks) or not all(streams.values())):
                    raise ValueError("missing_published_participant_tracks_or_words")
                row["invalid_fa_words"] = discarded
                yield row, streams, tracks, None
            except (ValueError, IndexError, KeyError, TypeError) as exc:
                yield row, None, None, str(exc)
