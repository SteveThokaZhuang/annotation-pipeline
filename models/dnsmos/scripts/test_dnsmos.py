#!/usr/bin/env python3
"""Regular DNSMOS P.835; official windowing/calibration with optimized ORT sessions.

Reference: microsoft/DNS-Challenge/DNSMOS/dnsmos_local.py.
Upstream reference and license are in scripts/upstream.
SIG=speech quality, BAK=background quality, OVRL=overall quality (higher is better).
Short recordings are repeated as in the official implementation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(os.environ.get("DNSMOS_SAMPLE", "sample.wav"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "cache" / "numba"))


class DNSMOS:
    def __init__(self, checkpoint, device="cpu", threads=2, p808=False):
        checkpoint=Path(checkpoint)
        import onnxruntime as ort
        if device == "cuda":
            if not hasattr(ort, "preload_dlls"):
                raise RuntimeError("Install with setup_dnsmos.sh --gpu first.")
            ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA provider unavailable. Install with --gpu or use --device cpu.")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

        def load(name):
            session = ort.InferenceSession(str(checkpoint / name), sess_options=options, providers=providers)
            if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("CUDA initialization failed. Check driver/libraries or use --device cpu.")
            return session

        self.primary = load("sig_bak_ovr.onnx")
        self.p808 = load("model_v8.onnx") if p808 else None
        self.providers = self.primary.get_providers()

    def score(self, path):
        import librosa
        import numpy as np
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float64", always_2d=True)
        audio = audio.mean(axis=1)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        duration = len(audio) / 16000
        # Preserve the official doubling and hop count for comparable scores.
        length = int(9.01 * 16000)
        while len(audio) < length:
            audio = np.concatenate((audio, audio))
        hops = int(np.floor(len(audio) / 16000) - 9.01) + 1
        raw_scores, calibrated, p808_scores = [], [], []
        for index in range(hops):
            segment = audio[index * 16000:index * 16000 + length]
            if len(segment) < length:
                continue
            features = segment.astype(np.float32)[None, :]
            raw = self.primary.run(None, {self.primary.get_inputs()[0].name: features})[0][0]
            sig, bak, ovrl = map(float, raw)
            raw_scores.append([sig, bak, ovrl])
            calibrated.append([
                np.polyval([-0.08397278, 1.22083953, 0.0052439], sig),
                np.polyval([-0.13166888, 1.60915514, -0.39604546], bak),
                np.polyval([-0.06766283, 1.11546468, 0.04602535], ovrl),
            ])
            if self.p808 is not None:
                mel = librosa.feature.melspectrogram(
                    y=segment[:-160], sr=16000, n_fft=321, hop_length=160, n_mels=120,
                )
                mel = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T.astype(np.float32)[None]
                value = self.p808.run(None, {self.p808.get_inputs()[0].name: mel})[0]
                p808_scores.append(float(value.reshape(-1)[0]))
        if not calibrated or not np.isfinite(calibrated).all():
            raise RuntimeError(f"Invalid DNSMOS output for {path}")
        result = {"filename": str(path), "len_in_sec": duration, "sr": 16000, "num_hops": len(calibrated)}
        for i, key in enumerate(("SIG", "BAK", "OVRL")):
            result[key] = float(np.mean(calibrated, axis=0)[i])
            result[key + "_raw"] = float(np.mean(raw_scores, axis=0)[i])
        if self.p808 is not None:
            if not np.isfinite(p808_scores).all():
                raise RuntimeError(f"Invalid P.808 output for {path}")
            result["P808_MOS"] = float(np.mean(p808_scores))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--audio", type=Path)
    inputs.add_argument("--input-dir", type=Path, help="Recursively score .wav files")
    default_file = ROOT / "default-device.txt"
    default_device = default_file.read_text().strip() if default_file.exists() else "cpu"
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
    parser.add_argument("--threads", type=int, default=2, help="ORT CPU threads per session")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent files; sessions are reused")
    parser.add_argument("--p808", action="store_true", help="Also compute P.808 (additional work)")
    parser.add_argument("--output", type=Path, help="Save JSON report")
    args = parser.parse_args()
    if args.threads < 1 or args.workers < 1:
        parser.error("threads and workers must be positive")
    files = sorted(args.input_dir.rglob("*.wav")) if args.input_dir else [args.audio or SAMPLE]
    if not files or any(not file.is_file() for file in files):
        parser.error("No valid input files found; supply --audio or --input-dir")
    started = time.perf_counter()
    scorer = DNSMOS(ROOT / "checkpoint", args.device, args.threads, args.p808)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(scorer.score, files))
    elapsed = time.perf_counter() - started
    report = {
        "providers": scorer.providers, "model_load_seconds": load_seconds,
        "scoring_seconds": elapsed, "results": results,
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
