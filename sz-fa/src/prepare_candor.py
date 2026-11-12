"""Reuse existing ASR transcripts and cut the supplied Candor recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asr', type=Path, required=True)
    p.add_argument('--audio-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    import soundfile as sf
    a.output.mkdir(parents=True, exist_ok=False)
    transcripts = [json.loads(line) for line in a.asr.read_text().splitlines() if line.strip()]
    manifest, sources = [], []
    for recording in sorted({r['chunk_id'].rsplit(':c', 1)[0] for r in transcripts}):
        source = (a.audio_dir / (recording + '.mp3')).resolve()
        decoded = a.output / (recording + '.wav')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-ar', '16000', '-ac', '1', str(decoded)], check=True)
        audio, sr = sf.read(decoded, dtype='float32')
        chunks = [r for r in transcripts if r['chunk_id'].rsplit(':c', 1)[0] == recording]
        for r in chunks:
            start, end = round(r['start_s']*sr), round(r['end_s']*sr)
            if start < 0 or end > len(audio)+1 or end <= start:
                raise ValueError('Invalid source chunk: ' + r['chunk_id'])
            path = a.output / (r['chunk_id'].replace(':', '_') + '.wav')
            sf.write(path, audio[start:end], sr, subtype='PCM_16')
            manifest.append(dict(id=r['chunk_id'], recording_id=recording, source_audio=str(source),
                audio=str(path.resolve()), text=r['text'], language=r.get('language', 'English'),
                chunk_start=start/sr, chunk_end=min(end,len(audio))/sr,
                duration=len(audio[start:end])/sr))
        sources.append(dict(audio=str(source), seconds=len(audio)/sr,
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest(), chunks=len(chunks)))
        decoded.unlink()
        print(recording, len(chunks), len(audio)/sr, flush=True)
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    (a.output/'preparation.json').write_text(json.dumps(dict(
        asr=str(a.asr.resolve()), asr_sha256=hashlib.sha256(a.asr.read_bytes()).hexdigest(),
        sources=sources, chunks=len(manifest), audio_seconds=sum(r['duration'] for r in manifest),
        seconds=time.perf_counter()-started, sample_rate=16000, channels=1), indent=2)+'\n')


if __name__ == '__main__':
    main()
