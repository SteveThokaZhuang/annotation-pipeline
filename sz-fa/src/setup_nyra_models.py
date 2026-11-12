"""Restore fixed-revision official Nyra weights, including bundled WavLM."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '9c8e724072261c30641a91ca365ce4f7b4ae7f83'
WAVLM_SHA256 = '92063b9e2a29279a9fe4f7e6703f5eeece903c7b3e471aebf2d9997f24fcdc5b'
FILES = ['config.json', 'meta.json', 'model.npz', 'phone_map.json', 'lexicon.txt',
         'LICENSE.md', 'README.md', 'wavlm/config.json', 'wavlm/model.safetensors']


def download_large(url, temporary):
    request = urllib.request.Request(url,headers={'Range':'bytes=0-0'})
    with urllib.request.urlopen(request,timeout=60) as response:
        total = int(response.headers['Content-Range'].rsplit('/',1)[1])
    completed = temporary.stat().st_size if temporary.exists() else 0
    part_size = 32*1024*1024
    ranges = [(a,min(a+part_size,total)-1) for a in range(completed,total,part_size)]
    def part(bounds):
        a,b = bounds
        path = temporary.with_name(temporary.name+f'.{a}-{b}')
        if path.exists() and path.stat().st_size==b-a+1:
            return path
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
                with urllib.request.urlopen(request,timeout=180) as response, path.open('wb') as stream:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {a}-{b}/{total}':
                        raise ValueError('Unexpected HTTP range response')
                    shutil.copyfileobj(response,stream,1024*1024)
                if path.stat().st_size!=b-a+1:
                    raise ValueError('Incomplete model range')
                print(f'WavLM bytes {a}-{b}/{total}',flush=True)
                return path
            except Exception:
                if attempt==2:
                    raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=8) as executor:
        parts=list(executor.map(part,ranges))
    with temporary.open('ab') as stream:
        for path in parts:
            with path.open('rb') as source:
                shutil.copyfileobj(source,stream,4*1024*1024)
            path.unlink()
    if temporary.stat().st_size!=total:
        raise ValueError('Incomplete WavLM file')


def main():
    output = ROOT/'models/nyra-forced-aligner/checkpoints'
    output.mkdir(parents=True, exist_ok=True)
    def download(name):
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/nyralabs/nyra_forced_aligner_en/resolve/{REVISION}/{name}'
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.partial')
            if name=='wavlm/model.safetensors':
                download_large(url,temporary)
            else:
                with urllib.request.urlopen(url, timeout=300) as response, temporary.open('wb') as stream:
                    shutil.copyfileobj(response, stream, 4*1024*1024)
            temporary.rename(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024), b''):
                digest.update(chunk)
        if name=='wavlm/model.safetensors' and digest.hexdigest()!=WAVLM_SHA256:
            raise ValueError('WavLM checksum differs from the fixed-revision official LFS SHA-256')
        print(name, path.stat().st_size, flush=True)
        return dict(file=name, url=url, bytes=path.stat().st_size, sha256=digest.hexdigest())
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(download, FILES))
    (output/'download-provenance.json').write_text(json.dumps(dict(
        repository='nyralabs/nyra_forced_aligner_en', revision=REVISION, files=files), indent=2)+'\n')


if __name__ == '__main__':
    main()
