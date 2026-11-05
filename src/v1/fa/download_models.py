"""Download pinned public FA checkpoints to the corresponding models directories."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from common import ROOT

HF_MODELS=[('Qwen/Qwen3-ForcedAligner-0.6B','c7cbfc2048c462b0d63a45797104fc9db3ad62b7','qwen3-forced-aligner'),('nyralabs/nyra_forced_aligner_en','9c8e724072261c30641a91ca365ce4f7b4ae7f83','nyra-forced-aligner')]
def main():
    from huggingface_hub import snapshot_download
    for repo,revision,subdir in HF_MODELS:
        p=ROOT/'models'/subdir/'checkpoints'
        snapshot_download(repo,revision=revision,local_dir=str(p),max_workers=4)
        (p/'download-provenance.json').write_text(json.dumps(dict(repo=repo,revision=revision),indent=2))
    p=ROOT/'models/Montreal-Forced-Aligner/checkpoints';p.mkdir(parents=True,exist_ok=True)
    sources=[]
    for kind,name,ext,sha in [('acoustic','english_mfa','zip','2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068'),('dictionary','english_us_mfa','dict','cbf4ed8cee1338e009b9b387bcb9ee789ce70b98c61ca9bc6ed4f4e63d4812d5')]:
        url=f'https://github.com/MontrealCorpusTools/mfa-models/releases/download/{kind}-{name}-v3.1.0/{name}.{ext}'
        dest=p/(name+'.'+ext)
        if not dest.exists():
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,timeout=120) as r: data=r.read()
                    dest.write_bytes(data);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(2)
        actual=hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual!=sha: raise ValueError(f'Checksum mismatch for {dest}: {actual}')
        sources.append(dict(url=url,file=str(dest),sha256=actual))
    (p/'download-provenance.json').write_text(json.dumps(sources,indent=2))
if __name__=='__main__':main()
