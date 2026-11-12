"""Download a pinned official MFA English model and its matching dictionary."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = '9c8d5d8e1dd5d9cf188a5571203de0e09ca6bede'
FILES = ['acoustic/final.alimdl', 'acoustic/final.mdl', 'acoustic/lda.mat',
         'acoustic/meta.json', 'acoustic/phones.txt', 'acoustic/tree',
         'dictionary/english_us_arpa.dict']


def main():
    output = ROOT / 'models/mfa-english'
    output.mkdir(parents=True, exist_ok=True)
    def download(name):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/MontrealCorpusTools/english_us_arpa/resolve/{REVISION}/{name}'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=180) as response:
                path.write_bytes(response.read())
        print(name, path.stat().st_size, flush=True)
        return dict(file=name, url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(download, FILES))
    with zipfile.ZipFile(output/'english_us_arpa.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            if name.startswith('acoustic/'):
                z.write(output/name, 'english_us_arpa/'+Path(name).name)
    (output/'provenance.json').write_text(json.dumps(dict(repository='MontrealCorpusTools/english_us_arpa',
        revision=REVISION, files=files), indent=2)+'\n')


if __name__ == '__main__':
    main()
