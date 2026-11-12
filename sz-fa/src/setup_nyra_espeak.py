"""Install espeak runtime under models, without changing system packages."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]/'models/nyra-espeak'


def main():
    packages = ROOT/'packages'
    packages.mkdir(parents=True, exist_ok=True)
    for name in ['libespeak-ng1','espeak-ng-data','libpcaudio0','libsonic0']:
        if not list(packages.glob(name+'_*.deb')):
            subprocess.run(['apt-get','download',name],cwd=packages,check=True)
    provenance = []
    for path in packages.glob('*.deb'):
        # --no-same-owner is required on the shared project filesystem.
        archive = subprocess.Popen(['dpkg-deb','--fsys-tarfile',str(path)],stdout=subprocess.PIPE)
        subprocess.run(['tar','--extract','--no-same-owner','--directory',str(ROOT)],stdin=archive.stdout,check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError('Cannot extract '+str(path))
        provenance.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ROOT/'provenance.json').write_text(json.dumps(dict(source='Ubuntu jammy apt packages',files=provenance),indent=2)+'\n')


if __name__ == '__main__':
    main()
