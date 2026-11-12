#!/usr/bin/env python3
"""Build the official NIST ROVER C sources using GCC (no make required)."""
import json
import configparser
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "vendor/SCTK/src/sclite"
    compiler = shutil.which("gcc")
    if not compiler:
        raise SystemExit("GCC is required to build ROVER.")
    makefile = (source / "makefile.in").read_text().replace("\\\n", " ")
    files = []
    for group in ("glib_c", "netlib_c", "align_c", "statlib_c"):
        match = re.search(r"^" + group + r"\s*=\s*(.+)$", makefile, re.MULTILINE)
        if not match:
            raise SystemExit("Cannot find source group: " + group)
        files.extend(match.group(1).split())
    (ROOT / "bin").mkdir(exist_ok=True)
    command = [compiler, "-O2", "-std=gnu89", "-I", str(source)]
    command += [str(source / name) for name in files + ["rover.c"]]
    command += ["-lm", "-o", str(ROOT / "bin/rover")]
    with (ROOT / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit("Build failed; see " + str(ROOT / "build.log"))
    manifest = {"build_command": command, "repositories": {}}
    for name in ("SCTK", "hystoc"):
        repo = ROOT / "vendor" / name
        commit=repo/'SOURCE_COMMIT'
        manifest['repositories'][name]={'commit':commit.read_text().strip() if commit.exists() else 'see THIRD_PARTY_NOTICES.md'}
    (ROOT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Built " + str(ROOT / "bin/rover"))


if __name__ == "__main__":
    main()
