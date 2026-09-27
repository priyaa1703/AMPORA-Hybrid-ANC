#!/usr/bin/env python3
"""Download small/optional public resources; large datasets are documented, not silently fetched."""
import argparse, subprocess, urllib.request
from pathlib import Path

FSDD_REPO="https://github.com/Jakobovski/free-spoken-digit-dataset.git"
MUSAN_URL="https://www.openslr.org/resources/17/musan.tar.gz"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="data/raw")
    ap.add_argument("--fsdd",action="store_true")
    ap.add_argument("--musan",action="store_true")
    a=ap.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    if a.fsdd:
        dest=root/"fsdd"
        if not dest.exists(): subprocess.run(["git","clone","--depth","1",FSDD_REPO,str(dest)],check=True)
    if a.musan:
        dest=root/"musan.tar.gz"
        if not dest.exists():
            print("MUSAN is a large download; verify current OpenSLR terms before use.")
            urllib.request.urlretrieve(MUSAN_URL,dest)
    if not (a.fsdd or a.musan):
        print("No download requested. See DATASETS.md for DNS/DEMAND/MUSAN sources and licensing.")
if __name__=="__main__": main()
