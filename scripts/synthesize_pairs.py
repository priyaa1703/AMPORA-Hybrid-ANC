#!/usr/bin/env python3
"""Create deterministic clean/noisy pairs and metadata.

Noise files are expected to be categorized by their parent directory name:
stationary, nonstationary, transient, or mixed.
No raw audio is copied into the repository.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, csv, random
from pathlib import Path
import numpy as np
from src.preprocessing.audio import read_audio, mix_at_snr

def fit(x,n):
    if len(x)>=n: return x[:n]
    return np.resize(x,n).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--speech-csv",required=True)
    ap.add_argument("--noise-csv",required=True)
    ap.add_argument("--out",default="data/processed/pairs")
    ap.add_argument("--snrs",default="-5,0,5,10,15")
    ap.add_argument("--per-speech",type=int,default=1)
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    rng=random.Random(a.seed)
    import pandas as pd
    speech=pd.read_csv(a.speech_csv); noise=pd.read_csv(a.noise_csv)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for _,s in speech.iterrows():
        clean=read_audio(s.path); n=min(len(clean),int(1.28*16000)); clean=fit(clean,n)
        for _ in range(a.per_speech):
            nr=noise.iloc[rng.randrange(len(noise))]
            nz=fit(read_audio(nr.path),n)
            snr=float(rng.choice([float(x) for x in a.snrs.split(",")]))
            noisy,scaled=mix_at_snr(clean,nz,snr)
            # Write only generated evaluation pairs; these can be excluded from git.
            stem=f"{Path(s.path).stem}_{Path(nr.path).stem}_{str(snr).replace('-','m').replace('.','p')}"
            cf=out/f"{stem}_clean.wav"; nf=out/f"{stem}_noisy.wav"
            import soundfile as sf
            sf.write(cf,clean,16000); sf.write(nf,noisy.astype(np.float32),16000)
            rows.append([str(cf),str(nf),nr.noise_category,snr])
    with open(out/"pairs.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["clean","noisy","noise_category","snr_db"]); w.writerows(rows)
    print("Pairs:",len(rows),"manifest:",out/"pairs.csv")
if __name__=="__main__": main()
