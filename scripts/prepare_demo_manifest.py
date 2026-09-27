#!/usr/bin/env python3
"""Create reproducible clean/noisy/reference pairs for evaluation and baselines."""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, csv, random
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from src.preprocessing.audio import read_audio, mix_at_snr


def fit(x,n, rng):
    x=np.asarray(x,dtype=np.float32)
    if len(x)>=n:
        start=0 if len(x)==n else int(rng.integers(0,len(x)-n+1)); return x[start:start+n]
    reps=int(np.ceil(n/len(x))) if len(x) else 1
    return np.tile(x,reps)[:n] if len(x) else np.zeros(n,dtype=np.float32)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--speech-csv',required=True); ap.add_argument('--noise-csv',required=True)
    ap.add_argument('--out',default='data/processed/eval_pairs'); ap.add_argument('--snrs',default='-5,0,5,10,15'); ap.add_argument('--per-speech',type=int,default=1); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed); random.seed(a.seed)
    speech=pd.read_csv(a.speech_csv); noise=pd.read_csv(a.noise_csv); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; snrs=[float(x) for x in a.snrs.split(',')]
    for _,s in speech.iterrows():
        clean=fit(read_audio(s.path), int(1.28*16000), rng)
        for j in range(a.per_speech):
            nr=noise.iloc[int(rng.integers(len(noise)))]; nz=fit(read_audio(nr.path),len(clean),rng); snr=float(rng.choice(snrs))
            noisy,scaled=mix_at_snr(clean,nz,snr)
            stem=f'{Path(s.path).stem}_{Path(nr.path).stem}_{snr:g}_{j}'
            cf=out/f'{stem}_clean.wav'; nf=out/f'{stem}_noisy.wav'; rf=out/f'{stem}_reference.wav'
            sf.write(cf,clean,16000); sf.write(nf,noisy,16000); sf.write(rf,scaled,16000)
            rows.append([str(cf),str(nf),str(rf),nr.noise_category,snr])
    mf=out/'manifest.csv'; pd.DataFrame(rows,columns=['clean','noisy','noise_reference','noise_category','snr_db']).to_csv(mf,index=False); print('Saved',mf,'rows=',len(rows))
if __name__=='__main__': main()
