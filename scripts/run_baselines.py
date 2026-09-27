#!/usr/bin/env python3
"""Run classical baselines on a paired manifest.

Manifest columns: clean,noisy,noise_category,snr_db,noise_reference
The reference signal should be the actual injected noise for the reference-assisted
NLMS baseline. This is explicitly a simulation baseline, not a claim of single-mic ANC.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
import pandas as pd
import soundfile as sf
from src.baselines import spectral_subtraction, wiener_filter, nlms_reference_anc
from src.evaluation.metrics import pair_metrics
from src.preprocessing.audio import read_audio


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('--out',default='results/baselines.csv')
    a=ap.parse_args(); df=pd.read_csv(a.manifest); rows=[]
    for _,r in df.iterrows():
        c=read_audio(r.clean); y=read_audio(r.noisy); ref=read_audio(r.noise_reference)
        outputs={
            'SpectralSubtraction': spectral_subtraction(y,ref),
            'Wiener': wiener_filter(y,ref),
            'NLMS': nlms_reference_anc(y,ref),
        }
        for name,e in outputs.items():
            m=pair_metrics(c,y,e); m.update({'Method':name,'Noise_Type':r.noise_category,'SNR_dB':r.snr_db}); rows.append(m)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).groupby('Method')[['Delta_SNR_dB','Delta_SI_SDR_dB']].mean().round(4)); print('Saved',out)

if __name__=='__main__': main()
