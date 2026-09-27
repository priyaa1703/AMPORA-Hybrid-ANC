#!/usr/bin/env python3
"""Evaluate paired clean/noisy/enhanced WAVs.

CSV columns required: clean,noisy,enhanced,noise_category,snr_db
For this repository's current high-band model, set --high-band to make the
scope explicit. Full-band STOI/PESQ should be run only after low/high branches
are recombined.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, pandas as pd
from pathlib import Path
from src.preprocessing.audio import read_audio, extract_high_band
from src.evaluation.metrics import pair_metrics, optional_stoi_pesq

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--out",default="results/metrics.csv")
    ap.add_argument("--high-band",action="store_true")
    ap.add_argument("--extended",action="store_true",
                     help="also compute SegSNR / freq-weighted SegSNR / impulsive-only "
                          "Delta-SI-SDR from src/evaluation/extended_metrics.py")
    a=ap.parse_args()
    df=pd.read_csv(a.manifest); rows=[]
    for _,r in df.iterrows():
        c=read_audio(r.clean); n=read_audio(r.noisy); e=read_audio(r.enhanced)
        if a.high_band: c,n,e=map(extract_high_band,(c,n,e))
        m=pair_metrics(c,n,e); m.update({"Noise_Type":r.noise_category,"Test_SNR":r.snr_db})
        if not a.high_band: m.update(optional_stoi_pesq(c,n,e))
        if a.extended:
            from src.evaluation.extended_metrics import extended_pair_metrics
            m.update(extended_pair_metrics(c, n, e))
        rows.append(m)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    result=pd.DataFrame(rows)
    print(result.select_dtypes("number").mean(numeric_only=True).round(4))
    print("Saved",out)
if __name__=="__main__": main()
