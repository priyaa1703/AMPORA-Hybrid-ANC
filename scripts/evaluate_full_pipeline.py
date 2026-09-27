#!/usr/bin/env python3
"""Run DC-CRN high-band inference, software hybrid recombination, and evaluation.

This is the main Colab evaluation path. It produces:
  results/dccrn_highband.csv
  results/hybrid_fullband.csv
  results/baselines.csv (run separately with run_baselines.py)
  results/plots/*.png

The hybrid experiment is reference-assisted in software: the injected noise waveform
acts as the reference microphone signal. This isolates algorithmic behavior without
claiming that a single microphone can supply that reference in a physical setup.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, math
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from src.models.dccrn import AMPORA_DCCRN
from src.preprocessing.audio import read_audio
from src.hybrid import hybrid_with_reference
from src.evaluation.metrics import pair_metrics, optional_stoi_pesq

SR=16000; NFFT=512; WIN=320; HOP=128; LOW=1500.; HIGH=7900.

def enhance_high(model, x, device):
    t=torch.from_numpy(x).unsqueeze(0).to(device); w=torch.hann_window(WIN,device=device)
    X=torch.stft(t,n_fft=NFFT,hop_length=HOP,win_length=WIN,window=w,return_complex=True,center=True)
    lo=int(math.ceil(LOW/(SR/NFFT))); hi=int(math.floor(HIGH/(SR/NFFT)))+1; Xh=X[:,lo:hi,:]
    M=model(torch.stack([Xh.real,Xh.imag],1)); Mc=torch.complex(M[:,0],M[:,1]); Yh=Xh*(1+Mc)
    Y=torch.zeros_like(X); Y[:,lo:hi,:]=Yh
    y=torch.istft(Y,w=w,n_fft=NFFT,hop_length=HOP,win_length=WIN,length=len(x),center=True)
    return y.squeeze(0).detach().cpu().numpy().astype(np.float32)

def save_plot(df, out):
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    for metric in ['Delta_SI_SDR_dB','Delta_SNR_dB']:
        if 'Method' not in df: continue
        g=df.groupby('Method')[metric].mean().sort_values()
        fig,ax=plt.subplots(figsize=(8,4)); g.plot.bar(ax=ax); ax.set_ylabel(metric); ax.set_title(metric+' by method'); fig.tight_layout(); fig.savefig(out/(metric+'_by_method.png'),dpi=160); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',default='results')
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); wavdir=out/'enhanced_wav'; wavdir.mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=AMPORA_DCCRN(16,64).to(device)
    ck=torch.load(a.checkpoint,map_location=device); model.load_state_dict(ck.get('model',ck)); model.eval()
    df=pd.read_csv(a.manifest); high_rows=[]; hybrid_rows=[]
    for idx,r in df.iterrows():
        clean=read_audio(r.clean); noisy=read_audio(r.noisy); ref=read_audio(r.noise_reference); n=min(len(clean),len(noisy),len(ref)); clean,noisy,ref=clean[:n],noisy[:n],ref[:n]
        high=enhance_high(model,noisy,device); hybrid=hybrid_with_reference(noisy,ref,high)
        hf=wavdir/f'{idx:05d}_dccrn_high.wav'; yf=wavdir/f'{idx:05d}_hybrid.wav'; sf.write(hf,high,SR); sf.write(yf,hybrid,SR)
        hm=pair_metrics(clean,noisy,high); hm.update({'Method':'DC-CRN-highband','Noise_Type':r.noise_category,'SNR_dB':r.snr_db}); high_rows.append(hm)
        fm=pair_metrics(clean,noisy,hybrid); fm.update({'Method':'AMPORA-hybrid-software','Noise_Type':r.noise_category,'SNR_dB':r.snr_db}); fm.update(optional_stoi_pesq(clean,noisy,hybrid)); hybrid_rows.append(fm)
    hdf=pd.DataFrame(high_rows); fdf=pd.DataFrame(hybrid_rows); hdf.to_csv(out/'dccrn_highband.csv',index=False); fdf.to_csv(out/'hybrid_fullband.csv',index=False)
    fdf.groupby('Noise_Type').mean(numeric_only=True).to_csv(out/'hybrid_by_noise_type.csv'); fdf.groupby('SNR_dB').mean(numeric_only=True).to_csv(out/'hybrid_by_snr.csv')
    save_plot(pd.concat([hdf,fdf],ignore_index=True),out/'plots')
    print('\nHybrid summary:'); print(fdf[['Delta_SNR_dB','Delta_SI_SDR_dB','Noisy_STOI','Enhanced_STOI','Noisy_PESQ','Enhanced_PESQ']].mean(numeric_only=True).round(4))
    print('Saved results under',out)
if __name__=='__main__': main()
