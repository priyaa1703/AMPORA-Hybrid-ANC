#!/usr/bin/env python3
"""Run the trained DC-CRN on a WAV and save enhanced high-band output.

This is intentionally explicit about scope: it enhances the high band only. For
full-band hybrid output use the hybrid experiment script after the low-band branch
is supplied with a reference signal.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, math
import numpy as np, soundfile as sf, torch
from src.models.dccrn import AMPORA_DCCRN
from src.preprocessing.audio import read_audio

SR=16000; NFFT=512; WIN=320; HOP=128; LOW=1500.; HIGH=7900.

def run(model,x,device):
    t=torch.from_numpy(x).unsqueeze(0).to(device); w=torch.hann_window(WIN,device=device)
    X=torch.stft(t,n_fft=NFFT,hop_length=HOP,win_length=WIN,window=w,return_complex=True,center=True)
    lo=int(math.ceil(LOW/(SR/NFFT))); hi=int(math.floor(HIGH/(SR/NFFT)))+1; Xh=X[:,lo:hi,:]
    inp=torch.stack([Xh.real,Xh.imag],1); M=model(inp); Mc=torch.complex(M[:,0],M[:,1]); Yh=Xh*(1+Mc)
    Y=X.clone(); Y[:,lo:hi,:]=Yh
    return torch.istft(Y,w=w,n_fft=NFFT,hop_length=HOP,win_length=WIN,length=len(x),center=True).squeeze(0).cpu().numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--input',required=True); ap.add_argument('--output',default='results/enhanced.wav')
    a=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=AMPORA_DCCRN(16,64).to(device)
    ck=torch.load(a.model,map_location=device); m.load_state_dict(ck.get('model',ck)); m.eval(); x=read_audio(a.input); y=run(m,x,device); sf.write(a.output,y,SR); print('Saved',a.output)
if __name__=='__main__': main()
