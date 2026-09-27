#!/usr/bin/env python3
"""Reproducible high-band DC-CRN training.

This keeps the uploaded AMPORA architecture rather than replacing it.
It trains the high-band branch only. It does not implement the low-band NLMS branch.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, csv, math, random
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
from scipy import signal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.models.dccrn import AMPORA_DCCRN

SR=16000; NFFT=512; WIN=320; HOP=128; LOW=1500.; HIGH=7900.; SEG=1.28

def read(path):
    x,fs=sf.read(path,dtype="float32"); x=np.asarray(x,dtype=np.float32)
    if x.ndim==2: x=x.mean(1)
    if fs!=SR: x=signal.resample_poly(x,SR,fs).astype(np.float32)
    return np.nan_to_num(x)

def fit(x,n):
    if len(x)>=n: return x[:n]
    return np.resize(x,n).astype(np.float32)

def rms(x): return float(np.sqrt(np.mean(x*x)+1e-12))
def mix(c,n,s):
    g=rms(c)/(10**(s/20))/(rms(n)+1e-8); n=n*g; return c+n

def high(x):
    sos=signal.butter(4,[80,HIGH],btype="bandpass",fs=SR,output="sos"); x=signal.sosfilt(sos,x)
    sos=signal.butter(2,LOW,btype="highpass",fs=SR,output="sos"); return signal.sosfilt(sos,x).astype(np.float32)

class PairDataset(Dataset):
    def __init__(self,speech_csv,noise_csv,training=True,length=1200,seed=42):
        self.s=pd.read_csv(speech_csv); self.n=pd.read_csv(noise_csv); self.training=training
        self.length=length; self.seed=seed
    def __len__(self): return self.length if self.training else len(self.s)
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+i)
        c=fit(read(self.s.iloc[i%len(self.s)].path),int(SR*SEG))
        nr=self.n.iloc[int(rng.integers(len(self.n)))]
        n=fit(read(nr.path),len(c))
        snr=float(rng.choice([-5,0,5,10,15])) if self.training else float(rng.choice([-5,0,5,10,15]))
        y=mix(c,n,snr)
        return torch.from_numpy(high(y)).float(),torch.from_numpy(high(c)).float()


class PrecomputedPairDataset(Dataset):
    """Reads a manifest already containing paired (clean, noisy) audio -- i.e. the
    output of scripts/synthesize_pairs.py or scripts/synthesize_pairs_v2.py -- instead
    of mixing clean+noise on the fly like PairDataset does. Use this to train on the
    "stronger dataset" (compound noise, reverb, clipping, wider SNR range) produced by
    synthesize_pairs_v2.py, which PairDataset has no way to reproduce on the fly.

    Required columns: clean, noisy (both file paths). Any other columns
    (noise_reference, noise_category, snr_db, ...) are ignored here but are still
    useful for later per-category analysis in scripts/evaluate.py.
    """
    def __init__(self, pairs_csv):
        df = pd.read_csv(pairs_csv)
        missing = {"clean", "noisy"} - set(df.columns)
        if missing:
            raise ValueError(
                f"{pairs_csv} is missing required column(s) {missing}. "
                "Did you mean --train-speech/--val-speech/--noise instead of "
                "--train-pairs-csv/--val-pairs-csv? Those two modes take different "
                "manifest formats -- see the --help text."
            )
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        c = fit(read(row.clean), int(SR * SEG))
        y = fit(read(row.noisy), int(SR * SEG))
        return torch.from_numpy(high(y)).float(), torch.from_numpy(high(c)).float()

def stft(x,window):
    return torch.stft(x,n_fft=NFFT,hop_length=HOP,win_length=WIN,window=window,
                      return_complex=True,center=True)
def istft(x,window,length):
    return torch.istft(x,n_fft=NFFT,hop_length=HOP,win_length=WIN,window=window,
                       length=length,center=True)

def sisdr(est,tgt,eps=1e-8):
    est=est-est.mean(-1,keepdim=True); tgt=tgt-tgt.mean(-1,keepdim=True)
    a=(est*tgt).sum(-1,keepdim=True)/(tgt.pow(2).sum(-1,keepdim=True)+eps)
    p=a*tgt; e=est-p
    return 10*torch.log10((p.pow(2).sum(-1)+eps)/(e.pow(2).sum(-1)+eps))

def forward_audio(model,x,window):
    X=stft(x,window); lo=int(math.ceil(LOW/(SR/NFFT))); hi=int(math.floor(HIGH/(SR/NFFT)))+1
    Xh=X[:,lo:hi,:]; inp=torch.stack([Xh.real,Xh.imag],1)
    M=model(inp); Mc=torch.complex(M[:,0],M[:,1])
    Yh=Xh*(1+Mc)
    Y=torch.zeros_like(X); Y[:,lo:hi,:]=Yh
    return istft(Y,window,x.shape[-1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-speech"); ap.add_argument("--val-speech")
    ap.add_argument("--noise", help="noise manifest CSV (path,noise_category) -- used only with --train-speech/--val-speech")
    ap.add_argument("--train-pairs-csv", help="pre-generated (clean,noisy) pairs manifest from "
                     "synthesize_pairs.py / synthesize_pairs_v2.py -- an alternative to "
                     "--train-speech/--noise that trains on the pairs exactly as generated "
                     "(compound noise, reverb, clipping etc. if you used synthesize_pairs_v2.py) "
                     "instead of re-mixing clean+noise online every batch")
    ap.add_argument("--val-pairs-csv", help="same as --train-pairs-csv but for validation")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--out",default="models/trained")
    ap.add_argument("--device",default="auto")
    a=ap.parse_args()

    online_mode = a.train_speech is not None or a.val_speech is not None or a.noise is not None
    pairs_mode = a.train_pairs_csv is not None or a.val_pairs_csv is not None
    if online_mode and pairs_mode:
        raise SystemExit("Use either --train-speech/--val-speech/--noise OR "
                          "--train-pairs-csv/--val-pairs-csv, not both.")
    if online_mode and not (a.train_speech and a.val_speech and a.noise):
        raise SystemExit("--train-speech/--val-speech/--noise mode requires all three.")
    if pairs_mode and not (a.train_pairs_csv and a.val_pairs_csv):
        raise SystemExit("--train-pairs-csv/--val-pairs-csv mode requires both.")
    if not online_mode and not pairs_mode:
        raise SystemExit("Provide either --train-speech/--val-speech/--noise, or "
                          "--train-pairs-csv/--val-pairs-csv.")

    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else a.device if a.device!="auto" else "cpu")
    if device.type=="cpu": print("WARNING: CPU training is supported for smoke tests but may be slow.")

    if pairs_mode:
        print(f"[train.py] training on precomputed pairs: {a.train_pairs_csv} / {a.val_pairs_csv}")
        tr=PrecomputedPairDataset(a.train_pairs_csv); va=PrecomputedPairDataset(a.val_pairs_csv)
    else:
        print(f"[train.py] training with online clean+noise mixing: {a.train_speech} + {a.noise}")
        tr=PairDataset(a.train_speech,a.noise,True); va=PairDataset(a.val_speech,a.noise,False)

    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0)
    vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=0)
    model=AMPORA_DCCRN(16,64).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=2e-6)
    window=torch.hann_window(WIN,device=device)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    best=float("inf"); history=[]
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for noisy,clean in tl:
            noisy,clean=noisy.to(device),clean.to(device); opt.zero_grad()
            enh=forward_audio(model,noisy,window)
            loss=(-sisdr(enh,clean).mean()) + 0.05*F.l1_loss(torch.log1p(torch.abs(stft(enh,window))),torch.log1p(torch.abs(stft(clean,window))))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(loss.item())
        model.eval(); vlss=[]; improvements=[]
        with torch.no_grad():
            for noisy,clean in vl:
                noisy,clean=noisy.to(device),clean.to(device); enh=forward_audio(model,noisy,window)
                vlss.append(float((-sisdr(enh,clean).mean()).item()))
                improvements.extend((sisdr(enh,clean)-sisdr(noisy,clean)).cpu().numpy().tolist())
        sched.step(); v=float(np.mean(vlss)); imp=float(np.mean(improvements))
        tl_loss=float(np.mean(losses))
        print(f"epoch={epoch+1} train_loss={tl_loss:.5f} val_loss={v:.5f} delta_si_sdr={imp:.4f} dB")
        history.append({"epoch":epoch+1,"train_loss":tl_loss,"val_loss":v,"delta_si_sdr_db":imp})
        state={"epoch":epoch+1,"model":model.state_dict(),"optimizer":opt.state_dict(),"validation_loss":v}
        torch.save(state,out/"last.pt")
        if v<best: best=v; torch.save(state,out/"best.pt")
    import json
    with open(out/"history.json","w") as f: json.dump(history,f,indent=2)
    print("best checkpoint:",out/"best.pt")
    print("training history:",out/"history.json")

if __name__=="__main__": main()
