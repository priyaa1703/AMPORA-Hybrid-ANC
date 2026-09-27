from __future__ import annotations
import numpy as np

def si_sdr(estimate,target,eps=1e-8):
    estimate=estimate-np.mean(estimate); target=target-np.mean(target)
    alpha=np.sum(estimate*target)/(np.sum(target*target)+eps)
    proj=alpha*target; err=estimate-proj
    return float(10*np.log10((np.sum(proj*proj)+eps)/(np.sum(err*err)+eps)))

def snr(clean,estimate,eps=1e-8):
    err=clean-estimate
    return float(10*np.log10((np.sum(clean*clean)+eps)/(np.sum(err*err)+eps)))

def pair_metrics(clean,noisy,enhanced):
    ns=snr(clean,noisy); es=snr(clean,enhanced)
    ni=si_sdr(noisy,clean); ei=si_sdr(enhanced,clean)
    return {
        "Noisy_SNR_dB":ns, "Enhanced_SNR_dB":es, "Delta_SNR_dB":es-ns,
        "Noisy_SI_SDR_dB":ni, "Enhanced_SI_SDR_dB":ei, "Delta_SI_SDR_dB":ei-ni,
    }

def optional_stoi_pesq(clean,noisy,enhanced,sr=16000):
    out={}
    try:
        from pystoi import stoi
        out["Noisy_STOI"]=float(stoi(clean,noisy,sr,extended=False))
        out["Enhanced_STOI"]=float(stoi(clean,enhanced,sr,extended=False))
    except Exception:
        out["Noisy_STOI"]=np.nan; out["Enhanced_STOI"]=np.nan
    try:
        from pesq import pesq
        mode="wb" if sr==16000 else "nb"
        out["Noisy_PESQ"]=float(pesq(sr,clean,noisy,mode))
        out["Enhanced_PESQ"]=float(pesq(sr,clean,enhanced,mode))
    except Exception:
        out["Noisy_PESQ"]=np.nan; out["Enhanced_PESQ"]=np.nan
    return out
