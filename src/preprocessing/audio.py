from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

def read_audio(path: str, sr: int = 16000) -> np.ndarray:
    x, fs = sf.read(path, dtype="float32")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if fs != sr:
        g = math.gcd(int(fs), int(sr))
        x = signal.resample_poly(x, sr // g, int(fs) // g).astype(np.float32)
    return np.nan_to_num(x).astype(np.float32)

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x*x) + 1e-12))

def mix_at_snr(clean, noise, snr_db):
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    n = min(len(clean), len(noise))
    clean, noise = clean[:n], noise[:n]
    nr = rms(noise)
    if nr < 1e-8:
        return clean.copy(), np.zeros_like(clean)
    desired = rms(clean) / (10.0 ** (snr_db / 20.0))
    noise = noise * (desired / (nr + 1e-8))
    return clean + noise, noise

def bandpass(x, low_hz, high_hz, sr=16000, order=4):
    nyq = sr/2
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError("Band must lie strictly inside Nyquist.")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)

def highpass(x, cutoff_hz, sr=16000, order=2):
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)

def extract_high_band(x, sr=16000, low=1500, high=7900):
    return highpass(bandpass(x, 80, high, sr), low, sr)

def rms_snr(clean, estimate):
    err = clean - estimate
    return 10*np.log10((np.sum(clean**2)+1e-12)/(np.sum(err**2)+1e-12))
