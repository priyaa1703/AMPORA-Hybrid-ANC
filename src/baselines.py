from __future__ import annotations
import numpy as np
from scipy import signal


def _stft(x, sr=16000, n_fft=512, hop=128, win=320):
    w = signal.windows.hann(win, sym=False)
    f, t, z = signal.stft(x, fs=sr, window=w, nperseg=win, noverlap=win-hop,
                           nfft=n_fft, boundary=None, padded=False)
    return f, t, z


def _istft(z, sr=16000, hop=128, win=320, length=None):
    w = signal.windows.hann(win, sym=False)
    _, x = signal.istft(z, fs=sr, window=w, nperseg=win, noverlap=win-hop,
                         input_onesided=True, boundary=None)
    x = np.asarray(x, dtype=np.float32)
    if length is not None:
        if len(x) < length:
            x = np.pad(x, (0, length-len(x)))
        x = x[:length]
    return x


def spectral_subtraction(noisy, noise_reference, sr=16000, alpha=1.0, floor=0.05):
    """Classical magnitude spectral subtraction using a supplied noise reference."""
    _, _, X = _stft(noisy, sr)
    _, _, N = _stft(noise_reference, sr)
    frames = min(X.shape[1], N.shape[1])
    X, N = X[:, :frames], N[:, :frames]
    mag = np.abs(X); phase = np.angle(X)
    noise_mag = np.median(np.abs(N), axis=1, keepdims=True)
    clean_mag = np.maximum(mag - alpha * noise_mag, floor * mag)
    return _istft(clean_mag*np.exp(1j*phase), sr, length=len(noisy))


def wiener_filter(noisy, noise_reference, sr=16000, floor=1e-6):
    """Classical frequency-domain Wiener gain using a supplied noise reference."""
    _, _, X = _stft(noisy, sr)
    _, _, N = _stft(noise_reference, sr)
    frames = min(X.shape[1], N.shape[1])
    X, N = X[:, :frames], N[:, :frames]
    p_n = np.mean(np.abs(N)**2, axis=1, keepdims=True)
    p_y = np.abs(X)**2
    p_s = np.maximum(p_y-p_n, 0.0)
    gain = p_s/(p_s+p_n+floor)
    return _istft(gain*X, sr, length=len(noisy))


def nlms_reference_anc(primary, reference, mu=0.1, filter_len=64, eps=1e-8):
    """Reference-assisted NLMS ANC.

    The reference is assumed to correlate with the additive noise in primary.
    This is a software simulation baseline; it requires a reference signal.
    """
    primary = np.asarray(primary, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    n = min(len(primary), len(reference))
    d = primary[:n]
    x = reference[:n]
    w = np.zeros(filter_len, dtype=np.float32)
    ref = np.zeros(filter_len, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ref[1:] = ref[:-1]; ref[0] = x[i]
        y = float(np.dot(w, ref))
        e = d[i] - y
        norm = float(np.dot(ref, ref) + eps)
        w += (mu * e / norm) * ref
        out[i] = e
    return out
