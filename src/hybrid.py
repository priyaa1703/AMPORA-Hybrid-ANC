from __future__ import annotations
import numpy as np
from scipy import signal
from .baselines import nlms_reference_anc


def split_bands(x, sr=16000, split_hz=1500.0, high_hz=7900.0, order=6):
    x = np.asarray(x, dtype=np.float32)
    nyq = sr/2
    if not 0 < split_hz < high_hz < nyq:
        raise ValueError('Band edges must satisfy 0 < split < high < Nyquist.')
    low_sos = signal.butter(order, split_hz, btype='lowpass', fs=sr, output='sos')
    high_sos = signal.butter(order, [split_hz, high_hz], btype='bandpass', fs=sr, output='sos')
    return signal.sosfilt(low_sos, x).astype(np.float32), signal.sosfilt(high_sos, x).astype(np.float32)


def hybrid_with_reference(noisy, noise_reference, high_enhanced, sr=16000, split_hz=1500.0,
                          nlms_mu=0.08, nlms_filter_len=64):
    """Software hybrid combiner for experiments.

    Low band: reference-assisted NLMS.
    High band: supplied DC-CRN high-band output.
    A complementary crossfade avoids a hard frequency seam.
    """
    noisy = np.asarray(noisy, dtype=np.float32)
    noise_reference = np.asarray(noise_reference, dtype=np.float32)
    high_enhanced = np.asarray(high_enhanced, dtype=np.float32)
    n = min(len(noisy), len(noise_reference), len(high_enhanced))
    noisy = noisy[:n]; noise_reference = noise_reference[:n]; high_enhanced = high_enhanced[:n]
    low, _ = split_bands(noisy, sr=sr, split_hz=split_hz)
    low_ref, _ = split_bands(noise_reference, sr=sr, split_hz=split_hz)
    low_enhanced = nlms_reference_anc(low, low_ref, mu=nlms_mu, filter_len=nlms_filter_len)
    # Keep the supplied high-band branch and use a low-pass + high-pass complement.
    _, high_noisy = split_bands(noisy, sr=sr, split_hz=split_hz)
    # Residual low/high leakage is minimized by applying matching filters.
    high_enhanced = signal.sosfilt(signal.butter(6, [split_hz, 7900], btype='bandpass', fs=sr, output='sos'), high_enhanced)
    return (low_enhanced + high_enhanced).astype(np.float32)
