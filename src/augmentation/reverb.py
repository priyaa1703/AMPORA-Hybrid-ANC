"""Synthetic room-impulse-response (RIR) generation and convolutional reverb
augmentation. Used to make the dataset more "field-realistic" (open field,
vehicle cabin, urban street reflections) without requiring a licensed RIR
corpus, and can additionally be pointed at a real RIR set (e.g. OpenSLR26/28)
when available — see DATASETS.md.
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def synth_rir(sr: int = 16000, t60_s: float = 0.35, direct_delay_ms: float = 2.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Statistical/stochastic RIR model (Poisson-arrival decaying noise, a
    standard fast approximation to a room impulse response used widely in
    speech-enhancement augmentation pipelines)."""
    rng = rng or np.random.default_rng()
    n = int(sr * (t60_s + 0.05))
    t = np.arange(n) / sr
    decay = np.exp(-t / (t60_s / np.log(1000)))  # -60 dB at t60_s
    noise = rng.standard_normal(n).astype(np.float32)
    rir = (noise * decay).astype(np.float32)
    delay = int(sr * direct_delay_ms / 1000)
    rir[:delay] = 0.0
    rir[delay] = 1.0  # direct path spike
    rir /= (np.max(np.abs(rir)) + 1e-9)
    return rir.astype(np.float32)


def apply_reverb(x: np.ndarray, rir: np.ndarray, wet: float = 0.35) -> np.ndarray:
    """Convolve ``x`` with ``rir`` and mix wet/dry, preserving input length and
    peak scale so downstream SNR mixing stays well-defined."""
    x = np.asarray(x, dtype=np.float32)
    wet_sig = signal.fftconvolve(x, rir)[: len(x)].astype(np.float32)
    peak_in = np.max(np.abs(x)) + 1e-9
    peak_wet = np.max(np.abs(wet_sig)) + 1e-9
    wet_sig *= peak_in / peak_wet
    return ((1 - wet) * x + wet * wet_sig).astype(np.float32)
