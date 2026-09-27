"""Procedural synthetic acoustic *transient* generators for dataset augmentation.

Purpose
-------
Real defence-scenario impulsive recordings (gunshots, artillery, explosions) are
licensed/restricted and are NOT bundled with this repository (see ``DATASETS.md``).
To still give the impulsive-noise branch of the detector and the AI model enough
training variety, this module synthesises generic *broadband acoustic transients*
directly from first-principles DSP (a fast-attack / exponentially-decaying
band-limited noise burst, i.e. the same textbook model used for room acoustics
"impulse response" excitation and for foley/SFX percussive design). It contains
no ballistic, propellant, or device-specific information of any kind — it is a
noise-shaping utility, equivalent to what any DAW's "transient designer" plugin
does.

Use these to augment/replace real impulsive recordings when none are available,
or to enrich a real impulsive dataset with extra variety.
"""
from __future__ import annotations
import numpy as np
from scipy import signal


def synth_impulsive_burst(sr: int = 16000, duration_s: float = 0.6, decay_tau_s: float = 0.09,
                           center_hz: float = 900.0, bandwidth_hz: float = 2600.0,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """A single band-limited, exponentially decaying noise burst.

    Generic model: white noise -> band-pass shaping -> fast-attack/exp-decay
    envelope. Produces a broadband "thump/crack" transient shape without
    encoding any device-specific acoustic signature.
    """
    rng = rng or np.random.default_rng()
    n = int(sr * duration_s)
    white = rng.standard_normal(n).astype(np.float32)
    low = max(20.0, center_hz - bandwidth_hz / 2)
    high = min(sr / 2 - 100, center_hz + bandwidth_hz / 2)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    shaped = signal.sosfilt(sos, white).astype(np.float32)
    t = np.arange(n) / sr
    attack_s = 0.002
    env = 1.0 - np.exp(-t / attack_s)
    env *= np.exp(-t / decay_tau_s)
    burst = shaped * env
    peak = np.max(np.abs(burst)) + 1e-9
    return (burst / peak).astype(np.float32)


def synth_impulsive_train(sr: int = 16000, length_s: float = 2.0, rate_hz: float = 2.0,
                           jitter: float = 0.35, rng: np.random.Generator | None = None) -> np.ndarray:
    """A train of randomly-spaced synthetic bursts (e.g. sustained automatic fire /
    repeated impacts) laid over silence, for the "transient" noise category."""
    rng = rng or np.random.default_rng()
    n = int(sr * length_s)
    out = np.zeros(n, dtype=np.float32)
    period = sr / max(rate_hz, 1e-3)
    t = 0.0
    while t < n:
        jittered = t + rng.uniform(-jitter, jitter) * period
        idx = int(np.clip(jittered, 0, n - 1))
        if idx >= n - 1:
            t += period
            continue
        burst = synth_impulsive_burst(
            sr=sr,
            duration_s=rng.uniform(0.15, 0.5),
            decay_tau_s=rng.uniform(0.04, 0.14),
            center_hz=rng.uniform(400, 2500),
            bandwidth_hz=rng.uniform(1200, 4000),
            rng=rng,
        )
        end = min(n, idx + len(burst))
        out[idx:end] += burst[: end - idx] * rng.uniform(0.6, 1.0)
        t += period
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)


def synth_rotor_drone(sr: int = 16000, length_s: float = 2.0, base_hz: float = 95.0,
                       n_harmonics: int = 6, wobble_hz: float = 4.0,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic rotor/engine drone: a harmonic comb with slow amplitude wobble,
    approximating helicopter/vehicle engine noise for the "stationary/non-stationary"
    categories when licensed recordings are unavailable."""
    rng = rng or np.random.default_rng()
    n = int(sr * length_s)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    for h in range(1, n_harmonics + 1):
        amp = 1.0 / h
        phase = rng.uniform(0, 2 * np.pi)
        out += (amp * np.sin(2 * np.pi * base_hz * h * t + phase)).astype(np.float32)
    wobble = 1.0 + 0.15 * np.sin(2 * np.pi * wobble_hz * t + rng.uniform(0, 2 * np.pi))
    out *= wobble
    out += 0.05 * rng.standard_normal(n).astype(np.float32)
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)


def synthetic_noise_bank(sr: int = 16000, seed: int = 0):
    """Return a dict of {category: generator_fn(length_s)->np.ndarray} covering
    stationary / non-stationary / transient / mixed noise classes, usable to
    bootstrap or extend the dataset without any external download."""
    rng = np.random.default_rng(seed)
    return {
        "stationary_engine": lambda L: synth_rotor_drone(sr, L, base_hz=60.0, wobble_hz=0.5, rng=rng),
        "nonstationary_rotor": lambda L: synth_rotor_drone(sr, L, base_hz=95.0, wobble_hz=4.0, rng=rng),
        "transient_impulsive": lambda L: synth_impulsive_train(sr, L, rate_hz=rng.uniform(0.5, 3.0), rng=rng),
        "mixed_siren": lambda L: _synth_siren(sr, L, rng),
    }


def _synth_siren(sr: int, length_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * length_s)
    t = np.arange(n) / sr
    sweep_hz = 0.33
    f = 700 + 300 * signal.sawtooth(2 * np.pi * sweep_hz * t, width=0.5)
    phase = 2 * np.pi * np.cumsum(f) / sr
    out = np.sin(phase).astype(np.float32)
    out += 0.08 * rng.standard_normal(n).astype(np.float32)
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak).astype(np.float32)
