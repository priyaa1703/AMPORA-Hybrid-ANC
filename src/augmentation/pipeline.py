"""Stronger dataset augmentation pipeline.

Combines, with per-sample randomisation:
  * multi-noise mixing (2-3 simultaneous noise sources instead of 1)
  * synthetic or real impulsive overlay (gunshot/artillery-like bursts) on top
    of a stationary/non-stationary bed, so the model sees *compound* defence
    scenarios (e.g. engine drone + sudden gunfire) rather than one noise type
    at a time
  * synthetic room reverberation (field / vehicle-cabin acoustics)
  * clipping distortion (radio/handset front-end overload)
  * random gain and simple spectral-tilt jitter (microphone/channel variation)
  * wide, curriculum-style SNR range (-15 dB .. +20 dB)

This directly addresses "stronger dataset": more compound, more acoustically
diverse, and more front-end-realistic noisy/clean pairs than plain single-noise
additive mixing.
"""
from __future__ import annotations
import numpy as np
from scipy import signal

from .synthetic_transients import synth_impulsive_train, synthetic_noise_bank
from .reverb import synth_rir, apply_reverb


def _rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _fit(x, n):
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n].astype(np.float32)


def spectral_tilt(x, sr, tilt_db_per_oct, rng):
    """Cheap channel-coloration model: a single-pole shelving filter whose
    slope is randomised per-sample to emulate mic/handset frequency response
    variation."""
    b = [1.0, -0.97 * np.sign(tilt_db_per_oct) * min(abs(tilt_db_per_oct) / 12.0, 0.97)]
    return signal.lfilter(b, [1.0], x).astype(np.float32)


def make_compound_noise(base_noise: np.ndarray, sr: int, length: int,
                         rng: np.random.Generator, extra_noise_pool=None,
                         impulsive_prob: float = 0.5, extra_noise_prob: float = 0.4) -> np.ndarray:
    """Build a compound noise bed: base noise + (optional) a second real noise
    clip + (optional) a synthetic impulsive burst train, all at randomised
    relative levels."""
    noise = _fit(base_noise, length).copy()

    if extra_noise_pool and rng.random() < extra_noise_prob:
        extra = _fit(extra_noise_pool[rng.integers(len(extra_noise_pool))], length)
        rel_db = rng.uniform(-6, 6)
        g = _rms(noise) / (10 ** (rel_db / 20)) / (_rms(extra) + 1e-8)
        noise = noise + extra * g

    if rng.random() < impulsive_prob:
        burst_train = synth_impulsive_train(sr=sr, length_s=length / sr,
                                             rate_hz=rng.uniform(0.4, 2.5), rng=rng)
        rel_db = rng.uniform(-4, 10)  # bursts are often louder than the bed
        g = _rms(noise) / (10 ** (rel_db / 20)) / (_rms(burst_train) + 1e-8)
        noise = noise + burst_train * g

    return noise.astype(np.float32)


def augment_pair(clean: np.ndarray, noise: np.ndarray, sr: int, rng: np.random.Generator,
                  snr_db: float, extra_noise_pool=None, reverb_prob: float = 0.4,
                  clip_prob: float = 0.15, tilt_prob: float = 0.3,
                  impulsive_prob: float = 0.5):
    """Produce one strong augmented (noisy, clean_target, noise_reference) triple.

    Returns
    -------
    noisy : compound noisy mixture the model receives as input
    clean : (possibly reverberated) target the model should reconstruct
    noise_reference : the isolated noise-bed estimate, useful as a synthetic
        "reference microphone" signal for the low-band NLMS branch.
    """
    n = min(len(clean), len(noise))
    clean = clean[:n].copy()
    compound_noise = make_compound_noise(noise[:n], sr, n, rng, extra_noise_pool,
                                          impulsive_prob=impulsive_prob)

    target = clean
    if rng.random() < reverb_prob:
        rir = synth_rir(sr=sr, t60_s=rng.uniform(0.15, 0.6), rng=rng)
        target = apply_reverb(clean, rir, wet=rng.uniform(0.15, 0.45))

    nr = _rms(compound_noise)
    if nr < 1e-8:
        noisy = target.copy()
    else:
        desired = _rms(target) / (10 ** (snr_db / 20.0))
        compound_noise = compound_noise * (desired / (nr + 1e-8))
        noisy = target + compound_noise

    if rng.random() < tilt_prob:
        noisy = spectral_tilt(noisy, sr, rng.uniform(-6, 6), rng)

    if rng.random() < clip_prob:
        thresh = rng.uniform(0.6, 0.95) * (np.max(np.abs(noisy)) + 1e-9)
        noisy = np.clip(noisy, -thresh, thresh)

    gain = 10 ** (rng.uniform(-3, 3) / 20.0)
    noisy = (noisy * gain).astype(np.float32)

    peak = np.max(np.abs(noisy)) + 1e-9
    if peak > 0.99:
        scale = 0.99 / peak
        noisy = noisy * scale
        target = target * scale

    return noisy.astype(np.float32), target.astype(np.float32), compound_noise.astype(np.float32)
