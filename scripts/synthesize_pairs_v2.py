#!/usr/bin/env python3
"""Stronger dataset generator (v2).

Builds on ``scripts/synthesize_pairs.py`` but adds, per generated sample:

  * a wider curriculum SNR range (default -15 .. +20 dB, vs -5..15 before)
  * compound noise beds (base noise + a second real noise clip + a synthetic
    impulsive burst train), instead of one noise clip per sample
  * synthetic room reverberation
  * clipping / channel spectral-tilt / gain-jitter front-end realism
  * an isolated ``noise_reference`` .wav per sample, suitable as the
    simulated reference-microphone signal for the low-band NLMS branch
    (this is what ``src/hybrid.py::hybrid_with_reference`` expects)
  * an option to bootstrap noise purely from ``src/augmentation`` synthetic
    generators when no licensed defence-noise corpus is available yet
    (``--synthetic-noise-only``)

Output manifest columns:
    clean, noisy, noise_reference, noise_category, snr_db, reverb, clipped
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.preprocessing.audio import read_audio
from src.augmentation.pipeline import augment_pair
from src.augmentation.synthetic_transients import synthetic_noise_bank


def fit(x, n):
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speech-csv", required=True, help="CSV with a 'path' column of clean speech files")
    ap.add_argument("--noise-csv", help="CSV with 'path' and 'noise_category' columns (optional if --synthetic-noise-only)")
    ap.add_argument("--out", default="data/processed/pairs_v2")
    ap.add_argument("--snr-min", type=float, default=-15.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--per-speech", type=int, default=2, help="augmented variants generated per clean utterance")
    ap.add_argument("--segment-seconds", type=float, default=1.28)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reverb-prob", type=float, default=0.4)
    ap.add_argument("--clip-prob", type=float, default=0.15)
    ap.add_argument("--tilt-prob", type=float, default=0.3)
    ap.add_argument("--impulsive-prob", type=float, default=0.5,
                     help="probability of overlaying a synthetic impulsive burst train on top of the noise bed")
    ap.add_argument("--synthetic-noise-only", action="store_true",
                     help="ignore --noise-csv and bootstrap all noise from procedural generators "
                          "in src/augmentation/synthetic_transients.py (no download required)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pyrng = random.Random(a.seed)
    speech = pd.read_csv(a.speech_csv)
    seg_n = int(a.segment_seconds * a.sr)

    noise_bank = synthetic_noise_bank(sr=a.sr, seed=a.seed) if a.synthetic_noise_only else None
    if a.synthetic_noise_only:
        noise_rows = None
    else:
        if not a.noise_csv:
            raise SystemExit("--noise-csv is required unless --synthetic-noise-only is set")
        noise_rows = pd.read_csv(a.noise_csv)

    out = Path(a.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    (out / "noise_reference").mkdir(parents=True, exist_ok=True)

    rows = []
    for _, s in speech.iterrows():
        clean_full = read_audio(s.path, sr=a.sr)
        clean_seg = fit(clean_full, seg_n)

        for k in range(a.per_speech):
            if a.synthetic_noise_only:
                category = pyrng.choice(list(noise_bank.keys()))
                base_noise = noise_bank[category](a.segment_seconds)
                extra_pool = [noise_bank[c](a.segment_seconds) for c in pyrng.sample(
                    list(noise_bank.keys()), k=min(2, len(noise_bank)))]
            else:
                nr = noise_rows.iloc[pyrng.randrange(len(noise_rows))]
                category = getattr(nr, "noise_category", "unknown")
                base_noise = fit(read_audio(nr.path, sr=a.sr), seg_n)
                pool_idx = pyrng.sample(range(len(noise_rows)), k=min(2, len(noise_rows)))
                extra_pool = [fit(read_audio(noise_rows.iloc[i].path, sr=a.sr), seg_n) for i in pool_idx]

            snr = float(rng.uniform(a.snr_min, a.snr_max))
            noisy, target, noise_ref = augment_pair(
                clean_seg, base_noise, a.sr, rng, snr,
                extra_noise_pool=extra_pool,
                reverb_prob=a.reverb_prob, clip_prob=a.clip_prob,
                tilt_prob=a.tilt_prob, impulsive_prob=a.impulsive_prob,
            )

            stem = f"{Path(s.path).stem}_{category}_{k}_{str(round(snr,1)).replace('-','m').replace('.','p')}"
            cf = out / "clean" / f"{stem}.wav"
            nf = out / "noisy" / f"{stem}.wav"
            rf = out / "noise_reference" / f"{stem}.wav"
            sf.write(cf, target, a.sr)
            sf.write(nf, noisy, a.sr)
            sf.write(rf, noise_ref, a.sr)
            rows.append([str(cf), str(nf), str(rf), category, snr])

    manifest = out / "pairs_v2.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean", "noisy", "noise_reference", "noise_category", "snr_db"])
        w.writerows(rows)
    print(f"Generated {len(rows)} strong pairs -> {manifest}")


if __name__ == "__main__":
    main()
