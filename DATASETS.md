# AMPORA dataset documentation

## Scope

The repository contains **no raw audio**. The current notebook mixes clean speech with
synthetic noise and optionally external noise. The project-specific recordings referenced
by the original notebook are private/not redistributed.

The public sources below are recommended for reproducible experiments. Licensing must be
checked again at download time and the original license/attribution must be retained.

## Public sources

| Dataset/resource | Official source | License / terms | Role |
|---|---|---|---|
| Microsoft DNS Challenge resources | https://github.com/microsoft/DNS-Challenge | The DNS repository documents source-specific terms; its code is MIT, while datasets retain their original terms. | Speech/noise/RIR resources and reproducible synthesis |
| MUSAN | https://www.openslr.org/17/ | CC BY 4.0 | Speech/music/noise; use the noise subset for noise experiments |
| DEMAND | https://zenodo.org/record/1227121 | CC BY-SA 3.0 as documented by Microsoft's DNS repository | Environmental noise |
| FSDD | https://github.com/Jakobovski/free-spoken-digit-dataset | Check the repository's current license before redistribution | Additional clean speech |

The Microsoft DNS repository documents DEMAND as CC BY-SA 3.0 and OpenSLR26/28 RIR data
as Apache 2.0. MUSAN's OpenSLR page identifies its license as CC BY 4.0.

## Current project-specific data

`AMPORA_V3_HARD_REALVOICE_DATASET` is not included. The original notebook expects:

```text
AMPORA_V3_HARD_REALVOICE_DATASET/
  clean_speech/
  noise_original/
  noise_impulsive/
```

Do not claim this private/project collection is a public or defence dataset.

## Reproducible preparation

1. Obtain public data directly from its official source.
2. Place clean speech under `data/raw/speech/`.
3. Place noise under category directories:
   `data/raw/stationary/`, `data/raw/nonstationary/`, `data/raw/transient/`.
4. Run:

```bash
python scripts/prepare_dataset.py   --speech data/raw/speech   --noise data/raw/stationary data/raw/nonstationary data/raw/transient   --out data/processed
```

5. Generate deterministic pairs:

```bash
python scripts/synthesize_pairs.py   --speech-csv data/processed/test_speech.csv   --noise-csv data/processed/noise.csv   --out data/processed/test_pairs   --snrs -5,0,5,10,15
```

Each generated sample records clean file, noisy file, noise category and target SNR.

## Real audio bundled directly in this repository (NEW)

`data/raw/speech_real/` and `data/raw/noise_real/` contain real, Creative-Commons
-licensed speech and noise (helicopter, aircraft, truck engine, industrial fan/rumble,
explosion, jackhammer) bundled directly in the repo -- see `ATTRIBUTIONS.md` for exact
sources and per-file licenses, and the README's "Real bundled audio" section for how to
use it. This is not a substitute for the sources below (it's generic recordings, not
defence-specific, and small: 6 speakers, 10 noise clips) but it does let the whole
pipeline run on genuine non-synthetic audio without waiting on any of the downloads
below.

## Stronger dataset generation (v2)

`scripts/synthesize_pairs_v2.py` (see README's "Stronger dataset pipeline" section)
generates a harder, more compound, more field-realistic dataset than the steps above:
wider SNR range (-15..+20 dB), compound noise beds (base + second clip + synthetic
impulsive burst train), synthetic reverberation, clipping/gain/spectral-tilt jitter, and
an isolated `noise_reference.wav` per sample for the low-band NLMS branch. It can either
consume the same `data/raw/...` layout as above, or bootstrap noise with **zero
downloads** via `--synthetic-noise-only`, using the procedural generators in
`src/augmentation/synthetic_transients.py` (band-limited decaying noise bursts for
impulsive events, a harmonic-comb-plus-wobble model for rotor/engine drone, a swept tone
for sirens). Those synthetic generators are plain DSP approximations, not recordings of
real defence equipment, and should be supplemented with a real, properly licensed corpus
(the public sources above, or the private `AMPORA_V3_HARD_REALVOICE_DATASET`) before any
production claim.

## Leakage policy

Speech is split by speaker identifier when one can be reliably extracted. If a speaker
cannot be identified, each recording is isolated as its own group. Chunks from the same
recording must never appear in different splits.

A future production experiment should also reserve entire noise sources for an unseen-noise
test rather than only relying on random noise generation.

## Sampling rate

The AMPORA model is currently configured for 16 kHz. The previous notebook described a
1.5–8 kHz high band. In the improved reproducible configuration, the digital upper cutoff
is 7.9 kHz rather than exactly Nyquist (8 kHz), leaving a transition margin.

Changing the sample rate requires retraining/re-exporting the model and regenerating
evaluation data.
