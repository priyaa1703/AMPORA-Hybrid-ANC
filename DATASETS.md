# Dataset Sources

This document lists the speech, noise, and acoustic datasets used or referenced for the AMPORA high-band speech-enhancement workflow.

## 1. Speech Data

### Free Spoken Digit Dataset (FSDD)

FSDD is used as an additional clean-speech source for speech-processing and model-development experiments.

Repository:
https://github.com/Jakobovski/free-spoken-digit-dataset

### Project Real Speech Data

The AMPORA project also uses real human speech recordings included in:

`AMPORA_V3_HARD_REALVOICE_DATASET`

The current project dataset contains:

* 169 real human speech segments
* 6 distinct speakers
* Creative Commons-licensed recordings

The dataset is used for model development, speech/noise mixing, validation, and evaluation.

The raw project dataset is intentionally not included in this GitHub repository because of its size and to avoid publishing private or restricted recordings.

---

## 2. Speech Enhancement Benchmark Dataset

### VoiceBank + DEMAND

VoiceBank-DEMAND is a commonly used speech-enhancement dataset containing clean speech and environmental noise.

It is referenced/used as a benchmark dataset for speech-enhancement experiments and comparative evaluation.

Source:
https://flageval.baai.ac.cn/docs/en/audio/Speech-Enhancement/Speech-Enhancement.html

---

## 3. Noise Data

### MUSAN

MUSAN provides speech, music, and noise recordings that can be used for speech and noise-processing experiments.

Source:
https://www.openslr.org/17/

### RIRS_NOISES

RIRS_NOISES provides room impulse responses and related acoustic data for speech-processing experiments and acoustic augmentation.

Source:
https://www.openslr.org/28/

---

## 4. Project Real-World Noise Data

The AMPORA project additionally uses real-world noise recordings covering the three noise categories relevant to the problem statement.

### Stationary Noise

Examples include:

* Industrial fan
* Air-conditioning rumble
* Truck/vehicle engine noise

### Non-Stationary Noise

Examples include:

* Helicopter noise
* Aircraft landing noise
* Chainsaw noise

### Transient / Impulsive Noise

Examples include:

* Real gunfire recordings from multiple firearm models
* Real explosion recordings
* Sirens

The project description identifies these transient-noise sources as being obtained from SESA and an edge-collected gunshot dataset.

These sources are used for noise-condition testing and mixed-noise speech-enhancement experiments. They should not be described as a dedicated licensed defence-noise corpus.

---

## 5. Synthetic Noise Data

Synthetic noise is also used during pipeline development and augmentation.

The project includes procedurally generated:

* Engine/rotor drone
* Harmonic and wobbling low-frequency noise
* Impulsive burst trains
* Band-limited decaying noise bursts
* Siren-like signals

Synthetic noise was used to bootstrap and test the processing pipeline before/alongside real-world noise data.

---

## 6. Data Augmentation

The training pipeline can apply controlled augmentation and noise mixing, including:

* Compound multi-noise mixing
* Synthetic reverberation
* Gain variation
* Clipping variation
* Spectral-tilt variation
* Controlled SNR mixing
* SNR curriculum approximately from -15 dB to +20 dB

These augmentations increase the variety of acoustic conditions presented during model training.

---

## 7. Data Splitting and Leakage Prevention

Where speaker-labelled speech data is used, the project follows a speaker-grouped train/validation/test split.

The objective is to prevent the same speaker or recording from appearing across different splits.

This helps ensure that evaluation measures generalization to unseen speech rather than memorization of speakers or recordings.

---

## 8. Project Dataset Summary

| Category              | Dataset / Source                 | Role                                         |
| --------------------- | -------------------------------- | -------------------------------------------- |
| Clean speech          | FSDD                             | Additional clean-speech source               |
| Speech enhancement    | VoiceBank-DEMAND                 | Benchmark speech-enhancement dataset         |
| Speech                | AMPORA_V3_HARD_REALVOICE_DATASET | Project speech data                          |
| General noise         | MUSAN                            | Noise augmentation/experiments               |
| Acoustic augmentation | RIRS_NOISES                      | Room impulse responses/acoustic augmentation |
| Stationary noise      | Project real-world recordings    | Engine, fan, A/C noise                       |
| Non-stationary noise  | Project real-world recordings    | Helicopter, aircraft, chainsaw               |
| Transient noise       | SESA / edge-collected sources    | Gunfire, explosions, sirens                  |
| Synthetic noise       | AMPORA procedural generators     | Pipeline development and augmentation        |

---

## 9. Reproducibility

To reproduce training and evaluation, obtain the required public datasets separately and place them according to the directory structure expected by the training/evaluation notebooks.

The raw `AMPORA_V3_HARD_REALVOICE_DATASET` is not included in the repository because of its size and recording restrictions.

Do not commit large raw datasets, private recordings, or restricted source material to the GitHub repository.

The repository should contain the dataset preparation instructions, expected directory structure, metadata format, preprocessing scripts, and reproducible mixing/evaluation code.
