# Dataset Sources

This document lists the speech and noise datasets used or referenced for the AMPORA high-band speech-enhancement workflow.

## Speech Data

### Free Spoken Digit Dataset (FSDD)

FSDD is used as an additional clean-speech source.

Repository:
https://github.com/Jakobovski/free-spoken-digit-dataset

## Speech Enhancement Data

### VoiceBank + DEMAND

VoiceBank-DEMAND is a commonly used speech-enhancement dataset containing clean speech and environmental noise.

Source:
https://flageval.baai.ac.cn/docs/en/audio/Speech-Enhancement/Speech-Enhancement.html

## Noise Data

### MUSAN

MUSAN provides speech, music and noise recordings that can be used for speech and noise-processing experiments.

Source:
https://www.openslr.org/17/

### RIRS_NOISES

RIRS_NOISES provides room impulse responses and related acoustic data for speech-processing experiments.

Source:
https://www.openslr.org/28/

## Project Dataset

The project also uses the provided:

`AMPORA_V3_HARD_REALVOICE_DATASET`

The raw project dataset is intentionally not included in this GitHub repository because of its size and to avoid publishing private recordings.

## Reproducibility

To reproduce training, obtain the required datasets separately and place them according to the directory structure expected by the notebook.

Do not commit large raw datasets or private recordings to the repository.
