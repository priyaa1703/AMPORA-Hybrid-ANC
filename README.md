# AMPORA — AI/ML-Enabled Hybrid Adaptive Noise Cancellation

AMPORA is a software-based hybrid speech-enhancement project developed for **DRDO PS26052 — AI/ML-enabled ANC for defence communication**.

The repository contains the AI/ML speech-enhancement pipeline, including the DC-CRN model, dataset preparation, training, evaluation, and real-time audio processing.

## Repository Structure

| Folder                                                   | Part                                                                                             | Author       |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------ |
| `hardware_dsp_nlms/`                                     | Reference implementation of the NLMS-based signal-processing stage used for software experiments | @Vincili2005 |
| `/` (`src/`, `scripts/`, `data/`, `models/`, `results/`) | AI/ML hybrid ANC: DC-CRN model, dataset pipeline, training, evaluation, and real-time streaming  | @priyaa1703  |

The `hardware_dsp_nlms/` work was integrated into this repository from `Vincili2005/ANC_python`, with its commit history preserved.

See [`hardware_dsp_nlms/STAGES.md`](hardware_dsp_nlms/STAGES.md) for the stages included in the NLMS reference implementation.

---

## Project Overview

The software combines adaptive filtering and deep-learning-based speech enhancement.

```text
Input Audio
     |
     +----------------------+
     |                      |
     v                      v
Low-Frequency           High-Frequency
Processing              Processing
     |                      |
    NLMS                  DC-CRN
     |                      |
     +----------+-----------+
                |
                v
        Enhanced Speech
```

The low-frequency branch uses NLMS-based adaptive filtering, while the high-frequency branch uses a DC-CRN model for speech enhancement.

---

## Main Components

* NLMS adaptive filtering
* DC-CRN speech-enhancement model
* STFT / iSTFT processing
* Dataset generation and preprocessing
* Model training
* Offline inference
* SNR-based evaluation
* STOI and PESQ evaluation
* ONNX model export
* INT8 model support
* Real-time software streaming

---

## Repository Layout

```text
AMPORA-Hybrid-ANC/
│
├── hardware_dsp_nlms/
│   └── STAGES.md
│
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── evaluation/
│   └── realtime/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer_checkpoint.py
│   ├── synthesize_pairs.py
│   ├── synthesize_pairs_v2.py
│   ├── verify_onnx.py
│   ├── benchmark_latency.py
│   └── run_realtime.py
│
├── data/
│   └── processed_real/
│
├── models/
│   ├── ampora_lite_dccrn_fp32.onnx
│   └── ampora_lite_dccrn_int8.onnx
│
├── results/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── DATASETS.md
├── ATTRIBUTIONS.md
└── README.md
```

---

## DC-CRN Model

The neural speech-enhancement branch operates on the high-frequency portion of the input signal.

```text
Input Audio
     |
     v
Band Selection
     |
     v
STFT
     |
     v
DC-CRN
     |
     v
Complex Mask
     |
     v
iSTFT
     |
     v
Enhanced Audio
```

The model predicts a complex residual mask that is applied to the input spectrogram.

---

## Dataset

The repository includes scripts for preparing and generating speech/noise mixtures for training and evaluation.

The pipeline supports:

* speech preprocessing
* noise preprocessing
* SNR-controlled mixtures
* training/validation pair generation
* real-audio evaluation
* reproducible dataset manifests

Dataset information and attribution are provided in:

```text
DATASETS.md
ATTRIBUTIONS.md
```

---

## Training

Install the required packages:

```bash
pip install -r requirements.txt
```

Generate training pairs:

```bash
python scripts/synthesize_pairs_v2.py
```

Train the model:

```bash
python scripts/train.py
```

Training checkpoints and logs are stored under the configured model/output directories.

A Google Colab notebook is also provided under:

```text
notebooks/
```

---

## Evaluation

The project evaluates speech enhancement using objective metrics such as:

* SNR
* ΔSNR
* SI-SDR
* STOI
* PESQ
* SegSNR

Example:

```bash
python scripts/evaluate.py \
  results/demo/demo_manifest.csv \
  --out results/demo/metrics.csv
```

---

## ONNX Models

ONNX versions of the neural model are included for lightweight inference.

```text
models/
├── ampora_lite_dccrn_fp32.onnx
└── ampora_lite_dccrn_int8.onnx
```

The INT8 model provides a smaller representation for faster and more resource-efficient inference.

ONNX verification can be performed using:

```bash
python scripts/verify_onnx.py
```

---

## Real-Time Software Pipeline

The repository also contains a real-time software implementation.

The streaming pipeline handles:

```text
Audio Input
     |
     v
Streaming Preprocessing
     |
     v
Hybrid Enhancement
     |
     v
Audio Reconstruction
     |
     v
Enhanced Output
```

The real-time implementation is available under:

```text
src/realtime/
```

It can be tested with audio files and compatible microphone input.

Example:

```bash
python scripts/run_realtime.py
```

---

## Results

Example evaluation results from the current software experiments:

| Input SNR |     ΔSNR |  STOI |  PESQ |
| --------- | -------: | ----: | ----: |
| -10 dB    | +8.52 dB | 0.726 | 1.149 |
| 0 dB      | +0.96 dB | 0.778 | 1.159 |
| +10 dB    |        — | 0.867 | 1.366 |
| +15 dB    |        — | 0.950 | 1.784 |

These values correspond to the current software evaluation setup and are provided as experimental results rather than fixed system specifications.

More detailed results are available in the `results/` directory.

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
python scripts/synthesize_pairs.py
```

or:

```bash
python scripts/synthesize_pairs_v2.py
```

### 3. Train

```bash
python scripts/train.py
```

### 4. Evaluate

```bash
python scripts/evaluate.py
```

### 5. Verify ONNX models

```bash
python scripts/verify_onnx.py
```

### 6. Run real-time processing

```bash
python scripts/run_realtime.py
```

---

## Project Status

The repository currently focuses on the **software implementation and evaluation** of the hybrid ANC pipeline.

The main components available in the repository are:

* NLMS reference implementation
* DC-CRN speech enhancement
* Dataset preparation
* Model training
* Evaluation scripts
* ONNX models
* INT8 model
* Real-time software processing
* Reproducible experiment scripts

The project is intended to provide a software demonstration and evaluation of the proposed hybrid noise-cancellation approach.

---

## Team

**AI/ML Hybrid ANC:** @priyaa1703

**NLMS Reference Implementation:** @Vincili2005

The original NLMS implementation history has been preserved in the repository.
