# AMPORA – AI/ML Enabled Hybrid Adaptive Noise Cancellation

## Overview

AMPORA is a hybrid adaptive noise-cancellation architecture designed for real-time voice communication in challenging acoustic environments.

The system combines:
- NLMS adaptive filtering for the lower-frequency band
- Complex-domain DC-CRN neural processing for the higher-frequency band
- STFT / iSTFT based frequency-domain processing
- FP32 and INT8 ONNX model deployment
- Embedded / FPGA-oriented low-latency processing

The high-band neural processing covers approximately 1.5–8 kHz.

## System Architecture

Input Speech + Noise
→ Band Splitting
→ 80 Hz–1.5 kHz: NLMS
→ 1.5–8 kHz: STFT → Complex DC-CRN → Complex Mask → iSTFT
→ Band Recombination
→ Enhanced Speech

## High-Band Neural Model

The neural component uses a lightweight complex-domain DC-CRN architecture.

| Parameter | Value |
|---|---|
| Sampling rate | 16 kHz |
| High-band | 1.5–8 kHz |
| STFT NFFT | 512 |
| Window length | 320 |
| Hop length | 128 |
| Frequency resolution | 31.25 Hz |
| High-band bins | 48–256 |
| Base channels | 16 |
| GRU hidden size | 64 |
| GRU layers | 1 |
| GRU direction | Unidirectional |
| Mask | Complex residual mask |

The model operates on real and imaginary STFT components.

Complex residual mask:

Y = X × (1 + M)

where X is the noisy complex STFT, M is the predicted complex residual mask, and Y is the enhanced complex STFT.

## Repository Contents

```text
AMPORA-Hybrid-ANC/
├── README.md
├── notebooks/
│   └── AMPORA_HIGHBAND_DCCRN.ipynb
├── models/
│   ├── ampora_lite_dccrn_fp32.onnx
│   └── ampora_lite_dccrn_int8.onnx
├── results/
│   └── training_history.csv
└── checkpoints/
```

## Notebook

The complete training and export workflow is available in:

`notebooks/AMPORA_HIGHBAND_DCCRN.ipynb`

## Exported Models

### FP32 ONNX

`models/ampora_lite_dccrn_fp32.onnx`

Size: approximately 1.02 MB.

The exported FP32 ONNX model passed ONNX verification.

### INT8 ONNX

`models/ampora_lite_dccrn_int8.onnx`

Size: approximately 370 KB.

The INT8 model was successfully loaded and tested using ONNX Runtime with the expected output shape.

## Training Result

The best validation checkpoint occurred at epoch 17.

Measured validation result:

- Best validation loss: 1.0613255869
- Validation ΔSI-SDR: -1.4823 dB

The reported ΔSI-SDR value is the measured validation result from the current training run. It is not presented as an enhancement gain.

Training history is provided in:

`results/training_history.csv`

## ONNX Verification

The exported models were checked using ONNX / ONNX Runtime.

Input shape:

`(1, 2, 209, 161)`

Output shape:

`(1, 2, 209, 161)`

The two channels represent the real and imaginary components of the complex STFT representation.

## Dataset

The project uses clean speech and noise sources for training and validation.

Dataset information and source links should be documented in DATASETS.md.

Large raw datasets and private recordings are intentionally not included in this repository.

## Embedded Deployment Concept

The intended high-band deployment flow is:

Audio → STFT → INT8 DC-CRN → Complex Mask → iSTFT → High-band Enhanced Audio

The INT8 ONNX model is an intermediate deployment artifact. A final embedded controller binary depends on the target processor, vendor runtime and compiler/toolchain.

## Project Goals

- Low-latency voice enhancement
- Reduced computational load through band-wise processing
- Adaptive handling of stationary and non-stationary noise
- Lightweight neural inference
- FPGA / embedded deployment compatibility
- Improved speech intelligibility in difficult acoustic environments

## Reproducibility

1. Open the notebook in notebooks/.
2. Install the required Python packages.
3. Provide the required datasets.
4. Run the training and validation cells.
5. Export the FP32 ONNX model.
6. Perform INT8 quantization.
7. Verify inference using ONNX Runtime.

Do not upload large raw datasets or private recordings to the repository.

## License

Add the appropriate project or institutional license before public release.