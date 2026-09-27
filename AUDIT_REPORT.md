# AMPORA repository audit — 2026-09-25

## Scope
Audited the uploaded archive `AMPORA-Hybrid-ANC (2).zip` before adding files.

## Original archive contents
- `README.md`
- `DATASETS.md`
- `notebooks/AMPORA_HIGHBAND_DCCRN.ipynb`
- `models/ampora_lite_dccrn_fp32.onnx`
- `models/ampora_lite_dccrn_int8.onnx`
- `results/training_history.csv`
- `checkpoints/README.md`

## Verified implementation facts

### Dataset and mixing
The notebook loads:
- project clean speech from `clean_speech/`
- project noise from `noise_original/` and `noise_impulsive/`
- optional FSDD recordings

It generates synthetic noise types in code: engine, rotor, wind, radio, hum, babble,
impulse, pink, white and mixed. Training SNR is sampled from a curriculum, with later
epochs reaching -15 dB to +10 dB.

The original validation set is speech-disjoint when speaker IDs can be inferred, but
there is no independent test split and no held-out unseen-noise protocol.

### Signal processing
Original model:
- 16 kHz
- 512-point FFT
- 320-sample window
- 128-sample hop
- nominal high band 1.5–8 kHz
- bins 48–256 inclusive = 209 bins
- real/imaginary input channels
- base channels 16
- unidirectional GRU, hidden size 64
- complex residual mask `Y = X * (1 + M)`

The notebook's high-band output is high-band only; it is not recombined with the
low-band NLMS branch.

The original notebook uses centered STFT and symmetric temporal convolution padding.
Therefore the existing model is not proven strictly causal despite the unidirectional GRU.

### Training
The original loss combines:
- multi-resolution STFT magnitude terms
- complex STFT L1 loss
- SI-SDR loss

The original run contains 20 recorded epochs.

### Historical result
From the committed training history, the minimum validation loss is at recorded epoch 18:
- validation loss: 1.0613255869
- validation ΔSI-SDR: -1.4822678405 dB

This is retained as a historical measured result. No new positive performance number has
been invented.

### ONNX
The archive contains FP32 and INT8 ONNX models.
The original notebook verifies loading and output shapes, but does not establish:
- FP32/INT8 quality equivalence
- FP32/INT8 latency equivalence
- full end-to-end audio latency
- memory usage
- FPGA performance

## Changes added
- reproducible configuration files
- dataset preparation and leakage-resistant split script
- deterministic clean/noise mixing script with metadata
- public dataset documentation
- DSP event/noise detector components
- hysteresis controller
- evaluation script
- ONNX verification script
- model-only latency benchmark
- standalone training script using the existing DC-CRN architecture
- automated tests
- hardware timing/deployment documentation
- explicit measured-vs-target status in README
- explicit limitation that the current artifact is high-band only

## Validation performed on the strengthened archive
- DC-CRN forward shape test: passed
- automated unit tests: 3 passed
- no full model retraining was claimed because the uploaded archive does not contain the
  required raw training data or PyTorch checkpoint.

## Still required before claiming final performance
1. Obtain the actual training and held-out test datasets under their respective licenses.
2. Generate speaker/recording-disjoint train/validation/test manifests.
3. Reserve noise sources for an unseen-noise test.
4. Train/fine-tune the model.
5. Evaluate noisy vs enhanced on stationary, non-stationary, transient and mixed noise.
6. Evaluate across the configured SNRs.
7. Recombine the high-band output with a real low-band branch before claiming full-band
   STOI/PESQ.
8. Export/re-quantize the trained model and measure FP32 vs INT8 quality and latency.
9. Benchmark the complete pipeline on the actual embedded target.

No value above should be presented as an achieved final system result unless the corresponding
experiment is actually run.
