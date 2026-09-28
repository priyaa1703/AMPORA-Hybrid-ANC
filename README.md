# AMPORA — AI/ML-Enabled Hybrid Adaptive Noise Cancellation

**A dual-band hybrid speech-enhancement system for defence and mission-critical
communication: classical reference-assisted NLMS for stationary/low-frequency noise,
a neural DC-CRN for non-stationary and impulsive high-frequency noise, a rule-based
noise-state controller arbitrating between them, and a working real-time streaming
implementation — not just an offline experiment.**

Built against DRDO problem statement PS26052 (AI/ML-enabled ANC for defence
communications). See `AUDIT_REPORT.md` for the original scope audit and
`CHANGELOG_ADDITIONS.md` for a full history of what was added, when, and why.

## Team and repository structure

This repository combines two parts of one system (DRDO PS26052 — AI/ML-enabled ANC for defence communication):

| Folder | Part | Author |
|---|---|---|
| [`hardware_dsp_nlms/`](hardware_dsp_nlms/) | **Hardware DSP implementation track**: staged Python reference of the reference-assisted NLMS chain (band-pass, 1.5 kHz split, NLMS, adaptation gating, double-talk handling) | [@Vincili2005](https://github.com/Vincili2005) |
| `/` (`src/`, `scripts/`, `data/`, `models/`, `results/`) | **AI/ML hybrid ANC**: DC-CRN high-band model, dataset pipeline, training, evaluation, real-time streaming pipeline | [@priyaa1703](https://github.com/priyaa1703) |

Her full commit history is preserved in this repo (imported from `Vincili2005/ANC_python`).
See `hardware_dsp_nlms/STAGES.md` for what each stage script does and how it maps onto the low-band branch of the AI/ML system.

## Results snapshot (measured, not asserted)

Confidence-gated hybrid inference (`scripts/render_sih_demo.py`), evaluated across a
range of input SNR levels on this repo's bundled real speech + real noise (including
real gunfire/explosion clips — see `ATTRIBUTIONS.md`). Full table, honest caveats, and
exact reproduction command: `results/sih_demo_reference/README.md`.

| Input SNR | ΔSNR | Enhanced STOI | Enhanced PESQ | vs. PS target (SNR>15dB, STOI>0.85, PESQ>2.5) |
|---|---|---|---|---|
| -10 dB (harsh) | **+8.52 dB** | 0.726 | 1.149 | Real improvement; targets not yet reached at this difficulty |
| 0 dB | +0.96 dB | 0.778 | 1.159 | — |
| +10 dB | — | **0.867** ✅ | 1.366 | **STOI target met** |
| +15 dB (mild) | — | **0.950** ✅ | 1.784 | **STOI target met**; SNR/PESQ approaching target |

**What this honestly shows:** real, positive noise cancellation in hard conditions;
the intelligibility (STOI) target met at moderate-to-high input SNR; PESQ still short of
target across the board — closing that gap needs a larger/more varied training corpus
(currently 6 speakers, ~450 augmented pairs — see `DATASETS.md`), not further tuning of
this checkpoint. Nothing here is presented as a final defence-grade result — see
"Important implementation boundary" below.

## What's real vs. what's still a gap

| Capability | Status |
|---|---|
| Dual-band hybrid architecture (NLMS + DC-CRN) | ✅ Implemented, `src/hybrid.py` |
| Rule-based noise-state controller (spectral flux + kurtosis, hysteresis) | ✅ Implemented, `src/detector/` |
| Dataset pipeline: synthesis, augmentation, real bundled audio | ✅ `src/augmentation/`, `data/raw/*_real/` |
| Training + evaluation (SNR/SI-SDR/STOI/PESQ + SegSNR/impulsive-only) | ✅ `scripts/train.py`, `scripts/evaluate.py` |
| **Real-time streaming inference** (mic→speaker or file, measured RTF/latency) | ✅ `src/realtime/`, `scripts/run_realtime.py` |
| Confidence-gated blending (do real cleanup when noisy, no harm when clean) | ✅ `src/realtime/confidence_gate.py` |
| ONNX export + INT8 quantization | ✅ `models/*.onnx` |
| Meeting PS's absolute SNR/STOI/PESQ targets at all operating points | ❌ Needs a production-scale defence-noise dataset, not more tuning |
| TensorRT conversion / physical Jetson deployment | ❌ Not attempted — no hardware available |
| Training on licensed defence-specific gunfire/artillery recordings | ⚠️ Partial — real (non-defence-specific) gunfire/explosion/helicopter clips used; see `DATASETS.md` for the licensed sources still needed |

## Important implementation boundary

**The originally-shipped ONNX artifacts (`models/ampora_lite_dccrn_*.onnx`) implement
only the high-band branch** and were not retrained during this project. Since then, this
repo has grown a complete software low-band NLMS branch, full band recombination
(`src/hybrid.py::hybrid_with_reference`), a confidence-gated blend of both branches, and
a real-time streaming implementation of the whole pipeline (`src/realtime/`) — none of
that is a physical hardware claim. What is still genuinely missing: a final
embedded/FPGA/Jetson implementation (no such hardware was available while building this),
and training on a production-scale, licensed defence-specific noise corpus rather than
the small bundled/synthetic dataset described in `DATASETS.md`.

Therefore, results from the current checkpoints must not be described as a validated,
production-ready defence ANC system — see the "Results snapshot" table above for exactly
what is and isn't measured.

## Software-complete project path

The completion path is intentionally software/ML-focused for Google Colab. It does not require FPGA, Raspberry Pi, or other hardware to generate the core experimental evidence. The repository now includes classical baselines, a software reference-assisted low-band NLMS branch, high-band DC-CRN inference, hybrid recombination, unseen-noise evaluation hooks, plotting, ONNX checks, and a Colab workflow.

The reference-assisted NLMS experiment uses the known injected noise waveform as the simulated reference signal. This is clearly labelled as a software experiment and must not be described as a physical microphone result.

## Current architecture

```text
Input audio
   |
   +--> 80 Hz–1.5 kHz --> external NLMS branch (not included)
   |
   +--> 1.5 kHz–7.9 kHz --> STFT --> DC-CRN --> complex residual mask --> iSTFT
                                      |
                                      +--> high-band enhanced output
```

The model predicts a two-channel real/imaginary residual mask:

`Y = X × (1 + M)`

with a bounded `tanh` mask.

## Existing artifact audit

The uploaded repository originally contained:

- `notebooks/AMPORA_HIGHBAND_DCCRN.ipynb`
- FP32 and INT8 ONNX artifacts
- `results/training_history.csv`
- `DATASETS.md`
- checkpoint documentation

The original training history contains 20 epochs. The lowest validation loss occurs at
**row epoch 18** (the notebook's zero-based checkpoint index is 17), with validation
loss `1.0613255869` and validation ΔSI-SDR `-1.4822678405 dB`. This is a measured
historical result, not an enhancement claim.

## Important findings from the original implementation

1. It has train/validation splitting but no independent test split.
2. Noise is mostly generated on the fly, so there is no held-out unseen-noise experiment.
3. The external noise files are available to both train and validation sampling.
4. Unknown speech filenames are grouped per recording, which avoids some leakage but does
   not prove speaker-disjointness.
5. The original evaluation is high-band only. Standard full-band STOI/PESQ should not be
   presented as final system metrics until the low-band branch is recombined.
6. The original notebook uses `center=True` STFT and symmetric temporal convolution padding.
   Despite the unidirectional GRU, that is **not sufficient to claim a strictly causal model**.
7. The previous upper cutoff was exactly 8 kHz at a 16 kHz sample rate. The reproducible
   configuration now uses 7.9 kHz to leave a transition margin.
8. The original `load_training_checkpoint(path)` ignores its `path` argument and always
   loads `BEST_MODEL`; the new standalone `scripts/train.py` does not use that buggy loader.
9. The original training configuration declares SNR down to -15 dB, but the first five
   epochs and the final curriculum use different ranges. The new pipeline records SNR
   explicitly in manifests instead of relying on an implicit curriculum.

## Repository layout

```text
configs/
  sample_rate.yaml
  model.yaml
  detector_thresholds.yaml
  streaming.yaml            # NEW: real-time chunking/latency defaults

data/
  README.md
  raw/
    speech_real/             # NEW: 169 real, CC-licensed speech segments (6 speakers)
    noise_real/               # NEW: 10 real, CC-licensed noise clips (stationary/nonstationary/transient)
  processed_real/             # NEW: manifests already generated from the above

scripts/
  download_datasets.py
  prepare_dataset.py
  synthesize_pairs.py
  synthesize_pairs_v2.py    # NEW: stronger/compound-noise dataset generator
  train.py                   # now supports --train-pairs-csv/--val-pairs-csv too
  infer_checkpoint.py       # NEW: checkpoint -> demo-ready clean/noisy/enhanced audio
  render_sih_demo.py        # NEW: confidence-gated SNR sweep + demo clips + summary.md
  evaluate.py                # now supports --extended (SegSNR etc.)
  verify_onnx.py
  benchmark_latency.py
  run_realtime.py           # NEW: live mic <-> speaker, or file-simulated, real-time ANC

src/
  preprocessing/
  detector/
  models/
  evaluation/
    extended_metrics.py    # NEW: SegSNR, freq-weighted SegSNR, impulsive-only Delta-SI-SDR
  augmentation/             # NEW: synthetic transients/reverb + stronger-dataset pipeline
  realtime/                 # NEW: causal/stateful streaming DSP + AI inference
    confidence_gate.py      # NEW: SNR-aware blend so enhancement helps, never over-processes

tests/
results/
hardware/
models/
notebooks/
```

## Dataset and leakage control

See `DATASETS.md`.

The reproducible pipeline creates speaker/recording-disjoint train/validation/test speech
manifests and stores noise category and target SNR as metadata.

The recommended evaluation must include:

- seen-noise test
- unseen-noise test
- stationary
- non-stationary
- transient
- mixed
- multiple SNR levels

## Metrics

For paired clean/noisy/enhanced audio, the evaluation pipeline reports:

- SNR
- ΔSNR
- SI-SDR
- ΔSI-SDR

STOI/PESQ are available in the evaluator for **full-band recombined audio**. The current
high-band-only artifact is not treated as a final full-band perceptual result.

Running `scripts/evaluate.py --extended` additionally reports (see
`src/evaluation/extended_metrics.py` and "Extended output parameters" below):

- SegSNR / frequency-weighted SegSNR
- Impulsive-only ΔSI-SDR (performance specifically on frames the detector flags as transient)

No new performance numbers are claimed until the pipeline is run on the actual held-out
test set.

## Noise/event detector

The detector is intentionally DSP-based rather than another neural network:

```text
Audio
  ↓
Spectral Flux + Kurtosis + optional SNR estimate
  ↓
Noise/Event State Controller
  ↓
STATIONARY / COMPLEX / TRANSIENT / MIXED-UNCERTAIN
```

The controller uses smoothing/hysteresis. Thresholds in
`configs/detector_thresholds.yaml` are configuration placeholders, not measured optimums.

It should be described as detecting acoustic conditions, not identifying a specific event.

## ONNX and quantization

The repository contains the existing FP32 and INT8 artifacts. Use:

```bash
python scripts/verify_onnx.py models/ampora_lite_dccrn_fp32.onnx models/ampora_lite_dccrn_int8.onnx
```

The current files are approximately 1.02 MB (FP32) and 0.37 MB (INT8). File size alone
does not establish quality preservation.

Use `benchmark_latency.py` for model-only CPU inference. It must not be confused with
end-to-end audio latency.

## Current measured vs target status

| Quantity | Status |
|---|---|
| Historical validation ΔSI-SDR | **Measured: -1.4823 dB** |
| Independent test-set result | **Not yet measured** |
| Unseen-noise result | **Not yet measured** |
| Full-band STOI | **Not yet measured** |
| Full-band PESQ | **Not yet measured** |
| FP32 vs INT8 quality comparison | **Not yet measured** |
| Model-only inference latency | **Not yet measured** (run `scripts/benchmark_latency.py`) |
| End-to-end **streaming/real-time** latency and RTF | **Measured in software** (see `results/streaming_latency_report.md`, generated by `scripts/run_realtime.py benchmark`) -- this is a generic-CPU onnxruntime measurement, **not** an embedded/FPGA/Jetson measurement |
| FPGA / Jetson hardware performance | **Not yet measured** |
| Energy consumption | **Not yet measured** |

## Reproducibility workflow

```bash
pip install -r requirements.txt

# Option 1: use the real bundled audio (fastest path to a genuine, non-synthetic result)
python scripts/synthesize_pairs_v2.py --speech-csv data/processed_real/train_speech.csv \
  --noise-csv data/processed_real/noise.csv --out data/processed_real/pairs_v2_train
python scripts/synthesize_pairs_v2.py --speech-csv data/processed_real/val_speech.csv \
  --noise-csv data/processed_real/noise.csv --out data/processed_real/pairs_v2_val
python scripts/train.py --train-pairs-csv data/processed_real/pairs_v2_train/pairs_v2.csv \
  --val-pairs-csv data/processed_real/pairs_v2_val/pairs_v2.csv --epochs 40 --out models/trained_real_v1

# Option 2: your own data
python scripts/prepare_dataset.py --speech data/raw/speech --noise data/raw/... --out data/processed
python scripts/synthesize_pairs.py ...
python scripts/synthesize_pairs_v2.py --synthetic-noise-only --speech-csv ...  # stronger dataset, no downloads
python scripts/train.py --train-speech ... --val-speech ... --noise ...        # online-mixing mode

# Either way, from a checkpoint:
python scripts/infer_checkpoint.py --checkpoint models/trained_real_v1/best.pt \
  --speech-csv data/processed_real/val_speech.csv --noise-csv data/processed_real/noise.csv \
  --out results/demo
python scripts/evaluate.py results/demo/demo_manifest.csv --out results/demo/metrics.csv --extended --high-band
python scripts/verify_onnx.py ...
python scripts/benchmark_latency.py ...
python scripts/run_realtime.py benchmark --model models/ampora_lite_dccrn_fp32.onnx   # real-time RTF/latency
python scripts/run_realtime.py file --in noisy.wav --out enhanced.wav --model ...      # real-time demo
pytest -q
```

The existing notebook remains as the original training/export artifact; `notebooks/AMPORA_COLAB_COMPLETE.ipynb` is the recommended end-to-end Google Colab workflow, and the new scripts provide a cleaner reproducibility path without deleting the original work.

## Limitations

- The **physical** low-band reference-microphone path (real hardware mics/ADC) is not
  implemented here; a **software** reference-assisted NLMS branch is provided for algorithmic
  comparison, hybrid experiments, and now also for live/streaming use (see "Real-time
  streaming inference" below) when a reference channel is supplied (e.g. a second wav channel,
  or a second sound-card input).
- The current DC-CRN is not yet proven strictly causal (symmetric time-axis conv padding);
  `src/realtime/` works around this with a bounded causal sliding-window context rather than
  by changing the trained model -- see "Real-time streaming inference" for the exact tradeoff.
- No final full-band enhancement claim is made.
- No embedded/FPGA timing or power claim is made. The `results/streaming_latency_report.md`
  numbers are generic-CPU onnxruntime numbers, not embedded-hardware numbers.
- The public repository does not contain private raw recordings or checkpoints.
- Full training cannot be reproduced from this archive alone without obtaining the required
  datasets/checkpoints.
- The synthetic noise generators in `src/augmentation/synthetic_transients.py` are
  procedural DSP approximations (band-limited decaying noise bursts, harmonic drone models),
  not recordings of real defence equipment; they exist to bootstrap dataset variety when a
  licensed corpus isn't yet available, and should be supplemented with real, properly licensed
  recordings for a production system (see `DATASETS.md`).

## Real bundled audio (NEW)

`data/raw/speech_real/` (169 segments, 6 distinct speakers) and `data/raw/noise_real/`
(10 clips across all 3 required categories -- stationary, non-stationary, transient)
contain real, properly Creative-Commons-licensed audio bundled directly in this
repository -- not synthetic tones, not a download step you have to chase down. See
`ATTRIBUTIONS.md` for exact per-file sources and licenses. `data/processed_real/`
already has the leakage-safe train/val/test manifests generated from this data via
`prepare_dataset.py`, so you can go straight to dataset generation / training without
re-running that step (though it's fully reproducible if you do: same seed, same split).

**Read this before using it in a demo:** these are generic recordings (a civilian
helicopter, a construction jackhammer, ordinary truck engines, a firework/demolition
-style explosion) picked to exercise each PS-required noise category with *real*
acoustic complexity -- they are not gunfire, artillery, or military-vehicle recordings,
and there are only 10 noise clips and 6 speakers. This is enough to prove the whole
pipeline works on real (not purely synthetic) audio and produces genuinely improved
SI-SDR/SegSNR, but it is not a validated defence-scenario result -- say so explicitly
if you use it in a demo. `DATASETS.md` lists where to get the real thing.

```bash
# Generate a stronger, augmented dataset from the bundled real audio:
python scripts/synthesize_pairs_v2.py \
  --speech-csv data/processed_real/train_speech.csv \
  --noise-csv data/processed_real/noise.csv \
  --out data/processed_real/pairs_v2_train \
  --snr-min -10 --snr-max 15 --per-speech 2

python scripts/synthesize_pairs_v2.py \
  --speech-csv data/processed_real/val_speech.csv \
  --noise-csv data/processed_real/noise.csv \
  --out data/processed_real/pairs_v2_val \
  --snr-min -10 --snr-max 15 --per-speech 2

# Train directly on those generated pairs (see "Two ways to train" below):
python scripts/train.py \
  --train-pairs-csv data/processed_real/pairs_v2_train/pairs_v2.csv \
  --val-pairs-csv data/processed_real/pairs_v2_val/pairs_v2.csv \
  --epochs 40 --out models/trained_real_v1

# Turn the checkpoint into demo-ready before/after audio + a manifest:
python scripts/infer_checkpoint.py \
  --checkpoint models/trained_real_v1/best.pt \
  --speech-csv data/processed_real/val_speech.csv \
  --noise-csv data/processed_real/noise.csv \
  --out results/demo --n-examples 8

# Score it, including the extended metrics:
python scripts/evaluate.py results/demo/demo_manifest.csv \
  --out results/demo/metrics.csv --extended --high-band
```

## Two ways to train (NEW: `--train-pairs-csv`)

`scripts/train.py` originally only supported "online" mixing -- given a speech CSV and
a noise CSV, it randomly mixes a fresh clean+noise pair every batch at a fixed SNR
list. That mode still works (`--train-speech`/`--val-speech`/`--noise`) and is the
right choice for maximum training-time variety with limited disk. It had no way to
reproduce `synthesize_pairs_v2.py`'s stronger augmentation (compound noise, reverb,
clipping, wider SNR range), since those pairs are pre-generated once and written to
disk rather than mixed inline. `--train-pairs-csv`/`--val-pairs-csv` is the new second
mode: point it at a `pairs_v2.csv` (or a plain `synthesize_pairs.py` manifest -- same
column names) and it trains directly on those pre-generated pairs instead. Use
whichever matches what you generated; the two modes are mutually exclusive (the script
errors clearly if you mix arguments from both).

`scripts/train.py` also now writes `<out>/history.json` (per-epoch train/val loss and
ΔSI-SDR) alongside the checkpoints, so you can plot a training curve without having to
copy numbers out of the console log by hand.

`scripts/infer_checkpoint.py` (NEW) is the missing "checkpoint -> demo-ready audio"
step: it loads a trained checkpoint, runs it on held-out speech+noise at a chosen SNR,
and writes clean/noisy/enhanced `.wav` triples plus a manifest that
`scripts/evaluate.py` can score directly.

## Confidence-gated inference (NEW)

A model trained on a small dataset (this project's bundled real data: 6 speakers, ~450
augmented pairs) tends to apply a roughly *fixed* amount of "cleanup" regardless of how
noisy the input actually is. That helps when the input is very noisy but actively **hurts**
already-clean input -- this was observed directly during interactive evaluation (see
`CHANGELOG_ADDITIONS.md`): at +15dB input SNR (already fairly clean), the unblended
model output was *worse* than doing nothing.

`src/realtime/confidence_gate.py` fixes this without any retraining: it estimates the
*current* SNR from the reference-mic signal and blends the fully-enhanced output against
the raw input accordingly -- near-full trust in the model when it's genuinely noisy,
tapering down to mostly-leave-it-alone as the input gets cleaner. This is wired into
both:

- **Offline demo rendering**: `scripts/render_sih_demo.py` -- runs an SNR sweep
  (`results/*/sweep/metrics_by_snr.csv`), renders a handful of presentation-ready demo
  clips (a hard/noisy case, a "do no harm on clean input" case, and a real
  transient/impulsive case if the noise manifest has one), and writes a `summary.md`
  with headline numbers ready to paste into slides.
- **The real-time pipeline**: `RealTimeANC` (`src/realtime/streaming_pipeline.py`) now
  applies the same gate live, using an exponentially-smoothed running estimate of
  primary/reference hop levels (`use_confidence_gate=True` by default; inspect
  `anc.last_alpha` after calling `.process()` to see the current gate value).

```bash
python scripts/render_sih_demo.py \
  --checkpoint models/trained_real_v2/best.pt \
  --speech-csv data/processed_real/val_speech.csv data/processed_real/test_speech.csv \
  --noise-csv data/processed_real/noise.csv \
  --out results/sih_demo
```

See `results/sih_demo_reference/README.md` for a real measured table from this project
and how to read it honestly (including where the PS's STOI/PESQ/SNR targets are and
aren't met yet).

## Real-time streaming inference (NEW)

The original repository only had *offline* helpers that process a whole array at once
(`src/hybrid.py`, `scripts/evaluate.py`). `src/realtime/` adds a genuine chunk-wise,
stateful pipeline that can take live input, process it, and produce live output --
the "given input in real time, process it and give output like that" workflow.

```text
src/realtime/
  streaming_dsp.py        # CausalBandSplitter (stateful IIR), StreamingNLMS,
                           # OverlapAddSynthesizer (WOLA reconstruction)
  streaming_highband.py   # StreamingHighBandEnhancer: causal sliding-window
                           # wrapper around the existing DC-CRN ONNX model
  streaming_pipeline.py   # RealTimeANC: orchestrates both bands + delay-aligns
                           # the near-zero-latency low band with the AI high band
```

Run it three ways with `scripts/run_realtime.py`:

```bash
# 1) Simulate real-time streaming on any machine, no audio hardware needed:
python scripts/run_realtime.py file \
  --in path/to/noisy.wav --out path/to/enhanced.wav \
  --reference-file path/to/reference_mic.wav \
  --model models/ampora_lite_dccrn_int8.onnx \
  --context-frames 8 --infer-every 2

# 2) Live microphone -> live speaker (needs `pip install sounddevice` + PortAudio):
python scripts/run_realtime.py mic --model models/ampora_lite_dccrn_int8.onnx \
  --context-frames 8 --infer-every 2 --reference-channel

# 3) Measure real-time factor (RTF) / latency across settings on your machine:
python scripts/run_realtime.py benchmark --model models/ampora_lite_dccrn_fp32.onnx
```

**How the causality problem is handled.** The trained DC-CRN (see "Limitations") is not
strictly causal, so it cannot be fed one STFT frame at a time and produce a correct
per-frame mask on its own. `StreamingHighBandEnhancer` instead keeps a rolling buffer of
the last `context_frames` STFT frames (all *past* audio only, no lookahead) and re-runs
the model on that window every hop, keeping only the newest frame's output mask. This adds
`context_frames * hop_length` samples of algorithmic latency but needs no retraining. An
`infer_every_n_hops` setting additionally lets the network run less often than every hop
(holding the mask steady in between) to trade a little temporal resolution in the *mask*
for a large, predictable cut in compute -- audio is still produced every hop via
overlap-add; nothing is dropped.

If no ONNX model/runtime is available at all, `StreamingHighBandEnhancer` degrades to a
dependency-free causal spectral-subtraction fallback rather than erroring out or passing
raw noise straight through -- the pipeline always produces bounded, finite output.

**Measured, not assumed, real-time performance.** `scripts/run_realtime.py benchmark`
writes `results/streaming_latency_report.md` with the actual measured algorithmic latency
and RTF (processing-time / audio-time; must be < 1.0 to keep up with real time) for several
`(context_frames, infer_every_n_hops)` settings **on whatever machine you run it on**. This
directly follows the existing `hardware/deployment_notes.md` note: *"For a real-time claim,
use a causal or explicitly bounded-lookahead front end and measure the resulting end-to-end
latency."* -- that is now implemented and measurable, not just described. Re-run the
benchmark on your actual target hardware (Jetson AGX Orin + TensorRT, per
`hardware/deployment_notes.md`) before making any hardware real-time claim; the CPU numbers
here are a software feasibility check, not the target-hardware result.

## Stronger dataset pipeline (NEW)

`scripts/synthesize_pairs_v2.py` (backed by `src/augmentation/`) generates a noticeably
harder/more realistic dataset than `scripts/synthesize_pairs.py`:

| | `synthesize_pairs.py` (original) | `synthesize_pairs_v2.py` (new) |
|---|---|---|
| SNR range | -5 .. +15 dB | **-15 .. +20 dB**, continuous (not a fixed list) |
| Noise per sample | 1 clip | **compound**: base noise + optional 2nd noise clip + optional synthetic impulsive burst train |
| Reverberation | none | optional synthetic room/field/vehicle-cabin reverb (`src/augmentation/reverb.py`) |
| Channel realism | none | optional clipping, gain jitter, spectral-tilt jitter |
| Reference signal | not generated | isolated `noise_reference.wav` per sample, ready for the low-band NLMS branch |
| Noise source | requires a downloaded corpus | can bootstrap **with zero downloads** via `--synthetic-noise-only`, using procedural generators for engine/rotor drone, impulsive bursts, and sirens (`src/augmentation/synthetic_transients.py`) |

```bash
# With a real noise corpus (see DATASETS.md for sources):
python scripts/synthesize_pairs_v2.py \
  --speech-csv data/processed/train_speech.csv \
  --noise-csv data/processed/noise.csv \
  --out data/processed/pairs_v2_train \
  --snr-min -15 --snr-max 20 --per-speech 4

# Or, to bootstrap immediately with no downloads at all:
python scripts/synthesize_pairs_v2.py \
  --speech-csv data/processed/train_speech.csv \
  --out data/processed/pairs_v2_bootstrap \
  --synthetic-noise-only --per-speech 4
```

The synthetic generators are plain DSP (band-limited decaying noise bursts for
gunshot/artillery-*like* transients, a harmonic-comb-plus-wobble model for
rotor/engine drone, a swept tone for sirens) -- useful for bootstrapping dataset
*volume and variety* while you obtain a properly licensed defence-noise corpus; they
are not a substitute for real recordings in a fielded system (see `DATASETS.md`).

## Extended output parameters (NEW)

The PS asks for SNR > 15 dB, STOI > 0.85, PESQ > 2.5. Those three numbers say nothing
about whether the system holds up specifically during impulsive events, or whether it
can actually run in real time. `scripts/evaluate.py --extended` (backed by
`src/evaluation/extended_metrics.py`) adds:

- **SegSNR / frequency-weighted SegSNR** -- segment-level SNR, far less dominated by a
  few loud/quiet frames than a single whole-utterance SNR number.
- **Impulsive-only ΔSI-SDR** -- ΔSI-SDR computed *only* on the frames this project's own
  detector (`src/detector`) flags as transient, isolating performance on exactly the
  gunshot/artillery-type events the PS is about, rather than averaging them away.
- **Real-Time Factor (RTF) and algorithmic latency** -- from `RealTimeANC`, because
  "good offline metrics" and "runs in real time" are two separate, both-necessary claims.
- **WER proxy (optional)** -- `optional_wer()` runs a local Whisper ASR transcript against
  a known reference transcript if `openai-whisper` is installed; skipped (not faked) if not.
  STOI/PESQ are perceptual-*model* proxies for intelligibility; an actual ASR transcript is
  a stronger, task-grounded check for a system whose whole point is intelligible speech.

## What makes this hybrid design different (uniqueness)

Relative to a generic "train a speech-enhancement network on noisy/clean pairs" project:

1. **Dual-band hybrid, not a single end-to-end network.** A classical, cheap,
   easily-analysed reference-assisted NLMS filter handles the low band (where stationary
   engine/rotor energy concentrates and a reference mic gives a strong correlated cue),
   while the neural DC-CRN focuses capacity on the harder high band. This keeps the AI
   model small (the shipped ONNX artifacts are ~0.4-1 MB) and its failure modes easier to
   reason about than a single black-box full-band network.
2. **A rule-based noise/event controller in the loop, not just a filter cascade**
   (`src/detector/noise_state_controller.py`): spectral-flux + kurtosis (+ optional SNR)
   drive an explicit stationary / complex / transient / mixed-uncertain state machine with
   hysteresis, which in turn tells the NLMS branch to freeze/slow adaptation and the AI
   branch to prioritize exactly when a transient (e.g. gunshot) is detected -- an explicit,
   inspectable adaptive-strength policy rather than a fixed processing chain.
3. **Causality and real-time latency are treated as first-class, measured constraints**,
   not an afterthought bolted on after offline training -- see "Real-time streaming
   inference" above, including an honest accounting of where the trained model is and isn't
   strictly causal and what that costs in added latency.
4. **Impulsive-specific evaluation**, not just whole-file averages -- the "Extended output
   parameters" impulsive-only ΔSI-SDR metric directly measures the PS's core differentiator
   (impulsive/defence noise) rather than being diluted into an overall SNR number dominated
   by the (comparatively easy) stationary segments.
5. **A synthetic, zero-download noise bootstrap path** (`--synthetic-noise-only`) that lets
   the whole pipeline -- dataset, training, evaluation, real-time demo -- be exercised
   end-to-end before a licensed defence-noise corpus is in hand, while remaining clearly
   labelled as a bootstrap, not a substitute for real recordings.



Add the project's actual license before public release. Do not imply that dataset licenses
are covered by the project code license.


## Persistent Google Drive checkpointing

For Colab training, checkpoints are saved to:

`My Drive/AMPORA-Hybrid-ANC/checkpoints/`

A checkpoint is written after every completed epoch and contains the model state,
optimizer state, scheduler state, epoch number, history, and configuration.
`latest.json` points to the most recent completed epoch.

If a Colab runtime disconnects, reconnect and rerun the setup cells. Mount Drive,
recreate the model/optimizer/scheduler, then call `load_latest_checkpoint(...)`.
Training can continue from `epoch + 1` without losing completed epochs.

The `/content` filesystem is temporary; Google Drive is the persistent storage used
for training checkpoints.
