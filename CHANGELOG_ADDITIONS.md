# Changelog: additions on top of the uploaded AMPORA-Hybrid-ANC-COLAB-DRIVE-RESUME archive

This file maps each item requested for this revision to exactly what was added/changed,
so nothing is buried in prose elsewhere. Nothing in the original archive was deleted;
the original scripts, notebooks, ONNX artifacts, and audit docs are untouched except for
the standalone-execution bugfix noted at the bottom.

## 1. "if anything is missing on the DL part as of my workflow, add that too"

- `src/augmentation/` -- a proper augmentation stage was missing between raw
  clean/noise audio and the training manifest; added compound noise mixing, synthetic
  reverberation, clipping/gain/spectral-tilt jitter (`pipeline.py`), plus procedural
  noise generators for when a licensed corpus isn't available yet (`synthetic_transients.py`,
  `reverb.py`).
- `src/realtime/` -- the repository had offline (whole-array) DSP/inference only; there
  was no stateful/streaming path at all, so a trained model could not actually be run on
  live or chunked input. Added causal band-splitting, streaming NLMS, a windowed
  overlap-add reconstructor, and a causal sliding-window wrapper around the existing
  DC-CRN ONNX model (`streaming_dsp.py`, `streaming_highband.py`, `streaming_pipeline.py`).
- `src/evaluation/extended_metrics.py` -- SegSNR, frequency-weighted SegSNR,
  impulsive-frame-only ΔSI-SDR, RTF/latency reporting hooks, and an optional WER proxy;
  wired into `scripts/evaluate.py` via a non-breaking `--extended` flag.
- `scripts/synthesize_pairs_v2.py`, `scripts/run_realtime.py` -- CLI entry points for the
  above so they're runnable, not just importable library code.
- Bugfix: `scripts/train.py`, `evaluate.py`, `evaluate_full_pipeline.py`,
  `export_audio_demo.py`, `prepare_demo_manifest.py`, `run_baselines.py`,
  `synthesize_pairs.py` used `from src....` imports but had no path shim, so running them
  directly as `python scripts/whatever.py` from the repo root raised
  `ModuleNotFoundError: No module named 'src'` (they only worked when invoked in a way
  that happened to put the repo root on `sys.path`, e.g. `python -m pytest` from root).
  Added a small `sys.path` insert at the top of each so they run standalone.

## 2. "add a stronger dataset"

`scripts/synthesize_pairs_v2.py` + `src/augmentation/` -- see README's "Stronger dataset
pipeline" section for the full before/after comparison table. Summary: wider SNR range
(-15..+20 dB vs -5..+15 dB), compound multi-source noise beds instead of one noise clip
per sample, synthetic impulsive burst overlays, synthetic reverberation, channel realism
(clipping/gain/tilt jitter), a generated reference-mic signal per sample, and a
zero-download synthetic bootstrap mode (`--synthetic-noise-only`).

## 3. "can we do like this, give input in real time and process it and give output like that"

`src/realtime/` + `scripts/run_realtime.py` -- see README's "Real-time streaming
inference" section. Three runnable modes: `file` (hardware-free simulated streaming,
tested end-to-end against the actual shipped ONNX model), `mic` (live mic -> speaker via
`sounddevice`, requires PortAudio on the machine it's run on), and `benchmark` (measures
and reports real-time factor / latency, writing `results/streaming_latency_report.md`).
The causal-model retrofit, its added latency, and its `infer_every_n_hops` speed/latency
tradeoff are documented in the module docstrings and the README section, consistent with
the existing `hardware/deployment_notes.md` note that a real-time claim requires a causal
or bounded-lookahead front end with *measured* latency -- that measurement now exists in
software (generic CPU), pending re-measurement on the actual embedded target.

## 4. "what else to get the given output parameters, add"

`src/evaluation/extended_metrics.py`, wired into `scripts/evaluate.py --extended` -- see
README's "Extended output parameters" section: SegSNR / frequency-weighted SegSNR,
impulsive-frame-only ΔSI-SDR, RTF and algorithmic latency, and an optional local-ASR WER
proxy for intelligibility (distinct from STOI/PESQ, which are perceptual-model proxies
rather than a task-grounded ASR check).

## 5. "make ours unique apart from others"

No new code by itself proves uniqueness; what changed here is that the existing
differentiators already implicit in the repository (dual-band hybrid split, the rule-based
noise-state controller driving adaptive branch strength, the project's unusually careful
honesty about causality/latency) are now: (a) actually backed by a working real-time
implementation rather than only an offline experiment, and (b) explicitly written up as a
differentiation section in the README ("What makes this hybrid design different") so it
reads as a stated design position rather than something a reviewer has to infer from code.

## Testing

`tests/test_realtime_and_augmentation.py` was added alongside the existing
`tests/test_signal_and_detector.py`, covering the new streaming DSP primitives, the
end-to-end `RealTimeANC` pipeline (both with and without a reference channel), the
synthetic noise generators, the augmentation pipeline, and the extended metrics. Two real
bugs were caught and fixed during this testing (not left for the user to hit):

1. `OverlapAddSynthesizer` had a near-zero-division startup transient that produced a
   large output spike in the first hop after `reset()`.
2. `synth_impulsive_train` could compute a negative/empty array slice near the end of a
   generated clip when a jittered burst start landed past the clip boundary.

`python -m pytest -q` passes (12/12) as of this revision.

## 6. Real, licensed audio bundled directly in the repo (this session's follow-up)

After the first delivery, the demo run in Colab surfaced two real gaps, both fixed here:

- **No real human speech or real noise was bundled** -- only synthetic tones/DSP, so
  nothing in a demo would actually sound like speech. Fixed by sourcing and bundling
  169 real speech segments (6 distinct speakers, from Creative-Commons freesound.org
  recordings via https://github.com/voxserv/audio_quality_testing_samples) and 10 real
  noise clips covering all 3 required categories -- stationary (industrial fan, A/C
  rumble, truck engine), non-stationary (helicopter, aircraft landing, chainsaw), and
  transient (explosion, jackhammer) -- from Creative-Commons sources via
  https://github.com/ALEX11BR/AltSFX. Full per-file attribution in `ATTRIBUTIONS.md`,
  as required by the CC-BY-licensed files among them. This is bundled directly in
  `data/raw/speech_real/` and `data/raw/noise_real/`, with manifests already generated
  in `data/processed_real/` -- no download step needed to try it.
- **`scripts/train.py` had no way to train on `synthesize_pairs_v2.py`'s output**
  (different manifest columns) -- this was flagged as an open item in the first
  delivery and is now fixed: `train.py` gained a second mode,
  `--train-pairs-csv`/`--val-pairs-csv`, alongside the original
  `--train-speech`/`--val-speech`/`--noise` online-mixing mode. It also now writes
  `history.json` per run instead of only printing to the console.
- **No script turned a checkpoint into demo-ready audio** -- added
  `scripts/infer_checkpoint.py`, which loads a checkpoint and writes clean/noisy
  /enhanced `.wav` triples + a manifest ready for `scripts/evaluate.py`.

Honest limitation carried forward: none of this was trained inside this sandbox (no
GPU, and a `torch` install here ran the container out of disk space) -- the dataset
generation and manifest logic were verified end-to-end without torch; the actual
training run still has to happen in your Colab, same as the first delivery.

## 7. Confidence-gated inference (this session's second follow-up)

Interactive evaluation in Colab surfaced a real behavioral bug: the trained checkpoint
applied a roughly fixed amount of enhancement regardless of input SNR, which measurably
*hurt* already-clean input (observed: at +15dB input SNR, ΔSNR was -15.85dB -- the
"enhanced" output was worse than doing nothing). Root-caused and fixed by adding
`src/realtime/confidence_gate.py`: a lightweight, torch-free SNR-aware blend between the
fully-enhanced signal and the raw input, driven by a running estimate of the actual
current SNR from the reference-mic signal. Verified to produce the expected monotonic
alpha curve (1.0 → 0.05 as input SNR rises from -10dB to +15dB) both in the offline
demo-rendering path and in the live `RealTimeANC` streaming pipeline.

Added `scripts/render_sih_demo.py`, which runs an SNR sweep with this gate applied and
renders presentation-ready demo clips + a `summary.md`. Added
`results/sih_demo_reference/` with a real, dated, honestly-captioned reference result
from this project's actual training run, including where the PS's SNR/STOI/PESQ targets
are and aren't met (STOI target met at 10-15dB input SNR; PESQ still short across the
board, attributed to training-set scale, not architecture).

`tests/test_realtime_and_augmentation.py` gained a regression test pinning the gate's
output to the values observed during Colab validation, so the formula can't silently
drift. `python -m pytest -q` passes (13/13) as of this revision.
