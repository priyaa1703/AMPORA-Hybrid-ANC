# Streaming (real-time) latency report

Measured with `scripts/run_realtime.py benchmark` on the machine that generated this file (onnxruntime CPUExecutionProvider). These are software measurements on generic CPU, not the Jetson/embedded target numbers in `hardware/timing_budget.md` -- re-run this on the actual target hardware (and with the TensorRT execution provider once available) before making a hardware real-time claim.

| context_frames | infer_every_n_hops | algorithmic latency (ms) | RTF | real-time safe? |
|---|---|---|---|---|
| 16 | 1 | 140.0 | 1.127 | NO |
| 16 | 3 | 140.0 | 0.552 | yes |
| 8 | 1 | 76.0 | 0.719 | yes |
| 8 | 2 | 76.0 | 0.493 | yes |
| 8 | 3 | 76.0 | 0.418 | yes |
| 4 | 1 | 44.0 | 0.601 | yes |
| 4 | 2 | 44.0 | 0.421 | yes |

AI model loaded and used: True
