# Latency report

## Current status

No new benchmark was run against the uploaded ONNX artifacts in this audit environment.

| Measurement | Value |
|---|---|
| Model inference latency | Not yet measured |
| STFT latency | Not yet measured |
| iSTFT latency | Not yet measured |
| End-to-end audio latency | Not yet measured |
| Target hardware | Not specified |

Run:

```bash
python scripts/benchmark_latency.py models/ampora_lite_dccrn_fp32.onnx
python scripts/benchmark_latency.py models/ampora_lite_dccrn_int8.onnx
```

These numbers cover model inference only. They must not be reported as end-to-end latency.
