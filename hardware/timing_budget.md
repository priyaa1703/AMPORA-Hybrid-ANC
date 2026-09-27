# Timing budget

Measured values are intentionally blank until the target CPU/SoC and complete
pipeline are benchmarked.

| Stage | Measured ms | Status |
|---|---:|---|
| Input buffering | Not yet measured | target-dependent |
| STFT | Not yet measured | target-dependent |
| DC-CRN inference | Not yet measured | run scripts/benchmark_latency.py |
| iSTFT | Not yet measured | target-dependent |
| Filterbank/recombination | Not yet measured | target-dependent |
| End-to-end | Not yet measured | do not claim a latency number |

A sub-10-ms value is a design target only unless measured on the final target.
