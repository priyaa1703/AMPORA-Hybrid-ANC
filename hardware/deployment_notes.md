# Deployment notes

The current ONNX artifacts demonstrate model export and CPU inference compatibility,
not FPGA performance. Final deployment requires a selected SoC/FPGA, runtime,
compiler/toolchain, memory budget and measured timing.

Important signal-processing constraint: the previous notebook used centered STFT
and offline-style filtering. Those choices introduce look-ahead/non-causal behavior.
For a real-time claim, use a causal or explicitly bounded-lookahead front end and
measure the resulting end-to-end latency.
