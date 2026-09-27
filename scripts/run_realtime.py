#!/usr/bin/env python3
"""Real-time (or real-time-simulated) AMPORA hybrid ANC entry point.

This is the "take input, process it live, give output live" piece requested
on top of the original (offline-only) repository. Three modes:

  mic        Live microphone -> live speaker/headset output, using
             ``sounddevice``. Optionally uses a second input channel as the
             reference microphone (see the block diagram: primary + reference
             mics feeding the DSP chain).
  file       Simulates real-time operation by feeding an existing .wav file
             through the pipeline in fixed-size chunks exactly as a live
             callback would receive them, measuring wall-clock processing
             time per chunk, and writing an enhanced .wav. Use this to
             validate/demo the pipeline on any machine, with no microphone
             or audio hardware required.
  benchmark  Runs the file-mode pipeline over a short synthetic signal at
             several (context_frames, infer_every_n_hops) settings and
             prints the measured real-time factor (RTF) and latency for
             each, so you can pick a real-time-safe configuration for your
             target hardware. Writes results/streaming_latency_report.md.

Examples
--------
    python scripts/run_realtime.py file --in noisy.wav --out enhanced.wav \\
        --model models/ampora_lite_dccrn_fp32.onnx --context-frames 8 --infer-every 2

    python scripts/run_realtime.py mic --model models/ampora_lite_dccrn_int8.onnx \\
        --context-frames 8 --infer-every 2

    python scripts/run_realtime.py benchmark --model models/ampora_lite_dccrn_fp32.onnx
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import time
from pathlib import Path

import numpy as np

from src.realtime.streaming_pipeline import RealTimeANC


def _build_anc(args) -> RealTimeANC:
    return RealTimeANC(
        sr=args.sr,
        split_hz=args.split_hz,
        high_hz=args.high_hz,
        context_frames=args.context_frames,
        infer_every_n_hops=args.infer_every,
        onnx_path=args.model,
    )


def run_file(args):
    import soundfile as sf
    primary, sr = sf.read(args.infile, dtype="float32")
    if primary.ndim == 2:
        # Treat a stereo file as [primary, reference] mics, matching the
        # dual-microphone block diagram, unless --mono-only is given.
        reference = None if args.mono_only else primary[:, 1]
        primary = primary[:, 0]
    else:
        reference = None
    if args.reference_file:
        reference, sr_r = sf.read(args.reference_file, dtype="float32")
        if reference.ndim == 2:
            reference = reference.mean(axis=1)
        assert sr_r == sr, "reference file must share the primary file's sample rate"

    if sr != args.sr:
        raise SystemExit(f"Input file sample rate {sr} != --sr {args.sr}. Resample first "
                          f"(see src/preprocessing/audio.py::read_audio).")

    anc = _build_anc(args)
    print(f"[run_realtime] AI model active: {anc.using_ai_model} | "
          f"algorithmic latency: {anc.latency_ms:.1f} ms | chunk size: {args.chunk} samples "
          f"({1000*args.chunk/args.sr:.1f} ms)")

    out_chunks = []
    n = len(primary)
    t0 = time.perf_counter()
    for i in range(0, n, args.chunk):
        p_chunk = primary[i:i + args.chunk]
        r_chunk = reference[i:i + args.chunk] if reference is not None else None
        out_chunks.append(anc.process(p_chunk, r_chunk))
    wall = time.perf_counter() - t0
    enhanced = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)

    peak = np.max(np.abs(enhanced)) + 1e-9
    if peak > 0.999:
        enhanced = enhanced * (0.999 / peak)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.outfile, enhanced, args.sr)

    audio_s = n / args.sr
    print(f"[run_realtime] processed {audio_s:.2f}s of audio in {wall:.2f}s wall-clock "
          f"-> RTF={wall/audio_s:.3f} (must be < 1.0 to run in real time on this machine)")
    print(f"[run_realtime] wrote {args.outfile}")


def run_mic(args):
    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover - hardware-dependent
        raise SystemExit(
            "sounddevice / PortAudio is not available in this environment.\n"
            "Install it with `pip install sounddevice` and make sure a PortAudio "
            "runtime is present (e.g. `apt-get install libportaudio2` on Linux), "
            "or use `python scripts/run_realtime.py file ...` instead, which needs "
            "no audio hardware.\n"
            f"Original error: {e}"
        )

    anc = _build_anc(args)
    channels = 2 if args.reference_channel else 1
    print(f"[run_realtime] AI model active: {anc.using_ai_model} | "
          f"algorithmic latency: {anc.latency_ms:.1f} ms | listening on input device "
          f"{args.input_device!r}, output device {args.output_device!r} (Ctrl+C to stop)")

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=_sys.stderr)
        primary = indata[:, 0]
        reference = indata[:, 1] if channels == 2 else None
        enhanced = anc.process(primary, reference)
        if len(enhanced) < frames:
            enhanced = np.pad(enhanced, (0, frames - len(enhanced)))
        outdata[:, 0] = enhanced[:frames]

    with sd.Stream(samplerate=args.sr, blocksize=args.chunk, channels=(channels, 1),
                    dtype="float32", device=(args.input_device, args.output_device),
                    callback=callback):
        try:
            while True:
                time.sleep(0.5)
                print(f"\r[run_realtime] RTF={anc.real_time_factor:.3f}  "
                      f"frames_processed={anc.frames_processed}", end="", flush=True)
        except KeyboardInterrupt:
            print("\n[run_realtime] stopped.")


def run_benchmark(args):
    sr = args.sr
    rng = np.random.default_rng(0)
    t = np.arange(int(sr * args.bench_seconds)) / sr
    clean = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = 0.2 * rng.standard_normal(len(t)).astype(np.float32)
    noisy = clean + noise

    configs = [(16, 1), (16, 3), (8, 1), (8, 2), (8, 3), (4, 1), (4, 2)]
    rows = []
    for ctx, every in configs:
        anc = RealTimeANC(sr=sr, context_frames=ctx, infer_every_n_hops=every, onnx_path=args.model)
        for i in range(0, len(noisy), args.chunk):
            anc.process(noisy[i:i + args.chunk], noise[i:i + args.chunk])
        rows.append((ctx, every, anc.latency_ms, anc.real_time_factor, anc.using_ai_model))
        safe = "REAL-TIME SAFE" if anc.real_time_factor < 1.0 else "TOO SLOW on this CPU"
        print(f"context_frames={ctx:>2} infer_every={every}  latency={anc.latency_ms:6.1f} ms  "
              f"RTF={anc.real_time_factor:.3f}  [{safe}]")

    out_md = Path("results/streaming_latency_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Streaming (real-time) latency report\n\n")
        f.write("Measured with `scripts/run_realtime.py benchmark` on the machine that "
                "generated this file (onnxruntime CPUExecutionProvider). These are software "
                "measurements on generic CPU, not the Jetson/embedded target numbers in "
                "`hardware/timing_budget.md` -- re-run this on the actual target hardware "
                "(and with the TensorRT execution provider once available) before making a "
                "hardware real-time claim.\n\n")
        f.write("| context_frames | infer_every_n_hops | algorithmic latency (ms) | RTF | real-time safe? |\n")
        f.write("|---|---|---|---|---|\n")
        for ctx, every, lat, rtf, ai in rows:
            f.write(f"| {ctx} | {every} | {lat:.1f} | {rtf:.3f} | {'yes' if rtf < 1.0 else 'NO'} |\n")
        f.write(f"\nAI model loaded and used: {rows[0][4] if rows else 'n/a'}\n")
    print(f"\n[run_realtime] wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sr", type=int, default=16000)
    common.add_argument("--split-hz", type=float, default=1500.0)
    common.add_argument("--high-hz", type=float, default=7900.0)
    common.add_argument("--chunk", type=int, default=256, help="samples per real-time block")
    common.add_argument("--context-frames", type=int, default=8)
    common.add_argument("--infer-every", type=int, default=2,
                         help="run the neural net every N hops (>1 trades mask time-resolution for speed)")
    common.add_argument("--model", default="models/ampora_lite_dccrn_int8.onnx",
                         help="ONNX model path, or omit/pass a bad path to force the "
                              "dependency-free spectral-subtraction fallback")

    pf = sub.add_parser("file", parents=[common], help="simulate real-time streaming from a wav file")
    pf.add_argument("--in", dest="infile", required=True)
    pf.add_argument("--out", dest="outfile", required=True)
    pf.add_argument("--reference-file", default=None, help="optional separate reference-mic wav")
    pf.add_argument("--mono-only", action="store_true", help="ignore a 2nd channel even if present")
    pf.set_defaults(func=run_file)

    pm = sub.add_parser("mic", parents=[common], help="live microphone -> speaker (needs sounddevice/PortAudio)")
    pm.add_argument("--input-device", default=None)
    pm.add_argument("--output-device", default=None)
    pm.add_argument("--reference-channel", action="store_true",
                     help="treat input channel 2 as the reference microphone")
    pm.set_defaults(func=run_mic)

    pb = sub.add_parser("benchmark", parents=[common], help="measure RTF/latency across settings, no audio file needed")
    pb.add_argument("--bench-seconds", type=float, default=3.0)
    pb.set_defaults(func=run_benchmark)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
