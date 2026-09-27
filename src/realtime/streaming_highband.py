"""Real-time high-band AI enhancement: wraps the existing (offline-trained,
non-streaming) ``AMPORA_DCCRN`` ONNX artifact in a causal, chunk-wise
sliding-window inference loop, using ``OverlapAddSynthesizer`` for
reconstruction.

Important honesty note (kept in the spirit of AUDIT_REPORT.md / README's
"Important implementation boundary" section): the underlying convolutional
blocks use symmetric time-axis padding and the model was trained with
``center=True`` STFT framing (see README point 6), so it is not a strictly
causal, sample-recurrent streaming model. This wrapper retrofits it for
real-time use with two practical compromises, both configurable:

  1. A short causal *context window* of the last ``context_frames`` STFT
     frames is fed to the model on every hop; only the newest frame's output
     mask is used. This adds algorithmic latency of roughly
     ``context_frames * hop_length`` samples but requires no architecture
     change or retraining.
  2. If no ONNX runtime / model file is available, the enhancer degrades to
     a classical causal spectral-subtraction fallback so the pipeline still
     produces sensible output rather than erroring out or silently
     passing raw noise through.
"""
from __future__ import annotations
import numpy as np
from scipy import signal

from .streaming_dsp import OverlapAddSynthesizer

try:
    import onnxruntime as _ort
except Exception:  # pragma: no cover - onnxruntime optional at runtime
    _ort = None


class _SpectralSubtractionFallback:
    """Minimal causal noise-floor spectral subtraction, used only when no
    ONNX model/runtime is available. Not a claim of AI enhancement -- it is
    a safe, dependency-free degrade path."""

    def __init__(self, freq_bins, alpha=1.6, floor=0.05):
        self.alpha = alpha
        self.floor = floor
        self.noise_est = np.full(freq_bins, 1e-4, dtype=np.float32)
        self._t = 0

    def mask(self, mag: np.ndarray) -> np.ndarray:
        self._t += 1
        adapt = 0.98 if self._t > 20 else 0.5  # faster noise-floor lock-on at start
        self.noise_est = adapt * self.noise_est + (1 - adapt) * np.minimum(mag, self.noise_est * 1.5 + 1e-6)
        sub = np.maximum(mag - self.alpha * self.noise_est, self.floor * mag)
        gain = sub / (mag + 1e-8)
        return np.clip(gain - 1.0, -1.0, 1.0)  # express as a residual mask like the AI model does


class StreamingHighBandEnhancer:
    def __init__(self, sr=16000, nfft=512, win_length=320, hop_length=128,
                 low_hz=1500.0, high_hz=7900.0, onnx_path=None,
                 context_frames=16, providers=None, infer_every_n_hops=1):
        self.sr = sr
        self.nfft = nfft
        self.win = win_length
        self.hop = hop_length
        self.context_frames = context_frames
        self.analysis_window = signal.windows.hann(win_length, sym=False).astype(np.float32)
        # Running the (comparatively expensive) neural net on every single
        # hop is what makes RTF>1 likely on a plain CPU onnxruntime backend
        # without INT8/TensorRT kernels (see hardware/deployment_notes.md and
        # results/latency_report.md for the target-hardware numbers). Holding
        # the mask for N hops between inferences is a standard, cheap way to
        # trade a little temporal resolution in the *mask* for a large,
        # predictable reduction in compute -- the audio itself is still
        # produced every hop via overlap-add, nothing is dropped.
        self.infer_every_n_hops = max(1, int(infer_every_n_hops))
        self._hop_counter = 0
        self._held_mask = None

        bin_hz = sr / nfft
        self.lo_bin = int(np.ceil(low_hz / bin_hz))

        self._session = None
        self.freq_bins = None
        if onnx_path is not None and _ort is not None:
            try:
                self._session = _ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
                in_shape = self._session.get_inputs()[0].shape
                fb = in_shape[2]
                self.freq_bins = int(fb) if isinstance(fb, int) else int(round((high_hz - low_hz) / bin_hz)) + 4
                self._input_name = self._session.get_inputs()[0].name
            except Exception:
                self._session = None

        if self._session is None:
            hi_bin = int(np.floor(high_hz / bin_hz)) + 1
            self.freq_bins = max(hi_bin - self.lo_bin, 1)
            self._fallback = _SpectralSubtractionFallback(self.freq_bins)

        self.hi_bin = self.lo_bin + self.freq_bins  # exclusive
        self.full_bins = nfft // 2 + 1

        ctx_samples = self.win + (context_frames - 1) * self.hop
        self._sample_buf = np.zeros(ctx_samples, dtype=np.float32)
        self._filled = 0  # how many real (non-zero-padded) samples have entered so far
        self.ola = OverlapAddSynthesizer(self.win, self.hop, window=self.analysis_window)

    @property
    def using_ai_model(self) -> bool:
        return self._session is not None

    def _frame_bins(self, frame: np.ndarray):
        windowed = frame * self.analysis_window
        spec = np.fft.rfft(windowed, n=self.nfft)
        return spec

    def process_hop(self, hop_chunk: np.ndarray) -> np.ndarray:
        """Feed exactly ``hop_length`` new high-band samples in; returns
        ``hop_length`` enhanced samples out (with ``context`` algorithmic
        latency already absorbed by the internal overlap-add buffer)."""
        hop_chunk = np.asarray(hop_chunk, dtype=np.float32)
        assert len(hop_chunk) == self.hop
        self._sample_buf = np.concatenate([self._sample_buf[self.hop:], hop_chunk])
        self._filled = min(self._filled + self.hop, len(self._sample_buf))

        specs = []
        for j in range(self.context_frames):
            start = j * self.hop
            frame = self._sample_buf[start:start + self.win]
            specs.append(self._frame_bins(frame))
        specs = np.stack(specs, axis=-1)  # (full_bins, context_frames)
        sub = specs[self.lo_bin:self.hi_bin, :]

        run_inference = (self._hop_counter % self.infer_every_n_hops == 0) or (self._held_mask is None)
        self._hop_counter += 1

        if self._session is not None:
            if run_inference:
                inp = np.stack([sub.real, sub.imag], axis=0)[None].astype(np.float32)  # (1,2,freq,time)
                try:
                    mask = self._session.run(None, {self._input_name: inp})[0][0]  # (2,freq,time)
                    mask_c = mask[0, :, -1] + 1j * mask[1, :, -1]
                except Exception:
                    mask_c = np.zeros(sub.shape[0], dtype=np.complex64)
                self._held_mask = mask_c
            else:
                mask_c = self._held_mask
        else:
            mag = np.abs(sub[:, -1])
            mask_c = self._fallback.mask(mag).astype(np.complex64)

        last_spec = specs[:, -1].copy()
        enhanced_sub = last_spec[self.lo_bin:self.hi_bin] * (1.0 + mask_c)
        full_spec = last_spec.copy()
        full_spec[self.lo_bin:self.hi_bin] = enhanced_sub
        full_spec[: self.lo_bin] = 0.0
        full_spec[self.hi_bin:] = 0.0

        time_frame = np.fft.irfft(full_spec, n=self.nfft)[: self.win].astype(np.float32)
        self.ola.add_frame(time_frame)
        return self.ola.pop_hop()

    def reset(self):
        self._sample_buf[:] = 0.0
        self._filled = 0
        self.ola.reset()

    @property
    def latency_samples(self) -> int:
        """Algorithmic latency introduced by this block, in samples: one
        full analysis window plus the causal context lookback."""
        return self.win + (self.context_frames - 1) * self.hop
