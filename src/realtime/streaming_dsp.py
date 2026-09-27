"""Stateful, chunk-wise (streaming) DSP building blocks.

Unlike the offline helpers in ``src/hybrid.py`` / ``src/baselines.py`` (which
operate on a whole array at once), everything here keeps internal state
(filter memory, adaptive-filter weights, overlap-add buffers) between calls to
``process()``/``push()`` so that audio can be fed in fixed-size real-time
chunks, e.g. 128 or 256 samples at a time from a microphone callback.
"""
from __future__ import annotations
import numpy as np
from scipy import signal


class CausalBandSplitter:
    """Low/high band splitter that preserves IIR filter memory (``zi``)
    across chunk boundaries, so successive chunks concatenate seamlessly
    (no clicks/artifacts at chunk edges, unlike naive per-chunk ``sosfilt``)."""

    def __init__(self, sr=16000, split_hz=1500.0, high_hz=7900.0, order=6):
        nyq = sr / 2
        if not 0 < split_hz < high_hz < nyq:
            raise ValueError("Band edges must satisfy 0 < split < high < Nyquist.")
        self.low_sos = signal.butter(order, split_hz, btype="lowpass", fs=sr, output="sos")
        self.high_sos = signal.butter(order, [split_hz, high_hz], btype="bandpass", fs=sr, output="sos")
        self._zi_low = signal.sosfilt_zi(self.low_sos)
        self._zi_high = signal.sosfilt_zi(self.high_sos)

    def process(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        low, self._zi_low = signal.sosfilt(self.low_sos, chunk, zi=self._zi_low)
        high, self._zi_high = signal.sosfilt(self.high_sos, chunk, zi=self._zi_high)
        return low.astype(np.float32), high.astype(np.float32)

    def reset(self):
        self._zi_low = signal.sosfilt_zi(self.low_sos)
        self._zi_high = signal.sosfilt_zi(self.high_sos)


class StreamingNLMS:
    """Reference-assisted Normalized-LMS adaptive filter with persistent
    weights and a persistent input delay-line, so it can run sample-by-sample
    across an unbounded stream of small chunks (real-time equivalent of
    ``src/baselines.py::nlms_reference_anc``)."""

    def __init__(self, filter_len=64, mu=0.08, eps=1e-6):
        self.filter_len = filter_len
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(filter_len, dtype=np.float32)
        self._ref_hist = np.zeros(filter_len, dtype=np.float32)

    def process(self, primary_chunk: np.ndarray, reference_chunk: np.ndarray) -> np.ndarray:
        primary_chunk = np.asarray(primary_chunk, dtype=np.float32)
        reference_chunk = np.asarray(reference_chunk, dtype=np.float32)
        n = len(primary_chunk)
        out = np.empty(n, dtype=np.float32)
        w = self.w
        hist = self._ref_hist
        for i in range(n):
            hist = np.roll(hist, 1)
            hist[0] = reference_chunk[i]
            y_hat = float(np.dot(w, hist))
            e = primary_chunk[i] - y_hat
            norm = float(np.dot(hist, hist)) + self.eps
            w = w + (self.mu / norm) * e * hist
            out[i] = e
        self.w = w
        self._ref_hist = hist
        return out

    def reset(self):
        self.w[:] = 0.0
        self._ref_hist[:] = 0.0


class OverlapAddSynthesizer:
    """Generic weighted overlap-add (WOLA) accumulator.

    Normalises by the *actual* accumulated squared-window energy at each
    output sample, so reconstruction stays correct even for a (win, hop)
    pair that is not on a "textbook" COLA ratio, which real-time chunk sizes
    chosen for latency reasons often are not.
    """

    def __init__(self, win_length: int, hop_length: int, window: np.ndarray | None = None):
        self.win = win_length
        self.hop = hop_length
        self.window = window if window is not None else signal.windows.hann(win_length, sym=False).astype(np.float32)
        self._buf = np.zeros(win_length, dtype=np.float32)
        self._norm = np.zeros(win_length, dtype=np.float32)

    def add_frame(self, frame: np.ndarray):
        """``frame`` must be exactly ``win_length`` time-domain samples,
        already the raw (un-windowed) synthesis signal for this analysis hop."""
        w = self.window
        self._buf[: self.win] += frame[: self.win] * w
        self._norm[: self.win] += w * w

    def pop_hop(self) -> np.ndarray:
        # A relatively large floor (rather than 1e-8) is intentional: during the
        # first ~win/hop hops after reset() only 1-2 windows have overlapped a
        # given sample, so the true window-energy normaliser is legitimately
        # small there. Dividing by a near-zero float causes an audible startup
        # spike; flooring trades a few hops of slight under-normalisation
        # (quieter, not louder, startup) for guaranteed-bounded output.
        eps = 1e-2
        norm = np.maximum(self._norm[: self.hop], eps)
        out = (self._buf[: self.hop] / norm).astype(np.float32)
        self._buf = np.concatenate([self._buf[self.hop:], np.zeros(self.hop, dtype=np.float32)])
        self._norm = np.concatenate([self._norm[self.hop:], np.zeros(self.hop, dtype=np.float32)])
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._norm[:] = 0.0
