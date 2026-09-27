"""End-to-end real-time (chunk-wise, stateful) AMPORA hybrid ANC pipeline.

This is the "take live input, process it, produce live output" piece that
was missing from the original repository, which only had *offline*,
whole-array helpers (``src/hybrid.py``, ``scripts/evaluate.py``). It wires
together:

  * ``CausalBandSplitter`` + ``StreamingNLMS``   -> low-band branch
  * ``StreamingHighBandEnhancer`` (ONNX DC-CRN)  -> high-band branch
  * a sample-accurate delay line                  -> aligns the (near
    zero-latency) low band with the (context-window-latency) high band
    before summation, so the two bands are not summed out of phase.

Input can be fed in arbitrary-sized blocks (whatever a sound-card callback
or a chunked file reader hands you); output is produced in whatever
whole multiples of ``hop_length`` are available after each call, which is
the standard pattern for real-time audio callbacks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time

import numpy as np

from .streaming_dsp import CausalBandSplitter, StreamingNLMS
from .streaming_highband import StreamingHighBandEnhancer
from .confidence_gate import confidence_gate_alpha


@dataclass
class _DelayLine:
    delay: int
    buf: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.buf = np.zeros(self.delay, dtype=np.float32)

    def push_pop(self, x: np.ndarray) -> np.ndarray:
        if self.delay == 0:
            return x
        combined = np.concatenate([self.buf, x])
        out = combined[: len(x)]
        self.buf = combined[len(x):][-self.delay:]
        if len(self.buf) < self.delay:
            self.buf = np.concatenate([np.zeros(self.delay - len(self.buf), dtype=np.float32), self.buf])
        return out.astype(np.float32)


class RealTimeANC:
    """Stateful real-time hybrid ANC processor.

    Call :meth:`process` repeatedly with successive raw audio blocks
    (primary mic, optionally reference mic). Internally buffers to the
    fixed ``hop_length`` the AI branch needs and returns however many
    aligned output samples are ready on each call (this will lag the input
    by up to one hop plus the AI branch's algorithmic latency, which is
    reported in :pyattr:`latency_ms`).
    """

    def __init__(self, sr=16000, split_hz=1500.0, high_hz=7900.0,
                 nfft=512, win_length=320, hop_length=128, context_frames=16,
                 nlms_filter_len=64, nlms_mu=0.08, onnx_path=None,
                 infer_every_n_hops=1, use_confidence_gate=True,
                 gate_low_db=-5.0, gate_span_db=20.0, gate_smoothing=0.9):
        self.sr = sr
        self.hop = hop_length

        self.band_primary = CausalBandSplitter(sr=sr, split_hz=split_hz, high_hz=high_hz)
        self.band_reference = CausalBandSplitter(sr=sr, split_hz=split_hz, high_hz=high_hz)
        self.nlms = StreamingNLMS(filter_len=nlms_filter_len, mu=nlms_mu)
        self.high_enhancer = StreamingHighBandEnhancer(
            sr=sr, nfft=nfft, win_length=win_length, hop_length=hop_length,
            low_hz=split_hz, high_hz=high_hz, onnx_path=onnx_path, context_frames=context_frames,
            infer_every_n_hops=infer_every_n_hops,
        )
        self._low_delay = _DelayLine(self.high_enhancer.latency_samples)

        # Confidence gate: blend the enhanced output against the raw (delayed)
        # input based on a running estimate of the current SNR from the
        # primary/reference hop levels, so the system does real cleanup when
        # it's genuinely noisy and does little-to-no harm when the input is
        # already fairly clean -- see src/realtime/confidence_gate.py and
        # CHANGELOG_ADDITIONS.md for why this was added (without it, a model
        # trained on a small dataset applies a roughly fixed amount of
        # "cleanup" regardless of input SNR, which actively hurts clean input).
        self.use_confidence_gate = use_confidence_gate
        self.gate_low_db = gate_low_db
        self.gate_span_db = gate_span_db
        self.gate_smoothing = gate_smoothing
        self._raw_delay = _DelayLine(self.high_enhancer.latency_samples)
        self._ema_primary_level = 1e-4
        self._ema_reference_level = 1e-4
        self.last_alpha = 1.0

        self._in_primary = np.zeros(0, dtype=np.float32)
        self._in_reference = np.zeros(0, dtype=np.float32)
        self._have_reference = False

        self.frames_processed = 0
        self._proc_time_s = 0.0
        self._audio_time_s = 0.0

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.high_enhancer.latency_samples / self.sr

    @property
    def using_ai_model(self) -> bool:
        return self.high_enhancer.using_ai_model

    @property
    def real_time_factor(self) -> float:
        """processing_time / audio_time; must stay < 1.0 to keep up with
        real time (lower is better / more headroom)."""
        if self._audio_time_s <= 0:
            return 0.0
        return self._proc_time_s / self._audio_time_s

    def process(self, primary_block: np.ndarray, reference_block: np.ndarray | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        primary_block = np.asarray(primary_block, dtype=np.float32).reshape(-1)
        self._in_primary = np.concatenate([self._in_primary, primary_block])
        if reference_block is not None:
            self._have_reference = True
            reference_block = np.asarray(reference_block, dtype=np.float32).reshape(-1)
            self._in_reference = np.concatenate([self._in_reference, reference_block])
        elif self._have_reference:
            # keep buffers aligned in length if reference is intermittently absent
            self._in_reference = np.concatenate([self._in_reference, np.zeros_like(primary_block)])

        n_hops = len(self._in_primary) // self.hop
        out_chunks = []
        for _ in range(n_hops):
            hop_primary = self._in_primary[: self.hop]
            self._in_primary = self._in_primary[self.hop:]

            low, _ = self.band_primary.process(hop_primary)

            if self._have_reference:
                hop_ref = self._in_reference[: self.hop]
                self._in_reference = self._in_reference[self.hop:]
                low_ref, _ = self.band_reference.process(hop_ref)
                low_enhanced = self.nlms.process(low, low_ref)
            else:
                low_enhanced = low

            high_enhanced = self.high_enhancer.process_hop(hop_primary)
            low_aligned = self._low_delay.push_pop(low_enhanced)
            enhanced_hop = (low_aligned + high_enhanced).astype(np.float32)

            if self.use_confidence_gate and self._have_reference:
                b = self.gate_smoothing
                self._ema_primary_level = b * self._ema_primary_level + (1 - b) * float(np.sqrt(np.mean(hop_primary ** 2) + 1e-12))
                self._ema_reference_level = b * self._ema_reference_level + (1 - b) * float(np.sqrt(np.mean(hop_ref ** 2) + 1e-12))
                est_snr_db = 20.0 * np.log10(self._ema_primary_level / (self._ema_reference_level + 1e-8))
                alpha = confidence_gate_alpha(est_snr_db, low_db=self.gate_low_db, span_db=self.gate_span_db)
                self.last_alpha = alpha
                raw_aligned = self._raw_delay.push_pop(hop_primary)
                enhanced_hop = (alpha * enhanced_hop + (1 - alpha) * raw_aligned).astype(np.float32)
            else:
                self._raw_delay.push_pop(hop_primary)  # keep delay line advancing even if unused

            out_chunks.append(enhanced_hop)
            self.frames_processed += 1

        out = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
        self._proc_time_s += time.perf_counter() - t0
        self._audio_time_s += len(primary_block) / self.sr
        return out

    def reset(self):
        self.band_primary.reset()
        self.band_reference.reset()
        self.nlms.reset()
        self.high_enhancer.reset()
        self._in_primary = np.zeros(0, dtype=np.float32)
        self._in_reference = np.zeros(0, dtype=np.float32)
        self._have_reference = False
        self.frames_processed = 0
        self._proc_time_s = 0.0
        self._audio_time_s = 0.0
