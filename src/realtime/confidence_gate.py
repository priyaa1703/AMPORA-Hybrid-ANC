"""Confidence gate for blending AI/NLMS-enhanced audio against raw input based
on the actual current SNR (as estimated from a reference-mic noise signal).

Split into its own module (no torch dependency) so it can be unit-tested and
reused without pulling in the full model stack -- see
scripts/render_sih_demo.py for where this is used in the demo-rendering
pipeline, and CHANGELOG_ADDITIONS.md for why it was added: a model trained on
a small dataset tends to apply a roughly fixed amount of "cleanup" regardless
of input SNR, which helps when the input is very noisy but actively hurts
already-clean input. Gating the blend by the real current SNR fixes that
without any retraining.
"""
from __future__ import annotations
import numpy as np


def confidence_gate_alpha(actual_snr_db: float, low_db: float = -5.0, span_db: float = 20.0) -> float:
    """Blend weight for "trust the enhanced signal" vs "trust the raw input".

    ~1.0 (fully trust the enhanced output) when the input is very noisy
    (at/below ``low_db``), tapering linearly down to ~0.05 (mostly leave the
    signal alone) once the input is already fairly clean (at/above
    ``low_db + span_db``). Clamped to [0.05, 1.0] -- never fully 0 or 1, so
    there is always a small floor of enhancement and never a full hard cut.
    """
    alpha = 1.0 - (actual_snr_db - low_db) / span_db
    return float(np.clip(alpha, 0.05, 1.0))
