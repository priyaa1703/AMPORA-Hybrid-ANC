import numpy as np
from src.realtime.streaming_pipeline import RealTimeANC
from src.realtime.streaming_dsp import CausalBandSplitter, StreamingNLMS, OverlapAddSynthesizer
from src.augmentation.pipeline import augment_pair
from src.augmentation.synthetic_transients import synth_impulsive_burst, synth_rotor_drone, synthetic_noise_bank
from src.evaluation.extended_metrics import seg_snr, freq_weighted_seg_snr, extended_pair_metrics


def _tone_and_noise(sr=16000, seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    clean = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = (0.2 * rng.standard_normal(len(t))).astype(np.float32)
    return clean, noise


def test_causal_band_splitter_matches_offline_length():
    clean, noise = _tone_and_noise()
    noisy = clean + noise
    bp = CausalBandSplitter()
    low, high = bp.process(noisy)
    assert len(low) == len(noisy) == len(high)
    assert np.isfinite(low).all() and np.isfinite(high).all()


def test_streaming_nlms_is_bounded_and_finite():
    clean, noise = _tone_and_noise()
    noisy = clean + noise
    nlms = StreamingNLMS()
    out = nlms.process(noisy, noise)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) < 10.0  # generously bounded, catches divergence


def test_overlap_add_reconstructs_dc_within_tolerance():
    win, hop = 320, 128
    ola = OverlapAddSynthesizer(win, hop)
    # OverlapAddSynthesizer implements WOLA: add_frame() applies the synthesis
    # window on top of a frame that (in the real pipeline) already carries the
    # analysis window once. To exercise that contract directly, feed it an
    # already-analysis-windowed constant signal (x=1 times the window) and
    # check the round trip recovers x=1 in steady state.
    frame = ola.window.copy()
    outs = []
    for _ in range(10):
        ola.add_frame(frame)
        outs.append(ola.pop_hop())
    out = np.concatenate(outs)
    steady = out[3 * hop:]
    assert np.isfinite(steady).all()
    assert np.allclose(steady, 1.0, atol=0.35)


def test_realtime_anc_end_to_end_no_reference():
    clean, noise = _tone_and_noise(seconds=1.0)
    noisy = clean + noise
    anc = RealTimeANC(sr=16000, onnx_path=None)  # dependency-free fallback path
    outs = [anc.process(noisy[i:i + 256]) for i in range(0, len(noisy), 256)]
    out = np.concatenate(outs)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 2.0
    assert anc.latency_ms > 0


def test_realtime_anc_end_to_end_with_reference():
    clean, noise = _tone_and_noise(seconds=1.0)
    noisy = clean + noise
    anc = RealTimeANC(sr=16000, onnx_path=None)
    outs = [anc.process(noisy[i:i + 256], noise[i:i + 256]) for i in range(0, len(noisy), 256)]
    out = np.concatenate(outs)
    assert np.isfinite(out).all()
    assert anc.real_time_factor >= 0.0


def test_synthetic_transients_are_bounded():
    burst = synth_impulsive_burst()
    rotor = synth_rotor_drone()
    assert np.isfinite(burst).all() and np.max(np.abs(burst)) <= 1.0 + 1e-6
    assert np.isfinite(rotor).all() and np.max(np.abs(rotor)) <= 1.0 + 1e-6


def test_synthetic_noise_bank_categories():
    bank = synthetic_noise_bank(seed=1)
    assert set(bank.keys()) >= {"stationary_engine", "nonstationary_rotor", "transient_impulsive", "mixed_siren"}
    for gen in bank.values():
        x = gen(0.5)
        assert np.isfinite(x).all()


def test_augment_pair_bounded():
    clean, noise = _tone_and_noise(seconds=1.28)
    rng = np.random.default_rng(0)
    noisy, target, ref = augment_pair(clean, noise, 16000, rng, snr_db=0.0)
    assert np.isfinite(noisy).all() and np.isfinite(target).all() and np.isfinite(ref).all()
    assert np.max(np.abs(noisy)) <= 1.0 + 1e-6


def test_extended_metrics_finite_on_clean_vs_noisy():
    clean, noise = _tone_and_noise(seconds=1.0)
    noisy = clean + noise
    enhanced = clean + 0.02 * noise
    m = extended_pair_metrics(clean, noisy, enhanced)
    assert np.isfinite(m["SegSNR_dB"])
    assert np.isfinite(m["FreqWeighted_SegSNR_dB"])
    assert m["SegSNR_dB"] > m["Noisy_SegSNR_dB"]  # enhancement should improve SegSNR here


def test_confidence_gate_alpha_matches_observed_colab_values():
    # Values independently observed during interactive Colab validation for this
    # project (see CHANGELOG_ADDITIONS.md) -- pinned here as a regression test so
    # the gating formula can't silently drift.
    from src.realtime.confidence_gate import confidence_gate_alpha
    expected = {-10: 1.0, -5: 1.0, 0: 0.75, 5: 0.5, 10: 0.25, 15: 0.05}
    for snr, exp in expected.items():
        assert abs(confidence_gate_alpha(snr) - exp) < 1e-9


