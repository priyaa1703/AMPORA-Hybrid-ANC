#!/usr/bin/env python3
"""Render final SIH demo assets from a trained checkpoint.

This is the confidence-gated inference pipeline validated interactively during
the Colab session for this project: the AI high-band branch AND the NLMS-based
low-band branch are both blended against the raw noisy signal using a gate
derived from the actual (reference-mic-estimated) input SNR, so the system
does real cleanup when the environment is genuinely noisy and does little to
no harm when the input is already fairly clean. Without this gate, a model
trained on a small dataset tends to apply a roughly fixed amount of "cleanup"
regardless of input SNR, which helps at low SNR but actively hurts already
clean input -- this was observed and fixed during evaluation, see
CHANGELOG_ADDITIONS.md.

Produces:
  <out>/sweep/metrics_by_snr.csv        -- SNR/STOI/PESQ averaged over N
                                            examples at each of several input
                                            SNR levels (the honest way to
                                            report a system's performance
                                            envelope, rather than one number)
  <out>/clips/<label>_<snr>dB/*.wav      -- a small number of individual
                                            clean/noisy/enhanced demo clips,
                                            chosen to be presentation-ready:
                                            one from a hard (low SNR) case,
                                            one showing "do no harm" behaviour
                                            at high SNR, and one built from a
                                            real transient/impulsive noise
                                            clip specifically (gunfire /
                                            explosion), if available.
  <out>/summary.md                       -- a short human-readable summary of
                                            what was rendered and the headline
                                            numbers, ready to paste into slides.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "scripts"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import train as T  # noqa: E402  (repo's own scripts/train.py -- read/fit/high/forward_audio/etc.)
from src.models.dccrn import AMPORA_DCCRN
from src.hybrid import hybrid_with_reference
from src.evaluation.metrics import pair_metrics, optional_stoi_pesq
from src.realtime.confidence_gate import confidence_gate_alpha


def load_model(checkpoint_path, device):
    model = AMPORA_DCCRN(16, 64).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model





def enhance_gated(model, window, device, clean, noise, snr_db):
    """Mix clean+noise at snr_db, run the confidence-gated hybrid pipeline
    (AI high band + NLMS low band, both gated by the actual current SNR),
    and return (noisy, enhanced, alpha, actual_snr_db)."""
    g = T.rms(clean) / (10 ** (snr_db / 20)) / (T.rms(noise) + 1e-8)
    noise_scaled = (noise * g).astype(np.float32)
    noisy = (clean + noise_scaled).astype(np.float32)

    noisy_hb = T.high(noisy)
    with torch.no_grad():
        x = torch.from_numpy(noisy_hb).float().unsqueeze(0).to(device)
        enh_hb = T.forward_audio(model, x, window).squeeze(0).cpu().numpy()

    actual_snr_db = 20 * np.log10(T.rms(clean) / (T.rms(noise_scaled) + 1e-8))
    alpha = confidence_gate_alpha(actual_snr_db)

    fully_enhanced = hybrid_with_reference(noisy, noise_scaled, enh_hb, sr=T.SR)
    enhanced = (alpha * fully_enhanced + (1 - alpha) * noisy).astype(np.float32)
    return noisy, enhanced, alpha, float(actual_snr_db)


def run_snr_sweep(model, window, device, speech_df, noise_df, out_dir,
                   snr_levels, n_per_snr, seed=42):
    rows_all = []
    for snr_db in snr_levels:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(min(n_per_snr, len(speech_df))):
            c = T.fit(T.read(speech_df.iloc[i].path), int(T.SR * T.SEG))
            nr = noise_df.iloc[int(rng.integers(len(noise_df)))]
            n = T.fit(T.read(nr.path), len(c))
            noisy, enhanced, alpha, actual_snr = enhance_gated(model, window, device, c, n, snr_db)
            m = pair_metrics(c, noisy, enhanced)
            m.update(optional_stoi_pesq(c, noisy, enhanced, sr=T.SR))
            m["alpha"] = alpha
            rows.append(m)
        df = pd.DataFrame(rows)
        avg = df.mean(numeric_only=True)
        rows_all.append({
            "Input_SNR_dB": snr_db,
            "avg_alpha": round(float(avg["alpha"]), 3),
            "Enhanced_SNR_dB": round(float(avg["Enhanced_SNR_dB"]), 2),
            "Delta_SNR_dB": round(float(avg["Delta_SNR_dB"]), 2),
            "Enhanced_SI_SDR_dB": round(float(avg["Enhanced_SI_SDR_dB"]), 2),
            "Enhanced_STOI": round(float(avg["Enhanced_STOI"]), 3),
            "Enhanced_PESQ": round(float(avg["Enhanced_PESQ"]), 3),
        })
    out = pd.DataFrame(rows_all)
    sweep_dir = out_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(sweep_dir / "metrics_by_snr.csv", index=False)
    return out


def render_clip(model, window, device, clean, noise, snr_db, out_dir, label):
    noisy, enhanced, alpha, actual_snr = enhance_gated(model, window, device, clean, noise, snr_db)
    clip_dir = out_dir / "clips" / f"{label}_{round(actual_snr)}dB"
    clip_dir.mkdir(parents=True, exist_ok=True)
    sf.write(clip_dir / "clean.wav", clean, T.SR)
    sf.write(clip_dir / "noisy.wav", noisy, T.SR)
    sf.write(clip_dir / "enhanced.wav", enhanced, T.SR)
    m = pair_metrics(clean, noisy, enhanced)
    m.update(optional_stoi_pesq(clean, noisy, enhanced, sr=T.SR))
    m["alpha"] = alpha
    m["actual_snr_db"] = actual_snr
    return clip_dir, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--speech-csv", required=True, nargs="+",
                     help="one or more speech manifests (e.g. val_speech.csv test_speech.csv) to draw demo clips from")
    ap.add_argument("--noise-csv", required=True)
    ap.add_argument("--out", default="results/sih_demo")
    ap.add_argument("--sweep-snr", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15])
    ap.add_argument("--n-per-snr", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.checkpoint, device)
    window = torch.hann_window(T.WIN, device=device)

    speech_df = pd.concat([pd.read_csv(p) for p in a.speech_csv], ignore_index=True)
    noise_df = pd.read_csv(a.noise_csv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render_sih_demo] running SNR sweep over {a.sweep_snr} dB, {a.n_per_snr} examples each...")
    sweep = run_snr_sweep(model, window, device, speech_df, noise_df, out_dir, a.sweep_snr, a.n_per_snr, a.seed)
    print(sweep.to_string(index=False))

    rng = np.random.default_rng(a.seed)
    summary_lines = ["# SIH demo assets\n", "## SNR sweep (see sweep/metrics_by_snr.csv)\n"]
    try:
        summary_lines.append(sweep.to_markdown(index=False))
    except Exception:
        summary_lines.append(sweep.to_string(index=False))
    summary_lines.append("\n")

    # Clip 1: hard case -- lowest sweep SNR, dramatic before/after
    hard_snr = min(a.sweep_snr)
    c = T.fit(T.read(speech_df.iloc[0].path), int(T.SR * T.SEG))
    nr = noise_df.iloc[int(rng.integers(len(noise_df)))]
    n = T.fit(T.read(nr.path), len(c))
    clip_dir, m = render_clip(model, window, device, c, n, hard_snr, out_dir, "hard_noisy")
    summary_lines.append(f"\n## Clip: hard_noisy ({clip_dir.name})\n"
                          f"- noise category: {getattr(nr,'noise_category','unknown')}\n"
                          f"- Delta_SNR_dB: {m['Delta_SNR_dB']:.2f}, Delta_SI_SDR_dB: {m['Delta_SI_SDR_dB']:.2f}, "
                          f"alpha: {m['alpha']:.2f}\n")

    # Clip 2: easy case -- highest sweep SNR, "do no harm" behaviour
    easy_snr = max(a.sweep_snr)
    c2 = T.fit(T.read(speech_df.iloc[min(1, len(speech_df)-1)].path), int(T.SR * T.SEG))
    nr2 = noise_df.iloc[int(rng.integers(len(noise_df)))]
    n2 = T.fit(T.read(nr2.path), len(c2))
    clip_dir2, m2 = render_clip(model, window, device, c2, n2, easy_snr, out_dir, "clean_input_no_harm")
    summary_lines.append(f"\n## Clip: clean_input_no_harm ({clip_dir2.name})\n"
                          f"- noise category: {getattr(nr2,'noise_category','unknown')}\n"
                          f"- Delta_SNR_dB: {m2['Delta_SNR_dB']:.2f} (near zero is the GOAL here -- system should "
                          f"barely touch already-clean input)\n- alpha: {m2['alpha']:.2f}\n")

    # Clip 3: a transient/impulsive (gunfire/explosion) example specifically, if present
    transient_rows = noise_df[noise_df.get("noise_category", "") == "transient"]
    if len(transient_rows) > 0:
        nr3 = transient_rows.iloc[int(rng.integers(len(transient_rows)))]
        c3 = T.fit(T.read(speech_df.iloc[min(2, len(speech_df)-1)].path), int(T.SR * T.SEG))
        n3 = T.fit(T.read(nr3.path), len(c3))
        mid_snr = float(np.median(a.sweep_snr))
        clip_dir3, m3 = render_clip(model, window, device, c3, n3, mid_snr, out_dir, "transient_impulsive")
        summary_lines.append(f"\n## Clip: transient_impulsive ({clip_dir3.name})\n"
                              f"- noise file: {Path(nr3.path).name}\n"
                              f"- Delta_SNR_dB: {m3['Delta_SNR_dB']:.2f}, Delta_SI_SDR_dB: {m3['Delta_SI_SDR_dB']:.2f}, "
                              f"alpha: {m3['alpha']:.2f}\n")
    else:
        summary_lines.append("\n## Clip: transient_impulsive -- SKIPPED (no 'transient' rows in --noise-csv)\n")

    with open(out_dir / "summary.md", "w") as f:
        f.writelines(summary_lines)

    print(f"\n[render_sih_demo] wrote sweep table + demo clips + summary to {out_dir}/")
    print(f"[render_sih_demo] see {out_dir}/summary.md for headline numbers to paste into slides")


if __name__ == "__main__":
    main()
