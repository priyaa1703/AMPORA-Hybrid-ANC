#!/usr/bin/env python3
"""Run a trained checkpoint (from scripts/train.py) on held-out speech+noise and
write clean/noisy/enhanced triples + a manifest, so scripts/evaluate.py and any
demo/report tooling has real before/after audio to work with.

This is the "checkpoint -> demo-ready enhanced audio" step that a fresh
scripts/train.py run does not produce by itself.
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
import numpy as np, pandas as pd, soundfile as sf
import torch
import train as T  # reuses read/fit/mix/high/stft/istft/forward_audio, SR/NFFT/etc from train.py
from src.models.dccrn import AMPORA_DCCRN


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--speech-csv", required=True, help="a speech manifest with a 'path' column, "
                     "e.g. data/processed*/val_speech.csv or data/processed*/test_speech.csv")
    ap.add_argument("--noise-csv", required=True, help="a noise manifest with 'path,noise_category' "
                     "columns, e.g. data/processed*/noise.csv")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--snr-db", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMPORA_DCCRN(16, 64).to(device)
    ckpt = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    window = torch.hann_window(T.WIN, device=device)

    speech = pd.read_csv(a.speech_csv)
    noise = pd.read_csv(a.noise_csv)
    rng = np.random.default_rng(a.seed)

    out = Path(a.out)
    for sub in ["clean", "noisy", "enhanced"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    n = min(a.n_examples, len(speech))
    for i in range(n):
        c = T.fit(T.read(speech.iloc[i].path), int(T.SR * T.SEG))
        nr = noise.iloc[int(rng.integers(len(noise)))]
        nz = T.fit(T.read(nr.path), len(c))
        noisy = T.mix(c, nz, a.snr_db)
        clean_hb = T.high(c)
        noisy_hb = T.high(noisy)

        with torch.no_grad():
            x = torch.from_numpy(noisy_hb).float().unsqueeze(0).to(device)
            enh = T.forward_audio(model, x, window).squeeze(0).cpu().numpy()

        stem = f"ex{i:02d}_{Path(speech.iloc[i].path).stem}"
        sf.write(out / "clean" / f"{stem}.wav", clean_hb, T.SR)
        sf.write(out / "noisy" / f"{stem}.wav", noisy_hb, T.SR)
        sf.write(out / "enhanced" / f"{stem}.wav", enh, T.SR)
        rows.append({
            "clean": str(out / "clean" / f"{stem}.wav"),
            "noisy": str(out / "noisy" / f"{stem}.wav"),
            "enhanced": str(out / "enhanced" / f"{stem}.wav"),
            "noise_category": getattr(nr, "noise_category", "unknown"),
            "snr_db": a.snr_db,
        })

    manifest = out / "demo_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Wrote {n} clean/noisy/enhanced triples -> {manifest}")
    print(f"Next: python scripts/evaluate.py {manifest} --out {out}/metrics.csv --extended --high-band")


if __name__ == "__main__":
    main()
