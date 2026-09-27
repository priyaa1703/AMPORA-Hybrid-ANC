"""
Persistent Google Drive checkpoint helpers for AMPORA Colab training.
Checkpoints live in:
My Drive/AMPORA-Hybrid-ANC/checkpoints/
"""

from pathlib import Path
import json
import os
import time

def mount_ampora_drive(project_name="AMPORA-Hybrid-ANC"):
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    root = Path("/content/drive/MyDrive") / project_name
    for name in ("checkpoints", "logs", "results", "models", "plots"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root

def save_checkpoint(model, optimizer, scheduler=None, epoch=None,
                    best_metric=None, history=None, config=None,
                    project_root=None):
    import torch
    if epoch is None:
        raise ValueError("epoch is required")
    root = Path(project_root or "/content/drive/MyDrive/AMPORA-Hybrid-ANC")
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "history": history or {},
        "config": config or {},
        "saved_at_unix": time.time(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    final_path = ckpt_dir / f"epoch_{int(epoch):04d}.pt"
    tmp_path = ckpt_dir / f".epoch_{int(epoch):04d}.pt.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)

    latest = ckpt_dir / "latest.json"
    latest_tmp = ckpt_dir / ".latest.json.tmp"
    latest_tmp.write_text(json.dumps({
        "latest_checkpoint": final_path.name,
        "epoch": int(epoch),
        "saved_at_unix": state["saved_at_unix"]
    }, indent=2))
    os.replace(latest_tmp, latest)
    return final_path

def load_latest_checkpoint(model, optimizer, scheduler=None,
                           project_root=None, map_location="cpu"):
    import torch
    root = Path(project_root or "/content/drive/MyDrive/AMPORA-Hybrid-ANC")
    ckpt_dir = root / "checkpoints"
    candidate = None

    latest = ckpt_dir / "latest.json"
    if latest.exists():
        try:
            name = json.loads(latest.read_text())["latest_checkpoint"]
            p = ckpt_dir / name
            if p.exists():
                candidate = p
        except Exception:
            pass

    if candidate is None:
        files = sorted(ckpt_dir.glob("epoch_*.pt"))
        candidate = files[-1] if files else None

    if candidate is None:
        return None

    state = torch.load(candidate, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state
