from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ZDCDataset
from .model import SparseMambaPhotonRegressor, collate


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def train(config: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    root = config["data"]["output_dir"]; max_tokens = int(config["model"]["max_tokens"])
    train_set, val_set = ZDCDataset(root, 0, max_tokens), ZDCDataset(root, 1, max_tokens)
    manifest = train_set.manifest
    device = _device(str(config["training"]["device"]))
    model = SparseMambaPhotonRegressor(**{key: config["model"][key] for key in ("dim", "layers", "d_state", "d_conv", "expand", "dropout")}, energy_reference=tuple(manifest["detector_energy_reference"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    loaders = {"train": DataLoader(train_set, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"]["num_workers"]), collate_fn=collate), "val": DataLoader(val_set, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]), collate_fn=collate)}
    mean, std = torch.tensor(manifest["target_mean"], device=device), torch.tensor(manifest["target_std"], device=device)
    history, best = [], float("inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        record = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train"); total = 0.0; count = 0
            for batch in loaders[phase]:
                features, mask, target = (batch[key].to(device) for key in ("features", "mask", "targets"))
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model(features, mask)
                    loss = torch.nn.functional.huber_loss((prediction - target) / std, torch.zeros_like(target)) + 1e-3 * model.calibration_penalty()
                    if phase == "train": optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += float(loss.detach()) * len(target); count += len(target)
            record[f"{phase}_loss"] = total / max(count, 1)
        history.append(record)
        if record["val_loss"] < best:
            best = record["val_loss"]
            torch.save({"model": model.state_dict(), "manifest": manifest, "config": config, "epoch": epoch, "val_loss": best}, run_dir / "best.pt")
    provenance = {"config": config, "manifest": manifest, "best_val_loss": best, "torch": torch.__version__, "device": str(device), "cuda": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "platform": platform.platform(), "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return run_dir
