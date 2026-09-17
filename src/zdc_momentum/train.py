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
from .ninja_dataset import NinjaPhotonDataset, fit_statistics


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _split_limits(value: Any) -> tuple[int | None, int | None, int | None]:
    if value is None: return (None, None, None)
    if isinstance(value, (list, tuple)):
        if len(value) != 3: raise ValueError("max_events_per_split sequence must have train, val, and test entries")
        return tuple(None if item is None else int(item) for item in value)
    if isinstance(value, dict):
        return tuple(None if value.get(name) is None else int(value[name]) for name in ("train", "val", "test"))
    return (int(value), int(value), int(value))


def regression_loss(prediction: torch.Tensor, target: torch.Tensor, target_std: torch.Tensor, *, loss: str = "huber", energy_objective: str = "log", relative_energy_scale: float = 0.10, target_mode: str = "joint") -> torch.Tensor:
    """Joint three-target loss with an independently selectable energy term.

    ``energy_objective='relative'`` uses the physically evaluated fractional
    residual, (exp(predicted_log_energy) - true_energy) / true_energy.  The
    fixed scale expresses the Huber knee in fractional-energy units and keeps
    that term commensurate with the standardized angular residuals.
    """
    if loss not in ("huber", "mae"):
        raise ValueError("training.loss must be 'huber' or 'mae'")
    if energy_objective not in ("log", "relative"):
        raise ValueError("training.energy_objective must be 'log' or 'relative'")
    if target_mode not in ("joint", "energy", "angles"):
        raise ValueError("model.target_mode must be 'joint', 'energy', or 'angles'")
    if relative_energy_scale <= 0:
        raise ValueError("training.relative_energy_scale must be positive")

    def pointwise(residual: torch.Tensor) -> torch.Tensor:
        if loss == "mae":
            return residual.abs().mean()
        return torch.nn.functional.huber_loss(residual, torch.zeros_like(residual))

    if target_mode in ("joint", "energy") and energy_objective == "log":
        energy_loss = pointwise((prediction[:, 0] - target[:, 0]) / target_std[0])
    elif target_mode in ("joint", "energy"):
        true_energy = torch.exp(target[:, 0])
        relative_residual = (torch.exp(prediction[:, 0]) - true_energy) / true_energy
        energy_loss = pointwise(relative_residual / relative_energy_scale)
    if target_mode == "joint":
        angle_prediction = prediction[:, 1:]
    elif target_mode == "angles":
        angle_prediction = prediction
    if target_mode in ("joint", "angles"):
        angle_loss = pointwise((angle_prediction - target[:, 1:]) / target_std[1:])
    if target_mode == "energy":
        return energy_loss
    if target_mode == "angles":
        return angle_loss
    # Preserve the baseline's equal per-target weighting: one energy and two
    # angular components, rather than weighting the energy term as one half.
    return (energy_loss + 2.0 * angle_loss) / 3.0


def train(config: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]))
    root = config["data"]["output_dir"]
    max_tokens_value = config["model"].get("max_tokens")
    max_tokens = None if max_tokens_value is None else int(max_tokens_value)
    roots = [str(path) for path in config["data"].get("ninja_roots", [])]
    split_limits = _split_limits(config["data"].get("max_events_per_split"))
    split = config["split"]
    if roots:
        statistics = fit_statistics(roots, int(config["seed"]), float(split["train_fraction"]), float(split["val_fraction"]))
        train_set = NinjaPhotonDataset(roots, 0, int(config["seed"]), float(split["train_fraction"]), float(split["val_fraction"]), statistics, max_tokens, split_limits[0])
        val_set = NinjaPhotonDataset(roots, 1, int(config["seed"]), float(split["train_fraction"]), float(split["val_fraction"]), statistics, max_tokens, split_limits[1])
        test_set = NinjaPhotonDataset(roots, 2, int(config["seed"]), float(split["train_fraction"]), float(split["val_fraction"]), statistics, max_tokens, split_limits[2])
        split_counts = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}
        if not all(split_counts.values()):
            raise RuntimeError(f"Ninja split has an empty partition: {split_counts}. Increase the pilot sample size or change the seed.")
        manifest = {"format": "mmap_ninja", "roots": roots, "statistics": statistics, "split_counts": split_counts, "max_events_per_split": split_limits, **statistics}
        (run_dir / "normalization.json").write_text(json.dumps(statistics, indent=2) + "\n")
    else:
        train_set, val_set = ZDCDataset(root, 0, max_tokens), ZDCDataset(root, 1, max_tokens)
        manifest = train_set.manifest
        if config["model"].get("event_summary", "none") != "none":
            raise ValueError("event_summary modes require mmap_ninja input")
    device = _device(str(config["training"]["device"]))
    model_keys = ("dim", "layers", "d_state", "d_conv", "expand", "dropout", "position_frequencies", "input_normalization", "event_summary", "target_mode", "readout_mode")
    model_args = {key: config["model"][key] for key in model_keys if key in config["model"]}
    energy_reference = manifest.get("energy_reference", manifest.get("detector_energy_reference"))
    model = SparseMambaPhotonRegressor(**model_args, energy_reference=tuple(energy_reference), coordinate_min=manifest.get("coordinate_min"), coordinate_max=manifest.get("coordinate_max"), global_coordinate_min=manifest.get("global_coordinate_min"), global_coordinate_max=manifest.get("global_coordinate_max")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    loaders = {"train": DataLoader(train_set, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"]["num_workers"]), collate_fn=collate), "val": DataLoader(val_set, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]), collate_fn=collate)}
    mean, std = torch.tensor(manifest["target_mean"], device=device), torch.tensor(manifest["target_std"], device=device)
    loss_name = str(config["training"].get("loss", "huber"))
    energy_objective = str(config["training"].get("energy_objective", "log"))
    relative_energy_scale = float(config["training"].get("relative_energy_scale", 0.10))
    target_mode = str(config["model"].get("target_mode", "joint"))
    history, best = [], float("inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        record = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train"); total = 0.0; count = 0
            for batch in loaders[phase]:
                features, mask, target = (batch[key].to(device) for key in ("features", "mask", "targets"))
                summaries = batch["summaries"]
                summaries = summaries.to(device) if summaries is not None else None
                with torch.set_grad_enabled(phase == "train"):
                    prediction = model(features, mask, summaries)
                    objective = regression_loss(prediction, target, std, loss=loss_name, energy_objective=energy_objective, relative_energy_scale=relative_energy_scale, target_mode=target_mode)
                    total_loss = objective + 1e-3 * model.calibration_penalty()
                    if phase == "train": optimizer.zero_grad(set_to_none=True); total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                total += float(total_loss.detach()) * len(target); count += len(target)
            record[f"{phase}_loss"] = total / max(count, 1)
        history.append(record)
        if record["val_loss"] < best:
            best = record["val_loss"]
            torch.save({"model": model.state_dict(), "manifest": manifest, "config": config, "epoch": epoch, "val_loss": best}, run_dir / "best.pt")
    provenance = {"config": config, "manifest": manifest, "best_val_loss": best, "torch": torch.__version__, "device": str(device), "cuda": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "platform": platform.platform(), "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()}
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return run_dir
