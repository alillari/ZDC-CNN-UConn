from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader

from .contracts import target_to_cartesian
from .data import ZDCDataset
from .model import SparseMambaPhotonRegressor, collate
from .ninja_dataset import NinjaPhotonDataset
from .train import _split_limits


def _gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def robust_gaussian(values: np.ndarray, min_events: int) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if len(values) < min_events: return {"reported": False, "reason": "too_few_events", "n": int(len(values))}
    center, width = np.median(values), 1.4826 * np.median(np.abs(values - np.median(values)))
    if not np.isfinite(width) or width <= 0: return {"reported": False, "reason": "zero_or_nonfinite_width", "n": int(len(values))}
    core = values[np.abs(values - center) <= 2.5 * width]
    counts, edges = np.histogram(core, bins="auto")
    centers = (edges[1:] + edges[:-1]) / 2
    try:
        fitted, covariance = curve_fit(_gaussian, centers, counts, p0=(counts.max(), center, width), maxfev=10_000)
        sigma = abs(float(fitted[2]))
        if not np.isfinite(sigma) or sigma == 0 or not np.all(np.isfinite(covariance)):
            raise RuntimeError("invalid covariance or sigma")
        return {"reported": True, "n": int(len(values)), "core_fraction": float(len(core) / len(values)), "mean": float(fitted[1]), "sigma": sigma}
    except Exception as exc:
        return {"reported": False, "reason": str(exc), "n": int(len(values))}


@torch.no_grad()
def evaluate(config: dict[str, Any], run_dir: str | Path, split: int = 2) -> Path:
    run_dir = Path(run_dir); checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    max_tokens_value = config["model"].get("max_tokens")
    max_tokens = None if max_tokens_value is None else int(max_tokens_value)
    manifest = checkpoint["manifest"]
    roots = [str(path) for path in config["data"].get("ninja_roots", [])]
    if manifest.get("format") == "mmap_ninja":
        if not roots:
            roots = manifest["roots"]
        statistics = manifest["statistics"]
        dataset = NinjaPhotonDataset(roots, split, int(config["seed"]), float(config["split"]["train_fraction"]), float(config["split"]["val_fraction"]), statistics, max_tokens, _split_limits(manifest.get("max_events_per_split"))[split])
    else:
        dataset = ZDCDataset(config["data"]["output_dir"], split, max_tokens)
    if not len(dataset):
        raise RuntimeError(f"Requested evaluation split {split} is empty")
    device = torch.device("cuda" if config["training"]["device"] == "cuda" or (config["training"]["device"] == "auto" and torch.cuda.is_available()) else "cpu")
    model_keys = ("dim", "layers", "d_state", "d_conv", "expand", "dropout", "position_frequencies", "input_normalization", "event_summary", "target_mode", "readout_mode")
    model_args = {key: config["model"][key] for key in model_keys if key in config["model"]}
    energy_reference = manifest.get("energy_reference", manifest.get("detector_energy_reference"))
    model = SparseMambaPhotonRegressor(**model_args, energy_reference=tuple(energy_reference), coordinate_min=manifest.get("coordinate_min"), coordinate_max=manifest.get("coordinate_max"), global_coordinate_min=manifest.get("global_coordinate_min"), global_coordinate_max=manifest.get("global_coordinate_max")).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    predicted, truth = [], []
    for batch in DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), collate_fn=collate):
        summaries = batch["summaries"]
        summaries = summaries.to(device) if summaries is not None else None
        predicted.append(model(batch["features"].to(device), batch["mask"].to(device), summaries).cpu().numpy()); truth.append(batch["targets"].numpy())
    predicted, truth = np.concatenate(predicted), np.concatenate(truth)
    target_mode = str(config["model"].get("target_mode", "joint"))
    prediction_full = np.full((len(predicted), 3), np.nan, dtype=np.float32)
    if target_mode == "joint": prediction_full = predicted
    elif target_mode == "energy": prediction_full[:, 0] = predicted[:, 0]
    elif target_mode == "angles": prediction_full[:, 1:] = predicted
    else: raise ValueError("model.target_mode must be 'joint', 'energy', or 'angles'")
    records = []
    energy_truth = np.exp(truth[:, 0])
    relative = np.full(len(truth), np.nan, dtype=np.float32)
    energy_metrics = {"energy_relative_bias": None, "energy_relative_rms": None}
    if target_mode in ("joint", "energy"):
        energy_pred = np.exp(prediction_full[:, 0]); relative = (energy_pred - energy_truth) / energy_truth
        energy_metrics = {"energy_relative_bias": float(relative.mean()), "energy_relative_rms": float(np.sqrt(np.mean(relative**2)))}
        edges = np.asarray(config["evaluation"]["energy_bins_gev"], dtype=float)
        for low, high in zip(edges[:-1], edges[1:]):
            selected = (energy_truth >= low) & (energy_truth < high)
            residual = relative[selected]
            records.append({"energy_low_gev": float(low), "energy_high_gev": float(high), "n": int(selected.sum()), "relative_bias": float(np.mean(residual)) if len(residual) else None, "relative_rms": float(np.sqrt(np.mean(residual**2))) if len(residual) else None, "relative_p16": float(np.quantile(residual, .16)) if len(residual) else None, "relative_p84": float(np.quantile(residual, .84)) if len(residual) else None, "gaussian": robust_gaussian(residual, int(config["evaluation"]["min_fit_events"]))})
    angle_metrics = {"theta_x_mae_mrad": None, "theta_y_mae_mrad": None}
    if target_mode in ("joint", "angles"):
        angle_metrics = {"theta_x_mae_mrad": float(np.mean(np.abs(prediction_full[:, 1] - truth[:, 1])) * 1000), "theta_y_mae_mrad": float(np.mean(np.abs(prediction_full[:, 2] - truth[:, 2])) * 1000)}
    cartesian_rmse = None
    if target_mode == "joint":
        cart_pred, cart_truth = target_to_cartesian(prediction_full), target_to_cartesian(truth)
        cartesian_rmse = np.sqrt(np.mean((cart_pred - cart_truth) ** 2, axis=0)).tolist()
    output = {"split": ("train", "val", "test")[split], "n_events": int(len(truth)), "global": {**energy_metrics, **angle_metrics, "cartesian_rmse": cartesian_rmse}, "energy_bins": records}
    np.savez_compressed(run_dir / f"predictions_{output['split']}.npz", prediction=prediction_full, truth=truth, energy=energy_truth, relative_energy_residual=relative)
    (run_dir / f"evaluation_{output['split']}.json").write_text(json.dumps(output, indent=2) + "\n")
    return run_dir
