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
    dataset = ZDCDataset(config["data"]["output_dir"], split, int(config["model"]["max_tokens"]))
    device = torch.device("cuda" if config["training"]["device"] == "cuda" or (config["training"]["device"] == "auto" and torch.cuda.is_available()) else "cpu")
    model = SparseMambaPhotonRegressor(**{key: config["model"][key] for key in ("dim", "layers", "d_state", "d_conv", "expand", "dropout")}, energy_reference=tuple(dataset.manifest["detector_energy_reference"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    predicted, truth = [], []
    for batch in DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), collate_fn=collate):
        predicted.append(model(batch["features"].to(device), batch["mask"].to(device)).cpu().numpy()); truth.append(batch["targets"].numpy())
    predicted, truth = np.concatenate(predicted), np.concatenate(truth)
    energy_pred, energy_truth = np.exp(predicted[:, 0]), np.exp(truth[:, 0])
    relative = (energy_pred - energy_truth) / energy_truth
    records = []
    edges = np.asarray(config["evaluation"]["energy_bins_gev"], dtype=float)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (energy_truth >= low) & (energy_truth < high)
        residual = relative[selected]
        records.append({"energy_low_gev": float(low), "energy_high_gev": float(high), "n": int(selected.sum()), "relative_bias": float(np.mean(residual)) if len(residual) else None, "relative_rms": float(np.sqrt(np.mean(residual**2))) if len(residual) else None, "relative_p16": float(np.quantile(residual, .16)) if len(residual) else None, "relative_p84": float(np.quantile(residual, .84)) if len(residual) else None, "gaussian": robust_gaussian(residual, int(config["evaluation"]["min_fit_events"]))})
    cart_pred, cart_truth = target_to_cartesian(predicted), target_to_cartesian(truth)
    output = {"split": ("train", "val", "test")[split], "n_events": int(len(truth)), "global": {"energy_relative_bias": float(relative.mean()), "energy_relative_rms": float(np.sqrt(np.mean(relative**2))), "theta_x_mae_mrad": float(np.mean(np.abs(predicted[:, 1] - truth[:, 1])) * 1000), "theta_y_mae_mrad": float(np.mean(np.abs(predicted[:, 2] - truth[:, 2])) * 1000), "cartesian_rmse": np.sqrt(np.mean((cart_pred - cart_truth) ** 2, axis=0)).tolist()}, "energy_bins": records}
    np.savez_compressed(run_dir / f"predictions_{output['split']}.npz", prediction=predicted, truth=truth, energy=energy_truth, relative_energy_residual=relative)
    (run_dir / f"evaluation_{output['split']}.json").write_text(json.dumps(output, indent=2) + "\n")
    return run_dir
