"""Training-only two-detector calorimetric energy calibration baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import robust_gaussian
from .ninja_dataset import NinjaPhotonDataset, fit_statistics
from .train import _split_limits


def _energy_sums(dataset: NinjaPhotonDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return raw WSi/SiPM energy sums and photon truth energy for a split."""
    sums, truth = [], []
    for features, detector, targets, index, _, _ in dataset.items:
        event = np.asarray(features[index])
        det = np.asarray(detector[index])
        sums.append((float(event[det == 0, 0].sum()), float(event[det == 1, 0].sum())))
        truth.append(float(np.exp(np.asarray(targets[index])[0])))
    return np.asarray(sums, dtype=np.float64), np.asarray(truth, dtype=np.float64)


def fit_sum_baseline(config: dict[str, Any], run_dir: str | Path) -> Path:
    """Fit E=a+b_WSi S_WSi+b_SiPM S_SiPM on train; report held-out test."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    roots = [str(path) for path in config["data"].get("ninja_roots", [])]
    if not roots:
        raise ValueError("sum-baseline requires data.ninja_roots")
    split = config["split"]; seed = int(config["seed"])
    statistics = fit_statistics(roots, seed, float(split["train_fraction"]), float(split["val_fraction"]))
    limits = _split_limits(config["data"].get("max_events_per_split"))
    make = lambda partition, limit: NinjaPhotonDataset(roots, partition, seed, float(split["train_fraction"]), float(split["val_fraction"]), statistics, None, limit)
    train_set, test_set = make(0, limits[0]), make(2, limits[2])
    train_sums, train_energy = _energy_sums(train_set)
    test_sums, test_energy = _energy_sums(test_set)
    coefficients, _, _, _ = np.linalg.lstsq(np.column_stack((np.ones(len(train_sums)), train_sums)), train_energy, rcond=None)
    prediction = np.column_stack((np.ones(len(test_sums)), test_sums)) @ coefficients
    relative = (prediction - test_energy) / test_energy
    records = []
    edges = np.asarray(config["evaluation"]["energy_bins_gev"], dtype=float)
    for low, high in zip(edges[:-1], edges[1:]):
        values = relative[(test_energy >= low) & (test_energy < high)]
        records.append({"energy_low_gev": float(low), "energy_high_gev": float(high), "n": int(len(values)), "relative_bias": float(values.mean()) if len(values) else None, "relative_rms": float(np.sqrt(np.mean(values ** 2))) if len(values) else None, "relative_p16": float(np.quantile(values, .16)) if len(values) else None, "relative_p84": float(np.quantile(values, .84)) if len(values) else None, "gaussian": robust_gaussian(values, int(config["evaluation"]["min_fit_events"]))})
    output = {"split": "test", "n_events": int(len(test_energy)), "global": {"energy_relative_bias": float(relative.mean()), "energy_relative_rms": float(np.sqrt(np.mean(relative ** 2))), "theta_x_mae_mrad": None, "theta_y_mae_mrad": None, "cartesian_rmse": None}, "energy_bins": records}
    prediction_full = np.full((len(test_energy), 3), np.nan, dtype=np.float32); prediction_full[:, 0] = np.log(np.maximum(prediction, 1e-12))
    truth_full = np.full((len(test_energy), 3), np.nan, dtype=np.float32); truth_full[:, 0] = np.log(test_energy)
    (run_dir / "sum_baseline.json").write_text(json.dumps({"coefficients": {"intercept": float(coefficients[0]), "wsi_energy": float(coefficients[1]), "sipm_energy": float(coefficients[2])}, "n_train": len(train_energy), "n_test": len(test_energy), "config": config}, indent=2) + "\n")
    (run_dir / "evaluation_test.json").write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(run_dir / "predictions_test.npz", prediction=prediction_full, truth=truth_full, energy=test_energy, relative_energy_residual=relative)
    return run_dir
