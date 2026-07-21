#!/usr/bin/env python
"""
After-the-fact momentum diagnostics for the ZDC CNN/Mamba comparison.

This script loads trained best-checkpoint files from a study output directory,
runs inference on an explicit split, and writes residual/calibration plots.
It is intentionally inference-only: it does not retrain either model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_zdc_mamba_vs_cnn_study import (
    MomentumWSIImageDataset,
    StackedWSICNNMomentumRegressor,
    load_event_splits,
)
from zdc_mamba_baseline import (
    PARTICLE_NAMES,
    MomentumTransform,
    SparseZDCMomentumDataset,
    ZDCMambaConfig,
    ZDCMambaRegressor,
    collate_sparse_zdc,
    extract_momentum_target,
)


MODEL_LABELS = {
    "stacked_conv2d": "Stacked Conv2d",
    "mamba_layer_hilbert": "Layer-Hilbert Mamba",
}


def build_config(run_config: Mapping[str, object], mamba_checkpoint: Optional[Mapping[str, object]]) -> ZDCMambaConfig:
    cfg = ZDCMambaConfig()
    target = run_config.get("target", {})
    if isinstance(target, Mapping):
        cfg.target.mode = str(target.get("mode", cfg.target.mode))
        cfg.target.magnitude_transform = str(
            target.get("magnitude_transform", cfg.target.magnitude_transform)
        )
        cfg.target.target_key = str(target.get("target_key", cfg.target.target_key))
        cfg.target.magnitude_mean = target.get("magnitude_mean")
        cfg.target.magnitude_std = target.get("magnitude_std")
        cfg.target.eps = float(target.get("eps", cfg.target.eps))

    loss = run_config.get("loss", {})
    if isinstance(loss, Mapping):
        cfg.lambda_p = float(loss.get("lambda_p", cfg.lambda_p))
        cfg.lambda_dir = float(loss.get("lambda_dir", cfg.lambda_dir))

    if mamba_checkpoint is not None:
        metadata = mamba_checkpoint.get("metadata", {})
        config = metadata.get("config", {}) if isinstance(metadata, Mapping) else {}
        tokenization = config.get("tokenization", {}) if isinstance(config, Mapping) else {}
        if isinstance(tokenization, Mapping):
            for key, value in tokenization.items():
                if hasattr(cfg.tokenization, key):
                    setattr(cfg.tokenization, key, value)

        model = config.get("model", {}) if isinstance(config, Mapping) else {}
        if isinstance(model, Mapping):
            cfg.model_dim = int(model.get("model_dim", cfg.model_dim))
            cfg.num_layers = int(model.get("num_layers", cfg.num_layers))
            cfg.d_state = int(model.get("d_state", cfg.d_state))
            cfg.d_conv = int(model.get("d_conv", cfg.d_conv))
            cfg.expand = int(model.get("expand", cfg.expand))

    return cfg


def load_run_config(output_dir: Path) -> Dict[str, object]:
    with (output_dir / "run_config.json").open() as stream:
        return json.load(stream)


def load_checkpoint(path: Path, device: str) -> Mapping[str, object]:
    return torch.load(path, map_location=device)


@torch.no_grad()
def predict_cnn(
    events: Sequence[Mapping[str, object]],
    cfg: ZDCMambaConfig,
    checkpoint: Mapping[str, object],
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = MomentumWSIImageDataset(events, cfg.target, cache_images=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = StackedWSICNNMomentumRegressor(cfg.target, output_mode=cfg.target.mode).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    truth = []
    for batch in loader:
        image = batch["image"].to(device)
        outputs = model(image)
        predictions.append(outputs["cartesian"].detach().cpu().numpy())
        truth.append(batch["momentum"].numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(truth, axis=0)


@torch.no_grad()
def predict_mamba(
    events: Sequence[Mapping[str, object]],
    cfg: ZDCMambaConfig,
    checkpoint: Mapping[str, object],
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = SparseZDCMomentumDataset(events, config=cfg, cache_tokens=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sparse_zdc,
    )
    model = ZDCMambaRegressor(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    truth = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        energy = batch["energy"].to(device)
        outputs = model(features, mask, energy)
        predictions.append(outputs["cartesian"].detach().cpu().numpy())
        truth.append(batch["momentum"].numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(truth, axis=0)


def event_observables(events: Sequence[Mapping[str, object]], cfg: ZDCMambaConfig) -> Dict[str, np.ndarray]:
    true_p = np.stack(
        [extract_momentum_target(event, cfg.target.target_key) for event in events],
        axis=0,
    ).astype(np.float64)
    values = {
        "true_magnitude": np.linalg.norm(true_p, axis=1),
        "total_wsi_energy": np.zeros(len(events), dtype=np.float64),
        "neutron_wsi_energy": np.zeros(len(events), dtype=np.float64),
    }
    for idx, event in enumerate(events):
        total = 0.0
        for name in PARTICLE_NAMES:
            energy = np.asarray(event[name].get("E", []), dtype=np.float64)
            finite_positive = energy[np.isfinite(energy) & (energy > 0)]
            energy_sum = float(finite_positive.sum())
            total += energy_sum
            if name == "neutron":
                values["neutron_wsi_energy"][idx] = energy_sum
        values["total_wsi_energy"][idx] = total
    return values


def residual_table(pred: np.ndarray, truth: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-8
    pred_mag = np.maximum(np.linalg.norm(pred, axis=1), eps)
    true_mag = np.maximum(np.linalg.norm(truth, axis=1), eps)
    pred_dir = pred / pred_mag[:, None]
    true_dir = truth / true_mag[:, None]
    dot = np.sum(pred_dir * true_dir, axis=1).clip(-1.0, 1.0)
    angular_mrad = np.arccos(dot) * 1000.0
    lambda_angle_mrad = np.arctan2(
        np.linalg.norm(truth[:, :2], axis=1),
        truth[:, 2],
    ) * 1000.0
    ratio = pred_mag / true_mag
    linear = pred_mag - true_mag
    relative = linear / true_mag
    log_ratio = np.log(ratio)
    return {
        "pred_magnitude": pred_mag,
        "true_magnitude": true_mag,
        "ratio": ratio,
        "linear_residual": linear,
        "relative_residual": relative,
        "log_residual": log_ratio,
        "angular_mrad": angular_mrad,
        "lambda_angle_mrad": lambda_angle_mrad,
    }


def summarize_residuals(name: str, residuals: Mapping[str, np.ndarray]) -> Dict[str, float]:
    ratio = residuals["ratio"]
    return {
        "model": name,
        "n_events": int(ratio.size),
        "linear_bias": float(np.mean(residuals["linear_residual"])),
        "linear_rms": float(np.std(residuals["linear_residual"])),
        "relative_bias": float(np.mean(residuals["relative_residual"])),
        "relative_rms": float(np.std(residuals["relative_residual"])),
        "log_bias": float(np.mean(residuals["log_residual"])),
        "log_rms": float(np.std(residuals["log_residual"])),
        "median_ratio": float(np.median(ratio)),
        "ratio_p16": float(np.percentile(ratio, 16)),
        "ratio_p84": float(np.percentile(ratio, 84)),
        "angular_mrad_mean": float(np.mean(residuals["angular_mrad"])),
        "angular_mrad_rms": float(np.std(residuals["angular_mrad"])),
        "angular_mrad_p68": float(np.percentile(residuals["angular_mrad"], 68)),
        "angular_mrad_p95": float(np.percentile(residuals["angular_mrad"], 95)),
    }


def write_metrics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "model",
        "n_events",
        "linear_bias",
        "linear_rms",
        "relative_bias",
        "relative_rms",
        "log_bias",
        "log_rms",
        "median_ratio",
        "ratio_p16",
        "ratio_p84",
        "angular_mrad_mean",
        "angular_mrad_rms",
        "angular_mrad_p68",
        "angular_mrad_p95",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_range(values: Sequence[np.ndarray], q_low: float, q_high: float) -> Tuple[float, float]:
    joined = np.concatenate([v[np.isfinite(v)] for v in values])
    lo = float(np.percentile(joined, q_low))
    hi = float(np.percentile(joined, q_high))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    centers = []
    means = []
    errors = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi)
        if mask.sum() < 3:
            continue
        vals = y[mask]
        centers.append(float(np.median(x[mask])))
        means.append(float(np.mean(vals)))
        errors.append(float(np.std(vals) / np.sqrt(vals.size)))
    return np.asarray(centers), np.asarray(means), np.asarray(errors)


def binned_angular_resolution(
    lambda_angle_mrad: np.ndarray,
    angular_mrad: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(lambda_angle_mrad) & np.isfinite(angular_mrad)
    x = lambda_angle_mrad[finite]
    y = angular_mrad[finite]
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    centers = []
    mean_values = []
    rms_values = []
    p68_values = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi)
        if mask.sum() < 3:
            continue
        vals = y[mask]
        centers.append(float(np.median(x[mask])))
        mean_values.append(float(np.mean(vals)))
        rms_values.append(float(np.std(vals)))
        p68_values.append(float(np.percentile(vals, 68)))
    return (
        np.asarray(centers),
        np.asarray(mean_values),
        np.asarray(rms_values),
        np.asarray(p68_values),
    )


def plot_calibration(path: Path, model_residuals: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    n_models = len(model_residuals)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    all_true = [res["true_magnitude"] for res in model_residuals.values()]
    all_pred = [res["pred_magnitude"] for res in model_residuals.values()]
    lo, hi = finite_range(all_true + all_pred, 0.5, 99.5)
    for ax, (name, res) in zip(axes[0], model_residuals.items()):
        ax.hexbin(
            res["true_magnitude"],
            res["pred_magnitude"],
            gridsize=55,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(MODEL_LABELS.get(name, name))
        ax.set_xlabel("true |p|")
        ax.set_ylabel("predicted |p|")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distributions(path: Path, model_residuals: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    specs = [
        ("linear_residual", "p_pred - p_true", 1.0, 1.0, 99.0),
        ("relative_residual", "(p_pred - p_true) / p_true", 1.0, 1.0, 99.0),
        ("log_residual", "log(p_pred / p_true)", 1.0, 1.0, 99.0),
        ("ratio", "p_pred / p_true", 1.0, 1.0, 99.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (key, xlabel, _, q_low, q_high) in zip(axes, specs):
        lo, hi = finite_range([res[key] for res in model_residuals.values()], q_low, q_high)
        bins = np.linspace(lo, hi, 80)
        for name, res in model_residuals.items():
            ax.hist(
                res[key],
                bins=bins,
                histtype="step",
                density=True,
                linewidth=1.8,
                label=MODEL_LABELS.get(name, name),
            )
        ax.axvline(0.0 if key != "ratio" else 1.0, color="black", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_binned_residuals(
    path: Path,
    observables: Mapping[str, np.ndarray],
    model_residuals: Mapping[str, Mapping[str, np.ndarray]],
    n_bins: int,
) -> None:
    panels = [
        ("true_magnitude", "true |p|", "relative_residual", "mean relative residual"),
        ("true_magnitude", "true |p|", "log_residual", "mean log residual"),
        ("neutron_wsi_energy", "neutron WSi visible energy", "relative_residual", "mean relative residual"),
        ("total_wsi_energy", "total WSi visible energy", "relative_residual", "mean relative residual"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (x_key, x_label, y_key, y_label) in zip(axes, panels):
        x = observables[x_key]
        if "energy" in x_key:
            x = np.log1p(x)
            x_label = f"log1p({x_label})"
        for name, res in model_residuals.items():
            centers, means, errors = binned_mean(x, res[y_key], n_bins)
            ax.errorbar(
                centers,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.5,
                capsize=2,
                label=MODEL_LABELS.get(name, name),
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_integrated_angular_resolution(
    path: Path,
    model_residuals: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = finite_range([res["angular_mrad"] for res in model_residuals.values()], 0.0, 99.5)
    bins = np.linspace(max(0.0, lo), hi, 90)
    for name, res in model_residuals.items():
        angular = res["angular_mrad"]
        label = (
            f"{MODEL_LABELS.get(name, name)} "
            f"mean={np.mean(angular):.3f} mrad, "
            f"rms={np.std(angular):.3f} mrad"
        )
        ax.hist(
            angular,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("angular separation [mrad]")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_angular_resolution_vs_lambda_angle(
    path: Path,
    model_residuals: Mapping[str, Mapping[str, np.ndarray]],
    n_bins: int,
    centerline_mrad: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for name, res in model_residuals.items():
        centers, means, rms_values, p68_values = binned_angular_resolution(
            res["lambda_angle_mrad"],
            res["angular_mrad"],
            n_bins,
        )
        label = MODEL_LABELS.get(name, name)
        axes[0].plot(centers, means, marker="o", linewidth=1.5, label=label)
        axes[1].plot(centers, p68_values, marker="o", linewidth=1.5, label=f"{label} p68")
        axes[1].plot(centers, rms_values, marker="s", linewidth=1.2, linestyle="--", label=f"{label} rms")

    for ax in axes:
        ax.axvline(centerline_mrad, color="black", linewidth=1.2, linestyle=":")
        ax.set_xlabel("true Lambda angle from +z [mrad]")
        ax.legend()
    axes[0].set_ylabel("mean angular separation [mrad]")
    axes[1].set_ylabel("angular resolution [mrad]")
    axes[0].set_title("Integrated in equal-population angle bins")
    axes[1].set_title("p68 and RMS by Lambda angle")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-pickle", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path("zdc_mamba_vs_cnn_outputs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_LABELS), default=tuple(MODEL_LABELS))
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--centerline-mrad", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir or (args.study_dir / "momentum_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    run_config = load_run_config(args.study_dir)
    train_events, val_events, test_events = load_event_splits(
        args.splits_pickle,
        allow_random_split=False,
        seed=12345,
        train_fraction=0.7,
        val_fraction=0.15,
    )
    split_events = {"train": train_events, "val": val_events, "test": test_events}[args.split]

    mamba_path = args.study_dir / "mamba_layer_hilbert_best.pt"
    cnn_path = args.study_dir / "cnn_magnitude_direction_best.pt"
    mamba_checkpoint = load_checkpoint(mamba_path, device) if mamba_path.exists() else None
    cfg = build_config(run_config, mamba_checkpoint)
    if cfg.target.magnitude_transform == "standardize" and cfg.target.magnitude_mean is None:
        MomentumTransform(cfg.target).fit(train_events)

    model_residuals = {}
    metrics = []
    if "stacked_conv2d" in args.models:
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing CNN checkpoint: {cnn_path}")
        pred, truth = predict_cnn(
            split_events,
            cfg,
            load_checkpoint(cnn_path, device),
            args.batch_size,
            args.num_workers,
            device,
        )
        model_residuals["stacked_conv2d"] = residual_table(pred, truth)
        metrics.append(summarize_residuals("stacked_conv2d", model_residuals["stacked_conv2d"]))

    if "mamba_layer_hilbert" in args.models:
        if mamba_checkpoint is None:
            raise FileNotFoundError(f"Missing Mamba checkpoint: {mamba_path}")
        pred, truth = predict_mamba(
            split_events,
            cfg,
            mamba_checkpoint,
            args.batch_size,
            args.num_workers,
            device,
        )
        model_residuals["mamba_layer_hilbert"] = residual_table(pred, truth)
        metrics.append(summarize_residuals("mamba_layer_hilbert", model_residuals["mamba_layer_hilbert"]))

    observables = event_observables(split_events, cfg)
    write_metrics(output_dir / "momentum_diagnostic_metrics.csv", metrics)
    with (output_dir / "momentum_diagnostic_metrics.json").open("w") as stream:
        json.dump({"split": args.split, "metrics": metrics}, stream, indent=2)

    plot_calibration(output_dir / "magnitude_calibration.png", model_residuals)
    plot_distributions(output_dir / "magnitude_residual_distributions.png", model_residuals)
    plot_binned_residuals(
        output_dir / "binned_magnitude_residuals.png",
        observables,
        model_residuals,
        args.bins,
    )
    plot_integrated_angular_resolution(
        output_dir / "angular_resolution_integrated.png",
        model_residuals,
    )
    plot_angular_resolution_vs_lambda_angle(
        output_dir / "angular_resolution_vs_lambda_angle.png",
        model_residuals,
        args.bins,
        args.centerline_mrad,
    )
    print(f"Wrote momentum diagnostics for {args.split} split to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
