#!/usr/bin/env python3
"""Render a compact, reproducible diagnostic figure for one ZDC run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "zdc-matplotlib"))
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--residual-output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    output = args.output or run_dir / "diagnostics_test.png"
    residual_output = args.residual_output or run_dir / "residual_maps_test.png"
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    evaluation = json.loads((run_dir / "evaluation_test.json").read_text())
    prediction = np.load(run_dir / "predictions_test.npz")

    truth = np.exp(prediction["truth"][:, 0])
    estimated = np.exp(prediction["prediction"][:, 0])
    residual = prediction["relative_energy_residual"]
    theta_x = (prediction["prediction"][:, 1] - prediction["truth"][:, 1]) * 1e3
    theta_y = (prediction["prediction"][:, 2] - prediction["truth"][:, 2]) * 1e3
    has_energy = bool(np.all(np.isfinite(estimated)))
    has_angles = bool(np.all(np.isfinite(theta_x)) and np.all(np.isfinite(theta_y)))
    records = evaluation["energy_bins"]
    center = np.asarray([(item["energy_low_gev"] + item["energy_high_gev"]) / 2 for item in records])
    bias = np.asarray([item["relative_bias"] if item["relative_bias"] is not None else np.nan for item in records])
    p16 = np.asarray([item["relative_p16"] if item["relative_p16"] is not None else np.nan for item in records])
    p84 = np.asarray([item["relative_p84"] if item["relative_p84"] is not None else np.nan for item in records])
    sigma = np.asarray([item["gaussian"].get("sigma", np.nan) if item["gaussian"].get("reported") else np.nan for item in records])
    counts = np.asarray([item["n"] for item in records])

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    if history:
        epoch = [row["epoch"] for row in history]
        ax.plot(epoch, [row["train_loss"] for row in history], marker="o", label="train")
        ax.plot(epoch, [row["val_loss"] for row in history], marker="o", label="validation")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.set_title("Optimization history"); ax.grid(alpha=.25); ax.legend()
    else: ax.text(.5, .5, "Analytic calibration baseline\n(no optimization history)", ha="center", va="center"); ax.set_axis_off()

    ax = axes[0, 1]
    if has_energy:
        limit = max(float(truth.max()), float(estimated.max())) * 1.03
        image = ax.hexbin(truth, estimated, gridsize=48, mincnt=1, bins="log", cmap="viridis")
        ax.plot([0, limit], [0, limit], "w--", linewidth=1, label="ideal")
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="truth energy [GeV]", ylabel="predicted energy [GeV]", title="Energy response")
        ax.legend(loc="upper left"); figure.colorbar(image, ax=ax, label="events per hexagon (log)")
    else: ax.text(.5, .5, "Energy output disabled", ha="center", va="center"); ax.set_axis_off()

    ax = axes[1, 0]
    if has_energy:
        ax.fill_between(center, p16 * 100, p84 * 100, alpha=.25, label="central 68%")
        ax.plot(center, bias * 100, marker="o", label="mean residual")
        good = np.isfinite(sigma)
        ax.plot(center[good], sigma[good] * 100, "s", label="Gaussian $\\sigma$")
        for x, n in zip(center, counts): ax.annotate(f"n={n}", (x, -36), ha="center", fontsize=8)
        ax.axhline(0, color="black", linewidth=.8); ax.set(ylim=(-40, 40), xlabel="truth-energy bin center [GeV]", ylabel="$(E_{pred}-E_{true})/E_{true}$ [%]", title="Energy residual by truth energy")
        ax.grid(alpha=.25); ax.legend()
    else: ax.text(.5, .5, "Energy output disabled", ha="center", va="center"); ax.set_axis_off()

    ax = axes[1, 1]
    if has_angles:
        extent = max(float(np.quantile(np.abs(np.r_[theta_x, theta_y]), .995)), 1.)
        bins = np.linspace(-extent, extent, 80)
        ax.hist(theta_x, bins=bins, histtype="step", linewidth=1.5, label=rf"$\Delta\theta_x$, MAE={np.mean(np.abs(theta_x)):.2f} mrad")
        ax.hist(theta_y, bins=bins, histtype="step", linewidth=1.5, label=rf"$\Delta\theta_y$, MAE={np.mean(np.abs(theta_y)):.2f} mrad")
        ax.axvline(0, color="black", linewidth=.8); ax.set(xlabel="prediction − truth [mrad]", ylabel="events", title="Angular residual distributions")
        ax.grid(alpha=.25); ax.legend()
    else: ax.text(.5, .5, "Angular outputs disabled", ha="center", va="center"); ax.set_axis_off()

    figure.suptitle(f"ZDC photon momentum diagnostics — {run_dir.name} (test n={len(truth)})", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)

    residuals = []
    if has_energy: residuals.append((estimated - truth, r"$\Delta E = E_{pred}-E_{true}$ [GeV]", "Energy residual"))
    if has_angles: residuals.extend(((theta_x, r"$\Delta\theta_x$ [mrad]", "Theta-x residual"), (theta_y, r"$\Delta\theta_y$ [mrad]", "Theta-y residual")))
    figure, axes = plt.subplots(1, len(residuals), figsize=(5.5 * len(residuals), 4.8), constrained_layout=True, squeeze=False)
    axes = axes[0]
    x_max = float(np.quantile(truth, .999))
    for axis, (values, label, title) in zip(axes, residuals):
        low, high = np.quantile(values, (.005, .995))
        span = max(abs(float(low)), abs(float(high)))
        image = axis.hist2d(truth, values, bins=(60, 60), range=((0, x_max), (-span, span)), norm=LogNorm(), cmap="viridis", cmin=1)
        axis.axhline(0, color="white", linestyle="--", linewidth=1)
        axis.set(xlabel="true energy [GeV]", ylabel=label, title=title)
        figure.colorbar(image[3], ax=axis, label="events/bin (log)")
    figure.suptitle(f"ZDC photon residual maps — {run_dir.name} (test n={len(truth)})", fontsize=14)
    figure.savefig(residual_output, dpi=180)
    print(residual_output)


if __name__ == "__main__":
    main()
