#!/usr/bin/env python
"""
First apples-to-apples ZDC momentum study:

1. stacked Conv2d WSi image baseline with the same magnitude+unit-direction
   target as the sparse Mamba model;
2. Hilbert-serialized sparse Mamba with magnitude+unit-direction target.

Input is a pickle file containing either:

    {"train": train_events, "val": val_events, "test": test_events}

or, with --allow-random-split:

    {"events": events}

Each event must contain gamma1/gamma2/neutron hit dictionaries and either
event["momentum"] = [px, py, pz] or top-level px/py/pz.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    from test_three_particle_ml import make_wsi_image_stack
except ImportError:
    def make_wsi_image_stack(
        hit_x,
        hit_y,
        hit_layer,
        hit_E,
        n_layers=20,
        nx=64,
        ny=64,
        x_min=-300.0,
        x_max=300.0,
        y_min=-300.0,
        y_max=300.0,
    ):
        img = np.zeros((n_layers, nx, ny), dtype=np.float32)
        ix = ((hit_x - x_min) / (x_max - x_min) * nx).astype(int)
        iy = ((hit_y - y_min) / (y_max - y_min) * ny).astype(int)
        mask = (
            (hit_layer >= 0)
            & (hit_layer < n_layers)
            & (ix >= 0)
            & (ix < nx)
            & (iy >= 0)
            & (iy < ny)
        )
        np.add.at(img, (hit_layer[mask], ix[mask], iy[mask]), hit_E[mask])
        return img

from zdc_mamba_baseline import (
    MomentumTargetConfig,
    MomentumTransform,
    SparseZDCMomentumDataset,
    ZDCMambaConfig,
    collate_sparse_zdc,
    count_parameters,
    extract_momentum_target,
    momentum_diagnostics,
    sequence_length_summary,
    train_zdc_mamba_baseline,
    validate_serialization_examples,
    zdc_momentum_loss,
)


PARTICLE_NAMES = ("gamma1", "gamma2", "neutron")


class MomentumWSIImageDataset(Dataset):
    def __init__(
        self,
        events: Sequence[Mapping[str, object]],
        target_config: MomentumTargetConfig,
        cache_images: bool = True,
        progress_desc: Optional[str] = None,
    ):
        self.events = list(events)
        self.target_config = target_config
        self.cache_images = cache_images
        self._x_cache = None
        self._p_cache = np.stack(
            [
                extract_momentum_target(event, self.target_config.target_key)
                for event in self.events
            ],
            axis=0,
        ).astype(np.float32)

        if cache_images:
            iterator = range(len(self.events))
            if progress_desc and tqdm is not None:
                iterator = tqdm(iterator, desc=progress_desc, leave=False)
            self._x_cache = np.empty((len(self.events), 3, 20, 64, 64), dtype=np.float32)
            for idx in iterator:
                self._x_cache[idx] = self._encode_event(self.events[idx])

    def __len__(self) -> int:
        return len(self.events)

    def _encode_event(self, event: Mapping[str, object]) -> np.ndarray:
        images = []
        for name in PARTICLE_NAMES:
            hits = event[name]
            image = make_wsi_image_stack(
                hit_x=np.asarray(hits["x"]),
                hit_y=np.asarray(hits["y"]),
                hit_layer=np.asarray(hits["layer"], dtype=int),
                hit_E=np.asarray(hits["E"]),
            )
            images.append(image)
        return np.log1p(np.stack(images, axis=0)).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._x_cache is None:
            image = self._encode_event(self.events[idx])
        else:
            image = self._x_cache[idx]
        return {
            "image": torch.from_numpy(image),
            "momentum": torch.from_numpy(self._p_cache[idx]),
        }


class StackedWSICNNMomentumRegressor(nn.Module):
    def __init__(
        self,
        target_config: MomentumTargetConfig,
        output_mode: str = "magnitude_direction",
    ):
        super().__init__()
        self.target_config = target_config
        self.output_mode = output_mode
        output_dim = 4 if output_mode == "magnitude_direction" else 3

        self.cnn = nn.Sequential(
            nn.Conv2d(60, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.target_transform = MomentumTransform(target_config)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = image.shape[0]
        x = image.view(batch_size, 60, 64, 64)
        raw = self.regressor(self.cnn(x))

        if self.output_mode == "cartesian":
            return {"raw": raw, "cartesian": raw}

        magnitude_value = raw[..., 0]
        magnitude = self.target_transform.decode_magnitude(magnitude_value).clamp_min(
            self.target_config.eps
        )
        unit_direction = torch.nn.functional.normalize(
            raw[..., 1:4], p=2, dim=-1, eps=1e-8
        )
        return {
            "raw": raw,
            "magnitude_value": magnitude_value,
            "magnitude": magnitude,
            "unit_direction": unit_direction,
            "cartesian": magnitude[..., None] * unit_direction,
        }


def train_cnn_baseline(
    train_events: Sequence[Mapping[str, object]],
    val_events: Sequence[Mapping[str, object]],
    config: ZDCMambaConfig,
    output_dir: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    cache_images: bool,
    use_amp: bool,
    show_progress: bool,
) -> Tuple[StackedWSICNNMomentumRegressor, Dict[str, object]]:
    train_dataset = MomentumWSIImageDataset(
        train_events,
        target_config=config.target,
        cache_images=cache_images,
        progress_desc="Precomputing train CNN images" if show_progress else None,
    )
    val_dataset = MomentumWSIImageDataset(
        val_events,
        target_config=config.target,
        cache_images=cache_images,
        progress_desc="Precomputing val CNN images" if show_progress else None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = StackedWSICNNMomentumRegressor(
        target_config=config.target,
        output_mode=config.target.mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_val = float("inf")
    checkpoint_path = output_dir / "cnn_magnitude_direction_best.pt"

    print(
        f"Training stacked Conv2d baseline on {device}; "
        f"params={count_parameters(model)}",
        flush=True,
    )

    for epoch in range(n_epochs):
        epoch_start = perf_counter()
        train_metrics = _run_cnn_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            config,
            device,
            pin_memory,
            amp_enabled,
            train=True,
            show_progress=show_progress,
            desc=f"CNN epoch {epoch + 1:03d}/{n_epochs:03d} train",
        )
        val_metrics = _run_cnn_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            config,
            device,
            pin_memory,
            amp_enabled,
            train=False,
            show_progress=show_progress,
            desc=f"CNN epoch {epoch + 1:03d}/{n_epochs:03d} val",
        )
        epoch_seconds = perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "epoch_seconds": epoch_seconds,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print(
            f"CNN epoch {epoch + 1:03d} | train total={train_metrics['loss']:.5f} | "
            f"val total={val_metrics['loss']:.5f} | "
            f"val p={val_metrics.get('magnitude', float('nan')):.5f} | "
            f"val dir={val_metrics.get('direction', float('nan')):.5f} | "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val_loss": best_val,
                    "config": {
                        "target": asdict(config.target),
                        "model": "StackedWSICNNMomentumRegressor",
                        "parameter_count": count_parameters(model),
                    },
                },
                checkpoint_path,
            )

    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    metadata = {
        "model": "StackedWSICNNMomentumRegressor",
        "parameter_count": count_parameters(model),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_val_loss": best_val,
        "returned_checkpoint": "best_val",
    }
    return model, metadata


def _run_cnn_epoch(
    model: StackedWSICNNMomentumRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: ZDCMambaConfig,
    device: str,
    pin_memory: bool,
    amp_enabled: bool,
    train: bool,
    show_progress: bool,
    desc: str,
) -> Dict[str, float]:
    model.train(train)
    sums: Dict[str, float] = {}
    n_events = 0
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        image = batch["image"].to(device, non_blocking=pin_memory)
        momentum = batch["momentum"].to(device, non_blocking=pin_memory)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(image)
                losses = zdc_momentum_loss(outputs, momentum, config)
            if train:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

        batch_n = momentum.size(0)
        n_events += batch_n
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().item()) * batch_n
        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{float(losses['loss'].detach().item()):.5f}")

    return {key: value / max(n_events, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate_cnn(
    model: StackedWSICNNMomentumRegressor,
    events: Sequence[Mapping[str, object]],
    config: ZDCMambaConfig,
    batch_size: int,
    num_workers: int,
    cache_images: bool,
) -> Dict[str, object]:
    dataset = MomentumWSIImageDataset(events, config.target, cache_images=cache_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = next(model.parameters()).device
    predictions = []
    truth = []
    model.eval()
    for batch in loader:
        image = batch["image"].to(device)
        outputs = model(image)
        predictions.append(outputs["cartesian"].detach().cpu())
        truth.append(batch["momentum"])
    return summarize_predictions(torch.cat(predictions), torch.cat(truth))


@torch.no_grad()
def evaluate_mamba(
    model: torch.nn.Module,
    events: Sequence[Mapping[str, object]],
    config: ZDCMambaConfig,
    batch_size: int,
    num_workers: int,
    cache_tokens: bool,
) -> Dict[str, object]:
    dataset = SparseZDCMomentumDataset(events, config=config, cache_tokens=cache_tokens)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sparse_zdc,
    )
    device = next(model.parameters()).device
    predictions = []
    truth = []
    model.eval()
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        energy = batch["energy"].to(device)
        outputs = model(features, mask, energy)
        predictions.append(outputs["cartesian"].detach().cpu())
        truth.append(batch["momentum"])
    summary = summarize_predictions(torch.cat(predictions), torch.cat(truth))
    summary["sequence_lengths"] = sequence_length_summary(dataset)
    return summary


def summarize_predictions(
    predicted_momentum: torch.Tensor,
    truth_momentum: torch.Tensor,
) -> Dict[str, object]:
    diag = momentum_diagnostics(predicted_momentum, truth_momentum)
    cart = diag["cartesian_residual"].numpy()
    angular_mrad = diag["angular_separation_rad"] * 1000.0
    summary = {
        "n_events": int(truth_momentum.shape[0]),
        "magnitude_bias": float(diag["magnitude_residual"].mean().item()),
        "magnitude_resolution_rms": float(diag["magnitude_residual"].std().item()),
        "relative_magnitude_bias": float(
            diag["relative_magnitude_residual"].mean().item()
        ),
        "relative_magnitude_resolution_rms": float(
            diag["relative_magnitude_residual"].std().item()
        ),
        "angular_resolution_mrad_rms": float(
            torch.sqrt(torch.mean(angular_mrad.square())).item()
        ),
        "angular_mrad_std": float(angular_mrad.std().item()),
        "angular_mrad_mean": float(angular_mrad.mean().item()),
        "angular_mrad_p68": float(torch.quantile(angular_mrad, 0.68).item()),
        "angular_mrad_p95": float(torch.quantile(angular_mrad, 0.95).item()),
        "tx_residual_rms": float(diag["tx_residual"].std().item()),
        "ty_residual_rms": float(diag["ty_residual"].std().item()),
        "cartesian_residual_bias": {
            "px": float(cart[:, 0].mean()),
            "py": float(cart[:, 1].mean()),
            "pz": float(cart[:, 2].mean()),
        },
        "cartesian_residual_rms": {
            "px": float(cart[:, 0].std()),
            "py": float(cart[:, 1].std()),
            "pz": float(cart[:, 2].std()),
        },
    }
    return summary


def load_event_splits(
    path: Path,
    allow_random_split: bool,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)

    if all(key in payload for key in ("train", "val")):
        return payload["train"], payload["val"], payload.get("test", payload["val"])

    if "events" not in payload or not allow_random_split:
        raise ValueError(
            "Input pickle must contain train/val(/test) splits. To split a single "
            "'events' list, pass --allow-random-split."
        )

    events = list(payload["events"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))
    n_train = int(round(train_fraction * len(events)))
    n_val = int(round(val_fraction * len(events)))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return (
        [events[idx] for idx in train_idx],
        [events[idx] for idx in val_idx],
        [events[idx] for idx in test_idx],
    )


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "model",
        "n_events",
        "magnitude_bias",
        "magnitude_resolution_rms",
        "relative_magnitude_bias",
        "relative_magnitude_resolution_rms",
        "angular_mrad_mean",
        "angular_mrad_std",
        "angular_resolution_mrad_rms",
        "angular_mrad_p68",
        "angular_mrad_p95",
        "tx_residual_rms",
        "ty_residual_rms",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-pickle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("zdc_mamba_vs_cnn_outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--allow-random-split", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--model-dim", type=int, default=96)
    parser.add_argument("--mamba-layers", type=int, default=4)
    parser.add_argument("--head-hidden-dim", type=int, default=192)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-dir", type=float, default=1.0)
    parser.add_argument(
        "--magnitude-transform",
        choices=("log", "identity", "standardize"),
        default="log",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_events, val_events, test_events = load_event_splits(
        args.splits_pickle,
        allow_random_split=args.allow_random_split,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    config = ZDCMambaConfig()
    config.tokenization.method = "layer_hilbert"
    config.target.mode = "magnitude_direction"
    config.target.magnitude_transform = args.magnitude_transform
    config.lambda_p = args.lambda_p
    config.lambda_dir = args.lambda_dir
    config.model_dim = args.model_dim
    config.num_layers = args.mamba_layers
    config.head_hidden_dim = args.head_hidden_dim

    if config.target.magnitude_transform == "standardize":
        MomentumTransform(config.target).fit(train_events)

    validation_report = validate_serialization_examples(
        train_events, config.tokenization, n_examples=min(5, len(train_events))
    )
    with (args.output_dir / "serialization_validation.json").open("w") as stream:
        json.dump(validation_report, stream, indent=2)

    shared_run_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "target": asdict(config.target),
        "loss": {"lambda_p": config.lambda_p, "lambda_dir": config.lambda_dir},
        "split_sizes": {
            "train": len(train_events),
            "val": len(val_events),
            "test": len(test_events),
        },
    }
    with (args.output_dir / "run_config.json").open("w") as stream:
        json.dump(shared_run_config, stream, indent=2)

    cache = not args.no_cache
    use_amp = not args.no_amp
    show_progress = not args.quiet

    cnn_model, cnn_metadata = train_cnn_baseline(
        train_events,
        val_events,
        config=config,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        cache_images=cache,
        use_amp=use_amp,
        show_progress=show_progress,
    )
    cnn_eval = evaluate_cnn(
        cnn_model,
        test_events,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_images=cache,
    )

    mamba_checkpoint = args.output_dir / "mamba_layer_hilbert_best.pt"
    mamba_model, mamba_metadata = train_zdc_mamba_baseline(
        train_events,
        val_events,
        config=config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cache_tokens=cache,
        num_workers=args.num_workers,
        use_amp=use_amp,
        show_progress=show_progress,
        target_gpu_temp_c=None,
        checkpoint_path=str(mamba_checkpoint),
    )
    mamba_eval = evaluate_mamba(
        mamba_model,
        test_events,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_tokens=cache,
    )

    summary = {
        "run_config": shared_run_config,
        "cnn": {"metadata": cnn_metadata, "test": cnn_eval},
        "mamba_layer_hilbert": {"metadata": mamba_metadata, "test": mamba_eval},
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    rows = [
        {"model": "stacked_conv2d", **cnn_eval},
        {"model": "mamba_layer_hilbert", **mamba_eval},
    ]
    write_summary_csv(args.output_dir / "summary_metrics.csv", rows)
    print(f"Wrote study outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
