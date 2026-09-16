from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
import uproot
from torch.utils.data import Dataset

from .contracts import DETECTOR_SIPM, DETECTOR_WSI, photon_target, rotate_xz


def _keys(tree: uproot.behaviors.TBranch.HasBranches) -> list[str]:
    return [str(key) for key in tree.keys(recursive=True)]


def _branch(tree: uproot.behaviors.TBranch.HasBranches, suffix: str) -> str:
    matches = [key for key in _keys(tree) if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one branch ending {suffix!r}, found {matches}")
    return matches[0]


def _mc_branch(tree: uproot.behaviors.TBranch.HasBranches, suffix: str) -> str:
    """Resolve an MCParticles leaf without accidentally selecting hit contributions."""
    matches = [key for key in _keys(tree) if "MCParticles" in key and key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one MCParticles branch ending {suffix!r}, found {matches}")
    return matches[0]


def _collection_prefix(tree: uproot.behaviors.TBranch.HasBranches, name: str) -> str:
    candidates = [key[: -len(".position.x")] for key in _keys(tree) if key.endswith(".position.x")]
    exact = [item for item in candidates if item == name or item.endswith("/" + name)]
    if len(exact) != 1:
        raise KeyError(f"Expected one collection named {name!r}, found {exact}")
    prefix = exact[0]
    if f"{prefix}.energy" not in _keys(tree):
        raise KeyError(f"Collection {prefix!r} has no energy branch")
    return prefix


def audit_root(path: str | Path, wsi_collection: str, sipm_collection: str, entries: int) -> dict[str, Any]:
    tree = uproot.open(path)["events"]
    pdg_key, generator_status_key = _mc_branch(tree, ".PDG"), _mc_branch(tree, ".generatorStatus")
    px_key, py_key, pz_key = (_mc_branch(tree, suffix) for suffix in (".momentum.x", ".momentum.y", ".momentum.z"))
    report: dict[str, Any] = {"path": str(path), "entries": int(tree.num_entries), "sample_entries": min(entries, int(tree.num_entries))}
    # ``arrays`` works for both production TTrees and uproot's local RNTuple fixtures.
    truth = tree.arrays([pdg_key, generator_status_key], entry_stop=entries, library="ak")
    counts = np.asarray(ak.to_numpy(ak.sum((truth[pdg_key] == 22) & (truth[generator_status_key] == 1), axis=1)))
    report["status1_photon_candidates_per_event"] = {"min": int(counts.min()), "max": int(counts.max()), "exactly_one_fraction": float(np.mean(counts == 1))}
    for detector, name in (("wsi", wsi_collection), ("sipm", sipm_collection)):
        prefix = _collection_prefix(tree, name)
        arrays = tree.arrays([f"{prefix}.energy", f"{prefix}.position.x", f"{prefix}.position.y", f"{prefix}.position.z"], entry_stop=entries, library="ak")
        energy = arrays[f"{prefix}.energy"]
        flat_e = ak.to_numpy(ak.flatten(energy, axis=None))
        flat_e = flat_e[np.isfinite(flat_e) & (flat_e > 0)]
        report[detector] = {"collection": prefix, "positive_hits": int(len(flat_e)), "energy_p50_p99": [float(np.quantile(flat_e, q)) for q in (0.5, 0.99)] if len(flat_e) else [None, None]}
    # Resolve momentum fields during audit so schema failures occur before preparation.
    report["truth_momentum_branches"] = [px_key, py_key, pz_key]
    return report


def _split(source_id: int, seed: int, train: float, val: float) -> int:
    value = int.from_bytes(hashlib.blake2b(f"{seed}:{source_id}".encode(), digest_size=8).digest(), "little") / 2**64
    return 0 if value < train else (1 if value < train + val else 2)


def _selected_photon(pdg: Any, generator_status: Any, px: Any, py: Any, pz: Any) -> tuple[float, float, float] | None:
    """Select the propagated status-1 photon, excluding the status-2 origin row."""
    mask = (np.asarray(pdg) == 22) & (np.asarray(generator_status) == 1)
    if mask.sum() != 1:
        return None
    return float(np.asarray(px)[mask][0]), float(np.asarray(py)[mask][0]), float(np.asarray(pz)[mask][0])


def prepare_dataset(config: dict[str, Any]) -> Path:
    data, split = config["data"], config["split"]
    files = [Path(path) for path in data["photon_files"]]
    if len(files) != 2:
        raise ValueError("data.photon_files must contain exactly photon-1 and photon-2 ROOT files")
    output = Path(data["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    feature_rows: list[np.ndarray] = []
    targets: list[list[float]] = []
    offsets = [0]
    source_ids: list[int] = []
    split_ids: list[int] = []
    train_detector_energy: list[list[float]] = [[], []]
    rejected_truth = 0
    theta = float(data["rotation_theta_rad"])
    for source_file, source_kind in zip(files, range(2)):
        tree = uproot.open(source_file)["events"]
        pdg_key, generator_status_key = _mc_branch(tree, ".PDG"), _mc_branch(tree, ".generatorStatus")
        px_key, py_key, pz_key = (_mc_branch(tree, suffix) for suffix in (".momentum.x", ".momentum.y", ".momentum.z"))
        collections = [(DETECTOR_WSI, _collection_prefix(tree, data["wsi_collection"])), (DETECTOR_SIPM, _collection_prefix(tree, data["sipm_collection"]))]
        needed = [pdg_key, generator_status_key, px_key, py_key, pz_key] + [f"{prefix}.{field}" for _, prefix in collections for field in ("energy", "position.x", "position.y", "position.z")]
        for chunk in tree.iterate(needed, step_size=int(data["chunk_entries"]), library="ak", report=True):
            arrays, report = chunk
            for local in range(len(arrays[pdg_key])):
                selected = _selected_photon(arrays[pdg_key][local], arrays[generator_status_key][local], arrays[px_key][local], arrays[py_key][local], arrays[pz_key][local])
                if selected is None:
                    rejected_truth += 1
                    continue
                px, py, pz = selected
                px, pz = rotate_xz(np.asarray(px), np.asarray(pz), theta)
                target = photon_target(float(px), float(py), float(pz))
                rows = []
                for detector, prefix in collections:
                    energy = np.asarray(arrays[f"{prefix}.energy"][local], dtype=np.float32)
                    x = np.asarray(arrays[f"{prefix}.position.x"][local], dtype=np.float32)
                    y = np.asarray(arrays[f"{prefix}.position.y"][local], dtype=np.float32)
                    z = np.asarray(arrays[f"{prefix}.position.z"][local], dtype=np.float32)
                    x, z = rotate_xz(x, z, theta)
                    valid = np.isfinite(energy) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (energy > float(data["min_positive_energy"]))
                    if valid.any():
                        rows.append(np.stack((energy[valid], x[valid], y[valid], z[valid], np.full(valid.sum(), detector)), axis=1))
                merged = np.concatenate(rows, axis=0).astype(np.float32) if rows else np.empty((0, 5), dtype=np.float32)
                if not len(merged):
                    continue
                # Paired source records share the original event index in the split hash.
                original_id = int(report.start + local)
                split_id = _split(original_id, int(config["seed"]), float(split["train_fraction"]), float(split["val_fraction"]))
                if split_id == 0:
                    for detector in (DETECTOR_WSI, DETECTOR_SIPM):
                        selected_energy = merged[merged[:, 4] == detector, 0]
                        if len(selected_energy): train_detector_energy[detector].extend(selected_energy.tolist())
                feature_rows.append(merged)
                offsets.append(offsets[-1] + len(merged))
                targets.append([target.log_energy, target.theta_x, target.theta_y])
                source_ids.append(source_kind * 1_000_000_000 + original_id)
                split_ids.append(split_id)
    if not targets:
        raise RuntimeError("Preparation produced no selected photon events")
    features = np.concatenate(feature_rows, axis=0)
    np.save(output / "features.npy", features)
    np.save(output / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(output / "targets.npy", np.asarray(targets, dtype=np.float32))
    np.save(output / "source_ids.npy", np.asarray(source_ids, dtype=np.int64))
    np.save(output / "split_ids.npy", np.asarray(split_ids, dtype=np.uint8))
    train_mask = np.asarray(split_ids) == 0
    if any(not values for values in train_detector_energy):
        raise RuntimeError("A detector has no positive training hits; refusing to create a calibrated dataset")
    scales = [float(np.quantile(np.asarray(train_detector_energy[detector]), 0.5)) for detector in (DETECTOR_WSI, DETECTOR_SIPM)]
    target_mean = np.asarray(targets, dtype=np.float32)[train_mask].mean(axis=0).tolist()
    target_std = np.asarray(targets, dtype=np.float32)[train_mask].std(axis=0).clip(1e-6).tolist()
    manifest = {"schema_version": 1, "truth_selector": "PDG == 22 and generatorStatus == 1", "n_events": len(targets), "n_hits": int(len(features)), "rejected_ambiguous_truth": rejected_truth, "detector_energy_reference": scales, "target_mean": target_mean, "target_std": target_std, "split_counts": {name: int(np.sum(np.asarray(split_ids) == i)) for i, name in enumerate(("train", "val", "test"))}, "config": config}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


class ZDCDataset(Dataset):
    def __init__(self, root: str | Path, split: int, max_tokens: int):
        root = Path(root)
        self.features = np.load(root / "features.npy", mmap_mode="r")
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r")
        self.targets = np.load(root / "targets.npy", mmap_mode="r")
        split_ids = np.load(root / "split_ids.npy", mmap_mode="r")
        self.indices = np.flatnonzero(split_ids == split)
        self.max_tokens = max_tokens
        self.manifest = json.loads((root / "manifest.json").read_text())

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        event = int(self.indices[index]); values = _aggregate_cells(np.asarray(self.features[self.offsets[event]:self.offsets[event + 1]], dtype=np.float32))
        if len(values) > self.max_tokens: values = _compress_tokens(values, self.max_tokens)
        return values, np.asarray(self.targets[event], dtype=np.float32)


def _aggregate_cells(values: np.ndarray) -> np.ndarray:
    """Aggregate exact repeated detector cells without altering the energy sum."""
    if len(values) < 2: return values
    keys, inverse = np.unique(values[:, 1:5], axis=0, return_inverse=True)
    energy = np.zeros(len(keys), dtype=np.float32); np.add.at(energy, inverse, values[:, 0])
    return np.column_stack((energy, keys)).astype(np.float32)


def _compress_tokens(values: np.ndarray, maximum: int) -> np.ndarray:
    """Energy-conserving deterministic spatial/depth merge, separately per detector."""
    if maximum < 2: raise ValueError("max_tokens must be at least two")
    counts = np.array([(values[:, 4] == detector).sum() for detector in (DETECTOR_WSI, DETECTOR_SIPM)])
    quotas = np.maximum(1, np.floor(maximum * counts / counts.sum()).astype(int))
    while quotas.sum() < maximum: quotas[np.argmax(counts - quotas)] += 1
    output = []
    for detector, quota in zip((DETECTOR_WSI, DETECTOR_SIPM), quotas):
        subset = values[values[:, 4] == detector]
        if not len(subset): continue
        order = np.lexsort((subset[:, 2], subset[:, 1], subset[:, 3])); subset = subset[order]
        for group in np.array_split(subset, min(int(quota), len(subset))):
            weight = group[:, 0]; total = weight.sum()
            output.append([total, *(np.average(group[:, 1:4], axis=0, weights=weight)), detector])
    return np.asarray(output, dtype=np.float32)
