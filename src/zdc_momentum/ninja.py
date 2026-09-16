"""Streaming ROOT-to-mmap_ninja photon skim."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mmap_ninja
import numpy as np
import uproot

from .contracts import DETECTOR_SIPM, DETECTOR_WSI, photon_target, rotate_xz
from .data import _collection_prefix, _mc_branch, _selected_photon


def build_photon_ninja(
    root_file: str | Path,
    output_dir: str | Path,
    *,
    max_events: int | None = None,
    chunk_entries: int = 1000,
    wsi_collection: str = "ZDC_WSi_Hits",
    sipm_collection: str = "HcalFarForwardZDCHits",
    rotation_theta_rad: float = 0.025,
    min_positive_energy: float = 0.0,
    progress_every: int = 1000,
) -> Path:
    """Write a minimal canonical photon product; refuse to overwrite an output."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    output.mkdir(parents=True)
    tree = uproot.open(root_file)["events"]
    pdg_key, generator_status_key = _mc_branch(tree, ".PDG"), _mc_branch(tree, ".generatorStatus")
    px_key, py_key, pz_key = (_mc_branch(tree, suffix) for suffix in (".momentum.x", ".momentum.y", ".momentum.z"))
    collections = [(DETECTOR_WSI, _collection_prefix(tree, wsi_collection)), (DETECTOR_SIPM, _collection_prefix(tree, sipm_collection))]
    branches = [pdg_key, generator_status_key, px_key, py_key, pz_key] + [f"{prefix}.{field}" for _, prefix in collections for field in ("energy", "position.x", "position.y", "position.z")]
    features_store = mmap_ninja.RaggedMmap(output / "features", mode="r+")
    detector_store = mmap_ninja.RaggedMmap(output / "detector_id", mode="r+")
    targets: list[list[float]] = []
    event_ids: list[int] = []
    counters = {"input_events": 0, "written_events": 0, "ambiguous_truth": 0, "empty_hits": 0}
    for arrays, report in tree.iterate(branches, entry_stop=max_events, step_size=chunk_entries, library="ak", report=True):
        feature_batch, detector_batch = [], []
        target_batch, event_batch = [], []
        for local in range(len(arrays[pdg_key])):
            counters["input_events"] += 1
            selected = _selected_photon(arrays[pdg_key][local], arrays[generator_status_key][local], arrays[px_key][local], arrays[py_key][local], arrays[pz_key][local])
            if selected is None:
                counters["ambiguous_truth"] += 1; continue
            px, py, pz = selected; px, pz = rotate_xz(np.asarray(px), np.asarray(pz), rotation_theta_rad)
            rows, detector_rows = [], []
            for detector, prefix in collections:
                e = np.asarray(arrays[f"{prefix}.energy"][local], dtype=np.float32)
                x = np.asarray(arrays[f"{prefix}.position.x"][local], dtype=np.float32)
                y = np.asarray(arrays[f"{prefix}.position.y"][local], dtype=np.float32)
                z = np.asarray(arrays[f"{prefix}.position.z"][local], dtype=np.float32)
                x, z = rotate_xz(x, z, rotation_theta_rad)
                keep = np.isfinite(e) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (e > min_positive_energy)
                if keep.any():
                    rows.append(np.stack((e[keep], x[keep], y[keep], z[keep]), axis=1))
                    detector_rows.append(np.full(keep.sum(), detector, dtype=np.uint8))
            if not rows:
                counters["empty_hits"] += 1; continue
            target = photon_target(float(px), float(py), float(pz))
            feature_batch.append(np.concatenate(rows).astype(np.float32, copy=False))
            detector_batch.append(np.concatenate(detector_rows))
            target_batch.append([target.log_energy, target.theta_x, target.theta_y])
            event_batch.append(int(report.start + local))
        if feature_batch:
            features_store.extend(feature_batch)
            detector_store.extend(detector_batch)
            targets.extend(target_batch); event_ids.extend(event_batch)
            counters["written_events"] += len(feature_batch)
        if progress_every and counters["input_events"] % progress_every == 0:
            print(f"processed={counters['input_events']} written={counters['written_events']} ambiguous_truth={counters['ambiguous_truth']} empty_hits={counters['empty_hits']}", flush=True)
    if not targets:
        raise RuntimeError("No events passed photon truth and hit selection")
    mmap_ninja.np_from_ndarray(output / "targets", np.asarray(targets, dtype=np.float32))
    mmap_ninja.np_from_ndarray(output / "source_event_id", np.asarray(event_ids, dtype=np.int64))
    # Validate sidecar/event alignment before declaring the product complete.
    if len(features_store) != len(detector_store) or len(features_store) != len(targets):
        raise RuntimeError("Ragged/dense sidecar event counts disagree")
    for index in range(len(features_store)):
        if len(features_store[index]) != len(detector_store[index]):
            raise RuntimeError(f"Ragged sidecar hit count mismatch at event {index}")
    manifest: dict[str, Any] = {"schema_version": 1, "source_root": str(root_file), "truth_selector": "PDG == 22 and generatorStatus == 1", "max_events": max_events, "chunk_entries": chunk_entries, "rotation_theta_rad": rotation_theta_rad, "collections": {"wsi": collections[0][1], "sipm": collections[1][1]}, "features": {"path": "features", "dtype": "float32", "columns": ["E", "x_mm", "y_mm", "z_mm"]}, "detector_id": {"path": "detector_id", "dtype": "uint8", "encoding": {"0": "WSi", "1": "SiPM"}}, "targets": ["log_energy", "theta_x_rad", "theta_y_rad"], "counters": counters}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "_SUCCESS").write_text("\n")
    return output
