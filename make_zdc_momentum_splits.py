#!/usr/bin/env python
"""
Build train/validation/test pickle splits for the first ZDC momentum study.

This script extracts the same WSi event representation used by the current CNN
notebooks and adds the momentum target expected by run_zdc_mamba_vs_cnn_study.py:

    event["momentum"] = [lambda_px, lambda_py, lambda_pz]

The Lambda momentum target is reconstructed as the sum of the two photon and
neutron MC momenta after the optional ZDC-frame rotation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import uproot as up
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires uproot. Install it in the active environment with "
        "`pip install uproot awkward`."
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


TREE_NAME = "events"

ZDC_FACE_Z_MM = 35856.4
ZDC_HALF_WIDTH_X_MM = 300.0
ZDC_HALF_WIDTH_Y_MM = 300.0
ROTATION_THETA_RAD = 0.025

M_GAMMA = 0.0
M_NEUTRON = 0.939565

WSI_FRONT_Z_MM = 35856.4
ZDC_TUNGSTEN_THICKNESS_MM = 3.5
ZDC_GLUE_THICKNESS_MM = 0.11
ZDC_PAD_THICKNESS_MM = 0.320
ZDC_PCB_THICKNESS_MM = 1.6
ZDC_SI_AIR_THICKNESS_MM = 1.0

WSI_LAYER_THICKNESS_MM = (
    ZDC_TUNGSTEN_THICKNESS_MM
    + ZDC_GLUE_THICKNESS_MM
    + ZDC_PAD_THICKNESS_MM
    + ZDC_GLUE_THICKNESS_MM
    + ZDC_PCB_THICKNESS_MM
    + ZDC_SI_AIR_THICKNESS_MM
)

WSI_SI_CENTER_OFFSET_MM = (
    ZDC_TUNGSTEN_THICKNESS_MM
    + ZDC_GLUE_THICKNESS_MM
    + 0.5 * ZDC_PAD_THICKNESS_MM
)

WSI_LAYER_Z_MM = (
    WSI_FRONT_Z_MM + WSI_SI_CENTER_OFFSET_MM + np.arange(20) * WSI_LAYER_THICKNESS_MM
)


def rotate_vectors_numpy(x, y, z, theta=ROTATION_THETA_RAD):
    c = np.cos(theta)
    s = np.sin(theta)
    return c * x + s * z, y, -s * x + c * z


def first_per_event(arr):
    return np.array([x[1] if len(x) > 1 else np.nan for x in arr])


def load_first_mc_particle(
    filename: Path,
    mass_assumption: float,
    rotate: bool,
    entry_stop: int | None = None,
) -> Dict[str, np.ndarray]:
    tree = up.open(filename)[TREE_NAME]

    vx = first_per_event(tree["MCParticles.vertex.x"].array(library="np", entry_stop=entry_stop))
    vy = first_per_event(tree["MCParticles.vertex.y"].array(library="np", entry_stop=entry_stop))
    vz = first_per_event(tree["MCParticles.vertex.z"].array(library="np", entry_stop=entry_stop))

    px = first_per_event(tree["MCParticles.momentum.x"].array(library="np", entry_stop=entry_stop))
    py = first_per_event(tree["MCParticles.momentum.y"].array(library="np", entry_stop=entry_stop))
    pz = first_per_event(tree["MCParticles.momentum.z"].array(library="np", entry_stop=entry_stop))

    if rotate:
        vx, vy, vz = rotate_vectors_numpy(vx, vy, vz)
        px, py, pz = rotate_vectors_numpy(px, py, pz)

    p = np.sqrt(px**2 + py**2 + pz**2)
    energy = np.sqrt(p**2 + mass_assumption**2)
    mass = np.full_like(px, mass_assumption, dtype=float)

    return {
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "px": px,
        "py": py,
        "pz": pz,
        "p": p,
        "E": energy,
        "m": mass,
    }


def collection_prefix(filename: Path, collection_name: str) -> str:
    tree = up.open(filename)[TREE_NAME]
    keys = set(tree.keys())
    candidates = [collection_name, f"{collection_name}/{collection_name}"]
    for prefix in candidates:
        if f"{prefix}.position.x" in keys and f"{prefix}.energy" in keys:
            return prefix
    raise KeyError(
        f"Could not find hit collection {collection_name!r} in {filename}. "
        f"Tried {candidates}."
    )


def extract_hit_information(
    filename: Path,
    collection_name: str,
    rotate: bool,
    entry_stop: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = up.open(filename)[TREE_NAME]
    prefix = collection_prefix(filename, collection_name)
    hit_x = tree[f"{prefix}.position.x"].array(library="np", entry_stop=entry_stop)
    hit_y = tree[f"{prefix}.position.y"].array(library="np", entry_stop=entry_stop)
    hit_z = tree[f"{prefix}.position.z"].array(library="np", entry_stop=entry_stop)
    hit_E = tree[f"{prefix}.energy"].array(library="np", entry_stop=entry_stop)

    if rotate:
        hit_x, hit_y, hit_z = rotate_vectors_numpy(hit_x, hit_y, hit_z)

    return hit_x, hit_y, hit_z, hit_E


def assign_layers_from_geometry(
    z_column: Sequence[np.ndarray],
    layer_z: np.ndarray = WSI_LAYER_Z_MM,
    tolerance: float = 1.0,
) -> np.ndarray:
    all_layers = []
    for zevt in z_column:
        zevt = np.asarray(zevt)
        if len(zevt) == 0:
            all_layers.append(np.array([], dtype=np.int32))
            continue

        dz = np.abs(zevt[:, None] - layer_z[None, :])
        layers = np.argmin(dz, axis=1)
        min_dz = dz[np.arange(len(zevt)), layers]
        valid = min_dz < tolerance
        layers_out = np.full(len(zevt), -1, dtype=np.int32)
        layers_out[valid] = layers[valid]
        all_layers.append(layers_out)

    return np.array(all_layers, dtype=object)


def project_to_zdc(p: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = ZDC_FACE_Z_MM - p["vz"]
        x_zdc = p["vx"] + (p["px"] / p["pz"]) * dz
        y_zdc = p["vy"] + (p["py"] / p["pz"]) * dz
    return x_zdc, y_zdc


def intersects_zdc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(x) < ZDC_HALF_WIDTH_X_MM)
        & (np.abs(y) < ZDC_HALF_WIDTH_Y_MM)
    )


def event_has_valid_activity(*particle_hit_blocks, require_positive_energy=True) -> np.ndarray:
    n_events = len(particle_hit_blocks[0]["E"])
    keep = np.zeros(n_events, dtype=bool)
    for block in particle_hit_blocks:
        for idx, (layers, energies) in enumerate(zip(block["layer"], block["E"])):
            layers = np.asarray(layers)
            energies = np.asarray(energies)
            if len(layers) == 0 or len(energies) == 0:
                continue
            valid = layers >= 0
            if require_positive_energy:
                valid &= energies > 0
            keep[idx] |= bool(np.any(valid))
    return keep


def filter_accepted_events(
    photon1_file: Path,
    photon2_file: Path,
    neutron_file: Path,
    rotate: bool,
    entry_stop: int | None = None,
) -> np.ndarray:
    gamma1 = load_first_mc_particle(photon1_file, M_GAMMA, rotate, entry_stop=entry_stop)
    gamma2 = load_first_mc_particle(photon2_file, M_GAMMA, rotate, entry_stop=entry_stop)
    neutron = load_first_mc_particle(neutron_file, M_NEUTRON, rotate, entry_stop=entry_stop)

    x1, y1 = project_to_zdc(gamma1)
    x2, y2 = project_to_zdc(gamma2)
    xn, yn = project_to_zdc(neutron)

    acc1 = intersects_zdc(x1, y1)
    acc2 = intersects_zdc(x2, y2)
    accn = intersects_zdc(xn, yn)
    forward = (gamma1["pz"] > 0) & (gamma2["pz"] > 0) & (neutron["pz"] > 0)
    return acc1 & acc2 & accn & forward


def apply_event_mask_to_hit_tuple(hit_tuple, mask):
    return tuple(component[mask] for component in hit_tuple)


def build_events(args) -> List[Dict[str, object]]:
    geometric_mask = filter_accepted_events(
        args.photon1_file,
        args.photon2_file,
        args.neutron_file,
        rotate=not args.no_rotate,
        entry_stop=args.read_entries,
    )
    print(f"Geometrically accepted events: {geometric_mask.sum()} / {len(geometric_mask)}")

    gamma1 = load_first_mc_particle(
        args.photon1_file, M_GAMMA, rotate=not args.no_rotate, entry_stop=args.read_entries
    )
    gamma2 = load_first_mc_particle(
        args.photon2_file, M_GAMMA, rotate=not args.no_rotate, entry_stop=args.read_entries
    )
    neutron = load_first_mc_particle(
        args.neutron_file, M_NEUTRON, rotate=not args.no_rotate, entry_stop=args.read_entries
    )

    wsi_g1 = extract_hit_information(
        args.photon1_file, args.collection, rotate=not args.no_rotate, entry_stop=args.read_entries
    )
    wsi_g2 = extract_hit_information(
        args.photon2_file, args.collection, rotate=not args.no_rotate, entry_stop=args.read_entries
    )
    wsi_n = extract_hit_information(
        args.neutron_file, args.collection, rotate=not args.no_rotate, entry_stop=args.read_entries
    )

    wsi_g1 = apply_event_mask_to_hit_tuple(wsi_g1, geometric_mask)
    wsi_g2 = apply_event_mask_to_hit_tuple(wsi_g2, geometric_mask)
    wsi_n = apply_event_mask_to_hit_tuple(wsi_n, geometric_mask)

    wsi_layers_g1 = assign_layers_from_geometry(wsi_g1[2], tolerance=args.layer_tolerance_mm)
    wsi_layers_g2 = assign_layers_from_geometry(wsi_g2[2], tolerance=args.layer_tolerance_mm)
    wsi_layers_n = assign_layers_from_geometry(wsi_n[2], tolerance=args.layer_tolerance_mm)

    has_wsi = event_has_valid_activity(
        {"layer": wsi_layers_g1, "E": wsi_g1[3]},
        {"layer": wsi_layers_g2, "E": wsi_g2[3]},
        {"layer": wsi_layers_n, "E": wsi_n[3]},
    )
    print(f"WSi detector-hit requirement: {has_wsi.sum()} / {len(has_wsi)}")

    decay_x = gamma1["vx"][geometric_mask][has_wsi]
    decay_y = gamma1["vy"][geometric_mask][has_wsi]
    decay_z = gamma1["vz"][geometric_mask][has_wsi]

    lam_px = (gamma1["px"][geometric_mask] + gamma2["px"][geometric_mask] + neutron["px"][geometric_mask])[has_wsi]
    lam_py = (gamma1["py"][geometric_mask] + gamma2["py"][geometric_mask] + neutron["py"][geometric_mask])[has_wsi]
    lam_pz = (gamma1["pz"][geometric_mask] + gamma2["pz"][geometric_mask] + neutron["pz"][geometric_mask])[has_wsi]

    wsi_g1 = apply_event_mask_to_hit_tuple(wsi_g1, has_wsi)
    wsi_g2 = apply_event_mask_to_hit_tuple(wsi_g2, has_wsi)
    wsi_n = apply_event_mask_to_hit_tuple(wsi_n, has_wsi)
    wsi_layers_g1 = wsi_layers_g1[has_wsi]
    wsi_layers_g2 = wsi_layers_g2[has_wsi]
    wsi_layers_n = wsi_layers_n[has_wsi]

    n_events = len(decay_z)
    if args.limit_events is not None:
        n_events = min(n_events, args.limit_events)

    iterator = range(n_events)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Assembling events", leave=False)

    events = []
    for idx in iterator:
        momentum = np.array([lam_px[idx], lam_py[idx], lam_pz[idx]], dtype=np.float32)
        events.append(
            {
                "gamma1": {
                    "x": wsi_g1[0][idx],
                    "y": wsi_g1[1][idx],
                    "z": wsi_g1[2][idx],
                    "layer": wsi_layers_g1[idx],
                    "E": wsi_g1[3][idx],
                },
                "gamma2": {
                    "x": wsi_g2[0][idx],
                    "y": wsi_g2[1][idx],
                    "z": wsi_g2[2][idx],
                    "layer": wsi_layers_g2[idx],
                    "E": wsi_g2[3][idx],
                },
                "neutron": {
                    "x": wsi_n[0][idx],
                    "y": wsi_n[1][idx],
                    "z": wsi_n[2][idx],
                    "layer": wsi_layers_n[idx],
                    "E": wsi_n[3][idx],
                },
                "x_vertex": float(decay_x[idx]),
                "y_vertex": float(decay_y[idx]),
                "z_vertex": float(decay_z[idx]),
                "lam_mom_x": float(lam_px[idx]),
                "lam_mom_y": float(lam_py[idx]),
                "lam_mom_z": float(lam_pz[idx]),
                "momentum": momentum,
            }
        )

    return events


def split_events(
    events: Sequence[Mapping[str, object]],
    seed: int,
    test_fraction: float,
    val_fraction_of_remaining: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))
    n_test = int(round(test_fraction * len(events)))
    test_idx = order[:n_test]
    rest_idx = order[n_test:]
    n_val = int(round(val_fraction_of_remaining * len(rest_idx)))
    val_idx = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]
    return {
        "train": [events[idx] for idx in train_idx],
        "val": [events[idx] for idx in val_idx],
        "test": [events[idx] for idx in test_idx],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photon1-file", type=Path, default=Path("../HEPMC-testing/photon1_test_50000.root"))
    parser.add_argument("--photon2-file", type=Path, default=Path("../HEPMC-testing/photon2_test_50000.root"))
    parser.add_argument("--neutron-file", type=Path, default=Path("../HEPMC-testing/neutron_test_50000.root"))
    parser.add_argument("--output", type=Path, default=Path("zdc_momentum_splits.pkl"))
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--collection", default="ZDC_WSi_Hits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction-of-remaining", type=float, default=0.1765)
    parser.add_argument("--layer-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--limit-events", type=int, default=None)
    parser.add_argument(
        "--read-entries",
        type=int,
        default=None,
        help="Only read the first N ROOT tree entries before acceptance cuts.",
    )
    parser.add_argument("--no-rotate", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    events = build_events(args)
    splits = split_events(
        events,
        seed=args.seed,
        test_fraction=args.test_fraction,
        val_fraction_of_remaining=args.val_fraction_of_remaining,
    )
    payload = {
        **splits,
        "metadata": {
            "photon1_file": str(args.photon1_file),
            "photon2_file": str(args.photon2_file),
            "neutron_file": str(args.neutron_file),
            "collection": args.collection,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "val_fraction_of_remaining": args.val_fraction_of_remaining,
            "rotate_to_zdc_frame": not args.no_rotate,
            "layer_tolerance_mm": args.layer_tolerance_mm,
            "read_entries": args.read_entries,
            "n_events": len(events),
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "target": "Lambda momentum = gamma1 + gamma2 + neutron MC momentum",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    with metadata_path.open("w") as stream:
        json.dump(payload["metadata"], stream, indent=2)

    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")
    print(f"Split sizes: {payload['metadata']['split_sizes']}")


if __name__ == "__main__":
    main()
