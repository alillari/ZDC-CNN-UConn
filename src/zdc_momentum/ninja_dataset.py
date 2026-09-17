"""mmap_ninja loading, train-only normalization, and deterministic token order."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import mmap_ninja
from torch.utils.data import Dataset
from .data import _aggregate_cells, _compress_tokens, _split


def _hilbert(x: int, y: int, bits: int = 6) -> int:
    d = 0; n = 1 << bits; s = n // 2
    while s:
        rx, ry = int(bool(x & s)), int(bool(y & s)); d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx: x, y = n - 1 - x, n - 1 - y
            x, y = y, x
        s //= 2
    return d


def serialize(features: np.ndarray, detector: np.ndarray, coordinate_min: np.ndarray, coordinate_max: np.ndarray, max_tokens: int | None) -> np.ndarray:
    values = np.column_stack((features, detector)).astype(np.float32, copy=False)
    values = _aggregate_cells(values)
    coordinates, detector = values[:, 1:4], values[:, 4].astype(np.uint8)
    lo = coordinate_min[detector]
    span = np.maximum(coordinate_max[detector] - coordinate_min[detector], 1e-6)
    xy = np.clip(((coordinates[:, :2] - lo[:, :2]) / span[:, :2] * 63).astype(int), 0, 63)
    layer = np.clip(((coordinates[:, 2] - lo[:, 2]) / span[:, 2] * np.where(detector == 0, 19, 63)).astype(int), 0, 63)
    h = np.fromiter((_hilbert(int(x), int(y)) for x, y in xy), dtype=np.int64, count=len(xy))
    values = values[np.lexsort((h, layer, detector))]
    return _compress_tokens(values, max_tokens) if max_tokens is not None and len(values) > max_tokens else values


def fit_statistics(roots: list[str], seed: int, train_fraction: float, val_fraction: float) -> dict:
    mins = np.full((2, 3), np.inf); maxs = np.full((2, 3), -np.inf); energy = [[], []]; event_energy = [[], []]; targets = []
    for root in roots:
        f, d = mmap_ninja.RaggedMmap(Path(root) / 'features'), mmap_ninja.RaggedMmap(Path(root) / 'detector_id')
        ids, truth = mmap_ninja.np_open_existing(Path(root) / 'source_event_id'), mmap_ninja.np_open_existing(Path(root) / 'targets')
        for i, event_id in enumerate(ids):
            if _split(int(event_id), seed, train_fraction, val_fraction) != 0: continue
            x, det = np.asarray(f[i]), np.asarray(d[i])
            targets.append(np.asarray(truth[i], dtype=np.float32))
            for k in (0, 1):
                part = x[det == k]
                if len(part):
                    mins[k] = np.minimum(mins[k], part[:, 1:].min(0)); maxs[k] = np.maximum(maxs[k], part[:, 1:].max(0))
                    energy[k].append(part[:, 0]); event_energy[k].append(float(part[:, 0].sum()))
    if not all(parts for parts in energy): raise RuntimeError('Each detector needs positive train hits')
    target_values = np.stack(targets)
    return {
        'coordinate_min': mins.tolist(), 'coordinate_max': maxs.tolist(),
        'global_coordinate_min': mins.min(axis=0).tolist(), 'global_coordinate_max': maxs.max(axis=0).tolist(),
        'energy_reference': [float(np.median(np.concatenate(x))) for x in energy],
        'event_energy_reference': [float(np.median(np.asarray(x))) for x in event_energy],
        'target_mean': target_values.mean(axis=0).tolist(), 'target_std': target_values.std(axis=0).clip(1e-6).tolist(),
    }


class NinjaPhotonDataset(Dataset):
    def __init__(self, roots: list[str], split: int, seed: int, train_fraction: float, val_fraction: float, statistics: dict, max_tokens: int | None, max_events: int | None = None):
        self.items=[]; self.max_tokens=max_tokens; self.minimum=np.asarray(statistics['coordinate_min']); self.maximum=np.asarray(statistics['coordinate_max']); self.event_energy_reference=np.asarray(statistics['event_energy_reference'])
        for source_index, root in enumerate(roots):
            root=Path(root); f=mmap_ninja.RaggedMmap(root/'features'); d=mmap_ninja.RaggedMmap(root/'detector_id'); t=mmap_ninja.np_open_existing(root/'targets'); ids=mmap_ninja.np_open_existing(root/'source_event_id')
            if not (len(f)==len(d)==len(t)==len(ids)): raise RuntimeError(f'Misaligned Ninja sidecars in {root}')
            self.items += [(f, d, t, i, source_index, int(event_id)) for i, event_id in enumerate(ids) if _split(int(event_id),seed,train_fraction,val_fraction)==split]
        if max_events is not None:
            if max_events < 1: raise ValueError("max_events must be positive")
            # A stable hash avoids a source-file-order or generation-scan bias in pilots.
            self.items.sort(key=lambda item: hashlib.blake2b(f"{seed}:{split}:{item[4]}:{item[5]}".encode(), digest_size=8).digest())
            self.items = self.items[:max_events]
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        f,d,t,i,_,_=self.items[index]
        features, detector = np.asarray(f[i]), np.asarray(d[i])
        values = _aggregate_cells(np.column_stack((features, detector)).astype(np.float32, copy=False))
        detector = values[:, 4].astype(np.uint8)
        summary = np.array([
            np.log1p(values[detector == 0, 0].sum() / self.event_energy_reference[0]),
            np.log1p(values[detector == 1, 0].sum() / self.event_energy_reference[1]),
            np.log1p(float((detector == 0).sum())),
            np.log1p(float((detector == 1).sum())),
        ], dtype=np.float32)
        return serialize(features, np.asarray(d[i]), self.minimum, self.maximum, self.max_tokens), np.asarray(t[i],dtype=np.float32), summary
