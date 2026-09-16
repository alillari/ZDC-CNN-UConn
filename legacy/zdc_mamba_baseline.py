"""
Sparse supervised Mamba baseline for three-particle ZDC WSi regression.

This module is intentionally additive to the existing CNN study in
test_three_particle_ml.py. It reuses the same event dictionary convention:

    {
        "gamma1": {"x": ..., "y": ..., "layer": ..., "E": ...},
        "gamma2": {"x": ..., "y": ..., "layer": ..., "E": ...},
        "neutron": {"x": ..., "y": ..., "layer": ..., "E": ...},
        ...
    }

The serialization follows the FM4NPP/CLAS12 pattern of making detector layer
identity dominate the order, then applying a 2D Hilbert code only within each
layer. The Hilbert lookup is precomputed once for the 64x64 transverse grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import subprocess
from time import sleep
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - exercised only when mamba-ssm is absent
    Mamba = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    from test_three_particle_ml import (
        get_gpu_temperature_c,
        wait_for_gpu_temperature,
    )
except ImportError:
    def get_gpu_temperature_c():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        first_temp = result.stdout.strip().splitlines()[0]
        return float(first_temp)

    def wait_for_gpu_temperature(target_temp_c, sleep_seconds=15):
        if target_temp_c is None:
            return None
        temp_c = get_gpu_temperature_c()
        while temp_c is not None and temp_c > target_temp_c:
            sleep(sleep_seconds)
            temp_c = get_gpu_temperature_c()
        return temp_c


PARTICLE_NAMES = ("gamma1", "gamma2", "neutron")
PARTICLE_TYPE_IDS = {"gamma1": 0, "gamma2": 0, "neutron": 1}

TOKEN_FEATURE_NAMES = (
    "log_energy",
    "x_norm",
    "y_norm",
    "z_norm",
    "layer_norm",
    "hilbert_norm",
    "particle_type",
    "is_layer_start",
    "hilbert_gap_norm",
    "layer_delta_norm",
    "is_layer_summary",
)


@dataclass
class ZDCTokenizationConfig:
    method: str = "layer_hilbert"
    n_layers: int = 20
    nx: int = 64
    ny: int = 64
    x_min: float = -300.0
    x_max: float = 300.0
    y_min: float = -300.0
    y_max: float = 300.0
    z_min: float = 0.0
    z_max: float = 19.0
    hilbert_bits: int = 6
    add_layer_summary_tokens: bool = False
    shuffle_seed: int = 12345


@dataclass
class MomentumTargetConfig:
    mode: str = "magnitude_direction"  # "magnitude_direction" or "cartesian"
    magnitude_transform: str = "log"  # "log", "identity", or "standardize"
    target_key: str = "momentum"
    magnitude_mean: Optional[float] = None
    magnitude_std: Optional[float] = None
    eps: float = 1e-8


@dataclass
class ZDCMambaConfig:
    tokenization: ZDCTokenizationConfig = field(default_factory=ZDCTokenizationConfig)
    target: MomentumTargetConfig = field(default_factory=MomentumTargetConfig)
    input_dim: int = len(TOKEN_FEATURE_NAMES)
    model_dim: int = 96
    num_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    particle_type_embedding_dim: int = 8
    head_hidden_dim: int = 192
    lambda_p: float = 1.0
    lambda_dir: float = 1.0
    lambda_cartesian: float = 1.0
    use_gru_fallback_without_mamba: bool = True


def hilbert_xy_to_index(x: int, y: int, bits: int = 6) -> int:
    """Map one 2D integer coordinate to a Hilbert distance on a 2**bits grid.

    Adapted for this ZDC use from the Hilbert serialization reviewed in
    ../../ML-work/CLAS12/fm4npp/hilbert.py. The implementation here is the
    standard compact xy->d form, used only to build a cached 64x64 lookup.
    """

    n = 1 << bits
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"Hilbert coordinate ({x}, {y}) is outside 0..{n - 1}")

    d = 0
    xi = int(x)
    yi = int(y)
    s = n // 2
    while s > 0:
        rx = 1 if (xi & s) else 0
        ry = 1 if (yi & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                xi = n - 1 - xi
                yi = n - 1 - yi
            xi, yi = yi, xi
        s //= 2
    return int(d)


_HILBERT_LOOKUP_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def get_hilbert_lookup(nx: int = 64, ny: int = 64, bits: int = 6) -> np.ndarray:
    key = (nx, ny, bits)
    if key not in _HILBERT_LOOKUP_CACHE:
        grid_size = 1 << bits
        if nx > grid_size or ny > grid_size:
            raise ValueError(
                f"Grid {nx}x{ny} does not fit in a 2**{bits} Hilbert plane"
            )
        lookup = np.empty((nx, ny), dtype=np.int64)
        for ix in range(nx):
            for iy in range(ny):
                lookup[ix, iy] = hilbert_xy_to_index(ix, iy, bits=bits)
        _HILBERT_LOOKUP_CACHE[key] = lookup
    return _HILBERT_LOOKUP_CACHE[key]


def _as_array(hit: Mapping[str, object], key: str, dtype=None) -> np.ndarray:
    if key not in hit:
        return np.empty(0, dtype=dtype or np.float32)
    return np.asarray(hit[key], dtype=dtype)


def _extract_particle_arrays(
    hits: Mapping[str, object],
    cfg: ZDCTokenizationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _as_array(hits, "x", np.float64)
    y = _as_array(hits, "y", np.float64)
    layer = _as_array(hits, "layer", np.int64)
    energy = _as_array(hits, "E", np.float64)
    z = _as_array(hits, "z", np.float64)

    if z.size == 0:
        z = layer.astype(np.float64)

    if not (x.size == y.size == layer.size == energy.size == z.size):
        raise ValueError("Hit arrays x, y, layer, E, and optional z must align")

    ix = ((x - cfg.x_min) / (cfg.x_max - cfg.x_min) * cfg.nx).astype(np.int64)
    iy = ((y - cfg.y_min) / (cfg.y_max - cfg.y_min) * cfg.ny).astype(np.int64)

    mask = (
        (layer >= 0)
        & (layer < cfg.n_layers)
        & (ix >= 0)
        & (ix < cfg.nx)
        & (iy >= 0)
        & (iy < cfg.ny)
        & np.isfinite(energy)
        & (energy > 0)
    )

    return ix[mask], iy[mask], layer[mask], energy[mask], z[mask]


def _aggregate_cells(
    ix: np.ndarray,
    iy: np.ndarray,
    layer: np.ndarray,
    energy: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if energy.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_i, empty_i, empty_f, empty_f

    cell = np.stack([layer, ix, iy], axis=1)
    unique, inverse = np.unique(cell, axis=0, return_inverse=True)
    energy_sum = np.zeros(len(unique), dtype=np.float64)
    z_weighted = np.zeros(len(unique), dtype=np.float64)
    np.add.at(energy_sum, inverse, energy)
    np.add.at(z_weighted, inverse, energy * z)
    z_mean = z_weighted / np.maximum(energy_sum, 1e-12)

    return (
        unique[:, 1].astype(np.int64),
        unique[:, 2].astype(np.int64),
        unique[:, 0].astype(np.int64),
        energy_sum.astype(np.float32),
        z_mean.astype(np.float32),
    )


def serialize_particle_hits(
    hits: Mapping[str, object],
    method: Optional[str] = None,
    particle_type_id: int = 0,
    config: Optional[ZDCTokenizationConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Serialize one particle's sparse WSi hits into occupied-cell tokens."""

    cfg = config or ZDCTokenizationConfig(method=method or "layer_hilbert")
    method = cfg.method if method is None else method
    if method not in {"layer_hilbert", "layer_raster", "shuffled"}:
        raise ValueError(f"Unsupported serialization method: {method}")

    ix, iy, layer, energy, z = _extract_particle_arrays(hits, cfg)
    ix, iy, layer, energy, z = _aggregate_cells(ix, iy, layer, energy, z)

    hilbert_lookup = get_hilbert_lookup(cfg.nx, cfg.ny, bits=cfg.hilbert_bits)
    hilbert = hilbert_lookup[ix, iy] if energy.size else np.empty(0, dtype=np.int64)
    raster = ix * cfg.ny + iy

    if method == "layer_hilbert":
        order = np.lexsort((hilbert, layer))
    elif method == "layer_raster":
        order = np.lexsort((raster, layer))
    else:
        order = np.arange(energy.size)
        local_rng = rng if rng is not None else np.random.default_rng(cfg.shuffle_seed)
        local_rng.shuffle(order)

    ix = ix[order]
    iy = iy[order]
    layer = layer[order]
    energy = energy[order]
    z = z[order]
    hilbert = hilbert[order]

    max_hilbert = float((1 << (2 * cfg.hilbert_bits)) - 1)
    x_norm = ix.astype(np.float32) / max(cfg.nx - 1, 1)
    y_norm = iy.astype(np.float32) / max(cfg.ny - 1, 1)
    z_norm = ((z - cfg.z_min) / max(cfg.z_max - cfg.z_min, 1e-12)).astype(np.float32)
    layer_norm = layer.astype(np.float32) / max(cfg.n_layers - 1, 1)
    hilbert_norm = hilbert.astype(np.float32) / max_hilbert

    if energy.size:
        first_in_layer = np.ones(energy.size, dtype=np.float32)
        first_in_layer[1:] = (layer[1:] != layer[:-1]).astype(np.float32)
        hilbert_gap = np.zeros(energy.size, dtype=np.float32)
        hilbert_gap[1:] = (hilbert[1:] - hilbert[:-1]).astype(np.float32) / max_hilbert
        layer_delta = np.zeros(energy.size, dtype=np.float32)
        layer_delta[1:] = (layer[1:] - layer[:-1]).astype(np.float32) / max(
            cfg.n_layers - 1, 1
        )
    else:
        first_in_layer = np.empty(0, dtype=np.float32)
        hilbert_gap = np.empty(0, dtype=np.float32)
        layer_delta = np.empty(0, dtype=np.float32)

    features = np.stack(
        [
            np.log1p(energy).astype(np.float32),
            x_norm,
            y_norm,
            z_norm,
            layer_norm,
            hilbert_norm,
            np.full_like(x_norm, float(particle_type_id)),
            first_in_layer,
            hilbert_gap,
            layer_delta,
            np.zeros_like(x_norm),
        ],
        axis=1,
    ).astype(np.float32)

    if cfg.add_layer_summary_tokens and energy.size:
        features, layer, hilbert, ix, iy, energy = _insert_layer_summary_tokens(
            features, layer, hilbert, ix, iy, energy, cfg, particle_type_id
        )

    return {
        "features": features,
        "layer": layer.astype(np.int64),
        "hilbert": hilbert.astype(np.int64),
        "ix": ix.astype(np.int64),
        "iy": iy.astype(np.int64),
        "energy": energy.astype(np.float32),
    }


def _insert_layer_summary_tokens(
    features: np.ndarray,
    layer: np.ndarray,
    hilbert: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    energy: np.ndarray,
    cfg: ZDCTokenizationConfig,
    particle_type_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    layers_out = []
    hilbert_out = []
    ix_out = []
    iy_out = []
    energy_out = []
    for layer_value in np.unique(layer):
        layer_mask = layer == layer_value
        layer_energy = energy[layer_mask]
        total_e = float(layer_energy.sum())
        cx = float(np.sum(ix[layer_mask] * layer_energy) / max(total_e, 1e-12))
        cy = float(np.sum(iy[layer_mask] * layer_energy) / max(total_e, 1e-12))
        summary = np.zeros(features.shape[1], dtype=np.float32)
        summary[0] = np.log1p(total_e)
        summary[1] = cx / max(cfg.nx - 1, 1)
        summary[2] = cy / max(cfg.ny - 1, 1)
        summary[3] = layer_value / max(cfg.n_layers - 1, 1)
        summary[4] = layer_value / max(cfg.n_layers - 1, 1)
        summary[5] = 0.0
        summary[6] = float(particle_type_id)
        summary[7] = 1.0
        summary[8] = 0.0
        summary[9] = 0.0
        summary[10] = 1.0
        rows.append(summary[None, :])
        rows.append(features[layer_mask])
        layers_out.extend([layer_value])
        layers_out.extend(layer[layer_mask].tolist())
        hilbert_out.extend([-1])
        hilbert_out.extend(hilbert[layer_mask].tolist())
        ix_out.extend([int(round(cx))])
        ix_out.extend(ix[layer_mask].tolist())
        iy_out.extend([int(round(cy))])
        iy_out.extend(iy[layer_mask].tolist())
        energy_out.extend([total_e])
        energy_out.extend(layer_energy.tolist())
        rows[-1][:, 7] = 0.0
        rows[-1][0, 7] = 0.0
    return (
        np.concatenate(rows, axis=0),
        np.asarray(layers_out, dtype=np.int64),
        np.asarray(hilbert_out, dtype=np.int64),
        np.asarray(ix_out, dtype=np.int64),
        np.asarray(iy_out, dtype=np.int64),
        np.asarray(energy_out, dtype=np.float32),
    )


class MomentumTransform:
    def __init__(self, config: MomentumTargetConfig):
        self.config = config

    def fit(self, events: Sequence[Mapping[str, object]]) -> None:
        if self.config.magnitude_transform != "standardize":
            return
        magnitudes = []
        for ev in events:
            p = extract_momentum_target(ev, self.config.target_key)
            magnitudes.append(float(np.linalg.norm(p)))
        vals = np.asarray(magnitudes, dtype=np.float32)
        self.config.magnitude_mean = float(vals.mean())
        self.config.magnitude_std = float(vals.std() if vals.std() > 1e-8 else 1.0)

    def encode_magnitude(self, p_mag: torch.Tensor) -> torch.Tensor:
        mode = self.config.magnitude_transform
        if mode == "log":
            return torch.log(p_mag.clamp_min(self.config.eps))
        if mode == "identity":
            return p_mag
        if mode == "standardize":
            mean = self.config.magnitude_mean
            std = self.config.magnitude_std
            if mean is None or std is None:
                raise ValueError("Magnitude standardization stats are not fitted")
            return (p_mag - mean) / max(std, self.config.eps)
        raise ValueError(f"Unknown magnitude transform: {mode}")

    def decode_magnitude(self, p_value: torch.Tensor) -> torch.Tensor:
        mode = self.config.magnitude_transform
        if mode == "log":
            return torch.exp(p_value)
        if mode == "identity":
            return p_value
        if mode == "standardize":
            mean = self.config.magnitude_mean
            std = self.config.magnitude_std
            if mean is None or std is None:
                raise ValueError("Magnitude standardization stats are not fitted")
            return p_value * std + mean
        raise ValueError(f"Unknown magnitude transform: {mode}")

    def metadata(self) -> Dict[str, object]:
        return asdict(self.config)


def extract_momentum_target(event: Mapping[str, object], target_key: str) -> np.ndarray:
    if target_key in event:
        value = event[target_key]
        if isinstance(value, Mapping):
            return np.asarray([value["px"], value["py"], value["pz"]], dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    if all(key in event for key in ("px", "py", "pz")):
        return np.asarray([event["px"], event["py"], event["pz"]], dtype=np.float32)

    raise KeyError(
        f"Event is missing momentum target '{target_key}' or top-level px/py/pz"
    )


class SparseZDCMomentumDataset(Dataset):
    def __init__(
        self,
        events: Sequence[Mapping[str, object]],
        config: Optional[ZDCMambaConfig] = None,
        target_transform: Optional[MomentumTransform] = None,
        cache_tokens: bool = True,
        progress_desc: Optional[str] = None,
    ):
        self.events = list(events)
        self.config = config or ZDCMambaConfig()
        self.target_transform = target_transform or MomentumTransform(self.config.target)
        self.cache_tokens = cache_tokens
        self._cache = None

        if (
            self.config.target.magnitude_transform == "standardize"
            and self.config.target.magnitude_mean is None
        ):
            self.target_transform.fit(self.events)

        if cache_tokens:
            iterator: Iterable[int] = range(len(self.events))
            if progress_desc and tqdm is not None:
                iterator = tqdm(iterator, desc=progress_desc, leave=False)
            self._cache = [self._encode_event(self.events[i], i) for i in iterator]

    def __len__(self) -> int:
        return len(self.events)

    def _encode_event(self, event: Mapping[str, object], index: int) -> Dict[str, object]:
        rng = np.random.default_rng(self.config.tokenization.shuffle_seed + index)
        particles = []
        for name in PARTICLE_NAMES:
            particles.append(
                serialize_particle_hits(
                    event[name],
                    method=self.config.tokenization.method,
                    particle_type_id=PARTICLE_TYPE_IDS[name],
                    config=self.config.tokenization,
                    rng=rng,
                )
            )

        momentum = extract_momentum_target(event, self.config.target.target_key)
        return {"particles": particles, "momentum": momentum}

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self._cache[idx] if self._cache is not None else self._encode_event(self.events[idx], idx)
        return item


def collate_sparse_zdc(batch: Sequence[Mapping[str, object]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(
        1,
        max(max(p["features"].shape[0] for p in item["particles"]) for item in batch),
    )
    feature_dim = batch[0]["particles"][0]["features"].shape[1]
    features = torch.zeros(batch_size, 3, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, 3, max_len, dtype=torch.bool)
    energy = torch.zeros(batch_size, 3, max_len, dtype=torch.float32)
    lengths = torch.zeros(batch_size, 3, dtype=torch.long)
    momenta = torch.empty(batch_size, 3, dtype=torch.float32)

    for bidx, item in enumerate(batch):
        momenta[bidx] = torch.as_tensor(item["momentum"], dtype=torch.float32)
        for pidx, particle in enumerate(item["particles"]):
            n = particle["features"].shape[0]
            lengths[bidx, pidx] = n
            if n == 0:
                continue
            features[bidx, pidx, :n] = torch.from_numpy(particle["features"])
            energy[bidx, pidx, :n] = torch.from_numpy(particle["energy"])
            mask[bidx, pidx, :n] = True

    return {
        "features": features,
        "mask": mask,
        "energy": energy,
        "lengths": lengths,
        "momentum": momenta,
    }


class RMSNorm(nn.Module):
    """Local copy of the RMSNorm pattern used by FM4NPP Mamba blocks."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class _MambaOrFallbackBlock(nn.Module):
    def __init__(self, cfg: ZDCMambaConfig):
        super().__init__()
        self.has_mamba = Mamba is not None
        self.allow_fallback = cfg.use_gru_fallback_without_mamba
        self.mamba_block = None
        self.fallback_block = None
        if self.has_mamba:
            self.mamba_block = Mamba(
                d_model=cfg.model_dim,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
            )
        if self.allow_fallback:
            self.fallback_block = nn.GRU(
                cfg.model_dim,
                cfg.model_dim,
                num_layers=1,
                batch_first=True,
            )
        if not self.has_mamba and not self.allow_fallback:
            raise ImportError("mamba-ssm is required for ZDCMambaRegressor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_mamba and x.is_cuda:
            return self.mamba_block(x)
        if self.fallback_block is None:
            raise RuntimeError(
                "mamba-ssm is installed but its available kernel requires CUDA; "
                "enable use_gru_fallback_without_mamba for CPU smoke tests."
            )
        out, _ = self.fallback_block(x)
        return out


class ParticleMambaEncoder(nn.Module):
    def __init__(self, cfg: ZDCMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.particle_embedding = nn.Embedding(2, cfg.particle_type_embedding_dim)
        projected_dim = cfg.input_dim + cfg.particle_type_embedding_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(projected_dim),
            nn.Linear(projected_dim, cfg.model_dim),
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": RMSNorm(cfg.model_dim),
                        "mamba": _MambaOrFallbackBlock(cfg),
                        "dropout": nn.Dropout(cfg.dropout),
                    }
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = RMSNorm(cfg.model_dim)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        pid = features[..., TOKEN_FEATURE_NAMES.index("particle_type")].long().clamp(0, 1)
        x = torch.cat([features, self.particle_embedding(pid)], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)

        for layer in self.layers:
            z = layer["mamba"](layer["norm"](x))
            x = (x + layer["dropout"](z)) * mask.unsqueeze(-1).to(x.dtype)

        x = self.norm(x)
        return self.pool(x, mask, energy)

    def pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (x * mask_f).sum(dim=1) / denom

        neg_inf = torch.finfo(x.dtype).min
        max_pool = x.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        w = torch.log1p(energy).unsqueeze(-1) * mask_f
        w_denom = w.sum(dim=1).clamp_min(1e-8)
        e_mean = (x * w).sum(dim=1) / w_denom

        return torch.cat([mean, max_pool, e_mean], dim=-1)


class ZDCMambaRegressor(nn.Module):
    def __init__(self, config: Optional[ZDCMambaConfig] = None):
        super().__init__()
        self.config = config or ZDCMambaConfig()
        output_dim = 4 if self.config.target.mode == "magnitude_direction" else 3
        self.encoder = ParticleMambaEncoder(self.config)
        pooled_dim = self.config.model_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim * 3),
            nn.Linear(pooled_dim * 3, self.config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden_dim, self.config.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.head_hidden_dim, output_dim),
        )
        self.target_transform = MomentumTransform(self.config.target)

    @property
    def mamba_available(self) -> bool:
        return Mamba is not None

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        energy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        embeddings = []
        for particle_idx in range(3):
            embeddings.append(
                self.encoder(
                    features[:, particle_idx],
                    mask[:, particle_idx],
                    energy[:, particle_idx],
                )
            )
        event_embedding = torch.cat(embeddings, dim=-1)
        raw = self.head(event_embedding)

        if self.config.target.mode == "cartesian":
            return {"cartesian": raw, "event_embedding": event_embedding}

        magnitude_value = raw[..., 0]
        magnitude = self.target_transform.decode_magnitude(magnitude_value).clamp_min(
            self.config.target.eps
        )
        unit_direction = F.normalize(raw[..., 1:4], p=2, dim=-1, eps=1e-8)
        momentum = magnitude[..., None] * unit_direction
        return {
            "raw": raw,
            "magnitude_value": magnitude_value,
            "magnitude": magnitude,
            "unit_direction": unit_direction,
            "cartesian": momentum,
            "event_embedding": event_embedding,
        }


def zdc_momentum_loss(
    outputs: Mapping[str, torch.Tensor],
    truth_momentum: torch.Tensor,
    config: ZDCMambaConfig,
) -> Dict[str, torch.Tensor]:
    if config.target.mode == "cartesian":
        cart = F.mse_loss(outputs["cartesian"], truth_momentum)
        return {"loss": config.lambda_cartesian * cart, "cartesian": cart}

    transform = MomentumTransform(config.target)
    true_mag = torch.linalg.norm(truth_momentum, dim=-1).clamp_min(config.target.eps)
    true_dir = F.normalize(truth_momentum, p=2, dim=-1, eps=1e-8)
    true_mag_value = transform.encode_magnitude(true_mag)
    p_loss = F.smooth_l1_loss(outputs["magnitude_value"], true_mag_value)
    dir_loss = (1.0 - (outputs["unit_direction"] * true_dir).sum(dim=-1)).mean()
    total = config.lambda_p * p_loss + config.lambda_dir * dir_loss
    return {"loss": total, "magnitude": p_loss, "direction": dir_loss}


@torch.no_grad()
def momentum_diagnostics(
    predicted_momentum: torch.Tensor,
    truth_momentum: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    pred_mag = torch.linalg.norm(predicted_momentum, dim=-1).clamp_min(eps)
    true_mag = torch.linalg.norm(truth_momentum, dim=-1).clamp_min(eps)
    pred_dir = predicted_momentum / pred_mag.unsqueeze(-1)
    true_dir = truth_momentum / true_mag.unsqueeze(-1)
    dot = (pred_dir * true_dir).sum(dim=-1).clamp(-1.0, 1.0)
    angular = torch.acos(dot)
    pred_pz = torch.where(
        predicted_momentum[..., 2].abs() < eps,
        predicted_momentum[..., 2].sign().clamp(min=0).mul(2).sub(1) * eps,
        predicted_momentum[..., 2],
    )
    true_pz = torch.where(
        truth_momentum[..., 2].abs() < eps,
        truth_momentum[..., 2].sign().clamp(min=0).mul(2).sub(1) * eps,
        truth_momentum[..., 2],
    )
    pred_tx = predicted_momentum[..., 0] / pred_pz
    pred_ty = predicted_momentum[..., 1] / pred_pz
    true_tx = truth_momentum[..., 0] / true_pz
    true_ty = truth_momentum[..., 1] / true_pz
    return {
        "magnitude_residual": pred_mag - true_mag,
        "relative_magnitude_residual": (pred_mag - true_mag) / true_mag,
        "cartesian_residual": predicted_momentum - truth_momentum,
        "angular_separation_rad": angular,
        "tx_residual": pred_tx - true_tx,
        "ty_residual": pred_ty - true_ty,
    }


def sequence_length_summary(dataset: SparseZDCMomentumDataset) -> Dict[str, object]:
    totals = []
    by_particle = {name: [] for name in PARTICLE_NAMES}
    dense_cells = (
        dataset.config.tokenization.n_layers
        * dataset.config.tokenization.nx
        * dataset.config.tokenization.ny
    )
    for idx in range(len(dataset)):
        item = dataset[idx]
        lengths = [p["features"].shape[0] for p in item["particles"]]
        totals.append(sum(lengths))
        for name, length in zip(PARTICLE_NAMES, lengths):
            by_particle[name].append(length)

    total_arr = np.asarray(totals, dtype=np.float32)
    summary = {
        "tokens_per_event": _describe(total_arr),
        "active_fraction_of_dense_volume": _describe(total_arr / (3.0 * dense_cells)),
        "tokens_per_particle": {
            name: _describe(np.asarray(values, dtype=np.float32))
            for name, values in by_particle.items()
        },
    }
    return summary


def _describe(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_zdc_mamba_baseline(
    train_events: Sequence[Mapping[str, object]],
    val_events: Sequence[Mapping[str, object]],
    config: Optional[ZDCMambaConfig] = None,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    cache_tokens: bool = True,
    num_workers: int = 0,
    use_amp: bool = True,
    show_progress: bool = True,
    target_gpu_temp_c: Optional[float] = 80,
    temp_check_interval: int = 1,
    temp_cooldown_sleep: int = 30,
    checkpoint_path: Optional[str] = None,
) -> Tuple[ZDCMambaRegressor, Dict[str, object]]:
    cfg = config or ZDCMambaConfig()
    target_transform = MomentumTransform(cfg.target)
    if cfg.target.magnitude_transform == "standardize":
        target_transform.fit(train_events)

    train_dataset = SparseZDCMomentumDataset(
        train_events,
        config=cfg,
        target_transform=target_transform,
        cache_tokens=cache_tokens,
        progress_desc="Precomputing train tokens" if show_progress else None,
    )
    val_dataset = SparseZDCMomentumDataset(
        val_events,
        config=cfg,
        target_transform=target_transform,
        cache_tokens=cache_tokens,
        progress_desc="Precomputing val tokens" if show_progress else None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_sparse_zdc,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_sparse_zdc,
    )

    model = ZDCMambaRegressor(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    amp_enabled = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val = float("inf")
    history = []

    metadata = {
        "config": {
            "tokenization": asdict(cfg.tokenization),
            "target": target_transform.metadata(),
            "model": {
                "model_dim": cfg.model_dim,
                "num_layers": cfg.num_layers,
                "d_state": cfg.d_state,
                "d_conv": cfg.d_conv,
                "expand": cfg.expand,
                "mamba_available": model.mamba_available,
                "uses_real_mamba_this_run": device == "cuda" and model.mamba_available,
                "cpu_gru_fallback_enabled": cfg.use_gru_fallback_without_mamba,
                "parameter_count": count_parameters(model),
            },
            "loss": {
                "lambda_p": cfg.lambda_p,
                "lambda_dir": cfg.lambda_dir,
                "lambda_cartesian": cfg.lambda_cartesian,
            },
        },
        "train_sequence_lengths": sequence_length_summary(train_dataset),
        "val_sequence_lengths": sequence_length_summary(val_dataset),
    }

    print(
        f"Training ZDC Mamba baseline on {device}; "
        f"real_mamba_this_run={device == 'cuda' and model.mamba_available}; "
        f"params={count_parameters(model)}",
        flush=True,
    )
    if target_gpu_temp_c is not None and get_gpu_temperature_c() is None:
        print("GPU temperature guard requested, but nvidia-smi query failed.", flush=True)

    for epoch in range(n_epochs):
        epoch_start = perf_counter()
        train_metrics = _run_mamba_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            pin_memory,
            amp_enabled,
            train=True,
            show_progress=show_progress,
            desc=f"Epoch {epoch + 1:03d}/{n_epochs:03d} train",
            target_gpu_temp_c=target_gpu_temp_c,
            temp_check_interval=temp_check_interval,
            temp_cooldown_sleep=temp_cooldown_sleep,
        )
        val_metrics = _run_mamba_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            cfg,
            device,
            pin_memory,
            amp_enabled,
            train=False,
            show_progress=show_progress,
            desc=f"Epoch {epoch + 1:03d}/{n_epochs:03d} val",
            target_gpu_temp_c=target_gpu_temp_c,
            temp_check_interval=temp_check_interval,
            temp_cooldown_sleep=temp_cooldown_sleep,
        )
        epoch_seconds = perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1,
            "epoch_seconds": epoch_seconds,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1:03d} | train total={train_metrics['loss']:.5f} | "
            f"val total={val_metrics['loss']:.5f} | "
            f"val p={val_metrics.get('magnitude', float('nan')):.5f} | "
            f"val dir={val_metrics.get('direction', float('nan')):.5f} | "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )

        if checkpoint_path and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": metadata,
                    "history": history,
                    "best_val_loss": best_val,
                },
                checkpoint_path,
            )

    if checkpoint_path:
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        metadata["returned_checkpoint"] = "best_val"
    metadata["history"] = history
    metadata["best_val_loss"] = best_val
    return model, metadata


def _run_mamba_epoch(
    model: ZDCMambaRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: ZDCMambaConfig,
    device: str,
    pin_memory: bool,
    amp_enabled: bool,
    train: bool,
    show_progress: bool,
    desc: str,
    target_gpu_temp_c: Optional[float],
    temp_check_interval: int,
    temp_cooldown_sleep: int,
) -> Dict[str, float]:
    model.train(train)
    sums: Dict[str, float] = {}
    n_events = 0
    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch_idx, batch in enumerate(iterator):
        if batch_idx % max(1, temp_check_interval) == 0:
            wait_for_gpu_temperature(target_gpu_temp_c, temp_cooldown_sleep)

        features = batch["features"].to(device, non_blocking=pin_memory)
        mask = batch["mask"].to(device, non_blocking=pin_memory)
        energy = batch["energy"].to(device, non_blocking=pin_memory)
        momentum = batch["momentum"].to(device, non_blocking=pin_memory)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(features, mask, energy)
                losses = zdc_momentum_loss(outputs, momentum, cfg)

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


def validate_serialization_examples(
    events: Sequence[Mapping[str, object]],
    config: Optional[ZDCTokenizationConfig] = None,
    n_examples: int = 3,
) -> List[Dict[str, object]]:
    cfg = config or ZDCTokenizationConfig()
    reports = []
    for event_idx, event in enumerate(events[:n_examples]):
        for name in PARTICLE_NAMES:
            tokens = serialize_particle_hits(
                event[name],
                method=cfg.method,
                particle_type_id=PARTICLE_TYPE_IDS[name],
                config=cfg,
            )
            layer = tokens["layer"]
            hilbert = tokens["hilbert"]
            front_to_back = bool(np.all(layer[1:] >= layer[:-1])) if layer.size > 1 else True
            hilbert_within_layer = True
            for layer_id in np.unique(layer):
                h = hilbert[layer == layer_id]
                h = h[h >= 0]
                if h.size > 1 and not np.all(h[1:] >= h[:-1]):
                    hilbert_within_layer = False
                    break
            reports.append(
                {
                    "event": event_idx,
                    "particle": name,
                    "n_tokens": int(tokens["features"].shape[0]),
                    "front_to_back_layers": front_to_back,
                    "hilbert_sorted_within_layer": hilbert_within_layer,
                    "only_occupied_cells": bool(np.all(tokens["energy"] > 0)),
                    "layers": layer.tolist(),
                    "hilbert": hilbert.tolist(),
                }
            )
    return reports
