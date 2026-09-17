from __future__ import annotations

import numpy as np
import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover
    Mamba = None


def collate(batch):
    if len(batch[0]) == 3:
        values, targets, summaries = zip(*batch)
        summaries = torch.from_numpy(np.stack(summaries).astype(np.float32, copy=False))
    else:
        values, targets = zip(*batch)
        summaries = None
    width = max(len(item) for item in values)
    features = torch.zeros((len(values), width, 5), dtype=torch.float32)
    mask = torch.zeros((len(values), width), dtype=torch.bool)
    for index, item in enumerate(values):
        features[index, :len(item)] = torch.from_numpy(item)
        mask[index, :len(item)] = True
    # ``targets`` is a tuple of small NumPy arrays.  Stack once before crossing
    # into Torch; constructing a tensor directly from that tuple is very slow.
    return {"features": features, "mask": mask, "targets": torch.from_numpy(np.stack(targets).astype(np.float32, copy=False)), "summaries": summaries}


class SparseMambaPhotonRegressor(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 6, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1, energy_reference: tuple[float, float] = (1.0, 1.0), coordinate_min=None, coordinate_max=None, global_coordinate_min=None, global_coordinate_max=None, position_frequencies: int = 2, input_normalization: str = "layernorm", event_summary: str = "none", target_mode: str = "joint", readout_mode: str = "shared"):
        super().__init__()
        self.register_buffer("energy_reference", torch.tensor(energy_reference, dtype=torch.float32).clamp_min(1e-12))
        self.register_buffer("coordinate_min", torch.tensor(coordinate_min if coordinate_min is not None else [[-1., -1., -1.], [-1., -1., -1.]], dtype=torch.float32))
        self.register_buffer("coordinate_max", torch.tensor(coordinate_max if coordinate_max is not None else [[1., 1., 1.], [1., 1., 1.]], dtype=torch.float32))
        self.register_buffer("global_coordinate_min", torch.tensor(global_coordinate_min if global_coordinate_min is not None else [-1., -1., -1.], dtype=torch.float32))
        self.register_buffer("global_coordinate_max", torch.tensor(global_coordinate_max if global_coordinate_max is not None else [1., 1., 1.], dtype=torch.float32))
        self.position_frequencies = position_frequencies
        if input_normalization not in ("layernorm", "none"):
            raise ValueError("input_normalization must be 'layernorm' or 'none'")
        if event_summary not in ("none", "calorimeter"):
            raise ValueError("event_summary must be 'none' or 'calorimeter'")
        if target_mode not in ("joint", "energy", "angles"):
            raise ValueError("target_mode must be 'joint', 'energy', or 'angles'")
        if readout_mode not in ("shared", "separate"):
            raise ValueError("readout_mode must be 'shared' or 'separate'")
        self.input_normalization = input_normalization
        self.event_summary = event_summary
        self.target_mode = target_mode
        self.readout_mode = readout_mode
        self.energy_log_gain = nn.Parameter(torch.zeros(2))
        self.detector_embedding = nn.Embedding(2, 8)
        # Local coordinates describe each detector at its native scale.  The global
        # coordinates preserve their relative placement in the combined ZDC system.
        input_dim = 1 + 3 + 3 + 2 * 3 * position_frequencies + 8
        # This switch concerns only raw 27-feature normalization.  ``self.norm``
        # below remains the latent 96-D normalization before attention pooling.
        input_norm = nn.LayerNorm(input_dim) if input_normalization == "layernorm" else nn.Identity()
        self.input_projection = nn.Sequential(input_norm, nn.Linear(input_dim, dim), nn.GELU())
        self.blocks = nn.ModuleList([_MambaOrGRU(dim, d_state, d_conv, expand) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        # In calorimeter mode, four train-normalized event observables are
        # concatenated after pooling: log total WSi energy, log total SiPM
        # energy, log merged WSi hit count, and log merged SiPM hit count.
        summary_dim = 4 if event_summary == "calorimeter" else 0
        if readout_mode == "shared":
            output_dim = {"joint": 3, "energy": 1, "angles": 2}[target_mode]
            self.head = nn.Sequential(nn.Linear(dim + summary_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, output_dim))
        else:
            # Separate heads preserve a shared Mamba trunk but avoid requiring
            # the final MLP to use calorimetric sums for angular regression.
            if target_mode in ("joint", "energy"):
                self.energy_head = nn.Sequential(nn.Linear(dim + summary_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
            if target_mode in ("joint", "angles"):
                self.angle_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 2))

    def forward(self, features: torch.Tensor, mask: torch.Tensor, summaries: torch.Tensor | None = None) -> torch.Tensor:
        energy, coordinates, detector = features[..., 0], features[..., 1:4], features[..., 4].long().clamp(0, 1)
        scaled_energy = torch.log1p(energy / (self.energy_reference[detector] * torch.exp(self.energy_log_gain[detector])))
        lo, hi = self.coordinate_min[detector], self.coordinate_max[detector]
        local_coordinates = (2.0 * (coordinates - lo) / (hi - lo).clamp_min(1e-6) - 1.0).clamp(-1.0, 1.0)
        global_coordinates = (2.0 * (coordinates - self.global_coordinate_min) / (self.global_coordinate_max - self.global_coordinate_min).clamp_min(1e-6) - 1.0).clamp(-1.0, 1.0)
        angles = [torch.pi * (2 ** band) * local_coordinates for band in range(self.position_frequencies)]
        positional = torch.cat([item for angle in angles for item in (torch.sin(angle), torch.cos(angle))], dim=-1)
        x = torch.cat((scaled_energy.unsqueeze(-1), local_coordinates, global_coordinates, positional, self.detector_embedding(detector)), dim=-1)
        x = self.input_projection(x) * mask.unsqueeze(-1)
        for block in self.blocks:
            x = (x + block(x)) * mask.unsqueeze(-1)
        x = self.norm(x)
        scores = self.attention(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        weight = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)
        if self.event_summary == "calorimeter":
            if summaries is None or summaries.shape != (features.shape[0], 4):
                raise ValueError("calorimeter event_summary requires summaries with shape [batch, 4]")
        pooled_with_summary = torch.cat((pooled, summaries), dim=-1) if self.event_summary == "calorimeter" else pooled
        if self.readout_mode == "shared":
            return self.head(pooled_with_summary)
        if self.target_mode == "energy":
            return self.energy_head(pooled_with_summary)
        if self.target_mode == "angles":
            return self.angle_head(pooled)
        return torch.cat((self.energy_head(pooled_with_summary), self.angle_head(pooled)), dim=-1)

    def calibration_penalty(self) -> torch.Tensor:
        return self.energy_log_gain.square().mean()


class _MambaOrGRU(nn.Module):
    """Use a GRU only for local CPU/import smoke tests; CUDA uses real Mamba."""
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) if Mamba is not None else None
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mamba is not None and x.is_cuda:
            return self.mamba(x)
        return self.gru(x)[0]
