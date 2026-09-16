from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover
    Mamba = None


def collate(batch):
    values, targets = zip(*batch)
    width = max(len(item) for item in values)
    features = torch.zeros((len(values), width, 5), dtype=torch.float32)
    mask = torch.zeros((len(values), width), dtype=torch.bool)
    for index, item in enumerate(values):
        features[index, :len(item)] = torch.from_numpy(item)
        mask[index, :len(item)] = True
    return {"features": features, "mask": mask, "targets": torch.tensor(targets, dtype=torch.float32)}


class SparseMambaPhotonRegressor(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 6, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1, energy_reference: tuple[float, float] = (1.0, 1.0)):
        super().__init__()
        self.register_buffer("energy_reference", torch.tensor(energy_reference, dtype=torch.float32).clamp_min(1e-12))
        self.energy_log_gain = nn.Parameter(torch.zeros(2))
        self.detector_embedding = nn.Embedding(2, 8)
        self.input_projection = nn.Sequential(nn.LayerNorm(12), nn.Linear(12, dim), nn.GELU())
        self.blocks = nn.ModuleList([_MambaOrGRU(dim, d_state, d_conv, expand) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 3))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        energy, coordinates, detector = features[..., 0], features[..., 1:4], features[..., 4].long().clamp(0, 1)
        scaled_energy = torch.log1p(energy / (self.energy_reference[detector] * torch.exp(self.energy_log_gain[detector])))
        x = torch.cat((scaled_energy.unsqueeze(-1), coordinates / 40_000.0, self.detector_embedding(detector)), dim=-1)
        x = self.input_projection(x) * mask.unsqueeze(-1)
        for block in self.blocks:
            x = (x + block(x)) * mask.unsqueeze(-1)
        x = self.norm(x)
        scores = self.attention(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        weight = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weight.unsqueeze(-1), dim=1)
        return self.head(pooled)

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
