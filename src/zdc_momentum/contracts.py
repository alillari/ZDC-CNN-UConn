from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

DETECTOR_WSI = 0
DETECTOR_SIPM = 1
FEATURE_DIM = 5  # log-energy, x, y, z, detector id


@dataclass(frozen=True)
class PhotonTarget:
    log_energy: float
    theta_x: float
    theta_y: float


def rotate_xz(x: np.ndarray, z: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(theta), math.sin(theta)
    return c * x + s * z, -s * x + c * z


def photon_target(px: float, py: float, pz: float) -> PhotonTarget:
    energy = float(np.sqrt(px * px + py * py + pz * pz))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("Photon momentum must have finite positive magnitude")
    return PhotonTarget(float(np.log(energy)), float(np.arctan2(px, pz)), float(np.arctan2(py, pz)))


def target_to_cartesian(values: np.ndarray) -> np.ndarray:
    """Convert [..., log(E), theta_x, theta_y] to Cartesian photon momentum."""
    energy = np.exp(values[..., 0])
    tx, ty = np.tan(values[..., 1]), np.tan(values[..., 2])
    pz = energy / np.sqrt(1.0 + tx * tx + ty * ty)
    return np.stack((tx * pz, ty * pz, pz), axis=-1)
