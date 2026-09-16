from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("data", "split", "model", "training", "evaluation"):
        if key not in config or not isinstance(config[key], dict):
            raise ValueError(f"Missing mapping {key!r} in configuration")
    return config
