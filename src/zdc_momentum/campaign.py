"""Manifest-driven campaign helper; invoke individual runs with the zdc CLI."""

from __future__ import annotations
import json
from pathlib import Path


def completed(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    return (path / "best.pt").exists() and (path / "evaluation_test.json").exists()


def write_manifest(root: str | Path, run_dirs: list[str]) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "campaign_manifest.json").write_text(json.dumps({"runs": run_dirs}, indent=2) + "\n")
