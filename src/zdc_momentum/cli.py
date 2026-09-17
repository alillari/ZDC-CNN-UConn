from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from .config import load_config
from .data import audit_root, prepare_dataset
from .evaluate import evaluate
from .sum_baseline import fit_sum_baseline
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="zdc")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "prepare", "train", "evaluate", "sum-baseline", "campaign"):
        item = sub.add_parser(command); item.add_argument("--config", required=True)
        if command in ("train", "evaluate", "sum-baseline"): item.add_argument("--run-dir", required=True)
        if command == "campaign":
            item.add_argument("--output-dir", required=True)
            item.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args(); config = load_config(args.config)
    if args.command == "audit":
        files = config["data"]["photon_files"]
        if len(files) != 2: raise SystemExit("data.photon_files must contain two ROOT paths")
        report = [audit_root(path, config["data"]["wsi_collection"], config["data"]["sipm_collection"], int(config["data"]["audit_entries"])) for path in files]
        print(json.dumps(report, indent=2))
    elif args.command == "prepare": print(prepare_dataset(config))
    elif args.command == "train": print(train(config, args.run_dir))
    elif args.command == "evaluate": print(evaluate(config, args.run_dir))
    elif args.command == "sum-baseline": print(fit_sum_baseline(config, args.run_dir))
    else:
        root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
        runs = []
        for seed in args.seeds:
            rendered = dict(config); rendered["seed"] = seed
            run = root / f"seed_{seed:04d}"; run.mkdir(exist_ok=True)
            (run / "config.yaml").write_text(yaml.safe_dump(rendered, sort_keys=True))
            runs.append({"seed": seed, "run_dir": str(run), "state": "complete" if (run / "evaluation_test.json").exists() else "pending"})
        (root / "campaign_manifest.json").write_text(json.dumps({"runs": runs}, indent=2) + "\n")
        print(root / "campaign_manifest.json")


if __name__ == "__main__":
    main()
