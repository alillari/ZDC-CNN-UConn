#!/usr/bin/env python
"""Inspect MCParticles row semantics before defining a ZDC truth selector."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import awkward as ak
import uproot


def _branch(tree, suffix: str) -> str:
    matches = [str(key) for key in tree.keys(recursive=True) if "MCParticles" in str(key) and str(key).endswith(suffix)]
    if len(matches) != 1: raise RuntimeError(f"Expected one MCParticles branch ending {suffix!r}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path)
    parser.add_argument("--events", type=int, default=20)
    args = parser.parse_args()
    tree = uproot.open(args.root_file)["events"]
    fields = {name: _branch(tree, suffix) for name, suffix in {"pdg": ".PDG", "generator_status": ".generatorStatus", "simulator_status": ".simulatorStatus", "px": ".momentum.x", "py": ".momentum.y", "pz": ".momentum.z", "vx": ".vertex.x", "vy": ".vertex.y", "vz": ".vertex.z"}.items()}
    arrays = tree.arrays(list(fields.values()), entry_stop=args.events, library="ak")
    row_counts, row_pdgs = Counter(), {}
    for event in range(len(arrays[fields["pdg"]])):
        nrows = len(arrays[fields["pdg"]][event]); row_counts[nrows] += 1
        print(f"event={event} n_mcparticles={nrows}")
        for row in range(nrows):
            pdg = int(arrays[fields["pdg"]][event][row])
            row_pdgs.setdefault(row, Counter())[pdg] += 1
            values = {name: float(arrays[key][event][row]) if name not in ("pdg", "generator_status", "simulator_status") else int(arrays[key][event][row]) for name, key in fields.items()}
            print(f"  row={row} {values}")
    print("row_count_distribution", dict(sorted(row_counts.items())))
    print("pdg_by_row", {row: dict(counts) for row, counts in sorted(row_pdgs.items())})


if __name__ == "__main__": main()
