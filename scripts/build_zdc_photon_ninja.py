#!/usr/bin/env python
"""Build a minimal canonical photon mmap_ninja dataset from one ROOT file."""
from __future__ import annotations
import argparse
from pathlib import Path
from zdc_momentum.ninja import build_photon_ninja


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--chunk-entries", type=int, default=1000)
    parser.add_argument("--wsi-collection", default="ZDC_WSi_Hits")
    parser.add_argument("--sipm-collection", default="HcalFarForwardZDCHits")
    parser.add_argument("--rotation-theta-rad", type=float, default=.025)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    print(build_photon_ninja(args.root_file, args.output_dir, max_events=args.max_events, chunk_entries=args.chunk_entries, wsi_collection=args.wsi_collection, sipm_collection=args.sipm_collection, rotation_theta_rad=args.rotation_theta_rad, progress_every=args.progress_every))


if __name__ == "__main__": main()
