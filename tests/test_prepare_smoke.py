import tempfile
import unittest
from pathlib import Path

import awkward as ak
import uproot

from zdc_momentum.data import ZDCDataset, audit_root, prepare_dataset
from zdc_momentum.ninja import build_photon_ninja
from zdc_momentum.ninja_dataset import NinjaPhotonDataset, fit_statistics
import mmap_ninja


def _branches():
    return {
        "MCParticles/MCParticles.PDG": ak.Array([[11, 22], [11, 22], [11, 22]]),
        "MCParticles/MCParticles.generatorStatus": ak.Array([[2, 1], [2, 1], [2, 1]]),
        "MCParticles/MCParticles.momentum.x": ak.Array([[0., 1.], [0., 2.], [0., 3.]]),
        "MCParticles/MCParticles.momentum.y": ak.Array([[0., 0.], [0., 0.], [0., 0.]]),
        "MCParticles/MCParticles.momentum.z": ak.Array([[1., 100.], [1., 110.], [1., 120.]]),
        **{f"{prefix}/{prefix}.{field}": ak.Array([[value, value], [value], [value]]) for prefix, value in (("ZDC_WSi_Hits", .1), ("HcalFarForwardZDCHits", 1.0)) for field in ("energy", "position.x", "position.y", "position.z")},
    }


class PrepareSmokeTest(unittest.TestCase):
    def test_audit_and_prepare_rntuple(self):
        root = Path(tempfile.mkdtemp(prefix="zdc-test-"))
        for name in ("p1", "p2"):
            with uproot.recreate(root / f"{name}.root") as stream:
                stream["events"] = _branches()
        report = audit_root(root / "p1.root", "ZDC_WSi_Hits", "HcalFarForwardZDCHits", 3)
        self.assertEqual(report["status1_photon_candidates_per_event"]["exactly_one_fraction"], 1.0)
        config = {"seed": 1, "data": {"photon_files": [str(root / "p1.root"), str(root / "p2.root")], "output_dir": str(root / "dataset"), "wsi_collection": "ZDC_WSi_Hits", "sipm_collection": "HcalFarForwardZDCHits", "rotation_theta_rad": .025, "chunk_entries": 2, "min_positive_energy": 0}, "split": {"train_fraction": .7, "val_fraction": .15}, "model": {"max_tokens": 2}}
        prepared = prepare_dataset(config)
        dataset = ZDCDataset(prepared, 0, 2)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.manifest["n_events"], 6)

    def test_ninja_builder_alignment(self):
        root = Path(tempfile.mkdtemp(prefix="zdc-ninja-test-"))
        source = root / "p1.root"
        with uproot.recreate(source) as stream: stream["events"] = _branches()
        output = build_photon_ninja(source, root / "ninja", max_events=3, chunk_entries=2)
        features, detector = mmap_ninja.RaggedMmap(output / "features"), mmap_ninja.RaggedMmap(output / "detector_id")
        self.assertEqual(len(features), 3)
        self.assertEqual(len(features), len(detector))
        self.assertEqual(len(features[0]), len(detector[0]))
        self.assertTrue((output / "_SUCCESS").exists())

    def test_ninja_loader_statistics_and_serialization(self):
        root = Path(tempfile.mkdtemp(prefix="zdc-ninja-loader-test-"))
        source = root / "p1.root"
        with uproot.recreate(source) as stream:
            stream["events"] = _branches()
        output = build_photon_ninja(source, root / "ninja", max_events=3, chunk_entries=2)
        statistics = fit_statistics([str(output)], seed=1, train_fraction=.7, val_fraction=.15)
        dataset = NinjaPhotonDataset([str(output)], 0, 1, .7, .15, statistics, max_tokens=2)
        tokens, target, summary = dataset[0]
        self.assertEqual(tokens.shape[1], 5)
        self.assertLessEqual(len(tokens), 2)
        self.assertEqual(target.shape, (3,))
        self.assertEqual(summary.shape, (4,))
        self.assertEqual(len(statistics["coordinate_min"]), 2)
