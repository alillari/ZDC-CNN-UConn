import unittest
import numpy as np
import torch

from zdc_momentum.contracts import photon_target, rotate_xz, target_to_cartesian
from zdc_momentum.data import _aggregate_cells, _compress_tokens, _selected_photon, _split
from zdc_momentum.evaluate import robust_gaussian
from zdc_momentum.model import SparseMambaPhotonRegressor


class ContractsTest(unittest.TestCase):
    def test_rotation_preserves_norm(self):
        x, z = rotate_xz(np.array([3.0]), np.array([4.0]), 0.025)
        self.assertTrue(np.allclose(x * x + z * z, 25.0))


    def test_target_round_trip(self):
        target = photon_target(2.0, -1.0, 100.0)
        cartesian = target_to_cartesian(np.array([[target.log_energy, target.theta_x, target.theta_y]]))[0]
        self.assertTrue(np.allclose(cartesian, [2.0, -1.0, 100.0], rtol=1e-5))


    def test_source_split_is_deterministic(self):
        self.assertEqual(_split(42, 7, .7, .15), _split(42, 7, .7, .15))

    def test_status_one_selects_the_propagated_photon(self):
        selected = _selected_photon(np.array([22, 22]), np.array([2, 1]), np.array([1., 2.]), np.array([0., 0.]), np.array([10., 20.]))
        self.assertEqual(selected, (2.0, 0.0, 20.0))

    def test_aggregation_and_compression_preserve_energy(self):
        values = np.array([[1, 0, 0, 1, 0], [2, 0, 0, 1, 0], [3, 1, 0, 2, 1], [4, 2, 0, 2, 1]], dtype=np.float32)
        aggregated = _aggregate_cells(values)
        compressed = _compress_tokens(aggregated, 2)
        self.assertEqual(len(aggregated), 3)
        self.assertAlmostEqual(float(compressed[:, 0].sum()), 10.0)


    def test_gaussian_fit_gate_and_fit(self):
        self.assertFalse(robust_gaussian(np.ones(3), 200)["reported"])
        values = np.random.default_rng(1).normal(.02, .1, 500)
        result = robust_gaussian(values, 200)
        self.assertTrue(result["reported"])
        self.assertLess(abs(result["mean"] - .02), .03)

    def test_cpu_model_smoke(self):
        model = SparseMambaPhotonRegressor(dim=16, layers=1, energy_reference=(.1, 1.0))
        output = model(torch.rand(2, 5, 5), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool))
        self.assertEqual(tuple(output.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
