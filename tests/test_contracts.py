import unittest
import numpy as np
import torch

from zdc_momentum.contracts import photon_target, rotate_xz, target_to_cartesian
from zdc_momentum.data import _aggregate_cells, _compress_tokens, _selected_photon, _split
from zdc_momentum.evaluate import robust_gaussian
from zdc_momentum.model import SparseMambaPhotonRegressor
from zdc_momentum.ninja_dataset import serialize
from zdc_momentum.train import regression_loss


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

    def test_no_raw_input_layernorm_smoke(self):
        model = SparseMambaPhotonRegressor(dim=16, layers=1, energy_reference=(.1, 1.0), input_normalization="none")
        self.assertIsInstance(model.input_projection[0], torch.nn.Identity)
        output = model(torch.rand(2, 5, 5), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool))
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_calorimeter_summary_readout_smoke(self):
        model = SparseMambaPhotonRegressor(dim=16, layers=1, energy_reference=(.1, 1.0), event_summary="calorimeter")
        output = model(
            torch.rand(2, 5, 5),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
            torch.rand(2, 4),
        )
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_separate_readouts_support_clean_target_arms(self):
        features = torch.rand(2, 5, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        energy = SparseMambaPhotonRegressor(dim=16, layers=1, energy_reference=(.1, 1.0), target_mode="energy", readout_mode="separate")
        angles = SparseMambaPhotonRegressor(dim=16, layers=1, energy_reference=(.1, 1.0), target_mode="angles", readout_mode="separate")
        joint = SparseMambaPhotonRegressor(dim=16, layers=1, energy_reference=(.1, 1.0), target_mode="joint", readout_mode="separate", event_summary="calorimeter")
        self.assertEqual(tuple(energy(features, mask).shape), (2, 1))
        self.assertEqual(tuple(angles(features, mask).shape), (2, 2))
        self.assertEqual(tuple(joint(features, mask, torch.rand(2, 4)).shape), (2, 3))

    def test_uncapped_serialization_preserves_all_merged_tokens(self):
        features = np.array([[1., 0., 0., 0.], [2., 1., 0., 0.], [3., 2., 0., 0.]], dtype=np.float32)
        detector = np.array([0, 0, 1], dtype=np.uint8)
        tokens = serialize(features, detector, np.array([[0., 0., 0.], [0., 0., 0.]]), np.array([[2., 1., 1.], [2., 1., 1.]]), max_tokens=None)
        self.assertEqual(len(tokens), 3)
        self.assertAlmostEqual(float(tokens[:, 0].sum()), 6.0)

    def test_relative_energy_huber_uses_fractional_residual(self):
        target = torch.tensor([[np.log(10.0), 0.0, 0.0]], dtype=torch.float32)
        prediction = torch.tensor([[np.log(11.0), 0.0, 0.0]], dtype=torch.float32)
        loss = regression_loss(prediction, target, torch.ones(3), energy_objective="relative", relative_energy_scale=.1)
        # Relative residual is exactly +10%; after division by the 10% fixed
        # scale, Huber(1) = 1/2.  The two angle losses are zero.
        self.assertAlmostEqual(float(loss), 1.0 / 6.0, places=5)

    def test_log_huber_matches_the_original_joint_objective(self):
        prediction = torch.tensor([[.2, -.1, .3], [.5, .2, -.4]])
        target = torch.tensor([[.1, -.2, .1], [.4, .5, -.1]])
        std = torch.tensor([.25, .5, .75])
        old = torch.nn.functional.huber_loss((prediction - target) / std, torch.zeros_like(target))
        new = regression_loss(prediction, target, std, loss="huber", energy_objective="log")
        self.assertTrue(torch.allclose(old, new))

    def test_mae_is_the_normalized_absolute_residual(self):
        target = torch.zeros((1, 3))
        std = torch.tensor([2.0, 4.0, 8.0])
        prediction = torch.tensor([[4.0, -2.0, 2.0]])
        loss = regression_loss(prediction, target, std, loss="mae")
        self.assertAlmostEqual(float(loss), (2.0 + .5 + .25) / 3.0, places=6)

    def test_single_task_losses_select_the_correct_targets(self):
        target = torch.tensor([[np.log(10.0), .2, -.4]], dtype=torch.float32)
        std = torch.tensor([1., .5, .25])
        energy_loss = regression_loss(torch.tensor([[np.log(11.0)]], dtype=torch.float32), target, std, target_mode="energy", energy_objective="relative", relative_energy_scale=.1)
        angle_loss = regression_loss(torch.tensor([[.3, -.5]]), target, std, target_mode="angles")
        self.assertAlmostEqual(float(energy_loss), .5, places=5)
        self.assertGreater(float(angle_loss), 0.0)


if __name__ == "__main__":
    unittest.main()
