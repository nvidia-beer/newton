# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression and synthetic identification checks for the uncoupled wheel."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.ancf_shell.kernels_element import compute_lumped_mass
from newton.examples.ancf._ancf_viz import material_row
from newton.solvers import isotropic_ancf_material, load_ancf_tire_usd
from newton.tests.ancf_example_probe import ASSETS
from newton.tests.ancf_wheel_calibration import experiment, fit_stiffness, make_wheel, observe


class TestWheelCalibrationSearch(unittest.TestCase):
    def test_recovers_unknown_scale_from_noisy_observations(self):
        t = np.linspace(0, 1, 40)

        def prediction(scale):
            return np.column_stack([0.1 * scale * t, 0.03 * np.sin(scale * t)])

        measured = prediction(1.37) + np.random.default_rng(7).normal(0, 0.0002, (40, 2))
        fit = fit_stiffness(measured, prediction)
        self.assertLess(abs(fit["stiffness_scale"] / 1.37 - 1), 0.01)
        self.assertFalse(fit["bound_hit"])

    def test_rejects_uninformative_observations(self):
        data = np.zeros((12, 2))
        with self.assertRaisesRegex(ValueError, "below measurement noise"):
            fit_stiffness(data, lambda scale: data.copy())

    def test_records_failed_candidate_without_selecting_it(self):
        data = np.full((12, 2), 1.2)

        def prediction(scale):
            if scale < 0.8:
                raise ValueError("Non-finite wheel state")
            return np.full((12, 2), scale)

        fit = fit_stiffness(data, prediction)
        self.assertTrue(any(trial["failure"] for trial in fit["trials"]))
        self.assertAlmostEqual(fit["stiffness_scale"], 1.2, delta=0.005)

    def test_rejects_all_failed_candidates(self):
        with self.assertRaisesRegex(ValueError, "All calibration trials failed"):
            fit_stiffness(np.zeros((12, 2)), lambda scale: np.full((12, 2), np.nan))

    def test_reports_parameter_at_bound(self):
        fit = fit_stiffness(np.full((12, 2), 3.0), lambda scale: np.full((12, 2), scale))
        self.assertTrue(fit["bound_hit"])
        self.assertAlmostEqual(fit["stiffness_scale"], 1.8)

    def test_rejects_invalid_observations_and_bounds(self):
        for data, bounds in [(np.zeros((4, 3)), (0.6, 1.8)), (np.zeros((4, 2)), (2, 1))]:
            with self.subTest(bounds=bounds, shape=data.shape), self.assertRaises(ValueError):
                fit_stiffness(data, lambda scale, data=data: data, bounds=bounds)


@unittest.skipUnless(wp.is_cuda_available(), "Wheel calibration requires CUDA")
class TestANCFWheelCalibration(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("ANCF_WHEEL_RECOVERY") == "1", "Set ANCF_WHEEL_RECOVERY=1 for full synthetic recovery"
    )
    def test_synthetic_stiffness_recovery_and_validation(self):
        output = Path(os.environ.get("ANCF_TEST_OUTPUT_DIR", tempfile.gettempdir())) / "wheel_recovery"
        report = experiment(output)
        self.assertTrue(report["recovery_pass"], report)
        self.assertTrue(report["validation_pass"], report)
        self.assertTrue(report["reference_time_refinement_pass"], report)
        self.assertLess(report["max_frame_sampled_penetration_m"], 0.01)
        self.assertFalse(report["physical_calibration"])
        # Production accuracy is reported independently, never certified by recovery.
        self.assertEqual(report["ready_for_coupling"], report["production_accuracy_pass"])

    def test_mass_assembly_is_reproducible_and_cartesian_isotropic(self):
        asset = ASSETS / "warthog_ancf_tire_simple.usda"
        model, _ = load_ancf_tire_usd(str(asset), device="cuda:0")
        mass = wp.zeros(model.n_nodes * 6, dtype=float, device="cuda:0")
        previous = None
        for _ in range(16):
            mass.zero_()
            wp.launch(
                compute_lumped_mass,
                dim=model.n_elems,
                inputs=[model.node_x0, model.node_D0, model.elem_nodes, model.elem_h, model.elem_mat, mass],
                device="cuda:0",
            )
            current = mass.numpy().reshape(-1, 6)
            for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
                np.testing.assert_array_equal(current[:, a], current[:, b])
            if previous is not None:
                np.testing.assert_array_equal(current, previous)
            previous = current

    def test_identical_drops_are_reproducible(self):
        first = observe(make_wheel(stiffness_scale=1.23), frames=48)
        second = observe(make_wheel(stiffness_scale=1.23), frames=48)
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)

    def test_single_wheel_material_override_reaches_solver(self):
        wheel = make_wheel(young_pa=3.7e6, density=987.0)
        expected = material_row(isotropic_ancf_material(E=3.7e6, nu=0.3, rho=987.0))
        np.testing.assert_allclose(wheel.ancf_model.elem_mat.numpy(), np.tile(expected, (wheel.ancf_model.n_elems, 1)))
        self.assertEqual(wheel.solver.n_envs, 1)
        self.assertEqual(wheel.model.body_count, 0, "This fixture must not introduce rigid coupling")

    def test_single_wheel_density_override_keeps_baked_stiffness(self):
        original = make_wheel()
        changed = make_wheel(density=1234.0)
        before, after = original.ancf_model.elem_mat.numpy(), changed.ancf_model.elem_mat.numpy()
        np.testing.assert_array_equal(before[:, :9], after[:, :9])
        np.testing.assert_array_equal(after[:, 9], 1234.0)

    def test_free_fall_cannot_identify_stiffness(self):
        low = observe(make_wheel(stiffness_scale=0.6), frames=6)
        high = observe(make_wheel(stiffness_scale=1.8), frames=6)
        np.testing.assert_allclose(low[:, :2], high[:, :2], atol=0.0005, rtol=0)
        t = np.arange(7) / 60
        np.testing.assert_allclose(low[:, 0], low[0, 0] - 0.5 * 9.81 * t**2, atol=9.81 / 600 * t[-1], rtol=0)
        self.assertEqual(float(low[:, 3].max()), 0.0)
        self.assertEqual(float(high[:, 3].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
