# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent finite-difference, adjoint, and production-force checks."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.examples.ancf.diffsim._tire_calibration import Measurement, fit_stiffness
from newton.examples.ancf.diffsim.example_diffsim_ancf_tire_lift import Example
from newton.solvers import ANCFTireEquilibrium


@unittest.skipUnless(wp.is_cuda_available(), "The physical wheel rig requires CUDA")
class TestANCFTireDifferentiation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = Example(None, Example.create_parser().parse_args(["--no-train"]))
        for _ in range(120):
            cls.example.step()
        cls.rig = cls.example.rig
        cls.solver = cls.rig.ancf_solver
        cls.eq = ANCFTireEquilibrium(
            cls.solver, cls.rig._bead_idx_np, cls.rig._m_rigid, cls.rig._f_load, cls.rig._build_pressures[0]
        )
        cls.initial = cls.eq.coordinates(cls.solver.node_x.numpy(), cls.solver.node_D.numpy(), cls.example.height)
        cls.pressure = cls.example.pressure_pa
        cls.result = cls.eq.solve(cls.pressure, cls.initial)
        cls.target = cls.eq.solve(sum(cls.example._pressure_bounds) / 2, cls.result.coordinates).height

    def test_analytic_residual_jacobian_matches_directional_difference(self):
        z = self.result.coordinates
        _, jac, rp = self.eq.evaluate(z, self.pressure)
        direction = np.random.default_rng(721).normal(size=z.size)
        direction /= np.linalg.norm(direction)
        for eps in (1e-5, 1e-6):
            plus = self.eq.evaluate(z + eps * direction, self.pressure, False)[0]
            minus = self.eq.evaluate(z - eps * direction, self.pressure, False)[0]
            fd = (plus - minus) / (2 * eps)
            self.assertLess(np.linalg.norm(jac @ direction - fd) / np.linalg.norm(fd), 2e-6)
        fd = (
            self.eq.evaluate(z, self.pressure + 10, False)[0] - self.eq.evaluate(z, self.pressure - 10, False)[0]
        ) / 20.0
        np.testing.assert_allclose(rp, fd, rtol=1e-8, atol=1e-10)

    def test_pressure_gradient_matches_resolved_equilibria(self):
        z = self.result.coordinates
        for pressure in np.linspace(*self.example._pressure_bounds, 4):
            solution = self.eq.solve(pressure, z)
            errors = []
            for eps in (100.0, 10.0):
                plus = self.eq.solve(pressure + eps, solution.coordinates)
                minus = self.eq.solve(pressure - eps, solution.coordinates)
                fd = (plus.height - minus.height) / (2 * eps)
                errors.append(abs(fd - solution.dh_dp) / abs(fd))
            self.assertLess(errors[-1], 1e-4)
            self.assertLess(errors[-1], errors[0])
            self.assertLess(solution.residual_norm, 1e-7)
            self.assertLess(solution.linear_residual, 1e-6)
            self.assertGreater(solution.dh_dp, 0.0)
            z = solution.coordinates

    def test_direct_sensitivity_equals_adjoint_loss_gradient(self):
        result = self.result
        _, jac, rp = self.eq.evaluate(result.coordinates, result.pressure)
        objective_z = np.zeros(self.eq.size)
        objective_z[-1] = result.height - 0.28
        adjoint = self.eq._solve_scaled(jac.T, objective_z)
        self.assertAlmostEqual(-adjoint @ rp, result.loss_gradient(0.28), delta=1e-14)

    def test_derivative_does_not_mutate_forward_solver(self):
        fields = ("node_x", "node_D", "node_xd", "node_Dd", "global_f_int", "global_f_ext", "cav_Kgas")
        before = {field: getattr(self.solver, field).numpy().copy() for field in fields}
        eas = self.solver.ancf.elem_eas_alpha.numpy().copy()
        result = self.eq.solve(self.pressure + 1000, self.result.coordinates, stiffness_scale=1.1)
        self.eq.pressure_step(result, self.target, self.example._pressure_bounds)
        for field in fields:
            np.testing.assert_array_equal(getattr(self.solver, field).numpy(), before[field])
        np.testing.assert_array_equal(self.solver.ancf.elem_eas_alpha.numpy(), eas)

    def test_equilibrium_height_matches_dynamic_settling(self):
        # After 2 s of HHT settling the rig sits 0.5 mm from the static equilibrium
        # (deterministic across runs): 0.2 % of the 0.256 m ride height and well
        # inside the 1 cm penetration screen used by the vehicle checks.
        self.assertLess(abs(self.result.height - self.example.height), 1e-3)

    def test_optimizer_reduces_loss_and_handles_pressure_bounds(self):
        result = self.result
        for _ in range(8):
            previous = abs(result.height - self.target)
            if previous < 5e-5:
                break
            result = self.eq.pressure_step(result, self.target, self.example._pressure_bounds)
            self.assertLess(abs(result.height - self.target), previous)
        self.assertLess(abs(result.height - self.target), 5e-5)
        with self.assertRaisesRegex(RuntimeError, "bounds"):
            self.eq.pressure_step(self.result, 0.5, (1.0, self.result.pressure))
        with self.assertRaisesRegex(RuntimeError, "converge"):
            self.eq.solve(45_000.0, self.initial, max_iterations=1)

    def test_small_blocks_preserve_single_tire_force_and_stiffness(self):
        # Compare the optimized launch against the previous 256-thread launch.
        self.solver._evaluate_forces_single(self.rig._sim_dt, update_eas=False)
        force, stiffness = self.solver.elem_f.numpy(), self.solver.elem_K.numpy()
        launch = wp.launch

        def previous_launch(kernel, *args, **kwargs):
            if kernel.key == "compute_element_forces_stiffness":
                kwargs["block_dim"] = 256
            return launch(kernel, *args, **kwargs)

        with patch.object(wp, "launch", previous_launch):
            self.solver._evaluate_forces_single(self.rig._sim_dt, update_eas=False)
        np.testing.assert_array_equal(self.solver.elem_f.numpy(), force)
        np.testing.assert_array_equal(self.solver.elem_K.numpy(), stiffness)

    def test_static_force_matches_production_single_tire_kernel(self):
        # Use a disposable solver state and restore it: production EAS update
        # first reaches its local fixed point, then a force-only call evaluates it.
        solver = self.solver
        fields = ("node_xd", "node_Dd", "node_f_ext_persistent", "global_f_int", "global_f_ext")
        before = {field: getattr(solver, field).numpy().copy() for field in fields}
        eas = solver.ancf.elem_eas_alpha.numpy().copy()
        try:
            solver.node_xd.zero_()
            solver.node_Dd.zero_()
            solver.node_f_ext_persistent.zero_()
            solver._evaluate_forces_single(self.rig._sim_dt, update_eas=True)
            solver._evaluate_forces_single(self.rig._sim_dt, update_eas=False)
            analytic, _ = self.eq._elastic(self.eq._full(self.initial), False)
            production = solver.elem_f.numpy()
            self.assertLess(np.linalg.norm(analytic - production) / np.linalg.norm(production), 2e-4)
            full_residual = solver.global_f_int.numpy() - solver.global_f_ext.numpy()
            reduced = np.r_[full_residual[self.eq.free], full_residual[self.eq.rim_dofs].sum() + self.eq.rim_load]
            expected = self.eq.evaluate(self.initial, self.pressure, False)[0]
            # Single-precision element roundoff accumulates over 64 bead nodes
            # in the rim equation; individual free-DOF forces remain tighter.
            self.assertLess(np.max(np.abs(expected[:-1] - reduced[:-1])), 0.02)
            self.assertLess(abs(expected[-1] - reduced[-1]) / self.eq.rim_load, 5e-4)
        finally:
            for field, value in before.items():
                getattr(solver, field).assign(value)
            solver.ancf.elem_eas_alpha.assign(eas)

    def test_stiffness_sensitivity_matches_resolved_equilibria(self):
        for scale in (0.8, 1.3):
            result = self.eq.solve(self.pressure, self.result.coordinates, stiffness_scale=scale)
            eps = 1e-4
            plus = self.eq.solve(self.pressure, result.coordinates, stiffness_scale=scale + eps)
            minus = self.eq.solve(self.pressure, result.coordinates, stiffness_scale=scale - eps)
            fd = (plus.height - minus.height) / (2 * eps)
            self.assertGreater(abs(fd), 1e-6)
            self.assertAlmostEqual(result.dh_dscale / fd, 1.0, delta=1e-4)

    def test_fit_recovers_stiffness_and_keeps_validation_out_of_updates(self):
        expected_scale = 1.35
        original_load = self.eq.rim_load
        rows = []
        try:
            for i, load in enumerate((200.0, 350.0, 450.0)):
                pressure = self.pressure + i * 3000
                self.eq.rim_load = self.eq.spindle_weight + load
                result = self.eq.solve(pressure, self.result.coordinates, stiffness_scale=expected_scale)
                # A deliberately biased held-out observation must not alter the fitted value.
                rows.append(
                    Measurement(
                        pressure,
                        load,
                        result.height + (0.002 if i == 2 else 0),
                        0.0002,
                        "validation" if i == 2 else "train",
                    )
                )
        finally:
            self.eq.rim_load = original_load
        report = fit_stiffness(self.eq, self.result.coordinates, rows)
        self.assertAlmostEqual(report["stiffness_scale"], expected_scale, delta=1e-4)
        self.assertLess(report["height_rms_m"]["train"], 1e-7)
        self.assertAlmostEqual(report["height_rms_m"]["validation"], 0.002, delta=1e-7)
        self.assertEqual(self.eq.rim_load, original_load)


if __name__ == "__main__":
    unittest.main()
