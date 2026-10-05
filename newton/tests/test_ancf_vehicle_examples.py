# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for full-vehicle acceptance checks and shared CLI options."""

import contextlib
import importlib
import io
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.ancf.example_vehicle_ancf_tires import Example
from newton.tests.ancf_example_probe import _configuration
from newton.tests.ancf_vehicle_checks import check_telemetry_final, check_terrain_final, check_vehicle_step


class TestANCFVehicleExamples(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Vehicle kinematics require CUDA")
    def test_trial_bead_kinematics_matches_full_rigid_setup(self):

        _, cli, _ = _configuration("03_braking", {"coupling-method": "adaptive"})
        example = Example(args=Example.create_parser().parse_args(cli))
        solver = example.ancf_solver
        fields = ("node_x", "node_D", "node_xd", "node_Dd", "node_xdd", "node_Ddd")
        rng = np.random.default_rng(123)
        for _ in range(4):
            velocity = rng.normal(size=example.state_0.joint_qd.shape).astype(np.float32)
            example.state_0.joint_qd.assign(velocity)
            example._gs_coupler._predict_pose(example.state_0, example._sim_dt)
            saved = {name: wp.clone(getattr(solver, name)) for name in fields}
            example.solver.step_kinematics(example.state_0, example.state_rigid, example.control, None, example._sim_dt)
            example._prescribe_beads(inv_dt=1 / example._sim_dt)
            expected = {name: getattr(solver, name).numpy() for name in fields}
            for name in fields:
                wp.copy(getattr(solver, name), saved[name])
            # The staging kernel must work with a different subtree reference:
            # it transports the same physical wheel velocity to that origin.
            example.solver.mjw_data.subtree_com.fill_(wp.vec3(2, -3, 4))
            example._trial_kinematics()
            example._prescribe_beads(inv_dt=1 / example._sim_dt)
            for name in fields:
                np.testing.assert_allclose(
                    getattr(solver, name).numpy(),
                    expected[name],
                    rtol=2e-4,
                    atol=2e-3 if name.endswith("dd") else 5e-6,
                    err_msg=name,
                )

    @unittest.skipUnless(wp.is_cuda_available(), "Vehicle kinematics require CUDA")
    def test_cached_rigid_response_matches_recomputed_setup(self):

        _, cli, _ = _configuration("03_braking", {"coupling-method": "adaptive"})
        example = Example(args=Example.create_parser().parse_args(cli))
        solver = example.solver
        cache = example._gs_coupler._kinematic_cache
        self.assertTrue(cache)
        solver.step_kinematics(example.state_0, example.state_rigid, example.control, None, example._sim_dt)
        cache.copy(restore=False)
        for torque in (1.0, -3.0, 0.0):
            # Trial bead staging overwrites wheel kinematics; dynamics advances
            # qpos/qvel. The next rigid response must still start at t_n.
            example._trial_kinematics()
            cache.copy(restore=True)
            solver.xfrc_applied.fill_(wp.spatial_vector(0.0, 0.0, torque, 0.0, 2.0, 0.0))
            solver.step_dynamics(example.state_rigid)
            actual = (example.state_rigid.joint_q.numpy(), example.state_rigid.joint_qd.numpy())
            solver.step_kinematics(example.state_0, example.state_rigid, example.control, None, example._sim_dt)
            solver.xfrc_applied.fill_(wp.spatial_vector(0.0, 0.0, torque, 0.0, 2.0, 0.0))
            solver.step_dynamics(example.state_rigid)
            np.testing.assert_allclose(actual[0], example.state_rigid.joint_q.numpy(), atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(actual[1], example.state_rigid.joint_qd.numpy(), atol=2e-6, rtol=2e-5)

    def test_all_tires_reject_nonfinite_positions(self):
        # A rear tire can fail while the front tire remains finite. Check the
        # complete batch, including infinities, before reporting a valid step.
        example = SimpleNamespace(_n_nodes=2, _frame=7, _t=7 / 60)
        for tire in range(4):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(tire=tire, invalid=invalid):
                    positions = np.zeros((8, 3))
                    positions[2 * tire, 1] = invalid
                    example.ancf_solver = SimpleNamespace(node_x=wp.array(positions, dtype=wp.vec3, device="cpu"))
                    with self.assertRaisesRegex(AssertionError, "non-finite.*frame 7"):
                        check_vehicle_step(example)
        example.ancf_solver.node_x.zero_()
        check_vehicle_step(example)

    @staticmethod
    def terrain_example():
        return SimpleNamespace(
            ancf_solver=SimpleNamespace(node_x=wp.zeros(4, dtype=wp.vec3, device="cpu")),
            fault_count=0,
            _pose=np.array([2.0, 0.0, 1.0]),
            spawn=(0.0, 0.0, 0.0),
            terrain=SimpleNamespace(height_at=lambda x, y: 0.0, name="test", w=0.4),
            _mode="manual",
            turns=[],
            lateral=0.0,
        )

    def test_terrain_validation_rejects_reset_and_stationary_runs(self):
        example = self.terrain_example()
        with contextlib.redirect_stdout(io.StringIO()):
            check_terrain_final(example)
        example.fault_count = 1
        with self.assertRaisesRegex(AssertionError, "simulation faults"):
            check_terrain_final(example)
        example.fault_count = 0
        example._pose[0] = 0.5
        with self.assertRaisesRegex(AssertionError, "drove only"):
            check_terrain_final(example)

    def test_telemetry_validation_keeps_terrain_and_ghost_checks(self):
        example = self.terrain_example()
        example._ghost_points = wp.array([[np.nan, 0, 0]], dtype=wp.vec3, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(AssertionError, "ghost points"):
                check_telemetry_final(example)
            example.fault_count = 1
            with self.assertRaisesRegex(AssertionError, "simulation faults"):
                check_telemetry_final(example)

    def test_full_vehicle_cli_preserves_coupling_overrides(self):
        # Each launcher must forward the same implicit solver/torque budget;
        # scene-specific parsers must not replace the optimized coupling options.
        for name in ("vehicle_ancf_tires", "vehicle_ancf_sand", "vehicle_terrain", "vehicle_telemetry"):
            with self.subTest(example=name):
                module = importlib.import_module(f"newton.examples.ancf.example_{name}")
                args = module.Example.create_parser().parse_args(
                    [
                        "--coupling-method",
                        "auto",
                        "--gs-iters",
                        "6",
                        "--substeps",
                        "6",
                        "--nr-iters",
                        "2",
                        "--pcg-iters",
                        "10",
                        "--torque-alpha",
                        "1",
                    ]
                )
                self.assertEqual(args.coupling_method, "auto")
                self.assertEqual((args.substeps, args.nr_iters, args.pcg_iters, args.gs_iters), (6, 2, 10, 6))
                self.assertEqual(args.torque_alpha, 1.0)

    def test_rigid_vehicle_defaults_match_launcher(self):
        for case, name in (
            ("03_driving", "vehicle_ancf_tires"),
            ("05_relief", "vehicle_terrain"),
            ("07_replay", "vehicle_telemetry"),
        ):
            with self.subTest(example=name):
                parser = importlib.import_module(f"newton.examples.ancf.example_{name}").Example.create_parser()
                self.assertEqual(parser.parse_args([]).coupling_method, "auto")
                _, cli, _ = _configuration(case)
                args = parser.parse_args(cli)
                self.assertEqual(args.coupling_method, "auto")
                self.assertEqual((args.substeps, args.nr_iters, args.pcg_iters, args.gs_iters), (6, 2, 10, 6))
                self.assertEqual(args.torque_alpha, 1.0)
                for method in ("coupled-newton", "adaptive"):
                    _, cli, _ = _configuration(case, {"coupling-method": method})
                    self.assertEqual(parser.parse_args(cli).coupling_method, method)

    def test_sand_preserves_separate_coupling_default(self):
        module = importlib.import_module("newton.examples.ancf.example_vehicle_ancf_sand")
        parser = module.Example.create_parser()
        self.assertEqual(parser.parse_args([]).coupling_method, "aitken")
        self.assertEqual(parser.parse_args(["--coupling-method", "coupled-newton"]).coupling_method, "coupled-newton")


if __name__ == "__main__":
    unittest.main()
