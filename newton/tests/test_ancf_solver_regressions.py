# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical and executed-work regressions for the implicit vehicle solver.

These tests require CUDA. Work counters execute inside the captured graph so
conditional branches are measured, rather than counted during graph creation.
They deliberately avoid wall-clock limits that depend on hardware and rendering.
"""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ancf_shell import solver_ancf_shell as fem
from newton._src.solvers.ancf_shell.model_ancf_shell import ANCFShellModel
from newton.examples.ancf.example_vehicle_ancf_tires import Example
from newton.tests.ancf_example_probe import _configuration
from newton.tests.test_ancf_shell_formulation import ElementFixture, rotation


@wp.kernel
def _count_work(counts: wp.array[int], index: int):
    counts[index] += 1


@unittest.skipUnless(wp.is_cuda_available(), "Implicit solver regressions require CUDA")
class TestANCFSolverRegressions(unittest.TestCase):
    def test_deferred_tangent_uses_current_state_in_replayed_graph(self):
        device, dt = "cuda:0", 1 / 600
        fixture = ElementFixture([rotation(0)], device)
        material = fixture.material.numpy()
        material[:, -1] = 0.009
        fixture.material.assign(material)
        ancf = ANCFShellModel(
            4,
            1,
            fixture.x0,
            fixture.d0,
            fixture.nodes,
            fixture.h,
            fixture.material,
            fixture.zeros((1, 5)),
            fixture.cos,
            fixture.sin,
            device=device,
        )
        model = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0).finalize(device=device)
        solver = fem.SolverANCFShell(model, ancf, n_envs=2, ground_z=-10, nr_max_iter=2, pcg_max_iter=10)
        solver._evaluate_forces_batched(dt)
        with wp.ScopedCapture(device=device) as force_capture:
            solver._evaluate_forces_batched(dt, assemble_tangent=False)
        with wp.ScopedCapture(device=device) as tangent_capture:
            solver._assemble_element_stiffness_batched()
        with wp.ScopedCapture(device=device) as full_capture:
            solver._evaluate_forces_batched(dt)

        rng = np.random.default_rng(2703)
        previous = None
        for replay in range(3):
            with self.subTest(replay=replay):
                positions = fixture.rest + rng.normal(scale=0.003, size=(2, 4, 3))
                directors = fixture.directors + rng.normal(scale=0.03, size=(2, 4, 3))
                velocities = rng.normal(scale=0.1, size=(2, 4, 3))
                director_velocities = rng.normal(scale=0.2, size=(2, 4, 3))
                expected_force, expected_tangent = [], []
                for env in range(2):
                    f, k = fixture.evaluate(positions[env], directors[env], velocities[env], director_velocities[env])
                    expected_force.append(f.ravel())
                    expected_tangent.append(k[0])
                for array, value in zip(
                    (solver.node_x, solver.node_D, solver.node_xd, solver.node_Dd),
                    (positions, directors, velocities, director_velocities),
                    strict=True,
                ):
                    array.assign(value.reshape(-1, 3))
                # A force-only evaluation must neither read nor overwrite the old tangent.
                solver.elem_K.fill_(float("nan"))
                wp.capture_launch(force_capture.graph)
                np.testing.assert_allclose(solver.global_f_int.numpy(), np.ravel(expected_force), rtol=2e-5, atol=2e-5)
                self.assertTrue(np.isnan(solver.elem_K.numpy()).all(), "Force-only evaluation rebuilt the tangent")
                external = solver.global_f_ext.numpy().copy()
                wp.capture_launch(tangent_capture.graph)
                current = solver.elem_K.numpy()
                np.testing.assert_allclose(current, expected_tangent, rtol=2e-5, atol=2e-5)
                if previous is not None:
                    self.assertGreater(float(np.max(np.abs(current - previous))), 1.0)
                previous = current.copy()

                # Standalone callers still receive a fresh tangent by default.
                solver.elem_K.fill_(float("nan"))
                wp.capture_launch(full_capture.graph)
                np.testing.assert_allclose(solver.elem_K.numpy(), expected_tangent, rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(solver.global_f_int.numpy(), np.ravel(expected_force), rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(solver.global_f_ext.numpy(), external)

    def test_pcg_graph_resets_scratch_and_reads_changed_systems(self):
        device, environments, nodes = "cuda:0", 2, 5
        n = 6 * nodes
        offsets = np.arange(nodes + 1, dtype=np.int32) * nodes
        columns = np.tile(np.arange(nodes, dtype=np.int32), nodes)
        solver = fem.PcgSolverBatched(environments, n, 36 * nodes * nodes, device, max_iters=4)
        solver.set_graph(offsets, columns)
        self.assertIsNotNone(solver._sweep_kernel)
        self.assertGreater(solver._n_dof_pad, n)
        o, c = [wp.array(a, dtype=int, device=device) for a in (offsets, columns)]
        values = wp.zeros(environments * solver.nnz, dtype=float, device=device)
        rhs = wp.zeros(environments * n, dtype=float, device=device)
        result = wp.zeros_like(rhs)
        rng = np.random.default_rng(913)
        # Warm compilation with a valid SPD matrix before capturing the solve.
        identity = np.tile(np.eye(n), (environments, 1, 1))
        values.assign(identity.reshape(environments, nodes, 6, nodes, 6).transpose(0, 1, 3, 2, 4).ravel())
        solver.solve(o, c, values, rhs, result, compute_residual_report=False)
        with wp.ScopedCapture(device=device) as capture:
            solver.solve(o, c, values, rhs, result, compute_residual_report=False)

        scratch = [*solver._r, *solver._p, *solver._rz, *solver._pAp, solver._x_pad, solver.Ap, solver.z, result]
        for replay in range(4):
            with self.subTest(replay=replay):
                seed = rng.normal(scale=0.1, size=(environments, n, n))
                matrix = (seed @ seed.transpose(0, 2, 1) + 5 * np.eye(n)).astype(np.float32)
                source = rng.normal(size=(environments, n)).astype(np.float32)
                source[replay % environments] = 0.0
                values.assign(matrix.reshape(environments, nodes, 6, nodes, 6).transpose(0, 1, 3, 2, 4).ravel())
                rhs.assign(source.ravel())
                # Finite poison models leftovers from a previous solve, including padding.
                for array in scratch:
                    array.fill_(17.0 + replay)
                wp.capture_launch(capture.graph)
                actual = result.numpy().reshape(environments, n)
                expected = np.linalg.solve(matrix.astype(np.float64), source.astype(np.float64)[..., None])[..., 0]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                residual = np.einsum("eij,ej->ei", matrix.astype(np.float64), actual) - source
                self.assertLess(float(np.linalg.norm(residual, axis=1).max()), 2e-5)
                np.testing.assert_array_equal(actual[replay % environments], 0.0)
                np.testing.assert_array_equal(solver._x_pad.numpy().reshape(environments, -1)[:, n:], 0.0)

    def test_moving_vehicle_skips_terminal_tangent_assembly(self):
        device = "cuda:0"
        _, cli, _ = _configuration("03_warthog_motion", {"coupling-method": "coupled-newton"})
        args = Example.create_parser().parse_args(cli)
        counts = wp.zeros(2, dtype=int, device=device)
        launch = wp.launch
        # Compile the counter before the example captures its conditional CUDA graph.
        launch(_count_work, dim=1, inputs=[counts, 0], device=device)

        def counted_launch(kernel, *args, **kwargs):
            if kernel is fem.compute_element_forces_stiffness_batched_gp:
                launch(_count_work, dim=1, inputs=[counts, 0], device=device)
            elif kernel is fem.compute_element_K_from_B:
                launch(_count_work, dim=1, inputs=[counts, 1], device=device)
            return launch(kernel, *args, **kwargs)

        with patch.object(wp, "launch", counted_launch):
            example = Example(args=args)
        coupler = example._gs_coupler
        self.assertIsNotNone(example._substep_graph, "Coupled CUDA graph was not captured")
        self.assertIsNotNone(coupler._coupled_solver, "Coupled Newton fell back to another method")
        self.assertEqual(example.ancf_solver.torque_alpha, 1.0)
        self.assertEqual((args.substeps, args.nr_iters, args.pcg_iters), (6, 2, 10))
        counts.zero_()
        start = coupler.interface_totals.numpy().copy()
        dofs = example.vehicle._axle_dofs.numpy()
        signs = example.vehicle._axle_sign.numpy()
        # Settle, accelerate, turn, and brake. The graph is replayed throughout.
        for frame in range(1, 721):
            example._target_wheel_speed = 0.0 if frame <= 120 or frame > 600 else (3.0 if frame <= 240 else 6.0)
            example.steer_angle = 0.5 if 480 < frame <= 600 else 0.0
            example.step()
            if frame % 120 == 0:
                with self.subTest(frame=frame):
                    steps, evaluations = coupler.interface_totals.numpy() - start
                    force_calls, tangent_calls = counts.numpy()
                    self.assertEqual(steps, frame * args.substeps)
                    self.assertEqual(force_calls, evaluations)
                    self.assertEqual(
                        tangent_calls, evaluations - steps, "Terminal evaluation assembled an unused tangent"
                    )
                    used = int(coupler.interface_iterations.numpy()[0])
                    self.assertGreater(used, 0)
                    self.assertLess(float(coupler.interface_residual.numpy()[used - 1]), 0.1)
                    self.assertTrue(np.isfinite(example.ancf_solver.node_x.numpy()).all())
                    speed = example.state_0.joint_qd.numpy()[dofs] * signs
                    self.assertTrue(np.isfinite(speed).all())
                    if frame == 480:
                        self.assertGreater(float(speed.min()), 5.0, "Vehicle did not reach the moving test condition")
                    if frame == 720:
                        self.assertLess(float(np.abs(speed).max()), 0.1, "Vehicle did not brake")


if __name__ == "__main__":
    unittest.main()
