# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Element invariants independent of vehicle tuning; run inside the Newton image."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ancf_shell import kernels_coarse, kernels_gather, kernels_pcg
from newton._src.solvers.ancf_shell.kernels_element import compute_lumped_mass
from newton._src.solvers.ancf_shell.kernels_stiffness import (
    compute_element_forces_stiffness_batched_gp,
    compute_element_K_from_B,
    compute_rest_jacobians,
    sum_element_quadrature_forces,
)
from newton._src.solvers.ancf_shell.model_ancf_shell import ANCFShellModel
from newton._src.solvers.ancf_shell.solver_ancf_shell import PcgSolverBatched
from newton.examples.ancf._vehicle_kernels import prescribe_beads
from newton.examples.ancf.example_vehicle_ancf_tires import Example as VehicleExample
from newton.solvers import InterfaceCouplerGS, SolverANCFShell


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class ElementFixture:
    """Evaluate the production batched element kernel on independent shell patches."""

    def __init__(self, rotations, device, halfwidth=0.1):
        self.device = device
        self.rotations = rotations
        self.count = len(rotations)
        self.rest = halfwidth * np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        self.directors = np.tile([0.0, 0.0, 1.0], (4, 1))
        self.x0 = self.array(np.concatenate([self.rest @ r.T for r in rotations]), wp.vec3)
        self.d0 = self.array(np.concatenate([self.directors @ r.T for r in rotations]), wp.vec3)
        self.nodes = self.array(np.arange(4 * self.count).reshape(-1, 4), wp.int32)
        self.h = self.array(np.full(self.count, 0.01))
        # Isotropic lambda = mu = 40 kPa, rho = 700 kg/m^3, no damping.
        self.material = self.array(
            np.tile([120000, 120000, 120000, 40000, 40000, 40000, 40000, 40000, 40000, 700, 0], (self.count, 1))
        )
        self.cos = self.array(np.ones(self.count))
        self.sin = self.zeros(self.count)
        self.rest_data = [self.zeros(self.count)]
        self.rest_data += [self.zeros(self.count, wp.vec3) for _ in range(8)]
        self.rest_data += [self.zeros(self.count, wp.mat33) for _ in range(8)]
        wp.launch(
            compute_rest_jacobians,
            dim=self.count,
            inputs=[self.x0, self.d0, self.nodes, self.h, self.cos, self.sin, *self.rest_data],
            device=device,
        )
        self.f = self.zeros((self.count, 24))
        self.gp_force = self.zeros((self.count, 24, 12))
        self.k = self.zeros((self.count, 24, 24))
        self.bd = self.zeros((self.count, 72, 12))
        self.bs = [self.zeros((self.count, 24, 12)) for _ in range(3)]
        self.weights = self.zeros(self.count * 12)
        self.velocity = self.zeros(4 * self.count, wp.vec3)

    def array(self, data, dtype=float):
        return wp.array(data, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype=float):
        return wp.zeros(shape, dtype=dtype, device=self.device)

    def evaluate(self, positions, directors, velocity=None, director_velocity=None):
        self.gp_force.zero_()
        v = (
            self.velocity
            if velocity is None
            else self.array(np.concatenate([velocity @ r.T for r in self.rotations]), wp.vec3)
        )
        dv = (
            self.velocity
            if director_velocity is None
            else self.array(np.concatenate([director_velocity @ r.T for r in self.rotations]), wp.vec3)
        )
        x = self.array(np.concatenate([positions @ r.T for r in self.rotations]), wp.vec3)
        d = self.array(np.concatenate([directors @ r.T for r in self.rotations]), wp.vec3)
        wp.launch(
            compute_element_forces_stiffness_batched_gp,
            dim=self.count * 12,
            inputs=[
                x,
                d,
                v,
                dv,
                self.x0,
                self.d0,
                self.nodes,
                self.h,
                self.material,
                self.gp_force,
                self.k,
                self.bd,
                *self.bs,
                self.weights,
                self.cos,
                self.sin,
                self.count,
                self.count * 4,
                12,
                3,
                *self.rest_data[9:],
            ],
            device=self.device,
        )
        wp.launch(
            sum_element_quadrature_forces, dim=(self.count, 24), inputs=[self.gp_force, 12, self.f], device=self.device
        )
        wp.launch(
            compute_element_K_from_B,
            dim=self.count * 36,
            inputs=[self.bd, *self.bs, self.weights, self.material, self.k, 12],
            device=self.device,
        )
        return self.f.numpy().copy().reshape(self.count, 4, 2, 3), self.k.numpy().copy()


def reference_energy(x, d, x0, d0, h=0.01):
    """Double-precision covariant ANS energy, independent of production derivatives."""

    def shape(u, v):
        return np.array([(1 - u) * (1 - v), (1 + u) * (1 - v), (1 + u) * (1 + v), (1 - u) * (1 + v)]) / 4

    def jacobian(x, d, u, v, w):
        du = np.array([v - 1, 1 - v, 1 + v, -1 - v]) / 4
        dv = np.array([u - 1, -1 - u, 1 + u, 1 - u]) / 4
        return np.column_stack((du @ (x + w * h / 2 * d), dv @ (x + w * h / 2 * d), h / 2 * shape(u, v) @ d))

    def strain(u, v, w=0):
        j = jacobian(x, d, u, v, w)
        j0 = jacobian(x0, d0, u, v, w)
        return (j.T @ j - j0.T @ j0) / 2

    e33 = np.array([strain(u, v)[2, 2] for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    energy = 0.0
    z, weights = np.polynomial.legendre.leggauss(3)
    for u in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
        for v in [-1 / np.sqrt(3), 1 / np.sqrt(3)]:
            for w, weight in zip(z, weights, strict=True):
                e = strain(u, v, w)
                e[2, 2] = shape(u, v) @ e33
                e[0, 2] = e[2, 0] = (1 - v) / 2 * strain(0, -1)[0, 2] + (1 + v) / 2 * strain(0, 1)[0, 2]
                e[1, 2] = e[2, 1] = (1 - u) / 2 * strain(-1, 0)[1, 2] + (1 + u) / 2 * strain(1, 0)[1, 2]
                j0 = jacobian(x0, d0, u, v, w)
                inverse = np.linalg.inv(j0)
                cartesian = inverse.T @ e @ inverse
                energy += (
                    weight * abs(np.linalg.det(j0)) * (20000 * np.trace(cartesian) ** 2 + 40000 * np.sum(cartesian**2))
                )
    return energy


class TestANCFShellFormulation(unittest.TestCase):
    def test_cavity_parallel_reduction_preserves_volume_under_translation_and_scale(self):
        corners = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32
        )
        faces = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
        vertices = np.concatenate([corners + np.array([3 * i, 0, 0]) for i in range(7)]).astype(np.float32)
        quads = np.concatenate([faces + 8 * i for i in range(7)])
        n, ne = len(vertices), len(quads)
        positions = np.concatenate([vertices, 2 * vertices + [100, -50, 32]]).astype(np.float32)
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                x = wp.array(positions, dtype=wp.vec3, device=device)
                nodes = wp.array(quads, dtype=int, device=device)
                signs = wp.ones(ne, dtype=float, device=device)
                centroid_parts = wp.zeros((2, (n + 31) // 32), dtype=wp.vec3d, device=device)
                volume_parts = wp.zeros((2, (ne + 31) // 32), dtype=wp.float64, device=device)
                centre = wp.zeros(2, dtype=wp.vec3, device=device)
                volume = wp.zeros(2, dtype=float, device=device)
                wp.launch(
                    kernels_gather.cavity_centroid_parts,
                    dim=centroid_parts.shape,
                    inputs=[x, n, centroid_parts],
                    device=device,
                )
                wp.launch(
                    kernels_gather.cavity_centroid_finish, dim=2, inputs=[centroid_parts, n, centre], device=device
                )
                wp.launch(
                    kernels_gather.cavity_volume_parts,
                    dim=volume_parts.shape,
                    inputs=[x, nodes, signs, centre, n, ne, volume_parts],
                    device=device,
                )
                wp.launch(kernels_gather.cavity_volume_finish, dim=2, inputs=[volume_parts, volume], device=device)
                np.testing.assert_allclose(volume.numpy(), [7, 56], rtol=1e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Coarse factorization uses a CUDA block")
    def test_coarse_factorization_reuses_matrix_for_different_residuals(self):
        rng = np.random.default_rng(195)
        n_envs = 64
        b = rng.normal(size=(n_envs, 9, 9))
        matrices = b.transpose(0, 2, 1) @ b + np.eye(9)
        units = np.logspace(-3, 3, 9)
        matrices[1::4] *= units[:, None] * units[None, :]
        # Pinned modes and an entirely pinned tire must produce zero corrections.
        matrices[2::4, 6:, :] = 0.0
        matrices[2::4, :, 6:] = 0.0
        matrices[3::4] = 0.0
        matrix = wp.array(matrices.reshape(-1), dtype=wp.float64, device="cuda:0")
        factors = wp.zeros_like(matrix)
        scales = wp.zeros(n_envs * 9, dtype=wp.float64, device="cuda:0")
        rhs = wp.zeros(n_envs * 9, dtype=wp.float64, device="cuda:0")
        actual = wp.zeros((n_envs, 9), dtype=float, device="cuda:0")
        reference = wp.zeros_like(actual)
        wp.launch(
            kernels_coarse.factor_coarse,
            dim=(n_envs, 128),
            block_dim=128,
            inputs=[matrix, factors, scales],
            device="cuda:0",
        )
        wp.launch(
            kernels_coarse.solve_coarse_factored, dim=n_envs, inputs=[factors, scales, rhs, actual], device="cuda:0"
        )
        with wp.ScopedCapture(device="cuda:0") as capture:
            wp.launch(
                kernels_coarse.solve_coarse_factored, dim=n_envs, inputs=[factors, scales, rhs, actual], device="cuda:0"
            )
        for _ in range(3):
            projected = rng.normal(size=(n_envs, 9))
            rhs.assign(projected.reshape(-1))
            wp.capture_launch(capture.graph)
            wp.launch(kernels_coarse.solve_coarse, dim=n_envs, inputs=[matrix, rhs, reference], device="cuda:0")
            np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=2e-6, atol=1e-7)
            expected = np.zeros((n_envs, 9))
            for i in range(n_envs):
                active = (9, 9, 6, 0)[i % 4]
                if active:
                    expected[i, :active] = np.linalg.solve(matrices[i, :active, :active], projected[i, :active])
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=1e-7)

    def test_coarse_modes_and_products_match_dense_reference(self):
        rng = np.random.default_rng(981)
        positions = rng.normal(size=(4, 3)).astype(np.float32)
        directors = rng.normal(size=(4, 3)).astype(np.float32)
        free = rng.integers(0, 2, 24).astype(np.float32)
        matrix = rng.normal(size=(24, 24)).astype(np.float32)
        expected = np.zeros((24, 9))
        for i in range(4):
            p = positions[i] - positions[0]
            d = directors[i]
            expected[i * 6 : i * 6 + 3, :3] = np.eye(3)
            for axis in range(3):
                expected[i * 6 : i * 6 + 3, axis + 3] = np.cross(np.eye(3)[axis], p)
                expected[i * 6 + 3 : i * 6 + 6, axis + 3] = np.cross(np.eye(3)[axis], d)
            expected[i * 6 : i * 6 + 3, 6:] = np.diag(p)
            expected[i * 6 + 3 : i * 6 + 6, 6:] = np.diag(d)
        expected *= free[:, None]
        blocks = np.concatenate(
            [matrix[6 * i : 6 * i + 6, 6 * j : 6 * j + 6].reshape(-1) for i in range(4) for j in range(4)]
        )
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                q, aq = (wp.zeros(9 * 32, dtype=float, device=device) for _ in range(2))
                wp.launch(
                    kernels_coarse.build_modes,
                    dim=(9, 24),
                    inputs=[
                        wp.array(positions, dtype=wp.vec3, device=device),
                        wp.array(directors, dtype=wp.vec3, device=device),
                        wp.array(free, dtype=float, device=device),
                        wp.array([0, 4, 8, 12, 16], dtype=int, device=device),
                        wp.array(list(range(4)) * 4, dtype=int, device=device),
                        wp.array(blocks, dtype=float, device=device),
                        24,
                        len(blocks),
                        32,
                        q,
                        aq,
                    ],
                    device=device,
                )
                np.testing.assert_allclose(q.numpy().reshape(9, 32)[:, :24].T, expected, atol=1e-7)
                np.testing.assert_allclose(aq.numpy().reshape(9, 32)[:, :24].T, matrix @ expected, atol=2e-6, rtol=2e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "The captured PCG solver requires CUDA")
    def test_force_history_matches_accepted_state(self):
        fixture = ElementFixture([rotation(0)], "cuda:0")
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
            device="cuda:0",
        )
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        solver = SolverANCFShell(
            builder.finalize(device="cuda:0"), ancf, n_envs=2, nr_max_iter=2, pcg_max_iter=10, ground_z=-10
        )
        solver.node_f_ext_persistent.assign(np.tile([[0, 0, 1000], [0, 0, 0], [0, 0, 0], [0, 0, 0]], (2, 1)))
        solver.capture_graph(1 / 600)
        solver.graph_step()
        expected, _ = fixture.evaluate(solver.node_x.numpy()[:4], solver.node_D.numpy()[:4])
        np.testing.assert_allclose(solver.global_f_int.numpy()[:24], expected.reshape(-1), rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "The captured PCG solver requires CUDA")
    def test_prescribed_motion_preserves_target_and_acceleration(self):
        fixture = ElementFixture([rotation(0)], "cuda:0")
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
            device="cuda:0",
        )
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
        solver = SolverANCFShell(
            builder.finalize(device="cuda:0"), ancf, n_envs=2, nr_max_iter=2, pcg_max_iter=10, ground_z=-10
        )
        solver.set_dirichlet_nodes(np.arange(8))
        acceleration = np.tile([1, 2, 3], (8, 1)).astype(np.float32)
        director_acceleration = np.tile([0.1, 0.2, 0.3], (8, 1)).astype(np.float32)
        solver.node_xdd.assign(acceleration)
        solver.node_Ddd.assign(director_acceleration)
        targets = [a.numpy() for a in (solver.node_x, solver.node_xd, solver.node_D, solver.node_Dd)]
        solver.capture_graph(1 / 600)
        solver.graph_step()
        for actual, expected in zip(
            (solver.node_x, solver.node_xd, solver.node_D, solver.node_Dd), targets, strict=True
        ):
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7)
        np.testing.assert_allclose(solver.node_xdd.numpy(), acceleration, atol=1e-6)
        np.testing.assert_allclose(solver.node_Ddd.numpy(), director_acceleration, atol=1e-6)
        solver.set_dirichlet_nodes(None)
        np.testing.assert_array_equal(solver.pcg.coarse_free.numpy(), 1.0)

    def test_bead_prediction_includes_rotation(self):
        fixture = ElementFixture([rotation(0)], "cpu")
        state = [fixture.zeros(1, wp.vec3) for _ in range(6)]
        wp.launch(
            prescribe_beads,
            dim=1,
            inputs=[
                fixture.array([[[0, 0, 0]]], wp.vec3),
                fixture.array([[[1, 0, 0, 0]]], wp.quat),
                fixture.array([[[0, 0, 2, 0, 0, 0]]], wp.spatial_vector),
                fixture.array([[[0, 0, 0]]], wp.vec3),
                *[fixture.array([0], int) for _ in range(3)],
                fixture.array([[0, 0, 1]], wp.vec3),
                fixture.array([[0, 0, 1]], wp.vec3),
                *state,
                1,
                1,
                0.0,
                0.05,
            ],
            device="cpu",
        )
        expected = [[np.sin(0.1), 0, np.cos(0.1)]]
        np.testing.assert_allclose(state[0].numpy(), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(state[3].numpy(), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(state[1].numpy(), [[2 * np.cos(0.1), 0, -2 * np.sin(0.1)]], atol=1e-6)

    def test_coupling_retry_restores_external_force_history(self):
        node_names = ("node_x", "node_xd", "node_xdd", "node_D", "node_Dd", "node_Ddd")
        force_names = ("global_f_int", "global_f_int0", "global_f_ext", "global_f_ext0")
        solver = SimpleNamespace(
            **{name: wp.zeros(4, dtype=wp.vec3, device="cpu") for name in node_names},
            **{name: wp.full(24, float(i + 1), device="cpu") for i, name in enumerate(force_names)},
        )
        state = SimpleNamespace(
            **{name: wp.zeros(1, device="cpu") for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
        )
        coupler = InterfaceCouplerGS(solver)
        coupler.allocate(state)
        solver.global_f_ext.fill_(100.0)
        solver.global_f_ext0.fill_(200.0)
        coupler._restore_extra()
        np.testing.assert_array_equal(solver.global_f_ext.numpy(), 3.0)
        np.testing.assert_array_equal(solver.global_f_ext0.numpy(), 4.0)

    def test_vehicle_support_check_uses_full_rigid_mass(self):
        # 100 kg vehicle. The single-wheel rig metadata deliberately differs
        # from the vehicle's quarter mass.
        def array(values, dtype=float):
            return wp.array(values, dtype=dtype, device="cpu")

        example = SimpleNamespace(
            _n_nodes=1,
            _bead_idx_np=np.array([0]),
            _bead_rest_np=np.zeros((1, 3)),
            _spindle_body_indices_np=[0, 1, 2, 3],
            _frame=0,
            _t=0.0,
            model=SimpleNamespace(body_mass=array([80.0, 5.0, 5.0, 5.0, 5.0])),
            spec=SimpleNamespace(rigid_corner_mass=1.0),
            state_0=SimpleNamespace(body_q=array([[0, 0, 0, 0, 0, 0, 1]] * 4, wp.transform)),
            ancf_solver=SimpleNamespace(
                node_x=array(np.zeros((4, 3)), wp.vec3),
                node_xd=array(np.zeros((4, 3)), wp.vec3),
                global_f_int=array(np.zeros(24)),
                _xfrc_stg_per_tire=[array([[0, 0, 0, 0, 25.0 * 9.81, 0]], wp.spatial_vector) for _ in range(4)],
            ),
        )
        VehicleExample.test_final(example)
        example.model.body_mass = array([800.0, 50.0, 50.0, 50.0, 50.0])
        with self.assertRaisesRegex(AssertionError, "F_z"):
            VehicleExample.test_final(example)

    def test_bending_is_independent_of_world_offset(self):
        # Binary-exact vertex coordinates isolate arithmetic cancellation from
        # quantisation of the input positions themselves.
        fixture = ElementFixture([rotation(0)], "cpu", halfwidth=0.125)
        directors = fixture.directors.copy()
        directors[:, 0] = fixture.rest[:, 0]
        force, stiffness = fixture.evaluate(fixture.rest, directors)
        translated_force, translated_stiffness = fixture.evaluate(
            fixture.rest + np.array([512.0, 1024.0, 2048.0]), directors
        )
        np.testing.assert_allclose(translated_force, force, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(translated_stiffness, stiffness, rtol=2e-4, atol=0.01)

    @unittest.skipUnless(wp.is_cuda_available(), "The captured PCG solver requires CUDA")
    def test_single_environment_history_preserves_internal_variables(self):
        fixture = ElementFixture([rotation(0)], "cuda:0")
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
            device="cuda:0",
        )
        model = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0).finalize(device="cuda:0")
        solver = SolverANCFShell(model, ancf, n_envs=1, nr_max_iter=2, pcg_max_iter=10, ground_z=-10)
        solver.node_f_ext_persistent.assign([[1000, 500, 1000], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        solver.capture_graph(1 / 600)
        solver.graph_step()
        accepted_force = solver.global_f_int.numpy().copy()
        accepted_alpha = ancf.elem_eas_alpha.numpy().copy()
        solver._evaluate_forces_single(1 / 600)
        np.testing.assert_allclose(solver.global_f_int.numpy(), accepted_force, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(ancf.elem_eas_alpha.numpy(), accepted_alpha)

    @unittest.skipUnless(wp.is_cuda_available(), "PCG uses CUDA tile kernels")
    def test_symmetric_preconditioner_matches_dense_reference(self):
        rng = np.random.default_rng(4)
        A = np.zeros((18, 18))
        for i in range(3):
            B = rng.normal(size=(6, 6))
            A[i * 6 : i * 6 + 6, i * 6 : i * 6 + 6] = B.T @ B + 10 * np.eye(6)
        for i in range(2):
            B = rng.normal(size=(6, 6)) * 0.3 - 2 * np.eye(6)
            A[i * 6 : i * 6 + 6, (i + 1) * 6 : (i + 2) * 6] = B
            A[(i + 1) * 6 : (i + 2) * 6, i * 6 : i * 6 + 6] = B.T
        rows = [0, 2, 5, 7]
        cols = [0, 1, 0, 1, 2, 1, 2]
        values = np.concatenate(
            [
                A[6 * i : 6 * (i + 1), 6 * j : 6 * (j + 1)].reshape(-1)
                for i in range(3)
                for j in cols[rows[i] : rows[i + 1]]
            ]
        )
        s = PcgSolverBatched(1, 18, len(values), "cuda:0", max_iters=10)
        s.set_graph(np.array(rows), np.array(cols))
        o = wp.array(rows, dtype=int, device="cuda:0")
        c = wp.array(cols, dtype=int, device="cuda:0")
        v = wp.array(values, dtype=float, device="cuda:0")
        s._prepare_preconditioner(o, c, v)
        rhs = rng.normal(size=128).astype(np.float32)
        rhs[18:] = 0
        s._precondition(o, c, v, wp.array(rhs, device="cuda:0"))
        colors = s._node_colors.numpy()
        D = np.zeros_like(A)
        L = np.zeros_like(A)
        for i in range(3):
            for j in range(3):
                ix = np.s_[6 * i : 6 * i + 6, 6 * j : 6 * j + 6]
                if i == j:
                    D[ix] = A[ix]
                elif colors[j] < colors[i]:
                    L[ix] = A[ix]
        expected = np.linalg.solve(D + L.T, D @ np.linalg.solve(D + L, rhs[:18]))
        np.testing.assert_allclose(s.z.numpy()[:18], expected, rtol=1e-5, atol=1e-7)

    @unittest.skipUnless(wp.is_cuda_available(), "PCG uses CUDA tile kernels")
    def test_captured_preconditioner_reloads_matrix_coefficients(self):
        # A star graph has uneven row degrees; a diagonal graph has no sweep edges.
        rng = np.random.default_rng(82)
        for n_nodes in (1, 5):
            with self.subTest(n_nodes=n_nodes):
                offsets, columns = [0], []
                for node in range(n_nodes):
                    columns.extend(range(n_nodes) if node == 0 else (0, node))
                    offsets.append(len(columns))
                n_envs, n_dof = 2, 6 * n_nodes
                nnz = 36 * len(columns)
                solver = PcgSolverBatched(n_envs, n_dof, nnz, "cuda:0")
                solver.set_graph(np.array(offsets), np.array(columns))
                self.assertIsNotNone(solver._sweep_kernel)
                o, c = [wp.array(a, dtype=int, device="cuda:0") for a in (offsets, columns)]
                values = wp.zeros(n_envs * nnz, dtype=float, device="cuda:0")
                rhs = wp.zeros(n_envs * solver._n_dof_pad, dtype=float, device="cuda:0")
                colors = solver._node_colors.numpy()
                with wp.ScopedCapture(device="cuda:0") as capture:
                    solver._prepare_preconditioner(o, c, values)
                    solver._precondition(o, c, values, rhs)
                # Change matrix and RHS after capture, including full off-diagonal 6x6 blocks.
                for _ in range(2):
                    packed, expected, sources = [], [], []
                    for _env in range(n_envs):
                        a = 20 * np.eye(n_dof)
                        for i in range(n_nodes):
                            b = rng.normal(size=(6, 6))
                            a[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] += b.T @ b
                            if i:
                                a[:6, 6 * i : 6 * i + 6] = b * 0.2
                                a[6 * i : 6 * i + 6, :6] = b.T * 0.2
                        diagonal, lower = np.zeros_like(a), np.zeros_like(a)
                        for i in range(n_nodes):
                            for block in range(offsets[i], offsets[i + 1]):
                                j = columns[block]
                                index = np.s_[6 * i : 6 * i + 6, 6 * j : 6 * j + 6]
                                packed.extend(a[index].reshape(-1))
                                if i == j:
                                    diagonal[index] = a[index]
                                elif colors[j] < colors[i]:
                                    lower[index] = a[index]
                        b = rng.normal(size=n_dof)
                        sources.extend(np.pad(b, (0, solver._n_dof_pad - n_dof)))
                        expected.append(
                            np.linalg.solve(diagonal + lower.T, diagonal @ np.linalg.solve(diagonal + lower, b))
                        )
                    values.assign(np.asarray(packed, dtype=np.float32))
                    rhs.assign(np.asarray(sources, dtype=np.float32))
                    wp.capture_launch(capture.graph)
                    actual = solver.z.numpy().reshape(n_envs, -1)
                    np.testing.assert_allclose(actual[:, :n_dof], expected, rtol=2e-5, atol=1e-7)
                    np.testing.assert_array_equal(actual[:, n_dof:], 0.0)

    def test_fused_preconditioner_matches_distributed_sweeps(self):
        # Include Warthog-sized batches, padding, and unequal color groups.
        rng = np.random.default_rng(73)
        for height, width in ((8, 30), (7, 9)):
            with self.subTest(shape=(height, width)):
                n_nodes, n_envs = height * width, 3
                offsets, columns = [0], []
                blocks = []
                for node in range(n_nodes):
                    y, x = divmod(node, width)
                    neighbors = sorted(
                        {((y + dy) % height) * width + (x + dx) % width for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                    )
                    for other in neighbors:
                        columns.append(other)
                        blocks.append((12.0 if other == node else -1.0) * np.eye(6))
                    offsets.append(len(columns))
                values = np.tile(np.array(blocks).reshape(-1), n_envs).astype(np.float32)
                s = PcgSolverBatched(n_envs, n_nodes * 6, len(values) // n_envs, "cuda:0", max_iters=10)
                s.set_graph(np.array(offsets), np.array(columns))
                self.assertIsNotNone(s._sweep_kernel)
                o, c = [wp.array(v, dtype=int, device="cuda:0") for v in (offsets, columns)]
                v = wp.array(values, device="cuda:0")
                s._prepare_preconditioner(o, c, v)
                rhs_np = rng.normal(size=(n_envs, s._n_dof_pad)).astype(np.float32)
                rhs_np[:, n_nodes * 6 :] = 0
                rhs = wp.array(rhs_np.reshape(-1), device="cuda:0")
                s._precondition(o, c, v, rhs)  # compile before capture
                with wp.ScopedCapture(device="cuda:0") as capture:
                    s._precondition(o, c, v, rhs)
                # Captured sweeps must read the current RHS and reset shared state.
                for scale in (1.0, -2.3):
                    rhs.assign((rhs_np * scale).reshape(-1))
                    wp.capture_launch(capture.graph)
                    fused = s.z.numpy()
                    kernel, packed_values = s._sweep_kernel, s._sweep_packed
                    s._sweep_kernel, s._sweep_packed = None, None
                    s._prepare_preconditioner(o, c, v)
                    s._precondition(o, c, v, rhs)
                    s._sweep_kernel, s._sweep_packed = kernel, packed_values
                    np.testing.assert_allclose(fused, s.z.numpy(), rtol=2e-6, atol=2e-7)
                    self.assertGreater(float(np.dot(fused, (rhs_np * scale).reshape(-1))), 0.0)
                    np.testing.assert_array_equal(fused.reshape(n_envs, -1)[:, n_nodes * 6 :], 0.0)

                # Exercise reuse of the Jacobi product inside actual PCG updates.
                rhs_solve = (rhs_np[:, : n_nodes * 6] * -2.3).copy()
                solution = wp.empty(n_envs * n_nodes * 6, dtype=float, device="cuda:0")
                s.solve(o, c, v, wp.array(rhs_solve.reshape(-1), device="cuda:0"), solution)
                x = solution.numpy().reshape(n_envs, n_nodes, 6)
                # The final preconditioner only supplies diagnostics, not another PCG update.
                for iterations in (1, 3, 10):
                    s.max_iters = iterations
                    source = wp.array(rhs_solve.reshape(-1), device="cuda:0")
                    s.solve(o, c, v, source, solution)
                    reference = solution.numpy()
                    with wp.ScopedCapture(device="cuda:0") as capture:
                        s.solve(o, c, v, source, solution, compute_residual_report=False)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(solution.numpy(), reference, rtol=2e-6, atol=2e-7)
                ax = np.zeros_like(x)
                for node in range(n_nodes):
                    for block in range(offsets[node], offsets[node + 1]):
                        ax[:, node] += x[:, columns[block]] @ blocks[block].T
                relative_residual = np.linalg.norm(ax.reshape(n_envs, -1) - rhs_solve, axis=1) / np.linalg.norm(
                    rhs_solve, axis=1
                )
                self.assertLess(float(relative_residual.max()), 5e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "PCG uses CUDA tile kernels")
    def test_packed_preconditioner_preserves_csr_arithmetic(self):
        rng = np.random.default_rng(907)
        height, width, environments = 8, 30, 2
        n_nodes = height * width
        offsets, columns, rows = [0], [], []
        for node in range(n_nodes):
            y, x = divmod(node, width)
            neighbors = sorted(
                {((y + dy) % height) * width + (x + dx) % width for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            )
            columns.extend(neighbors)
            rows.extend([node] * len(neighbors))
            offsets.append(len(columns))
        solver = PcgSolverBatched(environments, 6 * n_nodes, 36 * len(columns), "cuda:0")
        solver.set_graph(np.asarray(offsets), np.asarray(columns))
        self.assertIsNotNone(solver._sweep_packed)
        self.assertLessEqual(solver._sweep_sources.size, 4 * solver.nnz)
        reference = kernels_pcg.make_colored_sweep(solver._n_dof_pad, len(solver._color_nodes))
        o, c = [wp.array(a, dtype=int, device="cuda:0") for a in (offsets, columns)]
        matrix = wp.zeros(environments * solver.nnz, dtype=float, device="cuda:0")
        rhs = wp.zeros(environments * solver._n_dof_pad, dtype=float, device="cuda:0")
        with wp.ScopedCapture(device="cuda:0") as capture:
            solver._prepare_preconditioner(o, c, matrix)
            solver._precondition(o, c, matrix, rhs)
        # Dense 6x6 off-diagonal blocks exercise every packed row and column.
        # Mutate coefficients after capture to detect stale packed values.
        for _ in range(2):
            blocks = rng.normal(scale=0.1, size=(environments, len(columns), 6, 6)).astype(np.float32)
            for edge, (row, col) in enumerate(zip(rows, columns, strict=True)):
                if row == col:
                    blocks[:, edge] = np.eye(6, dtype=np.float32) * 12
            matrix.assign(blocks.reshape(-1))
            source = rng.normal(size=(environments, solver._n_dof_pad)).astype(np.float32)
            source[:, solver.n_dof :] = 0
            rhs.assign(source.reshape(-1))
            # The fused pack must not materialize the unused CSR-scaled matrix.
            solver._scaled_blocks.fill_(float("nan"))
            wp.capture_launch(capture.graph)
            packed = solver.z.numpy()
            self.assertTrue(
                np.isnan(solver._scaled_blocks.numpy()).all(), "Packed setup rebuilt unused CSR-scaled blocks"
            )
            packed_values, packed_kernel = solver._sweep_packed, solver._sweep_kernel
            solver._sweep_packed, solver._sweep_kernel = None, reference
            solver._prepare_preconditioner(o, c, matrix)
            solver._precondition(o, c, matrix, rhs)
            np.testing.assert_array_equal(packed, solver.z.numpy())
            solver._sweep_packed, solver._sweep_kernel = packed_values, packed_kernel

    def test_batched_damping_matches_matrix_product_and_dissipates_energy(self):
        rng = np.random.default_rng(51)
        for device in ("cpu", "cuda:0"):
            if device == "cuda:0" and not wp.is_cuda_available():
                continue
            with self.subTest(device=device):
                fixture = ElementFixture([rotation(0), rotation(0.4), rotation(-0.7)], device)
                material = fixture.material.numpy()
                material[:, :3] *= [1.0, 1.3, 1.6]
                material[:, 6:9] *= [0.8, 1.2, 1.5]
                material[:, 10] = [0.0, 0.003, 0.015]
                fixture.material.assign(material)
                fixture.cos.assign(np.full(3, np.cos(0.3), dtype=np.float32))
                fixture.sin.assign(np.full(3, np.sin(0.3), dtype=np.float32))
                x = fixture.rest + rng.normal(scale=0.002, size=(4, 3))
                d = fixture.directors + rng.normal(scale=0.02, size=(4, 3))
                v, dv = rng.normal(size=(2, 4, 3))
                elastic, stiffness = fixture.evaluate(x, d)
                damped, _ = fixture.evaluate(x, d, v, dv)
                dofs = np.stack([np.stack((v @ r.T, dv @ r.T), axis=1).reshape(24) for r in fixture.rotations])
                expected = material[:, 10, None] * np.einsum("eij,ej->ei", stiffness.astype(float), dofs)
                actual = (damped - elastic).reshape(3, 24)
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-5)
                self.assertTrue((np.einsum("ei,ei->e", actual, dofs) >= -1e-8).all())
                np.testing.assert_array_equal(damped[0], elastic[0])
                # Rigid motion has zero strain rate, even in the deformed state.
                spin = np.array([0.7, -0.2, 0.4])
                rigid_v = np.cross(spin, x) + np.array([1.0, -0.5, 0.3])
                rigid_dv = np.cross(spin, d)
                rigid, _ = fixture.evaluate(x, d, rigid_v, rigid_dv)
                np.testing.assert_allclose(rigid, elastic, rtol=2e-5, atol=2e-5)

    def test_force_is_energy_gradient(self):
        fixture = ElementFixture([rotation(0)], "cpu")
        x, d = fixture.rest.copy(), fixture.directors.copy()
        x[:, 0] *= 1.02
        x[2, 2] += 0.005
        d[:, 0] = fixture.rest[:, 0]
        f, _ = fixture.evaluate(x, d)
        q = np.stack((x, d), axis=1)
        expected = np.zeros_like(q)
        eps = 1e-6
        for index in np.ndindex(q.shape):
            plus, minus = q.copy(), q.copy()
            plus[index] += eps
            minus[index] -= eps
            expected[index] = (
                reference_energy(plus[:, 0], plus[:, 1], fixture.rest, fixture.directors)
                - reference_energy(minus[:, 0], minus[:, 1], fixture.rest, fixture.directors)
            ) / (2 * eps)
        np.testing.assert_allclose(f[0], expected, rtol=1e-3, atol=2e-5)

    def test_mass_and_director_inertia(self):
        fixture = ElementFixture([rotation(0)], "cpu")
        mass = fixture.zeros(24)
        wp.launch(
            compute_lumped_mass,
            dim=1,
            inputs=[fixture.x0, fixture.d0, fixture.nodes, fixture.h, fixture.material, mass],
            device="cpu",
        )
        values = mass.numpy().reshape(4, 6)
        node_mass = 700 * 0.2 * 0.2 * 0.01 / 4
        np.testing.assert_allclose(values[:, :3], node_mass, rtol=1e-5)
        np.testing.assert_allclose(values[:, 3:], node_mass * 0.01**2 / 12, rtol=1e-5)

    def test_rotated_reference_has_same_force_and_stiffness(self):
        # Rotate both reference and current configurations, including the material axes.
        # This catches global-coordinate ANS interpolation on curved tire patches.
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                rotations = [rotation(a) for a in np.deg2rad([0, 30, 60, 90])]
                fixture = ElementFixture(rotations, device)
                directors = fixture.directors.copy()
                directors[:, 0] = fixture.rest[:, 0]
                force, stiffness = fixture.evaluate(fixture.rest, directors)
                for i, r in enumerate(rotations[1:], 1):
                    np.testing.assert_allclose(force[i] @ r, force[0], rtol=2e-3, atol=2e-5)
                    transform = np.kron(np.eye(8), r)
                    np.testing.assert_allclose(
                        transform.T @ stiffness[i] @ transform, stiffness[0], rtol=2e-3, atol=0.01
                    )

    def test_rigid_translation_and_rotation_have_no_elastic_force(self):
        fixture = ElementFixture([rotation(0)], "cpu")
        r = rotation(0.7)
        force, _ = fixture.evaluate(fixture.rest @ r.T + [1, 2, 3], fixture.directors @ r.T)
        self.assertLess(np.linalg.norm(force), 0.05)


if __name__ == "__main__":
    unittest.main()
