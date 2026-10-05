# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent block-system and cavity checks for coupled shell corrections."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.ancf_shell.coupled_newton import (
    check,
    reduced_matrix,
    reduced_rhs,
    reduced_update,
    update_acc,
)
from newton._src.solvers.ancf_shell.kernels_coupled import make_inverse, project, rigid_mass
from newton._src.solvers.ancf_shell.kernels_gather import make_cavity_reduction
from newton._src.solvers.ancf_shell.schur import _project_rigid_impedance


@unittest.skipUnless(wp.is_cuda_available(), "Coupled shell corrections require CUDA")
class TestANCFCoupledNewton(unittest.TestCase):
    def test_nonzero_shell_residual_matches_full_block_solve(self):
        rng = np.random.default_rng(830)
        count, n, nv = 2, 18, 4
        dt, cv, weight = 0.002, 900.0, 0.56
        mass = rng.uniform(0.5, 2.0, (count, n)).astype(np.float32)
        contact = rng.uniform(0, 10, (count, n)).astype(np.float32)
        boundary = rng.normal(size=(count, 7, n)).astype(np.float32)
        boundary[:, 6] = 0
        jacobian = rng.normal(size=(1, count * 6, nv)).astype(np.float32)
        rigid = np.diag([20.0, 30.0, 40.0, 50.0]).astype(np.float32)
        arm = rng.uniform(0.1, 1, nv).astype(np.float32)
        guess = rng.normal(size=nv).astype(np.float32)
        raw = (guess + rng.normal(scale=0.1, size=nv)).astype(np.float32)
        displacement = np.zeros((count, 7, n), np.float32)
        velocity = np.zeros_like(displacement)
        # Assemble the independent full system in free-shell and rigid coordinates.
        full = np.zeros((count * 12 + nv, count * 12 + nv))
        rhs = np.zeros(full.shape[0])
        full[-nv:, -nv:] = rigid + np.diag(arm)
        rhs[-nv:] = (rigid + np.diag(arm)) @ (raw - guess)
        for env in range(count):
            seed = rng.normal(size=(n, n))
            matrix = seed.T @ seed + 30 * np.eye(n)
            b = boundary[env, :6].T.astype(float)
            j = jacobian[0, env * 6 : (env + 1) * 6].astype(float)
            cross = dt * matrix[6:, :6] @ b[:6] @ j
            residual = rng.normal(size=12)
            force_map = j.T @ b[6:].T * (cv * mass[env, 6:] + dt * weight * contact[env, 6:])
            boundary_mass = j.T @ b[:6].T @ np.diag(mass[env, :6] + dt * dt * weight * contact[env, :6]) @ b[:6] @ j
            rows = slice(env * 12, (env + 1) * 12)
            full[rows, rows] = matrix[6:, 6:]
            full[rows, -nv:] = cross
            full[-nv:, rows] = force_map
            full[-nv:, -nv:] += boundary_mass
            rhs[rows] = -residual
            displacement[env, :6, :6] = (dt * b[:6]).T
            displacement[env, :6, 6:] = -np.linalg.solve(matrix[6:, 6:], dt * matrix[6:, :6] @ b[:6]).T
            displacement[env, 6, 6:] = np.linalg.solve(matrix[6:, 6:], -residual)
            velocity[env] = cv * displacement[env]
            velocity[env, :6, :6] = b[:6].T
        expected = np.linalg.solve(full, rhs)
        dev = "cuda:0"

        def array(a, dtype=float):
            return wp.array(a, dtype=dtype, device=dev)

        b = array(boundary.reshape(count * 7, n))
        x = array(displacement.reshape(count * 7, n))
        v = array(velocity.reshape(count * 7, n))
        local = wp.zeros((count, 7, 7), dtype=wp.float64, device=dev)
        z = wp.zeros((count, nv, nv), dtype=wp.float64, device=dev)
        j = array(jacobian)
        bodies = array(np.arange(count), int)
        m = array(rigid[None])
        armature = array(arm)
        wp.launch_tiled(
            project,
            dim=count * 49,
            block_dim=128,
            inputs=[
                b,
                x,
                v,
                array(mass),
                array(contact),
                7,
                1,
                dt * weight,
                wp.ones(1, dtype=int, device=dev),
                local.flatten(),
            ],
            device=dev,
        )
        wp.launch(_project_rigid_impedance, dim=z.shape, inputs=[j, bodies, local, z], device=dev)
        reduced = wp.zeros((nv, nv), dtype=wp.float64, device=dev)
        inverse = wp.zeros_like(reduced)
        flags = [wp.zeros(1, dtype=int, device=dev) for _ in range(4)]
        wp.launch(reduced_matrix, dim=reduced.shape, inputs=[m, armature, z, reduced], device=dev)
        wp.launch(make_inverse(nv), dim=(1, 32), block_dim=32, inputs=[reduced, inverse, *flags], device=dev)
        right = wp.zeros(nv, dtype=wp.float64, device=dev)
        delta = wp.zeros(nv, dtype=float, device=dev)
        wp.launch(
            reduced_rhs, dim=nv, inputs=[m, armature, local, j, bodies, array(raw), array(guess), right], device=dev
        )
        wp.launch(reduced_update, dim=nv, inputs=[inverse, right, delta], device=dev)
        acceleration = wp.zeros(count * 3, dtype=wp.vec3, device=dev)
        director_acceleration = wp.zeros_like(acceleration)
        free = np.ones((count, n), np.float32)
        free[:, :6] = 0
        wp.launch(
            update_acc,
            dim=(count, 3),
            inputs=[x.flatten(), j, bodies, delta, array(free.ravel()), acceleration, director_acceleration, 3, 1.0],
            device=dev,
        )
        np.testing.assert_allclose(delta.numpy(), expected[-nv:], rtol=2e-5, atol=2e-7)
        result = np.concatenate(
            (acceleration.numpy().reshape(count, 3, 3), director_acceleration.numpy().reshape(count, 3, 3)), axis=2
        ).reshape(count, n)
        np.testing.assert_allclose(result[:, 6:].ravel(), expected[:-nv], rtol=2e-5, atol=2e-7)
        np.testing.assert_array_equal(result[:, :6], 0)

    def test_projection_reuses_tangent_and_updates_free_residual_in_graph(self):
        rng = np.random.default_rng(872)
        count, k, n = 2, 7, 257
        dev = "cuda:0"
        shape = (count * k, n)
        b, x, v = [wp.zeros(shape, dtype=float, device=dev) for _ in range(3)]
        mass = rng.uniform(0.5, 2.0, (count, n)).astype(np.float32)
        contact = rng.uniform(0, 10, (count, n)).astype(np.float32)
        m = wp.array(mass, device=dev)
        c = wp.array(contact, device=dev)
        refresh = wp.ones(1, dtype=int, device=dev)
        local = wp.zeros(count * k * k, dtype=wp.float64, device=dev)
        inputs = [b, x, v, m, c, k, (n + 127) // 128, 0.02, refresh, local]
        wp.launch_tiled(project, dim=count * k * k, block_dim=128, inputs=inputs, device=dev)
        with wp.ScopedCapture(device=dev) as capture:
            wp.launch_tiled(project, dim=count * k * k, block_dim=128, inputs=inputs, device=dev)
        previous = None
        for rebuild in (1, 0, 0, 1):
            boundary, displacement, velocity = [rng.normal(size=(count, k, n)).astype(np.float32) for _ in range(3)]
            b.assign(boundary.reshape(shape))
            x.assign(displacement.reshape(shape))
            v.assign(velocity.reshape(shape))
            refresh.fill_(rebuild)
            wp.capture_launch(capture.graph)
            actual = local.numpy().reshape(count, k, k)
            force = mass[:, None] * velocity + 0.02 * contact[:, None] * displacement
            expected = np.einsum("eab,ecb->eac", boundary.astype(np.float64), force.astype(np.float64))
            if rebuild:
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)
            else:
                np.testing.assert_array_equal(actual[:, :, :-1], previous[:, :, :-1])
                np.testing.assert_allclose(actual[:, :, -1], expected[:, :, -1], rtol=3e-5, atol=1e-5)
            previous = actual

    def test_parallel_rigid_mass_matches_kinetic_energy(self):
        rng = np.random.default_rng(42)
        nbody, ndof = 4, 10
        j = rng.normal(size=(nbody, 6, ndof)).astype(np.float32)
        seed = rng.normal(size=(nbody, 6, 6)).astype(np.float32)
        inertia = seed @ np.transpose(seed, (0, 2, 1)) + np.eye(6, dtype=np.float32)
        children = np.array([2, 0, 3, 1], np.int32)
        expected = sum(j[b].astype(float).T @ inertia[children[b]] @ j[b] for b in range(nbody))
        mass = wp.zeros((1, ndof, ndof), dtype=float, device="cuda:0")
        wp.launch(
            rigid_mass,
            dim=mass.shape,
            inputs=[
                wp.array([0], dtype=int, device="cuda:0"),
                wp.array([nbody], dtype=int, device="cuda:0"),
                wp.array(children, dtype=int, device="cuda:0"),
                wp.array(inertia, dtype=wp.spatial_matrix, device="cuda:0"),
                wp.array(j.reshape(1, nbody * 6, ndof), dtype=float, device="cuda:0"),
                mass,
            ],
            device="cuda:0",
        )
        actual = mass.numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
        velocity = rng.normal(size=ndof)
        twists = j @ velocity
        kinetic_energy = sum(twists[b] @ inertia[children[b]] @ twists[b] / 2 for b in range(nbody))
        self.assertAlmostEqual(float(velocity @ actual @ velocity / 2), kinetic_energy, delta=1e-4 * kinetic_energy)

    def test_parallel_interface_convergence_and_reserve_budget(self):
        # Include inactive lanes: only ten of the 32 CUDA threads own a DOF.
        n, dev = 10, "cuda:0"
        guess = np.linspace(-2, 2, n).astype(np.float32)
        inverse = np.eye(n) + 0.01 * np.ones((n, n))
        for iteration, error, converged, active in (
            (0, 0.0001, 0, 1),
            (2, 0.0001, 1, 0),
            (4, 0.004, 0, 1),
            (5, 0.004, 0, 0),
            (5, 0.04, 0, 1),
            (8, 0.04, 0, 1),
            (9, 0.004, 0, 0),
            (11, 0.04, 0, 0),
        ):
            with self.subTest(iteration=iteration, error=error):
                raw = guess.copy()
                raw[0] += error
                residual = raw - guess
                expected_norm = np.linalg.norm(residual.astype(np.float64))
                corrected = wp.zeros(n, dtype=float, device=dev)
                norms = wp.array([0.2] + [0.0] * 11, dtype=float, device=dev)
                history = wp.zeros(n, dtype=float, device=dev)
                flags = [wp.zeros(1, dtype=int, device=dev) for _ in range(4)]
                wp.launch(
                    check,
                    dim=(1, 32),
                    block_dim=32,
                    inputs=[
                        wp.array(guess, device=dev),
                        wp.array(raw, device=dev),
                        wp.array(inverse, dtype=wp.float64, device=dev),
                        corrected,
                        norms,
                        history,
                        *flags,
                        iteration,
                        11,
                        5,
                        0.001,
                        0.001,
                    ],
                    device=dev,
                )
                np.testing.assert_allclose(corrected.numpy(), guess + inverse @ residual, rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(history.numpy(), residual)
                self.assertAlmostEqual(float(norms.numpy()[iteration]), expected_norm, places=7)
                self.assertEqual(int(flags[0].numpy()[0]), active)
                self.assertEqual(int(flags[1].numpy()[0]), converged)
                self.assertEqual(int(flags[2].numpy()[0]), iteration + 1)
                self.assertEqual(int(flags[3].numpy()[0]), int(iteration >= 2))

    def test_cavity_volume_analytic_translated_boxes(self):
        # Outward quadrilateral faces; analytical volumes provide an independent oracle.
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32
        )
        faces = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], np.int32)
        shapes = np.array([[1, 2, 3], [0.5, 0.25, 2], [2, 3, 4]], np.float32)
        offsets = np.array([[0, 0, 0], [100, -50, 20], [-500, 300, 250]], np.float32)
        points = vertices[None] * shapes[:, None] + offsets[:, None]
        centre = wp.zeros(3, dtype=wp.vec3, device="cuda:0")
        volume = wp.zeros(3, dtype=float, device="cuda:0")
        args = [
            wp.array(points.reshape(-1, 3), dtype=wp.vec3, device="cuda:0"),
            wp.array(faces, dtype=int, device="cuda:0"),
            wp.ones(6, dtype=float, device="cuda:0"),
            centre,
            volume,
        ]
        wp.launch(make_cavity_reduction(8, 6, 128), dim=(3, 128), block_dim=128, inputs=args, device="cuda:0")
        np.testing.assert_allclose(volume.numpy(), np.prod(shapes, axis=1), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(centre.numpy(), shapes / 2, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
