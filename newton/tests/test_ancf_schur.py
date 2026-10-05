# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent linear and conservation checks for reduced ANCF coupling."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.ancf_shell.schur import (
    ShellSchurResponse,
    _project_rigid_impedance,
    _rigid_motion_map,
)


class TestANCFSchur(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "The shell PCG solver requires CUDA")
    def test_condensation_matches_direct_coupled_system(self):
        rng = np.random.default_rng(28)
        n, k, count = 18, 3, 2
        offsets, columns = np.arange(0, 10, 3), np.tile(np.arange(3), 3)
        solver = ShellSchurResponse(offsets, columns, count, k, "cuda:0", max_iters=40)
        arrays = {}
        free_np = np.ones((count, n), np.float32)
        free_np[:, :6] = 0.0
        mass_np = rng.uniform(0.1, 2.0, (count, n)).astype(np.float32)
        contact_np = rng.uniform(0.0, 3.0, (count, n)).astype(np.float32)
        for name, value in (("free", free_np), ("mass", mass_np), ("contact", contact_np)):
            arrays[name] = wp.array(value.reshape(-1), device="cuda:0")
        arrays["values"] = wp.zeros(count * 9 * 36, dtype=float, device="cuda:0")
        arrays["boundary"] = wp.zeros(count * k * n, dtype=float, device="cuda:0")
        dt, velocity_factor, boundary_scale, force_weight = 0.01, 150.0, 0.007, 0.56
        args = [arrays[key] for key in ("values", "free", "boundary", "mass", "contact")]
        args += [dt, velocity_factor, boundary_scale, force_weight]
        # Compile before capture; subsequent replays use entirely new coefficients.
        solver.solve(*args)
        with wp.ScopedCapture(device="cuda:0") as capture:
            solver.solve(*args)
        for _ in range(2):
            matrices, packed, boundary = [], [], []
            for _env in range(count):
                r = rng.normal(size=(n, n))
                a = (r.T @ r + 20.0 * np.eye(n)).astype(np.float32)
                matrices.append(a)
                packed.extend(a.reshape(3, 6, 3, 6).transpose(0, 2, 1, 3).reshape(-1))
                boundary.append(rng.normal(size=(k, n)).astype(np.float32))
            arrays["values"].assign(np.asarray(packed, np.float32))
            arrays["boundary"].assign(np.asarray(boundary).reshape(-1))
            wp.capture_launch(capture.graph)
            response = solver.displacement.numpy().reshape(count, k, n)
            velocities = solver.velocity.numpy().reshape(count, k, n)
            condensed = solver.impedance.numpy()
            for env in range(count):
                a, b = matrices[env].astype(float), boundary[env].T.astype(float)
                cross = boundary_scale * a[6:, :6] @ b[:6]
                x = np.concatenate((dt * b[:6], -np.linalg.solve(a[6:, 6:], cross)))
                v = np.concatenate((b[:6], velocity_factor * x[6:]))
                mass, contact = mass_np[env], contact_np[env]
                z = b.T @ (mass[:, None] * v + dt * force_weight * contact[:, None] * x)
                np.testing.assert_allclose(response[env].T, x, rtol=3e-5, atol=2e-8)
                np.testing.assert_allclose(velocities[env].T, v, rtol=3e-5, atol=2e-6)
                np.testing.assert_allclose(condensed[env], z, rtol=3e-5, atol=3e-6)
                # Compare a full interior/interface solve, not another condensation implementation.
                rigid_mass = np.eye(k) * 50.0
                lf = b[6:].T * (velocity_factor * mass[6:] + dt * force_weight * contact[6:])
                lb = b[:6].T @ ((mass[:6] + dt * dt * force_weight * contact[:6])[:, None] * b[:6])
                full = np.block([[a[6:, 6:], cross], [lf, rigid_mass + lb]])
                force = rng.normal(size=k)
                expected = np.linalg.solve(full, np.concatenate((np.zeros(n - 6), force)))
                reduced = np.linalg.solve(rigid_mass + condensed[env], force)
                np.testing.assert_allclose(reduced, expected[-k:], rtol=3e-5, atol=1e-8)

    def test_six_spindle_modes_preserve_point_velocity_and_virtual_work(self):
        rng = np.random.default_rng(349)
        count, nodes, k = 2, 5, 10
        positions = rng.normal(size=(count * nodes, 3)).astype(np.float32)
        directors = rng.normal(size=(count * nodes, 3)).astype(np.float32)
        com = rng.normal(size=(count, 3)).astype(np.float32)
        poses = np.zeros((count, 7), np.float32)
        poses[:, :3] = rng.normal(size=(count, 3))
        poses[:, 6] = 1.0
        jacobian = rng.normal(size=(1, count * 6, k)).astype(np.float32)
        local = rng.normal(size=(count, 6, 6))
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=device):
                boundary = wp.zeros(count * nodes * 36, dtype=float, device=device)
                bodies = wp.array(np.arange(count), dtype=int, device=device)
                wp.launch(
                    _rigid_motion_map,
                    dim=(count, nodes, 6),
                    inputs=[
                        bodies,
                        wp.array(poses, dtype=wp.transform, device=device),
                        wp.array(com, dtype=wp.vec3, device=device),
                        wp.array(positions, dtype=wp.vec3, device=device),
                        wp.array(directors, dtype=wp.vec3, device=device),
                        nodes,
                        6,
                        boundary,
                    ],
                    device=device,
                )
                b = boundary.numpy().reshape(count, 6, nodes, 6)
                output = wp.zeros((count, k, k), dtype=wp.float64, device=device)
                wp.launch(
                    _project_rigid_impedance,
                    dim=(count, k, k),
                    inputs=[
                        wp.array(jacobian, device=device),
                        bodies,
                        wp.array(local, dtype=wp.float64, device=device),
                        output,
                    ],
                    device=device,
                )
                speed = rng.normal(size=k)
                for tire in range(count):
                    j = jacobian[0, tire * 6 : tire * 6 + 6].astype(float)
                    twist = j @ speed
                    velocities = np.einsum("cnj,c->nj", b[tire], twist)
                    centres = poses[tire, :3] + com[tire]
                    for node in range(nodes):
                        p = positions[tire * nodes + node][[2, 0, 1]]
                        d = directors[tire * nodes + node][[2, 0, 1]]
                        vp = twist[:3] + np.cross(twist[3:], p - centres)
                        vd = np.cross(twist[3:], d)
                        np.testing.assert_allclose(velocities[node], np.r_[vp[[1, 2, 0]], vd[[1, 2, 0]]], atol=3e-6)
                    condensed = output.numpy()[tire]
                    np.testing.assert_allclose(condensed, j.T @ local[tire] @ j, atol=1e-12)
                    self.assertAlmostEqual(speed @ condensed @ speed, twist @ local[tire] @ twist, places=10)


if __name__ == "__main__":
    unittest.main()
