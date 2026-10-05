# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid tire contact must lie on the same triangles as the chassis and viewer."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.ancf._terrain_common import grid_triangles, triangulated_sample
from newton.solvers import TerrainSCM


class TestANCFTerrainContact(unittest.TestCase):
    def test_rigid_contact_matches_mesh_planes(self):
        # The saddle includes the 9 cm discrepancy found at the RELLIS chassis stall.
        heights = np.array([[1.0, 0.56], [0.56, 0.56]], np.float32)
        fractions = np.array([[0.72, 0.70], [0.30, 0.75], [0.75, 0.25], [0.25, 0.70]])
        triangles = grid_triangles(2, 2).reshape(-1, 3)
        vertices = np.array([[0, 0, 1], [0.25, 0, 0.56], [0, 0.25, 0.56], [0.25, 0.25, 0.56]])
        xy = fractions * 0.25
        expected_height, normals = [], []
        for point, fraction in zip(xy, fractions, strict=True):
            triangle = vertices[triangles[0 if fraction[0] >= fraction[1] else 1]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= np.linalg.norm(normal)
            height = triangle[0, 2] - np.dot(normal[:2], point - triangle[0, :2]) / normal[2]
            expected_height.append(height)
            normals.append(normal)
        normals = np.array(normals)
        clearance = np.array([-0.01, -0.02, 0.01, -0.005])
        expected = 10000 * np.maximum(-clearance * normals[:, 2], 0)[:, None] * normals
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            for origin in ((0.125, 0.125), (-13.125, 86.125)):
                with self.subTest(device=device, origin=origin):
                    grid = SimpleNamespace(origin=origin, hx=0.125, hy=0.125, cell=0.25)
                    world_xy = xy + np.array(origin) - 0.125
                    sampled = [triangulated_sample(grid, heights, *p) for p in world_xy]
                    np.testing.assert_allclose(sampled, expected_height, atol=1e-7)
                    points = np.column_stack((world_xy, np.array(expected_height) + clearance))
                    nodes = wp.array(points[:, [1, 2, 0]], dtype=wp.vec3, device=device)
                    terrain = TerrainSCM(
                        heights,
                        0.125,
                        0.125,
                        nodes,
                        wp.array([[0, 1, 2, 3]], dtype=int, device=device),
                        1,
                        rigid=True,
                        origin=origin,
                        device=device,
                    )
                    force = wp.zeros(24, dtype=float, device=device)
                    tangent = wp.zeros_like(force)
                    velocity = wp.zeros(4, dtype=wp.vec3, device=device)
                    terrain.apply_contact(nodes, velocity, 10000, 0, 0.1, 600, force, tangent)
                    np.testing.assert_allclose(terrain.node_f.numpy(), expected, atol=0.06, rtol=1e-3)
                    np.testing.assert_allclose(
                        force.numpy().reshape(4, 6)[:, :3], expected[:, [1, 2, 0]], atol=0.06, rtol=1e-3
                    )
                    np.testing.assert_array_equal(force.numpy().reshape(4, 6)[:, 3:], 0)
                    self.assertTrue(np.all(tangent.numpy() >= 0))

    def test_soil_retains_bilinear_surface(self):
        # Deformable soil is intentionally a separate surface; do not silently change it.
        heights = np.array([[1.0, 0.56], [0.56, 0.56]], np.float32)
        u, v = 0.7, 0.6
        height = 1.0 - 0.44 * (u + v - u * v)
        normal = np.array([0.44 * (1 - v), 0.44 * (1 - u), 1.0])
        normal /= np.linalg.norm(normal)
        nodes = wp.array([[v, height - 0.01, u]] * 4, dtype=wp.vec3, device="cpu")
        terrain = TerrainSCM(
            heights,
            0.5,
            0.5,
            nodes,
            wp.array([[0, 1, 2, 3]], dtype=int, device="cpu"),
            1,
            rigid=False,
            origin=(0.5, 0.5),
            device="cpu",
        )
        terrain.apply_contact(
            nodes, wp.zeros_like(nodes), 10000, 0, 0.1, 600, wp.zeros(24, device="cpu"), wp.zeros(24, device="cpu")
        )
        np.testing.assert_allclose(terrain.node_f.numpy(), np.tile(100 * normal[2] * normal, (4, 1)), atol=1e-3)


if __name__ == "__main__":
    unittest.main()
