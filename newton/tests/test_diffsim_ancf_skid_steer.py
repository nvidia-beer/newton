# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Response gradients, independent motor commands, and physical learning acceptance."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp

from newton.examples.ancf.diffsim._skid_steer_calibration import SkidResponse
from newton.examples.ancf.diffsim.example_diffsim_ancf_skid_steer import Example, _drive_sides
from newton.tests.ancf_diffsim_checks import check_skid_steer_result


class TestSkidResponse(unittest.TestCase):
    def setUp(self):
        self.response = SkidResponse(0.305, 1.13642)
        self.wheels = np.array([[3, 3], [1.5, 3], [3, 1.5], [0.75, 3.75], [3.75, 0.75]], dtype=float)
        self.truth = np.array([0.84, 0.03, 0.12])
        self.observations = self._synthetic_response(self.wheels)
        self.response.set_samples(self.wheels, self.observations)

    def _synthetic_response(self, wheels):
        inputs = self.response.features(wheels)
        return np.column_stack(
            (self.truth[0] * inputs[:, 0], self.truth[1] * inputs[:, 1] + self.truth[2] * inputs[:, 1] ** 3)
        )

    def test_gradient_matches_finite_difference(self):
        self.response.gains.assign([0.7, 0.2, 0.07])
        value, analytic = self.response.loss_gradient()
        original = self.response.gains.numpy().copy()
        numeric = np.zeros(3)
        epsilon = 1e-3
        for i in range(3):
            delta = np.eye(3)[i] * epsilon
            self.response.gains.assign(original + delta)
            plus = self.response.loss_gradient()[0]
            self.response.gains.assign(original - delta)
            minus = self.response.loss_gradient()[0]
            numeric[i] = (plus - minus) / (2 * epsilon)
        self.assertGreater(value, 0)
        np.testing.assert_allclose(analytic, numeric, atol=2e-5, rtol=1e-3)

    def test_fit_recovers_gains_and_inverts_an_unseen_turn(self):
        for _ in range(16):
            self.response.step()
        np.testing.assert_allclose(self.response.gains.numpy(), self.truth, atol=0.0011)
        self.assertTrue(np.all(np.diff(self.response.history) < 0))
        for target in ([0.6, 0.035], [0.7, -0.025], [0.5, 0.0]):
            commands = self.response.wheel_commands(*target)
            actual = self._synthetic_response(commands[None, :])[0]
            np.testing.assert_allclose(actual, target, atol=0.001)

    def test_rejects_unidentifiable_or_invalid_observations(self):
        for wheels, measured in (
            (np.ones((3, 2)), np.zeros((3, 2))),
            (np.array([[3, 3], [1, 3], [2, 3]]), np.zeros((3, 2))),
            (self.wheels[:2], self.observations[:2]),
            (self.wheels[:3], self.observations[:3]),
            (self.wheels, np.full((3, 2), np.nan)),
            (self.wheels, np.zeros((3, 3))),
        ):
            with self.subTest(wheels=wheels), self.assertRaises(ValueError):
                self.response.set_samples(wheels, measured)
        for radius, track in ((0, 1), (1, -1), (np.nan, 1)):
            with self.assertRaises(ValueError):
                SkidResponse(radius, track)

    def test_drive_maps_sides_and_axis_signs_including_counterrotation(self):
        dofs = wp.array([5, 2, 7, 0], dtype=int, device="cpu")
        signs = wp.array([1, -1, 1, -1], dtype=float, device="cpu")
        for command in ([2, 2], [1, 3], [3, 1], [-2, 2]):
            targets = wp.zeros(8, dtype=float, device="cpu")
            cmd = wp.array(command, dtype=float, device="cpu")
            wp.launch(_drive_sides, 4, inputs=[dofs, signs, cmd, targets], device="cpu")
            values = targets.numpy()
            np.testing.assert_array_equal(values[[5, 2, 7, 0]], np.tile(command, 2) * [1, -1, 1, -1])
            np.testing.assert_array_equal(values[[1, 3, 4, 6]], 0)

    def test_target_edits_reuse_learned_response(self):
        example = Example.__new__(Example)
        example.calibrated = True
        example.response = self.response
        original = self.response.gains.numpy().copy()
        example.set_target(0.5, -0.03)
        self.assertEqual(example._pending, "compare")
        np.testing.assert_array_equal(self.response.gains.numpy(), original)
        example.calibrated = False
        example.set_target(0.7, 0.01)
        self.assertEqual(example._pending, "learn")
        for speed, yaw in ((np.nan, 0), (0.1, 0), (0.5, 0.1)):
            with self.assertRaises(ValueError):
                example.set_target(speed, yaw)

    def test_report_cannot_overwrite_telemetry_preset_or_its_own_csv(self):
        example = Example.__new__(Example)
        example._config_path = Path("/tmp/07_vehicle_telemetry.json")
        for path in (example._config_path, Path("/tmp/observations.csv")):
            with self.assertRaises(ValueError):
                example._validate_report_path(path)
        example._validate_report_path(Path("/tmp/skid_report.json"))

    def test_reference_mesh_follows_target_pose_without_moving_the_vehicle(self):
        example = Example.__new__(Example)
        example.viewer = Mock(spec=["begin_frame", "log_state", "log_lines", "log_mesh", "end_frame"])
        example.model = SimpleNamespace(device="cpu")
        state = SimpleNamespace(body_q=wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu"))
        example.rig = SimpleNamespace(state_0=state)
        example._frame = 10
        example._paths = {}
        example._spin = np.zeros(2)
        example._ghost_pose = np.array([1.0, 2.0, math.pi / 2])
        example._ghost_z = 0.2
        example._ghost_local = wp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=wp.vec3, device="cpu")
        example._ghost_labels = wp.array([-1, -1, -1], dtype=int, device="cpu")
        example._ghost_indices = wp.array([0, 1, 2], dtype=int, device="cpu")
        example._ghost_hubs = wp.zeros(4, dtype=wp.vec3, device="cpu")
        example._ghost_spin = wp.zeros(4, dtype=float, device="cpu")
        example._ghost_points = wp.empty_like(example._ghost_local)
        example.phase = "Before"
        example._phase_frame = 200
        example._settle_frames = 180
        pose_before = state.body_q.numpy().copy()
        example.render()
        np.testing.assert_allclose(example._ghost_points.numpy(), [[1, 3, 0.2], [0, 2, 0.2], [1, 2, 1.2]], atol=1e-6)
        example._ghost_pose[0] += 0.5
        example.render()
        self.assertAlmostEqual(example._ghost_points.numpy()[0, 0], 1.5, places=6)
        np.testing.assert_array_equal(state.body_q.numpy(), pose_before)


@unittest.skipUnless(os.environ.get("ANCF_SKID_CALIBRATION") == "1", "Opt-in full MuJoCo/ANCF calibration")
class TestSkidSteerPhysical(unittest.TestCase):
    def test_calibration_replay_and_opposite_turn(self):
        if not wp.is_cuda_available():
            self.skipTest("MuJoCo/ANCF vehicle requires CUDA")
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "skid.json"
            args = Example.create_parser().parse_args(["--report", str(report)])
            source = args.telemetry_config.read_bytes()
            example = Example(None, args)
            for _ in range(4000):
                example.step()
            check_skid_steer_result(example)
            self.assertEqual(example.phase, "Ready")
            self.assertEqual(args.telemetry_config.read_bytes(), source)
            self.assertEqual(json.loads(report.read_text())["format"], "newton.skid_response/1")
            self.assertEqual(len(report.with_suffix(".csv").read_text().splitlines()), 601)
            original = example.results["Learned"].copy()
            gains = example.response.gains.numpy().copy()
            example.set_target(0.6, math.radians(2.0))
            for _ in range(1081):
                example.step()
            check_skid_steer_result(example)
            for key in original:
                self.assertAlmostEqual(original[key], example.results["Learned"][key], delta=0.005)
            example.set_target(0.6, math.radians(-2.0))
            for _ in range(1081):
                example.step()
            check_skid_steer_result(example)
            self.assertLess(example.results["Learned"]["yaw_rad_s"], 0)
            np.testing.assert_array_equal(example.response.gains.numpy(), gains)


if __name__ == "__main__":
    unittest.main()
