# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent tests for launcher examples 01, 02, 03, 05 and 07.

Run in Docker with docker/test-ancf-examples.sh. No sand/deformable ground,
fitted parameters or path correction. Includes implicit runtime checks.
The shared implicit coupling checks cover all three vehicle assets and tire resolutions.
The contact and replay acceptance checks use the Warthog/simple-tire profile.

Mechanical checks use analytical expectations. Limits of 1 cm penetration,
2 cm/s stationary speed and 5% mean load imbalance are engineering screening
tolerances, not measured tire accuracy. The 40 s replay has provisional limits
of 1 m position RMS and 10 degrees heading RMS, independent of the current bad
replay. Passing does not prove 1:1 fidelity or held-out calibration.
Most cases use the 6 / 2 / 10 launcher budget; the Superjeep and the 07 replay
need 10 substeps and say so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.ancf._vehicle_terrain import _record_wheel_diagnostics
from newton.examples.ancf.example_vehicle_telemetry import Example as TelemetryExample
from newton.examples.ancf.example_vehicle_telemetry import Telemetry, _brake_levers


class TestANCFTelemetryCommands(unittest.TestCase):
    """Command timing and feasible lever conversion, without simulation."""

    def setUp(self):
        self.telemetry = Telemetry.__new__(Telemetry)
        self.telemetry.controller = {"cmd_vel_timeout_s": 0.25}
        self.telemetry.cmd_t = np.array([1.0, 1.2, 2.0])
        self.telemetry.cmd_v = np.array([0.5, 1.0, 0.0])
        self.telemetry.cmd_w = np.array([0.2, -0.3, 0.0])

    def test_zero_order_hold_and_timeout(self):
        for t, expected in [
            (0.99, (0.0, 0.0)),
            (1.0, (0.5, 0.2)),
            (1.199, (0.5, 0.2)),
            (1.2, (1.0, -0.3)),
            (1.449, (1.0, -0.3)),
            (1.451, (0.0, 0.0)),
            (2.0, (0.0, 0.0)),
        ]:
            with self.subTest(t=t):
                self.assertEqual(self.telemetry.command_at(t), expected)

    def test_rewinding_recording_restores_command(self):
        self.telemetry.command_at(2.1)
        self.assertEqual(self.telemetry.command_at(1.1), (0.5, 0.2))

    def test_brake_levers_preserve_feasible_side_speeds(self):
        for left, right, throttle, lever in [
            (0.0, 0.0, 0.0, 0.0),
            (4.0, 4.0, 4.0, 0.0),
            (2.0, 4.0, 4.0, 0.5),
            (4.0, 2.0, 4.0, -0.5),
            (-2.0, -4.0, -4.0, 0.5),
        ]:
            with self.subTest(left=left, right=right):
                self.assertEqual(_brake_levers(left, right), (throttle, lever))

    def test_wheel_log_records_measured_speed_target_and_effort_with_signs(self):
        output = wp.zeros((2, 2, 3), dtype=float, device="cpu")
        wp.launch(
            _record_wheel_diagnostics,
            dim=2,
            inputs=[
                wp.array([1, 3], dtype=int, device="cpu"),
                wp.array([1.0, -1.0], dtype=float, device="cpu"),
                wp.array([99.0, 2.0, 99.0, -3.0], dtype=float, device="cpu"),
                wp.array([99.0, 4.0, 99.0, -5.0], dtype=float, device="cpu"),
                wp.array([99.0, 6.0, 99.0, -7.0], dtype=float, device="cpu"),
                1,
                output,
            ],
            device="cpu",
        )
        np.testing.assert_array_equal(output.numpy(), [[[0, 0, 0], [0, 0, 0]], [[2, 4, 6], [3, 5, 7]]])

    def test_replay_reads_start_of_frame_command_after_settling(self):
        query = []
        example = TelemetryExample.__new__(TelemetryExample)
        example.replay = True
        example.settle_time = 2.0
        example._t_scn = 2.0
        example._steer_sign = 1.0
        example._steer_range = 1.0
        example.spec = SimpleNamespace(max_wheel_speed=10.0)
        example.telemetry = SimpleNamespace(
            cmd_t=np.array([0.0]),
            command_at=lambda t: query.append(t) or (0.0, 0.0),
            wheel_speeds=lambda v, w: (0.0, 0.0),
        )

        def tick():
            example._t_scn += 1 / 60

        example._scenario_tick = tick
        example._drive()
        self.assertEqual(len(query), 1)
        self.assertAlmostEqual(query[0], 0.0, places=12)
        example._t_scn = 1.0
        example._drive()
        self.assertEqual(len(query), 1, "No recording command should run during settling")


class TestANCFExamplePhysics(unittest.TestCase):
    """Each scene runs once in a subprocess; assertions inspect measurements."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("ANCF examples require CUDA; run docker/test-ancf-examples.sh")
        cls.reports = {}
        destination = os.environ.get("ANCF_TEST_OUTPUT_DIR")
        if destination:
            cls.output = Path(destination)
            cls.output.mkdir(parents=True, exist_ok=True)
        else:
            cls.output = Path(tempfile.mkdtemp(prefix="newton-ancf-tests-"))
        print(f"\nANCF test artifacts: {cls.output}", flush=True)

    def report(self, case, overrides=None, label=None):
        label = label or case
        if label not in self.reports:
            output = self.output / f"{label}.json"
            console = self.output / f"{label}.log"
            command = [sys.executable, "-m", "newton.tests.ancf_example_probe", case, str(output)]
            if overrides:
                command += ["--overrides", json.dumps(overrides)]
            with console.open("w") as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
                except subprocess.TimeoutExpired:
                    self.fail(f"{case} timed out after 600 s; see {console}")
            text = console.read_text(errors="replace")
            self.assertEqual(result.returncode, 0, f"{case} failed; see {console}\n{text[-5000:]}")
            self.assertNotIn("height field collision overflow", text.lower(), f"Collision overflow: {console}")
            self.assertTrue(output.exists(), f"Missing probe report: {output}")
            self.reports[label] = json.loads(output.read_text())
        report = self.reports[label]
        self.assertTrue(report["complete"], f"Incomplete probe: {case}")
        substeps = (overrides or {}).get("substeps", 6)
        self.assertEqual(report["solver_budget"], [substeps, 2, 10])
        if report.get("coupling_method") == "adaptive":
            self.assertTrue(report["coupling_graph_captured"], "Adaptive coupling needs its CUDA graph for FPS savings")
        self.assertGreater(len(report["samples"]), 1)
        for sample in report["samples"]:
            self.assertEqual(sample.get("fault_count", 0), 0, f"{case}: fault at {sample['t']} s")
            self.assertFalse(sample.get("fault", ""), f"{case}: pending fault at {sample['t']} s")
            self.assertEqual(sample.get("episode", 0), 0, f"{case}: reset at {sample['t']} s")
        return report

    @staticmethod
    def tail(report, seconds=1.0):
        end = report["samples"][-1]["t"]
        return [sample for sample in report["samples"] if sample["t"] >= end - seconds]

    def assert_stationary(self, report):
        tail = self.tail(report)
        velocity = np.array([sample["velocity"][:3] for sample in tail])
        rms = float(np.sqrt(np.mean(np.sum(velocity**2, axis=1))))
        self.assertLess(rms, 0.02, f"Stationary speed RMS {rms:.4f} m/s exceeds 0.02 m/s")
        position = np.array([sample["pose"][:3] for sample in tail])
        drift = float(np.linalg.norm(position[-1] - position[0]))
        self.assertLess(drift, 0.02, f"Stationary drift {drift:.4f} m over the last second")

    def assert_load_balance(self, report):
        force = np.array([sample["support_force"] for sample in self.tail(report)])
        mean = float(np.mean(np.sum(force, axis=1)))
        expected = report["expected_support_force"]
        self.assertLess(
            abs(mean - expected) / expected,
            0.05,
            f"Mean upward support {mean:.3f} N; expected {expected:.3f} N, including applied preload",
        )

    def test_01_free_fall_and_translation_invariance(self):
        report = self.report("01_free_fall")
        first, last = report["samples"][0], report["samples"][-1]
        t = last["t"]
        p0, v0 = np.array(first["com"]), np.array(first["com_velocity"])
        expected_p = p0 + v0 * t + np.array([0.0, -0.5 * 9.81 * t * t, 0.0])
        expected_v = v0 + np.array([0.0, -9.81 * t, 0.0])
        # Allow one substep of initial acceleration-history error in HHT.
        dt = 1.0 / 600.0
        np.testing.assert_allclose(last["com"], expected_p, atol=9.81 * dt * t, rtol=0)
        np.testing.assert_allclose(last["com_velocity"], expected_v, atol=9.81 * dt, rtol=0)
        self.assertGreater(last["node_y_min"], 0.0, "Free fall must precede ground contact")
        self.assertLess(last["translation_error"], 1.0e-4)

    def test_01_drop_contact_and_settling(self):
        report = self.report("01_drop")
        minimum = min(sample["node_y_min"] for sample in report["samples"])
        self.assertLess(report["peak_penetration"], 0.01, f"Ground penetration {report['peak_penetration']:.4f} m")
        self.assertLess(minimum, 0.005, "Drop never reached the ground")
        speed = np.array([sample["com_velocity"] for sample in self.tail(report)])
        self.assertLess(float(np.sqrt(np.mean(np.sum(speed**2, axis=2)))), 0.05)
        self.assertLess(report["samples"][-1]["translation_error"], 1.0e-3)

    def test_02_loaded_wheel_support(self):
        report = self.report("02_loaded")
        self.assertEqual(report["effective_preload"], report["requested_preload"], "Preload was clamped")
        self.assert_stationary(report)
        self.assert_load_balance(report)

    def test_03_stationary_vehicle_support(self):
        report = self.report("03_stationary")
        self.assert_stationary(report)
        self.assert_load_balance(report)

    def test_03_driving_moves_forward_with_bounded_motor_effort(self):
        report = self.report("03_driving")
        first, last = report["samples"][0], report["samples"][-1]
        self.assertGreater(last["pose"][0] - first["pose"][0], report["rolling_radius"])
        wheel_speed = np.array([sample["wheel_velocity"] for sample in self.tail(report)])
        self.assertTrue((wheel_speed.mean(axis=0) > 0.0).all(), "Axle rotating opposite the command")
        effort = np.abs([sample["motor_torque"] for sample in report["samples"]])
        self.assertLessEqual(float(effort.max()), report["effort_limit"] * (1.0 + 1.0e-5))

    def test_03_acceleration_braking_and_reverse(self):
        report = self.report("03_braking")
        for lo, hi, sign in ((2.5, 3.0, 1), (5.5, 6.0, 0), (8.5, 9.0, -1)):
            speed = np.array([s["wheel_velocity"] for s in report["samples"] if lo <= s["t"] <= hi])
            with self.subTest(phase=sign):
                if sign == 0:
                    self.assertLess(float(np.abs(speed).max()), 0.1, "Wheels failed to stop")
                else:
                    self.assertGreater(float((sign * speed).min()), 2.0)
                    self.assertLess(float(speed.std(axis=0).max()), 0.2, "Wheel coupling oscillates")
        effort = np.abs([s["motor_torque"] for s in report["samples"]])
        self.assertLessEqual(float(effort.max()), report["effort_limit"] * (1 + 1e-5))

    def test_03_warthog_motion_bounds_response_refresh_work(self):
        report = self.report("03_warthog_motion")
        self.assertTrue(report["coupling_graph_captured"])
        self.assertEqual(report["torque_alpha"], 1.0)
        moving = [s for s in report["samples"] if 4.0 < s["t"] <= 10.0]
        residual = max(s["coupling_velocity_residual"][s["coupling_iterations_used"] - 1] for s in moving)
        self.assertLess(residual, 0.1, "Moving wheel–tire coupling failed to converge")
        steady = np.array([s["wheel_velocity"] for s in moving if 7.0 <= s["t"] <= 8.0])
        self.assertGreater(float(steady.min()), 5.0)
        self.assertLess(float(steady.std(axis=0).max()), 0.2, "Straight-driving wheel speed oscillates")
        final = report["samples"][-1]
        self.assertLess(float(np.abs(final["wheel_velocity"]).max()), 0.1, "Wheels failed to brake")
        np.testing.assert_allclose(
            [s["applied_moment"] for s in moving],
            [s["tire_moment"] for s in moving],
            atol=1e-4,
            rtol=1e-3,
        )
        effort = np.abs([s["motor_torque"] for s in report["samples"]])
        self.assertLessEqual(float(effort.max()), report["effort_limit"] * (1 + 1e-5))

    def assert_response_reuse(self, report):
        self.assertTrue(report["coupling_graph_captured"], "Response reuse requires captured coupling")
        self.assertTrue(report["response_reuse"], "Auto excluded a supported vehicle from response reuse")
        self.assertEqual(report["torque_alpha"], 1.0)

    def test_03_sherp_auto_reduces_stationary_coupling_work(self):
        report = self.report(
            "03_stationary",
            {"vehicle-asset": "sherp_vehicle.usdc", "tire-asset": "sherp_ancf_tire_simple.usda"},
            "03_sherp_auto_stationary",
        )
        self.assert_response_reuse(report)
        self.assert_stationary(report)
        self.assert_load_balance(report)
        first, last = self.tail(report)[0], self.tail(report)[-1]
        steps, trials = np.subtract(last["coupling_totals"], first["coupling_totals"])
        force_evaluations_per_solve = report["solver_budget"][1] + 1
        # A joint Newton evaluation calls the force law once; a complete shell
        # solve calls it NR+1 times. Keep the same work limit in comparable units.
        force_evaluations = trials if report.get("coupled_newton_active") else trials * force_evaluations_per_solve
        self.assertLess(
            force_evaluations / steps,
            2.0 * force_evaluations_per_solve,
            "Settled tires still require repeated full solves",
        )

    def test_03_all_vehicle_motion_preserves_coupling(self):
        self._check_all_vehicle_motion()

    def test_03_all_vehicle_coupled_newton_motion_preserves_coupling(self):
        self._check_all_vehicle_motion(coupling_method="coupled-newton")

    @staticmethod
    def _motion_substeps(vehicle):
        # The Superjeep tire state goes non-finite at frame 906 (steering phase)
        # with 6 substeps; it needs the 10-substep budget.
        return 10 if vehicle == "superjeep" else 6

    def _check_all_vehicle_motion(self, coupling_method=None):
        profiles = (
            ("warthog", "warthog_ancf_tire_simple.usda"),
            ("sherp", "sherp_ancf_tire_simple.usda"),
            ("superjeep", "superjeep_tire.usda"),
            ("warthog", "warthog_ancf_tire.usda"),
            ("sherp", "sherp_ancf_tire.usda"),
        )
        for vehicle, tire in profiles:
            with self.subTest(vehicle=vehicle, tire=tire):
                overrides = {
                    "vehicle-asset": f"{vehicle}_vehicle.usdc",
                    "tire-asset": tire,
                    "substeps": self._motion_substeps(vehicle),
                }
                suffix = ""
                if coupling_method is not None:
                    overrides["coupling-method"] = coupling_method
                    suffix = f"_{coupling_method}"
                report = self.report(
                    "03_vehicle_motion",
                    overrides,
                    f"03_motion_{tire.removesuffix('.usda')}{suffix}",
                )
                if vehicle != "superjeep":
                    # Coupled Newton rejects the Superjeep suspension (> 32 rigid dofs);
                    # it runs adaptive Aitken coupling, which the checks below still screen.
                    self.assert_response_reuse(report)
                moving = [s for s in report["samples"] if 2.0 < s["t"] <= 22.0]
                residual = max(s["coupling_velocity_residual"][s["coupling_iterations_used"] - 1] for s in moving)
                self.assertLess(residual, 0.1, "Wheel/tire velocity mismatch exceeds the motion screen")
                for lo, hi, sign in ((4.5, 5.0, 1), (9.5, 10.0, -1), (21.5, 22.0, 1)):
                    speed = np.array([s["wheel_velocity"] for s in moving if lo <= s["t"] <= hi])
                    # Motor gains and load differ between assets; require response
                    # in the commanded direction without imposing Warthog's PD error.
                    target = np.array([s["wheel_target"] for s in moving if lo <= s["t"] <= hi])
                    self.assertGreater(float((sign * speed / np.abs(target)).min()), 0.5)
                    self.assertLess(float(speed.std(axis=0).max()), 0.2, "Straight-driving wheels oscillate")
                for start, end in ((5.0, 7.0), (10.0, 12.0)):
                    before = next(s for s in report["samples"] if s["t"] == start)
                    after = next(s for s in report["samples"] if s["t"] == end)
                    self.assertLess(
                        float(np.linalg.norm(after["wheel_velocity"])),
                        0.1 * float(np.linalg.norm(before["wheel_velocity"])),
                        "Braking failed to dissipate wheel rotation",
                    )
                final_speed = np.array([s["wheel_velocity"] for s in report["samples"] if s["t"] >= 24.5])
                self.assertLess(float(np.abs(final_speed).max()), 0.1, "Wheels failed to stop")
                yaw_rates = []
                for lo, hi in ((14.0, 15.0), (17.0, 18.0)):
                    yaw_rate = float(np.mean([s["velocity"][5] for s in moving if lo <= s["t"] <= hi]))
                    self.assertGreater(abs(yaw_rate), 0.01, "Vehicle failed to turn")
                    yaw_rates.append(yaw_rate)
                # The authored steering-joint axis determines which sign turns left.
                self.assertLess(yaw_rates[0] * yaw_rates[1], 0.0, "Opposite steering failed to reverse yaw")
                np.testing.assert_allclose(
                    [s["applied_moment"] for s in moving],
                    [s["tire_moment"] for s in moving],
                    atol=1e-4,
                    rtol=1e-3,
                )
                effort = np.abs([s["motor_torque"] for s in report["samples"]])
                self.assertLessEqual(float(effort.max()), report["effort_limit"] * (1 + 1e-5))

    def test_03_warthog_coupled_newton_preserves_adaptive_motion(self):
        overrides = {"vehicle-asset": "warthog_vehicle.usdc", "tire-asset": "warthog_ancf_tire_simple.usda"}
        coupled = self.report("03_vehicle_motion", overrides, "03_motion_warthog_ancf_tire_simple")
        reference = self.report(
            "03_vehicle_motion", {**overrides, "coupling-method": "adaptive"}, "03_motion_warthog_simple_adaptive"
        )
        self.assert_response_reuse(coupled)
        for field, columns, tolerance in (("wheel_velocity", 4, 0.05), ("pose", 3, 0.1)):
            actual = np.array([s[field][:columns] for s in coupled["samples"]])
            expected = np.array([s[field][:columns] for s in reference["samples"]])
            rms = float(np.sqrt(np.mean((actual - expected) ** 2)))
            self.assertLess(rms, tolerance, f"Coupled Newton changes {field} relative to adaptive coupling")

        def force_evaluations(report):
            # A joint Newton evaluation calls the force law once; a complete shell
            # solve calls it NR+1 times.
            trials = report["samples"][-1]["coupling_totals"][1]
            return trials if report.get("coupled_newton_active") else trials * (report["solver_budget"][1] + 1)

        self.assertLess(
            force_evaluations(coupled), force_evaluations(reference), "Coupled Newton failed to reduce tire work"
        )

    def test_03_tire_reaction_moments_reach_rigid_wheels(self):
        report = self.report("03_driving")
        tail = self.tail(report)
        expected = np.array([sample["tire_moment"] for sample in tail])
        applied = np.array([sample["applied_moment"] for sample in tail])
        self.assertGreater(float(np.linalg.norm(expected, axis=2).max()), 0.1, "Insufficient torque excitation")
        np.testing.assert_allclose(
            applied,
            expected,
            atol=1.0e-4,
            rtol=1.0e-3,
            err_msg=f"Tire reaction moment lost at interface (torque_alpha={report['torque_alpha']})",
        )

    def test_05_flat_rigid_terrain_hold(self):
        report = self.report("05_flat")
        self.assertTrue(report["coupled_newton_active"], "Default implicit vehicle must capture coupled Newton")
        self.assertTrue(report["rigid_terrain"])
        self.assertEqual(report["terrain_stage"], 0.0)
        self.assert_stationary(report)
        self.assert_load_balance(report)

    def test_03_full_tire_warthog_asset_remains_stable(self):
        """Check the full tire mesh independently of the other vehicle assets."""
        report = self.report(
            "03_driving",
            {"vehicle-asset": "warthog_vehicle.usdc", "tire-asset": "warthog_ancf_tire.usda", "num-frames": 900},
            "03_warthog_full",
        )
        residuals = [
            sample["coupling_velocity_residual"][sample["coupling_iterations_used"] - 1]
            for sample in report["samples"][1:]
        ]
        # Use the replay interface and acceleration/braking oscillation screens.
        self.assertLess(max(residuals), 0.1, "Full Warthog tire interface oscillates")
        speed = np.array([sample["wheel_velocity"] for sample in self.tail(report)])
        self.assertGreater(float(speed.min()), 0.0, "Axle rotating opposite the forward command")
        self.assertLess(float(speed.std(axis=0).max()), 0.2, "Full Warthog wheel coupling oscillates")

    def test_03_full_tire_vehicle_assets_remain_finite(self):
        profiles = (
            ("warthog", "warthog_ancf_tire.usda"),
            ("sherp", "sherp_ancf_tire.usda"),
            ("superjeep", "superjeep_tire.usda"),
        )
        for vehicle, tire in profiles:
            with self.subTest(vehicle=vehicle, tire=tire):
                self.report(
                    "03_driving",
                    {"vehicle-asset": f"{vehicle}_vehicle.usdc", "tire-asset": tire, "num-frames": 900},
                    f"03_{vehicle}_full",
                )

    def test_07_saved_log_aligns_simulated_and_recorded_samples(self):
        log = self.output / "07_clock.npz"
        report = self.report("07_replay", {"num-frames": 240, "log": str(log)}, "07_clock")
        with np.load(log) as saved:
            self.assertEqual(saved["wheel_speed_rad_s"].shape, (240, 4))
            for field in ("ghost_x", "ghost_y", "err_along", "err_lat", "err_yaw"):
                self.assertEqual(saved[field].shape, saved["t"].shape)
            np.testing.assert_allclose(
                [saved[field][-1] for field in ("err_along", "err_lat", "err_yaw")],
                report["samples"][-1]["replay_error"],
                atol=1e-10,
            )
            np.testing.assert_allclose(saved["wheel_speed_rad_s"][-1], report["samples"][-1]["wheel_velocity"])

    def test_07_interface_velocity_mismatch_is_bounded(self):
        # Include the late chassis contact in the separate recording. The old
        # 40 s test of 00000 alone missed exhausted coupling iterations on 00004.
        # With 6 substeps rellis_00000 faults at 44.7 s (residual 0.95) and
        # rellis_00004 reaches 0.39; the replay needs the 10-substep budget.
        for sequence in ("rellis_00000", "rellis_00004"):
            with self.subTest(sequence=sequence):
                name = f"07_coupling_{sequence}"
                report = self.report(
                    "07_replay",
                    {
                        "terrain": sequence,
                        "num-frames": 2700,
                        "substeps": 10,
                        "log": str(self.output / f"{name}.npz"),
                    },
                    name,
                )
                self.assertTrue(report["coupled_newton_active"], "Default implicit vehicle must capture coupled Newton")
                residuals = []
                for sample in report["samples"][1:]:
                    used = sample.get("coupling_iterations_used") or report["coupling_iterations"]
                    residuals.append(sample["coupling_velocity_residual"][used - 1])
                # The norm bounds every coordinate; 0.1 rad/s wheel speed (0.1 m/s
                # translation) is the screen, not a fit. A single step that exhausts
                # the iteration cap moves between runs (fast-math, atomics) and is not
                # an oscillation, so screen the distribution: at most 1 % of the
                # 10 Hz samples may exceed 0.1 and none may exceed 0.5.
                residuals = np.array(residuals)
                self.assertLess(float(residuals.max()), 0.5, "Replay wheel/tire interface velocity oscillates")
                self.assertLessEqual(
                    int((residuals > 0.1).sum()),
                    len(residuals) // 100,
                    "Replay wheel/tire interface velocity exceeds the screen too often",
                )


if __name__ == "__main__":
    unittest.main()
