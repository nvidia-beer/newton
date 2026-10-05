# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Forward UI checks and coupled pressure-learning acceptance tests."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp

import newton
from newton.examples.ancf.diffsim._tire_calibration import read_measurements
from newton.examples.ancf.diffsim.example_diffsim_ancf_tire_lift import Example
from newton.solvers import ANCFTireEquilibrium
from newton.tests.ancf_diffsim_checks import check_tire_lift_forward, check_tire_lift_result


class TestDiffsimANCFTireLift(unittest.TestCase):
    def test_height_markers_move_while_paused_without_changing_physics(self):
        example = Example.__new__(Example)
        example.viewer = Mock(spec=["log_mesh"])
        pose = wp.array([wp.transform((0.0, 0.0, 0.263), wp.quat_identity())], dtype=wp.transform, device="cpu")
        example.rig = SimpleNamespace(_frame=240, _spindle_newton_idx=0, state_0=SimpleNamespace(body_q=pose))
        marker = newton.Mesh.create_cylinder(0.014, 0.38, up_axis=newton.Axis.Y, compute_inertia=False)
        example._marker_local_points = wp.array(marker.vertices, dtype=wp.vec3, device="cpu")
        example._marker_indices = wp.array(marker.indices, dtype=wp.int32, device="cpu")
        example._actual_marker_points = wp.empty_like(example._marker_local_points)
        example._target_marker_points = wp.empty_like(example._marker_local_points)
        example.height = 0.263
        example._height_bounds = (0.25, 0.28)
        original_pose = pose.numpy().copy()
        for target in (0.275, 0.251, 0.263):
            example.set_target_height(target)
            example._render_height_markers()
            for points, height in (
                (example._actual_marker_points, example.height),
                (example._target_marker_points, target),
            ):
                z = points.numpy()[:, 2]
                self.assertAlmostEqual((z.min() + z.max()) / 2, height, places=6)
            np.testing.assert_array_equal(pose.numpy(), original_pose)
        # Once the axle moves, the target ghost remains at the requested height.
        pose.assign([[0.0, 0.0, 0.255, 0.0, 0.0, 0.0, 1.0]])
        example._render_height_markers()
        self.assertAlmostEqual(example._actual_marker_points.numpy()[:, 2].mean(), 0.255, places=6)
        self.assertAlmostEqual(example._target_marker_points.numpy()[:, 2].mean(), 0.263, places=6)

    def test_target_edit_starts_learning_from_stopped_or_manual_mode(self):
        example = Example.__new__(Example)
        example.rig = SimpleNamespace(_frame=240)
        example.height = 0.263
        example._height_bounds = (0.261, 0.266)
        example.learning = False
        example.training_status = "Stopped: Target is outside the pressure bounds on this equilibrium branch"
        example.set_target_height(0.263)
        self.assertTrue(example.learning)
        self.assertEqual(example._next_update_frame, 241)
        self.assertEqual(example.training_status, "Target changed")
        example.learning = False
        example.training_status = "Manual"
        example.set_target_height(0.264)
        self.assertTrue(example.learning)
        self.assertEqual(example.training_status, "Target changed")

    def test_rejects_bad_interior_pressure_before_publishing_height_limits(self):
        for bad_contact in (True, False):
            with self.subTest(bad_contact=bad_contact):
                example = Example.__new__(Example)
                example.rig = SimpleNamespace(_pressure_currents=[2.0])
                example._equilibrium_initial = np.zeros(1)
                example._measurements = []
                example._height_bounds = example._reference_profiles = example._range_samples = None
                example._pressure_bounds = (1.0, 8.0)
                example._automatic_target = False
                example._ground_index = 0
                example.target_height = 0.26
                example._tread_profile = Mock(return_value=np.zeros((1, 2)))
                eq = Mock()
                eq.ground = 0.0

                def solve(pressure, _coordinates, bad_contact=bad_contact):
                    return SimpleNamespace(
                        pressure=pressure,
                        coordinates=np.array([pressure]),
                        height=0.24 if 1 < pressure < 2 and not bad_contact else 0.25 + 0.001 * pressure,
                        dh_dp=1e-6,
                    )

                eq.solve.side_effect = solve
                eq.positions.side_effect = lambda coordinates, bad_contact=bad_contact: np.array(
                    [[0.0, -0.03 if 1 < coordinates[0] < 2 and bad_contact else -0.01, 0.0]]
                )
                example._equilibrium = eq
                with self.assertRaisesRegex(RuntimeError, "contact limits|monotonic"):
                    example._prepare_equilibrium()
                self.assertIsNone(example._height_bounds)
                self.assertIsNone(example._reference_profiles)
                self.assertIsNone(example._range_samples)

    def test_range_checks_continue_from_high_pressure_through_contact_changes(self):
        example = Example.__new__(Example)
        example.rig = SimpleNamespace(_pressure_currents=[8.0])
        example._equilibrium_initial = np.array([8.0])
        example._measurements = []
        example._height_bounds = example._reference_profiles = example._range_samples = None
        example._pressure_bounds = (0.25, 8.0)
        example._automatic_target = False
        example._ground_index = 0
        example.target_height = 0.26
        example._tread_profile = Mock(return_value=np.zeros((1, 2)))
        eq = Mock(ground=0.0)

        def solve(pressure, coordinates):
            if max(pressure / coordinates[0], coordinates[0] / pressure) > 1.5:
                raise RuntimeError("Pressure jump crosses too many contact changes")
            return SimpleNamespace(
                pressure=pressure, coordinates=np.array([pressure]), height=0.25 + 0.001 * pressure, dh_dp=1e-3
            )

        eq.solve.side_effect = solve
        eq.positions.return_value = np.array([[0.0, -0.01, 0.0]])
        example._equilibrium = eq
        example._prepare_equilibrium()
        np.testing.assert_allclose(example._height_bounds, (0.25025, 0.258))
        self.assertEqual(len(example._range_samples), 11)
        self.assertTrue(np.all(np.diff([r["height_m"] for r in example._range_samples]) > 0))

    def test_measurement_schema_and_training_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.csv"
            header = "nominal_pressure_pa,additional_load_n,axle_height_m,height_std_m,split\n"
            rows = "14000,200,0.28,0.0002,train\n20000,350,0.27,0.0002,train\n25000,450,0.26,0.0002,validation\n"
            path.write_text(header + rows)
            measurements = read_measurements(path)
            self.assertEqual(len(measurements), 3)
            for bad in (
                rows.replace("0.0002", "0"),
                rows.replace("14000", "nan"),
                rows.replace("200,", "-200,"),
                rows.replace("train", "test"),
                rows.replace("20000,350", "14000,200"),
            ):
                path.write_text(header + bad)
                with self.assertRaises(ValueError):
                    read_measurements(path)
            path.write_text("pressure_psi,height\n2,0.28\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_measurements(path)

    def test_gradient_requires_converged_equilibrium(self):
        example = Example.__new__(Example)
        example._equilibrium_result = None
        with self.assertRaisesRegex(RuntimeError, "equilibrium"):
            example.pressure_gradient()

    def test_pressure_request_preserves_reference_and_applied_pressure(self):
        example = Example.__new__(Example)
        example.rig = SimpleNamespace(
            _pressure_targets=[30_000.0], _pressure_currents=[30_000.0], _build_pressures=[30_000.0]
        )
        example.set_pressure(40_000.0)
        self.assertEqual(example.rig._pressure_targets, [40_000.0])
        self.assertEqual(example.rig._build_pressures, [30_000.0])
        self.assertEqual(example.pressure_pa, 30_000.0)  # The forward rig applies its pressure ramp during step().
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(pressure=invalid), self.assertRaises(ValueError):
                example.set_pressure(invalid)

    def test_invalid_experiment_inputs(self):
        parser = Example.create_parser()
        for flag, value in (
            ("--target-height", "0"),
            ("--pressure-psi", "-1"),
            ("--build-pressure-psi", "nan"),
            ("--load", "-1"),
            ("--load", "inf"),
        ):
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag, value])


@unittest.skipUnless(wp.is_cuda_available(), "The physical wheel rig requires CUDA")
class TestTireLiftTraining(unittest.TestCase):
    def test_wider_demo_range_endpoints_and_learning_on_both_grounds(self):
        for ground in ("telemetry", "firm"):
            with self.subTest(ground=ground):
                example = Example(None, Example.create_parser().parse_args(["--ground", ground]))
                self.assertEqual(example._pressure_range, "demo")
                self.assertEqual(example._rig_args.pcg_iters, 20)
                np.testing.assert_allclose(np.array(example._pressure_bounds) / 6894.757293168, [0.25, 8])
                self.assertAlmostEqual(example.pressure_pa / 6894.757293168, 0.25)
                for _ in range(960):
                    example.step()
                check_tire_lift_result(example)
                self.assertGreater(example._height_bounds[1] - example._height_bounds[0], 0.017)
                self.assertEqual(len(example._range_samples), 11)
                self.assertTrue(np.all(np.diff([r["height_m"] for r in example._range_samples]) > 0))
                self.assertTrue(all(r["dh_dp"] > 0 and r["penetration_m"] < 0.025 for r in example._range_samples))
                self.assertGreater(example.pressure_pa / 6894.757293168, 6.0)
                # Check the low-pressure contact transition as well as both endpoints.
                shapes = []
                for sample in (example._range_samples[0], example._range_samples[2], example._range_samples[-1]):
                    pressure, height = sample["pressure_pa"], sample["height_m"]
                    example._manual_pressure(pressure)
                    # The softer tire takes longer to settle after deflation.
                    for _ in range(600):
                        example.step()
                    check_tire_lift_forward(example)
                    self.assertLess(abs(example.height - height), 2.5e-4)
                    shapes.append(example._tread_profile(example._observed_nodes, example.height))
                self.assertGreater(shapes[0][:, 1].min() - shapes[-1][:, 1].min(), 0.015)
                example.set_target_height(example._height_bounds[0])
                for _ in range(1440):
                    example.step()
                check_tire_lift_result(example)
                self.assertLess(example.pressure_pa / 6894.757293168, 0.26)
                self.assertLess(example.max_penetration, 0.025)
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaisesRegex(RuntimeError, "telemetry"):
                        example.export_config(Path(folder) / "demo.json")

    def test_learning_reaches_target_and_retargets_downward(self):
        example = Example(None, Example.create_parser().parse_args(["--pressure-range", "telemetry"]))
        for _ in range(960):
            example.step()
        check_tire_lift_forward(example)
        self.assertEqual(example.training_status, "Converged")
        self.assertGreaterEqual(example.training_iterations, 6)  # A visible learning phase, not one large jump.
        self.assertGreater(example.pressure_pa - example._run_start_pressure_pa, 1.5 * 6894.757)
        self.assertGreater(len(example._pressure_history), 20)
        self.assertTrue(np.all(np.diff(example._pressure_history) >= 0.0))
        self.assertLessEqual(np.max(np.diff(example._pressure_history)), 0.201)
        history = list(example._pressure_history)
        for _ in range(30):
            example.step()
        self.assertEqual(list(example._pressure_history), history)  # Keep the completed demonstration visible.
        self.assertLess(abs(example.height_error), 2.5e-4)
        self.assertLess(example._equilibrium_result.residual_norm, 1e-7)
        high_pressure = example.pressure_pa
        target = example.target_height
        example._restart_learning()
        self.assertEqual(example.target_height, target)
        self.assertEqual(example.rig._pressure_targets[0], example._initial_pressure_pa)
        self.assertEqual(example.training_iterations, 0)
        self.assertFalse(example._error_history)
        for _ in range(720):
            example.step()
        check_tire_lift_forward(example)
        self.assertEqual(example.training_status, "Converged")
        self.assertGreater(example.training_iterations, 0)
        self.assertLess(abs(example.height_error), 2.5e-4)
        self.assertLess(abs(example.height_error), abs(example._initial_height_error) / 5)
        self.assertTrue(example._error_history)
        high_pressure = example.pressure_pa
        self.assertLess(example._height_bounds[0], example.target_height)
        self.assertGreater(example._height_bounds[1], example.target_height)
        example.set_target_height(example._height_bounds[1] + 0.01)
        for _ in range(10):
            example.step()
        self.assertTrue(example.learning)
        self.assertEqual(example.training_status, "Target outside reachable height range")
        self.assertEqual(example.pressure_pa, high_pressure)
        lower_pressure = (example._pressure_bounds[0] + high_pressure) / 2
        target = example._equilibrium.solve(lower_pressure, example._equilibrium_result.coordinates).height
        example.set_target_height(target)
        self.assertFalse(example._target_outside_range)
        for _ in range(720):
            example.step()
        check_tire_lift_forward(example)
        self.assertEqual(example.training_status, "Converged")
        self.assertLess(abs(example.height_error), 2.5e-4)
        self.assertLess(example.pressure_pa, high_pressure)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prepared.json"
            example.export_config(path)
            config = json.loads(path.read_text())
            report = json.loads(path.with_suffix(".report.json").read_text())
            self.assertTrue(config["args"]["replay"])
            self.assertEqual(config["args"]["build-pressure"], 0.0)
            self.assertEqual(config["args"]["shell-tires"][0]["pressure"], example.pressure_pa)
            self.assertFalse(report["vehicle_calibration_validated"])
            self.assertEqual(report["data_source"], "demonstration")

    def test_shared_physics_and_zero_build_pressure_ramp(self):
        example = Example(None, Example.create_parser().parse_args(["--no-train", "--pressure-range", "telemetry"]))
        self.assertEqual(example.rig._build_pressures, [0.0])
        self.assertAlmostEqual(example.rig._f_load, 50.0 * 9.81)
        self.assertEqual(example.rig.ancf_solver.kn, example._rig_args.kn)
        pressure = sum(example._pressure_bounds) / 2
        example.set_pressure(pressure)
        for _ in range(10):
            example.step()
        self.assertAlmostEqual(example.pressure_pa, pressure)
        self.assertEqual(example.rig._build_pressures, [0.0])
        with self.assertRaisesRegex(ValueError, "limits"):
            example.set_pressure(example._pressure_bounds[1] + 100)
        # Starting the demo from --no-train must still generate a reachable target.
        example._restart_learning()
        self.assertTrue(example._automatic_target)
        for _ in range(720):
            example.step()
        self.assertEqual(example.training_status, "Converged")
        self.assertLess(abs(example.height_error), 2.5e-4)

    def test_ground_change_relearns_same_height_with_different_pressure(self):
        example = Example(None, Example.create_parser().parse_args(["--pressure-range", "telemetry"]))
        for _ in range(960):
            example.step()
        self.assertEqual(example.training_status, "Converged")
        initial_pressure, target = example.pressure_pa, example.target_height
        low, high = example._reference_profiles
        self.assertGreater(low[:, 1].min() - high[:, 1].min(), 0.005)
        original_low = low.copy()
        original_kn = example.rig.ancf_solver.kn
        source_nodes = example.rig.ancf_solver.node_x.numpy().copy()
        coordinates = example._equilibrium_result.coordinates.copy()
        reconstructed = example._equilibrium.positions(coordinates)
        np.testing.assert_allclose(
            reconstructed[example.rig._bead_idx_np, 1],
            example._equilibrium.rest[example.rig._bead_idx_np, 0, 1] + example._equilibrium_result.height,
        )
        np.testing.assert_array_equal(example.rig.ancf_solver.node_x.numpy(), source_nodes)
        np.testing.assert_array_equal(coordinates, example._equilibrium_result.coordinates)
        example._select_ground(1)
        self.assertEqual(example.rig.ancf_solver.kn, 2 * original_kn)
        self.assertEqual(example.target_height, target)
        self.assertIsNone(example._reference_profiles)
        for _ in range(960):
            example.step()
        check_tire_lift_forward(example)
        self.assertEqual(example.training_status, "Converged")
        self.assertEqual(example._equilibrium.kn, example.rig.ancf_solver.kn)
        self.assertLess(abs(example.height_error), 2.5e-4)
        self.assertGreater(initial_pressure - example.pressure_pa, 0.5 * 6894.757)
        self.assertFalse(np.allclose(example._reference_profiles[0], original_low))
        # Endpoint previews expose real deformation and cancel learning.
        example._manual_pressure(example._pressure_bounds[0])
        for _ in range(180):
            example.step()
        low_shape = example._tread_profile(example._observed_nodes, example.height)
        example._manual_pressure(example._pressure_bounds[1])
        for _ in range(180):
            example.step()
        high_shape = example._tread_profile(example._observed_nodes, example.height)
        self.assertFalse(example.learning)
        self.assertGreater(low_shape[:, 1].min() - high_shape[:, 1].min(), 0.005)
        np.testing.assert_allclose(low_shape, example._reference_profiles[0], atol=3e-4)
        np.testing.assert_allclose(high_shape, example._reference_profiles[1], atol=3e-4)
        with tempfile.TemporaryDirectory() as folder:
            # A manual inspection is not a converged fit eligible for export.
            with self.assertRaises(RuntimeError):
                example.export_config(Path(folder) / "manual.json")

    def test_csv_fit_exports_stiffness_and_rebuilds_the_preview(self):
        source = Example(None, Example.create_parser().parse_args(["--no-train", "--pressure-range", "telemetry"]))
        for _ in range(120):
            source.step()
        rig = source.rig
        eq = ANCFTireEquilibrium(rig.ancf_solver, rig._bead_idx_np, rig._m_rigid, rig._f_load, 0.0)
        initial = eq.coordinates(rig.ancf_solver.node_x.numpy(), rig.ancf_solver.node_D.numpy(), source.height)
        expected_scale = 1.25
        with tempfile.TemporaryDirectory() as folder:
            csv_path, export = Path(folder) / "synthetic.csv", Path(folder) / "replay.json"
            rows = ["nominal_pressure_pa,additional_load_n,axle_height_m,height_std_m,split"]
            for i, load in enumerate((200.0, 350.0, 450.0)):
                pressure = source.pressure_pa + i * 3000
                eq.rim_load = eq.spindle_weight + load
                result = eq.solve(pressure, initial, stiffness_scale=expected_scale)
                rows.append(f"{pressure},{load},{result.height},0.0002,{'validation' if i == 2 else 'train'}")
            csv_path.write_text("\n".join(rows) + "\n")
            args = Example.create_parser().parse_args(
                ["--calibration-csv", str(csv_path), "--data-source", "synthetic", "--export-config", str(export)]
            )
            fitted = Example(None, args)
            for _ in range(600):
                fitted.step()
            self.assertEqual(fitted._pressure_range, "telemetry")
            self.assertTrue(fitted._fit_done, fitted.training_status)
            self.assertAlmostEqual(fitted._material["E"] / source._material["E"], expected_scale, delta=1e-4)
            self.assertLess(abs(fitted.height_error), 2.5e-4)
            self.assertLess(fitted.max_penetration, 0.025)
            config = json.loads(export.read_text())
            report = json.loads(export.with_suffix(".report.json").read_text())
            self.assertEqual(config["args"]["shell-tires"][0]["E"], fitted._material["E"])
            self.assertEqual(config["args"]["shell-tires"][0]["pressure"], source.pressure_pa)
            self.assertEqual(report["data_source"], "synthetic")
            self.assertLess(report["static_fit"]["height_rms_m"]["validation"], 1e-7)


if __name__ == "__main__":
    unittest.main()
