# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Learn a skid-steer velocity response before moving to recorded telemetry.

Uses example 03's MuJoCo vehicle and four physical ANCF tires, with example 07's
assets/material/contact/coupling settings on flat ground. Independent left/right
wheel motors produce steering through tire forces; the chassis is never posed
by the controller. Positive yaw turns left, and opposite wheel speeds are allowed.

The demonstration first tries a target speed and yaw rate with ideal no-slip
kinematics. It then measures straight, gentle and stronger left/right maneuvers,
differentiates a steady velocity response with Warp, and retries the target.
Gray is the requested trajectory/ghost, orange the initial drive, and purple the
drive using the fitted response. Change the target or Replay to test the fit
again; Learn again repeats the measurements. This is a synthetic calibration
exercise: gradients pass through the response model, not MuJoCo or ANCF time
integration. The fitted gains include tire slip and actuator/rolling response;
they do not identify friction, stiffness, or a terrain-independent controller.

``--report PATH`` saves observations and before/after validation for preparation
of vehicle_telemetry. It does not modify the recording, its ROS controller
geometry, or the source telemetry preset. Real-data calibration remains a later step.

Run: python -m newton.examples diffsim_ancf_skid_steer (requires CUDA).
"""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples.ancf import _vehicle_config as vehicle_config
from newton.examples.ancf._terrain_common import quat_yaw
from newton.examples.ancf._vehicle_simulation import VehicleSimulation
from newton.examples.ancf._vehicle_usd import P_YUP_TO_ZU, VehicleUSD, asset_path, find_body, ghost_points
from newton.examples.ancf.diffsim._ancf_common import COLORS
from newton.examples.ancf.diffsim._skid_steer_calibration import SkidResponse

_DT = 1.0 / 60.0
_FIT_UPDATES = 16
_FIT_INTERVAL = 12


@wp.kernel
def _drive_sides(dofs: wp.array[int], signs: wp.array[float], cmd: wp.array[float], targets: wp.array[float]):
    wheel = wp.tid()
    targets[dofs[wheel]] = signs[wheel] * cmd[wheel % 2]


class _SkidRig(VehicleSimulation):
    """Example 03 physics with independent side motors and repeatable flat-ground resets."""

    def __init__(self, viewer, args):
        self.vehicle = VehicleUSD(asset_path(args.vehicle_asset))
        if self.vehicle.kind != "rigid_hull" or self.vehicle.steering != "skid":
            raise ValueError("Skid calibration requires a rigid-hull, skid-steered vehicle")
        self.requested = np.zeros(2)
        self.applied = np.zeros(2)
        super().__init__(viewer, args)
        self._initial = {
            "body_q": self.state_0.body_q.numpy().copy(),
            "joint_q": self.state_0.joint_q.numpy().copy(),
            "node_x": self.ancf_solver.node_x.numpy().copy(),
            "node_D": self.ancf_solver.node_D.numpy().copy(),
        }

    def _launch_drive(self):
        wp.launch(
            _drive_sides,
            dim=4,
            inputs=[self.vehicle._axle_dofs, self.vehicle._axle_sign, self.cmd, self.control.joint_target_qd],
            device=self.cmd.device,
        )

    def _update_controls(self):
        self.applied += np.clip(self.requested - self.applied, -0.2, 0.2)
        self.cmd.assign(self.applied.astype(np.float32))
        self._launch_drive()

    def reset(self):
        self.reset_state(
            self._initial["node_x"],
            self._initial["node_D"],
            body_q=self._initial["body_q"],
            joint_q=self._initial["joint_q"],
        )
        for state in (self.state_0, self.state_rigid):
            state.body_f.zero_()
        self.control.joint_f.zero_()
        self.requested.fill(0.0)
        self.applied.fill(0.0)
        self._update_controls()
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
        self._prescribe_beads()
        self._gs_prescribe_toggle = 0
        if self._gs_coupler is not None:
            self._gs_coupler.reset(self.state_0)
        self._frame = 0
        self._t = 0.0
        self._update_viz_buffers()


class Example:
    """Measure skid steering, learn its response, then compare physical drives."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.target_speed = float(args.target_speed)
        self.target_yaw = math.radians(args.target_yaw_deg)
        self._validate_target(self.target_speed, self.target_yaw)
        self._config_path = args.telemetry_config.resolve()
        if args.report is not None:
            self._validate_report_path(args.report)
        config = vehicle_config.load_telemetry_preset(self._config_path)
        rig_args = VehicleSimulation.create_parser().parse_args([])
        for key, value in config["args"].items():
            name = key.replace("-", "_")
            if hasattr(rig_args, name) and value is not None:
                setattr(rig_args, name, value)
        rig_args.wheel_speed = rig_args.steer_angle = 0.0
        rig_args.ground_z = rig_args.world_z_offset = 0.0
        rig_args.diag_period = 1_000_000
        self.rig = _SkidRig(viewer, rig_args)
        self.model = self.rig.model
        self._chassis = find_body(self.model, "chassis")
        self.response = SkidResponse(self.rig.vehicle.r_roll, 2 * self.rig.spec.half_track, self.model.device)
        self._frame = 0
        self._settle_frames = 180
        self._drive_frames = 360
        self._queue = deque()
        self._pending = None
        self._samples = []
        self._paths = {name: [] for name in COLORS}
        self.results = {}
        self.calibrated = False
        self.status = "Ready"
        self.phase = "Ready"
        self._report_written = False
        self._observe()
        self._ghost_pose = np.array([0.0, 0.0, 0.0])
        self._ghost_z = self.pose[2]
        self._spin = np.zeros(2)
        self._build_ghost()
        if viewer is not None:
            viewer.set_camera(pos=(5.5, -8.5, 5.8), pitch=-30.0, yaw=115.0)
        self._start(learn=args.train)

    def _build_ghost(self):
        rig, vehicle = self.rig, self.rig.vehicle
        points, triangles, labels = [], [], []
        count = 0

        def add(p, tri, wheel):
            nonlocal count
            points.append(p)
            triangles.append(np.asarray(tri).reshape(-1, 3) + count)
            labels.append(np.full(len(p), wheel, dtype=np.int32))
            count += len(p)

        for _name, p, tri, _color in vehicle.hull_visuals:
            add(p, tri, -1)
        p = rig.ancf_model.node_x0.numpy() @ P_YUP_TO_ZU.T
        quads = rig.ancf_model.elem_nodes.numpy()
        tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        spindle = rig._tire_meta.spindle
        for wheel in range(4):
            add(p, tri, wheel)
            if spindle is not None:
                mirror = vehicle.hubs_local[wheel, 1] < 0
                transform = np.diag([1.0, -1.0, 1.0]) @ P_YUP_TO_ZU if mirror else P_YUP_TO_ZU
                indices = spindle.triangle_indices[:, [0, 2, 1]] if mirror else spindle.triangle_indices
                add(spindle.points @ transform.T, indices, wheel)
        dev = self.model.device
        self._ghost_local = wp.array(np.concatenate(points), dtype=wp.vec3, device=dev)
        self._ghost_labels = wp.array(np.concatenate(labels), dtype=int, device=dev)
        self._ghost_indices = wp.array(np.concatenate(triangles).ravel(), dtype=int, device=dev)
        # Reference motion uses the chassis frame, including its offset from axle midpoint.
        self._ghost_hubs = wp.array(vehicle.hubs_local, dtype=wp.vec3, device=dev)
        self._ghost_spin = wp.zeros(4, dtype=float, device=dev)
        self._ghost_points = wp.empty_like(self._ghost_local)

    @staticmethod
    def _validate_target(speed, yaw):
        if not np.isfinite([speed, yaw]).all() or not 0.2 <= speed <= 0.9 or abs(yaw) > math.radians(3.0):
            raise ValueError("Choose speed 0.2–0.9 m/s and turn rate -3 to 3 degrees/s")

    def set_target(self, speed: float, yaw_rate: float) -> None:
        """Set the next reference speed [m/s] and yaw rate [rad/s]."""
        self._validate_target(speed, yaw_rate)
        self.target_speed, self.target_yaw = speed, yaw_rate
        self._pending = "compare" if self.calibrated else "learn"

    def _start(self, *, learn):
        self._paths = {name: [] for name in COLORS}
        self.results = {}
        self._report_written = False
        self._queue = deque(["Before"])
        if learn:
            self.calibrated = False
            self.response.gains.assign([1.0, 1.0, 0.0])
            self.response.history.clear()
            self._samples.clear()
            self._queue.extend(
                [
                    "Measure straight",
                    "Measure left",
                    "Measure right",
                    "Measure hard left",
                    "Measure hard right",
                    "Learn",
                ]
            )
        if learn or self.calibrated:
            self._queue.append("Learned")
        self._next_phase()

    def _next_phase(self):
        if not self._queue:
            self.phase = self.status = "Ready"
            return
        self.phase = self._queue.popleft()
        self._phase_frame = 0
        self._window = []
        if self.phase == "Learn":
            self.response.set_samples(
                np.array([r["command"] for r in self._samples]), np.array([r["measured"] for r in self._samples])
            )
            self.status = "Learning skid response"
        else:
            self.rig.reset()
            self._observe()
            self._ghost_pose[:] = [self.pose[0], self.pose[1], quat_yaw(self.pose)]
            self._ghost_z = self.pose[2]
            self._spin.fill(0.0)
            if self.phase in COLORS:
                self._paths[self.phase] = []
                self._paths["Reference"] = []
            self.status = f"{self.phase}: settling"
        print(f"[skid] {self.phase}", flush=True)

    def _observe(self):
        self.pose = self.rig.state_0.body_q.numpy()[self._chassis]
        qd = self.rig.state_0.body_qd.numpy()[self._chassis]
        yaw = quat_yaw(self.pose)
        self.measured = np.array([qd[0] * math.cos(yaw) + qd[1] * math.sin(yaw), qd[5]])
        nodes = self.rig.ancf_solver.node_x.numpy()
        if not np.isfinite(nodes).all() or not np.isfinite(self.pose).all() or not np.isfinite(qd).all():
            raise RuntimeError("Non-finite skid-steer simulation state")
        if nodes[:, 1].min() < -0.05:
            raise RuntimeError("Skid-steer tire penetration exceeded 50 mm")

    def step(self):
        self._frame += 1
        if self._pending is not None:
            self._start(learn=self._pending == "learn")
            self._pending = None
        if self.phase == "Ready":
            if self.args.report and self.calibrated and not self._report_written:
                self.export_report(self.args.report)
                self._report_written = True
            return
        self._phase_frame += 1
        if self.phase == "Learn":
            if self._phase_frame % _FIT_INTERVAL == 0:
                loss = self.response.step()
                self.status = f"Learning: {len(self.response.history) - 1}/{_FIT_UPDATES}, loss {loss:.5f}"
            if self._phase_frame >= _FIT_UPDATES * _FIT_INTERVAL:
                self.calibrated = True
                self._next_phase()
            return
        drive_frame = self._phase_frame - self._settle_frames
        if drive_frame <= 0:
            self.rig.requested.fill(0.0)
        else:
            ramp = min(drive_frame * _DT / 0.5, 1.0)
            if self.phase.startswith("Measure"):
                command = {
                    "Measure straight": (3.0, 3.0),
                    "Measure left": (1.5, 3.0),
                    "Measure right": (3.0, 1.5),
                    "Measure hard left": (0.75, 3.75),
                    "Measure hard right": (3.75, 0.75),
                }[self.phase]
                self.rig.requested[:] = ramp * np.array(command)
            else:
                command = self.response.wheel_commands(
                    self.target_speed, self.target_yaw, learned=self.phase == "Learned"
                )
                if np.max(np.abs(command)) > self.rig.spec.max_wheel_speed:
                    raise ValueError("Target exceeds the vehicle's wheel-speed limits; reduce speed or turn rate")
                self.rig.requested[:] = ramp * command
                v, w = ramp * self.target_speed, ramp * self.target_yaw
                mid = self._ghost_pose[2] + 0.5 * w * _DT
                self._ghost_pose += [v * math.cos(mid) * _DT, v * math.sin(mid) * _DT, w * _DT]
                self._spin += self.response.wheel_commands(v, w, learned=False) * _DT
            self.status = f"{self.phase}: {drive_frame * _DT:.1f} / {self._drive_frames * _DT:.0f} s"
        self.rig.step()
        self._observe()
        if drive_frame == 0:
            self._ghost_pose[:] = [self.pose[0], self.pose[1], quat_yaw(self.pose)]
            self._ghost_z = self.pose[2]
        if drive_frame > 0:
            if self.phase in ("Before", "Learned") and drive_frame % 3 == 0:
                self._paths[self.phase].append([self.pose[0], self.pose[1], 0.025])
                self._paths["Reference"].append([self._ghost_pose[0], self._ghost_pose[1], 0.03])
            if drive_frame > self._drive_frames - 120:
                self._window.append(self.measured.copy())
                if self.phase.startswith("Measure"):
                    self._samples.append(
                        {
                            "maneuver": self.phase,
                            "command": self.rig.applied.tolist(),
                            "measured": self.measured.tolist(),
                        }
                    )
        if drive_frame >= self._drive_frames:
            if self.phase in ("Before", "Learned"):
                mean = np.mean(self._window, axis=0)
                self.results[self.phase] = {
                    "speed_m_s": float(mean[0]),
                    "yaw_rad_s": float(mean[1]),
                    "speed_error_m_s": float(mean[0] - self.target_speed),
                    "yaw_error_rad_s": float(mean[1] - self.target_yaw),
                    "position_error_m": float(np.linalg.norm(self.pose[:2] - self._ghost_pose[:2])),
                }
                print(f"[skid] {self.phase}: {self.results[self.phase]}", flush=True)
            self._next_phase()

    def export_report(self, path: Path) -> None:
        """Save synthetic calibration and validation, plus command/velocity observations."""
        if not self.calibrated or "Learned" not in self.results:
            raise RuntimeError("Finish learning and a validation drive before exporting")
        self._validate_report_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "format": "newton.skid_response/1",
            "data_source": "synthetic MuJoCo/ANCF maneuvers",
            "telemetry_config": str(self._config_path),
            "vehicle_asset": self.rig.vehicle.name,
            "target_speed_m_s": self.target_speed,
            "target_yaw_rad_s": self.target_yaw,
            "speed_gain": float(self.response.gains.numpy()[0]),
            "yaw_gain": float(self.response.gains.numpy()[1]),
            "yaw_cubic_s2": float(self.response.gains.numpy()[2]),
            "nominal_pressure_pa": self.rig.ctis.currents if self.rig.ctis else [0.0] * 4,
            "loss_history": self.response.history,
            "validation": self.results,
            "gradient_model": "steady speed/yaw response; physical rollout is not differentiated",
        }
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with path.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["maneuver", "left_command_rad_s", "right_command_rad_s", "forward_speed_m_s", "yaw_rate_rad_s"]
            )
            for row in self._samples:
                writer.writerow([row["maneuver"], *row["command"], *row["measured"]])

    def _validate_report_path(self, path: Path) -> None:
        if path.resolve() == self._config_path:
            raise ValueError("Write a new report; preserve the source telemetry preset")
        if path.suffix.lower() != ".json":
            raise ValueError("Use a .json report path; observations go to a sibling .csv")

    def gui(self, ui):
        if getattr(self, "_ui_frames", 0) < 2:
            for label in ("Model Information", "Visualization", "Rendering Options", "Controls"):
                ui.get_state_storage().set_int(ui.get_id(label), 0)
            self._ui_frames = getattr(self, "_ui_frames", 0) + 1
        ui.text("Learn skid-steer response")
        ui.set_next_item_width(ui.get_content_region_avail().x)
        c1, speed = ui.slider_float("##speed", self.target_speed, 0.2, 0.9, "Speed: %.2f m/s")
        ui.set_next_item_width(ui.get_content_region_avail().x)
        c2, yaw = ui.slider_float("##yaw", math.degrees(self.target_yaw), -3.0, 3.0, "Turn: %.2f deg/s")
        if c1 or c2:
            self.set_target(float(np.clip(speed, 0.2, 0.9)), math.radians(float(np.clip(yaw, -3.0, 3.0))))
        if ui.button("Learn again"):
            self._pending = "learn"
        ui.same_line()
        ui.begin_disabled(not self.calibrated)
        if ui.button("Replay before / after"):
            self._pending = "compare"
        ui.end_disabled()
        ui.text_wrapped(self.status)
        ui.text(f"Left / right: {self.rig.applied[0]:.2f} / {self.rig.applied[1]:.2f} rad/s")
        ui.text(f"Actual speed: {self.measured[0]:.2f} m/s")
        ui.text(f"Actual turn: {math.degrees(self.measured[1]):.2f} deg/s")
        for name, color in COLORS.items():
            ui.text_colored(ui.ImVec4(*color, 1.0), "Gray target" if name == "Reference" else name)
            if name in self.results:
                r = self.results[name]
                ui.same_line()
                ui.text(f"gap {r['position_error_m']:.2f} m")
        ui.text_wrapped("Left turns slow the left wheels. The gray vehicle shows the requested motion.")
        if ui.collapsing_header("Advanced"):
            gains = self.response.gains.numpy()
            ui.text(f"Speed gain: {gains[0]:.3f}")
            ui.text(f"Turn: {gains[1]:.3f} linear + {gains[2]:.3f} cubic")
            ui.text_wrapped(
                "Warp learns speed and nonlinear turning response from simulated maneuvers. MuJoCo and all four tires test the resulting commands."
            )
            ui.text_wrapped("This prepares telemetry calibration; these are synthetic flat-ground observations.")

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._frame * _DT)
        self.viewer.log_state(self.rig.state_0)
        for name, path in self._paths.items():
            points = np.asarray(path, dtype=np.float32).reshape(-1, 3)
            self.viewer.log_lines(
                f"/skid/{name}",
                wp.array(points[:-1], dtype=wp.vec3, device=self.model.device),
                wp.array(points[1:], dtype=wp.vec3, device=self.model.device),
                COLORS[name],
                width=3.0,
            )
        self._ghost_spin.assign(np.tile(self._spin, 2).astype(np.float32))
        x, y, yaw = self._ghost_pose
        wp.launch(
            ghost_points,
            dim=len(self._ghost_local),
            inputs=[
                self._ghost_local,
                self._ghost_labels,
                self._ghost_hubs,
                self._ghost_spin,
                wp.transform(wp.vec3(x, y, self._ghost_z), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw))),
                self._ghost_points,
            ],
            device=self.model.device,
        )
        self.viewer.log_mesh(
            "/skid/ghost",
            self._ghost_points,
            self._ghost_indices,
            color=COLORS["Reference"],
            # During settling the two hulls coincide exactly and would depth-fight.
            hidden=self.phase.startswith("Measure")
            or self.phase == "Learn"
            or (self.phase != "Ready" and self._phase_frame <= self._settle_frames),
            backface_culling=False,
            roughness=0.5,
            metallic=0.0,
        )
        key = self.viewer._qualify("/skid/ghost") if hasattr(self.viewer, "_qualify") else "/skid/ghost"
        ghost = getattr(self.viewer, "objects", {}).get(key)
        if ghost is not None:
            ghost.alpha = 0.3
            ghost.cast_shadow = False
        self.viewer.end_frame()

    def test_post_step(self):
        self._observe()

    def test_final(self):
        from newton.tests.ancf_diffsim_checks import check_skid_steer_result  # noqa: PLC0415

        check_skid_steer_result(self)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=4000)
        parser.add_argument("--telemetry-config", type=Path, default=vehicle_config.TELEMETRY_PRESET)
        parser.add_argument("--target-speed", type=float, default=0.6, help="Target forward speed [m/s], 0.2–0.9.")
        parser.add_argument("--target-yaw-deg", type=float, default=2.0, help="Target yaw rate [degrees/s], -3 to 3.")
        parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--report", type=Path, help="Write a new calibration report and sibling observation CSV.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
