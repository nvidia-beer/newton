# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The recorded drive of a real vehicle (grey ghost) next to the simulated one, on the same terrain.

Extends the shared VehicleTerrain runtime. The RELLIS-3D bundles ``rellis_0000N`` come with the drive the real
Warthog made across them (``assets/vehicle_telemetry/<vehicle>/<sequence>/trajectory.csv``: the
pose at 10 Hz plus the smoothed forward speed / yaw rate derived from it, in the terrain's scene
frame). Here that recording is played back as a transparent grey copy of the vehicle - the
"telemetry car" - while the simulated vehicle (4 ANCF FEM tires) follows the same path.

* The ghost is driven by the telemetry alone: nothing in the simulation touches it. Position,
  roll / pitch / yaw are the recorded pose (interpolated at the simulation clock, 0 at pose 0
  after ``--settle-time``); the vehicle frame origin sits under the recorded base_link by the
  median height of the recorded pose over the reconstructed ground. Only the terrain stage
  ``w`` scales the height (heights = w * I_N), so the ghost stays on the terrain when the relief
  is changed; at w = 1 it is the pure recording.
* The ghost's wheels roll and its manoeuvres show: each side's wheel angle integrates the
  no-slip skid-steer kinematics of the recorded speed ``v`` and yaw rate ``w`` (left ``(v - w b) / r``,
  right ``(v + w b) / r``, half track ``b``, rolling radius ``r``). The dataset has no wheel
  signals, so these are derived, not measured. The wheels are the very meshes of the simulated
  tires (the baked ANCF shell at rest + the rim / hub).
* The simulated vehicle drives in ``--controller track`` (default): pure pursuit on the
  recorded path, at the recorded speed. Its separation from the ghost (along the ghost's heading /
  across it / heading error) is shown in the side panel and logged (``--log``). The HUD strip
  charts show the recorded (grey) against the simulated (orange) yaw rate and speed, the minimap a
  grey marker for the ghost. Every other control of vehicle_terrain (manual driving, camera,
  stage, CTIS panel, terrain overlays) works as there.
* ``--replay``: the robot's own recorded commands instead of pure pursuit, open loop. Every
  ``cmd_vel`` (speed, yaw rate) of the full-stack bag export goes through the robot's velocity
  controller (``drive_controller`` in the telemetry metadata: the left / right wheel speeds it
  commanded) into the brake levers and straight to the wheel motors, so the simulated vehicle gets
  what the real one got; where the two part is the model error. Needs a full-stack bag
  (newton-rellis-3d-tool: ``download_real_data.sh --bag N``, ``export_telemetry.py``).

Command: python -m newton.examples vehicle_telemetry [--terrain rellis_00000] [--vehicle-asset warthog_vehicle.usdc]
         [--ghost-alpha 0.35] [--replay]
(docker/config/07_vehicle_telemetry.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf import _vehicle_config as vehicle_config
from newton.examples.ancf._terrain_common import quat_yaw, require_vehicle_asset
from newton.examples.ancf._vehicle_terrain import MINIMAP_SIZE, SkidCommand, VehicleTerrain, wrap_pi
from newton.examples.ancf._vehicle_usd import P_YUP_TO_ZU, asset_path, ghost_points
from newton.solvers import load_ancf_tire_usd

_GHOST_COLOR = (0.62, 0.62, 0.66)
_GHOST_MESH = "/telemetry/ghost"
_SEQUENCE_FORMAT = "rellis.vehicle_telemetry/1"
_COLUMNS = (
    "elapsed_s",
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "forward_speed_m_s",
    "yaw_rate_rad_s",
)


def telemetry_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "vehicle_telemetry")


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vector ``v`` (3,) rotated by the unit quaternion ``q = (x, y, z, w)``."""
    u = q[:3]
    return v + 2.0 * q[3] * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _slerp(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1.0 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def _brake_levers(w_left: float, w_right: float) -> tuple[float, float]:
    """``(wheel speed [rad/s], lever (+ = left))`` of the brake-steer drive (``_vehicle_usd._drive_skid``: the
    braked side runs at ``w (1 - |lever|)``) that gives the per-side wheel speeds ``w_left`` / ``w_right``. The
    faster side keeps the throttle speed; a side asked to turn against the other stands (a brake only slows it)."""
    left_fast = abs(w_left) >= abs(w_right)
    w, slow = (w_left, w_right) if left_fast else (w_right, w_left)
    if w == 0.0:
        return 0.0, 0.0
    lever = 1.0 - min(max(slow / w, 0.0), 1.0)
    return w, (-lever if left_fast else lever)


class Telemetry:
    """One recorded drive: ``vehicle_telemetry/<vehicle>/<sequence>`` (``rellis.vehicle_telemetry/1``)."""

    def __init__(self, directory: str):
        with open(os.path.join(directory, "metadata.json")) as f:
            meta = json.load(f)
        if meta.get("format") != _SEQUENCE_FORMAT or meta.get("pose_frame") != "scene":
            raise ValueError(f"{directory}: unsupported telemetry format {meta.get('format')!r}")
        self.vehicle = str(meta["vehicle"])
        self.sequence = str(meta["sequence"])
        path = os.path.join(directory, meta["trajectory_csv"])
        with open(path) as f:
            header = f.readline().strip().split(",")
        a = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=[header.index(c) for c in _COLUMNS])
        self.t = a[:, 0]  # [s] since the first pose
        if not np.isfinite(a[:, :8]).all() or len(self.t) < 2 or not np.all(np.diff(self.t) > 0.0):
            raise ValueError(f"{path}: needs >= 2 finite, time-ordered poses")
        self.pos = a[:, 1:4]  # base_link in the scene frame [m]
        self.quat = a[:, 4:8] / np.linalg.norm(a[:, 4:8], axis=1, keepdims=True)  # (x, y, z, w)
        self.v = np.nan_to_num(a[:, 8])  # forward speed [m/s]
        self.w = np.nan_to_num(a[:, 9])  # yaw rate [rad/s]
        self.duration = float(self.t[-1])
        self.yaw_rate_peak = float(np.percentile(np.abs(self.w), 99.0))  # [rad/s]
        self.base_height = 0.0  # base_link over the ground [m], set by bind
        self._spin: np.ndarray | None = None
        # The robot's recorded commands (a full-stack bag export) and the velocity controller that turned them
        # into wheel speeds; cmd_t stays None without them.
        self.controller: dict | None = meta.get("drive_controller")
        self.cmd_t: np.ndarray | None = None
        self.cmd_source = ""
        if self.controller:
            self._load_commands(directory, meta)

    def _load_commands(self, directory: str, meta: dict) -> None:
        """The controller's command topic from the measured ROS recording that has the most of it."""
        best = None
        for rec in meta.get("measured_ros", []):
            path = os.path.join(directory, rec["metadata"])
            with open(path) as f:
                topic = json.load(f)["topics"].get(self.controller["command_topic"])
            if topic and topic.get("csv") and (best is None or topic["messages"] > best[0]):
                best = (topic["messages"], os.path.join(os.path.dirname(path), topic["csv"]))
        if best is None:
            return
        origin = int(meta["timestamp_origin_ns"])  # the recording's time zero (frame 0)
        with open(best[1], newline="") as f:
            rows = [(int(r["timestamp_ns"]), float(r["linear.x"]), float(r["angular.z"])) for r in csv.DictReader(f)]
        rows.sort()
        self.cmd_t = np.array([(ns - origin) * 1e-9 for ns, _v, _w in rows])  # [s], integer ns subtracted first
        self.cmd_v = np.array([v for _ns, v, _w in rows])  # [m/s]
        self.cmd_w = np.array([w for _ns, _v, w in rows])  # [rad/s]
        self.cmd_source = os.path.relpath(best[1], directory)

    def command_at(self, t: float) -> tuple[float, float]:
        """The recorded command ``(v [m/s], yaw rate [rad/s])`` in force at ``t`` [s]: the last message, or 0 once
        it is older than the controller's cmd_vel timeout (the robot stops then too)."""
        if self.cmd_t is None:
            return 0.0, 0.0
        i = int(np.searchsorted(self.cmd_t, t, side="right")) - 1
        if i < 0 or t - self.cmd_t[i] > float(self.controller["cmd_vel_timeout_s"]):
            return 0.0, 0.0
        return float(self.cmd_v[i]), float(self.cmd_w[i])

    def wheel_speeds(self, v: float, w: float) -> tuple[float, float]:
        """Left / right wheel speeds [rad/s] the robot's velocity controller commanded for ``(v, w)``."""
        c = self.controller
        track = float(c["wheel_separation_m"]) * float(c["wheel_separation_multiplier"])
        r = float(c["wheel_radius_m"]) * float(c["wheel_radius_multiplier"])
        return (v - 0.5 * w * track) / r, (v + 0.5 * w * track) / r

    def bind(self, ground_z: np.ndarray, half_track: float, r_roll: float) -> None:
        """Tie the recording to a vehicle and its terrain.

        Args:
            ground_z: Reconstructed ground height under every pose [m], from the terrain's
                reference track (same frames as the recording).
            half_track: Half the distance between the left and right wheels [m].
            r_roll: Wheel rolling radius [m].
        """
        if len(ground_z) != len(self.t):
            raise ValueError(f"the terrain track has {len(ground_z)} poses, the telemetry {len(self.t)}")
        self.base_height = float(np.median(self.pos[:, 2] - ground_z))
        dt = np.diff(self.t)
        omega = np.stack([(self.v - self.w * half_track), (self.v + self.w * half_track)], axis=1) / r_roll
        spin = np.zeros((len(self.t), 2))
        spin[1:] = np.cumsum(0.5 * (omega[1:] + omega[:-1]) * dt[:, None], axis=0)
        self._spin = spin  # [rad] left / right wheel angle, no-slip kinematics

    def _locate(self, t: float) -> tuple[int, float]:
        i = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        return i, float(np.clip((t - self.t[i]) / (self.t[i + 1] - self.t[i]), 0.0, 1.0))

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(base_link position (3,), quaternion (x, y, z, w), (left, right) wheel angle [rad])`` at ``t`` [s]."""
        i, f = self._locate(t)
        p = (1.0 - f) * self.pos[i] + f * self.pos[i + 1]
        return p, _slerp(self.quat[i], self.quat[i + 1], f), (1.0 - f) * self._spin[i] + f * self._spin[i + 1]

    def speed_at(self, t: float) -> float:
        """Recorded forward speed [m/s]; 0 after the end of the recording."""
        if t > self.duration:
            return 0.0
        i, f = self._locate(t)
        return float((1.0 - f) * self.v[i] + f * self.v[i + 1])

    def yaw_rate_at(self, t: float) -> float:
        """Recorded yaw rate [rad/s]; 0 after the end of the recording."""
        if t > self.duration:
            return 0.0
        i, f = self._locate(t)
        return float((1.0 - f) * self.w[i] + f * self.w[i + 1])


class Example(VehicleTerrain):
    """vehicle_terrain plus the recorded drive of the real vehicle as a transparent grey ghost."""

    def __init__(self, viewer=None, args=None):
        self._ghost_clock = 0.0  # [s] since the last reset, settle time included
        self.replay = bool(args.replay)  # drive the recorded commands open loop instead of pure pursuit
        seq = args.telemetry or args.terrain
        self.telemetry = Telemetry(os.path.join(telemetry_dir(), args.telemetry_vehicle, seq))
        veh = require_vehicle_asset(args)
        if veh.steering != "skid" or veh.kind != "rigid_hull":
            raise NotImplementedError(
                f"{veh.name}: the telemetry ghost needs a skid-steered rigid-hull vehicle "
                f"(its wheel motion is derived from the recorded speed and yaw rate)"
            )
        if self.telemetry.vehicle not in veh.name:
            print(f"[TELEMETRY] warning: the recording is of the {self.telemetry.vehicle}, the vehicle is {veh.name}")
        super().__init__(viewer, args)

        t = self.terrain
        if t.track is None or t.name != self.telemetry.sequence:
            raise ValueError(
                f"telemetry {self.telemetry.sequence} belongs to the terrain of the same name, not '{t.name}'"
            )
        self.telemetry.bind(t.track.ground[:, 2], float(self.spec.half_track), float(self.vehicle.r_roll))
        if viewer is not None:
            viewer.camera_speed = float(args.camera_speed)  # the maps are 100s of m across
        self.show_ghost = True
        self.ghost_alpha = float(args.ghost_alpha)
        self._ghost_err = (
            0.0,
            0.0,
            0.0,
        )  # sim minus recorded: along the ghost's heading [m], across [m], heading [rad]
        self._ghost_xy = (0.0, 0.0)
        self._ghost_yaw = 0.0
        self._err_sq, self._err_n = 0.0, 0
        self._log.update(
            {k: [] for k in ("ghost_x", "ghost_y", "ghost_yaw", "ghost_v", "err_along", "err_lat", "err_yaw")}
        )
        self._build_ghost()
        self._ghost_update()
        print(
            f"[TELEMETRY] {self.telemetry.vehicle} {self.telemetry.sequence}: {len(self.telemetry.t)} poses, "
            f"{self.telemetry.duration:.0f} s, base_link {self.telemetry.base_height:.3f} m over the ground, "
            f"peak yaw rate {math.degrees(self.telemetry.yaw_rate_peak):.0f} deg/s"
        )
        tel = self.telemetry
        if tel.cmd_t is not None:
            c = tel.controller
            print(
                f"[TELEMETRY] recorded commands: {len(tel.cmd_t)} over {tel.cmd_t[-1] - tel.cmd_t[0]:.0f} s "
                f"({tel.cmd_source}), controller track {c['wheel_separation_m'] * c['wheel_separation_multiplier']:.4g} m, "
                f"wheel radius {c['wheel_radius_m'] * c['wheel_radius_multiplier']:.3g} m"
                + ("; REPLAY: driven open loop" if self.replay else "")
            )
        elif self.replay:
            raise ValueError(
                f"--replay: {tel.sequence} has no recorded commands (or no drive_controller): download its full-stack "
                f"bag and re-export (newton-rellis-3d-tool: download_real_data.sh --only poses --bag N, export_telemetry.py)"
            )

    # ── Ghost geometry ─────────────────────────────────────────────────────────

    def _build_ghost(self) -> None:
        """One mesh in the vehicle frame: the hull's display meshes, and per wheel the baked tire
        shell at rest (as the simulated tires start) plus the tire asset's rim / hub."""
        veh, dev = self.vehicle, "cuda:0"
        local: list[np.ndarray] = []
        wheel: list[np.ndarray] = []
        tris: list[np.ndarray] = []
        n = 0

        def add(points: np.ndarray, triangles: np.ndarray, body: int) -> None:
            nonlocal n
            local.append(points.astype(np.float32))
            wheel.append(np.full(len(points), body, dtype=np.int32))
            tris.append(triangles.reshape(-1, 3).astype(np.int32) + n)
            n += len(points)

        for _name, pts, tri, _color in veh.hull_visuals:
            add(pts + veh.hull_origin, tri, -1)
        # The ghost is only drawn, so it always wears the vehicle's full-resolution tire (with the tread),
        # whatever tire the simulation runs.
        model, meta = load_ancf_tire_usd(asset_path(veh.default_tire_asset), device=dev)
        x0 = model.node_x0.numpy().astype(np.float64) @ P_YUP_TO_ZU.T  # tire frame -> vehicle frame
        quads = model.elem_nodes.numpy()
        shell = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
        sp = meta.spindle
        for i in range(len(veh.hubs)):
            add(x0, shell, i)
            if sp is not None:
                mirror = bool(veh.hubs_local[i, 1] < 0.0)  # right-hand wheels: hub cap outboard, as in the build
                P = np.diag([1.0, -1.0, 1.0]) @ P_YUP_TO_ZU if mirror else P_YUP_TO_ZU
                st = sp.triangle_indices[:, [0, 2, 1]] if mirror else sp.triangle_indices
                add(sp.points.astype(np.float64) @ P.T, st, i)
        self._ghost_local = wp.array(np.concatenate(local), dtype=wp.vec3, device=dev)
        self._ghost_wheel = wp.array(np.concatenate(wheel), dtype=int, device=dev)
        self._ghost_hubs = wp.array(veh.hubs.astype(np.float32), dtype=wp.vec3, device=dev)
        self._ghost_spin = wp.zeros(len(veh.hubs), dtype=float, device=dev)
        self._ghost_points = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._ghost_indices = wp.array(np.concatenate(tris).reshape(-1), dtype=wp.int32, device=dev)
        r = np.hypot(x0[:, 0], x0[:, 2])
        print(
            f"[TELEMETRY] ghost mesh: {n} points, {len(self._ghost_indices) // 3} triangles; tire shell centre "
            f"({x0[:, 0].mean():+.3f}, {x0[:, 1].mean():+.3f}, {x0[:, 2].mean():+.3f}) m, max radius {r.max():.3f} m "
            f"(tire R_outer {meta.R_outer:.3f} m)"
        )

    # ── Ghost motion (the recording only) ──────────────────────────────────────

    def _ghost_time(self) -> float:
        """Recording time [s]: the pose 0 stands still through the settle time, like the simulated vehicle."""
        return max(0.0, self._ghost_clock - self.settle_time)

    def _ghost_update(self, record: bool = False) -> None:
        tel, t = self.telemetry, self.terrain
        tg = self._ghost_time()
        p, q, spin = tel.at(tg)
        origin = p - _rotate(q, np.array([0.0, 0.0, tel.base_height]))  # the ground point under base_link
        origin[2] *= t.w  # heights = w * I_N: the ghost follows the stage (w = 1: the recording)
        origin[:2] += t.origin
        pose = wp.transform(wp.vec3(*(float(c) for c in origin)), wp.quat(*(float(c) for c in q)))
        self._ghost_spin.assign(np.array([spin[0], spin[1], spin[0], spin[1]], dtype=np.float32))  # FL FR RL RR
        wp.launch(
            ghost_points,
            dim=len(self._ghost_points),
            inputs=[self._ghost_local, self._ghost_wheel, self._ghost_hubs, self._ghost_spin, pose],
            outputs=[self._ghost_points],
            device="cuda:0",
        )
        yaw_g = quat_yaw(np.concatenate([origin, q]))
        self._ghost_xy, self._ghost_yaw = (float(origin[0]), float(origin[1])), yaw_g
        # Where the simulated vehicle is against it. The chassis body sits at -hull_origin from the vehicle centre.
        s = self._pose
        yaw_s = quat_yaw(s)
        c, sn = math.cos(yaw_s), math.sin(yaw_s)
        hx, hy = -float(self.vehicle.hull_origin[0]), -float(self.vehicle.hull_origin[1])
        dx = float(s[0]) + c * hx - sn * hy - origin[0]
        dy = float(s[1]) + sn * hx + c * hy - origin[1]
        cg, sg = math.cos(yaw_g), math.sin(yaw_g)
        along, across = dx * cg + dy * sg, -dx * sg + dy * cg
        dyaw = wrap_pi(yaw_s - yaw_g)
        self._ghost_err = (along, across, dyaw)
        if tg > 0.0 and tg <= tel.duration:
            self._err_sq += along * along + across * across
            self._err_n += 1
        if record and self._log_path:
            L = self._log
            for k, v in zip(
                ("ghost_x", "ghost_y", "ghost_yaw", "ghost_v", "err_along", "err_lat", "err_yaw"),
                (origin[0], origin[1], yaw_g, tel.speed_at(tg), along, across, dyaw),
                strict=True,
            ):
                L[k].append(float(v))

    # ── Simulated vehicle: same path, recorded speed ───────────────────────────

    def _apply_speed(self) -> None:
        super()._apply_speed()
        # the HUD yaw chart spans 1.5 x yaw_set: here the largest recorded yaw rate
        self.yaw_set = self.telemetry.yaw_rate_peak / 1.5

    def _track_route(self) -> tuple[float, float, int]:
        _v, yaw_cmd, turn = super()._track_route()
        v_rec = self.telemetry.speed_at(self._t_scn - self.settle_time)
        return min(max(v_rec, 0.0), self.plant.v_max), yaw_cmd, turn

    def _drive(self) -> None:
        if not self.replay or self.telemetry.cmd_t is None:
            super()._drive()
            return
        # Open loop: the robot's recorded (v, w) -> its controller's left / right wheel speeds -> brake levers,
        # straight to the wheel motors. The controller metadata is a stock-config assumption.
        self._scenario_tick()  # clock, turns, off-track readout; its pure pursuit command is replaced below
        tel = self.telemetry
        command_time = self._t_scn - vehicle_config.FRAME_DT - self.settle_time
        self.v_cmd, self.yaw_cmd = tel.command_at(command_time) if command_time >= 0.0 else (0.0, 0.0)
        w, lever = _brake_levers(*tel.wheel_speeds(self.v_cmd, self.yaw_cmd))
        w_max = float(self.spec.max_wheel_speed)
        w = max(-w_max, min(w_max, w))
        self._cmd = SkidCommand(w / w_max, lever)
        self.steer_angle = self._steer_sign * lever * self._steer_range
        self._target_wheel_speed = self.wheel_speed = w

    def _on_reset(self) -> None:
        self._ghost_clock = 0.0
        self._err_sq, self._err_n = 0.0, 0
        super()._on_reset()

    def step(self) -> None:
        self._ghost_clock += vehicle_config.FRAME_DT
        super().step()

    def _telemetry(self) -> None:
        super()._telemetry()
        if not self._fault:
            # Record the reference at the same end-of-frame time as the simulated
            # state, before the base step writes the final log.
            self._ghost_update(record=True)
        if self._fault or not self._trace_yaw:
            return
        tg = self._ghost_time()  # grey = recorded, orange = simulated
        self._trace_yaw[-1] = (self.telemetry.yaw_rate_at(tg), self.yaw_rate)
        self._trace_v[-1] = (self.telemetry.speed_at(tg), self.v_fwd)

    # ── GUI / HUD ──────────────────────────────────────────────────────────────

    def _gui_scenario(self, ui) -> None:
        tel = self.telemetry
        ui.text(
            f"Telemetry  {tel.vehicle} {tel.sequence}   t = {min(self._ghost_time(), tel.duration):.1f} / {tel.duration:.0f} s"
        )
        v = self.viewer
        if v is not None and hasattr(v, "camera_speed"):
            _c, v.camera_speed = ui.slider_float("free camera speed [m/s] (WASD)", v.camera_speed, 1.0, 100.0)
        _c, self.show_ghost = ui.checkbox("Show the recorded vehicle (grey ghost)", self.show_ghost)
        _c, self.ghost_alpha = ui.slider_float("ghost opacity", self.ghost_alpha, 0.05, 1.0)
        along, across, dyaw = self._ghost_err
        ui.text(
            f"  simulated vs recorded: {along:+.2f} m along  {across:+.2f} m across  {math.degrees(dyaw):+.1f} deg heading"
        )
        how = (
            "drives the robot's recorded commands (open loop)"
            if self.replay and tel.cmd_t is not None
            else "follows the recorded path at the recorded speed"
        )
        ui.text(
            f"  the simulated vehicle {how}; {len(self.turns)} turns done, peak yaw {math.degrees(tel.yaw_rate_peak):.0f} deg/s"
        )
        if tel.cmd_t is not None:
            _c, self.replay = ui.checkbox(
                "Replay the robot's recorded commands (open loop, no pure pursuit)", self.replay
            )
        if self._skid and not self.replay:
            _c, self.plant.yaw_authority = ui.slider_float(
                "yaw authority (measured / kinematic yaw per lever)", self.plant.yaw_authority, 0.1, 1.5
            )

    def _draw_minimap(self, imgui, x0, y0) -> None:
        super()._draw_minimap(imgui, x0, y0)
        if not self.show_ghost:
            return
        s = self._map_scale
        cx, cy = x0 + 0.5 * MINIMAP_SIZE, y0 + 0.5 * MINIMAP_SIZE
        p = imgui.ImVec2(cx + self._ghost_xy[0] * s, cy - self._ghost_xy[1] * s)
        tip = imgui.ImVec2(p.x + 9.0 * math.cos(self._ghost_yaw), p.y - 9.0 * math.sin(self._ghost_yaw))
        grey = self._col(imgui, 0.8, 0.8, 0.85)
        draw = imgui.get_window_draw_list()
        draw.add_line(p, tip, grey, 2.0)
        draw.add_circle_filled(p, 4.0, grey)

    # ── Render ─────────────────────────────────────────────────────────────────

    def _render_overlays(self) -> None:
        super()._render_overlays()
        v = self.viewer
        v.log_mesh(
            _GHOST_MESH,
            self._ghost_points,
            self._ghost_indices,
            hidden=not self.show_ghost,
            backface_culling=False,
            color=_GHOST_COLOR,
            roughness=0.9,
            metallic=0.0,
        )
        # The GL viewer draws a mesh with alpha < 1 blended over the scene (nearest layer only);
        # a shadow of the recording would read as a second vehicle on the ground, so it casts none.
        obj = getattr(v, "objects", {}).get(v._qualify(_GHOST_MESH) if hasattr(v, "_qualify") else _GHOST_MESH)
        if obj is not None:
            obj.alpha = self.ghost_alpha
            obj.cast_shadow = False

    # ── Diagnostics / tests ────────────────────────────────────────────────────

    def _print_diag(self, fps: float = 0.0) -> None:
        super()._print_diag(fps)
        along, across, dyaw = self._ghost_err
        print(
            f"  telemetry t={self._ghost_time():.1f} s: recorded v={self.telemetry.speed_at(self._ghost_time()):.2f} m/s  "
            f"simulated - recorded: {along:+.2f} m along  {across:+.2f} m across  {math.degrees(dyaw):+.1f} deg heading"
        )

    # The runner requires these hooks; test code is loaded only in test mode.
    def test_post_step(self) -> None:
        from newton.tests.ancf_vehicle_checks import check_vehicle_step  # noqa: PLC0415

        check_vehicle_step(self)

    def test_final(self) -> None:
        from newton.tests.ancf_vehicle_checks import check_telemetry_final  # noqa: PLC0415

        check_telemetry_final(self)

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = VehicleTerrain.create_parser()
        parser.add_argument(
            "--telemetry",
            type=str,
            default=None,
            help="Recorded sequence under assets/vehicle_telemetry/<--telemetry-vehicle>/; default: the terrain's name.",
        )
        parser.add_argument("--telemetry-vehicle", type=str, default="warthog", help="Vehicle folder of the recording.")
        parser.add_argument(
            "--camera-speed",
            type=float,
            default=12.0,
            help="Free-camera WASD / QE speed [m/s] (the viewer default is 4).",
        )
        parser.add_argument("--ghost-alpha", type=float, default=0.35, help="Opacity of the recorded vehicle, 0..1.")
        parser.add_argument(
            "--replay",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Drive the robot's recorded commands open loop instead of pure pursuit (needs a full-stack bag export).",
        )
        # A recording exists for the RELLIS bundles; the simulated vehicle takes its speed from it (--speed
        # only scales the HUD charts; the lateral-acceleration cap would slow it below the recording).
        parser.set_defaults(
            terrain="rellis_00000",
            vehicle_asset="warthog_vehicle.usdc",
            tire_asset="warthog_ancf_tire_simple.usda",  # low-resolution wheels (240 nodes); --tire-asset warthog_ancf_tire.usda for the chevron tread
            speed=1.5,
            a_lat=0.0,
            num_frames=20000,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
