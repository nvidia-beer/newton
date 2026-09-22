# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Double-wishbone vehicle on 4 ANCF FEM tires manoeuvring around a track terrain.

Extends vehicle_ancf_tires (3): the flat plane is replaced by a heightfield baked by
``third_party/newton-terrain-tool`` - a closed rocky corridor with lunar mountains as
boundaries plus the route metadata (centerline, borders, spawn, checkpoints, arc-length /
signed-distance grids) that tells the example where the track is.  Rigid heightfield contact
through ``TerrainSCM(rigid=True)``; the Bekker-Wong soil mode of that class is not wired in here.

  --terrain NAME   loads examples/ancf/assets/terrain/NAME/NAME_terrain.json
                   (bake with third_party/newton-terrain-tool/regenerate_all.sh)

The base example spawns the car at the origin heading +x, so the terrain is shifted under it
by (-spawn.x, -spawn.y): the MuJoCo hfield through the shape transform, the ANCF tire
contact through ``TerrainSCM(origin=...)``.  The tool bakes the route with the spawn heading
already along +x (``align_spawn_heading``).

Two heightfield shapes share the grid: the fine one is visible and non-colliding (rendered by
the viewer), a block-mean copy at ``--mujoco-terrain-cell`` (0.5 m) is the hidden MuJoCo
collider for rims, arms and chassis.  The 0.5 m block mean is kept for this example; the field
example collides the rigid parts with the fine grid itself (the vendored mujoco_warp collects up
to ``MJ_MAXHFPRISM = 128`` hfield prisms per geom).  The ANCF tires always see the fine grid.

Driver: a :class:`~newton.examples.ancf._track_driver.TrackDriver` policy selected with
``--driver`` - ``manual`` (the inherited steer / throttle sliders) or ``pure_pursuit``
(lookahead steering + curvature-limited speed).  Each frame the example builds a
``TrackObservation`` (speed, heading error, lateral offset, border distance, curvature ahead,
...) and applies the returned normalized ``DriveCommand``; a learned policy is one more
subclass in ``_track_driver.DRIVERS``.  Steering is the base example's parallel kingpin angle;
its sign is read from the kingpin joint axis so "+" is a left turn.  The wheel speed is passed
with "+" = forward; the vehicle asset's drive applies the per-axle hinge sign.  UI from the
newton vehicles MPPI track example without the planner and cones: follow camera, minimap,
G-meter, speed / throttle panel, corridor telemetry, plus the inherited CTIS panel.  ``--test``
runs pure pursuit and asserts progress along the route inside the corridor.

Command: python -m newton.examples vehicle_track [--terrain test_03_fourier_hard]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf import example_vehicle_ancf_tires as _dw
from newton.examples.ancf._field_task import downsample, load_height_png
from newton.examples.ancf._terrain_common import (
    FAR_BELOW,
    SPAWN_CLEARANCE,
    quat_roll_pitch,
    quat_yaw,
    set_follow_camera,
    terrain_contact_spikes,
)
from newton.examples.ancf._track_driver import (
    DRIVERS,
    TRAJECTORY_SAMPLES,
    DriveCommand,
    ManualDriver,
    PurePursuitDriver,
    TrackDriver,
    TrackObservation,
)
from newton.solvers import TerrainSCM

# Intended-path arrow (driver.trajectory): ribbon width, head size, lift above the terrain [m]
_ARROW_HALF_WIDTH = 0.15
_ARROW_HEAD_HALF_WIDTH = 0.5
_ARROW_HEAD_LENGTH = 1.2
_ARROW_LIFT = 0.35

# MuJoCo hfield collider cell: a 0.5 m block mean of the 0.25 m grid, kept for this example (the
# field example collides with the fine grid; the vendored mujoco_warp collects up to
# MJ_MAXHFPRISM = 128 prisms per geom). 0.5 m still resolves the ~1 m boulders for the rim stops.
_MJ_TERRAIN_CELL = 0.5  # [m]
_TEST_SPEED = 3.0  # [m/s] pure-pursuit cruise under --test

# ── HUD (from the MPPI track example) ─────────────────────────────────────────
MINIMAP_SIZE = 240.0
MINIMAP_MARGIN = 12.0
MINIMAP_PAD = 0.08
MINIMAP_TRAIL_MAX = 3600
GMETER_SIZE = 150.0
HUD_GAP = 10.0
HUD_PANEL_W = 496.0
HUD_BAR_SEGMENTS = 48
GMETER_G_EDGE = 2.5
GMETER_TRAIL_MAX = 20
GMETER_EMA = 0.3
HUD_FONT_CANDIDATES = (
    str(Path.home() / ".local/share/fonts/JetBrainsMono-Regular.ttf"),
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
)
HUD_FONT_BOLD_CANDIDATES = (
    str(Path.home() / ".local/share/fonts/JetBrainsMono-Bold.ttf"),
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/jetbrains-mono/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Bold.ttf",
)


# ── Terrain asset ─────────────────────────────────────────────────────────────


def terrain_dir(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "terrain", name)


class TrackTerrain:
    """A baked terrain directory, re-expressed in the vehicle world (spawn at the origin).

    All coordinates returned by this class are world coordinates: the tool's grid frame
    shifted by ``(-spawn.x, -spawn.y)``.  ``origin`` is the world position of the grid centre.
    """

    def __init__(self, name: str):
        d = terrain_dir(name)
        meta_path = os.path.join(d, f"{name}_terrain.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"terrain '{name}' not found ({meta_path}); bake it with "
                f"third_party/newton-terrain-tool/regenerate_all.sh --name {name}"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        g = self.meta["grid"]
        files = self.meta["files"]
        sp = self.meta["spawn"]
        if abs(float(sp["yaw"])) > 1.0e-3:
            raise ValueError(
                f"terrain '{name}' spawn yaw is {math.degrees(sp['yaw']):.1f} deg; the vehicle spawns heading +x. "
                "Re-bake with newton-terrain-tool (align_spawn_heading, default on)."
            )
        self.name = name
        self.cell = float(g["cell"])
        self.hx, self.hy = float(g["hx"]), float(g["hy"])
        self.max_h = float(g["max_h"])
        self.origin = (-float(sp["x"]), -float(sp["y"]))
        self.spawn_h = float(sp["h"])
        self.heights = (load_height_png(os.path.join(d, files["height_png"])) * self.max_h).astype(np.float32)
        assert self.heights.shape == (int(g["nrow"]), int(g["ncol"])), "height PNG does not match terrain.json"
        self.sdf = np.load(os.path.join(d, files["sdf_npy"])).astype(np.float32)
        self.arc = np.load(os.path.join(d, files["s_npy"])).astype(np.float32)
        with open(os.path.join(d, files["route_json"])) as f:
            r = json.load(f)
        shift = np.array([self.origin[0], self.origin[1], 0.0], dtype=np.float32)
        self.center = np.asarray(r["center"], dtype=np.float32) + shift
        self.left = np.asarray(r["left"], dtype=np.float32) + shift
        self.right = np.asarray(r["right"], dtype=np.float32) + shift
        self.half_width = float(r["half_width"])
        self.length = float(r["length"])
        self.checkpoints = [
            {**c, "x": c["x"] + self.origin[0], "y": c["y"] + self.origin[1]} for c in self.meta.get("checkpoints", [])
        ]
        nxt = np.roll(self.center[:, :2], -1, axis=0)
        self.seg = np.linalg.norm(nxt - self.center[:, :2], axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)[:-1]])
        # Menger curvature per centerline point (for the drivers' speed cap / RL observation)
        a = np.roll(self.center[:, :2], 1, axis=0)
        b = self.center[:, :2]
        cross = np.abs((b - a)[:, 0] * (nxt - b)[:, 1] - (b - a)[:, 1] * (nxt - b)[:, 0])
        den = np.linalg.norm(b - a, axis=1) * np.linalg.norm(nxt - b, axis=1) * np.linalg.norm(nxt - a, axis=1)
        self.curvature = (2.0 * cross / np.maximum(den, 1e-9)).astype(np.float32)

    def _sample(self, grid: np.ndarray, x: float, y: float) -> float:
        nrow, ncol = grid.shape
        fx = min(max((x - self.origin[0] + self.hx) / self.cell, 0.0), ncol - 1 - 1e-6)
        fy = min(max((y - self.origin[1] + self.hy) / self.cell, 0.0), nrow - 1 - 1e-6)
        c, r = int(fx), int(fy)
        tx, ty = fx - c, fy - r
        return float(
            (grid[r, c] * (1 - tx) + grid[r, c + 1] * tx) * (1 - ty)
            + (grid[r + 1, c] * (1 - tx) + grid[r + 1, c + 1] * tx) * ty
        )

    def height_at(self, x: float, y: float) -> float:
        return self._sample(self.heights, x, y)

    def sdf_at(self, x: float, y: float) -> float:
        """Signed distance to the corridor border [m], negative inside."""
        return self._sample(self.sdf, x, y)

    def s_at(self, x: float, y: float) -> float:
        """Arc length of the nearest centerline point [m] (nearest cell: the field wraps at the seam)."""
        c = int(min(max(round((x - self.origin[0] + self.hx) / self.cell), 0), self.arc.shape[1] - 1))
        r = int(min(max(round((y - self.origin[1] + self.hy) / self.cell), 0), self.arc.shape[0] - 1))
        return float(self.arc[r, c])

    def _segment(self, s: float) -> tuple[int, int, float]:
        s = s % self.length
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.cum) - 1))
        j = (i + 1) % len(self.cum)
        u = float(np.clip((s - self.cum[i]) / max(self.seg[i], 1e-9), 0.0, 1.0))
        return i, j, u

    def point_at(self, s: float) -> np.ndarray:
        """Centerline point (3,) at arc length ``s`` (wrapped)."""
        i, j, u = self._segment(s)
        return self.center[i] + (self.center[j] - self.center[i]) * u

    def curvature_at(self, s: float) -> float:
        """Centerline curvature [1/m] at ``s``."""
        return float(self.curvature[self._segment(s)[0]])


# ── Example ───────────────────────────────────────────────────────────────────


class Example(_dw.Example):
    """Double-wishbone car + 4 ANCF tires on a newton-terrain-tool track."""

    # MuJoCo contact budget for the hfield pairs (rims, arm capsules, chassis boxes; up to
    # MJ_MAXHFPRISM = 128 candidate prisms per geom, of which only the ones under the part touch).
    _NCONMAX = 512
    _NJMAX = 2048

    def __init__(self, viewer=None, args=None):
        if args is None:
            args = argparse.Namespace()
        self._args = args
        self.terrain = TrackTerrain(args.terrain)
        t = self.terrain
        print(
            f"[TRACK] {t.name}: {t.heights.shape[1]}x{t.heights.shape[0]} px  cell {t.cell} m  max_h {t.max_h:.1f} m  "
            f"route {t.length:.0f} m  corridor {2 * t.half_width:.0f} m  spawn h {t.spawn_h:.2f} m  grid centre {t.origin}"
        )
        # Flat plane off, whole vehicle lifted to the terrain height at the spawn.
        args.ground_z = FAR_BELOW
        args.world_z_offset = t.spawn_h + SPAWN_CLEARANCE
        super().__init__(viewer, args)

        # Throttle range: the slider limit is the vehicle asset's maxWheelSpeed (jeep 12 rad/s =
        # 6 m/s, Sherp 12.3 rad/s = 40 km/h) unless --max-wheel-speed widens it (the spec is frozen;
        # the base only reads max_wheel_speed when it draws the slider).
        if args.max_wheel_speed is not None:
            self.spec = dataclasses.replace(self.spec, max_wheel_speed=float(args.max_wheel_speed))
        # Wheel-speed ramp per frame: the axles are velocity servos, so the ramp is capped at what
        # the tires can transmit, a rim acceleration of mu g, i.e. mu g / r_roll [rad/s^2] (jeep 17.6,
        # Sherp 9.8); a faster ramp only spins the wheels. The base example's own ramp finishes the
        # last _dw._WHEEL_SPEED_RATE of every step, this adds the rest.
        mu = float(getattr(args, "mu", _dw._MU))
        self._wheel_speed_rate = max(mu * _dw._GRAVITY / self.vehicle.r_roll * _dw._FRAME_DT, _dw._WHEEL_SPEED_RATE)

        # Kingpin sign: a positive hinge angle about +Z is a left (CCW) turn.
        self._steer_sign = 1.0 if self._kingpin_axis_z > 0.0 else -1.0

        self._test = bool(getattr(args, "test", False))
        self.follow_camera = True
        self.drive_cmd = 0.0
        self.steer_cmd = 0.0

        # ── Driver (policy) behind the TrackDriver interface ──
        self._pp_speed = _TEST_SPEED if self._test else float(args.speed)
        self._pp_a_lat = 2.5 if self._test else float(args.a_lat)
        self.driver: TrackDriver = self._make_driver("pure_pursuit" if self._test else str(args.driver))
        self._obs = TrackObservation(route=t, half_width=t.half_width)

        dev = "cuda:0"
        self._chassis_body = _dw._find_body(self.model, "chassis")
        self._chassis_q = wp.zeros(1, dtype=wp.transform, device=dev)
        self._chassis_qd = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
        self._pose = self._read_chassis()
        self._prev_v = None
        self._accel_car_g = (0.0, 0.0)
        self._gmeter_trail = deque(maxlen=GMETER_TRAIL_MAX)
        self._trail = deque(maxlen=MINIMAP_TRAIL_MAX)
        self._s_prev = t.s_at(float(self._pose[0]), float(self._pose[1]))
        self._s_total = 0.0
        self._tele = {
            "speed": 0.0,
            "yaw_rate": 0.0,
            "s": self._s_prev,
            "meters": 0.0,
            "laps": 0,
            "n": 0.0,
            "sdf": t.sdf_at(0.0, 0.0),
            "roll": 0.0,
            "pitch": 0.0,
            "h": 0.0,
        }
        self._hero_yaw = 0.0
        self._hud_ok = True
        self._minimap_ok = True
        self._minimap_boundary_px = None
        self._hud_font = None
        self._hud_font_bold = None
        self._hud_font_tried = False
        self._init_minimap()

        # Route borders as white ribbons (world-space width; the viewer's log_lines width is a
        # single global pixel value shared with the bead rings and contact spikes).
        self._boundary_ribbons = []
        for poly in (t.left, t.right):
            self._boundary_ribbons.append(self._ribbon(poly, half_width=0.125, lift=0.15, device=dev))
        # Curved 3D arrow for the driver's intended path: fixed vertex count (2K shaft + 3 head)
        # so the device buffers are allocated once and only refilled per frame.
        k = TRAJECTORY_SAMPLES
        self._arrow_host = np.zeros((2 * k + 3, 3), dtype=np.float32)
        self._arrow_points = wp.zeros(2 * k + 3, dtype=wp.vec3, device=dev)
        self._arrow_indices = wp.array(self._arrow_mesh_indices(k), dtype=wp.int32, device=dev)
        self._arrow_visible = False
        if viewer is not None:
            set_follow_camera(viewer, self._pose)
            if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
                viewer.camera.fov = 65.0

    # ── Terrain hook (MJCF imported, ANCF solver built, nothing captured yet) ──

    def _on_car_builder(self, car: newton.ModelBuilder) -> None:
        t = self.terrain
        # Rigid heightfield contact for the tire nodes, grid centred at t.origin in the world.
        self.terrain_scm = TerrainSCM(
            heights=t.heights,
            hx=t.hx,
            hy=t.hy,
            node_x0=self.ancf_model.node_x0,
            elem_nodes=self.ancf_model.elem_nodes,
            n_envs=_dw._N_TIRES,
            patch_width=self.spec.tire_width,
            rigid=True,
            mu_rigid=self.ancf_solver.mu,
            origin=t.origin,
            device="cuda:0",
        )
        self.ancf_solver.terrain = self.terrain_scm

        # Fine grid: visible, non-colliding (the viewer renders hfields; SolverMuJoCo maps
        # has_shape_collision=False to contype = conaffinity = 0).
        h = t.heights
        vis_cfg = car.default_shape_cfg.copy()
        vis_cfg.has_shape_collision = False
        car.add_shape_heightfield(
            xform=wp.transform(wp.vec3(t.origin[0], t.origin[1], 0.0), wp.quat_identity()),
            heightfield=newton.Heightfield(
                data=h,
                nrow=h.shape[0],
                ncol=h.shape[1],
                hx=t.hx,
                hy=t.hy,
                min_z=float(h.min()),
                max_z=float(max(h.max(), h.min() + 1.0e-3)),
            ),
            cfg=vis_cfg,
            color=wp.vec3(0.22, 0.22, 0.24),  # dark grey rock
            label="terrain_visual",
        )
        # Coarse block-mean copy: the hidden MuJoCo collider for rims, arms and chassis boxes.
        mj_cell = float(self._args.mujoco_terrain_cell)
        k = max(1, int(round(mj_cell / t.cell)))
        # The mean (not the max) keeps the coarse collider at or below the boulder tops the tires
        # see, so rims never ride on a rock the fine grid does not have.
        hc = downsample(h, k, "mean")
        hx_c = 0.5 * (hc.shape[1] - 1) * k * t.cell
        hy_c = 0.5 * (hc.shape[0] - 1) * k * t.cell
        # coarse node (0, 0) coincides with fine node (0, 0) at (-hx, -hy): shift the centre accordingly
        col_cfg = car.default_shape_cfg.copy()
        col_cfg.is_visible = False
        car.add_shape_heightfield(
            xform=wp.transform(wp.vec3(t.origin[0] - t.hx + hx_c, t.origin[1] - t.hy + hy_c, 0.0), wp.quat_identity()),
            heightfield=newton.Heightfield(
                data=hc,
                nrow=hc.shape[0],
                ncol=hc.shape[1],
                hx=hx_c,
                hy=hy_c,
                min_z=float(hc.min()),
                max_z=float(max(hc.max(), hc.min() + 1.0e-3)),
            ),
            cfg=col_cfg,
            label="terrain_collider",
        )
        print(
            f"[TRACK] MuJoCo collider grid {hc.shape[1]}x{hc.shape[0]} at {k * t.cell:.2f} m (block mean of the {t.cell} m grid)"
        )
        # The MJCF worldbody plane (the base puts it back to z = 0) is not the floor here.
        for s in range(car.shape_count):
            if car.shape_body[s] == -1 and car.shape_type[s] == newton.GeoType.PLANE:
                xf = car.shape_transform[s]
                car.shape_transform[s] = wp.transform(wp.vec3(xf.p[0], xf.p[1], FAR_BELOW), xf.q)
        self._kingpin_axis_z, _ = self.vehicle.drive_axes(car)

    # ── Driver ─────────────────────────────────────────────────────────────────

    def _make_driver(self, name: str) -> TrackDriver:
        if name not in DRIVERS:
            raise KeyError(f"unknown driver '{name}'; available: {sorted(DRIVERS)}")
        if name == PurePursuitDriver.name:
            # Bicycle-model pure pursuit emits delta = atan(2 L sin(alpha) / L_d), normalized by
            # max_steer. A brake-steered vehicle (VehicleUSD._drive_skid: the braked side slows by
            # the lever travel) yaws at v * turn / (2 half_track) = v * turn / track, so it is built
            # with L = track and max_steer = 1: the small-angle output is then exactly the lever
            # command its drive expects.
            skid = self.vehicle.steering == "skid"
            drv = PurePursuitDriver(
                wheelbase=2.0 * self.spec.half_track if skid else 2.0 * self.spec.half_wheelbase,
                max_steer=self.spec.max_steer,
                tire_radius=self.vehicle.r_roll if skid else self.spec.tire_R_outer,
                max_wheel_speed=self.spec.max_wheel_speed,
                target_speed=self._pp_speed,
                a_lat_max=self._pp_a_lat,
            )
        else:
            drv = DRIVERS[name]()
        drv.reset()
        return drv

    def _slider_command(self) -> DriveCommand:
        """The inherited sliders as a normalized command (+ steer = left, + drive = forward)."""
        return DriveCommand(
            self._target_wheel_speed / self.spec.max_wheel_speed,
            self._steer_sign * self.steer_angle / self.spec.max_steer,
        )

    def _apply_command(self, cmd: DriveCommand) -> None:
        self.steer_angle = self._steer_sign * cmd.steer * self.spec.max_steer
        self._target_wheel_speed = cmd.drive * self.spec.max_wheel_speed

    def step(self) -> None:
        if isinstance(self.driver, ManualDriver):
            self.driver.command = self._slider_command()  # sliders are the source of truth
        cmd = self.driver.act(self._obs)
        self._apply_command(cmd)
        self.drive_cmd, self.steer_cmd = cmd.drive, cmd.steer
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) > self._wheel_speed_rate:
            self.wheel_speed += math.copysign(self._wheel_speed_rate - _dw._WHEEL_SPEED_RATE, d)
        super().step()
        self._update_telemetry()

    # ── Telemetry (two chassis rows read back per frame) ───────────────────────

    def _read_chassis(self) -> np.ndarray:
        wp.copy(self._chassis_q, self.state_0.body_q, src_offset=self._chassis_body, count=1)
        return self._chassis_q.numpy()[0]

    def _update_telemetry(self) -> None:
        q = self._read_chassis()
        wp.copy(self._chassis_qd, self.state_0.body_qd, src_offset=self._chassis_body, count=1)
        qd = self._chassis_qd.numpy()[0]
        self._pose = q
        tr = self.terrain
        x, y = float(q[0]), float(q[1])
        yaw = quat_yaw(q)
        v = np.array([float(qd[0]), float(qd[1])])
        s = tr.s_at(x, y)
        ds = s - self._s_prev
        if ds < -0.5 * tr.length:
            ds += tr.length
        elif ds > 0.5 * tr.length:
            ds -= tr.length
        if abs(ds) < 5.0:
            self._s_total += ds
        self._s_prev = s
        near = tr.point_at(s)
        nxt = tr.point_at(s + 1.0)
        tang = np.array([float(nxt[0] - near[0]), float(nxt[1] - near[1])])
        tang /= max(np.linalg.norm(tang), 1e-9)
        lateral = float(-(x - float(near[0])) * tang[1] + (y - float(near[1])) * tang[0])
        roll, pitch = quat_roll_pitch(q)
        if self._prev_v is not None:
            a = (v - self._prev_v) / _dw._FRAME_DT
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            right = np.array([fwd[1], -fwd[0]])
            g = (float(a @ right) / 9.81, float(a @ fwd) / 9.81)
            self._accel_car_g = (
                GMETER_EMA * g[0] + (1.0 - GMETER_EMA) * self._accel_car_g[0],
                GMETER_EMA * g[1] + (1.0 - GMETER_EMA) * self._accel_car_g[1],
            )
            self._gmeter_trail.append(self._accel_car_g)
        self._prev_v = v
        self._trail.append((x, y))
        self._hero_yaw = yaw
        self._tele.update(
            speed=float(np.linalg.norm(v)),
            yaw_rate=float(qd[5]),
            s=s,
            meters=self._s_total,
            laps=int(self._s_total // tr.length) if self._s_total >= 0.0 else 0,
            n=lateral,
            sdf=tr.sdf_at(x, y),
            roll=roll,
            pitch=pitch,
            h=float(q[2]) - tr.height_at(x, y),
        )
        # observation for the driver
        route_yaw = math.atan2(float(tang[1]), float(tang[0]))
        o = self._obs
        o.x, o.y, o.yaw = x, y, yaw
        o.speed, o.yaw_rate = self._tele["speed"], float(qd[5])
        o.roll, o.pitch = roll, pitch
        o.s, o.lateral = s, lateral
        o.heading_error = (route_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        o.border_distance = self._tele["sdf"]
        o.curvature_ahead = tuple(tr.curvature_at(s + d) for d in TrackObservation.CURVATURE_LOOKAHEADS)

    # ── GUI ────────────────────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        t = self._tele
        ui.text(
            f"Track  {self.terrain.name}   {self.terrain.length:.0f} m loop, corridor {2 * self.terrain.half_width:.0f} m"
        )
        _changed, self.follow_camera = ui.checkbox("Follow camera", self.follow_camera)
        ui.text("Driver")
        for k, name in enumerate(DRIVERS):
            if k:
                ui.same_line()
            if ui.button(("[%s]" if self.driver.name == name else " %s ") % name):
                self._pp_speed = getattr(self.driver, "target_speed", self._pp_speed)
                self.driver = self._make_driver(name)
        self.driver.gui(ui)
        ui.text(
            f"  drive {self.drive_cmd:+.2f}   steer {self.steer_cmd:+.2f} (+ = left)   heading err {math.degrees(self._obs.heading_error):+.0f} deg"
        )
        ui.text(
            f"Speed {t['speed']:.2f} m/s (chassis)   no-slip {self.wheel_speed * self.spec.tire_R_outer:.2f} m/s   "
            f"yaw rate {math.degrees(t['yaw_rate']):+.0f} deg/s"
        )
        ui.text(f"Progress s = {t['s']:.0f} m   laps {t['laps']}   distance {t['meters']:.0f} m")
        ui.text(
            f"Lateral offset {t['n']:+.2f} m   {'IN corridor' if t['sdf'] < 0.0 else 'OFF corridor'} (border {t['sdf']:+.1f} m)"
        )
        ui.text(
            f"Roll {math.degrees(t['roll']):+.1f} deg   pitch {math.degrees(t['pitch']):+.1f} deg   chassis h {t['h']:.2f} m"
        )
        ui.separator()
        if not isinstance(self.driver, ManualDriver):
            ui.text(f"{self.driver.name} drives: move a steer / throttle slider to take over manually.")
        steer_before, throttle_before = self.steer_angle, self._target_wheel_speed
        super().gui(ui)  # steer / throttle sliders, CTIS, live tire loads
        if not isinstance(self.driver, ManualDriver) and (
            self.steer_angle != steer_before or self._target_wheel_speed != throttle_before
        ):
            self.driver = self._make_driver(ManualDriver.name)  # otherwise the policy overwrites the slider next frame
        self._draw_minimap(ui)
        self._draw_gmeter(ui)
        self._draw_hud_bars(ui)

    # ── Minimap / HUD (ported from example_vehicle_mppi_track) ─────────────────

    def _init_minimap(self) -> None:
        self._map_inner = self.terrain.left[:, :2]
        self._map_outer = self.terrain.right[:, :2]
        pts = np.vstack([self._map_inner, self._map_outer])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        extent = (hi - lo) * (1.0 + 2.0 * MINIMAP_PAD)
        self._map_center = 0.5 * (lo + hi)
        self._map_scale = MINIMAP_SIZE / max(float(extent[0]), float(extent[1]), 1e-6)

    @staticmethod
    def _hud_load_font(imgui, candidates):
        for path in candidates:
            if Path(path).is_file():
                try:
                    font = imgui.get_io().fonts.add_font_from_file_ttf(path, imgui.get_font_size())
                except Exception:
                    font = None
                if font is not None:
                    return font
        return None

    def _hud_get_font(self, imgui, bold=False):
        if not self._hud_font_tried:
            self._hud_font_tried = True
            self._hud_font = self._hud_load_font(imgui, HUD_FONT_CANDIDATES)
            self._hud_font_bold = self._hud_load_font(imgui, HUD_FONT_BOLD_CANDIDATES)
        font = (self._hud_font_bold or self._hud_font) if bold else self._hud_font
        return font if font is not None else imgui.get_font()

    def _hud_text_size(self, imgui, text, font_size, bold=False):
        imgui.push_font(self._hud_get_font(imgui, bold), font_size)
        try:
            return imgui.calc_text_size(text).x
        finally:
            imgui.pop_font()

    def _hud_caption(self, imgui, draw, x, y, text, align="left", span=0.0):
        font_size = 1.1 * imgui.get_font_size()
        if align == "right":
            x -= self._hud_text_size(imgui, text, font_size, bold=True)
        elif align == "center":
            x += 0.5 * (span - self._hud_text_size(imgui, text, font_size, bold=True))
        col = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        draw.add_text(self._hud_get_font(imgui, bold=True), font_size, imgui.ImVec2(x, y), col, text)

    @staticmethod
    def _hud_window_flags(imgui):
        return (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_inputs
            | imgui.WindowFlags_.no_nav
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_saved_settings
        )

    def _hud_window(self, imgui, name, x0, y0, w, h, body):
        imgui.set_next_window_pos(imgui.ImVec2(x0, y0))
        imgui.set_next_window_size(imgui.ImVec2(w, h))
        imgui.set_next_window_bg_alpha(0.45)
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 8.0)
        try:
            visible = imgui.begin(name, None, self._hud_window_flags(imgui))[0]
            try:
                if visible:
                    body()
            finally:
                imgui.end()
        finally:
            imgui.pop_style_var()

    def _draw_minimap(self, imgui):
        if not self._minimap_ok:
            return
        try:
            viewport = imgui.get_main_viewport()
            x0 = viewport.pos.x + viewport.size.x - MINIMAP_SIZE - MINIMAP_MARGIN
            y0 = viewport.pos.y + viewport.size.y - MINIMAP_SIZE - MINIMAP_MARGIN
            self._hud_window(
                imgui,
                "##track_minimap",
                x0,
                y0,
                MINIMAP_SIZE,
                MINIMAP_SIZE,
                lambda: self._draw_minimap_contents(imgui, x0, y0),
            )
        except Exception:
            self._minimap_ok = False  # degrade silently on missing imgui API

    def _draw_minimap_contents(self, imgui, x0, y0):
        cx = x0 + 0.5 * MINIMAP_SIZE
        cy = y0 + 0.5 * MINIMAP_SIZE
        mx, my = float(self._map_center[0]), float(self._map_center[1])
        s = self._map_scale

        def to_px(p):
            return imgui.ImVec2(cx + (p[0] - mx) * s, cy - (p[1] - my) * s)

        draw = imgui.get_window_draw_list()
        gray = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.55, 0.55, 0.6, 0.9))
        blue = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.66, 1.0, 0.9))
        orange = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.55, 0.1, 1.0))
        cp_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.3, 0.5, 1.0, 0.8))
        if self._minimap_boundary_px is None or self._minimap_boundary_px[0] != (x0, y0):
            loops = [[to_px(p) for p in poly] for poly in (self._map_inner, self._map_outer)]
            cps = [to_px((c["x"], c["y"])) for c in self.terrain.checkpoints]
            self._minimap_boundary_px = ((x0, y0), loops, cps)
        for loop in self._minimap_boundary_px[1]:
            draw.add_polyline(loop, gray, imgui.ImDrawFlags_.closed, 1.5)
        for p in self._minimap_boundary_px[2]:
            draw.add_circle_filled(p, 2.0, cp_col)
        if len(self._trail) >= 2:
            step = max(1, len(self._trail) // 600)
            trail = list(self._trail)[::step]
            if trail[-1] != self._trail[-1]:
                trail.append(self._trail[-1])
            draw.add_polyline([to_px(p) for p in trail], blue, imgui.ImDrawFlags_.none, 1.5)
        car = to_px((float(self._pose[0]), float(self._pose[1])))
        tick = 9.0
        tip = imgui.ImVec2(car.x + tick * math.cos(self._hero_yaw), car.y - tick * math.sin(self._hero_yaw))
        draw.add_line(car, tip, orange, 2.0)
        draw.add_circle_filled(car, 4.0, orange)
        self._hud_caption(imgui, draw, x0 + 8.0, y0 + 6.0, "TRACK")
        self._hud_caption(
            imgui,
            draw,
            x0 + MINIMAP_SIZE - 8.0,
            y0 + MINIMAP_SIZE - 26.0,
            f"LAP {self._tele['laps'] + 1}",
            align="right",
        )

    def _draw_gmeter(self, imgui):
        if not self._hud_ok:
            return
        try:
            viewport = imgui.get_main_viewport()
            mm_x0 = viewport.pos.x + viewport.size.x - MINIMAP_SIZE - MINIMAP_MARGIN
            x0 = mm_x0 - HUD_GAP - GMETER_SIZE
            y0 = viewport.pos.y + viewport.size.y - GMETER_SIZE - MINIMAP_MARGIN
            self._hud_window(
                imgui,
                "##hud_gmeter",
                x0,
                y0,
                GMETER_SIZE,
                GMETER_SIZE,
                lambda: self._draw_gmeter_contents(imgui, x0, y0),
            )
        except Exception:
            self._hud_ok = False

    def _draw_gmeter_contents(self, imgui, x0, y0):
        cx = x0 + 0.5 * GMETER_SIZE
        cy = y0 + 0.5 * GMETER_SIZE
        radius = 0.5 * GMETER_SIZE - 14.0
        ppg = radius / GMETER_G_EDGE
        draw = imgui.get_window_draw_list()
        cross = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.75, 0.75, 0.8, 0.8))
        ring = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.6, 0.6, 0.65, 0.35))
        red = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.95, 0.15, 0.15, 1.0))
        trail_col = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.95, 0.35, 0.35, 0.35))
        center = imgui.ImVec2(cx, cy)
        for g in (1.0, 2.0):
            draw.add_circle(center, g * ppg, ring, 48, 1.0)
        draw.add_line(imgui.ImVec2(cx - radius, cy), imgui.ImVec2(cx + radius, cy), cross, 1.0)
        draw.add_line(imgui.ImVec2(cx, cy - radius), imgui.ImVec2(cx, cy + radius), cross, 1.0)

        def to_dot(a_right, a_fwd):
            dx, dy = a_right * ppg, -a_fwd * ppg
            mag = math.hypot(dx, dy)
            if mag > radius:
                dx *= radius / mag
                dy *= radius / mag
            return imgui.ImVec2(cx + dx, cy + dy)

        for a_right, a_fwd in self._gmeter_trail:
            draw.add_circle_filled(to_dot(a_right, a_fwd), 2.0, trail_col)
        a_right, a_fwd = self._accel_car_g
        draw.add_circle_filled(to_dot(a_right, a_fwd), 5.0, red)
        self._hud_caption(imgui, draw, x0, y0 + 6.0, "ACCELERATION", align="center", span=GMETER_SIZE)
        self._hud_caption(
            imgui,
            draw,
            x0 + GMETER_SIZE - 8.0,
            y0 + GMETER_SIZE - 26.0,
            f"{math.hypot(a_right, a_fwd):.1f} G",
            align="right",
        )

    def _draw_hud_bars(self, imgui):
        if not self._hud_ok:
            return
        try:
            viewport = imgui.get_main_viewport()
            mm_x0 = viewport.pos.x + viewport.size.x - MINIMAP_SIZE - MINIMAP_MARGIN
            x0 = mm_x0 - HUD_GAP - GMETER_SIZE - HUD_GAP - HUD_PANEL_W
            y0 = viewport.pos.y + viewport.size.y - GMETER_SIZE - MINIMAP_MARGIN
            self._hud_window(
                imgui,
                "##hud_bars",
                x0,
                y0,
                HUD_PANEL_W,
                GMETER_SIZE,
                lambda: self._draw_hud_bars_contents(imgui, x0, y0, HUD_PANEL_W),
            )
        except Exception:
            self._hud_ok = False

    def _draw_hud_bars_contents(self, imgui, x0, y0, width):
        throttle = max(0.0, min(1.0, self.drive_cmd))
        brake = max(0.0, min(1.0, -self.drive_cmd))
        speed_kmh = self._tele["speed"] * 3.6
        draw = imgui.get_window_draw_list()
        white = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        outline = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.7, 0.7, 0.75, 0.5))
        blue = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.0, 0.66, 1.0, 0.95))
        red = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.9, 0.15, 0.15, 0.95))
        pad = 12.0
        bx = x0 + pad
        bw = width - 2.0 * pad
        label = f"{int(round(speed_kmh))} km/h"
        font_size = 2.0 * imgui.get_font_size()
        cap_gap = 14.0
        num_w = self._hud_text_size(imgui, label, font_size, bold=True)
        cap_w = self._hud_text_size(imgui, "SPEED", 1.1 * imgui.get_font_size(), bold=True)
        gx = x0 + 0.5 * (width - cap_w - cap_gap - num_w)
        self._hud_caption(imgui, draw, gx, y0 + 14.0, "SPEED")
        draw.add_text(
            self._hud_get_font(imgui, bold=True), font_size, imgui.ImVec2(gx + cap_w + cap_gap, y0 + 8.0), white, label
        )
        bar_h = 14.0
        seg_gap = 3.0
        seg_w = (bw - (HUD_BAR_SEGMENTS - 1) * seg_gap) / HUD_BAR_SEGMENTS
        # No friction brake in this vehicle: the red bar is the reverse half of the throttle axis.
        for label_y, frac, fill, name in ((y0 + 52.0, throttle, blue, "THROTTLE"), (y0 + 100.0, brake, red, "REVERSE")):
            self._hud_caption(imgui, draw, x0 + pad, label_y, name)
            y = label_y + 1.2 * imgui.get_font_size() + 4.0
            for i in range(HUD_BAR_SEGMENTS):
                sx = bx + i * (seg_w + seg_gap)
                draw.add_rect(imgui.ImVec2(sx, y), imgui.ImVec2(sx + seg_w, y + bar_h), outline, 1.0)
                lit = max(0.0, min(1.0, frac * HUD_BAR_SEGMENTS - i))
                if lit > 0.0:
                    draw.add_rect_filled(imgui.ImVec2(sx, y), imgui.ImVec2(sx + seg_w * lit, y + bar_h), fill, 1.0)

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        if self.follow_camera:
            set_follow_camera(self.viewer, self._pose)
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, colors=(1.0, 0.45, 0.0))
        self.viewer.log_lines("bead_spokes", self._spoke_start_zu, self._bead_pos_zu, colors=(1.0, 0.90, 0.1))
        wp.launch(
            terrain_contact_spikes,
            dim=_dw._N_TIRES * self._n_nodes,
            inputs=[
                self.terrain_scm.node_f,
                self.state_0.particle_q,
                2.0e-4,
                self._contact_line_s,
                self._contact_line_e,
            ],
            device="cuda:0",
        )
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, colors=(0.0, 1.0, 1.0))
        for i, (points, indices) in enumerate(self._boundary_ribbons):
            self.viewer.log_mesh(f"/track/boundary_{i}", points, indices, color=(1.0, 1.0, 1.0), backface_culling=False)
        traj = self.driver.trajectory(self._obs)
        if traj is not None and len(traj) == TRAJECTORY_SAMPLES:
            self._update_arrow(np.asarray(traj, dtype=np.float64))
            self._arrow_visible = True
        elif self._arrow_visible:
            self._arrow_visible = False
        self.viewer.log_mesh(
            "/track/intended_path",
            self._arrow_points,
            self._arrow_indices,
            color=(0.1, 0.9, 0.3),
            backface_culling=False,
            hidden=not self._arrow_visible,
        )
        self.viewer.end_frame()

    @staticmethod
    def _arrow_mesh_indices(k: int) -> np.ndarray:
        """Triangles of the arrow: quad strip over vertices [0..k) left / [k..2k) right, head = last 3."""
        i = np.arange(k - 1)
        shaft = np.concatenate(
            [np.stack([i, i + 1, k + i], axis=1), np.stack([i + 1, k + i + 1, k + i], axis=1)], axis=0
        )
        head = np.array([[2 * k, 2 * k + 1, 2 * k + 2]])
        return np.concatenate([shaft, head], axis=0).flatten().astype(np.int32)

    def _update_arrow(self, xy: np.ndarray) -> None:
        """Drape the (K, 2) path on the terrain as a ribbon ending in a triangular head."""
        t = self.terrain
        d = np.gradient(xy, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-d[:, 1], d[:, 0]], axis=1)
        z = np.array([t.height_at(float(p[0]), float(p[1])) + _ARROW_LIFT for p in xy])
        k = len(xy)
        h = self._arrow_host
        h[:k, :2] = xy + normal * _ARROW_HALF_WIDTH
        h[k : 2 * k, :2] = xy - normal * _ARROW_HALF_WIDTH
        h[:k, 2] = z
        h[k : 2 * k, 2] = z
        tip_base = xy[-1]
        fwd = d[-1]
        side = normal[-1]
        h[2 * k, :2] = tip_base + fwd * _ARROW_HEAD_LENGTH
        h[2 * k + 1, :2] = tip_base + side * _ARROW_HEAD_HALF_WIDTH
        h[2 * k + 2, :2] = tip_base - side * _ARROW_HEAD_HALF_WIDTH
        h[2 * k :, 2] = z[-1]
        self._arrow_points.assign(h)

    def _ribbon(self, poly: np.ndarray, half_width: float, lift: float, device: str) -> tuple[wp.array, wp.array]:
        """Closed quad-strip mesh along ``poly`` (N, 3): ``2 * half_width`` wide, ``lift`` above the terrain."""
        t = self.terrain
        xy = poly[:, :2].astype(np.float64)
        d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-d[:, 1], d[:, 0]], axis=1) * half_width
        z = np.array([t.height_at(float(p[0]), float(p[1])) + lift for p in xy])
        left = np.column_stack([xy + normal, z])
        right = np.column_stack([xy - normal, z])
        points = np.concatenate([left, right], axis=0).astype(np.float32)  # [0..N) left edge, [N..2N) right edge
        n = len(xy)
        i = np.arange(n)
        j = (i + 1) % n
        tris = np.concatenate([np.stack([i, j, n + i], axis=1), np.stack([j, n + j, n + i], axis=1)], axis=0)
        return wp.array(points, dtype=wp.vec3, device=device), wp.array(
            tris.flatten().astype(np.int32), dtype=wp.int32, device=device
        )

    # ── Tests ──────────────────────────────────────────────────────────────────

    def _print_diag(self, fps: float = 0.0) -> None:
        """Base per-tire lines plus the command-vs-motion numbers that separate a command problem
        (wheel speed not reaching its target) from a traction problem (chassis much slower than
        the no-slip speed)."""
        super()._print_diag(fps)
        t = self._tele
        v_noslip = self.wheel_speed * self.spec.tire_R_outer
        slip = 1.0 - t["speed"] / v_noslip if v_noslip > 0.05 else 0.0
        print(
            f"  drive={self.drive_cmd:+.2f} steer={self.steer_cmd:+.2f} ({self.driver.name})  "
            f"axle {self.wheel_speed:.1f}/{self._target_wheel_speed:.1f} rad/s  no-slip {v_noslip:.2f} m/s  "
            f"chassis {t['speed']:.2f} m/s  slip {100 * slip:.0f} %  s={t['s']:.0f} m  n={t['n']:+.2f} m  "
            f"roll={math.degrees(t['roll']):+.0f} pitch={math.degrees(t['pitch']):+.0f} deg  h={t['h']:.2f} m"
        )

    def test_post_step(self) -> None:
        super().test_post_step()
        roll, pitch = quat_roll_pitch(self._pose)
        if abs(roll) > math.radians(60.0) or abs(pitch) > math.radians(60.0):
            raise AssertionError(
                f"vehicle overturned: roll {math.degrees(roll):.0f} deg pitch {math.degrees(pitch):.0f} deg at t={self._t:.1f} s"
            )

    def test_final(self) -> None:
        super().test_final()
        t = self._tele
        assert t["meters"] > 10.0, f"FAIL: {self.driver.name} advanced only {t['meters']:.1f} m along the route"
        assert t["sdf"] < 1.0, (
            f"FAIL: vehicle left the corridor: {t['sdf']:.1f} m outside the border (lateral offset {t['n']:+.1f} m)"
        )
        print(
            f"[PASS] {self.terrain.name} / {self.driver.name}: {t['meters']:.0f} m along the route, lateral offset {t['n']:+.2f} m, {t['speed']:.1f} m/s at the end"
        )

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = _dw.Example.create_parser()
        parser.add_argument(
            "--terrain",
            type=str,
            default="test_03_fourier_hard",
            help="Terrain name under examples/ancf/assets/terrain/ (newton-terrain-tool output).",
        )
        parser.add_argument(
            "--driver",
            type=str,
            default="pure_pursuit",
            choices=sorted(DRIVERS),
            help="Policy driving the car (TrackDriver subclass).",
        )
        parser.add_argument("--speed", type=float, default=4.0, help="pure_pursuit cruise speed [m/s].")
        parser.add_argument(
            "--a-lat",
            type=float,
            default=2.5,
            help="pure_pursuit lateral-acceleration cap for bends [m/s^2]; 0 = no cap (run flat out).",
        )
        parser.add_argument(
            "--max-wheel-speed",
            type=float,
            default=None,
            help="Throttle slider limit [rad/s]; default = the vehicle asset's maxWheelSpeed (its rated top speed).",
        )
        parser.add_argument(
            "--mujoco-terrain-cell",
            type=float,
            default=_MJ_TERRAIN_CELL,
            help="Cell of the block-averaged hfield MuJoCo collides with [m]; the tires use the full-resolution grid.",
        )
        parser.set_defaults(num_frames=1200)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
