# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Vehicle on 4 ANCF FEM tires over a terrain bundle: drive it yourself or follow the bundle's track.

Extends the shared VehicleSimulation runtime with a terrain bundle (``newton_terrain_tool.field/2``: the synthetic
``boulders`` / ``craters`` of newton-terrain-tool or a RELLIS-3D reconstruction ``rellis_0000N`` of
newton-rellis-3d-tool). The FEM tires contact the fine grid (``TerrainSCM``), the rigid parts (rims,
arms, chassis) a MuJoCo heightfield on the same grid. The vehicle spawns at pose 0 of the bundle's
reference track (position and heading), at the origin heading +x when the bundle has none.

* ``--controller manual``: the base example's steer / throttle sliders and CTIS panel, plus the
  driving keys I / K (throttle up / down), J / L (steer left / right, self-centring), SPACE (stop).
  The camera is free by default: the viewer's WASD / QE + mouse gaming controls (``--camera``).
* ``--controller track``: pure pursuit on the reference track gives the yaw-rate command; a
  kingpin-steered vehicle turns it into a steer angle through the bicycle model, a skid-steered
  one into brake levers through the drive kinematics (:class:`SkidPlant`, ``--yaw-authority``;
  no PID: the wheel motors hold the wheel speeds). The speed is capped in bends
  (``--a-lat``). Every stretch of the track bending one way is a "turn" with yaw IAE / RMS /
  time-to-band; ``--switch-pressure-turns`` / ``--switch-stage-turns`` alternate the condition;
  ``--log file.npz`` stores the signals and the per-turn metrics. Any driving key or slider hands over to
  manual.
* Terrain stage ``w``: ``heights = w * I_N`` applied live from the UI - the ANCF terrain grid, the
  MuJoCo collider and the terrain mesh all rescale, then the vehicle is re-dropped at
  the spawn pose. "Reset vehicle" restores the at-rest snapshot taken at construction.
* HUD overlay: yaw-rate / speed strip charts (command grey, measured orange), drive bars, per-turn
  IAE bars, minimap with the arena, the corridor and the trail. Scene: orange arrow = the arc the
  command describes, cyan arrow = the arc the chassis is on, blue trail, yellow reference track.
* Camera modes follow / top / free, sun / ambient sliders, debug overlays for the MuJoCo collider
  grid and its terrain contacts, and a watchdog that dumps the last frames and resets the vehicle
  on a simulation fault (non-finite chassis state, runaway tread node or chassis speed).

Command: python -m newton.examples vehicle_terrain --vehicle-asset <usd> [--terrain boulders | craters | rellis_00000]
         [--controller manual | track] [--difficulty 1.0]
(docker/config/05_vehicle_terrain.json)
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf import _vehicle_config as vehicle_config
from newton.examples.ancf._capture_utils import snapshot_arrays
from newton.examples.ancf._field_task import FieldTerrain
from newton.examples.ancf._terrain_common import (
    FAR_BELOW,
    SPAWN_CLEARANCE,
    alloc_chassis_buffers,
    grid_triangles,
    park_mjcf_plane,
    quat_roll_pitch,
    quat_yaw,
    read_chassis_q,
    read_chassis_qd,
    require_vehicle_asset,
    rock_clearance_height,
    set_follow_camera,
    terrain_contact_spikes,
    terrain_mesh_points,
    wheel_speed_rate,
)
from newton.examples.ancf._vehicle_simulation import VehicleSimulation
from newton.examples.ancf._vehicle_usd import find_body
from newton.solvers import TerrainSCM


def wrap_pi(angle: float) -> float:
    """``angle`` [rad] wrapped into [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@wp.kernel
def _record_wheel_diagnostics(
    dofs: wp.array[int],
    signs: wp.array[float],
    velocity: wp.array[float],
    target: wp.array[float],
    effort: wp.array[float],
    row: int,
    output: wp.array3d[float],
):
    wheel = wp.tid()
    dof = dofs[wheel]
    output[row, wheel, 0] = signs[wheel] * velocity[dof]
    output[row, wheel, 1] = signs[wheel] * target[dof]
    output[row, wheel, 2] = signs[wheel] * effort[dof]


MODES = ("manual", "track")

# MuJoCo hfield collider cell. The vendored mujoco_warp collects up to MJ_MAXHFPRISM = 128 prisms
# per geom (raised from MuJoCo's 50): the 1.69 x 0.46 m chassis box yawed 45 deg covers 8 x 8 cells
# of the 0.25 m grid = 128 prisms, so the rigid parts can use the same fine grid as the tires.
_MJ_TERRAIN_CELL = 0.25  # [m]
# Simulation-fault detection (the vehicle is reset, the state is garbage): tread nodes faster
# than four times the tire surface speed at the asset's maxWheelSpeed (self._fault_node_speed), or a
# rigid body faster than the car can drive, or any non-finite chassis state.
_FAULT_BODY_SPEED = 30.0  # [m/s]
# Actuator ramps, for the driving keys and the controller alike: full lock in 0.5 s (a fast human at
# the wheel), full throttle in 2 s.
_KEY_STEER_TIME = 0.5  # [s]
_KEY_THROTTLE_TIME = 2.0  # [s]
_TERRAIN_COLOR = (0.45, 0.43, 0.40)  # dry ground, grey-brown
_END_MARGIN = 6.0  # [m] before the last sample of an open track: the vehicle stops here
_CORRIDOR_HALF_WIDTH = 4.0  # [m] corridor around a reference track that has none baked (a recorded drive)
_OOB_MARGIN = 3.0  # [m] inside the arena rectangle before the vehicle is put back at the spawn
_CURVATURE_HORIZON = 12.0  # [m] past the lookahead the speed cap looks for the tightest bend
_SPEED_KP = 0.6  # [1/s] proportional speed term of the kingpin drive (the skid drive has none)

# Scene overlays: draped arrows for the commanded / measured arc (ribbon half width, head, lift [m]),
# the driven trail (segments, lift above the terrain [m])
_ARROW_SAMPLES = 24
_ARROW_HALF_WIDTH = 0.15
_ARROW_HEAD_HALF_WIDTH = 0.5
_ARROW_HEAD_LENGTH = 1.2
_ARROW_LIFT = 0.35
_ARROW_HORIZON = 4.0  # [s] of motion shown ahead of the vehicle
_TRAIL3D_LEN = 900  # segments (15 s at one point per frame)
_TRAIL_LIFT = 0.25
_TRACE_LEN = 240  # frames of strip chart (4 s)

# ── HUD (imgui overlay, bottom row) ──────────────────────────────────────────
HUD_H = 150.0
MINIMAP_SIZE = HUD_H
MINIMAP_MARGIN = 12.0
MINIMAP_PAD = 0.08
MINIMAP_TRAIL_MAX = 3600
HUD_GAP = 10.0
CONTROL_W = 520.0
HUD_BAR_SEGMENTS = 24
HUD_FONT_CANDIDATES = (
    str(Path.home() / ".local/share/fonts/JetBrainsMono-Regular.ttf"),
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
)
HUD_FONT_BOLD_CANDIDATES = (
    str(Path.home() / ".local/share/fonts/JetBrainsMono-Bold.ttf"),
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
)


def terrain_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "terrain")


@wp.kernel
def _mj_contact_points(
    pos: wp.array[wp.vec3],
    geom: wp.array2d[int],
    nacon: wp.array[int],
    hf_geom: int,
    far: float,
    out: wp.array[wp.vec3],
):
    """MuJoCo contact points involving the terrain collider; the unused slots are parked far below."""
    i = wp.tid()
    if i < nacon[0] and (geom[i, 0] == hf_geom or geom[i, 1] == hf_geom):
        out[i] = pos[i]
    else:
        out[i] = wp.vec3(0.0, 0.0, far)


@wp.kernel
def _diag_reduce(node_f: wp.array[wp.vec3], node_xd: wp.array[wp.vec3], out: wp.array[float]):
    """out = [max |terrain force|, max |node velocity|, number of nodes in terrain contact]."""
    i = wp.tid()
    f = wp.length(node_f[i])
    wp.atomic_max(out, 0, f)
    wp.atomic_max(out, 1, wp.length(node_xd[i]))
    if f > 0.0:
        wp.atomic_add(out, 2, 1.0)


# ── Drive command ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class SkidCommand:
    """Normalized command: ``throttle`` = mean wheel speed / max_wheel_speed, ``turn`` = steer (+ = left):
    the brake lever of a skid steer, the steer angle / max_steer of a kingpin-steered vehicle."""

    throttle: float = 0.0
    turn: float = 0.0

    def clipped(self) -> SkidCommand:
        return SkidCommand(max(-1.0, min(1.0, float(self.throttle))), max(-1.0, min(1.0, float(self.turn))))


@dataclasses.dataclass
class SkidPlant:
    """Drive kinematics of a skid-steered vehicle (numbers from the vehicle asset): the braked side slows by
    the lever travel, the other keeps the throttle speed (``_vehicle_usd._drive_skid``), so the yaw rate is
    ``authority * w r turn / (2 b)`` and the mean speed ``w r (1 - |turn| / 2)``. No feedback: the wheel
    motors hold the wheel speeds and pure pursuit closes the loop on the path."""

    half_track: float  # [m]
    r_roll: float  # [m]
    max_wheel_speed: float  # [rad/s]
    v_min_ff: float = 0.5  # [m/s] below this the yaw mapping uses v_min_ff (no division by ~0)
    # measured yaw rate / kinematic yaw rate for a given lever (1 = no-slip kinematics). A four-wheel
    # skid steer scrubs all four tires sideways while turning, so its authority is well below 1.
    yaw_authority: float = 1.0

    @property
    def v_max(self) -> float:
        return self.r_roll * self.max_wheel_speed

    def yaw_scale(self, v_ref: float) -> float:
        """Yaw rate at full lever for throttle speed ``v_ref`` [rad/s]: ``v / (2 b)``."""
        return max(abs(v_ref), self.v_min_ff) / (2.0 * self.half_track)

    def command(self, v_cmd: float, yaw_cmd: float) -> SkidCommand:
        """Levers for ``v_cmd`` [m/s] and ``yaw_cmd`` [rad/s], solved jointly: the unbraked side runs at
        ``w r = v_cmd / (1 - |l| / 2)`` (so the mean of both sides is ``v_cmd``) and the yaw rate is
        ``authority * w r * l / (2 b)``; with ``q = yaw_cmd / (authority * v_cmd / (2 b))`` that gives
        ``l / (1 - |l| / 2) = q``  ->  ``l = q / (1 + |q| / 2)``."""
        q = yaw_cmd / (self.yaw_scale(v_cmd) * max(self.yaw_authority, 0.05))
        turn = max(-1.0, min(1.0, q / (1.0 + 0.5 * abs(q))))
        w = v_cmd / (self.r_roll * max(1.0 - 0.5 * abs(turn), 0.5))
        return SkidCommand(w / self.max_wheel_speed, turn).clipped()


# ── Example ───────────────────────────────────────────────────────────────────


class VehicleTerrain(VehicleSimulation):
    """Vehicle + 4 ANCF tires on a terrain bundle: manual driving or track following, live stage."""

    # MuJoCo contact budget for the hfield pairs (rims, arm capsules, chassis boxes; up to
    # MJ_MAXHFPRISM = 128 candidate prisms per geom, of which only the ones under the part touch).
    _NCONMAX = 512
    _NJMAX = 2048

    CAMERA_MODES = ("follow", "top", "free")

    def __init__(self, viewer=None, args=None):
        if args is None:
            args = argparse.Namespace()
        self._args = args
        self._test = bool(getattr(args, "test", False))
        w0 = 0.4 if self._test else float(args.difficulty)
        self.terrain = FieldTerrain(terrain_dir(), args.terrain, w0)
        t = self.terrain
        print(
            f"[TERRAIN] {t.name}: {t.ncol}x{t.nrow} px  cell {t.cell} m  I_N max {t.max_h:.2f} m  "
            f"grid {2 * t.hx:.0f}x{2 * t.hy:.0f} m  stage w = {t.w:.2f}"
            + (
                f"  track {t.track.length:.0f} m ({'closed' if t.track.closed else 'open'}, {t.track.n_turns} turns)"
                if t.track is not None
                else "  no reference track"
            )
        )
        # Flat plane off. The vehicle is built at the terrain's start pose, lifted above the rocks
        # under it: the base class's graph-capture warm-up steps this pose, and MuJoCo keeps
        # warm-start data the base does not restore, so it must be a legitimate drop, not a car
        # buried in a boulder. Resets restore this snapshot. The vehicle asset is needed before the
        # base builds (spindle footprint for the lift).
        self.vehicle = require_vehicle_asset(args)
        self.spawn = t.start_pose
        self._snapshot_z = self._lift_at_spawn()
        args.ground_z = FAR_BELOW
        args.world_z_offset = self._snapshot_z
        args.spawn_pose = self.spawn
        print(
            f"[TERRAIN] spawn ({self.spawn[0]:+.1f}, {self.spawn[1]:+.1f}) m heading {math.degrees(self.spawn[2]):+.0f} deg"
        )
        super().__init__(viewer, args)

        # Throttle limit = the vehicle asset's maxWheelSpeed unless --max-wheel-speed overrides it;
        # wheel-speed ramp capped at the traction limit mu g / r_roll.
        if args.max_wheel_speed is not None:
            self.spec = dataclasses.replace(self.spec, max_wheel_speed=float(args.max_wheel_speed))
        mu = float(getattr(args, "mu", vehicle_config.MU))
        self._wheel_speed_rate = wheel_speed_rate(
            mu, self.vehicle.r_roll, vehicle_config.GRAVITY, vehicle_config.FRAME_DT, vehicle_config.WHEEL_SPEED_RATE
        )
        # Watchdog threshold (diagnostic, not physics) [m/s]: a tread node on a rolling tire moves at
        # up to v_chassis + omega R = 2 omega R at full throttle without slip, so 2x that is the fault line.
        self._fault_node_speed = 4.0 * float(self.spec.max_wheel_speed) * float(self.spec.tire_R_outer)
        # + steer = left for the keys and the controller; skid steer takes a lever command in [-1, 1].
        self._skid = self.vehicle.steering == "skid"
        self._steer_sign = 1.0 if self._kingpin_axis_z > 0.0 else -1.0
        self._steer_range = 1.0 if self._skid else float(self.spec.max_steer)
        self.camera_mode = str(args.camera)
        self.top_view_height = 30.0  # [m] above the car; 30 m at 65 deg fov shows ~38 m across

        # ── Device buffers ──
        dev = "cuda:0"
        self._chassis_body = find_body(self.model, "chassis")
        self._chassis_q, self._chassis_qd = alloc_chassis_buffers(dev)
        self._field_dev = wp.array(t.field, dtype=float, device=dev)
        # terrain mesh for the viewer: one vertex per grid node, a plain terrain colour
        self._terrain_points = wp.zeros(t.nrow * t.ncol, dtype=wp.vec3, device=dev)
        # Only the cells the terrain bundle marks as worth drawing (files.draw_png); the collider and
        # the tire contact use the whole grid regardless.
        draw_mask = None if t.draw_mask is None else t.draw_mask[:-1, :-1]
        if draw_mask is not None:
            print(f"[TERRAIN] drawing {int(draw_mask.sum())} of {draw_mask.size} cells (files.draw_png)")
        self._terrain_indices = wp.array(grid_triangles(t.nrow, t.ncol, draw_mask), dtype=wp.int32, device=dev)
        # MuJoCo collider overlay (the viewer's own "show collision" draws the Newton shape at the
        # stage it was built with; this one follows the stage).
        self.show_collider = False
        self._logged_show_collider, self._collider_logged = False, False
        self._hc_dev = wp.array(self._hc, dtype=float, device=dev)
        self._collider_points = wp.zeros(self._hc.shape[0] * self._hc.shape[1], dtype=wp.vec3, device=dev)
        self._collider_indices = wp.array(
            grid_triangles(self._hc.shape[0], self._hc.shape[1]), dtype=wp.int32, device=dev
        )
        # Watchdog: three device-reduced scalars per frame, a rolling history dumped on the first
        # fault so a divergence can be traced back, not just noticed.
        self._diag_dev = wp.zeros(3, dtype=float, device=dev)
        self._history = deque(maxlen=90)
        self._fault = ""
        self.fault_count = 0
        self.show_reference_track = True
        self._reference_track_starts = None
        self._reference_track_ends = None
        self._refresh_terrain_mesh()
        # MuJoCo contacts with the terrain collider (debug overlay)
        self.show_mj_contacts = False
        n_con = int(self.solver.mjw_data.naconmax)
        self._mj_contact_pts = wp.full(n_con, wp.vec3(0.0, 0.0, FAR_BELOW), dtype=wp.vec3, device=dev)
        self._mj_contact_colors = wp.full(n_con, wp.vec3(1.0, 0.1, 0.1), dtype=wp.vec3, device=dev)
        self._mj_contact_radii = wp.full(n_con, 0.08, dtype=wp.float32, device=dev)

        # MuJoCo hfield elevation data (normalised, one hfield = the collider) for live stage changes.
        mjm = self.solver.mjw_model
        if int(mjm.nhfield) != 1:
            raise ValueError(f"expected exactly one MuJoCo hfield (the collider), got {mjm.nhfield}")
        if int(mjm.nhfielddata) != self._hc_norm.size:
            raise ValueError("MuJoCo hfield data size does not match the collider grid")
        self._collider_shape = list(self.model.shape_label).index("terrain_collider")
        self._collider_hf_args = {
            "nrow": self._hc.shape[0],
            "ncol": self._hc.shape[1],
            "hx": 0.5 * (self._hc.shape[1] - 1) * self._hc_k * t.cell,
            "hy": 0.5 * (self._hc.shape[0] - 1) * self._hc_k * t.cell,
        }
        geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()[0]  # (ngeom,) for world 0
        hits = np.where(geom_to_shape == self._collider_shape)[0]
        if len(hits) != 1:
            raise ValueError(f"terrain_collider maps to {len(hits)} MuJoCo geoms")
        self._collider_geom = int(hits[0])
        self._apply_stage_to_collider()

        # ── Snapshot of the at-rest vehicle at the spawn pose, for resets ──
        wp.synchronize_device(dev)
        self._snapshot = self._take_snapshot()
        self._pose = self._read_chassis()
        self._chassis_vel = np.zeros(6, dtype=np.float32)
        self._w_pending = t.w
        self._constructed = False

        # ── Controller ──
        a = args
        self.plant = SkidPlant(
            half_track=float(self.spec.half_track),
            r_roll=float(self.vehicle.r_roll),
            max_wheel_speed=float(self.spec.max_wheel_speed),
            yaw_authority=float(a.yaw_authority),
        )
        if self._skid:
            print(
                f"[TERRAIN] skid steering: brake levers from the drive kinematics, yaw authority {self.plant.yaw_authority:.3f}"
            )
        else:
            print(
                f"[TERRAIN] kingpin steering: bicycle-model pure pursuit, wheelbase {2.0 * self.spec.half_wheelbase:.2f} m"
            )
        self._mode = "manual"
        # ── Track following ──
        self.v_set = float(a.speed)  # [m/s]
        self.a_lat = float(a.a_lat)  # [m/s^2] speed cap in bends (0 = off)
        self.yaw_set = math.radians(25.0)  # [rad/s] manual default; _load_route: v* / bend radius
        self.lookahead_min = float(a.lookahead_min)  # [m]
        self.lookahead_time = float(a.lookahead_time)  # [s]
        self.lateral = 0.0  # [m] signed offset from the track (+ = left of it)
        self._half_width_arg = a.corridor_half_width
        self._half_width = float(a.corridor_half_width or t.corridor_half_width or _CORRIDOR_HALF_WIDTH)
        self._lap = 0
        self._prev_turn = -1
        self._s_track = 0.0  # arc length of the last nearest-point hit (windowed search)
        self._finished = False  # open track: the vehicle got within _END_MARGIN of the end and holds still
        self._load_route()
        self.settle_time = float(a.settle_time)
        self.band = float(a.band)
        self.switch_pressure_turns = int(a.switch_pressure_turns)
        self.switch_stage_turns = int(a.switch_stage_turns)
        self.p_levels = (float(a.p_high), float(a.p_low))  # [CTIS panel units]
        self.stage_levels = (float(a.stage_a), float(a.stage_b))
        self._p_idx = 0
        self._stage_idx = 0
        self.v_cmd = 0.0
        self.yaw_cmd = 0.0
        self._t_scn = 0.0
        self._turn = -1  # id of the current turn (-1 = settling / straight / outside the corridor)
        self._turn_t0 = 0.0
        self._turn_acc = None
        self.turns: list[dict] = []
        self.episode = 0
        # actuator slew = the driving keys': fair vs the human / key baseline
        self._steer_rate = vehicle_config.FRAME_DT / _KEY_STEER_TIME
        self._throttle_rate = vehicle_config.FRAME_DT / _KEY_THROTTLE_TIME
        self._cmd = SkidCommand()  # normalized (throttle, turn) actually applied (slew-limited)
        mode = "track" if self._test and t.track is not None else str(a.controller)
        if mode == "track" and t.track is None:
            print(f"[TERRAIN] '{t.name}' has no reference track: manual driving")
            mode = "manual"
        self._set_mode(mode)

        # ── Telemetry / HUD state ──
        self.v_fwd = 0.0
        self.yaw_rate = 0.0
        self._trail = deque(maxlen=MINIMAP_TRAIL_MAX)
        self._trace_yaw = deque(maxlen=_TRACE_LEN)
        self._trace_v = deque(maxlen=_TRACE_LEN)
        self._hud_ok = True
        self._hud_font = None
        self._hud_font_bold = None
        self._hud_font_tried = False
        self._init_minimap()
        # scene overlays: orange arrow = the arc the command (v*, yaw*) describes over the next
        # _ARROW_HORIZON s, cyan arrow = the arc the chassis is actually on, blue trail
        k = _ARROW_SAMPLES
        self._arrow_host = np.zeros((2 * k + 3, 3), dtype=np.float32)
        self._arrow_indices = wp.array(self._arrow_mesh_indices(k), dtype=wp.int32, device=dev)
        self._arrow_cmd = wp.zeros(2 * k + 3, dtype=wp.vec3, device=dev)
        self._arrow_meas = wp.zeros(2 * k + 3, dtype=wp.vec3, device=dev)
        self._arrow_cmd_visible = False
        self._arrow_meas_visible = False
        self._trail_s_host = np.full((_TRAIL3D_LEN, 3), FAR_BELOW, dtype=np.float32)
        self._trail_e_host = np.full((_TRAIL3D_LEN, 3), FAR_BELOW, dtype=np.float32)
        self._trail_s = wp.array(self._trail_s_host, dtype=wp.vec3, device=dev)
        self._trail_e = wp.array(self._trail_e_host, dtype=wp.vec3, device=dev)
        self._trail_i = 0
        self._trail_prev = None
        self._log_path = a.log
        self._log = {
            k: [] for k in ("t", "v_cmd", "yaw_cmd", "v", "yaw_rate", "throttle", "turn", "lat", "w", "p", "episode")
        }
        self._log_written = False
        # Buffer measured wheel signals on device; read once when saving the log.
        self._wheel_log = None
        self._wheel_log_count = 0
        if self._log_path:
            self._wheel_log = wp.zeros(
                (max(1, int(getattr(a, "num_frames", 0) or 600)), vehicle_config.N_TIRES, 3),
                dtype=float,
                device=dev,
            )

        if self._test and self._mode == "manual":
            self._target_wheel_speed = 3.0  # [rad/s] drive straight over the rocks
        self._constructed = True
        # scene light: sun elevation above the horizon and ambient level (GL renderer knobs)
        self.sun_elevation_deg = float(args.sun_elevation)
        # the sun azimuth is given relative to the vehicle's initial heading, so the scene is lit
        # the same way whichever way the track starts (RELLIS maps spawn at any yaw)
        self.sun_azimuth_deg = float(args.sun_azimuth) + math.degrees(quat_yaw(self._pose))
        self.ambient = float(args.ambient)
        if viewer is not None:
            # the free camera starts above and behind the vehicle looking along its heading (down
            # the reference track); from there the viewer's WASD / QE + mouse take over
            set_follow_camera(viewer, self._pose)
            self._update_camera()
            if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
                viewer.camera.fov = 65.0
            self._apply_light()
            self._register_hud()

    # ── Terrain hook (MJCF imported, ANCF solver built, nothing captured yet) ──

    def _on_car_builder(self, car: newton.ModelBuilder) -> None:
        t = self.terrain
        self.terrain_scm = TerrainSCM(
            heights=t.heights(),
            hx=t.hx,
            hy=t.hy,
            node_x0=self.ancf_model.node_x0,
            elem_nodes=self.ancf_model.elem_nodes,
            n_envs=vehicle_config.N_TIRES,
            patch_width=self.spec.tire_width,
            rigid=True,
            mu_rigid=self.ancf_solver.mu,
            origin=t.origin,
            device="cuda:0",
        )
        self.ancf_solver.terrain = self.terrain_scm

        # Hidden collider for rims / arms / chassis: the fine grid itself (cell 0.25 m, k = 1) now
        # that the vendored mujoco_warp collects 128 prisms per geom. Coarser cells are still
        # available (--mujoco-terrain-cell): block MAX is the rock envelope (rims then rest on
        # neighbouring rock tops between boulders and the car loses traction), block MEAN sits
        # below the rock tops (chassis sinks into rock); neither is right, the fine grid is.
        # newton.Heightfield normalises its data by the data's own min / max (min_z / max_z only
        # set the world mapping), so it is built from I_N (w = 1): normalised elevation =
        # hc / hc_max spanning [0, hc_max]. The stage is applied afterwards by writing
        # w * hc / hc_max into MuJoCo's elevation array (_apply_stage_to_collider).
        mj_cell = float(self._args.mujoco_terrain_cell)
        k = max(1, int(round(mj_cell / t.cell)))
        self._hc_mode = str(self._args.mujoco_terrain_agg)
        hc = t.downsample(t.field, k, self._hc_mode)
        self._hc = np.maximum(hc, 0.0).astype(np.float32)
        self._hc_k = k
        hc_max = float(max(hc.max(), 1.0e-3))
        self._hc_norm = (hc / hc_max).astype(np.float32).flatten()
        hx_c = 0.5 * (hc.shape[1] - 1) * k * t.cell
        hy_c = 0.5 * (hc.shape[0] - 1) * k * t.cell
        col_cfg = car.default_shape_cfg.copy()
        col_cfg.is_visible = False
        car.add_shape_heightfield(
            # coarse node (0, 0) coincides with fine node (0, 0) at (-hx, -hy) of the grid
            xform=wp.transform(wp.vec3(t.origin[0] - t.hx + hx_c, t.origin[1] - t.hy + hy_c, 0.0), wp.quat_identity()),
            heightfield=newton.Heightfield(
                data=np.maximum(hc, 0.0).astype(np.float32),
                nrow=hc.shape[0],
                ncol=hc.shape[1],
                hx=hx_c,
                hy=hy_c,
                min_z=0.0,
                max_z=hc_max,
            ),
            cfg=col_cfg,
            label="terrain_collider",
        )
        print(
            f"[TERRAIN] MuJoCo collider grid {hc.shape[1]}x{hc.shape[0]} at {k * t.cell:.2f} m (block {self._hc_mode} of the {t.cell} m grid)"
        )
        park_mjcf_plane(car)
        self._kingpin_axis_z, _ = self.vehicle.drive_axes(car)

    # ── Stage (live terrain relief) ────────────────────────────────────────────

    def set_difficulty(self, w: float) -> None:
        """Rescale the terrain to stage ``w`` everywhere it lives, then re-drop the car."""
        w = max(0.0, min(1.0, float(w)))
        t = self.terrain
        t.w = w
        self._w_pending = w
        self.terrain_scm.h0.assign(t.heights())
        self._apply_stage_to_collider()
        self._refresh_terrain_mesh()
        self._sync_viewer_collision_shape()
        self.reset_vehicle()

    def _sync_viewer_collision_shape(self) -> None:
        """Keep the Newton collider shape (what the viewer's own "show collision" draws) at the
        current stage. The viewer builds the heightfield mesh when it populates the model, so after
        updating the model it is asked to re-read it; set_model drops every example callback and
        logged object, so the side panel, the HUD and the terrain mesh are put back."""
        if self.viewer is None:
            return
        t = self.terrain
        hc_max = float(max(self._hc.max(), 1.0e-3))
        # data normalised by its own range -> hc / hc_max; world z = 0 + that * (w * hc_max) = w * hc
        self.model.shape_source[self._collider_shape] = newton.Heightfield(
            data=self._hc, min_z=0.0, max_z=max(t.w * hc_max, 1.0e-3), **self._collider_hf_args
        )
        self.viewer.set_model(self.model)
        self._terrain_dirty, self._collider_logged = True, False  # set_model dropped the logged meshes
        if self._constructed and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui, ex=self: ex.gui(ui), position="side")
            self._register_hud()

    def _apply_stage_to_collider(self) -> None:
        """MuJoCo elevation = w * hc / hc_max (the hfield spans [0, hc_max])."""
        self.solver.mjw_model.hfield_data.assign((self.terrain.w * self._hc_norm).astype(np.float32))

    def _refresh_terrain_mesh(self) -> None:
        t = self.terrain
        self._terrain_dirty = True  # render() re-logs the meshes
        starts, ends = t.reference_track_segments()
        if len(starts):
            if self._reference_track_starts is None:
                self._reference_track_starts = wp.array(starts, dtype=wp.vec3, device="cuda:0")
                self._reference_track_ends = wp.array(ends, dtype=wp.vec3, device="cuda:0")
            else:
                self._reference_track_starts.assign(starts)
                self._reference_track_ends.assign(ends)
        wp.launch(
            terrain_mesh_points,
            dim=(t.nrow, t.ncol),
            inputs=[self._field_dev, t.w, t.hx, t.hy, t.cell, t.origin[0], t.origin[1], self._terrain_points],
            device="cuda:0",
        )
        nr, nc = self._hc.shape
        hx_c = 0.5 * (nc - 1) * self._hc_k * t.cell
        hy_c = 0.5 * (nr - 1) * self._hc_k * t.cell
        # coarse node (0, 0) sits on fine node (0, 0): centre = origin - (hx, hy) + (hx_c, hy_c)
        wp.launch(
            terrain_mesh_points,
            dim=(nr, nc),
            inputs=[
                self._hc_dev,
                t.w,
                hx_c,
                hy_c,
                self._hc_k * t.cell,
                t.origin[0] - t.hx + hx_c,
                t.origin[1] - t.hy + hy_c,
                self._collider_points,
            ],
            device="cuda:0",
        )

    # ── Reset (snapshot + z shift) ─────────────────────────────────────────────

    def _take_snapshot(self) -> dict:
        a = self.ancf_solver
        return snapshot_arrays(
            {"node_x": a.node_x, "node_D": a.node_D, "s0_bq": self.state_0.body_q, "s0_jq": self.state_0.joint_q}
        )

    def _lift_at_spawn(self) -> float:
        """Lift of the tire bottoms so nothing starts inside a rock at the spawn pose: the highest
        rock under the four tire discs, or under the chassis / suspension footprint minus the belly
        clearance (rocks between the wheels can stand higher than the ones under the tires),
        plus the drop clearance [m]."""
        return rock_clearance_height(self.terrain, self.terrain.heights(), self.vehicle, self.spawn) + SPAWN_CLEARANCE

    def reset_vehicle(self) -> float:
        """Put the at-rest snapshot back at the spawn pose, lifted to clear the rocks at the
        current stage; velocities, internal forces and EAS parameters zeroed. Returns the lift."""
        self._s_track = 0.0  # the spawn is pose 0: progress along the track restarts
        lift = self._lift_at_spawn()
        dz = lift - self._snapshot_z
        snap = self._snapshot
        bq = snap["s0_bq"].copy()
        bq[:, 2] += dz
        jq = snap["s0_jq"].copy()
        jq[self.chassis_q0 + 2] += dz
        nx = snap["node_x"].copy()
        nx[:, 1] += dz  # ANCF is Y-up: a1 is the world z
        self.reset_state(nx, snap["node_D"], body_q=bq, joint_q=jq)
        self.terrain_scm.node_f.zero_()
        # controls to rest, then the same warm-up the constructor does
        self.steer_angle = 0.0
        self.wheel_speed = 0.0
        self._target_wheel_speed = 0.0
        self._update_controls()
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
        self._check_reset(nx)
        self._prescribe_beads()
        if self._gs_coupler is not None:
            self._gs_coupler.reset(self.state_0)
        self._update_viz_buffers()
        self._history.clear()
        self._pose = self._read_chassis()
        print(f"[TERRAIN] vehicle reset at the spawn pose, lift {lift:.2f} m, stage w = {self.terrain.w:.2f}")
        if self._constructed:
            self._on_reset()
        return lift

    def _check_reset(self, node_x_placed: np.ndarray) -> None:
        """Compare the restored bead nodes with where the MuJoCo spindles now expect them (the base
        diagnostic's drift): an inconsistent reset shows up here as metres, not as a NaN later."""
        xpos_all = self.solver.xpos.numpy()
        xquat_all = self.solver.xquat.numpy()
        smj = self._spindle_mj_arr.numpy()
        rest_np = self._bead_rest_np
        nn = self._n_nodes
        worst = 0.0
        for e in range(vehicle_config.N_TIRES):
            pos_zu = xpos_all[0, int(smj[e])]
            qm = xquat_all[0, int(smj[e])]
            qx, qy, qz, qw = float(qm[1]), float(qm[2]), float(qm[3]), float(qm[0])
            u = np.array([qx, qy, qz])
            r_mj = np.stack([rest_np[:, 2], rest_np[:, 0], rest_np[:, 1]], axis=1)
            r_rot = (
                r_mj * (qw * qw - u @ u)
                + 2.0 * (r_mj @ u)[:, None] * u[None, :]
                + 2.0 * qw * np.cross(np.broadcast_to(u, r_mj.shape), r_mj)
            )
            p_mj = r_rot + pos_zu
            expected = np.stack([p_mj[:, 1], p_mj[:, 2], p_mj[:, 0]], axis=1)
            placed = node_x_placed[e * nn : (e + 1) * nn][self._bead_idx_np]
            worst = max(worst, float(np.max(np.linalg.norm(placed - expected, axis=1))))
        print(f"[TERRAIN] reset check: max bead mismatch vs MuJoCo spindles {1e3 * worst:.1f} mm")
        if worst > 0.05:
            raise RuntimeError(f"reset inconsistent: ANCF beads {worst:.3f} m from their spindles")

    def _on_reset(self) -> None:
        """Vehicle back at the spawn (fault, stage change, out of the arena, button): new episode,
        command and manoeuvre state cleared."""
        self._close_turn()
        self._turn = -1
        self._t_scn = 0.0
        self._finished = False
        self.episode += 1
        self._cmd = SkidCommand()
        self._trail.clear()
        self._trail_s_host[:] = FAR_BELOW
        self._trail_e_host[:] = FAR_BELOW
        self._trail_prev = None
        self._lap, self._prev_turn, self.lateral = 0, -1, 0.0

    # ── Controller / mode ──────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise KeyError(f"unknown controller '{mode}'; available: {MODES}")
        if mode == "track" and self.terrain.track is None:
            raise ValueError(f"terrain '{self.terrain.name}' has no reference track to follow")
        self._mode = mode
        self.steer_angle = 0.0
        self._target_wheel_speed = 0.0
        self._cmd = SkidCommand()

    def _load_route(self) -> None:
        """Track-following numbers from the bundle's :class:`~newton.examples.ancf._field_task.ReferenceTrack`:
        turns = the stretches of the track bending one way, yaw* = v* / typical bend radius (a plot
        scale and the test threshold), the corridor borders for the minimap."""
        tr = self.terrain.track
        if tr is None:
            self._route_R, self._loop_turns, self._loop_len, self._loop_period = math.inf, 0, 0.0, 1.0
            self._corridor_xy = []
            return
        self._route_R = tr.turn_radius()
        self._loop_turns = tr.n_turns
        self._loop_len = tr.length
        n = np.stack([-tr.tangent[:, 1], tr.tangent[:, 0]], axis=1) * self._half_width
        self._corridor_xy = [(tr.xy + n, tr.closed), (tr.xy - n, tr.closed)]  # (boundary (N, 2), closed)
        self._apply_speed()
        print(
            f"[TERRAIN] track: {tr.n_turns} turns, bend R {self._route_R:.1f} m -> yaw* {math.degrees(self.yaw_set):.0f} deg/s at "
            f"{self.v_set} m/s; corridor +-{self._half_width} m; pure pursuit lookahead max({self.lookahead_min} m, {self.lookahead_time} s * v)"
        )

    def _apply_speed(self) -> None:
        """Setpoints that follow the speed on the fixed track: yaw* = v* / R and the loop time."""
        v = max(self.v_set, 1e-3)
        if math.isfinite(self._route_R):
            self.yaw_set = v / self._route_R
        self._loop_period = self._loop_len / v

    # ── Track following ────────────────────────────────────────────────────────

    def _scenario_tick(self) -> None:
        """Advance the manoeuvre clock, set (v_cmd, yaw_cmd), close / open turns, apply condition switches."""
        self._t_scn += vehicle_config.FRAME_DT
        t = self._t_scn
        if t < self.settle_time:
            self.v_cmd, self.yaw_cmd = 0.0, 0.0
            return
        self.v_cmd, self.yaw_cmd, turn = self._track_route()
        tr = self.terrain.track
        if not tr.closed and not self._finished and self._s_track >= tr.length - _END_MARGIN:
            self._finished = True
            print(f"[TERRAIN] end of the track ({tr.length:.0f} m): stopping")
        if self._finished:
            self.v_cmd, self.yaw_cmd = 0.0, 0.0
        if turn != self._turn:
            self._close_turn()
            self._turn = turn
            if turn >= 0:
                self._turn_t0 = t
                self._turn_acc = {"iae": 0.0, "sq": 0.0, "lat_sq": 0.0, "n": 0, "t_band": None}
                self._maybe_switch(turn)

    def _track_route(self) -> tuple[float, float, int]:
        """Pure pursuit on the track: ``(v_cmd, yaw_cmd, turn)`` - the speed capped by the tightest bend
        ahead (a_lat), the yaw-rate command toward the lookahead point, and the id of the turn the
        vehicle is in (position-based; -1 on a straight or outside the corridor)."""
        tr = self.terrain.track
        q = self._pose
        x, y, yaw = float(q[0]), float(q[1]), quat_yaw(q)
        _i, s_here, self.lateral = tr.nearest(x, y, self._s_track)
        self._s_track = s_here
        look = max(self.lookahead_min, self.lookahead_time * max(self.v_fwd, 0.0))
        p = tr.point_at(s_here + look)
        alpha = wrap_pi(math.atan2(p[1] - y, p[0] - x) - yaw)
        v_ref = max(self.v_fwd, 0.5)
        # circular arc through the lookahead point. Not capped: this position feedback is what keeps the
        # vehicle on the track when it turns less than asked (skid scrub); the levers / steer saturate at +-1.
        yaw_cmd = 2.0 * v_ref * math.sin(alpha) / look
        v_cmd = self.v_set
        if self.a_lat > 0.0:
            kappa = max(tr.curvature_at(s_here + d) for d in np.arange(0.0, look + _CURVATURE_HORIZON, 2.0))
            if kappa > 1e-4:
                v_cmd = min(v_cmd, math.sqrt(self.a_lat / kappa))
        turn = tr.turn_at(s_here)
        if turn < 0 or abs(self.lateral) > self._half_width:
            return v_cmd, yaw_cmd, -1
        # turn ids run 0 .. n-1 along the loop (turn 0 spans the seam): a drop in the id is a new lap
        if self._prev_turn >= 0 and turn < self._prev_turn:
            self._lap += 1
        self._prev_turn = turn
        return v_cmd, yaw_cmd, self._lap * self._loop_turns + turn

    def _maybe_switch(self, turn: int) -> None:
        if (
            self.switch_pressure_turns > 0
            and turn > 0
            and turn % self.switch_pressure_turns == 0
            and self.ctis is not None
        ):
            self._p_idx ^= 1
            p = self.p_levels[self._p_idx]
            self.ctis.set_all(self.ctis._build + p * self.ctis._per_unit)
            print(f"[TERRAIN] turn {turn}: CTIS setpoint -> {p:g} {self.ctis._unit} (ramps at the fill rate)")
        if self.switch_stage_turns > 0 and turn > 0 and turn % self.switch_stage_turns == 0:
            self._stage_idx ^= 1
            w = self.stage_levels[self._stage_idx]
            print(f"[TERRAIN] turn {turn}: terrain stage -> {w:g} (re-drop = new episode)")
            self.set_difficulty(w)  # calls reset_vehicle -> _on_reset

    def _close_turn(self) -> None:
        acc = self._turn_acc
        if acc is None or acc["n"] == 0:
            return
        n = acc["n"]
        self.turns.append(
            {
                "turn": self._turn,
                "episode": self.episode,
                "iae": acc["iae"],
                "rms": math.sqrt(acc["sq"] / n),
                "lat_rms": math.sqrt(acc["lat_sq"] / n),
                "t_band": acc["t_band"] if acc["t_band"] is not None else math.nan,
                "p": self.p_levels[self._p_idx],
                "w": self.terrain.w,
                "yaw_cmd": self.yaw_cmd,  # sign = turn direction
            }
        )
        r = self.turns[-1]
        print(
            f"[TERRAIN] turn {self._turn} done: yaw IAE {acc['iae']:.3f} rad  RMS {math.degrees(r['rms']):.1f} deg/s"
            f"  lateral RMS {r['lat_rms']:.2f} m  time-to-band {r['t_band']:.2f} s  p {r['p']:g}  w {self.terrain.w:.2f}"
        )
        self._turn_acc = None

    def _accumulate_turn(self) -> None:
        acc = self._turn_acc
        if acc is None:
            return
        e = self.yaw_cmd - self.yaw_rate
        acc["iae"] += abs(e) * vehicle_config.FRAME_DT
        acc["sq"] += e * e
        acc["lat_sq"] += self.lateral * self.lateral
        acc["n"] += 1
        if acc["t_band"] is None and abs(e) <= self.band * abs(self.yaw_cmd):
            acc["t_band"] = self._t_scn - self._turn_t0

    def _drive(self) -> None:
        """One frame of track following: the pure pursuit command through the drive kinematics,
        slew-limited into the base example's actuators (steer_angle, _target_wheel_speed)."""
        self._scenario_tick()
        if self._skid:
            cmd = self.plant.command(self.v_cmd, self.yaw_cmd)
        else:
            # kingpin: bicycle model delta = atan(L yaw* / v), throttle = wheel speed for v_cmd + P term
            v_ref = max(self.v_fwd, 0.5)
            steer = math.atan(2.0 * float(self.spec.half_wheelbase) * self.yaw_cmd / v_ref) / self._steer_range
            w = (self.v_cmd + _SPEED_KP * (self.v_cmd - self.v_fwd)) / float(self.spec.tire_R_outer)
            cmd = SkidCommand(w / float(self.spec.max_wheel_speed), steer).clipped()
        c = self._cmd
        d = cmd.turn - c.turn
        c.turn += math.copysign(min(abs(d), self._steer_rate), d) if d else 0.0
        d = cmd.throttle - c.throttle
        c.throttle += math.copysign(min(abs(d), self._throttle_rate), d) if d else 0.0
        self.steer_angle = self._steer_sign * c.turn * self._steer_range  # skid: the lever itself (+ = left)
        self._target_wheel_speed = c.throttle * float(self.spec.max_wheel_speed)

    # ── Step / driving ─────────────────────────────────────────────────────────

    def step(self) -> None:
        if self._fault:
            # the physics left its valid range: back to the snapshot (positions, velocities, tire
            # state, MuJoCo warm start all restored)
            self.fault_count += 1
            self._fault = ""
            self.reset_vehicle()
            return
        self._read_keys()
        if self._mode != "manual":
            self._drive()
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) > self._wheel_speed_rate:
            self.wheel_speed += math.copysign(self._wheel_speed_rate - vehicle_config.WHEEL_SPEED_RATE, d)
        super().step()
        self._watch()
        self._telemetry()
        if (
            self._log_path
            and not self._log_written
            and self._frame >= int(getattr(self._args, "num_frames", 0) or 0) > 0
        ):
            self._write_log()

    def _read_keys(self) -> None:
        """Driving keys, chosen not to collide with the viewer's WASD / QE / arrow camera controls
        (the other Newton keyboard examples use the same block): I / K ramp the throttle, J / L ramp
        the steer toward full lock and it self-centres when released, SPACE zeroes the throttle.
        Under a controller, any driving key hands over to manual."""
        v = self.viewer
        if v is None or not hasattr(v, "is_key_down"):
            return
        try:
            up, down = v.is_key_down("i"), v.is_key_down("k")
            left, right = v.is_key_down("j"), v.is_key_down("l")
            brake = v.is_key_down("space")
        except Exception:
            return
        if self._mode != "manual":
            if up or down or left or right or brake:
                print("[TERRAIN] driving key: manual driving")
                self._set_mode("manual")
            return
        vmax = float(self.spec.max_wheel_speed)
        dv = vmax * vehicle_config.FRAME_DT / _KEY_THROTTLE_TIME
        ds = self._steer_range * vehicle_config.FRAME_DT / _KEY_STEER_TIME
        if brake:
            self._target_wheel_speed = 0.0
        elif up and not down:
            self._target_wheel_speed = min(vmax, self._target_wheel_speed + dv)
        elif down and not up:
            self._target_wheel_speed = max(-vmax, self._target_wheel_speed - dv)
        s = self.steer_angle
        if left and not right:
            s += self._steer_sign * ds
        elif right and not left:
            s -= self._steer_sign * ds
        elif s != 0.0:
            s = math.copysign(max(0.0, abs(s) - ds), s)
        self.steer_angle = max(-self._steer_range, min(self._steer_range, s))

    # ── Watchdog / telemetry ───────────────────────────────────────────────────

    def _read_chassis(self) -> np.ndarray:
        return read_chassis_q(self.state_0, self._chassis_body, self._chassis_q)

    def _watch(self) -> None:
        """Chassis pose for the camera + the fault test (one transform, one spatial vector and
        three reduced floats read back per frame)."""
        q = self._read_chassis()
        qd = read_chassis_qd(self.state_0, self._chassis_body, self._chassis_qd)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
            self._fault = "non-finite chassis state"
            self._dump_history(self._fault)
            return  # step() resets the vehicle
        self._pose = q
        self._chassis_vel = qd
        t = self.terrain
        x, y, z = float(q[0]), float(q[1]), float(q[2])
        roll, pitch = quat_roll_pitch(q)
        self._diag_dev.zero_()
        wp.launch(
            _diag_reduce,
            dim=vehicle_config.N_TIRES * self._n_nodes,
            inputs=[self.terrain_scm.node_f, self.ancf_solver.node_xd, self._diag_dev],
            device="cuda:0",
        )
        dg = self._diag_dev.numpy()
        if dg[1] > self._fault_node_speed:
            self._fault = f"tread node at {dg[1]:.0f} m/s (> {self._fault_node_speed:.0f})"
        elif float(np.linalg.norm(qd[:3])) > _FAULT_BODY_SPEED:
            self._fault = f"chassis at {np.linalg.norm(qd[:3]):.0f} m/s (> {_FAULT_BODY_SPEED:.0f})"
        if self._fault:
            self._dump_history(self._fault)
        self._history.append(
            (
                self._frame,
                round(self.steer_angle, 3),
                round(self.wheel_speed, 2),
                round(float(np.linalg.norm(qd[:2])), 3),
                round(z - t.height_at(x, y), 3),
                round(math.degrees(roll)),
                round(math.degrees(pitch)),
                int(dg[2]),
                float(dg[0]),
                float(dg[1]),
            )
        )

    def _dump_history(self, why: str) -> None:
        print(
            f"[TERRAIN] SIM FAULT: {why} at frame {self._frame}. Last frames (frame, steer, axle rad/s, chassis m/s, h, roll, pitch, contact nodes, max|node_f| N, max|node_xd| m/s):"
        )
        for rec in self._history:
            print("   ", rec)
        if self._test:
            raise RuntimeError(f"[TERRAIN] simulation diverged: {why}")

    def _telemetry(self) -> None:
        """Speed / yaw rate from the chassis row the watchdog read, HUD traces, per-turn metrics, log,
        and the arena bound (leaving it puts the vehicle back at the spawn)."""
        if self._fault:
            return
        q, qd = self._pose, self._chassis_vel
        yaw = quat_yaw(q)
        self.v_fwd = float(qd[0]) * math.cos(yaw) + float(qd[1]) * math.sin(yaw)
        self.yaw_rate = float(qd[5])
        t = self.terrain
        x, y = float(q[0]), float(q[1])
        self._trail.append((x, y))
        self._trace_yaw.append((self.yaw_cmd, self.yaw_rate))
        self._trace_v.append((self.v_cmd, self.v_fwd))
        if self._mode != "manual":
            self._accumulate_turn()
        if self._log_path:
            L = self._log
            L["t"].append(self._t)
            L["v_cmd"].append(self.v_cmd)
            L["yaw_cmd"].append(self.yaw_cmd)
            L["v"].append(self.v_fwd)
            L["yaw_rate"].append(self.yaw_rate)
            L["throttle"].append(self._cmd.throttle)
            L["turn"].append(self._cmd.turn)
            L["lat"].append(self.lateral)
            L["w"].append(t.w)
            L["p"].append(list(self.ctis.live_p) if self.ctis is not None else [0.0] * vehicle_config.N_TIRES)
            L["episode"].append(self.episode)
            if self._wheel_log_count == self._wheel_log.shape[0]:
                grown = wp.zeros(
                    (2 * self._wheel_log.shape[0], vehicle_config.N_TIRES, 3), dtype=float, device=self.model.device
                )
                wp.copy(grown, self._wheel_log, count=self._wheel_log.size)
                self._wheel_log = grown
            wp.launch(
                _record_wheel_diagnostics,
                dim=vehicle_config.N_TIRES,
                inputs=[
                    self.vehicle._axle_dofs,
                    self.vehicle._axle_sign,
                    self.state_0.joint_qd,
                    self.control.joint_target_qd,
                    self.control.joint_f,
                    self._wheel_log_count,
                    self._wheel_log,
                ],
                device=self.model.device,
            )
            self._wheel_log_count += 1
        if abs(x) > t.arena_half_x - _OOB_MARGIN or abs(y) > t.arena_half_y - _OOB_MARGIN:
            print(f"[TERRAIN] left the arena at ({x:+.1f}, {y:+.1f}) m: reset")
            self.reset_vehicle()

    def _write_log(self) -> None:
        self._close_turn()
        self._log_written = True
        turns = self.turns
        keys = ("turn", "episode", "iae", "rms", "lat_rms", "t_band", "p", "w")
        out = {k: np.asarray(v) for k, v in self._log.items()}
        wheel = self._wheel_log.numpy()[: self._wheel_log_count]
        out["wheel_speed_rad_s"] = wheel[:, :, 0]
        out["wheel_target_rad_s"] = wheel[:, :, 1]
        out["wheel_motor_torque_nm"] = wheel[:, :, 2]
        out["wheel_order"] = np.asarray([label for label, _ in vehicle_config.WHEEL_ORDER])
        out["wheel_sample_timing"] = np.asarray(
            "End-of-frame angular velocity; target and effort from the final substep; signs normalized forward."
        )
        out.update({f"turn_{k}": np.asarray([d[k] for d in turns], dtype=np.float64) for k in keys})
        out["controller"] = np.asarray(f"{self._mode}/{'skid' if self._skid else 'kingpin'}")
        out["plant"] = np.asarray(
            [
                self.plant.half_track,
                self.plant.r_roll,
                self.plant.max_wheel_speed,
                self.plant.yaw_authority,
            ],
            dtype=np.float64,
        )
        out["stage_w"] = np.asarray(self.terrain.w)
        out["frame_dt"] = np.asarray(vehicle_config.FRAME_DT)
        out["scenario"] = np.asarray(
            [
                self.v_set,
                self.yaw_set,
                self.settle_time,
                self._route_R,
                self._loop_turns,
                self._loop_len,
                self._loop_period,
            ],
            dtype=np.float64,
        )
        np.savez(self._log_path, **out)
        print(f"[TERRAIN] log written: {self._log_path} ({len(out['t'])} frames, {len(turns)} turns)")

    # ── GUI (side panel) ───────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        t = self.terrain
        ui.text(f"Terrain  {t.name}   {2 * t.hx:.0f} x {2 * t.hy:.0f} m   I_N max {t.max_h:.1f} m")
        ui.text("Controller")
        for k, name in enumerate(MODES):
            if k:
                ui.same_line()
            if ui.button(("[%s]" if self._mode == name else " %s ") % name) and (
                name != "track" or t.track is not None
            ):
                self._set_mode(name)
        self._gui_scenario(ui)
        ui.text(
            f"cmd  v {self.v_cmd:.2f} m/s  yaw {math.degrees(self.yaw_cmd):+.0f} deg/s   meas  v {self.v_fwd:.2f} m/s  yaw {math.degrees(self.yaw_rate):+.0f} deg/s"
            f"   steer {self._cmd.turn:+.2f}  throttle {self._cmd.throttle:+.2f}   off-track {self.lateral:+.2f} m"
        )
        if self._mode != "manual":
            ui.text(f"{self._mode} drives: a driving key (I K J L) or a steer / throttle slider switches to manual.")
        ui.separator()

        ui.text("Camera")
        for k, mode in enumerate(self.CAMERA_MODES):
            if k:
                ui.same_line()
            if ui.button(("[%s]" if self.camera_mode == mode else " %s ") % mode):
                self.camera_mode = mode
        if self.camera_mode == "top":
            _c, self.top_view_height = ui.slider_float("top view height [m]", self.top_view_height, 8.0, 80.0)
        c1, self.sun_elevation_deg = ui.slider_float("sun elevation [deg]", self.sun_elevation_deg, 10.0, 90.0)
        c2, self.sun_azimuth_deg = ui.slider_float("sun azimuth [deg]", self.sun_azimuth_deg, -180.0, 180.0)
        c3, self.ambient = ui.slider_float("ambient light", self.ambient, 0.2, 2.5)
        if c1 or c2 or c3:
            self._apply_light()
        ui.separator()

        # terrain relief: applied on slider release (a stage change rebuilds the terrain arrays and re-drops the car)
        ui.text("Terrain")
        _c, self._w_pending = ui.slider_float("stage w (heights = w * I_N)", self._w_pending, 0.0, 1.0)
        if ui.is_item_deactivated_after_edit() and abs(self._w_pending - t.w) > 1e-3:
            self.set_difficulty(self._w_pending)
        for k, s in enumerate(t.stages):
            if k:
                ui.same_line()
            if ui.button(("[%.1f]" if abs(s - t.w) < 1e-3 else " %.1f ") % s):
                self.set_difficulty(s)
        if ui.button("Reset vehicle (snapshot at the spawn pose)"):
            self.reset_vehicle()
        ui.same_line()
        ui.text(f"sim faults {self.fault_count}   episode {self.episode}")
        _c, self.show_mj_contacts = ui.checkbox(
            "Show MuJoCo contacts with the terrain collider (red points)", self.show_mj_contacts
        )
        _c, self.show_collider = ui.checkbox(
            f"Show MuJoCo collider grid ({self._hc_k * t.cell:g} m block {self._hc_mode}, follows the stage)",
            self.show_collider,
        )
        if self._reference_track_starts is not None:
            _c, self.show_reference_track = ui.checkbox("Show reference track", self.show_reference_track)
        ui.separator()

        # the base example's steer / throttle sliders + CTIS panel + live readouts
        ui.text(
            "drive: I / K throttle   J / L steer (self-centring)   SPACE stop      camera (free): WASD / QE + mouse"
        )
        steer_before, throttle_before = self.steer_angle, self._target_wheel_speed
        super().gui(ui)
        if self._mode != "manual" and (self.steer_angle != steer_before or self._target_wheel_speed != throttle_before):
            s, w = self.steer_angle, self._target_wheel_speed
            self._set_mode("manual")  # otherwise the controller overwrites the slider next frame
            self.steer_angle, self._target_wheel_speed = s, w

    def _gui_scenario(self, ui) -> None:
        """Track-following setpoints (a subclass with its own manoeuvre replaces this section)."""
        ui.text("Track")
        c_v, self.v_set = ui.slider_float("speed setpoint [m/s]", self.v_set, 0.0, self.plant.v_max)
        _c, self.a_lat = ui.slider_float("lateral accel cap in bends [m/s^2] (0 = off)", self.a_lat, 0.0, 8.0)
        if self._skid:
            _c, self.plant.yaw_authority = ui.slider_float(
                "yaw authority (measured / kinematic yaw per lever)", self.plant.yaw_authority, 0.1, 1.5
            )
        if c_v:
            self._apply_speed()
        if self.terrain.track is None:
            ui.text("  no reference track in this bundle")
            return
        ui.text(
            f"  {self._loop_turns} turns / {self._loop_len:.0f} m ({self._loop_period:.0f} s at v*), "
            f"bend R {self._route_R:.1f} m -> yaw* {math.degrees(self.yaw_set):.0f} deg/s, corridor +-{self._half_width:g} m"
        )
        ui.text(
            f"  turn {self._turn}  turns done {len(self.turns)}"
            + (
                f"   last IAE {self.turns[-1]['iae']:.3f} rad  t-band {self.turns[-1]['t_band']:.2f} s"
                if self.turns
                else ""
            )
        )
        ui.text(
            f"  switches: pressure every {self.switch_pressure_turns or '-'} turns ({self.p_levels[0]:g}/{self.p_levels[1]:g}), "
            f"stage every {self.switch_stage_turns or '-'} turns ({self.stage_levels[0]:g}/{self.stage_levels[1]:g})"
        )

    # ── HUD overlay ────────────────────────────────────────────────────────────

    def _register_hud(self) -> None:
        if self.viewer is not None and hasattr(self.viewer, "register_ui_callback"):
            # "free": always drawn (the side panel only renders while expanded)
            self.viewer.register_ui_callback(self._draw_hud, position="free")

    def _init_minimap(self) -> None:
        t = self.terrain
        extent = 2.0 * max(t.arena_half_x, t.arena_half_y) * (1.0 + 2.0 * MINIMAP_PAD)
        self._map_scale = MINIMAP_SIZE / max(extent, 1e-6)

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

    def _hud_caption(self, imgui, draw, x, y, text, align="left", span=0.0, small=False):
        font_size = (0.9 if small else 1.1) * imgui.get_font_size()
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

    def _draw_hud(self, imgui) -> None:
        if not self._hud_ok:
            return
        try:
            vp = imgui.get_main_viewport()
            # bottom row, left of the viewport just past the viewer's side panel: control panel
            # (yaw / speed strips, drive bars, turn IAE) then the minimap
            sidebar = getattr(self.viewer, "_sidebar_width_fb_px", lambda: 320.0)()
            y0 = vp.pos.y + vp.size.y - HUD_H - MINIMAP_MARGIN
            x_ctl = vp.pos.x + sidebar + 2.0 * MINIMAP_MARGIN
            self._hud_window(
                imgui, "##terrain_control", x_ctl, y0, CONTROL_W, HUD_H, lambda: self._draw_control(imgui, x_ctl, y0)
            )
            x_map = x_ctl + CONTROL_W + HUD_GAP
            self._hud_window(
                imgui, "##terrain_minimap", x_map, y0, MINIMAP_SIZE, HUD_H, lambda: self._draw_minimap(imgui, x_map, y0)
            )
        except Exception as e:
            self._hud_ok = False
            print(f"[TERRAIN] HUD disabled: {e!r}")
            traceback.print_exc()

    def _col(self, imgui, r, g, b, a=1.0):
        return imgui.color_convert_float4_to_u32(imgui.ImVec4(r, g, b, a))

    def _draw_minimap(self, imgui, x0, y0):
        t = self.terrain
        cx, cy, s = x0 + 0.5 * MINIMAP_SIZE, y0 + 0.5 * MINIMAP_SIZE, self._map_scale

        def to_px(p):
            return imgui.ImVec2(cx + p[0] * s, cy - p[1] * s)

        draw = imgui.get_window_draw_list()
        gray, blue, orange = (
            self._col(imgui, 0.55, 0.55, 0.6, 0.9),
            self._col(imgui, 0.0, 0.66, 1.0, 0.9),
            self._col(imgui, 1.0, 0.55, 0.1),
        )
        hx, hy = t.arena_half_x, t.arena_half_y
        draw.add_rect(to_px((-hx, hy)), to_px((hx, -hy)), gray, thickness=1.5)
        draw.add_rect(to_px((-t.hx, t.hy)), to_px((t.hx, -t.hy)), self._col(imgui, 0.4, 0.4, 0.45, 0.5), thickness=1.0)
        if len(self._trail) >= 2:
            step = max(1, len(self._trail) // 600)
            trail = list(self._trail)[::step]
            if trail[-1] != self._trail[-1]:
                trail.append(self._trail[-1])
            draw.add_polyline([to_px(p) for p in trail], blue, flags=imgui.ImDrawFlags_.none, thickness=1.5)
        white = self._col(imgui, 1.0, 1.0, 1.0, 0.85)
        for b, closed in self._corridor_xy:  # corridor borders around the reference track
            flags = imgui.ImDrawFlags_.closed if closed else imgui.ImDrawFlags_.none
            draw.add_polyline([to_px(p) for p in b], white, flags=flags, thickness=1.0)
        car = to_px((float(self._pose[0]), float(self._pose[1])))
        yaw = quat_yaw(self._pose)
        tip = imgui.ImVec2(car.x + 9.0 * math.cos(yaw), car.y - 9.0 * math.sin(yaw))
        draw.add_line(car, tip, orange, 2.0)
        draw.add_circle_filled(car, 4.0, orange)
        self._hud_caption(imgui, draw, x0 + 8.0, y0 + 6.0, t.name.upper())
        self._hud_caption(
            imgui,
            draw,
            x0 + MINIMAP_SIZE - 8.0,
            y0 + MINIMAP_SIZE - 26.0,
            f"w {t.w:.1f}  ep {self.episode}",
            align="right",
        )

    def _strip(self, imgui, draw, x, y, w, h, trace, lo, hi, title):
        """Command (grey) vs measured (colour) over the last _TRACE_LEN frames."""
        frame = self._col(imgui, 0.7, 0.7, 0.75, 0.5)
        draw.add_rect(imgui.ImVec2(x, y), imgui.ImVec2(x + w, y + h), frame, thickness=1.0)
        span = max(hi - lo, 1e-6)
        zero_y = y + h * (1.0 - (0.0 - lo) / span)
        if lo < 0.0 < hi:
            draw.add_line(imgui.ImVec2(x, zero_y), imgui.ImVec2(x + w, zero_y), frame, 1.0)
        n = len(trace)
        if n >= 2:
            dx = w / (_TRACE_LEN - 1)
            xs = x + w - dx * (n - 1)

            def pts(i):
                return [
                    imgui.ImVec2(xs + k * dx, y + h * (1.0 - (min(max(v[i], lo), hi) - lo) / span))
                    for k, v in enumerate(trace)
                ]

            draw.add_polyline(
                pts(0), self._col(imgui, 0.8, 0.8, 0.85, 0.9), flags=imgui.ImDrawFlags_.none, thickness=1.5
            )
            draw.add_polyline(
                pts(1), self._col(imgui, 1.0, 0.55, 0.1, 1.0), flags=imgui.ImDrawFlags_.none, thickness=1.5
            )
        self._hud_caption(imgui, draw, x + 4.0, y + 2.0, title, small=True)

    def _second_strip(self):
        """(trace, lo, hi, title) of the control panel's second strip chart: speed here (a subclass may
        show its own loop variable instead)."""
        v_hi = max(1.5 * self.v_set, 1.0)
        return self._trace_v, -0.2 * v_hi, v_hi, f"SPEED  {self.v_fwd:.2f} m/s"

    def _draw_control(self, imgui, x0, y0):
        draw = imgui.get_window_draw_list()
        pad = 10.0
        strip_w = 150.0
        strip_h = HUD_H - 2.0 * pad
        y_ref = max(1.5 * abs(self.yaw_set), math.radians(15.0))
        self._strip(
            imgui,
            draw,
            x0 + pad,
            y0 + pad,
            strip_w,
            strip_h,
            self._trace_yaw,
            -y_ref,
            y_ref,
            f"YAW  {math.degrees(self.yaw_rate):+.0f} deg/s",
        )
        trace, lo, hi, title = self._second_strip()
        self._strip(imgui, draw, x0 + 2 * pad + strip_w, y0 + pad, strip_w, strip_h, trace, lo, hi, title)
        # drive bars: the wheel-speed differential of a skid steer (LEFT / RIGHT), steer and throttle
        # of a kingpin-steered vehicle
        bx = x0 + 3 * pad + 2 * strip_w
        bw = CONTROL_W - (bx - x0) - pad
        throttle = self.wheel_speed / float(self.spec.max_wheel_speed)
        if self._skid:
            lever = self._steer_sign * self.steer_angle
            bars = ((throttle * (1.0 - max(lever, 0.0)), "LEFT"), (throttle * (1.0 - max(-lever, 0.0)), "RIGHT"))
        else:
            bars = ((self._steer_sign * self.steer_angle / self._steer_range, "STEER (+ left)"), (throttle, "THROTTLE"))
        outline, blue, red = (
            self._col(imgui, 0.7, 0.7, 0.75, 0.5),
            self._col(imgui, 0.0, 0.66, 1.0, 0.95),
            self._col(imgui, 0.9, 0.15, 0.15, 0.95),
        )
        bar_h, seg_gap = 10.0, 2.0
        seg_w = (bw - (HUD_BAR_SEGMENTS - 1) * seg_gap) / HUD_BAR_SEGMENTS
        for label_y, (side, name) in zip((y0 + 8.0, y0 + 46.0), bars, strict=True):
            self._hud_caption(imgui, draw, bx, label_y, name, small=True)
            y = label_y + 0.9 * imgui.get_font_size() + 3.0
            frac, fill = min(1.0, abs(side)), blue if side >= 0.0 else red
            for i in range(HUD_BAR_SEGMENTS):
                sx = bx + i * (seg_w + seg_gap)
                draw.add_rect(imgui.ImVec2(sx, y), imgui.ImVec2(sx + seg_w, y + bar_h), outline, thickness=1.0)
                lit = max(0.0, min(1.0, frac * HUD_BAR_SEGMENTS - i))
                if lit > 0.0:
                    draw.add_rect_filled(imgui.ImVec2(sx, y), imgui.ImVec2(sx + seg_w * lit, y + bar_h), fill)
        # per-turn yaw IAE (last 12 turns): blue = high pressure, red = low, brightness = stage
        ty = y0 + 86.0
        self._hud_caption(imgui, draw, bx, ty, f"TURN IAE  {self._mode}", small=True)
        base_y = y0 + HUD_H - pad
        top_y = ty + 0.9 * imgui.get_font_size() + 4.0
        turns = self.turns[-12:]
        if turns:
            ref = max(1e-6, *(d["iae"] for d in turns))
            tw = bw / 12.0
            for k, d in enumerate(turns):
                hgt = (base_y - top_y) * d["iae"] / ref
                shade = 0.55 + 0.45 * d["w"]
                col = (
                    self._col(imgui, 0.0, 0.66 * shade, 1.0 * shade, 0.95)
                    if d["p"] >= self.p_levels[0]
                    else self._col(imgui, 0.9 * shade, 0.15, 0.15, 0.95)
                )
                x = bx + k * tw
                draw.add_rect_filled(imgui.ImVec2(x + 1.0, base_y - hgt), imgui.ImVec2(x + tw - 1.0, base_y), col)
        if self._turn_acc is not None:
            self._hud_caption(
                imgui, draw, bx + bw, ty, f"#{self._turn} {self._turn_acc['iae']:.2f}", align="right", small=True
            )

    # ── Render ─────────────────────────────────────────────────────────────────

    def _apply_light(self) -> None:
        """Push the sun direction / light colour / ambient terms to the GL renderer (no-op for others)."""
        r = getattr(self.viewer, "renderer", None)
        if r is None:
            return
        el, az = math.radians(self.sun_elevation_deg), math.radians(self.sun_azimuth_deg)
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        if hasattr(r, "_sun_direction"):
            r._sun_direction = d / np.linalg.norm(d)
        if hasattr(r, "_light_color"):
            r._light_color = (1.0, 1.0, 1.0)
        if hasattr(r, "ambient_sky"):
            r.ambient_sky = tuple(min(2.0, self.ambient * c) for c in (0.8, 0.8, 0.85))
        if hasattr(r, "ambient_ground"):
            r.ambient_ground = tuple(min(2.0, self.ambient * c) for c in (0.45, 0.45, 0.5))
        if hasattr(r, "shadow_extents"):
            # the shadow map is an orthographic box of +-extents around the camera (the viewer's
            # default is 10 m): anything outside it renders unlit, so it has to cover the terrain
            t = self.terrain
            r.shadow_extents = 1.2 * max(t.hx, t.hy)

    def _update_camera(self) -> None:
        """follow = chase camera behind the car; top = straight down from above the car, heading up
        on screen; free = leave the viewer's own navigation alone."""
        if self.camera_mode == "follow":
            set_follow_camera(self.viewer, self._pose)
        elif self.camera_mode == "top":
            q = self._pose
            self.viewer.set_camera(
                pos=wp.vec3(float(q[0]), float(q[1]), float(q[2]) + self.top_view_height),
                pitch=-89.0,
                yaw=math.degrees(quat_yaw(q)),
            )

    def render(self) -> None:
        if self.viewer is None:
            return
        self._update_camera()
        v = self.viewer
        v.begin_frame(self._t)
        v.log_state(self.state_0)
        # log_mesh recomputes normals and re-uploads every vertex, so the (static) terrain meshes are
        # only logged when they change (stage / set_model) or the collider overlay is toggled; the
        # collider is not logged at all until it is first shown.
        if self._terrain_dirty or self.show_collider != self._logged_show_collider:
            v.log_mesh(
                "/terrain/mesh",
                self._terrain_points,
                self._terrain_indices,
                color=_TERRAIN_COLOR,
                backface_culling=False,
                hidden=self.show_collider,
            )
            if self.show_collider or self._collider_logged:
                v.log_mesh(
                    "/terrain/collider",
                    self._collider_points,
                    self._collider_indices,
                    color=(0.9, 0.3, 0.9),
                    backface_culling=False,
                    hidden=not self.show_collider,
                )
                self._collider_logged = True
            self._logged_show_collider = self.show_collider
            self._terrain_dirty = False
        if self._reference_track_starts is not None:
            v.log_lines(
                "/terrain/reference_track",
                self._reference_track_starts,
                self._reference_track_ends,
                colors=(1.0, 0.85, 0.1),
                hidden=not self.show_reference_track,
            )
        if self.show_mj_contacts:
            d = self.solver.mjw_data
            wp.launch(
                _mj_contact_points,
                dim=len(self._mj_contact_pts),
                inputs=[d.contact.pos, d.contact.geom, d.nacon, self._collider_geom, FAR_BELOW, self._mj_contact_pts],
                device="cuda:0",
            )
        v.log_points(
            "/terrain/mj_contacts",
            self._mj_contact_pts,
            radii=self._mj_contact_radii,
            colors=self._mj_contact_colors,
            hidden=not self.show_mj_contacts,
        )
        v.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, colors=(1.0, 0.45, 0.0))
        v.log_lines("bead_spokes", self._spoke_start_zu, self._bead_pos_zu, colors=(1.0, 0.90, 0.1))
        wp.launch(
            terrain_contact_spikes,
            dim=vehicle_config.N_TIRES * self._n_nodes,
            inputs=[
                self.terrain_scm.node_f,
                self.state_0.particle_q,
                2.0e-4,
                self._contact_line_s,
                self._contact_line_e,
            ],
            device="cuda:0",
        )
        v.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, colors=(0.0, 1.0, 1.0))
        self._render_overlays()
        v.end_frame()

    # ── Scene overlays (arrows + trail) ────────────────────────────────────────

    @staticmethod
    def _arc_xy(x: float, y: float, yaw: float, v: float, omega: float, horizon: float, n: int) -> np.ndarray:
        """(n, 2) plan-view points of constant-speed / constant-yaw-rate motion from (x, y, yaw)."""
        ts = np.linspace(0.0, horizon, n)
        if abs(omega) < 1e-3:
            return np.stack([x + v * ts * math.cos(yaw), y + v * ts * math.sin(yaw)], axis=1)
        r = v / omega
        th = yaw + omega * ts
        return np.stack([x + r * (np.sin(th) - math.sin(yaw)), y - r * (np.cos(th) - math.cos(yaw))], axis=1)

    @staticmethod
    def _arrow_mesh_indices(k: int) -> np.ndarray:
        """Triangles of the arrow: quad strip over vertices [0..k) left / [k..2k) right, head = last 3."""
        i = np.arange(k - 1)
        shaft = np.concatenate(
            [np.stack([i, i + 1, k + i], axis=1), np.stack([i + 1, k + i + 1, k + i], axis=1)], axis=0
        )
        head = np.array([[2 * k, 2 * k + 1, 2 * k + 2]])
        return np.concatenate([shaft, head], axis=0).flatten().astype(np.int32)

    def _update_arrow(self, xy: np.ndarray, target: wp.array) -> None:
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
        tip_base, fwd, side = xy[-1], d[-1], normal[-1]
        h[2 * k, :2] = tip_base + fwd * _ARROW_HEAD_LENGTH
        h[2 * k + 1, :2] = tip_base + side * _ARROW_HEAD_HALF_WIDTH
        h[2 * k + 2, :2] = tip_base - side * _ARROW_HEAD_HALF_WIDTH
        h[2 * k :, 2] = z[-1]
        target.assign(h)

    def _update_trail3d(self) -> None:
        """Append the chassis ground point to the ring of trail segments (one small upload per frame)."""
        q = self._pose
        x, y = float(q[0]), float(q[1])
        p = np.array([x, y, self.terrain.height_at(x, y) + _TRAIL_LIFT], dtype=np.float32)
        if self._trail_prev is not None:
            i = self._trail_i % _TRAIL3D_LEN
            self._trail_s_host[i] = self._trail_prev
            self._trail_e_host[i] = p
            self._trail_i += 1
        self._trail_prev = p
        self._trail_s.assign(self._trail_s_host)
        self._trail_e.assign(self._trail_e_host)

    def _commanded_path_xy(self, x: float, y: float, yaw: float, k: int) -> np.ndarray:
        """(k, 2) path the current command describes (orange arrow): the (v*, yaw*) arc here."""
        return self._arc_xy(x, y, yaw, self.v_cmd, self.yaw_cmd, _ARROW_HORIZON, k)

    def _render_overlays(self) -> None:
        q = self._pose
        x, y, yaw = float(q[0]), float(q[1]), quat_yaw(q)
        k = _ARROW_SAMPLES
        # commanded arc: only while a controller has a moving command (settling / manual: hidden)
        self._arrow_cmd_visible = self._mode != "manual" and abs(self.v_cmd) > 0.1
        if self._arrow_cmd_visible:
            self._update_arrow(self._commanded_path_xy(x, y, yaw, k), self._arrow_cmd)
        # measured arc: what the chassis is doing now, extrapolated over the same horizon
        self._arrow_meas_visible = abs(self.v_fwd) > 0.3
        if self._arrow_meas_visible:
            self._update_arrow(self._arc_xy(x, y, yaw, self.v_fwd, self.yaw_rate, _ARROW_HORIZON, k), self._arrow_meas)
        self._update_trail3d()
        v = self.viewer
        v.log_mesh(
            "/terrain/commanded_arc",
            self._arrow_cmd,
            self._arrow_indices,
            color=(1.0, 0.55, 0.1),
            backface_culling=False,
            hidden=not self._arrow_cmd_visible,
        )
        v.log_mesh(
            "/terrain/measured_arc",
            self._arrow_meas,
            self._arrow_indices,
            color=(0.0, 0.85, 1.0),
            backface_culling=False,
            hidden=not self._arrow_meas_visible,
        )
        v.log_lines("terrain_trail", self._trail_s, self._trail_e, colors=(0.0, 0.66, 1.0))

    # ── Diagnostics / tests ────────────────────────────────────────────────────

    def _print_diag(self, fps: float = 0.0) -> None:
        super()._print_diag(fps)
        q = self._pose
        t = self.terrain
        roll, pitch = quat_roll_pitch(q)
        print(
            f"  [{self._mode}] cmd v={self.v_cmd:.2f} yaw={math.degrees(self.yaw_cmd):+.0f}  meas v={self.v_fwd:.2f} yaw={math.degrees(self.yaw_rate):+.0f} deg/s"
            f"  steer {self.steer_angle:+.3f}  axle {self.wheel_speed:.1f}/{self._target_wheel_speed:.1f} rad/s  "
            f"h={float(q[2]) - t.height_at(float(q[0]), float(q[1])):.2f} m  roll={math.degrees(roll):+.0f} pitch={math.degrees(pitch):+.0f} deg"
            f"  off-track {self.lateral:+.2f} m  turn {self._turn} ep {self.episode} turns {len(self.turns)}  w={t.w:.2f}  faults {self.fault_count}"
        )
        # heading audit: body yaw vs the direction the chassis actually moves vs the track tangent
        # (a spawn-yaw or drive-sign bug shows up here as a constant offset)
        qd = self._chassis_vel
        v_xy = math.hypot(float(qd[0]), float(qd[1]))
        course = math.degrees(math.atan2(float(qd[1]), float(qd[0]))) if v_xy > 0.2 else math.nan
        tr = t.track
        if tr is not None:
            _i, s_here, _lat = tr.nearest(float(q[0]), float(q[1]), self._s_track)
            tang = tr.tangent_at(s_here)
            track_yaw = math.degrees(math.atan2(float(tang[1]), float(tang[0])))
        else:
            track_yaw = math.nan
        print(
            f"  heading: body yaw {math.degrees(quat_yaw(q)):+.0f} deg  course (velocity dir) {course:+.0f} deg  "
            f"track tangent {track_yaw:+.0f} deg  spawn yaw {math.degrees(self.spawn[2]):+.0f} deg  pos ({float(q[0]):+.1f}, {float(q[1]):+.1f})"
        )

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = VehicleSimulation.create_parser()
        parser.add_argument(
            "--terrain",
            type=str,
            default="boulders",
            help="Terrain bundle under examples/ancf/assets/terrain/ (newton_terrain_tool.field/2: boulders, craters, rellis_00000..).",
        )
        parser.add_argument(
            "--difficulty",
            type=float,
            default=1.0,
            help="Terrain stage w in [0, 1]: heights = w * I_N (changeable in the UI).",
        )
        parser.add_argument(
            "--controller",
            type=str,
            default="track",
            choices=MODES,
            help="manual = keys / sliders; track = follow the bundle's reference track.",
        )
        parser.add_argument("--speed", type=float, default=3.0, help="Track-following speed setpoint [m/s].")
        parser.add_argument(
            "--a-lat", type=float, default=2.5, help="Lateral-acceleration cap in bends [m/s^2]; 0 = no cap."
        )
        parser.add_argument(
            "--corridor-half-width",
            type=float,
            default=None,
            help=f"Corridor half width around the track [m]; default: the bundle's baked corridor, else {_CORRIDOR_HALF_WIDTH} m.",
        )
        parser.add_argument("--lookahead-min", type=float, default=4.0, help="Pure-pursuit lookahead floor [m].")
        parser.add_argument("--lookahead-time", type=float, default=1.0, help="Pure-pursuit lookahead per m/s [s].")
        parser.add_argument(
            "--settle-time",
            type=float,
            default=3.0,
            help="Standstill before the manoeuvre [s] (tires settle after the drop).",
        )
        parser.add_argument(
            "--band", type=float, default=0.15, help="Time-to-band threshold as a fraction of the commanded yaw rate."
        )
        parser.add_argument(
            "--switch-pressure-turns",
            type=int,
            default=0,
            help="Toggle the CTIS setpoint high/low every N turns (0 = off).",
        )
        parser.add_argument(
            "--p-high", type=float, default=2.0, help="High CTIS setpoint [panel units, psi for the Sherp]."
        )
        parser.add_argument("--p-low", type=float, default=1.0, help="Low CTIS setpoint [panel units].")
        parser.add_argument(
            "--switch-stage-turns",
            type=int,
            default=0,
            help="Toggle the terrain stage a/b every M turns (0 = off; a stage change re-drops the vehicle).",
        )
        parser.add_argument("--stage-a", type=float, default=0.2)
        parser.add_argument("--stage-b", type=float, default=0.8)
        parser.add_argument(
            "--yaw-authority",
            type=float,
            default=1.0,
            help="Skid steer: measured / kinematic yaw rate per lever (1 = no-slip kinematics); scales the brake levers.",
        )
        parser.add_argument(
            "--log",
            type=str,
            default=None,
            help="Write per-frame signals + per-turn metrics to this .npz at num-frames.",
        )
        parser.add_argument(
            "--max-wheel-speed",
            type=float,
            default=None,
            help="Axle servo range [rad/s]; default = the vehicle asset's maxWheelSpeed.",
        )
        parser.add_argument(
            "--mujoco-terrain-cell",
            type=float,
            default=_MJ_TERRAIN_CELL,
            help="Cell of the hfield MuJoCo collides with [m] (0.25 = the tire grid itself).",
        )
        parser.add_argument(
            "--mujoco-terrain-agg",
            type=str,
            default="max",
            choices=("max", "mean"),
            help="Aggregation of the fine grid into the collider cell: max = rock envelope (rigid parts never inside rock), mean = below rock tops.",
        )
        parser.add_argument(
            "--sun-elevation", type=float, default=75.0, help="Sun elevation above the horizon [deg] (GL viewer)."
        )
        parser.add_argument(
            "--sun-azimuth",
            type=float,
            default=-60.0,
            help="Sun azimuth [deg] relative to the vehicle's initial heading (0 = behind the vehicle) (GL viewer).",
        )
        parser.add_argument(
            "--ambient", type=float, default=1.3, help="Ambient light level (GL viewer; 1 = the viewer default)."
        )
        parser.add_argument(
            "--camera",
            type=str,
            default="free",
            choices=VehicleTerrain.CAMERA_MODES,
            help="Start camera: free = the viewer's WASD / QE + mouse controls; follow = chase camera; top = above the car.",
        )
        parser.set_defaults(num_frames=3600)
        return parser
