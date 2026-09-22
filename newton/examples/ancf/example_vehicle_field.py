# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Vehicle on 4 ANCF FEM tires driven by hand over a boulder field whose relief changes live.

Extends vehicle_ancf_tires (3) with a ``newton-terrain-tool`` field (``kind: field``, e.g.
``field_01_boulders``): rocks stacked over hills, no track. The FEM tires contact the fine grid
(``TerrainSCM``), the rigid parts (rims, arms, chassis) a MuJoCo heightfield on the same grid.

* Terrain stage ``w``: ``heights = w * I_N`` applied live from the UI - the ANCF terrain grid,
  the MuJoCo collider and the rendered mesh all rescale, then the car is re-dropped onto the rocks
  (lifted by the highest rock under its tire footprints and hull).
* Driving: the base example's steer / throttle sliders and CTIS panel, plus the arrow keys
  (UP / DOWN throttle, LEFT / RIGHT steer with self-centring, SPACE stop).
* "Reset vehicle" puts the at-rest snapshot taken at construction back at the origin.
* Camera modes follow / top / free, debug overlays for the MuJoCo collider grid and its terrain
  contacts, and a watchdog that dumps the last frames and auto-resets the vehicle on a
  simulation fault (non-finite chassis state, runaway tread node or chassis speed).

Command: python -m newton.examples vehicle_field --vehicle-asset <usd> [--terrain field_01_boulders] [--difficulty 1.0]
(docker/config/06_vehicle_field.json starts at --difficulty 0.4)
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from collections import deque

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf import example_vehicle_ancf_tires as _dw
from newton.examples.ancf._field_task import FieldTerrain
from newton.examples.ancf._terrain_common import (
    FAR_BELOW,
    SPAWN_CLEARANCE,
    quat_roll_pitch,
    quat_yaw,
    set_follow_camera,
    terrain_contact_spikes,
)
from newton.examples.ancf._vehicle_usd import VehicleUSD
from newton.solvers import TerrainSCM

# MuJoCo hfield collider cell. The vendored mujoco_warp collects up to MJ_MAXHFPRISM = 128 prisms
# per geom (raised from MuJoCo's 50): the 1.69 x 0.46 m chassis box yawed 45 deg covers 8 x 8 cells
# of the 0.25 m grid = 128 prisms, so the rigid parts can use the same fine grid as the tires.
_MJ_TERRAIN_CELL = 0.25  # [m]
# Simulation-fault detection (the vehicle is reset, the state is garbage): tread nodes faster
# than four times the tire surface speed at the asset's maxWheelSpeed (self._fault_node_speed), or a
# rigid body faster than the car can drive, or any non-finite chassis state.
_FAULT_BODY_SPEED = 30.0  # [m/s]
# Arrow-key ramps: full lock in 0.5 s (a fast human at the wheel), full throttle in 2 s.
_KEY_STEER_TIME = 0.5  # [s]
_KEY_THROTTLE_TIME = 2.0  # [s]


def terrain_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "terrain")


# ── Kernels ───────────────────────────────────────────────────────────────────


@wp.kernel
def _terrain_mesh_points(
    h: wp.array2d[float], w: float, hx: float, hy: float, cell: float, ox: float, oy: float, pts: wp.array[wp.vec3]
):
    """Grid node (r, c) -> world vertex at stage w, grid centre at (ox, oy)."""
    r, c = wp.tid()
    ncol = h.shape[1]
    pts[r * ncol + c] = wp.vec3(ox - hx + float(c) * cell, oy - hy + float(r) * cell, w * h[r, c])


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


# ── Example ───────────────────────────────────────────────────────────────────


class Example(_dw.Example):
    """Vehicle + 4 ANCF tires on a newton-terrain-tool boulder field with a live stage."""

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
            f"[FIELD] {t.name}: {t.ncol}x{t.nrow} px  cell {t.cell} m  I_N max {t.max_h:.2f} m  grid {2 * t.hx:.0f}x{2 * t.hy:.0f} m  stage w = {t.w:.2f}"
        )
        # Flat plane off. The car is built at the origin heading +x, lifted above the rocks under
        # it: the base class's graph-capture warm-up steps this pose, and MuJoCo keeps warm-start
        # data the base does not restore, so it must be a legitimate drop, not a car buried in a
        # boulder. Resets restore this snapshot. The vehicle asset is needed before the base
        # builds (spindle footprint for the lift).
        vehicle_asset = getattr(args, "vehicle_asset", None)
        if not vehicle_asset:
            raise ValueError("--vehicle-asset is required (a vehicle USD baked by newton-tire-tool)")
        self.vehicle = VehicleUSD(_dw._asset_path(vehicle_asset))
        self._spindle_xy = self.vehicle.spindle_positions_zu()[:, :2].astype(np.float64)
        self._tire_r = float(self.vehicle.r_roll)
        self._belly = float(self.vehicle.belly)  # from the vehicle asset's chassis colliders / hull
        self._snapshot_z = self._lift_at_origin()
        args.ground_z = FAR_BELOW
        args.world_z_offset = self._snapshot_z
        super().__init__(viewer, args)

        # Throttle limit = the vehicle asset's maxWheelSpeed unless --max-wheel-speed overrides it;
        # wheel-speed ramp capped at the traction limit mu g / r_roll (see the track example).
        if args.max_wheel_speed is not None:
            self.spec = dataclasses.replace(self.spec, max_wheel_speed=float(args.max_wheel_speed))
        mu = float(getattr(args, "mu", _dw._MU))
        self._wheel_speed_rate = max(mu * _dw._GRAVITY / self.vehicle.r_roll * _dw._FRAME_DT, _dw._WHEEL_SPEED_RATE)
        # Watchdog threshold (diagnostic, not physics) [m/s]: a tread node on a rolling tire moves at
        # up to v_chassis + omega R = 2 omega R at full throttle without slip, so 2x that is the fault line.
        self._fault_node_speed = 4.0 * float(self.spec.max_wheel_speed) * float(self.spec.tire_R_outer)
        # + steer = left for the arrow keys; skid steer takes a lever command in [-1, 1].
        self._steer_sign = 1.0 if self._kingpin_axis_z > 0.0 else -1.0
        self._steer_range = 1.0 if self.vehicle.steering == "skid" else float(self.spec.max_steer)
        self.camera_mode = "follow"
        self.top_view_height = 30.0  # [m] above the car; 30 m at 65 deg fov shows ~38 m across

        # ── Device buffers ──
        dev = "cuda:0"
        self._chassis_body = _dw._find_body(self.model, "chassis")
        self._chassis_q = wp.zeros(1, dtype=wp.transform, device=dev)
        self._chassis_qd = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
        self._field_dev = wp.array(t.field, dtype=float, device=dev)
        self._terrain_points = wp.zeros(t.nrow * t.ncol, dtype=wp.vec3, device=dev)
        self._terrain_indices = wp.array(self._grid_triangles(t.nrow, t.ncol), dtype=wp.int32, device=dev)
        # MuJoCo collider overlay (the viewer's own "show collision" draws the Newton shape at the
        # stage it was built with; this one follows the stage).
        self.show_collider = False
        self._hc_dev = wp.array(self._hc, dtype=float, device=dev)
        self._collider_points = wp.zeros(self._hc.shape[0] * self._hc.shape[1], dtype=wp.vec3, device=dev)
        self._collider_indices = wp.array(
            self._grid_triangles(self._hc.shape[0], self._hc.shape[1]), dtype=wp.int32, device=dev
        )
        # Watchdog: three device-reduced scalars per frame, a rolling history dumped on the first
        # fault so a divergence can be traced back, not just noticed.
        self._diag_dev = wp.zeros(3, dtype=float, device=dev)
        self._history = deque(maxlen=90)
        self._fault = ""
        self.fault_count = 0
        self._refresh_terrain_mesh()
        # MuJoCo contacts with the terrain collider (debug overlay)
        self.show_mj_contacts = False
        n_con = int(self.solver.mjw_data.naconmax)
        self._mj_contact_pts = wp.full(n_con, wp.vec3(0.0, 0.0, FAR_BELOW), dtype=wp.vec3, device=dev)
        self._mj_contact_colors = wp.full(
            n_con, wp.vec3(1.0, 0.1, 0.1), dtype=wp.vec3, device=dev
        )  # the GL backend wants per-point colors
        self._mj_contact_radii = wp.full(n_con, 0.08, dtype=wp.float32, device=dev)

        # MuJoCo hfield elevation data (normalised, one hfield = the collider) for live stage changes.
        mjm = self.solver.mjw_model
        assert int(mjm.nhfield) == 1, f"expected exactly one MuJoCo hfield (the collider), got {mjm.nhfield}"
        assert int(mjm.nhfielddata) == self._hc_norm.size, "MuJoCo hfield data size does not match the collider grid"
        self._collider_shape = list(self.model.shape_label).index("terrain_collider")
        self._collider_hf_args = {
            "nrow": self._hc.shape[0],
            "ncol": self._hc.shape[1],
            "hx": 0.5 * (self._hc.shape[1] - 1) * self._hc_k * t.cell,
            "hy": 0.5 * (self._hc.shape[0] - 1) * self._hc_k * t.cell,
        }
        geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()[0]  # (ngeom,) for world 0
        hits = np.where(geom_to_shape == self._collider_shape)[0]
        assert len(hits) == 1, f"terrain_collider maps to {len(hits)} MuJoCo geoms"
        self._collider_geom = int(hits[0])
        self._apply_stage_to_collider()

        # ── Snapshot of the at-rest vehicle (origin, heading +x) for resets ──
        wp.synchronize_device(dev)
        self._snapshot = self._take_snapshot()
        self._chassis_joint_q0 = self._chassis_free_joint_q_start()

        self._pose = self._read_chassis()
        self._w_pending = t.w
        self._constructed = False

        # --debug-substeps: no frame graph, the substep loop runs on the host with a finiteness
        # check after every stage (kinematics, bead prescription, ANCF step, wrenches, dynamics).
        self._debug_substeps = bool(args.debug_substeps)
        if self._debug_substeps:
            self._substep_graph = None
            print("[FIELD] debug-substeps: frame graph disabled, checking every stage for non-finite values")

        if self._test:
            self._target_wheel_speed = 3.0  # [rad/s] drive straight over the rocks (~30 m in 1200 frames)
        self._constructed = True
        # scene light: sun elevation above the horizon and ambient level (GL renderer knobs)
        self.sun_elevation_deg = float(args.sun_elevation)
        self.sun_azimuth_deg = float(args.sun_azimuth)
        self.ambient = float(args.ambient)
        if viewer is not None:
            self._update_camera()
            if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
                viewer.camera.fov = 65.0
            self._apply_light()

    # ── Terrain hook (MJCF imported, ANCF solver built, nothing captured yet) ──

    def _on_car_builder(self, car: newton.ModelBuilder) -> None:
        t = self.terrain
        self.terrain_scm = TerrainSCM(
            heights=t.heights(),
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
            f"[FIELD] MuJoCo collider grid {hc.shape[1]}x{hc.shape[0]} at {k * t.cell:.2f} m (block {self._hc_mode} of the {t.cell} m grid)"
        )
        for s in range(car.shape_count):
            if car.shape_body[s] == -1 and car.shape_type[s] == newton.GeoType.PLANE:
                xf = car.shape_transform[s]
                car.shape_transform[s] = wp.transform(wp.vec3(xf.p[0], xf.p[1], FAR_BELOW), xf.q)
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
        updating the model it is asked to re-read it. MuJoCo was exported at build time and is
        untouched by this."""
        if self.viewer is None:
            return
        t = self.terrain
        hc_max = float(max(self._hc.max(), 1.0e-3))
        # data normalised by its own range -> hc / hc_max; world z = 0 + that * (w * hc_max) = w * hc
        self.model.shape_source[self._collider_shape] = newton.Heightfield(
            data=self._hc, min_z=0.0, max_z=max(t.w * hc_max, 1.0e-3), **self._collider_hf_args
        )
        self.viewer.set_model(self.model)
        # set_model drops the example's side-panel callback (newton.examples.run registers it once
        # after construction): put this panel back, otherwise the "Example" options vanish after
        # the first stage change. Not during construction, or run() would register a second copy.
        if self._constructed and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(lambda ui, ex=self: ex.gui(ui), position="side")

    def _apply_stage_to_collider(self) -> None:
        """MuJoCo elevation = w * hc / hc_max (the hfield spans [0, hc_max])."""
        self.solver.mjw_model.hfield_data.assign((self.terrain.w * self._hc_norm).astype(np.float32))

    def _refresh_terrain_mesh(self) -> None:
        t = self.terrain
        wp.launch(
            _terrain_mesh_points,
            dim=(t.nrow, t.ncol),
            inputs=[self._field_dev, t.w, t.hx, t.hy, t.cell, t.origin[0], t.origin[1], self._terrain_points],
            device="cuda:0",
        )
        nr, nc = self._hc.shape
        hx_c = 0.5 * (nc - 1) * self._hc_k * t.cell
        hy_c = 0.5 * (nr - 1) * self._hc_k * t.cell
        # coarse node (0, 0) sits on fine node (0, 0): centre = origin - (hx, hy) + (hx_c, hy_c)
        wp.launch(
            _terrain_mesh_points,
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

    @staticmethod
    def _grid_triangles(nrow: int, ncol: int) -> np.ndarray:
        r, c = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
        i0 = (r * ncol + c).ravel()
        i1 = i0 + 1
        i2 = i0 + ncol
        i3 = i2 + 1
        return (
            np.concatenate([np.stack([i0, i1, i3], axis=1), np.stack([i0, i3, i2], axis=1)], axis=0)
            .flatten()
            .astype(np.int32)
        )

    # ── Reset (snapshot + z shift) ─────────────────────────────────────────────

    def _take_snapshot(self) -> dict:
        a = self.ancf_solver
        return {
            "node_x": a.node_x.numpy().copy(),
            "node_D": a.node_D.numpy().copy(),
            "s0_bq": self.state_0.body_q.numpy().copy(),
            "s0_jq": self.state_0.joint_q.numpy().copy(),
        }

    def _chassis_free_joint_q_start(self) -> int:
        m = self.model
        child = m.joint_child.numpy()
        jtype = m.joint_type.numpy()
        qs = m.joint_q_start.numpy()
        for j in range(m.joint_count):
            if int(child[j]) == self._chassis_body:
                assert int(jtype[j]) == int(newton.JointType.FREE), "chassis joint is not a free joint"
                return int(qs[j])
        raise RuntimeError("no joint with the chassis as child")

    def _lift_at_origin(self) -> float:
        """Lift of the tire bottoms so nothing starts inside a rock at the origin: the highest rock
        under the four tire discs, or under the chassis / suspension footprint minus the belly
        clearance (rocks between the wheels can stand higher than the ones under the tires),
        plus the drop clearance [m]."""
        t = self.terrain
        tires = t.max_height_in_discs(self._spindle_xy, self._tire_r + t.cell)
        x0, x1, y0, y1 = self.vehicle.footprint
        body = t.max_height_in_rect(x0 - t.cell, x1 + t.cell, y0 - t.cell, y1 + t.cell) - self._belly
        return max(tires, body) + SPAWN_CLEARANCE

    def reset_vehicle(self) -> float:
        """Put the at-rest snapshot back at the origin heading +x, lifted to clear the rocks at the
        current stage; velocities, internal forces and EAS parameters zeroed. Returns the lift."""
        lift = self._lift_at_origin()
        dz = lift - self._snapshot_z
        snap = self._snapshot
        bq = snap["s0_bq"].copy()
        bq[:, 2] += dz
        jq = snap["s0_jq"].copy()
        jq[self._chassis_joint_q0 + 2] += dz
        nx = snap["node_x"].copy()
        nx[:, 1] += dz  # ANCF is Y-up: a1 is the world z
        a = self.ancf_solver
        for st in (self.state_0, self.state_rigid):
            st.body_q.assign(bq)
            st.body_qd.zero_()
            st.joint_q.assign(jq)
            st.joint_qd.zero_()
        a.node_x.assign(nx)
        a.node_D.assign(snap["node_D"])
        a.node_xd.zero_()
        a.node_xdd.zero_()
        a.node_Dd.zero_()
        a.node_Ddd.zero_()
        a.global_f_int.zero_()
        a.global_f_int0.zero_()
        a.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()
        self.solver.xfrc_applied.zero_()
        self.terrain_scm.node_f.zero_()
        # MuJoCo's Newton-solver warm start belongs to the previous state
        ws = getattr(self.solver.mjw_data, "qacc_warmstart", None)
        if ws is not None:
            ws.zero_()
        # controls to rest, then the same warm-up the constructor does
        self.steer_angle = 0.0
        self.wheel_speed = 0.0
        self._target_wheel_speed = 0.0
        self._update_controls()
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
        self._check_reset(nx)
        self._prescribe_beads()
        self._update_viz_buffers()
        self._history.clear()
        self._pose = self._read_chassis()
        print(f"[FIELD] vehicle reset at the origin, lift {lift:.2f} m, stage w = {self.terrain.w:.2f}")
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
        for e in range(_dw._N_TIRES):
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
        print(f"[FIELD] reset check: max bead mismatch vs MuJoCo spindles {1e3 * worst:.1f} mm")
        if worst > 0.05:
            raise RuntimeError(f"reset inconsistent: ANCF beads {worst:.3f} m from their spindles")

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
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) > self._wheel_speed_rate:
            self.wheel_speed += math.copysign(self._wheel_speed_rate - _dw._WHEEL_SPEED_RATE, d)
        super().step()
        self._watch()

    def _read_keys(self) -> None:
        """Arrow keys drive when the viewer has them (the follow camera re-aims every frame, so
        the viewer's own arrow-key camera nudge is harmless while it is on): UP / DOWN ramp the
        throttle, LEFT / RIGHT ramp the steer toward full lock and it self-centres when released,
        SPACE zeroes the throttle."""
        v = self.viewer
        if v is None or not hasattr(v, "is_key_down"):
            return
        try:
            up, down = v.is_key_down("up"), v.is_key_down("down")
            left, right = v.is_key_down("left"), v.is_key_down("right")
            brake = v.is_key_down("space")
        except Exception:
            return
        vmax = float(self.spec.max_wheel_speed)
        dv = vmax * _dw._FRAME_DT / _KEY_THROTTLE_TIME
        ds = self._steer_range * _dw._FRAME_DT / _KEY_STEER_TIME
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

    def simulate(self) -> None:
        if not self._debug_substeps or self._gs_coupler is not None:
            super().simulate()
            return
        ancf = self.ancf_solver
        dt = self._sim_dt

        def check(stage: str, k: int) -> None:
            bad = []
            for name, arr in (
                ("ancf.node_x", ancf.node_x),
                ("ancf.node_xd", ancf.node_xd),
                ("ancf.node_D", ancf.node_D),
                ("ancf.global_f_int", ancf.global_f_int),
                ("terrain.node_f", self.terrain_scm.node_f),
                ("mj.xfrc_applied", self.solver.xfrc_applied),
                ("mj.xpos", self.solver.xpos),
                ("mj.qpos", self.solver.mjw_data.qpos),
                ("mj.qvel", self.solver.mjw_data.qvel),
                ("mj.qacc", self.solver.mjw_data.qacc),
                ("state_rigid.body_q", self.state_rigid.body_q),
                ("state_0.body_q", self.state_0.body_q),
                ("state_0.joint_q", self.state_0.joint_q),
            ):
                v = arr.numpy()
                if not np.all(np.isfinite(v)):
                    bad.append(f"{name} ({int((~np.isfinite(v)).sum())} non-finite of {v.size})")
            if bad:
                nf = self.terrain_scm.node_f.numpy()
                xf = self.solver.xfrc_applied.numpy()
                raise RuntimeError(
                    f"[FIELD] non-finite after '{stage}' in substep {k} of frame {self._frame}: "
                    + "; ".join(bad)
                    + f"\n  |terrain node_f| max {np.nanmax(np.linalg.norm(nf, axis=1)):.3g} N   |xfrc| max {np.nanmax(np.abs(xf)):.3g}"
                    + f"   ncon {int(self.solver.mjw_data.ncon.numpy()[0]) if hasattr(self.solver.mjw_data, 'ncon') else '?'}"
                )

        if self._frame == 0:
            self._debug_placement()
        for k in range(self._substeps):
            self._gs_kinematics_fn(self.state_0, self.state_rigid, self.control, None, dt)
            check("step_kinematics", k)
            self._prescribe_beads(inv_dt=1.0 / dt, vel_predict_dt=dt)
            check("prescribe_beads (predict)", k)
            ancf.graph_step()
            check("ancf.graph_step", k)
            self._prescribe_beads()
            check("prescribe_beads", k)
            self._accumulate_wrenches()
            check("accumulate_wrenches", k)
            self._gs_dynamics_fn(self.state_rigid)
            check("step_dynamics", k)
            self._copy_rigid_to_state0()
            if self._frame == 0 and k < 3:
                self._debug_forces(k)
            if self._frame < 3:
                nf = np.linalg.norm(self.terrain_scm.node_f.numpy(), axis=1)
                xd = np.linalg.norm(ancf.node_xd.numpy(), axis=1)
                qd = self.state_0.body_qd.numpy()
                print(
                    f"[FIELD]   frame {self._frame} substep {k}: contact nodes {int((nf > 0).sum())}  |node_f| max {nf.max():.3g} N  "
                    f"|node_xd| max {xd.max():.3g} m/s  |body_qd| max {np.abs(qd).max():.3g}  |xfrc| max {np.abs(self.solver.xfrc_applied.numpy()).max():.3g}"
                )
        if self._frame == 0:
            nf = self.terrain_scm.node_f.numpy()
            print(
                f"[FIELD] debug frame 0 ok: |terrain node_f| max {np.linalg.norm(nf, axis=1).max():.3g} N  ncon {int(self.solver.mjw_data.ncon.numpy()[0]) if hasattr(self.solver.mjw_data, 'ncon') else '?'}"
            )

    def _dump_history(self, why: str) -> None:
        print(
            f"[FIELD] SIM FAULT: {why} at frame {self._frame}. Last frames (frame, steer, axle rad/s, chassis m/s, h, roll, pitch, contact nodes, max|node_f| N, max|node_xd| m/s):"
        )
        for rec in self._history:
            print("   ", rec)
        if self._test:
            raise RuntimeError(f"[FIELD] simulation diverged: {why}")

    def _debug_placement(self) -> None:
        """Frame-0 audit: the origin the contact kernel reads, and per tire the lowest node against
        the terrain height under it (host sampling of the same field)."""
        t = self.terrain
        print(
            f"[FIELD]   kernel origin array {self.terrain_scm._origin.numpy().tolist()}  FieldTerrain.origin {t.origin}  h0 max {float(self.terrain_scm.h0.numpy().max()):.2f} m"
        )
        x = self.ancf_solver.node_x.numpy()  # Y-up: (y, z, x)
        nn = self._n_nodes
        for e, (label, _) in enumerate(_dw._WHEEL_ORDER):
            xe = x[e * nn : (e + 1) * nn]
            i = int(np.argmin(xe[:, 1]))
            wx, wy, wz = float(xe[i, 2]), float(xe[i, 0]), float(xe[i, 1])
            zs = [wz - t.height_at(float(p[2]), float(p[0])) for p in xe]
            print(
                f"[FIELD]   tire {label}: lowest node world ({wx:+.2f}, {wy:+.2f}, {wz:.2f})  terrain there {t.height_at(wx, wy):.2f}  min node clearance {min(zs):+.3f} m"
            )
        gp = self.solver.mjw_model.geom_pos.numpy()[0, self._collider_geom]
        gx = self.solver.mjw_data.geom_xpos.numpy()[0, self._collider_geom]
        print(f"[FIELD]   MuJoCo collider geom_pos {gp.tolist()}  geom_xpos (world, used by contacts) {gx.tolist()}")

    def _debug_forces(self, k: int) -> None:
        """Which body moves, and which MuJoCo force term drives it."""
        d = self.solver.mjw_data
        qd = self.state_0.body_qd.numpy()
        mag = np.abs(qd)
        b, comp = np.unravel_index(int(np.argmax(mag)), mag.shape)
        labels = list(getattr(self.model, "body_label", getattr(self.model, "body_key", [])))
        name = labels[b] if b < len(labels) else str(b)
        print(
            f"[FIELD]   substep {k}: fastest body '{name}' component {comp} ({'lin' if comp < 3 else 'ang'}) = {qd[b, comp]:+.3g}   body_qd row {np.round(qd[b], 3).tolist()}"
        )
        for term in (
            "ctrl",
            "qfrc_actuator",
            "qfrc_constraint",
            "qfrc_passive",
            "qfrc_smooth",
            "qfrc_applied",
            "qvel",
            "qacc",
        ):
            arr = getattr(d, term, None)
            if arr is None:
                continue
            v = arr.numpy()[0]
            i = int(np.argmax(np.abs(v)))
            print(f"[FIELD]     {term:16s} max |.| {abs(v[i]):.3g} at dof {i}")
        tq = self.control.joint_target_qd.numpy()
        print(f"[FIELD]     joint_target_qd nonzero {np.flatnonzero(tq).tolist()} -> {tq[np.flatnonzero(tq)].tolist()}")

    # ── Watchdog ───────────────────────────────────────────────────────────────

    def _read_chassis(self) -> np.ndarray:
        wp.copy(self._chassis_q, self.state_0.body_q, src_offset=self._chassis_body, count=1)
        return self._chassis_q.numpy()[0]

    def _watch(self) -> None:
        """Chassis pose for the camera + the fault test (one transform, one spatial vector and
        three reduced floats read back per frame)."""
        q = self._read_chassis()
        wp.copy(self._chassis_qd, self.state_0.body_qd, src_offset=self._chassis_body, count=1)
        qd = self._chassis_qd.numpy()[0]
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
            self._fault = "non-finite chassis state"
            self._dump_history(self._fault)
            return  # step() resets the vehicle
        self._pose = q
        t = self.terrain
        x, y, z = float(q[0]), float(q[1]), float(q[2])
        roll, pitch = quat_roll_pitch(q)
        self._diag_dev.zero_()
        wp.launch(
            _diag_reduce,
            dim=_dw._N_TIRES * self._n_nodes,
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

    # ── GUI ────────────────────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        t = self.terrain
        ui.text(f"Field  {t.name}   {2 * t.hx:.0f} x {2 * t.hy:.0f} m   I_N max {t.max_h:.1f} m")
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
        c3, self.ambient = ui.slider_float("ambient light", self.ambient, 0.2, 1.6)
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
        if ui.button("Reset vehicle (snapshot at the origin)"):
            self.reset_vehicle()
        ui.same_line()
        ui.text(f"sim faults {self.fault_count}")
        _c, self.show_mj_contacts = ui.checkbox(
            "Show MuJoCo contacts with the terrain collider (red points)", self.show_mj_contacts
        )
        _c, self.show_collider = ui.checkbox(
            f"Show MuJoCo collider grid ({self._hc_k * t.cell:g} m block {self._hc_mode}, follows the stage)",
            self.show_collider,
        )
        ui.separator()

        # the base example's steer / throttle sliders + CTIS panel + live readouts
        ui.text("keys: UP / DOWN throttle   LEFT / RIGHT steer (self-centring)   SPACE stop")
        super().gui(ui)

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
            r.ambient_sky = tuple(min(1.0, self.ambient * c) for c in (0.8, 0.8, 0.85))
        if hasattr(r, "ambient_ground"):
            r.ambient_ground = tuple(min(1.0, self.ambient * c) for c in (0.45, 0.45, 0.5))

    def _update_camera(self) -> None:
        """follow = chase camera behind the car; top = straight down from above the car, heading up
        on screen; free = leave the viewer's own navigation alone."""
        if self.camera_mode == "follow":
            set_follow_camera(self.viewer, self._pose)
        elif self.camera_mode == "top":
            q = self._pose
            yaw = quat_yaw(q)
            self.viewer.set_camera(
                pos=wp.vec3(float(q[0]), float(q[1]), float(q[2]) + self.top_view_height),
                pitch=-89.0,
                yaw=math.degrees(yaw),
            )

    def render(self) -> None:
        if self.viewer is None:
            return
        self._update_camera()
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/field/terrain",
            self._terrain_points,
            self._terrain_indices,
            color=(0.45, 0.43, 0.40),
            backface_culling=False,
            hidden=self.show_collider,
        )
        self.viewer.log_mesh(
            "/field/collider",
            self._collider_points,
            self._collider_indices,
            color=(0.9, 0.3, 0.9),
            backface_culling=False,
            hidden=not self.show_collider,
        )
        if self.show_mj_contacts:
            d = self.solver.mjw_data
            wp.launch(
                _mj_contact_points,
                dim=len(self._mj_contact_pts),
                inputs=[d.contact.pos, d.contact.geom, d.nacon, self._collider_geom, FAR_BELOW, self._mj_contact_pts],
                device="cuda:0",
            )
        self.viewer.log_points(
            "/field/mj_contacts",
            self._mj_contact_pts,
            radii=self._mj_contact_radii,
            colors=self._mj_contact_colors,
            hidden=not self.show_mj_contacts,
        )
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
        self.viewer.end_frame()

    # ── Diagnostics / tests ────────────────────────────────────────────────────

    def _print_diag(self, fps: float = 0.0) -> None:
        super()._print_diag(fps)
        q = self._pose
        t = self.terrain
        roll, pitch = quat_roll_pitch(q)
        print(
            f"  steer {self.steer_angle:+.3f}  axle {self.wheel_speed:.1f}/{self._target_wheel_speed:.1f} rad/s  "
            f"chassis ({float(q[0]):+.1f}, {float(q[1]):+.1f})  h={float(q[2]) - t.height_at(float(q[0]), float(q[1])):.2f} m  "
            f"roll={math.degrees(roll):+.0f} pitch={math.degrees(pitch):+.0f} deg  stage w={t.w:.2f}  faults {self.fault_count}"
        )

    def test_final(self) -> None:
        # The base F_z check assumes four tires on a plane; on rocks the loads are anything.
        x_np = self.ancf_solver.node_x.numpy()
        assert np.all(np.isfinite(x_np)), "non-finite node_x at test_final"
        assert self.fault_count == 0, f"FAIL: {self.fault_count} simulation faults"
        q = self._pose
        h = float(q[2]) - self.terrain.height_at(float(q[0]), float(q[1]))
        assert h > 0.0, f"FAIL: chassis {h:.2f} m below the terrain"
        assert float(q[0]) > 1.0, f"FAIL: the vehicle drove only {float(q[0]):.1f} m forward"
        print(
            f"[PASS] {self.terrain.name} w={self.terrain.w:.2f}: drove {float(q[0]):.1f} m, chassis {h:.2f} m above the terrain"
        )

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = _dw.Example.create_parser()
        parser.add_argument(
            "--terrain",
            type=str,
            default="field_01_boulders",
            help="Field terrain under examples/ancf/assets/terrain/ (newton-terrain-tool, kind: field).",
        )
        parser.add_argument(
            "--difficulty",
            type=float,
            default=1.0,
            help="Terrain stage w in [0, 1]: heights = w * I_N (changeable in the UI).",
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
            "--sun-elevation", type=float, default=60.0, help="Sun elevation above the horizon [deg] (GL viewer)."
        )
        parser.add_argument("--sun-azimuth", type=float, default=-60.0, help="Sun azimuth [deg] (GL viewer).")
        parser.add_argument("--ambient", type=float, default=1.0, help="Ambient light level (GL viewer).")
        parser.add_argument(
            "--debug-substeps",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Run the substep loop without the frame graph and stop at the first non-finite stage (slow; diagnosis only).",
        )
        parser.set_defaults(num_frames=1200)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
