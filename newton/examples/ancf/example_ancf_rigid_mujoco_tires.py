# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""N parallel ANCF FEM tires, each on its own MuJoCo single-wheel test rig.

Rig per environment (Chrono ChTireTestRig topology, built with ModelBuilder — no MJCF):
``world -slide_x- chassis -slide_z- spindle``. slide_x is kinematic at the no-slip speed
``rim_omega * r_roll`` (joint_qd written before step_kinematics; a lin_motor velocity
actuator is a soft backup), slide_z is free so MuJoCo carries the vertical load, and the
wheel spin is self-tracked (``rim_phi`` / ``rim_omega`` device arrays, no hinge). The
spindle body is the vehicle's own wheel (--vehicle-asset, mass / inertia / hub shapes) or
the tire asset's baked ``/Tire/Spindle`` prim.

Architecture
------------
  Rigid rigs     - SolverMuJoCo (Z-up, N worlds, one rig per world)
  FEM tires x N  - SolverANCFShellRigid (Y-up, n_envs=N, batched)

Coupling per substep (one captured CUDA graph, launched ``substeps`` times per frame):
  1. rim_phi += rim_omega dt;  joint_qd[slide_x] = rim_omega r_roll;  lin_motor target
  2. step_kinematics -> xpos / cvel on GPU
  3. prescribe ANCF bead nodes from the spindle pose (+ v dt predictor)   rigid -> soft
  4. mass-proportional damping into node_f_ext_persistent; ANCF NR+PCG step
  5. re-pin the beads; accumulate_wheel_wrenches -> xfrc_applied          soft -> rigid
     (+ --f-load preload; the forward component is zeroed since slide_x is kinematic)
  6. step_dynamics; re-pin joint_qd[slide_x]; copy state_rigid -> state_0

Solver budget (substeps / NR / PCG, CFL and HHT stability): docker/ANCF_FINDINGS.md.

Coordinate systems
------------------
  MuJoCo Z-up : x_fwd, y_lat, z_up
  ANCF Y-up   : x_lat, y_up,  z_fwd  (axle along X, tread at Y=0)
  Z-up -> Y-up : (x,y,z) -> (y, z, x)
  Y-up -> Z-up : (x,y,z) -> (z, x, y)

Command: python -m newton.examples ancf_rigid_mujoco_tires
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.ancf._ancf_viz import (
    _ancf_yup_to_zu,
    _build_ring_lines,
    _gather_zu,
    bead_row_indices,
    material_row,
    quad_triangles,
    ring_segments,
)
from newton.examples.ancf._vehicle_usd import VehicleUSD, find_body
from newton.solvers import SolverANCFShellRigid, isotropic_ancf_material, load_ancf_tire_usd

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# ── Shell defaults (overridden by the --shell-tires JSON entries) ─────────────
# Mesh geometry (radii, width, n_circ, section divisions) is baked into the tire asset.
_H_SHELL = 0.006  # [m] shell thickness
_E_TIRE = 1.0e7  # [Pa]
_NU_TIRE = 0.45
_RHO_TIRE = 700.0  # [kg/m^3]
_ALPHA_D = 0.0  # Rayleigh stiffness damping [s]; HHT alpha supplies the numerical damping by default
_PRESSURE = 30_000.0  # [Pa] gauge inflation pressure

# ── Solver parameter defaults (overridden by CLI args) ───────────────────────
_SIM_SUBSTEPS = 20
_NR_ITERS = 16
_PCG_ITERS = 25
_FRAME_DT = 1.0 / 60.0

# ── Vertical spindle load (simulates vehicle body weight) ─────────────────────
# Clamped at runtime to kn * 0.025 m (beyond ~25 mm penetration the shell inverts).
# F_load acts downward (-Z in MuJoCo Z-up) on each spindle independently.
_F_LOAD = 0.0  # [N] per-spindle downward load (0 = tire weight only)

# ── Ground contact defaults (overridden by CLI args / the tire asset) ─────────
_KN = 10_000.0  # [N/m]
_KD = 20.0  # [N·s/m]
_MU = 0.9

# ── Test rig ──────────────────────────────────────────────────────────────────
_DROP_CLEARANCE = 0.000  # [m] quasi-static start (tread at ground, no impact)
_GRAVITY = 9.81  # [m/s^2]
_RPM_RATE = 4.0  # [rpm/frame] max RPM change per step() call

# Rig joints (Chrono ChTireTestRig topology).
_KV_LIN_MOTOR = 5000.0  # [N·s/m] slide_x velocity actuator gain (≈ kinematic no-slip forward speed)
_D_SLIDE_X = 5.0  # [N·s/m] slide_x joint damping
_D_SLIDE_Z = 200.0  # [N·s/m] slide_z damping: v_term = M·g/D = 31.88·9.81/200 ≈ 1.56 m/s


# ── Warp kernels ──────────────────────────────────────────────────────────────


@wp.kernel
def _update_mass_damp(
    node_xd: wp.array[wp.vec3],
    lm_flat: wp.array[float],  # lumped_mass [n_nodes*6]; lm[6*i]=pos-DOF mass
    alpha_m: float,  # mass-proportional Rayleigh coeff [s^-1]
    f_pers: wp.array[wp.vec3],  # node_f_ext_persistent (written, not added)
):
    """Write mass-proportional damping into node_f_ext_persistent each substep.

    f_damp[i] = -alpha_m * m_node[i] * xd[i]; damps the low-frequency free
    oscillations that HHT barely attenuates (omega dt << 1).

    Derivation: alpha_m = 2*zeta*omega_n where omega_n is the constrained
    tire breathing frequency and zeta=0.10 (10% critical damping).
    """
    i = wp.tid()
    m = lm_flat[i * 6]  # position-DOF mass (same for x,y,z by isotropy)
    xd = node_xd[i]
    f_pers[i] = wp.vec3(-alpha_m * m * xd[0], -alpha_m * m * xd[1], -alpha_m * m * xd[2])


@wp.kernel
def _advance_rim_phi_batched(
    rim_phi: wp.array[float],  # (N,) in/out
    rim_omega: wp.array[float],  # (N,) rad/s
    dt: float,
):
    """dim=N. Kinematic rim angle integration: phi += omega * dt (spin is self-tracked, no hinge DOF)."""
    e = wp.tid()
    rim_phi[e] = rim_phi[e] + rim_omega[e] * dt


@wp.kernel
def _spin_spindle_body_q(
    body_q: wp.array[wp.transform],
    rim_phi: wp.array[float],
    spindle_idx: int,
    n_bodies_per_world: int,
):
    """Write the kinematic rim angle into the spindle body orientation (Z-up, axle = +Y).

    The MuJoCo spindle has no hinge DOF (spin lives in rim_phi), so its xquat stays
    identity; without this the spindle mesh would render without turning. Same sense as
    _prescribe_beads_gpu: +phi about the ANCF +X axle == +Y in Z-up (cyclic axis map).
    MuJoCo never reads body_q back (update_data_interval=0), so this is state-only.
    """
    e = wp.tid()
    b = spindle_idx + e * n_bodies_per_world
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), rim_phi[e])
    body_q[b] = wp.transform(wp.transform_get_translation(body_q[b]), q)


@wp.kernel
def _set_rolling_ctrl(
    rim_omega: wp.array[float],  # (N,) self-tracked, not from cvel
    joint_target_vel: wp.array[float],
    slide_x_qd_dof: int,
    n_qd_per_world: int,
    R_outer: float,
):
    """dim=N. No-slip rolling: lin_motor target = rim_omega * R_outer (Chrono ChLinkMotorLinearSpeed)."""
    env = wp.tid()
    joint_target_vel[slide_x_qd_dof + env * n_qd_per_world] = rim_omega[env] * R_outer


@wp.kernel
def _prescribe_slide_x_vel(
    joint_qd: wp.array[float],
    rim_omega: wp.array[float],  # (N,) rad/s
    slide_x_qd_dof: int,
    n_qd_per_world: int,
    r_outer: float,
):
    """dim=N.  Write joint_qd[slide_x] = omega * R BEFORE step_kinematics.

    step_kinematics then integrates joint_q[slide_x] += (omega * R) * dt, giving exact
    no-slip rolling. The lin_motor stays as a soft-constraint backup but contributes
    near-zero force in steady state.
    """
    e = wp.tid()
    joint_qd[slide_x_qd_dof + e * n_qd_per_world] = rim_omega[e] * r_outer


@wp.kernel
def _prescribe_beads_gpu(
    xpos: wp.array2d[wp.vec3],
    cvel: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    rim_phi: wp.array[float],
    rim_omega: wp.array[float],  # (N,) self-tracked spin — NOT from cvel[ang_y]
    bead_idx: wp.array[wp.int32],
    bead_rest: wp.array[wp.vec3],
    bead_D0: wp.array[wp.vec3],
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_xdd: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
    n_bead: int,
    n_nodes: int,
    lateral_offsets: wp.array[float],  # (N,) ANCF X (= Z-up Y) of each tire's hub; the MuJoCo rigs all sit at Y = 0
    R_outer: float,
    vel_predict_dt: float,  # > 0: linearly extrapolate hub pos by vel × dt (reduces coupling lag O(dt²))
    inv_dt: float,  # > 0: write bead accel as d(prescribed velocity)/dt; 0: leave accel alone
):
    """Prescribe bead nodes: rotate rest offset by phi, translate to hub position.

    Spin is kinematic (rim_omega from GPU array, not MuJoCo cvel[ang_y]).
    Forward velocity = rim_omega × R_outer (no-slip).
    Vertical velocity from cvel[spindle][5] (MuJoCo vz = slide_z velocity).
    MuJoCo Z-up → ANCF Y-up: (x,y,z)→(y,z,x).
    vel_predict_dt: extrapolate hub position by spindle velocity × dt before prescribing.
    This upgrades interface accuracy from O(dt) → O(dt²), stabilising at 2× larger substep dt.
    """
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]

    pos_zu = xpos[env, spindle_mj]
    sv = cvel[env, spindle_mj]

    omega = rim_omega[env]  # kinematic spin — self-tracked, not from MuJoCo
    v_fwd = rim_omega[env] * R_outer  # no-slip forward velocity = omega × R

    # Linear predictor: advance hub position by spindle velocity × dt.
    # Converts first-order coupling lag (pos from prev step) to second-order.
    # vel_predict_dt = 0.0 → original behaviour (no prediction).
    v_vert = sv[5]  # MuJoCo cvel: [0:3]=angular, [3:6]=linear; sv[5] = vz (vertical)
    v_fwd_mj = sv[3]  # MuJoCo vx (forward)

    hub_x = pos_zu[1] + lateral_offsets[env]
    hub_y = pos_zu[2] + v_vert * vel_predict_dt  # ANCF Y = MuJoCo Z (vertical)
    hub_z = pos_zu[0] + v_fwd_mj * vel_predict_dt  # ANCF Z = MuJoCo X (forward)

    phi = rim_phi[env]
    r = bead_rest[i]
    d0 = bead_D0[i]

    cp = wp.cos(phi)
    sp = wp.sin(phi)
    y_loc = r[1] * cp - r[2] * sp
    z_new = r[1] * sp + r[2] * cp

    node_x[global_idx] = wp.vec3(hub_x + r[0], hub_y + y_loc, hub_z + z_new)
    # Material velocity of a hub-attached point: hub velocity (slide_z vertical, no-slip
    # forward from slide_x) + spin x r. With the consistent ANCF mass matrix, a bead velocity
    # that disagrees with the bead motion lands as an inertia error on the sidewall nodes next
    # to the hub every substep ("soggy" rim), so the hub velocity must be included.
    v_new = wp.vec3(0.0, -omega * z_new + v_vert, omega * y_loc + v_fwd)
    d_y = d0[1] * cp - d0[2] * sp
    d_z = d0[1] * sp + d0[2] * cp
    dd_new = wp.vec3(0.0, -omega * d_z, omega * d_y)

    # Bead acceleration consistent with the prescribed velocity (finite difference of the
    # previous prescribed velocity); only on the pre-step call.
    if inv_dt > 0.0:
        node_xdd[global_idx] = (v_new - node_xd[global_idx]) * inv_dt
        node_Ddd[global_idx] = (dd_new - node_Dd[global_idx]) * inv_dt

    node_xd[global_idx] = v_new
    node_D[global_idx] = wp.vec3(d0[0], d_y, d_z)
    node_Dd[global_idx] = dd_new


@wp.kernel
def _zero_xfrc_forward(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
):
    """Zero the forward (MuJoCo X) force component of xfrc_applied.

    slide_x is kinematically prescribed — ANCF forward forces must NOT reach
    step_dynamics.  At 300 RPM the forward force can be O(10 kN), creating a
    velocity impulse large enough to overflow joint_qd and produce NaN.
    slide_z (vertical) is free and remains two-way; torques are scaled by the
    solver's torque_alpha (0 by default).
    """
    e = wp.tid()
    cur = xfrc_applied[e, spindle_mj]
    xfrc_applied[e, spindle_mj] = wp.spatial_vector(
        0.0,
        cur[1],
        cur[2],
        cur[3],
        cur[4],
        cur[5],
    )


@wp.kernel
def _fill_spindle_positions_env(
    body_q: wp.array[wp.transform],  # state_0.body_q (Z-up transforms)
    spindle_idx: int,  # world-0 Newton body index
    n_bodies_world: int,  # bodies per world
    out: wp.array[wp.vec3],  # (N*n_bead,)
    n_bead: int,
    lateral_offsets: wp.array[float],  # (N,) Y offset of each tire in Z-up (= ANCF X)
):
    """Fill spoke-start positions from body_q + the tire's lateral display offset.
    MuJoCo spindles share Y=0; add the tire's offset so spokes point to the right tire.
    """
    tid = wp.tid()
    env = tid // n_bead
    idx = env * n_bodies_world + spindle_idx
    p = wp.transform_get_translation(body_q[idx])
    out[tid] = wp.vec3(p[0], p[1] + lateral_offsets[env], p[2])


@wp.kernel
def _apply_vertical_load(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    f_load: float,  # downward force per spindle [N], positive = down
):
    """Apply constant vertical (downward) preload to each spindle (vehicle body weight).

    xfrc layout (mujoco_warp FORCE FIRST): [2]=fz, negative = downward in Z-up.
    """
    env = wp.tid()
    cur = xfrc_applied[env, spindle_mj]
    xfrc_applied[env, spindle_mj] = wp.spatial_vector(
        cur[0],
        cur[1],
        cur[2] - f_load,
        cur[3],
        cur[4],
        cur[5],
    )


@wp.kernel
def _gather_contact_spikes(
    node_x: wp.array[wp.vec3],  # ANCF Y-up, all envs flat
    ground_y: float,  # ground level in ANCF Y-up (= 0.0)
    vis_scale: float,  # penetration amplification for visibility
    line_starts: wp.array[wp.vec3],  # Z-up output
    line_ends: wp.array[wp.vec3],  # Z-up output
):
    """GPU-only contact visualization: a spike of height pen*vis_scale per penetrating node.

    Non-contact nodes emit a zero-length segment (invisible).
    ANCF Y-up → Z-up: (x, y, z) → (z, x, y); the spike goes up (+z) from the ground plane.
    """
    i = wp.tid()
    p = node_x[i]  # ANCF Y-up
    pen = ground_y - p[1]  # positive = inside ground

    base = wp.vec3(p[2], p[0], ground_y)  # clamp to ground surface for spike base

    if pen > 0.0:
        line_starts[i] = base
        line_ends[i] = wp.vec3(base[0], base[1], base[2] + pen * vis_scale)
    else:
        line_starts[i] = base
        line_ends[i] = base  # zero-length → invisible


# ── Example class ─────────────────────────────────────────────────────────────


class Example:
    """N ANCF FEM tires on MuJoCo single-wheel test rigs — the rigid <-> soft coupling test."""

    def __init__(self, viewer=None, args=None):
        device = "cuda:0"
        if args.fast_math:
            wp.config.fast_math = True  # ~5-15% on stiffness; risks NaN with E<2MPa
        wp.init()

        self._frame = 0
        self._t = 0.0
        self.viewer = viewer
        self._diag_period = int(args.diag_period)
        self._t_wall = time.perf_counter()
        self._t_step = 0.0
        self._t_render = 0.0
        # GUI live readouts (updated in _print_diag)
        self._gui_sp_z = 0.0
        self._gui_fz = 0.0
        self._gui_bead_drift = 0.0
        self._gui_v_max = 0.0
        self._gui_fps = 0.0
        self._gui_slip_vel = 0.0  # instantaneous: v_hub - omega*R [m/s]
        self._gui_pos_slip = 0.0  # cumulative: hub_x - phi*R [m]
        self._gui_v_roll = 0.0  # omega * R_outer [m/s]
        self._gui_v_fwd = 0.0  # hub forward velocity [m/s]
        drop_clearance = float(args.drop_clearance)

        # ── Tire configs from --shell-tires JSON list; n_envs = len(list) ───────
        # Each entry defines one parallel environment (tire). All tires share the mesh and
        # material of the first entry (batched solver); pressure and position are per env.
        tires_raw = args.shell_tires
        if isinstance(tires_raw, str):
            parsed = json.loads(tires_raw)
            tire_cfgs = parsed if isinstance(parsed, list) else [parsed]
        elif isinstance(tires_raw, list):
            tire_cfgs = tires_raw if tires_raw else [{}]
        elif isinstance(tires_raw, dict):
            tire_cfgs = [tires_raw]
        else:
            tire_cfgs = [{}]

        n_envs = len(tire_cfgs) if args.n_envs is None else int(args.n_envs)
        # Pad (repeating the last entry) / trim the configs to --n-envs.
        while len(tire_cfgs) < n_envs:
            tire_cfgs.append(tire_cfgs[-1])
        tire_cfgs = tire_cfgs[:n_envs]

        tire_cfg = tire_cfgs[0]  # geometry/material from first entry (shared across envs)
        pressures = [float(c.get("pressure", _PRESSURE)) for c in tire_cfgs]

        # Mesh geometry is fixed at bake time (third_party/newton-tire-tool/scripts/bake_tire.py).
        # Only material, thickness and pressure are runtime-overridable (applied to the loaded
        # mesh below). "thickness": null keeps the asset's baked per-section shell thickness.
        e_tire = float(tire_cfg.get("E", _E_TIRE))
        nu_tire = float(tire_cfg.get("nu", _NU_TIRE))
        rho_tire = float(tire_cfg.get("rho", _RHO_TIRE))
        h_cfg = tire_cfg.get("thickness", _H_SHELL)
        h_shell = None if h_cfg is None else float(h_cfg)
        alpha_d = float(tire_cfg.get("alpha-damp", _ALPHA_D))
        # --vehicle-asset: the rig's spindle is that vehicle's own wheel body (mass, inertia,
        # hub / axle shapes) and the tire defaults to the vehicle's tire; without it the tire
        # asset's baked /Tire/Spindle stands in.
        vehicle_asset_arg = args.vehicle_asset
        self.vehicle = None
        if vehicle_asset_arg:
            vpath = (
                vehicle_asset_arg if os.path.isabs(vehicle_asset_arg) else os.path.join(_ASSETS_DIR, vehicle_asset_arg)
            )
            self.vehicle = VehicleUSD(vpath)
        tire_asset_arg = args.tire_asset or (self.vehicle.default_tire_asset if self.vehicle else None)
        if not tire_asset_arg:
            raise ValueError("--tire-asset (or --vehicle-asset with a defaultTireAsset) is required")

        # kn / kd / pcg default to the tire asset's recommendation (resolved after the tire is
        # loaded below); the CLI / JSON override when given.
        kn_arg = args.kn
        kd_arg = args.kd
        pcg_arg = args.pcg_iters
        mu = float(args.mu)
        nr_iters = int(args.nr_iters)
        substeps = int(args.substeps)
        thickness_gp = int(tire_cfg.get("thickness-gp", args.thickness_gp))

        sim_dt = _FRAME_DT / substeps
        mat = isotropic_ancf_material(E=e_tire, nu=nu_tire, rho=rho_tire, alpha_damp=alpha_d)

        # ── ANCF tire mesh (Y-up: axle along X, tread at Y=0) ────────────────
        tire_asset_path = tire_asset_arg if os.path.isabs(tire_asset_arg) else os.path.join(_ASSETS_DIR, tire_asset_arg)
        self.ancf_model, tire_meta = load_ancf_tire_usd(tire_asset_path, device=device)
        kn = float(kn_arg) if kn_arg is not None else float(tire_meta.contact_kn or _KN)
        kd = float(kd_arg) if kd_arg is not None else kn * (_KD / _KN)  # same damping ratio as the baseline
        pcg_iters = int(pcg_arg) if pcg_arg is not None else int(tire_meta.pcg_iters or _PCG_ITERS)
        if tire_meta.spindle is None:
            raise ValueError(
                f"{tire_asset_path} has no /Tire/Spindle prim — re-bake it with newton-tire-tool "
                "(bake_tire.py --spindle-* / bake_sherp_ancf_tire.py)."
            )
        n_elems = self.ancf_model.n_elems
        self.ancf_model.elem_mat = wp.array(np.tile(material_row(mat), (n_elems, 1)), dtype=float, device=device)
        if h_shell is None and tire_meta.shell_thickness is not None:
            h_shell = float(tire_meta.shell_thickness)  # the asset's validated uniform thickness
        if h_shell is None:
            # Keep the baked per-section thickness; use its mean for the lumped mass estimates below.
            h_shell = float(self.ancf_model.elem_h.numpy().mean())
        else:
            self.ancf_model.elem_h = wp.array(np.full(n_elems, h_shell, dtype=np.float32), device=device)

        r_outer, r_inner, width = tire_meta.R_outer, tire_meta.R_inner, tire_meta.width
        n_circ = tire_meta.n_circ
        n_bead_rows = tire_meta.n_bead_rows
        n_ax_divs = n_elems // n_circ
        n_bead_per_ring = n_circ
        n_bead = 2 * n_bead_rows * n_bead_per_ring

        n_nodes = self.ancf_model.n_nodes
        x0_np = self.ancf_model.node_x0.numpy()
        d0_np = self.ancf_model.node_D0.numpy()
        d0_np_tiled = np.tile(d0_np, (n_envs, 1)).astype(np.float32)

        # Rolling / contact radius = outermost node radius. Equals R_outer for a
        # smooth crown; for a lugged bake (Sherp) it is the lug-top radius, so the
        # tread starts exactly at ground level and no-slip v = omega * r_roll.
        r_roll = float(np.sqrt(x0_np[:, 1] ** 2 + x0_np[:, 2] ** 2).max())
        self._r_outer = r_roll  # the radius _prescribe_beads / the slide_x kinematics use

        drop_h = r_roll + drop_clearance
        env_spacing = float(args.env_spacing)  # [m] lateral gap between tires (default)
        # Per-tire positions from config (ANCF X = lateral); fallback to uniform spacing.
        tire_positions = [float(tire_cfgs[e].get("position", [e * env_spacing, 0.0, 0.0])[0]) for e in range(n_envs)]
        drop_heights = [drop_h] * n_envs  # all tires at same height; spaced laterally

        m_tire = rho_tire * h_shell * (2.0 * math.pi * r_outer * width + 2.0 * math.pi * (r_outer**2 - r_inner**2))
        fz_tare = m_tire * 9.81

        # ── Mass-proportional Rayleigh damping (auto-derived) ─────────────────
        _l_sw = math.sqrt((r_outer - r_inner) ** 2 + (width / 2.0) ** 2)
        _k_sw = e_tire * h_shell * 2.0 * math.pi * r_outer / _l_sw
        _m_free = m_tire * (1.0 - n_bead / n_nodes)
        _omega_n = math.sqrt(_k_sw / max(_m_free, 1e-9))
        alpha_m_damp = float(getattr(args, "alpha_m_damp", 2.0 * 0.10 * _omega_n))

        _f_load_raw = float(args.f_load)
        _f_load_safe = kn * 0.025  # 25mm max penetration = safe shell deformation
        if _f_load_raw > _f_load_safe:
            print(
                f"[WARN] --f-load {_f_load_raw:.0f}N exceeds safe limit "
                f"{_f_load_safe:.0f}N at kn={kn:.0f} → clamped. "
                f"Raise kn to apply more load (kn=300k → safe up to 7500N)."
            )
            _f_load_raw = _f_load_safe
        self._f_load = _f_load_raw
        self._f_load_safe_max = _f_load_safe
        self._substeps = substeps
        self._sim_dt = sim_dt
        self._n_bead_per_ring = n_bead_per_ring
        self._n_bead_rows = n_bead_rows
        self._n_bead = n_bead
        self._fz_tare = fz_tare
        self._m_tire = m_tire
        self._alpha_m_damp = alpha_m_damp
        # Rigid wheel mass for the contact-force check: the asset spindle's mass unless --m-rigid overrides.
        if args.m_rigid is not None:
            self._m_rigid = float(args.m_rigid)
        else:
            self._m_rigid = float(self.vehicle.wheel_mass if self.vehicle else tire_meta.spindle.mass)

        # ── Bead ring node indices: n_bead_rows rows pinned per side (TireAssetMeta) ──
        bead_np = bead_row_indices(n_bead_per_ring, n_bead_rows, n_ax_divs)
        assert len(bead_np) == n_bead, f"expected {n_bead} bead nodes, got {len(bead_np)}"

        self._bead_idx = wp.array(bead_np, dtype=wp.int32, device=device)
        # Tiled bead indices into the N*n_nodes particle array (Z-up, all envs).
        bead_idx_all_np = np.concatenate([bead_np + e * n_nodes for e in range(n_envs)])
        self._bead_idx_all = wp.array(bead_idx_all_np.astype(np.int32), device=device)
        self._bead_rest = wp.array(x0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_idx_np = bead_np
        self._bead_rest_np = x0_np[bead_np]
        self._bead_D0 = wp.array(d0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)

        # ── ANCF solver (Y-up, gravity along -Y) ─────────────────────────────
        ancf_builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        ancf_newton_model = ancf_builder.finalize(device=device)
        self.ancf_solver = SolverANCFShellRigid(
            model=ancf_newton_model,
            ancf_model=self.ancf_model,
            ground_z=0.0,
            kn=kn,
            kd=kd,
            mu=mu,
            nr_max_iter=nr_iters,
            pcg_max_iter=pcg_iters,
            n_envs=n_envs,
            thickness_gp=thickness_gp,
        )

        # Place each ANCF tire at its config position (ANCF X = lateral = Z-up Y). The MuJoCo
        # rigs all sit at Y=0; the per-tire lateral offsets in the coupling kernels supply the
        # hub offset so the coupling is exact for any position list.
        world_x = np.concatenate(
            [x0_np + np.array([tire_positions[e], drop_heights[e], 0.0], dtype=np.float32) for e in range(n_envs)],
            axis=0,
        )  # shape (N*n_nodes, 3)
        self.ancf_solver.node_x.assign(world_x)
        # _step_batched requires GLOBAL flat indices (env * n_nodes + local_idx).
        bead_global_np = np.concatenate([bead_np + e * n_nodes for e in range(n_envs)])
        self.ancf_solver.set_dirichlet_nodes(bead_global_np)
        self._build_pressures = pressures
        self._pressure_targets = list(pressures)
        self._pressure_currents = list(pressures)
        self._build_pressure = pressures[0]  # > 0 enables the CTIS panel / ramp
        self.ancf_solver.set_cavity(pressures, pressures)

        # ── MuJoCo rigid-body rig (Z-up) from the asset's spindle ────────────
        # Chrono ChTireTestRig topology: world -slide_x- chassis -slide_z- spindle.
        # slide_x carries a velocity actuator (lin_motor) that prescribes the
        # no-slip forward speed; slide_z is free so MuJoCo handles the vertical
        # load. Spin is kinematic (rim_phi / rim_omega GPU arrays), no hinge.
        # The spindle body (visual mesh, mass, inertia) is the /Tire/Spindle prim
        # baked into the tire USD by newton-tire-tool; its frame is the tire's
        # Y-up (axle X), converted to Z-up here with (x,y,z) -> (z,x,y).
        sp = tire_meta.spindle

        # Build a single-world template to extract per-world DOF strides.
        car_template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car_template)
        car_template.add_ground_plane()
        chassis = car_template.add_link(
            mass=1e-3, inertia=wp.mat33(1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6), label="chassis"
        )
        j_slide_x = car_template.add_joint_prismatic(
            -1,
            chassis,
            axis=newton.Axis.X,
            damping=_D_SLIDE_X,
            target_vel=0.0,
            target_kd=_KV_LIN_MOTOR,
            actuator_mode=newton.JointTargetMode.VELOCITY,
            label="slide_x",
        )
        if self.vehicle is not None:
            # The vehicle's wheel body, exactly as the car carries it (VehicleUSD.add_spindle_link).
            spindle = self.vehicle.add_spindle_link(
                car_template, sp, label="spindle", r_bead=r_inner, half_bead=0.5 * width
            )
        else:
            _P = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Y-up -> Z-up
            sp_pts_zu = (sp.points.astype(np.float64) @ _P.T).astype(np.float32)
            sp_com_zu = _P @ sp.com
            sp_inertia_zu = _P @ sp.inertia @ _P.T
            spindle_mesh = newton.Mesh(sp_pts_zu, sp.triangle_indices.reshape(-1), compute_inertia=False)
            vis_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0, has_shape_collision=False, has_particle_collision=False
            )
            spindle = car_template.add_link(
                mass=sp.mass,
                com=wp.vec3(*sp_com_zu.tolist()),
                inertia=wp.mat33(*sp_inertia_zu.reshape(-1).tolist()),
                label="spindle",
            )
            car_template.add_shape_mesh(
                spindle, mesh=spindle_mesh, cfg=vis_cfg, color=(0.35, 0.35, 0.75), label="spindle_vis"
            )
        j_slide_z = car_template.add_joint_prismatic(
            chassis, spindle, axis=newton.Axis.Z, damping=_D_SLIDE_Z, label="slide_z"
        )
        car_template.add_articulation([j_slide_x, j_slide_z])

        slide_z_q_dof_local = int(car_template.joint_q_start[j_slide_z])
        slide_x_qd_dof_local = int(car_template.joint_qd_start[j_slide_x])  # for lin_motor
        n_q_per_world = int(car_template.joint_coord_count)
        n_qd_per_world = int(car_template.joint_dof_count)

        # Build the multi-world MuJoCo model (N identical spindle rigs).
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        for _e in range(n_envs):
            car.add_world(car_template)  # no xform: fwd_position ignores body_q init

        self._slide_z_q_dof = slide_z_q_dof_local
        self._slide_x_qd_dof = slide_x_qd_dof_local  # lin_motor: no-slip rolling
        self._n_q_per_world = n_q_per_world
        self._n_qd_per_world = n_qd_per_world

        # ── RPM targets (ramped in step() into rim_omega) ─────────────────────
        default_rpm = float(args.rpm)
        self._target_rpm = [float(c.get("rpm", default_rpm)) for c in tire_cfgs]
        self._current_rpm = list(self._target_rpm)

        # ── Rendering model ───────────────────────────────────────────────────
        # Convert all N*n_nodes positions from ANCF Y-up to Z-up for the viewer.
        world_x_zu_all = []
        for _e in range(n_envs):
            wx_e = world_x[_e * n_nodes : (_e + 1) * n_nodes]
            wx_zu = np.stack([wx_e[:, 2], wx_e[:, 0], wx_e[:, 1]], axis=1).astype(np.float32)
            world_x_zu_all.append(wx_zu)
        world_x_zu = np.concatenate(world_x_zu_all, axis=0)  # (N*n_nodes, 3)

        # Particles go DIRECTLY on car (preserves the N-world structure; builder.add_world(car)
        # would flatten N worlds to 1). The ground plane comes from the rig template.
        car.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x_zu],
            vel=[(0.0, 0.0, 0.0)] * (n_nodes * n_envs),
            mass=[0.0] * (n_nodes * n_envs),
            radius=[0.001] * (n_nodes * n_envs),
        )
        all_tris = quad_triangles(self.ancf_model.elem_nodes.numpy(), n_nodes, n_envs)
        car.add_triangles(
            i=all_tris[:, 0].tolist(),
            j=all_tris[:, 1].tolist(),
            k=all_tris[:, 2].tolist(),
        )
        self.model = car.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()

        # joint_q[slide_z] sets the hub height. Sync model.joint_q and both states so
        # step_kinematics reads the correct spindle position from the first substep.
        jq_init = self.model.joint_q.numpy().copy()
        for _e in range(n_envs):
            jq_init[slide_z_q_dof_local + _e * n_q_per_world] = drop_heights[_e]
        self.model.joint_q.assign(jq_init)
        self.state_0.joint_q.assign(self.model.joint_q)
        self.state_rigid.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        wp.launch(
            _ancf_yup_to_zu,
            dim=n_envs * n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device=device,
        )

        # ── MuJoCo solver ─────────────────────────────────────────────────────
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            iterations=2,
            ls_iterations=2,
            update_data_interval=0,
        )
        # Disable capture_while in mujoco_warp solver: use Python for-loop (no GPU syncs).
        self.solver.mjw_model.opt.graph_conditional = False

        # ── Spindle body index mapping ────────────────────────────────────────
        self._spindle_newton_idx = find_body(self.model, "spindle")
        btow = self.solver.mjc_body_to_newton.numpy()[0]
        m = np.where(btow == self._spindle_newton_idx)[0]
        assert len(m), (
            f"spindle Newton body {self._spindle_newton_idx} not in MuJoCo body map. "
            f"body_label={list(self.model.body_label)}"
        )
        self._spindle_mj = int(m[0])

        # ── Register per-wheel coupling in SolverANCFShellRigid ──────────────
        # N-parallel-world layout: env e lives in MuJoCo world e (world_idx=e); the spindle
        # body index is the same in every world. lateral_offset = the tire's config position so
        # the moment arm is taken from the ANCF hub centre (MuJoCo spindles all sit at Y=0).
        for _e in range(n_envs):
            bead_global_e = (bead_np + _e * n_nodes).astype(np.int32)
            self.ancf_solver.setup_wheel(
                tire_idx=_e,
                spindle_mj=self._spindle_mj,
                bead_idx_np=bead_global_e,
                tare_fz=fz_tare,
                world_idx=_e,  # parallel-world: env == world
                lateral_offset=float(tire_positions[_e]),
                device=device,
            )

        # ── GPU scalars / per-env arrays ─────────────────────────────────────
        self._rim_phi = wp.zeros(n_envs, dtype=float, device=device)  # kinematic, self-tracked
        self._rim_omega_wp = wp.zeros(n_envs, dtype=float, device=device)  # kinematic spin (not from MuJoCo cvel)
        # Host mirror of _rim_omega_wp; kept in sync by step() so the RPM ramp
        # never reads back from the device.  Starts zeroed to match the array above.
        self._rim_omega_np = np.zeros(n_envs, dtype=np.float32)
        self._n_nodes = n_nodes
        self._n_envs = n_envs
        self._tire_positions = tire_positions  # per-env ANCF X offsets from config
        self._lateral_offsets_wp = wp.array(np.asarray(tire_positions, dtype=np.float32), dtype=float, device=device)
        # Per-world body stride from the template (model.body_count includes extra bodies).
        self._n_bodies_per_world = car_template.body_count
        spindle_src = f"vehicle {self.vehicle.name}" if self.vehicle is not None else "tire asset /Tire/Spindle"
        print(
            f"[STM] tire={os.path.basename(tire_asset_path)} R_outer={r_outer:.4f} r_roll={r_roll:.4f}"
            f"  spindle={spindle_src} (newton={self._spindle_newton_idx}, mj={self._spindle_mj})"
            f"  n_envs={n_envs}  m_tire={m_tire:.3f} kg  m_rigid={self._m_rigid:.2f} kg"
            f"  alpha_m_damp={alpha_m_damp:.1f} s^-1  substeps={substeps} nr={nr_iters} pcg={pcg_iters}"
        )

        # ── Bead ring visualization buffers (N envs) ─────────────────────────
        seg_s_all, seg_e_all = ring_segments(n_bead_per_ring, n_bead_rows, n_bead, n_envs)
        n_segs_total = len(seg_s_all)
        self._bead_pos_zu = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)
        self._spoke_start_zu = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=device)
        self._ring_line_s = wp.zeros(n_segs_total, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(n_segs_total, dtype=wp.vec3, device=device)
        self._n_ring_segs = n_segs_total

        # ── Contact spike visualization buffers (one segment per FEM node, GPU-only) ──
        self._contact_line_s = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=device)
        self._contact_line_e = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=device)
        self._contact_vis_scale = 100.0  # pen 4 mm → 400 mm spike at kn=10k; tune per kn

        # ── Graph capture ─────────────────────────────────────────────────────
        # Must be set before capture_graph(): _step_batched (used by CUDA graph)
        # only zeros NR corrector at Dirichlet nodes when this flag is True.
        # Without it, bead nodes drift during the NR loop → growing spurious forces.
        self.ancf_solver._fix_dirichlet_in_batched = True
        self.ancf_solver.capture_graph(self._sim_dt)

        # Restore ANCF state consumed by the capture_graph warmup step.
        # Zero f_int/f_int0: set_cavity(p,p) gives p_gauge=0 at V_ref, so the
        # reference config has no internal force.
        self.ancf_solver.node_x.assign(world_x)
        self.ancf_solver.node_xd.zero_()
        self.ancf_solver.node_xdd.zero_()
        self.ancf_solver.node_D.assign(d0_np_tiled)
        self.ancf_solver.node_Dd.zero_()
        self.ancf_solver.node_Ddd.zero_()
        self.ancf_solver.global_f_int.zero_()
        self.ancf_solver.global_f_int0.zero_()
        self.ancf_solver.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
        self._prescribe_beads()

        # Seed bead ring and spoke visualization buffers (all N envs).
        wp.launch(
            _gather_zu,
            dim=self._n_envs * self._n_bead,
            inputs=[self.state_0.particle_q, self._bead_idx_all, self._bead_pos_zu],
            device=device,
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=device,
        )
        wp.launch(
            _fill_spindle_positions_env,
            dim=self._n_envs * self._n_bead,
            inputs=[
                self.state_0.body_q,
                self._spindle_newton_idx,
                self._n_bodies_per_world,
                self._spoke_start_zu,
                self._n_bead,
                self._lateral_offsets_wp,
            ],
            device=device,
        )

        # MuJoCo kinematics / dynamics as standalone graphs (the fallback substep loop uses
        # them), then the full substep as one CUDA graph. Uses ancf.step() (unrolled NR+PCG)
        # because a graph launch is not capturable inside another capture.
        self._kin_graph = None
        self._dyn_graph = None
        self._substep_graph = None
        self._try_capture_mujoco_graphs()
        self._try_capture_substep_graph()

        if viewer is not None:
            viewer.set_model(self.model)
            viewer.set_camera(pos=wp.vec3(-6.0, -9.0, 4.5), pitch=-22.0, yaw=48.0)
            # Per-world display offsets from the tire positions so each spindle appears
            # centred on its tire (viewer.world_offsets: wp.array[wp.vec3] of model.world_count).
            if self._n_envs > 1:
                offsets = np.zeros((self._n_envs, 3), dtype=np.float32)
                for _e in range(self._n_envs):
                    offsets[_e, 1] = self._tire_positions[_e]  # ANCF X → Z-up Y
                viewer.world_offsets = wp.array(offsets, dtype=wp.vec3, device=device)

    # ── Substep graph capture ─────────────────────────────────────────────────

    def _try_capture_substep_graph(self) -> None:
        """Capture one complete substep as a single CUDA graph.

        Saves and restores ALL simulation state — ANCF, state_0, and state_rigid —
        around the capture warmup.  ancf.step() is used (unrolled NR+PCG) because
        ancf.graph_step() cannot be nested inside wp.capture_begin.
        """
        dev = "cuda:0"
        ancf = self.ancf_solver
        dt = self._sim_dt

        wp.synchronize_device(dev)
        s = {
            "node_x": ancf.node_x.numpy().copy(),
            "node_xd": ancf.node_xd.numpy().copy(),
            "node_xdd": ancf.node_xdd.numpy().copy(),
            "node_D": ancf.node_D.numpy().copy(),
            "node_Dd": ancf.node_Dd.numpy().copy(),
            "node_Ddd": ancf.node_Ddd.numpy().copy(),
            "f_int": ancf.global_f_int.numpy().copy(),
            "f_int0": ancf.global_f_int0.numpy().copy(),
            "eas": self.ancf_model.elem_eas_alpha.numpy().copy(),
            "f_ext": ancf.node_f_ext_persistent.numpy().copy(),
            "s0_bq": self.state_0.body_q.numpy().copy(),
            "s0_bqd": self.state_0.body_qd.numpy().copy(),
            "s0_jq": self.state_0.joint_q.numpy().copy(),
            "s0_jqd": self.state_0.joint_qd.numpy().copy(),
            "sr_bq": self.state_rigid.body_q.numpy().copy(),
            "sr_bqd": self.state_rigid.body_qd.numpy().copy(),
            "sr_jq": self.state_rigid.joint_q.numpy().copy(),
            "sr_jqd": self.state_rigid.joint_qd.numpy().copy(),
        }

        def _restore_ancf():
            ancf.node_x.assign(s["node_x"])
            ancf.node_xd.assign(s["node_xd"])
            ancf.node_xdd.assign(s["node_xdd"])
            ancf.node_D.assign(s["node_D"])
            ancf.node_Dd.assign(s["node_Dd"])
            ancf.node_Ddd.assign(s["node_Ddd"])
            ancf.global_f_int.assign(s["f_int"])
            ancf.global_f_int0.assign(s["f_int0"])
            self.ancf_model.elem_eas_alpha.assign(s["eas"])
            ancf.node_f_ext_persistent.assign(s["f_ext"])

        def _restore_rigid():
            self.state_0.body_q.assign(s["s0_bq"])
            self.state_0.body_qd.assign(s["s0_bqd"])
            self.state_0.joint_q.assign(s["s0_jq"])
            self.state_0.joint_qd.assign(s["s0_jqd"])
            self.state_rigid.body_q.assign(s["sr_bq"])
            self.state_rigid.body_qd.assign(s["sr_bqd"])
            self.state_rigid.joint_q.assign(s["sr_jq"])
            self.state_rigid.joint_qd.assign(s["sr_jqd"])

        # Pre-warmup: wp.capture_end() creates the graph but NOT the exec (lazy). Launching
        # once here instantiates it OUTSIDE any capture context (error 900 otherwise).
        ancf.graph_step()
        wp.synchronize_device(dev)
        _restore_ancf()

        # Capture with direct calls: cudaGraphLaunch is not allowed during stream capture
        # (error 900), so ancf.step() replaces ancf.graph_step() and step_kinematics /
        # step_dynamics replace their graphs. All kernels are already compiled.
        def _kin():
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)

        def _ancf_step():
            ancf.step(None, None, None, None, dt)

        def _dyn():
            self.solver.step_dynamics(self.state_rigid)

        try:
            wp.capture_begin(device=dev)
            self._one_substep(_kin, _ancf_step, _dyn)  # capture ONE substep only
            self._substep_graph = wp.capture_end(device=dev)
            print(f"[SUBSTEP GRAPH] Captured 1 substep — {self._substeps} launches/frame")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._substep_graph = None
            print(f"[SUBSTEP GRAPH] Capture failed ({e!r}) — falling back to the per-substep loop")

        wp.synchronize_device(dev)
        _restore_ancf()
        _restore_rigid()

    def _try_capture_mujoco_graphs(self) -> None:
        """Capture step_kinematics and step_dynamics as standalone CUDA graphs (fallback loop)."""
        dev = "cuda:0"
        dt = self._sim_dt
        wp.synchronize_device(dev)
        try:
            wp.capture_begin(device=dev)
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            self._kin_graph = wp.capture_end(device=dev)
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._kin_graph = None
            print(f"[MJ GRAPHS] Kinematics capture failed: {e!r}")

        wp.synchronize_device(dev)
        try:
            wp.capture_begin(device=dev)
            self.solver.step_dynamics(self.state_rigid)
            self._dyn_graph = wp.capture_end(device=dev)
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._dyn_graph = None
            print(f"[MJ GRAPHS] Dynamics capture failed: {e!r}")

        # Launch each graph once so its exec is instantiated outside any capture context.
        wp.synchronize_device(dev)
        if self._kin_graph is not None:
            wp.capture_launch(self._kin_graph)
        if self._dyn_graph is not None:
            wp.capture_launch(self._dyn_graph)
        wp.synchronize_device(dev)

    # ── Coupling helpers ───────────────────────────────────────────────────────

    def _prescribe_beads(self, vel_predict_dt: float = 0.0, inv_dt: float = 0.0) -> None:
        wp.launch(
            _prescribe_beads_gpu,
            dim=self._n_envs * self._n_bead,
            inputs=[
                self.solver.xpos,
                self.solver.cvel,
                self._spindle_mj,
                self._rim_phi,
                self._rim_omega_wp,
                self._bead_idx,
                self._bead_rest,
                self._bead_D0,
                self.ancf_solver.node_x,
                self.ancf_solver.node_xd,
                self.ancf_solver.node_xdd,
                self.ancf_solver.node_D,
                self.ancf_solver.node_Dd,
                self.ancf_solver.node_Ddd,
                self._n_bead,
                self._n_nodes,
                self._lateral_offsets_wp,
                self._r_outer,
                vel_predict_dt,
                inv_dt,
            ],
            device="cuda:0",
        )

    def _accumulate_wrenches(self) -> None:
        self.solver.xfrc_applied.zero_()
        self.ancf_solver.accumulate_wheel_wrenches(
            xfrc_applied=self.solver.xfrc_applied,
            xpos=self.solver.xpos,
            device="cuda:0",
        )
        if self._f_load > 0.0:
            wp.launch(
                _apply_vertical_load,
                dim=self._n_envs,
                inputs=[self.solver.xfrc_applied, self._spindle_mj, self._f_load],
                device="cuda:0",
            )
        # slide_x is kinematic — zero forward force so step_dynamics cannot
        # perturb joint_qd[slide_x].  slide_z (vertical) remains two-way.
        wp.launch(
            _zero_xfrc_forward, dim=self._n_envs, inputs=[self.solver.xfrc_applied, self._spindle_mj], device="cuda:0"
        )

    def _one_substep(self, kin_fn, ancf_step_fn, dyn_fn) -> None:
        """One coupled substep; ``kin_fn`` / ``ancf_step_fn`` / ``dyn_fn`` are direct calls
        under capture and graph launches in the fallback loop (same kernels either way)."""
        dev = "cuda:0"
        dt = self._sim_dt
        ancf = self.ancf_solver
        # 1. Advance kinematic rim angle: phi += omega × dt
        wp.launch(
            _advance_rim_phi_batched, dim=self._n_envs, inputs=[self._rim_phi, self._rim_omega_wp, dt], device=dev
        )
        # 2a. Kinematic velocity injection: joint_qd[slide_x] = omega × R so
        #     step_kinematics advances slide_x by exactly omega × R × dt.
        wp.launch(
            _prescribe_slide_x_vel,
            dim=self._n_envs,
            inputs=[
                self.state_0.joint_qd,
                self._rim_omega_wp,
                self._slide_x_qd_dof,
                self._n_qd_per_world,
                self._r_outer,
            ],
            device=dev,
        )
        # 2b. Set lin_motor target (backup soft-constraint, near-zero force in steady state)
        wp.launch(
            _set_rolling_ctrl,
            dim=self._n_envs,
            inputs=[
                self._rim_omega_wp,
                self.control.joint_target_vel,
                self._slide_x_qd_dof,
                self._n_qd_per_world,
                self._r_outer,
            ],
            device=dev,
        )
        # 3. MuJoCo kinematics (slide_x, slide_z) → updates xpos
        kin_fn()
        # Predictor: extrapolate spindle pos by vel × dt → O(dt²) interface (vs O(dt) without).
        self._prescribe_beads(vel_predict_dt=dt, inv_dt=1.0 / dt)
        if self._alpha_m_damp > 0.0:
            wp.launch(
                _update_mass_damp,
                dim=self._n_envs * self._n_nodes,
                inputs=[ancf.node_xd, ancf.lumped_mass_tiled, self._alpha_m_damp, ancf.node_f_ext_persistent],
                device=dev,
            )
        ancf_step_fn()
        # Corrector: pin beads to current spindle pos (no extrapolation) after FEM solve.
        self._prescribe_beads(vel_predict_dt=0.0)
        self._accumulate_wrenches()
        dyn_fn()
        # Re-pin slide_x velocity after step_dynamics so the copy carries the
        # kinematic value (not whatever residual lin_motor/xfrc computed).
        wp.launch(
            _prescribe_slide_x_vel,
            dim=self._n_envs,
            inputs=[
                self.state_rigid.joint_qd,
                self._rim_omega_wp,
                self._slide_x_qd_dof,
                self._n_qd_per_world,
                self._r_outer,
            ],
            device=dev,
        )
        wp.copy(self.state_0.body_q, self.state_rigid.body_q)
        wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
        wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
        wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self) -> None:
        # One small graph per substep (~1 NR block): the GPU executes while the CPU
        # launches the next one, so the frame is GPU-bound rather than launch-bound.
        if self._substep_graph is not None:
            for _sub in range(self._substeps):
                wp.capture_launch(self._substep_graph)
            wp.synchronize_device()
            return

        # Fallback: the same substep from the pre-captured kinematics / dynamics / ANCF graphs.
        def _kin():
            if self._kin_graph is not None:
                wp.capture_launch(self._kin_graph)
            else:
                self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)

        def _dyn():
            if self._dyn_graph is not None:
                wp.capture_launch(self._dyn_graph)
            else:
                self.solver.step_dynamics(self.state_rigid)

        for _sub in range(self._substeps):
            self._one_substep(_kin, self.ancf_solver.graph_step, _dyn)

    def step(self) -> None:
        # Ramp CTIS pressure per-env toward GUI targets (2000 Pa/frame each)
        if self._build_pressure > 0.0:
            for e in range(self._n_envs):
                d = self._pressure_targets[e] - self._pressure_currents[e]
                step = min(abs(d), 2000.0) * (1.0 if d >= 0.0 else -1.0)
                self._pressure_currents[e] += step
            self.ancf_solver.set_cavity(
                self._pressure_currents,
                self._build_pressures,
            )

        # Ramp rim_omega_wp toward GUI target RPM.  The host is the sole writer of
        # rim_omega_wp (kernels only read it), so ramp the host mirror and upload
        # only when it actually changes — steady RPM costs no PCIe traffic.
        rim_omega_np = self._rim_omega_np
        rpm_changed = False
        for e in range(self._n_envs):
            d = self._target_rpm[e] - self._current_rpm[e]
            if abs(d) <= _RPM_RATE:
                self._current_rpm[e] = self._target_rpm[e]
            else:
                self._current_rpm[e] += _RPM_RATE if d > 0.0 else -_RPM_RATE
            omega = self._current_rpm[e] * (2.0 * math.pi / 60.0)
            if omega != rim_omega_np[e]:
                rim_omega_np[e] = omega
                rpm_changed = True
        if rpm_changed:
            self._rim_omega_wp.assign(rim_omega_np)

        _t0 = time.perf_counter()
        self.simulate()
        self._t_step += time.perf_counter() - _t0
        self._frame += 1
        self._t += _FRAME_DT

        # Update render particles (all N*n_nodes) and bead ring / spoke buffers (all N envs).
        wp.launch(
            _ancf_yup_to_zu,
            dim=self._n_envs * self._n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device="cuda:0",
        )
        wp.launch(
            _gather_zu,
            dim=self._n_envs * self._n_bead,
            inputs=[self.state_0.particle_q, self._bead_idx_all, self._bead_pos_zu],
            device="cuda:0",
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device="cuda:0",
        )
        wp.launch(
            _fill_spindle_positions_env,
            dim=self._n_envs * self._n_bead,
            inputs=[
                self.state_0.body_q,
                self._spindle_newton_idx,
                self._n_bodies_per_world,
                self._spoke_start_zu,
                self._n_bead,
                self._lateral_offsets_wp,
            ],
            device="cuda:0",
        )

        if self._frame % self._diag_period == 0:
            _now = time.perf_counter()
            _fps = self._diag_period / max(_now - self._t_wall, 1e-9)
            _ms = 1e3 / max(_fps, 1e-3)
            self._print_diag(_fps, _ms)
            self._t_wall = _now
            self._t_step = self._t_render = 0.0

    # ── Diagnostics (host readback only every --diag-period frames) ───────────

    def _print_diag(self, fps: float = 0.0, ms: float = 0.0) -> None:
        with wp.ScopedTimer("diag", use_nvtx=False, color="red"):
            x_all = self.ancf_solver.node_x.numpy()  # (N*n_nodes, 3)
            xd_all = self.ancf_solver.node_xd.numpy()  # (N*n_nodes, 3)
            # Per-tire staging wrench the solver fed to xfrc_applied (ANCF Y-up; [4] = vertical).
            stg_all = [self.ancf_solver._xfrc_stg_per_tire[e].numpy()[0] for e in range(self._n_envs)]
            phi_all = self._rim_phi.numpy()  # (N,)
            fi_all = self.ancf_solver.global_f_int.numpy()  # (N*n_nodes*6,)
            bq_np = self.state_0.body_q.numpy()
            xpos_all = self.solver.xpos.numpy()  # (N, nbody)
            cvel_all = self.solver.cvel.numpy()  # (N, nbody) spatial_vector

        N = self._n_envs
        nn = self._n_nodes
        nf = nn * 6
        rest_np = self._bead_rest_np

        print(f"\n[{self._frame:4d}] t={self._t:.2f}s  fps={fps:.1f}")
        for e in range(N):
            x_e = x_all[e * nn : (e + 1) * nn]
            xd_e = xd_all[e * nn : (e + 1) * nn].reshape(-1, 3)
            fi_e = fi_all[e * nf : (e + 1) * nf]
            stg_e = stg_all[e]
            phi_e = float(phi_all[e])
            sp_tf = bq_np[self._spindle_newton_idx + e * self._n_bodies_per_world]
            sp_z = float(sp_tf[2])

            any_nan = bool(np.any(np.isnan(x_e)))
            fi_finite = bool(np.all(np.isfinite(fi_e)))
            fi_max = float(np.max(np.abs(fi_e))) if fi_finite else float("nan")
            v_max = float(np.max(np.linalg.norm(xd_e, axis=1)))

            bead_x = x_e[self._bead_idx_np]
            cp, sp_ = math.cos(phi_e), math.sin(phi_e)
            hub = np.array([float(sp_tf[1]) + self._tire_positions[e], float(sp_tf[2]), float(sp_tf[0])])
            expected = np.stack(
                [
                    hub[0] + rest_np[:, 0],
                    hub[1] + rest_np[:, 1] * cp - rest_np[:, 2] * sp_,
                    hub[2] + rest_np[:, 1] * sp_ + rest_np[:, 2] * cp,
                ],
                axis=1,
            )
            drift_mm = float(np.max(np.linalg.norm(bead_x - expected, axis=1))) * 1e3
            fz = float(stg_e[4]) + self._fz_tare

            # What the coupling kernel actually uses as hub position
            xpos_e = xpos_all[e, self._spindle_mj]  # MuJoCo Z-up (x_fwd,y_lat,z_up)
            hub_x_mj = float(xpos_e[1]) + self._tire_positions[e]  # → ANCF X
            hub_y_mj = float(xpos_e[2])  # → ANCF Y (height)

            # ── Rolling slip test ─────────────────────────────────────────────
            # No-slip rolling: v_hub_fwd == omega_axle * R_outer.
            # ang_y (MuJoCo cvel[1]) = axle spin; vx (cvel[3]) = hub forward velocity.
            # slip_vel > 0: hub moves faster than rotation says (braking / skid)
            # slip_vel < 0: wheel spins faster than hub moves (wheel-spin / friction loss)
            # pos_slip: cumulative — hub_x_fwd vs phi*R_outer since t=0.
            omega_e = float(cvel_all[e, self._spindle_mj][1])  # axle ang_y [rad/s]
            v_fwd_e = float(cvel_all[e, self._spindle_mj][3])  # hub vx [m/s]
            v_roll_e = omega_e * self._r_outer  # expected v_fwd [m/s]
            slip_vel_e = v_fwd_e - v_roll_e  # instantaneous slip [m/s]
            fwd_pos_e = float(xpos_e[0])  # hub x_fwd in MuJoCo [m]
            roll_pos_e = phi_e * self._r_outer  # expected fwd dist [m]
            pos_slip_e = fwd_pos_e - roll_pos_e  # cumulative slip [m]

            print(
                f"  env{e}: {'NaN!' if any_nan else 'ok  '}"
                f"  sp_Z={sp_z:+.3f}m  xpos_hub=({hub_x_mj:.3f},{hub_y_mj:.3f})"
                f"  bead_drift={drift_mm:.2f}mm"
                f"  v_max={v_max:.2f}m/s"
                f"  fi_max={fi_max:.2e}"
                f"  Fz={fz:+.0f}N"
                f"  | slip: v_roll={v_roll_e:+.3f}m/s v_hub={v_fwd_e:+.3f}m/s"
                f" dv={slip_vel_e:+.4f}m/s pos={pos_slip_e:+.4f}m"
            )

        # Cache env-0 for GUI
        x_np = x_all[:nn]
        xd_np = xd_all[:nn].reshape(-1, 3)
        sp_tf0 = bq_np[self._spindle_newton_idx]
        any_nan = bool(np.any(np.isnan(x_np)))
        bead_x = x_np[self._bead_idx_np]
        phi_val = float(phi_all[0])
        cp, sp_ = math.cos(phi_val), math.sin(phi_val)
        hub_ancf = np.array([float(sp_tf0[1]), float(sp_tf0[2]), float(sp_tf0[0])])
        expected = np.stack(
            [
                hub_ancf[0] + rest_np[:, 0],
                hub_ancf[1] + rest_np[:, 1] * cp - rest_np[:, 2] * sp_,
                hub_ancf[2] + rest_np[:, 1] * sp_ + rest_np[:, 2] * cp,
            ],
            axis=1,
        )
        bead_drift_mm = float(np.max(np.linalg.norm(bead_x - expected, axis=1))) * 1e3
        node_v_max = float(np.max(np.linalg.norm(xd_np, axis=1)))
        fi_np = fi_all[:nf]
        fi_finite = bool(np.all(np.isfinite(fi_np)))
        fi_max = float(np.max(np.abs(fi_np))) if fi_finite else float("nan")
        fz_contact = float(stg_all[0][4]) + self._fz_tare
        fz_exp = self._m_rigid * _GRAVITY
        sp_z = float(sp_tf0[2])
        xpos_zu = xpos_all[0, self._spindle_mj]

        # Env-0 slip for GUI
        omega_0 = float(cvel_all[0, self._spindle_mj][1])
        v_fwd_0 = float(cvel_all[0, self._spindle_mj][3])
        v_roll_0 = omega_0 * self._r_outer
        slip_vel_0 = v_fwd_0 - v_roll_0
        phi_0 = float(phi_all[0])
        pos_slip_0 = float(xpos_zu[0]) - phi_0 * self._r_outer

        # Cache for GUI readouts (updated once per diag_period, not every frame)
        self._gui_sp_z = sp_z
        self._gui_fz = fz_contact
        self._gui_bead_drift = bead_drift_mm
        self._gui_v_max = node_v_max
        self._gui_fps = fps
        self._gui_slip_vel = slip_vel_0
        self._gui_pos_slip = pos_slip_0
        self._gui_v_roll = v_roll_0
        self._gui_v_fwd = v_fwd_0

        _step_ms = 1e3 * self._t_step / self._diag_period
        _render_ms = 1e3 * self._t_render / self._diag_period
        print(
            f"[{self._frame:4d}] {'NaN!' if any_nan else 'ok  '}"
            f"  t={self._t:.2f}s"
            f"  sp_Z={sp_z:+.4f}m"
            f"  F_z={fz_contact:+.0f}N (wt~{fz_exp:.0f}N)"
            f"  bead_drift={bead_drift_mm:.3f}mm"
            f"  v_max={node_v_max:.2f}m/s"
            f"  fi_max={fi_max:.2e}"
            f"  fps={fps:.1f} ({ms:.1f}ms/frame"
            f"  step={_step_ms:.1f}ms  render={_render_ms:.1f}ms)"
        )

    # ── GUI ────────────────────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        if self._build_pressure > 0.0:
            ui.separator()
            ui.text("CTIS pressure (per tire)")
            p_max = self._build_pressure * 2.0
            for e in range(self._n_envs):
                changed, val = ui.slider_float(f"Tire {e} [Pa]##pres{e}", self._pressure_targets[e], 0.0, p_max)
                if changed:
                    self._pressure_targets[e] = float(val)
                cur = self._pressure_currents[e]
                ui.text(f"  {cur:7.0f} Pa  ({cur / 6894.76:.1f} psi)  gauge {cur - self._build_pressures[e]:+.0f} Pa")

        ui.separator()
        ui.text("Spindle RPM")
        for e in range(self._n_envs):
            changed, val = ui.slider_float(f"Env {e}##rpm{e}", self._target_rpm[e], -300.0, 300.0)
            if changed:
                self._target_rpm[e] = float(val)
            omega = self._target_rpm[e] * (2.0 * math.pi / 60.0)
            ui.text(f"  env{e}  {self._target_rpm[e]:+7.1f} RPM  ({omega:+.2f} rad/s)")
        # rim_omega_wp is ramped toward the target in step(); no immediate write here.

        ui.separator()
        ui.text("Live")
        ui.text(f"  fps          {self._gui_fps:6.1f}")
        ui.text(f"  spindle Z    {self._gui_sp_z:+.4f} m")
        ui.text(f"  Fz contact   {self._gui_fz:+.0f} N  (wt~{self._m_rigid * _GRAVITY:.0f} N)")
        ui.text(f"  bead drift   {self._gui_bead_drift:.3f} mm")
        ui.text(f"  v_max        {self._gui_v_max:.2f} m/s")
        ui.separator()
        ui.text("Vertical load (vehicle weight)")
        changed, val = ui.slider_float("F_load [N]##fload", self._f_load, 0.0, self._f_load_safe_max)
        if changed:
            self._f_load = float(val)
        F_total = self._f_load + self._m_rigid * _GRAVITY
        ui.text(f"  F_normal ~{F_total:.0f} N")
        ui.text(f"  safe max {self._f_load_safe_max:.0f} N  (raise kn=300k → 7500 N)")

        ui.separator()
        ui.text("Contact spikes")
        changed, val = ui.slider_float("pen scale##cvis", self._contact_vis_scale, 1.0, 500.0)
        if changed:
            self._contact_vis_scale = float(val)
        ui.text("  cyan spikes = in-contact nodes, height = pen × scale")

        ui.separator()
        ui.text("Rolling slip  (env 0)")
        ui.text(f"  v_roll       {self._gui_v_roll:+.4f} m/s  (omega * R)")
        ui.text(f"  v_hub_fwd    {self._gui_v_fwd:+.4f} m/s  (spindle vx)")
        ui.text(f"  slip vel     {self._gui_slip_vel:+.4f} m/s  (v_hub - v_roll)")
        ui.text(f"  slip pos     {self._gui_pos_slip:+.4f} m   (hub_x - phi*R)")

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        _t0 = time.perf_counter()
        self.viewer.begin_frame(self._t)

        # Spindle spin is kinematic (rim_phi); put it into body_q so the mesh turns with the
        # tire. Per-world lateral offsets are the viewer's world_offsets.
        wp.launch(
            _spin_spindle_body_q,
            dim=self._n_envs,
            inputs=[self.state_0.body_q, self._rim_phi, self._spindle_newton_idx, self._n_bodies_per_world],
            device=self.state_0.body_q.device,
        )
        self.viewer.log_state(self.state_0)
        # Orange rings: the bead rings as closed polygons.
        self.viewer.log_lines(
            "bead_rings",
            self._ring_line_s,
            self._ring_line_e,
            colors=(1.0, 0.45, 0.0),
        )
        # Yellow spokes: spindle centre → each bead node.
        self.viewer.log_lines(
            "bead_spokes",
            self._spoke_start_zu,
            self._bead_pos_zu,
            colors=(1.0, 0.90, 0.1),
        )
        # Cyan spikes: one per FEM node, height ∝ contact penetration (GPU-only).
        wp.launch(
            _gather_contact_spikes,
            dim=self._n_envs * self._n_nodes,
            inputs=[self.ancf_solver.node_x, 0.0, self._contact_vis_scale, self._contact_line_s, self._contact_line_e],
            device="cuda:0",
        )
        self.viewer.log_lines(
            "contact_spikes",
            self._contact_line_s,
            self._contact_line_e,
            colors=(0.0, 1.0, 1.0),
        )
        self.viewer.end_frame()
        self._t_render += time.perf_counter() - _t0

    # ── Tests ──────────────────────────────────────────────────────────────────

    def test_post_step(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()[: self._n_nodes]  # check env-0
        if np.any(np.isnan(x_np)):
            raise AssertionError(f"NaN in node_x at frame {self._frame}, t={self._t:.3f}s")

    def test_final(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()[: self._n_nodes]  # env-0
        xd_np = self.ancf_solver.node_xd.numpy()[: self._n_nodes]  # env-0
        assert not np.any(np.isnan(x_np)), "NaN in node_x at test_final"
        assert not np.any(np.isnan(xd_np)), "NaN in node_xd at test_final"
        assert not np.any(np.isinf(x_np)), "Inf in node_x at test_final"

        bq_np = self.state_0.body_q.numpy()
        sp_tf = bq_np[self._spindle_newton_idx]
        hub_ancf = np.array([float(sp_tf[1]), float(sp_tf[2]), float(sp_tf[0])])
        phi_val = float(self._rim_phi.numpy()[0])
        bead_x = x_np[self._bead_idx_np]
        rest_np = self._bead_rest_np
        cp, sp = math.cos(phi_val), math.sin(phi_val)
        expected = np.stack(
            [
                hub_ancf[0] + rest_np[:, 0],
                hub_ancf[1] + rest_np[:, 1] * cp - rest_np[:, 2] * sp,
                hub_ancf[2] + rest_np[:, 1] * sp + rest_np[:, 2] * cp,
            ],
            axis=1,
        )
        max_drift_m = float(np.max(np.linalg.norm(bead_x - expected, axis=1)))
        assert max_drift_m < 1e-3, f"FAIL: bead drift {max_drift_m * 1e3:.2f} mm > 1 mm"

        # Env-0 vertical spindle load as fed to MuJoCo: solver staging [4] (ANCF Y) + tare.
        stg_np = self.ancf_solver._xfrc_stg_per_tire[0].numpy()[0]
        fz = abs(float(stg_np[4]) + self._fz_tare)
        fz_exp = self._m_rigid * _GRAVITY
        rel_err = abs(fz - fz_exp) / max(fz_exp, 1.0)
        assert rel_err < 0.50, f"FAIL: F_z={fz:.1f} N, expected~{fz_exp:.1f} N (err={rel_err * 100:.1f}% > 50%)"

        fi_np = self.ancf_solver.global_f_int.numpy()
        assert np.all(np.isfinite(fi_np)), "Non-finite values in global_f_int"

        print(
            f"[PASS] frame={self._frame}  t={self._t:.1f}s\n"
            f"       bead drift  = {max_drift_m * 1e3:.3f} mm  (limit 1 mm)\n"
            f"       F_z contact = {fz:.1f} N  (rigid_wt~{fz_exp:.1f} N,"
            f" err={rel_err * 100:.1f}%)\n"
            f"       No NaN/Inf in node_x, node_xd, global_f_int. OK"
        )

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--n-envs",
            type=int,
            default=None,
            help="Number of parallel environments (tires + spindles). Default: len(--shell-tires), else 1.",
        )
        parser.add_argument(
            "--m-rigid",
            type=float,
            default=None,
            help="Total rigid body mass [kg] used for contact force validation. "
            "Default: None → the spindle mass baked into the tire asset (/Tire/Spindle).",
        )
        parser.add_argument(
            "--vehicle-asset",
            type=str,
            default=None,
            help="Vehicle USD (filename under assets/ or absolute path): the rig uses its wheel body as the "
            "spindle and its default tire. run-examples.sh prompts for it.",
        )
        parser.add_argument(
            "--tire-asset",
            type=str,
            default=None,
            help="Baked ANCF tire USD (filename under assets/ or absolute path). Default: the vehicle's tire.",
        )
        parser.add_argument(
            "--drop-clearance",
            type=float,
            default=_DROP_CLEARANCE,
            help="Initial hub clearance above ground [m]. Default 0 (quasi-static start).",
        )
        parser.add_argument(
            "--diag-period",
            type=int,
            default=5,
            help="Print diagnostics every N frames.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=_SIM_SUBSTEPS,
            help="Substeps per frame.  dt = 1/60/substeps.",
        )
        parser.add_argument(
            "--kn",
            type=float,
            default=None,
            help="Ground contact normal stiffness [N/m] (default: the tire asset's recommendation).  "
            "HHT stable when kn < m_node/((0.5-beta)*dt^2).",
        )
        parser.add_argument(
            "--kd",
            type=float,
            default=None,
            help="Ground contact normal damping [N·s/m] (default: kn x 20/10000).  kd < 2*m_node/dt.",
        )
        parser.add_argument(
            "--mu",
            type=float,
            default=_MU,
            help="Ground friction coefficient.",
        )
        parser.add_argument(
            "--nr-iters",
            type=int,
            default=_NR_ITERS,
            help="Newton-Raphson iterations per substep.",
        )
        parser.add_argument(
            "--pcg-iters",
            type=int,
            default=None,
            help="PCG iterations per NR step (default: the tire asset's recommendation).",
        )
        parser.add_argument(
            "--thickness-gp",
            type=int,
            default=3,
            choices=[3, 5],
            help="Through-thickness Gauss points (3=faster, 5=full accuracy).",
        )
        parser.add_argument(
            "--env-spacing",
            type=float,
            default=1.2,
            help="Lateral gap [m] between tire centres in Z-up Y (ANCF X). "
            "Set 0 to co-locate all tires (useful for unit tests).",
        )
        parser.add_argument(
            "--shell-tires",
            type=str,
            default=None,
            help="JSON list of tire configs — one entry per parallel environment. "
            "n_envs = len(list) unless --n-envs overrides. "
            'Example: \'[{"E": 1e7, "nu": 0.45, "pressure": 30000}, ...]\' ',
        )
        parser.add_argument(
            "--rpm",
            type=float,
            default=0.0,
            help="Initial spindle RPM for all envs. Positive = forward. Default: 0.",
        )
        parser.add_argument(
            "--f-load",
            type=float,
            default=_F_LOAD,
            help="Vertical preload per spindle [N] — simulates vehicle body weight. "
            "Clamped to kn*0.025 at runtime to prevent shell inversion. "
            "kn=10k → max 250 N.  kn=300k → max 7500 N.  Default: 0.",
        )
        parser.add_argument(
            "--fast-math",
            action="store_true",
            default=False,
            help="Enable Warp fast-math (~5-15%% speedup). Safe for E>=2MPa; may cause NaN with very soft materials.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
