# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""ANCF FEM tire on a MuJoCo Chrono-style TireTestRig — 3-phase rolling test.

Topology
--------
  world → carrier  [slide_z (free vertical), slide_x (motor-driven longitudinal)]
                  → spindle [spin_y (motor-driven rotation)]
                            → ANCF FEM tire (bead-nodes prescribed)

Phases
------
  DROP   - tire falls under gravity until tread contacts ground (sp_z <= r_outer * 1.002).
  SETTLE - motors idle; tire deforms to static equilibrium for --settle-delay seconds.
  TEST   - lin_motor drives carrier at --long-speed [m/s]; rot_motor spins spindle at
           --ang-rpm [RPM]. Sliders in GUI allow live adjustment.

Normal load
-----------
  carrier_mass = normal_load / g.  Default 3000 N → 306 kg.
  Spindle inertial mass is 0.001 kg (near-zero, Chrono convention); carrier provides load.

Coordinate systems
------------------
  MuJoCo Z-up : x_fwd, y_lat, z_up
  ANCF Y-up   : x_lat, y_up,  z_fwd  (axle along X, tread at Y=0)
  Z-up -> Y-up : (x,y,z) -> (y, z, x)
  Y-up -> Z-up : (x,y,z) -> (z, x, y)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.ancf_shell import (
    build_ancf_parabolic_mesh,
    build_ancf_tire_mesh,
    isotropic_ancf_material,
)
from newton._src.solvers.ancf_shell.solver_ancf_shell_rigid import SolverANCFShellRigid as SolverANCFShell

# ── Phase labels ──────────────────────────────────────────────────────────────
_PHASE_DROP = "drop"
_PHASE_SETTLE = "settle"
_PHASE_TEST = "test"

# ── Rig defaults ──────────────────────────────────────────────────────────────
_NORMAL_LOAD = 0.0  # [N]  applied via xfrc on carrier (GUI slider)
_LONG_SPEED = 0.2  # [m/s] carrier longitudinal speed (TEST phase)
_ANG_RPM = 10.0  # [RPM] spindle angular speed (TEST phase)
_SETTLE_DELAY = 1.0  # [s]  settle time after contact before TEST
_DROP_CLEARANCE_RIG = 0.100  # [m] 10 cm drop height (matches Chrono convention)
_DROP_SPEED = 0.05  # [m/s] carrier lowering speed (Chrono default is 0.1 m/s)

_ASSETS_RIG = os.path.join(os.path.dirname(__file__), "assets", "ancf_tire_testrig.xml")

# ── Tire geometry (FEDA 335/65R22.5) ─────────────────────────────────────────
_R_OUTER = 0.499  # [m] tread crown radius
_R_INNER = 0.286  # [m] bead / rim seat radius
_WIDTH = 0.335  # [m] bead-to-bead width
_N_CIRC = 20  # circumferential divisions
_SEC_DIVS = (1, 2, 3)  # axial element divisions per half-section; 1 bead row per side
_H_SHELL = 0.006  # [m] shell thickness

# ── Shell material defaults (overridden by --shell-tire JSON) ────────────────
_E_TIRE = 1.0e7  # [Pa]  10 MPa
_NU_TIRE = 0.45
_RHO_TIRE = 700.0  # [kg/m^3]
_ALPHA_D = 0.0  # Rayleigh stiffness damping

_PRESSURE = 30_000.0  # [Pa] gauge inflation pressure

# ── FEDA masses — spindle only (carrier provides normal load) ─────────────────
_M_SPINDLE = 0.001  # [kg] near-zero spindle (carrier mass = normal_load/g)
_M_RIGID = _M_SPINDLE

# ── Solver parameter defaults (overridden by CLI args) ───────────────────────
_SIM_SUBSTEPS = 20
_NR_ITERS = 16
_PCG_ITERS = 25
_FRAME_DT = 1.0 / 60.0
_USE_GRAPH = True

# ── Coupling: tare + alpha ────────────────────────────────────────────────────
_M_TIRE = _RHO_TIRE * _H_SHELL * (2.0 * math.pi * _R_OUTER * _WIDTH + 2.0 * math.pi * (_R_OUTER**2 - _R_INNER**2))
_FZ_TARE = _M_TIRE * 9.81
_COUPLING_ALPHA = 1.0

# ── Ground contact defaults (overridden by CLI args) ─────────────────────────
_KN = 10_000.0  # [N/m]
_KD = 20.0  # [N·s/m]
_MU = 0.9

# ── Misc ──────────────────────────────────────────────────────────────────────
_SIM_DURATION = 5.0  # [s]
_GRAVITY = 9.81  # [m/s^2]

# ── Bead ring indices ─────────────────────────────────────────────────────────
_N_AX_DIVS = 2 * sum(_SEC_DIVS)
_N_BEAD_PER_RING = _N_CIRC
_N_BEAD = 2 * _SEC_DIVS[0] * _N_BEAD_PER_RING


# ── Warp kernels ──────────────────────────────────────────────────────────────


@wp.kernel
def _update_mass_damp(
    node_xd: wp.array[wp.vec3],
    lm_flat: wp.array[float],
    alpha_m: float,
    f_pers: wp.array[wp.vec3],
):
    """Write mass-proportional damping into node_f_ext_persistent each substep."""
    i = wp.tid()
    m = lm_flat[i * 6]
    xd = node_xd[i]
    f_pers[i] = wp.vec3(-alpha_m * m * xd[0], -alpha_m * m * xd[1], -alpha_m * m * xd[2])


@wp.kernel
def _fill_phi_from_joint_q(
    joint_q: wp.array[float],
    axle_dof: int,
    rim_phi: wp.array[float],
):
    rim_phi[0] = joint_q[axle_dof]


@wp.kernel
def _prescribe_beads_gpu(
    xpos: wp.array2d[wp.vec3],
    cvel: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    rim_phi: wp.array[float],
    bead_idx: wp.array[wp.int32],
    bead_rest: wp.array[wp.vec3],
    bead_D0: wp.array[wp.vec3],
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_xdd: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
):
    """Prescribe bead nodes: rotate rest offset by phi, translate to hub position."""
    i = wp.tid()
    pos_zu = xpos[0, spindle_mj]
    sv = cvel[0, spindle_mj]

    hub_x = pos_zu[1]
    hub_y = pos_zu[2]
    hub_z = pos_zu[0]
    omega = sv[1]  # MuJoCo ang_y  = ANCF axle spin
    v_lat = sv[4]  # MuJoCo v_y    = ANCF v_x (lateral hub velocity)
    v_vert = sv[5]  # MuJoCo v_z    = ANCF v_y (vertical hub velocity — DROP MOTOR!)

    phi = rim_phi[0]
    idx = bead_idx[i]
    r = bead_rest[i]
    d0 = bead_D0[i]

    cp = wp.cos(phi)
    sp = wp.sin(phi)
    y_loc = r[1] * cp - r[2] * sp
    z_new = r[1] * sp + r[2] * cp

    node_x[idx] = wp.vec3(hub_x + r[0], hub_y + y_loc, hub_z + z_new)
    # Bead velocity: full hub kinematics in ANCF Y-up frame.
    # MuJoCo cvel = (ang_x, ang_y, ang_z, v_x, v_y, v_z) — ANGULAR FIRST.
    # Mapping: MuJoCo (v_x=fwd, v_y=lat, v_z=up) → ANCF (v_z=fwd, v_x=lat, v_y=up)
    # Missing v_vert caused huge constraint forces during DROP (hub descending at 0.05 m/s
    # while bead velocity said 0 → velocity discontinuity → -7184m spindle position).
    v_fwd = sv[3]  # MuJoCo v_x = ANCF v_z (forward, slide_x)
    node_xd[idx] = wp.vec3(v_lat, -omega * z_new + v_vert, omega * y_loc + v_fwd)
    node_xdd[idx] = wp.vec3(0.0, 0.0, 0.0)

    d_y = d0[1] * cp - d0[2] * sp
    d_z = d0[1] * sp + d0[2] * cp
    node_D[idx] = wp.vec3(d0[0], d_y, d_z)
    node_Dd[idx] = wp.vec3(0.0, -omega * d_z, omega * d_y)
    node_Ddd[idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _accum_bead_wrench(
    global_f_int: wp.array[float],
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    xpos: wp.array2d[wp.vec3],
    spindle_mj: int,
    staging: wp.array[wp.spatial_vector],
):
    """Accumulate bead constraint reactions into staging wrench (Newton 3rd law)."""
    i = wp.tid()
    idx = bead_idx[i]
    base = idx * 6
    fx = -global_f_int[base + 0]
    fy = -global_f_int[base + 1]
    fz = -global_f_int[base + 2]
    f = wp.vec3(fx, fy, fz)
    pos_zu = xpos[0, spindle_mj]
    hub_ancf = wp.vec3(pos_zu[1], pos_zu[2], pos_zu[0])
    r = node_x[idx] - hub_ancf
    tau = wp.cross(r, f)
    wp.atomic_add(staging, 0, wp.spatial_vector(tau[0], tau[1], tau[2], fx, fy, fz))


@wp.kernel
def _staging_to_xfrc(
    staging: wp.array[wp.spatial_vector],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    alpha: float,
    tare_fz: float,
):
    """Map ANCF Y-up staging wrench -> MuJoCo Z-up xfrc_applied."""
    w = staging[0]
    tau_zu = wp.vec3(w[2], w[0], w[1])
    f_zu = wp.vec3(w[5], w[3], w[4] + tare_fz)
    cur = xfrc_applied[0, spindle_mj]
    xfrc_applied[0, spindle_mj] = wp.spatial_vector(
        cur[0] + alpha * f_zu[0],
        cur[1] + alpha * f_zu[1],
        cur[2] + alpha * f_zu[2],
        cur[3] + alpha * tau_zu[0],
        cur[4] + alpha * tau_zu[1],
        cur[5] + alpha * tau_zu[2],
    )


@wp.kernel
def _ancf_yup_to_zu(
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
):
    """ANCF Y-up -> Z-up particle positions for rendering: (x,y,z)->(z,x,y)."""
    i = wp.tid()
    p = src[i]
    dst[i] = wp.vec3(p[2], p[0], p[1])


@wp.kernel
def _gather_yup_to_zu(
    src: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    dst: wp.array[wp.vec3],
):
    """Gather indexed nodes from ANCF Y-up and convert to Z-up."""
    i = wp.tid()
    p = src[indices[i]]
    dst[i] = wp.vec3(p[2], p[0], p[1])


@wp.kernel
def _build_ring_lines(
    bead_pos_zu: wp.array[wp.vec3],
    seg_s: wp.array[wp.int32],
    seg_e: wp.array[wp.int32],
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
):
    """Build line-segment start/end arrays from gathered bead positions."""
    i = wp.tid()
    line_starts[i] = bead_pos_zu[seg_s[i]]
    line_ends[i] = bead_pos_zu[seg_e[i]]


@wp.kernel
def _fill_spindle_positions(
    xpos: wp.array2d[wp.vec3],
    spindle_mj: int,
    out: wp.array[wp.vec3],
):
    """Fill every spoke-start element with the spindle centre (Z-up)."""
    i = wp.tid()
    out[i] = xpos[0, spindle_mj]


@wp.kernel
def _apply_normal_load(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    body_mj: int,
    f_load: float,  # downward force [N]
):
    """Apply programmable normal load on carrier body (downward = -Z in MuJoCo Z-up).
    xfrc layout (mujoco_warp FORCE FIRST): [0]=fx [1]=fy [2]=fz [3]=tx [4]=ty [5]=tz."""
    cur = xfrc_applied[0, body_mj]
    xfrc_applied[0, body_mj] = wp.spatial_vector(
        cur[0],
        cur[1],
        cur[2] - f_load,
        cur[3],
        cur[4],
        cur[5],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_dof(builder: newton.ModelBuilder, joint_name: str) -> int:
    for j, label in enumerate(builder.joint_label):
        if label.split("/")[-1] == joint_name:
            return int(builder.joint_qd_start[j])
    raise KeyError(f"joint '{joint_name}' not found")


def _find_dof_q(builder: newton.ModelBuilder, joint_name: str) -> int:
    for j, label in enumerate(builder.joint_label):
        if label.split("/")[-1] == joint_name:
            return int(builder.joint_q_start[j])
    raise KeyError(f"joint '{joint_name}' not found")


def _find_body(model: newton.Model, body_name: str) -> int:
    for j, label in enumerate(model.body_label):
        if label.split("/")[-1] == body_name:
            return j
    raise KeyError(f"body '{body_name}' not found")


# ── Example class ─────────────────────────────────────────────────────────────


class Example:
    """ANCF FEM tire on a MuJoCo Chrono TireTestRig — drop/settle/test phases."""

    def __init__(self, viewer=None, args=None):
        device = "cuda:0"
        wp.init()

        self._frame = 0
        self._t = 0.0
        self.viewer = viewer
        self._diag_period = int(getattr(args, "diag_period", 5))
        self._t_wall = time.perf_counter()
        self._t_step = 0.0
        self._t_render = 0.0
        self._t_kin = 0.0
        self._t_ancf = 0.0
        self._t_dyn = 0.0
        self._t_couple = 0.0  # wrench accumulation (coupling)
        self._t_copy = 0.0  # state_rigid → state_0 wp.copy
        self._t_viz = 0.0  # render viz kernel launches
        self._t_sync = 0.0  # GPU→CPU sync (numpy) calls

        # GUI live readouts
        self._gui_sp_z = 1e9  # set to actual drop height after eval_fk in __init__
        self._gui_sp_x = 0.0
        self._gui_fz = 0.0
        self._gui_bead_drift = 0.0
        self._gui_v_max = 0.0
        self._gui_fps = 0.0
        # Debug: decouple ANCF from MuJoCo — ANCF runs alone with fixed beads
        self._decouple_mujoco = bool(getattr(args, "decouple_mujoco", False))
        if self._decouple_mujoco:
            print("[RIG] *** DECOUPLED MODE: ANCF only, MuJoCo frozen, no wrench ***")
        # NaN tracking — set once and stays True until reset
        self._nan_detected = False
        self._nan_first_frame = -1
        self._nan_sources = []  # which arrays had NaN

        # Rig parameters
        self._normal_load = float(getattr(args, "normal_load", _NORMAL_LOAD))
        self._long_speed = float(getattr(args, "long_speed", _LONG_SPEED))
        self._ang_rpm = float(getattr(args, "ang_rpm", _ANG_RPM))
        self._settle_delay = float(getattr(args, "settle_delay", _SETTLE_DELAY))

        # Phase state
        self._phase = _PHASE_DROP
        self._t_contact = None

        drop_clearance = float(getattr(args, "drop_clearance", _DROP_CLEARANCE_RIG))

        # ── Tire material from --shell-tires JSON ─────────────────────────────
        tire_raw = getattr(args, "shell_tires", None) or getattr(args, "shell_tire", None)
        if isinstance(tire_raw, str):
            parsed = json.loads(tire_raw)
            tire_cfg = parsed[0] if isinstance(parsed, list) else parsed
        elif isinstance(tire_raw, list):
            tire_cfg = tire_raw[0] if tire_raw else {}
        elif isinstance(tire_raw, dict):
            tire_cfg = tire_raw
        else:
            tire_cfg = {}

        r_outer = float(tire_cfg.get("r-outer", _R_OUTER))
        r_inner = float(tire_cfg.get("r-inner", _R_INNER))
        width = float(tire_cfg.get("width", _WIDTH))
        e_tire = float(tire_cfg.get("E", _E_TIRE))
        nu_tire = float(tire_cfg.get("nu", _NU_TIRE))
        rho_tire = float(tire_cfg.get("rho", _RHO_TIRE))
        h_shell = float(tire_cfg.get("thickness", _H_SHELL))
        alpha_d = float(tire_cfg.get("alpha-damp", _ALPHA_D))
        n_circ = int(tire_cfg.get("n-circ", _N_CIRC))
        sec_divs = tuple(int(x) for x in tire_cfg.get("sec-divs", list(_SEC_DIVS)))
        pressure = float(tire_cfg.get("pressure", _PRESSURE))
        mesh_type = str(tire_cfg.get("mesh-type", "polaris"))
        n_ax = int(tire_cfg.get("n-ax", 4))

        self._r_outer = r_outer
        drop_h = r_outer + drop_clearance

        kn = float(getattr(args, "kn", _KN))
        kd = float(getattr(args, "kd", _KD))
        mu = float(getattr(args, "mu", _MU))
        nr_iters = int(getattr(args, "nr_iters", _NR_ITERS))
        pcg_iters = int(getattr(args, "pcg_iters", _PCG_ITERS))
        substeps = int(getattr(args, "substeps", _SIM_SUBSTEPS))

        sim_dt = _FRAME_DT / substeps
        mat = isotropic_ancf_material(E=e_tire, nu=nu_tire, rho=rho_tire, alpha_damp=alpha_d)

        # ── ANCF tire mesh (Y-up: axle along X, tread at Y=0) ────────────────
        if mesh_type == "parabolic":
            self.ancf_model = build_ancf_parabolic_mesh(
                R_outer=r_outer,
                R_inner=r_inner,
                width=width,
                n_circ=n_circ,
                n_ax=n_ax,
                material=mat,
                h_shell=h_shell,
                pressure=pressure,
                device=device,
            )
            n_ax_divs = n_ax
            n_bead_per_ring = n_circ
            n_bead = 2 * n_circ
        else:
            self.ancf_model = build_ancf_tire_mesh(
                R_outer=r_outer,
                R_inner=r_inner,
                width=width,
                n_circ=n_circ,
                section_divs=sec_divs,
                section_mats=(mat, mat, mat),
                section_h=(h_shell, h_shell, h_shell),
                pressure=pressure,
                device=device,
            )
            n_ax_divs = 2 * sum(sec_divs)
            n_bead_per_ring = n_circ
            n_bead = 2 * sec_divs[0] * n_bead_per_ring

        n_nodes = self.ancf_model.n_nodes
        x0_np = self.ancf_model.node_x0.numpy()
        d0_np = self.ancf_model.node_D0.numpy()

        m_tire = rho_tire * h_shell * (2.0 * math.pi * r_outer * width + 2.0 * math.pi * (r_outer**2 - r_inner**2))
        fz_tare = m_tire * 9.81

        # ── Mass-proportional Rayleigh damping (auto-derived) ─────────────────
        _l_sw = math.sqrt((r_outer - r_inner) ** 2 + (width / 2.0) ** 2)
        _k_sw = e_tire * h_shell * 2.0 * math.pi * r_outer / _l_sw
        _m_free = m_tire * (1.0 - n_bead / n_nodes)
        _omega_n = math.sqrt(_k_sw / max(_m_free, 1e-9))
        alpha_m_damp = float(getattr(args, "alpha_m_damp", 2.0 * 0.10 * _omega_n))

        self._substeps = substeps
        self._kn = kn
        self._nr_iters = nr_iters
        self._pcg_iters = pcg_iters
        print(f"[RIG] substeps={substeps}  sim_dt={sim_dt:.6f}s  nr={nr_iters}  pcg={pcg_iters}")
        self._sim_dt = sim_dt
        self._n_bead_per_ring = n_bead_per_ring
        self._sec_divs = sec_divs
        self._n_ax_divs = n_ax_divs
        self._n_bead = n_bead
        self._fz_tare = fz_tare
        self._m_tire = m_tire
        self._alpha_m_damp = alpha_m_damp
        # m_rigid = spindle only; normal_load drives carrier mass in MJCF
        self._m_rigid = float(getattr(args, "m_rigid", _M_RIGID))

        # ── Bead ring node indices ────────────────────────────────────────────
        n_bead_rows = 1 if mesh_type == "parabolic" else sec_divs[0]
        left_rows = [
            np.arange(k * n_bead_per_ring, (k + 1) * n_bead_per_ring, dtype=np.int32) for k in range(n_bead_rows)
        ]
        right_rows = [
            np.arange((n_ax_divs - k) * n_bead_per_ring, (n_ax_divs - k + 1) * n_bead_per_ring, dtype=np.int32)
            for k in range(n_bead_rows)
        ]
        bead_np = np.concatenate(left_rows + right_rows)
        assert len(bead_np) == n_bead, f"expected {n_bead} bead nodes, got {len(bead_np)}"

        self._bead_idx = wp.array(bead_np, dtype=wp.int32, device=device)
        self._bead_rest = wp.array(x0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_D0 = wp.array(d0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)

        # ── ANCF solver (Y-up, gravity along -Y) ─────────────────────────────
        ancf_builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        ancf_newton_model = ancf_builder.finalize(device=device)
        self.ancf_solver = SolverANCFShell(
            model=ancf_newton_model,
            ancf_model=self.ancf_model,
            ground_z=0.0,
            kn=kn,
            kd=kd,
            mu=mu,
            nr_max_iter=nr_iters,
            pcg_max_iter=pcg_iters,
            n_envs=1,
        )

        world_x = x0_np + np.array([0.0, drop_h, 0.0], dtype=np.float32)
        self.ancf_solver.node_x.assign(world_x)
        self.ancf_solver.set_dirichlet_nodes(bead_np)
        self._build_pressure = pressure
        self._pressure_target = pressure
        self._pressure_current = pressure
        self.ancf_solver.set_cavity([pressure], [pressure])

        # ── MuJoCo rigid-body builder (Z-up) ─────────────────────────────────
        _mjcf_path = getattr(args, "mjcf", None)
        if _mjcf_path is None:
            _mjcf_path = _ASSETS_RIG
        elif not os.path.isabs(_mjcf_path):
            _mjcf_path = os.path.join(os.path.dirname(_ASSETS_RIG), _mjcf_path)
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        car.add_mjcf(_mjcf_path, up_axis="Z")

        self._slide_z_q_dof = _find_dof_q(car, "slide_z")
        self._spin_y_qd_dof = _find_dof(car, "spin_y")
        self._spin_y_q_dof = _find_dof_q(car, "spin_y")
        self._slide_x_qd_dof = _find_dof(car, "slide_x")
        print(
            f"[RIG] slide_z q_dof={self._slide_z_q_dof}"
            f"  spin_y qd_dof={self._spin_y_qd_dof}  q_dof={self._spin_y_q_dof}"
            f"  slide_x qd_dof={self._slide_x_qd_dof}"
        )

        # ── Rendering model ───────────────────────────────────────────────────
        world_x_zu = np.stack([world_x[:, 2], world_x[:, 0], world_x[:, 1]], axis=1).astype(np.float32)

        builder = newton.ModelBuilder()
        builder.add_world(car)
        # Ground plane already defined in ancf_tire_testrig.xml worldbody —
        # do NOT call add_ground_plane() here or the plane appears twice.
        builder.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x_zu],
            vel=[(0.0, 0.0, 0.0)] * n_nodes,
            mass=[0.0] * n_nodes,
            radius=[0.001] * n_nodes,
        )
        en_np = self.ancf_model.elem_nodes.numpy()
        tris = np.empty((len(en_np) * 2, 3), dtype=np.int32)
        tris[0::2] = en_np[:, [0, 1, 2]]
        tris[1::2] = en_np[:, [0, 2, 3]]
        builder.add_triangles(
            i=tris[:, 0].tolist(),
            j=tris[:, 1].tolist(),
            k=tris[:, 2].tolist(),
        )
        self.model = builder.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()
        self._ctrl_np = np.zeros(self.control.mujoco.ctrl.shape[0], dtype=np.float32)

        jq_init = self.model.joint_q.numpy().copy()
        # All bodies at same Z (chassis pos="0 0 0") — stable constraint chain.
        # The carrier_strut visual goes UP from hub height, giving the Chrono appearance.
        jq_init[self._slide_z_q_dof] = drop_h
        self.model.joint_q.assign(jq_init)
        self.state_0.joint_q.assign(self.model.joint_q)
        self.state_rigid.joint_q.assign(self.model.joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        wp.launch(
            _ancf_yup_to_zu, dim=n_nodes, inputs=[self.ancf_solver.node_x, self.state_0.particle_q], device=device
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
        self.solver.mjw_model.opt.graph_conditional = False
        print(
            f"[MJ] opt.iterations={self.solver.mjw_model.opt.iterations}"
            f"  graph_conditional={self.solver.mjw_model.opt.graph_conditional}"
            f"  njmax={self.solver.mjw_data.njmax}"
        )

        # ── Spindle body index mapping ────────────────────────────────────────
        self._spindle_newton_idx = _find_body(self.model, "spindle")
        btow = self.solver.mjc_body_to_newton.numpy()[0]
        m = np.where(btow == self._spindle_newton_idx)[0]
        assert len(m), (
            f"spindle Newton body {self._spindle_newton_idx} not in MuJoCo body map. "
            f"body_label={list(self.model.body_label)}"
        )
        self._spindle_mj = int(m[0])

        # Carrier body index for normal-load xfrc injection
        _carrier_newton = _find_body(self.model, "carrier")
        _m2 = np.where(btow == _carrier_newton)[0]
        assert len(_m2), "carrier body not in MuJoCo body map"
        self._carrier_mj = int(_m2[0])

        print(f"[RIG] spindle newton={self._spindle_newton_idx}  mj={self._spindle_mj}")
        print(
            f"[RIG] m_tire={m_tire:.3f} kg  fz_tare={fz_tare:.2f} N"
            f"  coupling_alpha={_COUPLING_ALPHA:.4f}"
            f"  alpha_m_damp={alpha_m_damp:.1f} s^-1"
        )

        # ── Register single-wheel coupling in SolverANCFShellRigid ───────────
        # Single env, single world: bead_np is already the global index array.
        # lateral_offset=0: the spindle sits at its real MuJoCo position in xpos.
        self.ancf_solver.setup_wheel(
            tire_idx=0,
            spindle_mj=self._spindle_mj,
            bead_idx_np=bead_np.astype(np.int32),
            tare_fz=fz_tare,
            world_idx=0,
            lateral_offset=0.0,
            device=device,
        )
        # Spindle inertia is near-zero (0.001 kg); carrier (306 kg) provides the
        # normal load through the constraint chain.  Full coupling (alpha=1.0)
        # is correct here — unlike DW where MuJoCo also computes primary support.
        self.ancf_solver._coupling_alpha = _COUPLING_ALPHA

        # ── GPU scalars ───────────────────────────────────────────────────────
        self._rim_phi = wp.zeros(1, dtype=float, device=device)
        self._xfrc_stg = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self._n_nodes = n_nodes

        # ── Bead ring visualization buffers ───────────────────────────────────
        N = self._n_bead_per_ring
        n_rings = 2 * self._sec_divs[0]
        seg_s_np = np.array([i + k * N for k in range(n_rings) for i in range(N)], dtype=np.int32)
        seg_e_np = np.array([(i + 1) % N + k * N for k in range(n_rings) for i in range(N)], dtype=np.int32)
        n_segs = len(seg_s_np)
        self._bead_pos_zu = wp.zeros(self._n_bead, dtype=wp.vec3, device=device)
        self._spoke_start_zu = wp.zeros(self._n_bead, dtype=wp.vec3, device=device)
        self._ring_seg_s = wp.array(seg_s_np, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_np, dtype=wp.int32, device=device)
        self._ring_line_s = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._n_ring_segs = n_segs

        # ── Graph capture ─────────────────────────────────────────────────────
        self.ancf_solver._fix_dirichlet_in_batched = True
        self.ancf_solver.capture_graph(self._sim_dt)

        self.ancf_solver.node_x.assign(world_x)
        self.ancf_solver.node_xd.zero_()
        self.ancf_solver.node_xdd.zero_()
        self.ancf_solver.node_D.assign(d0_np.astype(np.float32))
        self.ancf_solver.node_Dd.zero_()
        self.ancf_solver.node_Ddd.zero_()
        self.ancf_solver.global_f_int.zero_()
        self.ancf_solver.global_f_int0.zero_()
        self.ancf_solver.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
        self._prescribe_beads()

        # Initialize gui_sp_z from actual xpos so DROP detection doesn't fire on frame 0.
        # (0.0 default would satisfy sp_z <= r_outer immediately → instant SETTLE transition)
        wp.synchronize_device()
        self._gui_sp_z = float(self.solver.xpos.numpy()[0, self._spindle_mj][2])
        print(f"[RIG] initial spindle Z = {self._gui_sp_z:.4f} m  (contact threshold = {self._r_outer * 1.002:.4f} m)")

        wp.launch(
            _gather_yup_to_zu,
            dim=self._n_bead,
            inputs=[self.ancf_solver.node_x, self._bead_idx, self._bead_pos_zu],
            device=device,
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=device,
        )
        wp.launch(
            _fill_spindle_positions,
            dim=self._n_bead,
            inputs=[self.solver.xpos, self._spindle_mj, self._spoke_start_zu],
            device=device,
        )

        if not _USE_GRAPH:
            _anc = self.ancf_solver
            _dt_ref = self._sim_dt
            _anc.graph_step = lambda: _anc.step(None, None, None, None, _dt_ref)

        # ── MuJoCo CUDA graph capture ──────────────────────────────────────────
        # Capture step_kinematics and step_dynamics as standalone graphs so the
        # substep loop launches 3 graphs instead of ~80 raw kernel dispatches.
        self._kin_graph = None
        self._dyn_graph = None
        _dev = "cuda:0"
        wp.synchronize_device(_dev)
        try:
            wp.capture_begin(device=_dev)
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
            self._kin_graph = wp.capture_end(device=_dev)
            print("[RIG] Kinematics graph captured ✓")
        except Exception as e:
            try:
                wp.capture_end(device=_dev)
            except Exception:
                pass
            print(f"[RIG] Kinematics capture failed: {e!r}")

        wp.synchronize_device(_dev)
        try:
            wp.capture_begin(device=_dev)
            self.solver.step_dynamics(self.state_rigid)
            self._dyn_graph = wp.capture_end(device=_dev)
            print("[RIG] Dynamics graph captured ✓")
        except Exception as e:
            try:
                wp.capture_end(device=_dev)
            except Exception:
                pass
            print(f"[RIG] Dynamics capture failed: {e!r}")

        # Force graph exec instantiation outside any capture context
        wp.synchronize_device(_dev)
        if self._kin_graph:
            wp.capture_launch(self._kin_graph)
        if self._dyn_graph:
            wp.capture_launch(self._dyn_graph)
        wp.synchronize_device(_dev)
        print("[RIG] MuJoCo graph execs instantiated ✓")

        # ── Combined substep CUDA graph — matches ancf_rigid_mujoco_tires ───────
        # Captures all self._substeps iterations into ONE graph launch per frame.
        # Uses ancf.step() (unrolled NR+PCG) — ancf.graph_step() cannot be nested.
        # Two _prescribe_beads() calls per substep: before AND after ancf.step(),
        # matching ancf_rigid_mujoco_tires._one_substep_direct(). The second call
        # corrects bead positions shifted by the HHT predictor (v_bead × dt drift).
        self._substep_graph = None
        ancf = self.ancf_solver
        dt = self._sim_dt

        # Pre-warm ancf.step() kernels BEFORE capture_begin (lazy compile → error 900).
        ancf.step(None, None, None, None, dt)
        wp.synchronize_device(_dev)
        # Restore ANCF state after pre-warm
        ancf.node_x.assign(self.ancf_solver.node_x.numpy().copy() * 0 + self.ancf_solver.node_x.numpy())
        ancf.global_f_int.zero_()
        ancf.global_f_int0.zero_()
        ancf.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()

        def _one_substep_direct():
            """One substep: kin → prescribe → ancf.step → prescribe → accumulate → dyn → copy."""
            wp.launch(
                _fill_phi_from_joint_q,
                dim=1,
                inputs=[self.state_0.joint_q, self._spin_y_q_dof, self._rim_phi],
                device=_dev,
            )
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            self._prescribe_beads()
            if self._alpha_m_damp > 0.0:
                wp.launch(
                    _update_mass_damp,
                    dim=self._n_nodes,
                    inputs=[ancf.node_xd, ancf.lumped_mass, self._alpha_m_damp, ancf.node_f_ext_persistent],
                    device=_dev,
                )
            ancf.step(None, None, None, None, dt)
            self._prescribe_beads()  # correct HHT predictor drift
            self._accumulate_wrenches()
            self.solver.step_dynamics(self.state_rigid)
            wp.copy(self.state_0.body_q, self.state_rigid.body_q)
            wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

        try:
            wp.capture_begin(device=_dev)
            for _ in range(self._substeps):
                _one_substep_direct()
            self._substep_graph = wp.capture_end(device=_dev)
            print(f"[RIG FRAME GRAPH] Captured {self._substeps} substeps — 1 launch/frame ✓")
        except Exception as e:
            try:
                wp.capture_end(device=_dev)
            except Exception:
                pass
            self._substep_graph = None
            print(f"[RIG FRAME GRAPH] Capture failed ({e!r}) — falling back to Python loop")

        # Restore ANCF + rigid state after capture warmup
        wp.synchronize_device(_dev)
        ancf.global_f_int.zero_()
        ancf.global_f_int0.zero_()
        ancf.node_f_ext_persistent.zero_()
        self.ancf_model.elem_eas_alpha.zero_()

        if viewer is not None:
            viewer.set_model(self.model)

        print(
            f"[RIG] normal_load={self._normal_load:.0f} N"
            f"  carrier_mass={self._normal_load / _GRAVITY:.1f} kg"
            f"  long_speed={self._long_speed:.3f} m/s"
            f"  ang_rpm={self._ang_rpm:.1f} RPM"
        )

    # ── Coupling helpers ───────────────────────────────────────────────────────

    def _prescribe_beads(self) -> None:
        wp.launch(
            _prescribe_beads_gpu,
            dim=self._n_bead,
            inputs=[
                self.solver.xpos,
                self.solver.cvel,
                self._spindle_mj,
                self._rim_phi,
                self._bead_idx,
                self._bead_rest,
                self._bead_D0,
                self.ancf_solver.node_x,
                self.ancf_solver.node_xd,
                self.ancf_solver.node_xdd,
                self.ancf_solver.node_D,
                self.ancf_solver.node_Dd,
                self.ancf_solver.node_Ddd,
            ],
            device="cuda:0",
        )

    def _accumulate_wrenches(self) -> None:
        self.solver.xfrc_applied.zero_()
        # ANCF bead reactions → spindle xfrc
        self.ancf_solver.accumulate_wheel_wrenches(
            xfrc_applied=self.solver.xfrc_applied,
            xpos=self.solver.xpos,
            device="cuda:0",
        )
        # Normal load: downward xfrc on carrier. Applied only during SETTLE/TEST —
        # during DROP the carrier falls freely under its 1 kg self-weight only,
        # preventing a hard impact that would cause element inversion → NaN.
        if self._normal_load > 0.0 and self._phase != _PHASE_DROP:
            wp.launch(
                _apply_normal_load,
                dim=1,
                inputs=[self.solver.xfrc_applied, self._carrier_mj, self._normal_load],
                device="cuda:0",
            )

    def _set_drop_motor(self, speed: float) -> None:
        """Set drop_motor vertical speed [m/s]. Negative=downward. 0=hold."""
        self._ctrl_np[0] = float(speed)
        self.control.mujoco.ctrl.assign(self._ctrl_np)

    def _engage_motors(self) -> None:
        """Write velocity targets: drop_motor=0 (hold), lin_motor, rot_motor."""
        self._ctrl_np[0] = 0.0  # drop_motor: hold
        self._ctrl_np[1] = float(self._long_speed)  # lin_motor
        self._ctrl_np[2] = float(self._ang_rpm * math.pi / 30.0)  # rot_motor
        self.control.mujoco.ctrl.assign(self._ctrl_np)

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self) -> None:
        # ── Debug: decouple ANCF from MuJoCo ────────────────────────────────
        if self._decouple_mujoco:
            for _sub in range(self._substeps):
                self.ancf_solver.graph_step()
            return
        # ────────────────────────────────────────────────────────────────────

        # Frame graph path (matches ancf_rigid_mujoco_tires: 1 launch per frame)
        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
            wp.synchronize_device()
            return

        # Fallback: Python loop with raw kin/dyn calls (same as ancf_rigid_mujoco_tires fallback)
        for _sub in range(self._substeps):
            _ta = time.perf_counter()
            wp.launch(
                _fill_phi_from_joint_q,
                dim=1,
                inputs=[self.state_0.joint_q, self._spin_y_q_dof, self._rim_phi],
                device="cuda:0",
            )
            # Use raw call (not graph) — matches ancf_single_tire_mujoco.py which is stable.
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
            self._prescribe_beads()
            if self._alpha_m_damp > 0.0:
                wp.launch(
                    _update_mass_damp,
                    dim=self._n_nodes,
                    inputs=[
                        self.ancf_solver.node_xd,
                        self.ancf_solver.lumped_mass,
                        self._alpha_m_damp,
                        self.ancf_solver.node_f_ext_persistent,
                    ],
                    device="cuda:0",
                )
            self._t_kin += time.perf_counter() - _ta

            _ta = time.perf_counter()
            self.ancf_solver.graph_step()
            self._t_ancf += time.perf_counter() - _ta

            # Second prescribe_beads: corrects bead positions shifted by HHT predictor.
            # The predictor moves beads by v_bead × dt each substep; without correction
            # the drift accumulates and causes huge constraint reactions → NaN.
            self._prescribe_beads()

            _ta = time.perf_counter()
            self._accumulate_wrenches()
            self._t_couple += time.perf_counter() - _ta

            _ta = time.perf_counter()
            # Use raw call (not graph) — matches ancf_single_tire_mujoco.py which is stable.
            self.solver.step_dynamics(self.state_rigid)
            self._t_dyn += time.perf_counter() - _ta

            _ta = time.perf_counter()
            wp.copy(self.state_0.body_q, self.state_rigid.body_q)
            wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)
            self._t_copy += time.perf_counter() - _ta

    def step(self) -> None:
        # Phase transitions (Chrono-style motor-controlled lowering):
        #   DROP   — drop_motor lowers carrier at _DROP_SPEED; lin/rot motors idle
        #   SETTLE — drop_motor=0 (hold); normal load engages; tire settles
        #   TEST   — all motors engaged (lin_speed + rot_rpm)
        if self._phase == _PHASE_DROP:
            # Drive drop_motor downward — negative = downward in slide_z axis
            self._set_drop_motor(-_DROP_SPEED)
            if self._gui_sp_z <= self._r_outer * 1.002:
                self._phase = _PHASE_SETTLE
                self._t_contact = self._t
                self._set_drop_motor(0.0)  # stop lowering — hold position
                print(f"[RIG] contact at t={self._t:.3f}s  sp_z={self._gui_sp_z:.4f}m  -> SETTLE")
        elif self._phase == _PHASE_SETTLE:
            if self._t - self._t_contact >= self._settle_delay:
                self._phase = _PHASE_TEST
                self._engage_motors()
                print(f"[RIG] settled -> TEST  long={self._long_speed:.3f} m/s  rpm={self._ang_rpm:.1f}")

        # Ramp CTIS pressure toward GUI target (2000 Pa/frame).
        # Skip set_cavity when pressure is already at target — avoids a GPU array
        # write every frame that triggers implicit stream synchronization overhead.
        if self._build_pressure > 0.0:
            d = self._pressure_target - self._pressure_current
            if abs(d) > 1.0:  # only update when still ramping (>1 Pa from target)
                step = min(abs(d), 2000.0) * (1.0 if d >= 0.0 else -1.0)
                self._pressure_current += step
                self.ancf_solver.set_cavity([self._pressure_current], [self._build_pressure])

        _t0 = time.perf_counter()
        self.simulate()
        self._t_step += time.perf_counter() - _t0
        self._frame += 1
        self._t += _FRAME_DT

        # ── Render viz buffers ────────────────────────────────────────────────
        _ta = time.perf_counter()
        wp.launch(
            _ancf_yup_to_zu,
            dim=self._n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device="cuda:0",
        )
        wp.launch(
            _gather_yup_to_zu,
            dim=self._n_bead,
            inputs=[self.ancf_solver.node_x, self._bead_idx, self._bead_pos_zu],
            device="cuda:0",
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device="cuda:0",
        )
        wp.launch(
            _fill_spindle_positions,
            dim=self._n_bead,
            inputs=[self.solver.xpos, self._spindle_mj, self._spoke_start_zu],
            device="cuda:0",
        )
        self._t_viz += time.perf_counter() - _ta

        # ── Per-frame NaN check (lightweight — reads 1 vec3 only) ───────────────
        # Piggybacked on the xpos read that already happens each frame for DROP.
        # For non-DROP phases: check every 5 frames to keep sync cost low.
        _do_nan_check = (self._phase == _PHASE_DROP) or (self._frame % 5 == 0)
        if _do_nan_check and not self._nan_detected:
            _ta = time.perf_counter()
            _sp = self.solver.xpos.numpy()[0, self._spindle_mj]
            _x0 = self.ancf_solver.node_x.numpy()[0]  # just first node
            self._t_sync += time.perf_counter() - _ta
            if self._phase == _PHASE_DROP:
                self._gui_sp_z = float(_sp[2])
            if math.isnan(float(_sp[2])) or math.isnan(float(_x0[0])):
                self._nan_detected = True
                self._nan_first_frame = self._frame
                self._nan_sources = ["detected by fast check — run full diag for details"]
                print(f"\n{'!' * 60}")
                print(f"[NaN] frame={self._frame}  t={self._t:.3f}s  phase={self._phase}")
                print(f"  spindle Z = {_sp[2]:.4f}   node_x[0] = {_x0}")
                print(
                    f"  (full details at next diag period, frame {self._frame + self._diag_period - self._frame % self._diag_period})"
                )
                print(f"{'!' * 60}\n")
        elif self._phase == _PHASE_DROP and not _do_nan_check:
            # Still need xpos for DROP detection even when not NaN-checking
            _ta = time.perf_counter()
            self._gui_sp_z = float(self.solver.xpos.numpy()[0, self._spindle_mj][2])
            self._t_sync += time.perf_counter() - _ta

        if self._frame % self._diag_period == 0:
            _now = time.perf_counter()
            _fps = self._diag_period / max(_now - self._t_wall, 1e-9)
            _ms = 1e3 / max(_fps, 1e-3)
            self._print_diag(_fps, _ms)
            self._t_wall = _now
            self._t_step = self._t_render = self._t_kin = self._t_ancf = 0.0
            self._t_dyn = self._t_couple = self._t_copy = self._t_viz = self._t_sync = 0.0

    def _print_diag(self, fps: float = 0.0, ms: float = 0.0) -> None:
        # ONE synchronize then bulk-pull all arrays — avoids N separate GPU drains.
        wp.synchronize_device()
        x_np = self.ancf_solver.node_x.numpy()
        xd_np = self.ancf_solver.node_xd.numpy().reshape(-1, 3)
        stg_np = self.ancf_solver._xfrc_stg_per_tire[0].numpy()[0]
        phi_val = float(self._rim_phi.numpy()[0])
        fi_np = self.ancf_solver.global_f_int.numpy()
        bq_np = self.state_0.body_q.numpy()  # bulk after sync — no extra stall
        xpos_all = self.solver.xpos.numpy()  # same sync window

        # ── NaN detection across all simulation arrays ────────────────────────
        _nan_sources = []
        if np.any(np.isnan(x_np)):
            _nan_sources.append("node_x")
        if np.any(np.isnan(xd_np)):
            _nan_sources.append("node_xd")
        if np.any(np.isnan(fi_np)):
            _nan_sources.append("global_f_int")
        if np.any(np.isnan(bq_np)):
            _nan_sources.append("body_q")
        if np.any(np.isnan(stg_np)):
            _nan_sources.append("wrench_stg")
        if np.any(np.isnan(xpos_all)):
            _nan_sources.append("xpos")
        any_nan = len(_nan_sources) > 0
        if any_nan and not self._nan_detected:
            self._nan_detected = True
            self._nan_first_frame = self._frame
            self._nan_sources = _nan_sources
            print(f"\n{'=' * 60}")
            print(f"[NaN DETECTED] frame={self._frame}  t={self._t:.3f}s  phase={self._phase}")
            print(f"  Affected arrays: {', '.join(_nan_sources)}")
            print(f"  node_x  nan_count={int(np.sum(np.isnan(x_np)))}/{x_np.size}")
            print(f"  node_xd nan_count={int(np.sum(np.isnan(xd_np)))}/{xd_np.size}")
            _fi_finite = fi_np[np.isfinite(fi_np)]
            _fi_max_str = f"{float(np.max(np.abs(_fi_finite))):.3e}" if _fi_finite.size else "all-NaN"
            print(f"  fi_max (last finite) = {_fi_max_str}")
            print(f"  sp_Z={float(bq_np[self._spindle_newton_idx][2]):.4f}m")
            print(f"  wrench=(fx={float(stg_np[3]):.1f} fy={float(stg_np[4]):.1f} N)")
            print(f"{'=' * 60}\n")

        sp_tf = bq_np[self._spindle_newton_idx]
        sp_z = float(sp_tf[2])  # MuJoCo Z = vertical
        sp_x = float(sp_tf[0])  # MuJoCo X = forward

        fz_contact = float(stg_np[4])
        fz_exp = self._normal_load

        bead_x = x_np[self._bead_idx.numpy()]
        rest_np = self._bead_rest.numpy()
        cp, sp_ = math.cos(phi_val), math.sin(phi_val)
        hub_ancf = np.array([float(sp_tf[1]), float(sp_tf[2]), float(sp_tf[0])])
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

        fi_finite = bool(np.all(np.isfinite(fi_np)))
        fi_max = float(np.max(np.abs(fi_np))) if fi_finite else float("nan")

        xpos_zu = xpos_all[0, self._spindle_mj]  # reuse bulk-fetched array
        float(abs(xpos_zu[2] - sp_tf[2])) * 1e3

        # Cache for GUI readouts
        self._gui_sp_z = sp_z
        self._gui_sp_x = sp_x
        self._gui_fz = fz_contact
        self._gui_bead_drift = bead_drift_mm
        self._gui_v_max = node_v_max
        self._gui_fps = fps

        _dp = max(self._diag_period, 1)
        _step_ms = 1e3 * self._t_step / _dp
        _render_ms = 1e3 * self._t_render / _dp
        _kin_ms = 1e3 * self._t_kin / _dp
        _ancf_ms = 1e3 * self._t_ancf / _dp
        _dyn_ms = 1e3 * self._t_dyn / _dp
        _couple_ms = 1e3 * self._t_couple / _dp
        _copy_ms = 1e3 * self._t_copy / _dp
        _viz_ms = 1e3 * self._t_viz / _dp
        _sync_ms = 1e3 * self._t_sync / _dp
        _total_py = _kin_ms + _ancf_ms + _dyn_ms + _couple_ms + _copy_ms + _viz_ms + _sync_ms
        _gpu_ms = _ancf_ms  # ANCF graph_step is the pure GPU path (no CPU overhead)
        print(
            f"[{self._frame:4d}] {'NaN!' if any_nan else 'ok  '}"
            f"  t={self._t:.2f}s  phase={self._phase}"
            f"  sp_Z={sp_z:+.4f}m  sp_X={sp_x:+.3f}m"
            f"  F_z={fz_contact:+.0f}N (load~{fz_exp:.0f}N)"
            f"  bead_drift={bead_drift_mm:.3f}mm"
            f"  v_max={node_v_max:.2f}m/s"
            f"  fi_max={fi_max:.2e}"
            f"  fps={fps:.1f}  wall={ms:.1f}ms/frame  phase={self._phase}\n"
            f"  [Python breakdown per frame]\n"
            f"    kin    (MuJoCo step_kin)  = {_kin_ms:6.2f} ms\n"
            f"    ancf   (ANCF graph_step)  = {_ancf_ms:6.2f} ms  ← GPU-bound\n"
            f"    dyn    (MuJoCo step_dyn)  = {_dyn_ms:6.2f} ms\n"
            f"    couple (accumulate wrench) = {_couple_ms:6.2f} ms\n"
            f"    copy   (rigid→state_0)    = {_copy_ms:6.2f} ms\n"
            f"    viz    (render kernels)   = {_viz_ms:6.2f} ms\n"
            f"    sync   (xpos.numpy DROP)  = {_sync_ms:6.2f} ms  ← CPU stall\n"
            f"    substeps={self._substeps}  nr={self._nr_iters}  pcg={self._pcg_iters}"
        )

    # ── GUI ────────────────────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        # ── NaN status banner ─────────────────────────────────────────────────
        if self._nan_detected:
            ui.text(f"*** NaN DETECTED  frame={self._nan_first_frame} ***")
            ui.text(f"  arrays: {', '.join(self._nan_sources)}")
        else:
            ui.text("Health: OK  (no NaN)")

        ui.separator()
        ui.text("Debug")
        changed, val = ui.checkbox("Decouple MuJoCo (ANCF only)", self._decouple_mujoco)
        if changed:
            self._decouple_mujoco = bool(val)
            print(
                f"[RIG] decouple_mujoco={self._decouple_mujoco}"
                f"  ({'ANCF only, beads fixed' if val else 'full coupling'})"
            )

        phase_label = {
            _PHASE_DROP: "DROP  (falling)",
            _PHASE_SETTLE: "SETTLE (contact, waiting)",
            _PHASE_TEST: "TEST  (rolling)",
        }
        ui.text(f"Phase: {phase_label.get(self._phase, self._phase)}")
        ui.separator()

        if self._phase == _PHASE_TEST:
            changed, val = ui.slider_float("long speed [m/s]", self._long_speed, 0.0, 5.0)
            if changed:
                self._long_speed = float(val)
                self._engage_motors()

            changed, val = ui.slider_float("ang RPM", self._ang_rpm, -300.0, 300.0)
            if changed:
                self._ang_rpm = float(val)
                self._engage_motors()
            ui.text(f"  {self._ang_rpm * math.pi / 30.0:+.3f} rad/s")

        if self._build_pressure > 0.0:
            ui.separator()
            ui.text("CTIS pressure")
            p_max = self._build_pressure * 2.0
            changed, val = ui.slider_float("nominal [Pa]", self._pressure_target, 0.0, p_max)
            if changed:
                self._pressure_target = float(val)
            ui.text(f"  current  {self._pressure_current:8.0f} Pa  ({self._pressure_current / 6894.76:.1f} psi)")
            ui.text(
                f"  build    {self._build_pressure:8.0f} Pa"
                f"  gauge {self._pressure_current - self._build_pressure:+.0f} Pa"
            )

        ui.separator()
        ui.text("Live")
        ui.text(f"  fps          {self._gui_fps:6.1f}")
        ui.text(f"  spindle Z    {self._gui_sp_z:+.4f} m")
        ui.text(f"  spindle X    {self._gui_sp_x:+.3f} m")
        ui.text(f"  Fz contact   {self._gui_fz:+.0f} N")
        changed, val = ui.slider_float("normal load [N]##nl", self._normal_load, 0.0, 5000.0)
        if changed:
            self._normal_load = float(val)
        _safe_kn = self._kn * 0.025  # 25mm safe penetration limit
        ui.text(
            f"  {self._normal_load:.0f} N  |  safe limit at kn={self._kn:.0f}: {_safe_kn:.0f} N"
            f"  ({'OK' if self._normal_load <= _safe_kn else 'TOO HIGH → NaN risk'})"
        )
        ui.text(f"  bead drift   {self._gui_bead_drift:.3f} mm")
        ui.text(f"  v_max        {self._gui_v_max:.2f} m/s")

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        _t0 = time.perf_counter()
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "bead_rings",
            self._ring_line_s,
            self._ring_line_e,
            colors=(1.0, 0.45, 0.0),
        )
        self.viewer.log_lines(
            "bead_spokes",
            self._spoke_start_zu,
            self._bead_pos_zu,
            colors=(1.0, 0.90, 0.1),
        )
        self.viewer.end_frame()
        self._t_render += time.perf_counter() - _t0

    # ── Tests ──────────────────────────────────────────────────────────────────

    def test_post_step(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()
        if np.any(np.isnan(x_np)):
            raise AssertionError(f"NaN in node_x at frame {self._frame}, t={self._t:.3f}s")

    def test_final(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()
        xd_np = self.ancf_solver.node_xd.numpy()
        assert not np.any(np.isnan(x_np)), "NaN in node_x at test_final"
        assert not np.any(np.isnan(xd_np)), "NaN in node_xd at test_final"
        assert not np.any(np.isinf(x_np)), "Inf in node_x at test_final"

        bq_np = self.state_0.body_q.numpy()
        sp_tf = bq_np[self._spindle_newton_idx]
        hub_ancf = np.array([float(sp_tf[1]), float(sp_tf[2]), float(sp_tf[0])])
        phi_val = float(self._rim_phi.numpy()[0])
        bead_x = x_np[self._bead_idx.numpy()]
        rest_np = self._bead_rest.numpy()
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

        stg_np = self.ancf_solver._xfrc_stg_per_tire[0].numpy()[0]
        fz = abs(float(stg_np[4]))
        fz_exp = self._normal_load
        rel_err = abs(fz - fz_exp) / max(fz_exp, 1.0)
        assert rel_err < 0.50, f"FAIL: F_z={fz:.1f} N, expected~{fz_exp:.1f} N (err={rel_err * 100:.1f}% > 50%)"

        fi_np = self.ancf_solver.global_f_int.numpy()
        assert np.all(np.isfinite(fi_np)), "Non-finite values in global_f_int"

        print(
            f"[PASS] frame={self._frame}  t={self._t:.1f}s\n"
            f"       bead drift  = {max_drift_m * 1e3:.3f} mm  (limit 1 mm)\n"
            f"       F_z contact = {fz:.1f} N  (normal_load~{fz_exp:.1f} N,"
            f" err={rel_err * 100:.1f}%)\n"
            f"       No NaN/Inf in node_x, node_xd, global_f_int. OK"
        )

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--decouple-mujoco",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Debug: run ANCF in isolation with MuJoCo frozen and beads fixed. "
            "Use to verify FEM stability independent of the coupling.",
        )
        parser.add_argument(
            "--normal-load",
            type=float,
            default=_NORMAL_LOAD,
            help="Normal load [N]. Carrier mass = normal_load / g. Default 3000 N.",
        )
        parser.add_argument(
            "--long-speed",
            type=float,
            default=_LONG_SPEED,
            help="Carrier longitudinal speed in TEST phase [m/s]. Default 0.2.",
        )
        parser.add_argument(
            "--ang-rpm",
            type=float,
            default=_ANG_RPM,
            help="Spindle angular speed in TEST phase [RPM]. Default 10.",
        )
        parser.add_argument(
            "--settle-delay",
            type=float,
            default=_SETTLE_DELAY,
            help="Time to wait after contact before engaging motors [s]. Default 1.0.",
        )
        parser.add_argument(
            "--m-rigid",
            type=float,
            default=_M_RIGID,
            help="Spindle inertial mass [kg] (near-zero; normal load comes from --normal-load).",
        )
        parser.add_argument(
            "--mjcf",
            type=str,
            default=None,
            help="Path to MJCF XML (absolute or relative to assets/). Default: ancf_tire_testrig.xml.",
        )
        parser.add_argument(
            "--drop-clearance",
            type=float,
            default=_DROP_CLEARANCE_RIG,
            help="Initial hub clearance above ground [m]. Default 0.010 m.",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=_SIM_DURATION,
            help="Simulation duration [s].",
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
            default=_KN,
            help="Ground contact normal stiffness [N/m].",
        )
        parser.add_argument(
            "--kd",
            type=float,
            default=_KD,
            help="Ground contact normal damping [N·s/m].",
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
            default=_PCG_ITERS,
            help="PCG iterations per NR step.",
        )
        parser.add_argument(
            "--shell-tire",
            type=str,
            default=None,
            help="JSON string with tire material overrides (single dict).",
        )
        parser.add_argument(
            "--shell-tires",
            type=str,
            default=None,
            help="JSON list of tire configs (single-element list for rig; "
            "first entry used). Accepts the same format as ancf_mujoco_tires.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
