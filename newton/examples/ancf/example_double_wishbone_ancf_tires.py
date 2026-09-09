# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""FEDA double-wishbone vehicle on 4 inflatable ANCF FEM tires.

Combines two examples:

  double_wishbone            (7) - FEDA (FED Alpha) vehicle mechanics: full
                                   double-wishbone suspension, TSDA air-spring
                                   approximations.  Ported from Chrono FEDA_Full.
  ancf_rigid_mujoco_tires    (2) - the MuJoCo <-> ANCF coupling: bead-node
                                   Dirichlet prescription from the spindle pose,
                                   reaction wrench fed back through xfrc_applied.

The rigid wheels of (7) are removed: the FEM tires are the only ground support.

Architecture
------------
  Rigid car    - SolverMuJoCo (Z-up, feda_rims_only.xml, 1 MuJoCo world)
  FEM tires x4 - SolverANCFShellRigid (Y-up, n_envs=4, batched)

Coupling per substep (default --gs-iters 2, newton.solvers.InterfaceCouplerGS):
  1. step_kinematics -> xpos / xquat / cvel on GPU
  2. prescribe ANCF bead nodes from the spindle pose and velocity  (rigid -> soft)
  3. ANCF NR+PCG step (captured CUDA graph)
  4. sum the tire's external load -> xfrc_applied on the spindle  (soft -> rigid)
  5. step_dynamics
  6. repeat from 1 seeded with this iteration's rigid state, gs_iters times.
  --gs-iters 1 runs the single explicit pass and captures the whole frame as one
  CUDA graph.

Coordinate systems
------------------
  MuJoCo Z-up : x_fwd, y_lat, z_up
  ANCF Y-up   : x_lat, y_up,  z_fwd  (axle along X, tread at Y=0)
  Z-up -> Y-up : (x,y,z) -> (y, z, x)
  Y-up -> Z-up : (x,y,z) -> (z, x, y)

Command: python -m newton.examples double_wishbone_ancf_tires
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
from newton._src.solvers.ancf_shell import (
    build_ancf_tire_mesh,
    isotropic_ancf_material,
)
from newton._src.solvers.ancf_shell.solver_ancf_shell_rigid import SolverANCFShellRigid as SolverANCFShell

# ── Vehicle geometry (FEDA 335/65R22.5) ──────────────────────────────────────
_WB_H = 1.651  # half-wheelbase [m]
_TR_H = 0.97663  # half-track [m]
_R_OUTER = 0.499  # tread crown radius [m]
_R_INNER = 0.286  # bead seat radius [m]
_WIDTH = 0.335  # bead-to-bead width [m]

# Spindle world positions in FEDA Z-up: (x_fwd, y_lat, z_up)
_SPINDLE_ZU = np.array(
    [
        [_WB_H, -_TR_H, _R_OUTER],  # front right
        [_WB_H, +_TR_H, _R_OUTER],  # front left
        [-_WB_H, -_TR_H, _R_OUTER],  # rear right
        [-_WB_H, +_TR_H, _R_OUTER],  # rear left
    ],
    dtype=np.float32,
)

_SPINDLE_NAMES = ("spindle_fl", "spindle_fr", "spindle_rl", "spindle_rr")
_AXLE_NAMES = ("axle_fl", "axle_fr", "axle_rl", "axle_rr")
_N_TIRES = 4

# MuJoCo is right-handed and this MJCF steers at +X, so +X is forward, +Z is up and
# +Y is the vehicle's LEFT (ISO 8855).  The FEDA MJCF nonetheless names its -Y bodies
# "_fl"/"_rl", which is inverted; the arrays above follow the MJCF order, so index 0 is
# the front RIGHT corner.  Steering is parallel (both kingpins take the same angle), so
# the swap only ever affected display names — not the dynamics.
# UI/diag order is the driver's FL, FR, RL, RR; each entry maps to its tire index.
_WHEEL_ORDER = (("FL", 1), ("FR", 0), ("RL", 3), ("RR", 2))

# ── Tire mesh defaults ────────────────────────────────────────────────────────
_N_CIRC = 16
# sec_divs[0] is the number of bead rows PER SIDE; the bead rings are the only
# thing resisting axial (spindle-axis) motion of the casing.  2 rows per side =
# 4 rings, 64 pinned nodes.
_SEC_DIVS = (2, 2, 3)
_H_SHELL = 0.010  # [m]

# ── Shell material (locked values, TASK_ANCF_JEEP.md) ─────────────────────────
_E_TIRE = 5.0e7  # [Pa]
_NU_TIRE = 0.45
_RHO_TIRE = 700.0  # [kg/m^3]
_ALPHA_D = 0.15  # Rayleigh stiffness-proportional damping [s]

# Cavity pressure above build.  Pressure and modulus are coupled through the hoop
# strain eps = p*R/(t*E); 30 kPa at 50 MPa gives ~7.5 cm static deflection at
# 14-17 kN per corner (measured 2026-09-09).
_PRESSURE = 30_000.0  # [Pa]
# Pressure the rest shape was meshed at.  The shell sees the GAUGE load
# (nominal - build), so build must be BELOW _PRESSURE or the tire is uninflated.
_BUILD_PRESSURE = 0.0

# ── CTIS envelope: Icelandic super jeep ───────────────────────────────────────
# Þórhallsson, "Automatic Control and User Interface for Central Tire Inflation
# System", MSc thesis, Reykjavík University, 2015 (docker/MSc.pdf):
#   §3.2  setpoint range 0.5-35 psi (0.5-10 psi covers off-road use on its own)
#   §1    typical off-road snow driving 2-8 psi; ~1 psi or less in extreme cases,
#         and the Arctic Trucks footprint study runs 20 psi down to 3 psi
#   §3.4  driver must retrim 2 -> 10 psi in 4 s while moving, i.e. 2 psi/s
#   §3.6  setpoint resolution 0.5 psi below 5 psi, 1 psi above (table 3.1)
# All readings are shown in psi, as the thesis requires.
_PSI = 6894.757  # [Pa/psi]
_CTIS_MIN_PSI = 0.5
_CTIS_MAX_PSI = 35.0
_CTIS_RATE_PSI_S = 2.0  # [psi/s]
# Reference levels: deep snow, soft snow, trail, gravel, highway.
_CTIS_PRESETS = ((2.0, "snow"), (5.0, "trail"), (10.0, "gravel"), (20.0, "road"), (35.0, "hwy"))

# ── FEDA masses (rim collapsed into spindle) ──────────────────────────────────
_M_SPINDLE = 13.08  # [kg]
_M_RIM = 18.80  # [kg]
_M_RIGID = _M_SPINDLE + _M_RIM  # 31.88 kg per corner

# ── Solver budget (locked, TASK_ANCF_JEEP.md) ─────────────────────────────────
# Verified 2026-09-09 to settle and drive; nr=8/pcg=100 and pcg=200 gave the
# same trajectories, so more iterations buy nothing here.
_SIM_SUBSTEPS = 10
_NR_ITERS = 2
_PCG_ITERS = 25
_FRAME_DT = 1.0 / 60.0

# Interface Gauss-Seidel iterations per substep (see --gs-iters).
_GS_ITERS = 2

# ── Ground contact (locked values) ────────────────────────────────────────────
_KN = 20_000.0  # [N/m]
_KD = 42.0  # [N·s/m]  (implicit: in the NR tangent since 2026-09-09)
_MU = 0.9

# Tire reaction torque fed back to the spindle: 0 = force only.  The explicit,
# one-substep-lagged torque path into the axle/kingpin hinges is unstable at
# vehicle loads (measured 2026-09-09: antisymmetric L/R mode, NaN in 6 frames).
_TORQUE_ALPHA = 0.0

_GRAVITY = 9.81
_MAX_STEER = 0.47947  # ±27.5° [rad]
_MAX_SPEED = 20.0  # [rad/s]
_WHEEL_SPEED_RATE = 0.2  # [rad/s per frame] max speed change per step() call

# Initial chassis lowering [m] so the tires start pre-compressed and can carry
# the TSDA preload on frame 0: ~F/(N_contact*kn) = 15.5e3/(24*20e3) ~ 3 cm.
_RIDE_DROP = 0.030

_MJCF_PATH = os.path.join(os.path.dirname(__file__), "assets", "feda_rims_only.xml")


# ── Warp kernels ──────────────────────────────────────────────────────────────


@wp.kernel
def _drive_feda(
    steer_dofs: wp.array[wp.int32],  # kingpin hinge DOFs (FL, FR) — kinematic steering
    cmd: wp.array[wp.float32],  # [0] steer_angle [rad], [1] wheel_speed [rad/s]
    throttle_dofs: wp.array[wp.int32],
    joint_target_pos: wp.array[wp.float32],
    joint_target_vel: wp.array[wp.float32],
):
    """dim=N_TIRES.  Write steering position and wheel-speed targets from a device buffer."""
    tid = wp.tid()
    if tid < steer_dofs.shape[0]:
        joint_target_pos[steer_dofs[tid]] = cmd[0]
    joint_target_vel[throttle_dofs[tid]] = cmd[1]


@wp.kernel
def _prescribe_beads_gpu_dw(
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    cvel: wp.array2d[wp.spatial_vector],
    subtree_com: wp.array2d[wp.vec3],  # mjw_data.subtree_com — cvel's linear reference point
    body_rootid: wp.array[int],  # mjw_model.body_rootid
    spindle_mj_arr: wp.array[wp.int32],
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
    inv_dt: float,  # >0: write bead accel as d(prescribed velocity)/dt; 0: leave accel alone
    vel_predict_dt: float,  # >0: extrapolate hub pos by v*dt (t_n pose -> t_{n+1})
):
    """Prescribe bead node position, director and velocities from the spindle pose.

    MuJoCo Z-up -> ANCF Y-up: (x,y,z) -> (y, z, x).
    """
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]

    mj = spindle_mj_arr[env]
    pos_zu = xpos[0, mj]
    # mujoco_warp stores xquat in MuJoCo order (w, x, y, z) inside a wp.quat whose
    # slots are (x, y, z, w) — reorder before wp.quat_rotate.
    qm = xquat[0, mj]
    q = wp.quat(qm[1], qm[2], qm[3], qm[0])
    sv = cvel[0, mj]

    w_mj = wp.vec3(sv[0], sv[1], sv[2])
    # cvel is "com-based": its linear part is the velocity of the point
    # subtree_com[root] (the whole vehicle's COM), not of the body origin.
    # Transport it to the spindle origin (same correction Newton applies in
    # mujoco/kernels.py mj_body_acceleration).  Without it the bead velocity is
    # wrong by omega x (xpos - com): vehicle pitch rate times the 1.65 m
    # half-wheelbase, opposite sign front vs rear.
    com = subtree_com[0, body_rootid[mj]]
    v_mj = wp.vec3(sv[3], sv[4], sv[5]) + wp.cross(w_mj, pos_zu - com)

    r = bead_rest[i]
    d0 = bead_D0[i]

    # Bead offset is defined in the ANCF frame (x_lat, y_up, z_fwd); express it
    # in the spindle's local MuJoCo frame and rotate by the spindle orientation
    # (xquat already contains the axle spin).
    r_mj = wp.vec3(r[2], r[0], r[1])
    d_mj = wp.vec3(d0[2], d0[0], d0[1])
    r_w = wp.quat_rotate(q, r_mj)
    d_w = wp.quat_rotate(q, d_mj)

    p_w = pos_zu + v_mj * vel_predict_dt + r_w

    # Material velocity of a hub-attached point: v_hub + omega x r.
    vel_w = v_mj + wp.cross(w_mj, r_w)
    dd_w = wp.cross(w_mj, d_w)

    v_new = wp.vec3(vel_w[1], vel_w[2], vel_w[0])
    dd_new = wp.vec3(dd_w[1], dd_w[2], dd_w[0])

    # Bead acceleration consistent with the prescribed velocity (finite difference
    # of the previous prescribed velocity); only on the pre-step call.
    if inv_dt > 0.0:
        node_xdd[global_idx] = (v_new - node_xd[global_idx]) * inv_dt
        node_Ddd[global_idx] = (dd_new - node_Dd[global_idx]) * inv_dt

    node_x[global_idx] = wp.vec3(p_w[1], p_w[2], p_w[0])
    node_xd[global_idx] = v_new
    node_D[global_idx] = wp.vec3(d_w[1], d_w[2], d_w[0])
    node_Dd[global_idx] = dd_new


@wp.kernel
def _ancf_yup_to_zu(src: wp.array[wp.vec3], dst: wp.array[wp.vec3]):
    """ANCF Y-up -> Z-up for particle rendering: (x,y,z) -> (z,x,y)."""
    i = wp.tid()
    p = src[i]
    dst[i] = wp.vec3(p[2], p[0], p[1])


@wp.kernel
def _gather_zu(src: wp.array[wp.vec3], indices: wp.array[wp.int32], dst: wp.array[wp.vec3]):
    """Gather indexed positions already in Z-up."""
    i = wp.tid()
    dst[i] = src[indices[i]]


@wp.kernel
def _gather_contact_spikes(
    node_x: wp.array[wp.vec3],  # ANCF Y-up, all envs flat
    ground_y: float,
    vis_scale: float,
    line_starts: wp.array[wp.vec3],  # Z-up output
    line_ends: wp.array[wp.vec3],  # Z-up output
):
    """GPU-only contact visualization: a spike of height pen*vis_scale per penetrating node."""
    i = wp.tid()
    p = node_x[i]
    pen = ground_y - p[1]
    base = wp.vec3(p[2], p[0], ground_y)
    line_starts[i] = base
    if pen > 0.0:
        line_ends[i] = wp.vec3(base[0], base[1], base[2] + pen * vis_scale)
    else:
        line_ends[i] = base


@wp.kernel
def _build_ring_lines(
    bead_pos_zu: wp.array[wp.vec3],
    seg_s: wp.array[wp.int32],
    seg_e: wp.array[wp.int32],
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
):
    i = wp.tid()
    line_starts[i] = bead_pos_zu[seg_s[i]]
    line_ends[i] = bead_pos_zu[seg_e[i]]


@wp.kernel
def _fill_spindle_positions_dw(
    xpos: wp.array2d[wp.vec3],
    spindle_mj_arr: wp.array[wp.int32],
    out: wp.array[wp.vec3],
    n_bead: int,
):
    """Fill spoke-start positions from xpos[0, spindle_mj_arr[env]] (already Z-up)."""
    tid = wp.tid()
    env = tid // n_bead
    out[tid] = xpos[0, spindle_mj_arr[env]]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_dof(builder: newton.ModelBuilder, joint_name: str) -> int:
    """Return qd DOF index for the joint whose last path segment matches joint_name."""
    for j, label in enumerate(builder.joint_label):
        if label.split("/")[-1] == joint_name:
            return int(builder.joint_qd_start[j])
    raise KeyError(f"joint '{joint_name}' not found in builder")


def _find_body(model: newton.Model, body_name: str) -> int:
    """Return Newton body index whose last path segment matches body_name."""
    for j, label in enumerate(model.body_label):
        if label.split("/")[-1] == body_name:
            return j
    raise KeyError(f"body '{body_name}' not found in model")


# ── Example class ─────────────────────────────────────────────────────────────


class Example:
    """FEDA double-wishbone rigid car + 4 ANCF FEM tires."""

    def __init__(self, viewer=None, args=None):
        device = "cuda:0"
        if getattr(args, "fast_math", False):
            wp.config.fast_math = True
        wp.init()

        self._frame = 0
        self._t = 0.0
        self.viewer = viewer
        self._diag_period = int(getattr(args, "diag_period", 30))
        self._t_wall = time.perf_counter()
        self._gui_fps = 0.0
        self._gui_fz = [0.0] * _N_TIRES
        self._gui_drift = [0.0] * _N_TIRES

        # ── Solver params ─────────────────────────────────────────────────────
        substeps = int(getattr(args, "substeps", _SIM_SUBSTEPS))
        nr_iters = int(getattr(args, "nr_iters", _NR_ITERS))
        pcg_iters = int(getattr(args, "pcg_iters", _PCG_ITERS))
        kn = float(getattr(args, "kn", _KN))
        kd = float(getattr(args, "kd", _KD))
        mu = float(getattr(args, "mu", _MU))
        sim_dt = _FRAME_DT / substeps
        self._substeps = substeps
        self._sim_dt = sim_dt

        # ── Tire material (shell-tires JSON array or flat CLI args) ──────────
        shell_tires = getattr(args, "shell_tires", None)
        if shell_tires:
            if isinstance(shell_tires, str):
                shell_tires = json.loads(shell_tires)
            t0 = shell_tires[0]  # first entry used for all 4 tires
            e_tire = float(t0.get("E", _E_TIRE))
            nu_tire = float(t0.get("nu", _NU_TIRE))
            rho_tire = float(t0.get("rho", _RHO_TIRE))
            h_shell = float(t0.get("thickness", _H_SHELL))
            alpha_d = float(t0.get("alpha-damp", _ALPHA_D))
            pressure = float(t0.get("pressure", _PRESSURE))
            sec_divs = tuple(int(x) for x in t0.get("sec-divs", list(_SEC_DIVS)))
            n_circ = int(t0.get("n-circ", _N_CIRC))
        else:
            e_tire = float(getattr(args, "e_tire", _E_TIRE))
            nu_tire = float(getattr(args, "nu_tire", _NU_TIRE))
            rho_tire = float(getattr(args, "rho_tire", _RHO_TIRE))
            h_shell = float(getattr(args, "h_shell", _H_SHELL))
            alpha_d = float(getattr(args, "alpha_d", _ALPHA_D))
            pressure = float(getattr(args, "pressure", _PRESSURE))
            sec_divs = _SEC_DIVS
            n_circ = _N_CIRC
        thick_gp = int(getattr(args, "thickness_gp", 3))

        mat = isotropic_ancf_material(E=e_tire, nu=nu_tire, rho=rho_tire, alpha_damp=alpha_d)

        # ── ANCF tire mesh (Y-up: axle along X, tread at Y=0) ────────────────
        self.ancf_model = build_ancf_tire_mesh(
            R_outer=_R_OUTER,
            R_inner=_R_INNER,
            width=_WIDTH,
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
        d0_np_tiled = np.tile(d0_np, (_N_TIRES, 1)).astype(np.float32)

        m_tire = rho_tire * h_shell * (2.0 * math.pi * _R_OUTER * _WIDTH + 2.0 * math.pi * (_R_OUTER**2 - _R_INNER**2))
        fz_tare = m_tire * _GRAVITY

        self._n_bead = n_bead
        self._n_nodes = n_nodes
        self._fz_tare = fz_tare

        # ── Bead ring node indices: sec_divs[0] rows pinned per side ─────────
        n_bead_rows = sec_divs[0]
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
        bead_idx_all_np = np.concatenate([bead_np + e * n_nodes for e in range(_N_TIRES)])
        self._bead_idx_all = wp.array(bead_idx_all_np.astype(np.int32), device=device)
        self._bead_rest = wp.array(x0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_D0 = wp.array(d0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_idx_np = bead_np
        self._bead_rest_np = x0_np[bead_np]
        # Free crown node (max radial distance from the axle) for the diagnostics.
        self._crown_idx = int(np.argmax(x0_np[:, 1] ** 2 + x0_np[:, 2] ** 2))
        self._crown_rest_np = x0_np[self._crown_idx]

        # ── ANCF solver (Y-up, gravity along -Y) ─────────────────────────────
        ancf_builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        ancf_newton_model = ancf_builder.finalize(device=device)
        self.ancf_solver = SolverANCFShell(
            model=ancf_newton_model,
            ancf_model=self.ancf_model,
            n_tires=_N_TIRES,
            torque_alpha=float(getattr(args, "torque_alpha", _TORQUE_ALPHA)),
            ground_z=0.0,
            kn=kn,
            kd=kd,
            mu=mu,
            nr_max_iter=nr_iters,
            pcg_max_iter=pcg_iters,
            thickness_gp=thick_gp,
        )

        # Place each ANCF tire at its spindle's position.
        # Spindle (x_fwd, y_lat, z_up) in Z-up -> ANCF (y_lat, z_up, x_fwd).
        world_x = np.concatenate(
            [
                x0_np + np.array([_SPINDLE_ZU[e, 1], _SPINDLE_ZU[e, 2], _SPINDLE_ZU[e, 0]], dtype=np.float32)
                for e in range(_N_TIRES)
            ],
            axis=0,
        )
        self.ancf_solver.node_x.assign(world_x)

        # Dirichlet bead nodes (global indices across all envs).
        bead_global_np = np.concatenate([bead_np + e * n_nodes for e in range(_N_TIRES)])
        self.ancf_solver.set_dirichlet_nodes(bead_global_np)
        self.ancf_solver._fix_dirichlet_in_batched = True
        self.ancf_solver.debug_residuals = bool(getattr(args, "debug_residuals", False))

        # CTIS: set_cavity takes per-env sequences, so each tire holds its own K_gas.
        # Start already inflated — the ramp is a realistic 2 psi/s, so filling from the
        # build pressure would leave the jeep on flat tires for seconds and would not
        # reproduce the settled baseline (h=0.42 m, 14-17 kN/corner).
        self._pressure = pressure
        self._build_pressure = float(getattr(args, "build_pressure", _BUILD_PRESSURE))
        self._pressure_all = pressure  # "all tires" slider position
        self._pressure_targets = [pressure] * _N_TIRES
        self._pressure_currents = [pressure] * _N_TIRES
        self.ancf_solver.set_cavity(self._pressure_currents, [self._build_pressure] * _N_TIRES)

        # ── MuJoCo car builder (Z-up, 1 world) ───────────────────────────────
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        car.default_shape_cfg.mu = mu
        car.add_mjcf(_MJCF_PATH, up_axis="Z")  # ground plane is in the MJCF worldbody

        steer_names = ("upright_fl_steer", "upright_fr_steer")
        self._steer_qd_dof_arr = wp.array([_find_dof(car, n) for n in steer_names], dtype=wp.int32, device=device)
        self._axle_qd_dof_arr = wp.array([_find_dof(car, n) for n in _AXLE_NAMES], dtype=wp.int32, device=device)
        # Steer/throttle commands ride on a 2-element device buffer so GUI changes
        # do not invalidate a captured graph.
        self._cmd_host = np.zeros(2, dtype=np.float32)
        self.cmd = wp.zeros(2, dtype=wp.float32, device=device)

        # ANCF visualization particles (massless, Z-up) + triangles.
        world_x_zu = np.stack([world_x[:, 2], world_x[:, 0], world_x[:, 1]], axis=1).astype(np.float32)
        car.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x_zu],
            vel=[(0.0, 0.0, 0.0)] * (_N_TIRES * n_nodes),
            mass=[0.0] * (_N_TIRES * n_nodes),
            radius=[0.001] * (_N_TIRES * n_nodes),
        )
        en_np = self.ancf_model.elem_nodes.numpy()
        tris = np.empty((len(en_np) * 2, 3), dtype=np.int32)
        tris[0::2] = en_np[:, [0, 1, 2]]
        tris[1::2] = en_np[:, [0, 2, 3]]
        all_tris = np.concatenate([tris + e * n_nodes for e in range(_N_TIRES)], axis=0)
        car.add_triangles(i=all_tris[:, 0].tolist(), j=all_tris[:, 1].tolist(), k=all_tris[:, 2].tolist())

        self.model = car.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()

        # Pre-compress the tires so they can carry the TSDA preload on frame 0.
        ride_drop = float(getattr(args, "ride_drop", _RIDE_DROP))
        if ride_drop != 0.0:
            jq = self.model.joint_q.numpy()
            jq[2] -= ride_drop  # chassis freejoint: [0:3]=pos, [3:7]=quat
            self.model.joint_q.assign(jq)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ── MuJoCo solver ─────────────────────────────────────────────────────
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            iterations=50,
            ls_iterations=10,
            njmax=500,
            nconmax=128,
            # Must be 1: InterfaceCouplerGS rewinds state_0 to t_n between GS
            # iterations and step_kinematics only pushes state_0 into mjData
            # when update_data_interval > 0.
            update_data_interval=1,
        )
        # graph_conditional stays at its default (True): mujoco_warp then wraps the
        # solver loop in a capture_while node and exits once the world converges.
        # With it disabled the frame graph ran all 50 iterations every substep
        # (500 per frame, ~13 launch-bound kernels each; nsys 2026-09-09).

        # ── Spindle names -> MuJoCo body indices; register per-wheel coupling ──
        btow = self.solver.mjc_body_to_newton.numpy()[0]
        spindle_mj_list = []
        for name in _SPINDLE_NAMES:
            nidx = _find_body(self.model, name)
            m = np.where(btow == nidx)[0]
            assert len(m), f"spindle '{name}' (newton={nidx}) not in MuJoCo body map"
            spindle_mj_list.append(int(m[0]))
        self._spindle_mj_arr = wp.array(spindle_mj_list, dtype=wp.int32, device=device)
        for w in range(_N_TIRES):
            self.ancf_solver.setup_wheel(
                tire_idx=w,
                spindle_mj=spindle_mj_list[w],
                bead_idx_np=(bead_np + w * n_nodes).astype(np.int32),
                tare_fz=fz_tare,
                world_idx=0,
                lateral_offset=0.0,
                device=device,
            )
        print(
            f"[DW] mjcf={os.path.basename(_MJCF_PATH)}  spindle_mj={spindle_mj_list}  "
            f"m_tire={m_tire:.3f} kg  torque_alpha={self.ancf_solver.torque_alpha:.3g}  "
            f"substeps={substeps} nr={nr_iters} pcg={pcg_iters}"
        )

        # ── GUI state ─────────────────────────────────────────────────────────
        self.steer_angle = float(getattr(args, "steer_angle", 0.0))
        self.wheel_speed = float(getattr(args, "wheel_speed", 0.0))
        self._target_wheel_speed = self.wheel_speed

        # ── Visualization buffers (bead rings, spokes, contact spikes) ───────
        N_r = n_bead_per_ring
        n_rings = 2 * sec_divs[0]
        seg_s_1 = np.array([i + k * N_r for k in range(n_rings) for i in range(N_r)], dtype=np.int32)
        seg_e_1 = np.array([(i + 1) % N_r + k * N_r for k in range(n_rings) for i in range(N_r)], dtype=np.int32)
        seg_s_all = np.concatenate([seg_s_1 + e * n_bead for e in range(_N_TIRES)])
        seg_e_all = np.concatenate([seg_e_1 + e * n_bead for e in range(_N_TIRES)])
        n_segs = len(seg_s_all)
        self._bead_pos_zu = wp.zeros(_N_TIRES * n_bead, dtype=wp.vec3, device=device)
        self._spoke_start_zu = wp.zeros(_N_TIRES * n_bead, dtype=wp.vec3, device=device)
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=device)
        self._ring_line_s = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._n_ring_segs = n_segs
        self._contact_line_s = wp.zeros(_N_TIRES * n_nodes, dtype=wp.vec3, device=device)
        self._contact_line_e = wp.zeros(_N_TIRES * n_nodes, dtype=wp.vec3, device=device)
        self._contact_vis_scale = 100.0

        # ── ANCF graph capture, then restore the state its warm-up consumed ───
        self.ancf_solver.capture_graph(sim_dt)
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

        # Warm up kinematics + bead prescription so MuJoCo data is initialised.
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, sim_dt)
        self._prescribe_beads()
        self._update_viz_buffers()

        # ── Interface Gauss-Seidel coupling ───────────────────────────────────
        self._gs_iters = max(1, int(getattr(args, "gs_iters", _GS_ITERS)))
        self._gs_coupler = None
        self._gs_prescribe_toggle = 0
        if self._gs_iters > 1:
            self._gs_coupler = newton.solvers.InterfaceCouplerGS(self.ancf_solver, n_iters=self._gs_iters, tol=0.0)
            # elem_eas_alpha is mutated by the ANCF step and is not part of the
            # coupler's built-in pack, so it must be registered to be rewound.
            self._gs_coupler.allocate(self.state_0, extra_arrays=[self.ancf_model.elem_eas_alpha])

        # MuJoCo kinematics/dynamics as standalone graphs (usable inside the GS
        # loop); with gs_iters == 1 the whole substep loop is one frame graph.
        self._kin_graph = None
        self._dyn_graph = None
        self._substep_graph = None
        self._try_capture_mujoco_graphs()
        if self._gs_coupler is None:
            self._try_capture_substep_graph()
        if self._kin_graph is not None:
            self._gs_kinematics_fn = lambda *a: wp.capture_launch(self._kin_graph)
        else:
            self._gs_kinematics_fn = self.solver.step_kinematics
        if self._dyn_graph is not None:
            self._gs_dynamics_fn = lambda *a: wp.capture_launch(self._dyn_graph)
        else:
            self._gs_dynamics_fn = self.solver.step_dynamics

        if viewer is not None:
            viewer.set_model(self.model)
            viewer.set_camera(pos=wp.vec3(-12.0, -18.0, 9.0), pitch=-22.0, yaw=48.0)

    # ── Graph capture ─────────────────────────────────────────────────────────

    def _try_capture_mujoco_graphs(self) -> None:
        """Capture step_kinematics and step_dynamics as standalone CUDA graphs."""
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
            print(f"[DW] kinematics graph capture failed: {e!r}")

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
            print(f"[DW] dynamics graph capture failed: {e!r}")

        # Instantiate the graph execs outside any capture context.
        wp.synchronize_device(dev)
        if self._kin_graph is not None:
            wp.capture_launch(self._kin_graph)
        if self._dyn_graph is not None:
            wp.capture_launch(self._dyn_graph)
        wp.synchronize_device(dev)

    def _try_capture_substep_graph(self) -> None:
        """Capture one full frame (all substeps, explicit coupling) as a single CUDA graph.

        Uses ancf.step() (unrolled NR+PCG) since a graph launch is not capturable.
        Saves and restores all simulation state around the capture warm-up.
        """
        dev = "cuda:0"
        ancf = self.ancf_solver
        dt = self._sim_dt

        wp.synchronize_device(dev)
        saved = {
            "node_x": ancf.node_x.numpy().copy(),
            "node_xd": ancf.node_xd.numpy().copy(),
            "node_xdd": ancf.node_xdd.numpy().copy(),
            "node_D": ancf.node_D.numpy().copy(),
            "node_Dd": ancf.node_Dd.numpy().copy(),
            "node_Ddd": ancf.node_Ddd.numpy().copy(),
            "f_int": ancf.global_f_int.numpy().copy(),
            "f_int0": ancf.global_f_int0.numpy().copy(),
            "eas": self.ancf_model.elem_eas_alpha.numpy().copy(),
            "s0_bq": self.state_0.body_q.numpy().copy(),
            "s0_bqd": self.state_0.body_qd.numpy().copy(),
            "s0_jq": self.state_0.joint_q.numpy().copy(),
            "s0_jqd": self.state_0.joint_qd.numpy().copy(),
            "sr_bq": self.state_rigid.body_q.numpy().copy(),
            "sr_bqd": self.state_rigid.body_qd.numpy().copy(),
            "sr_jq": self.state_rigid.joint_q.numpy().copy(),
            "sr_jqd": self.state_rigid.joint_qd.numpy().copy(),
        }

        def _restore():
            ancf.node_x.assign(saved["node_x"])
            ancf.node_xd.assign(saved["node_xd"])
            ancf.node_xdd.assign(saved["node_xdd"])
            ancf.node_D.assign(saved["node_D"])
            ancf.node_Dd.assign(saved["node_Dd"])
            ancf.node_Ddd.assign(saved["node_Ddd"])
            ancf.global_f_int.assign(saved["f_int"])
            ancf.global_f_int0.assign(saved["f_int0"])
            self.ancf_model.elem_eas_alpha.assign(saved["eas"])
            self.state_0.body_q.assign(saved["s0_bq"])
            self.state_0.body_qd.assign(saved["s0_bqd"])
            self.state_0.joint_q.assign(saved["s0_jq"])
            self.state_0.joint_qd.assign(saved["s0_jqd"])
            self.state_rigid.body_q.assign(saved["sr_bq"])
            self.state_rigid.body_qd.assign(saved["sr_bqd"])
            self.state_rigid.joint_q.assign(saved["sr_jq"])
            self.state_rigid.joint_qd.assign(saved["sr_jqd"])

        # Force ANCF graph exec instantiation outside any capture context.
        ancf.graph_step()
        wp.synchronize_device(dev)
        _restore()

        def _one_substep():
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            self._prescribe_beads(inv_dt=1.0 / dt, vel_predict_dt=dt)
            ancf.step(None, None, None, None, dt)
            self._prescribe_beads()
            self._accumulate_wrenches()
            self.solver.step_dynamics(self.state_rigid)
            self._copy_rigid_to_state0()

        try:
            wp.capture_begin(device=dev)
            for _ in range(self._substeps):
                _one_substep()
            self._substep_graph = wp.capture_end(device=dev)
            print(f"[DW] frame graph captured ({self._substeps} substeps, 1 launch/frame)")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._substep_graph = None
            print(f"[DW] frame graph capture failed ({e!r}) — falling back to loop mode")

        wp.synchronize_device(dev)
        _restore()

    # ── Coupling helpers ───────────────────────────────────────────────────────

    def _prescribe_beads(self, inv_dt: float = 0.0, vel_predict_dt: float = 0.0) -> None:
        wp.launch(
            _prescribe_beads_gpu_dw,
            dim=_N_TIRES * self._n_bead,
            inputs=[
                self.solver.xpos,
                self.solver.xquat,
                self.solver.cvel,
                self.solver.mjw_data.subtree_com,
                self.solver.mjw_model.body_rootid,
                self._spindle_mj_arr,
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
                inv_dt,
                vel_predict_dt,
            ],
            device="cuda:0",
        )

    def _prescribe_beads_gs(self) -> None:
        """Zero-arg ``prescribe_fn`` for ``InterfaceCouplerGS.substep()``.

        Called twice per GS iteration: before the ANCF step (predict the hub
        pose forward by v*dt at k=0 and write a consistent bead acceleration)
        and after it (snap to the solved pose).  At GS iteration k>0
        ``solver.xpos`` already holds the k-th estimate of the t_{n+1} pose, so
        no extrapolation is applied.
        """
        if self._gs_prescribe_toggle == 0:
            predict = self._sim_dt if self._gs_coupler.gs_iter == 0 else 0.0
            self._prescribe_beads(inv_dt=1.0 / self._sim_dt, vel_predict_dt=predict)
        else:
            self._prescribe_beads()
        self._gs_prescribe_toggle = 1 - self._gs_prescribe_toggle

    def _accumulate_wrenches(self) -> None:
        self.solver.xfrc_applied.zero_()
        self.ancf_solver.accumulate_wheel_wrenches(
            xfrc_applied=self.solver.xfrc_applied, xpos=self.solver.xpos, device="cuda:0"
        )

    def _copy_rigid_to_state0(self) -> None:
        wp.copy(self.state_0.body_q, self.state_rigid.body_q)
        wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
        wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
        wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self) -> None:
        """Run one frame (substeps steps)."""
        if self._gs_coupler is not None:
            for _ in range(self._substeps):
                self._gs_coupler.substep(
                    self.state_0,
                    self.state_rigid,
                    self.control,
                    self._sim_dt,
                    self._gs_kinematics_fn,
                    self._prescribe_beads_gs,
                    self._accumulate_wrenches,
                    self._gs_dynamics_fn,
                )
            return

        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
            return

        ancf = self.ancf_solver
        for _ in range(self._substeps):
            self._gs_kinematics_fn(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
            self._prescribe_beads(inv_dt=1.0 / self._sim_dt, vel_predict_dt=self._sim_dt)
            ancf.graph_step()
            self._prescribe_beads()
            self._accumulate_wrenches()
            self._gs_dynamics_fn(self.state_rigid)
            self._copy_rigid_to_state0()

    def _update_controls(self) -> None:
        """Upload steer/throttle commands for this frame (2 floats over the bus)."""
        self._cmd_host[0] = self.steer_angle
        self._cmd_host[1] = self.wheel_speed
        self.cmd.assign(self._cmd_host)
        wp.launch(
            _drive_feda,
            dim=_N_TIRES,
            inputs=[
                self._steer_qd_dof_arr,
                self.cmd,
                self._axle_qd_dof_arr,
                self.control.joint_target_q,
                self.control.joint_target_qd,
            ],
            device="cuda:0",
        )

    def _update_viz_buffers(self) -> None:
        dev = "cuda:0"
        wp.launch(
            _ancf_yup_to_zu,
            dim=_N_TIRES * self._n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device=dev,
        )
        wp.launch(
            _gather_zu,
            dim=_N_TIRES * self._n_bead,
            inputs=[self.state_0.particle_q, self._bead_idx_all, self._bead_pos_zu],
            device=dev,
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=dev,
        )
        wp.launch(
            _fill_spindle_positions_dw,
            dim=_N_TIRES * self._n_bead,
            inputs=[self.solver.xpos, self._spindle_mj_arr, self._spoke_start_zu, self._n_bead],
            device=dev,
        )

    def step(self) -> None:
        # CTIS pressure ramp toward the per-tire GUI targets; build stays fixed.
        if self._pressure > 0.0 and self._pressure_currents != self._pressure_targets:
            rate = _CTIS_RATE_PSI_S * _PSI * _FRAME_DT  # [Pa per frame]
            for e in range(_N_TIRES):
                d = self._pressure_targets[e] - self._pressure_currents[e]
                if d == 0.0:
                    continue
                self._pressure_currents[e] += min(abs(d), rate) * (1.0 if d > 0.0 else -1.0)
            self.ancf_solver.set_cavity(self._pressure_currents, [self._build_pressure] * _N_TIRES)

        # Ramp wheel speed toward the target.
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) <= _WHEEL_SPEED_RATE:
            self.wheel_speed = self._target_wheel_speed
        else:
            self.wheel_speed += _WHEEL_SPEED_RATE if d > 0.0 else -_WHEEL_SPEED_RATE

        self._update_controls()
        self.simulate()
        self._frame += 1
        self._t += _FRAME_DT

        self._update_viz_buffers()

        if self._frame % self._diag_period == 0:
            now = time.perf_counter()
            fps = self._diag_period / max(now - self._t_wall, 1e-9)
            self._gui_fps = fps
            self._t_wall = now
            self._print_diag(fps)

    # ── Diagnostics (host readback only every --diag-period frames) ───────────

    def _print_diag(self, fps: float = 0.0) -> None:
        x_all = self.ancf_solver.node_x.numpy()
        stg_per_wheel = [self.ancf_solver._xfrc_stg_per_tire[w].numpy()[0] for w in range(_N_TIRES)]
        xpos_all = self.solver.xpos.numpy()
        xquat_all = self.solver.xquat.numpy()
        smj = self._spindle_mj_arr.numpy()
        nn = self._n_nodes
        rest_np = self._bead_rest_np

        print(f"\n[{self._frame:4d}] t={self._t:.2f}s  fps={fps:.1f}")
        if self.ancf_solver.debug_residuals:
            for e in range(_N_TIRES):
                print(f"  {self.ancf_solver.residual_report(e)}")
        for label, e in _WHEEL_ORDER:
            x_e = x_all[e * nn : (e + 1) * nn]
            any_nan = bool(np.any(np.isnan(x_e)))
            fz = float(stg_per_wheel[e][4]) + self._fz_tare
            pos_zu = xpos_all[0, int(smj[e])]
            hub = np.array([float(pos_zu[1]), float(pos_zu[2]), float(pos_zu[0])])
            # Expected bead/crown positions from the spindle pose (mirrors _prescribe_beads_gpu_dw).
            qm = xquat_all[0, int(smj[e])]
            qx, qy, qz, qw = float(qm[1]), float(qm[2]), float(qm[3]), float(qm[0])
            u = np.array([qx, qy, qz])

            def rot(r_mj):
                return (
                    r_mj * (qw * qw - u @ u)
                    + 2.0 * (r_mj @ u)[..., None] * u
                    + 2.0 * qw * np.cross(np.broadcast_to(u, r_mj.shape), r_mj)
                )

            r_mj = np.stack([rest_np[:, 2], rest_np[:, 0], rest_np[:, 1]], axis=1)
            p_mj = rot(r_mj) + pos_zu
            expected = np.stack([p_mj[:, 1], p_mj[:, 2], p_mj[:, 0]], axis=1)
            drift_mm = float(np.max(np.linalg.norm(x_e[self._bead_idx_np] - expected, axis=1))) * 1e3
            crown_mj = np.array([[self._crown_rest_np[2], self._crown_rest_np[0], self._crown_rest_np[1]]])
            cp = rot(crown_mj)[0] + pos_zu
            crown_drift_mm = float(np.linalg.norm(x_e[self._crown_idx] - np.array([cp[1], cp[2], cp[0]]))) * 1e3
            self._gui_fz[e] = fz
            self._gui_drift[e] = drift_mm
            print(
                f"  {label}: {'NaN!' if any_nan else 'ok  '}"
                f"  hub=({hub[0]:.3f},{hub[1]:.3f},{hub[2]:.3f})"
                f"  drift={drift_mm:.2f}mm  crown_drift={crown_drift_mm:.1f}mm  Fz={fz:+.0f}N"
            )

    # ── GUI ────────────────────────────────────────────────────────────────────

    def gui(self, ui) -> None:
        ui.text("FEDA Double-Wishbone + ANCF Tires")
        ui.separator()

        changed, val = ui.slider_float("steer [rad]", self.steer_angle, -_MAX_STEER, _MAX_STEER)
        if changed:
            self.steer_angle = float(val)
        changed, val = ui.slider_float("throttle [rad/s]", self._target_wheel_speed, -_MAX_SPEED, _MAX_SPEED)
        if changed:
            self._target_wheel_speed = float(val)

        ui.separator()
        ui.text(f"forward speed ~ {self.wheel_speed * _R_OUTER:+.2f} m/s")
        if abs(self.steer_angle) > 1e-3:
            ui.text(f"turn radius   ~ {(2.0 * _WB_H) / math.tan(abs(self.steer_angle)):.2f} m")
        else:
            ui.text("turn radius   ~ straight")

        if self._pressure > 0.0:
            ui.separator()
            ui.text(f"CTIS setpoint [psi]   super jeep {_CTIS_MIN_PSI:g} - {_CTIS_MAX_PSI:g}")
            changed, val = ui.slider_float("all tires", self._pressure_all / _PSI, _CTIS_MIN_PSI, _CTIS_MAX_PSI)
            if changed:
                self._pressure_all = float(val) * _PSI
                self._pressure_targets = [self._pressure_all] * _N_TIRES
            for i, (psi, name) in enumerate(_CTIS_PRESETS):
                if i:
                    ui.same_line()
                if ui.button(f"{psi:g} {name}"):
                    self._pressure_all = psi * _PSI
                    self._pressure_targets = [self._pressure_all] * _N_TIRES
            for label, e in _WHEEL_ORDER:
                cur = self._pressure_targets[e] / _PSI
                changed, val = ui.slider_float(label, cur, _CTIS_MIN_PSI, _CTIS_MAX_PSI)
                if changed:
                    self._pressure_targets[e] = float(val) * _PSI
            # Shell load is the gauge value (nominal - build); set_cavity holds absolute.
            for label, e in _WHEEL_ORDER:
                gauge = self._pressure_currents[e] - self._build_pressure
                ui.text(f"  {label} {gauge / _PSI:5.2f} psi   ({self._pressure_currents[e]:7.0f} Pa abs)")

        ui.separator()
        ui.text("Live")
        ui.text(f"  fps   {self._gui_fps:6.1f}")
        for label, e in _WHEEL_ORDER:
            ui.text(f"  {label}  Fz={self._gui_fz[e]:+.0f} N  drift={self._gui_drift[e]:.2f} mm")

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, colors=(1.0, 0.45, 0.0))
        self.viewer.log_lines("bead_spokes", self._spoke_start_zu, self._bead_pos_zu, colors=(1.0, 0.90, 0.1))
        wp.launch(
            _gather_contact_spikes,
            dim=_N_TIRES * self._n_nodes,
            inputs=[self.ancf_solver.node_x, 0.0, self._contact_vis_scale, self._contact_line_s, self._contact_line_e],
            device="cuda:0",
        )
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, colors=(0.0, 1.0, 1.0))
        self.viewer.end_frame()

    # ── Tests ──────────────────────────────────────────────────────────────────

    def test_post_step(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()[: self._n_nodes]
        if np.any(np.isnan(x_np)):
            raise AssertionError(f"NaN in ANCF node_x at frame {self._frame}, t={self._t:.3f}s")

    def test_final(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()
        xd_np = self.ancf_solver.node_xd.numpy()
        assert np.all(np.isfinite(x_np)), "non-finite node_x at test_final"
        assert np.all(np.isfinite(xd_np)), "non-finite node_xd at test_final"

        stg_per_wheel = [self.ancf_solver._xfrc_stg_per_tire[w].numpy()[0] for w in range(_N_TIRES)]
        xpos_all = self.solver.xpos.numpy()
        xquat_all = self.solver.xquat.numpy()
        smj = self._spindle_mj_arr.numpy()
        rest_np = self._bead_rest_np
        nn = self._n_nodes

        for e in range(_N_TIRES):
            x_e = x_np[e * nn : (e + 1) * nn]
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
            max_drift = float(np.max(np.linalg.norm(x_e[self._bead_idx_np] - expected, axis=1)))
            assert max_drift < 1e-3, f"FAIL tire {e}: bead drift {max_drift * 1e3:.2f} mm > 1 mm"
            fz = abs(float(stg_per_wheel[e][4]))
            fz_exp = _M_RIGID * _GRAVITY
            rel = abs(fz - fz_exp) / max(fz_exp, 1.0)
            assert rel < 0.50, f"FAIL tire {e}: F_z={fz:.1f} N  expected~{fz_exp:.1f} N  err={rel * 100:.1f}% > 50%"

        assert np.all(np.isfinite(self.ancf_solver.global_f_int.numpy())), "non-finite global_f_int at test_final"
        print(f"[PASS] frame={self._frame}  t={self._t:.1f}s  all 4 tires OK")

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--wheel-speed", type=float, default=0.0, help="Throttle angular speed [rad/s].")
        parser.add_argument(
            "--steer-angle", type=float, default=0.0, help=f"Steering [rad], range ±{_MAX_STEER:.5f} (±27.5°)."
        )
        parser.add_argument("--substeps", type=int, default=_SIM_SUBSTEPS, help="Substeps per frame.")
        parser.add_argument("--nr-iters", type=int, default=_NR_ITERS, help="Newton-Raphson iterations per substep.")
        parser.add_argument("--pcg-iters", type=int, default=_PCG_ITERS, help="PCG iterations per NR step.")
        parser.add_argument("--kn", type=float, default=_KN, help="Contact normal stiffness [N/m].")
        parser.add_argument("--kd", type=float, default=_KD, help="Contact damping [N·s/m].")
        parser.add_argument("--mu", type=float, default=_MU, help="Friction coefficient.")
        parser.add_argument(
            "--shell-tires",
            type=str,
            default=None,
            help="JSON array of per-tire configs (same format as ancf_rigid_mujoco_tires); "
            "first entry used for all 4 tires, overrides the flat --e-tire etc. args.",
        )
        parser.add_argument("--pressure", type=float, default=_PRESSURE, help="Nominal cavity pressure [Pa].")
        parser.add_argument(
            "--build-pressure",
            type=float,
            default=_BUILD_PRESSURE,
            help="Pressure the rest shape was meshed at [Pa]; the shell sees pressure - build_pressure.",
        )
        parser.add_argument("--e-tire", type=float, default=_E_TIRE, help="Shell Young's modulus [Pa].")
        parser.add_argument("--nu-tire", type=float, default=_NU_TIRE, help="Poisson ratio.")
        parser.add_argument("--rho-tire", type=float, default=_RHO_TIRE, help="Density [kg/m^3].")
        parser.add_argument("--h-shell", type=float, default=_H_SHELL, help="Shell thickness [m].")
        parser.add_argument(
            "--thickness-gp", type=int, default=3, choices=[3, 5], help="Through-thickness Gauss points."
        )
        parser.add_argument(
            "--ride-drop",
            type=float,
            default=_RIDE_DROP,
            help="Lower the chassis by this many metres at t=0 so the tires start pre-compressed.",
        )
        parser.add_argument(
            "--torque-alpha",
            type=float,
            default=_TORQUE_ALPHA,
            help="Scale on the tire reaction torque fed back to the spindle (0 = force only).",
        )
        parser.add_argument(
            "--gs-iters",
            type=int,
            default=_GS_ITERS,
            help="Interface Gauss-Seidel iterations per substep. 1 = single explicit pass captured as one "
            "frame graph; 2 removes the leading-order coupling lag (Python loop, ~2x slower).",
        )
        parser.add_argument("--fast-math", action="store_true", default=False, help="Enable Warp fast-math.")
        parser.add_argument(
            "--diag-period", type=int, default=30, help="Print per-wheel diagnostics every N frames (host readback)."
        )
        parser.add_argument(
            "--debug-residuals",
            action="store_true",
            default=False,
            help="Report per-NR-iteration ||R|| and the final PCG reduction per tire each --diag-period frames.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
