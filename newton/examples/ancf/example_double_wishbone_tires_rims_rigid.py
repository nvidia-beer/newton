# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""FEDA double-wishbone rigid car with 4 ANCF FEM tires — SolverANCFShellRigid variant.

Same as double_wishbone_tires_rims but uses SolverANCFShellRigid, which exposes
a per-wheel wrench API as the insertion point for future bilateral rim-body
constraints (Chrono ChLinkNodeFrame style).  Same MJCF, mesh, and coupling loop
— only the solver class and _accumulate_wrenches differ.

Architecture
------------
  Rigid car    - SolverMuJoCo (Z-up, double_wishbone.xml, 1 MuJoCo world)
  FEM tires x4 - SolverANCFShellRigid (Y-up, n_envs=4, batched)

Coupling (per substep):
  1. Sync rim angle from each axle joint_q
  2. step_kinematics -> xpos on GPU
  3. Prescribe ANCF bead node positions from spindle xpos  (rigid -> soft)
  4. ANCF NR+PCG step
  5. Accumulate ANCF bead reactions -> xfrc_applied on each spindle  (soft -> rigid)
  6. step_dynamics

Key difference from ancf_mujoco_tires
--------------------------------------
  ancf_mujoco_tires uses N isolated MuJoCo worlds: xpos[env, spindle_mj].
  This file uses 1 MuJoCo world with 4 spindle bodies: xpos[0, spindle_mj_arr[env]].

Coordinate systems
------------------
  MuJoCo Z-up : x_fwd, y_lat, z_up
  ANCF Y-up   : x_lat, y_up,  z_fwd  (axle along X, tread at Y=0)
  Z-up -> Y-up : (x,y,z) -> (y, z, x)
  Y-up -> Z-up : (x,y,z) -> (z, x, y)

Command: python -m newton.examples double_wishbone_tires_rims
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
        [_WB_H, -_TR_H, _R_OUTER],  # FL
        [_WB_H, +_TR_H, _R_OUTER],  # FR
        [-_WB_H, -_TR_H, _R_OUTER],  # RL
        [-_WB_H, +_TR_H, _R_OUTER],  # RR
    ],
    dtype=np.float32,
)

_SPINDLE_NAMES = ("spindle_fl", "spindle_fr", "spindle_rl", "spindle_rr")
_AXLE_NAMES = ("axle_fl", "axle_fr", "axle_rl", "axle_rr")
_N_TIRES = 4

# ── Tire mesh defaults ────────────────────────────────────────────────────────
_N_CIRC = 20
_SEC_DIVS = (1, 2, 3)
_H_SHELL = 0.006  # [m]

# ── Shell material defaults ───────────────────────────────────────────────────
_E_TIRE = 1.0e7  # [Pa]  10 MPa
_NU_TIRE = 0.45
_RHO_TIRE = 700.0  # [kg/m^3]
_ALPHA_D = 0.0  # Rayleigh stiffness damping (zeroed; HHT provides numerical damping)

_PRESSURE = 30_000.0  # [Pa] gauge inflation pressure

# ── FEDA masses (rim collapsed into spindle) ──────────────────────────────────
_M_SPINDLE = 13.08  # [kg]
_M_RIM = 18.80  # [kg]
_M_RIGID = _M_SPINDLE + _M_RIM  # 31.88 kg per corner

# ── Solver defaults ───────────────────────────────────────────────────────────
_SIM_SUBSTEPS = 10
_NR_ITERS = 4
_PCG_ITERS = 25
_FRAME_DT = 1.0 / 60.0

# ── Coupling: tare + alpha ────────────────────────────────────────────────────
# alpha=1.0: direct xfrc coupling (no GS mass weighting needed).
# Tare cancels tire self-weight from bead reactions so pre-contact xfrc ≈ 0.
_COUPLING_ALPHA = 0.05  # MuJoCo handles primary support; ANCF adds deformation forces

# ── Ground contact defaults ───────────────────────────────────────────────────
_KN = 10_000.0  # [N/m]
_KD = 20.0  # [N·s/m]
_MU = 0.9

_GRAVITY = 9.81
_MAX_STEER = 0.47947  # ±27.5° [rad]
_MAX_SPEED = 20.0  # [rad/s]
_WHEEL_SPEED_RATE = 0.2  # [rad/s per frame] max speed change per step() call

_MJCF_PATH = os.path.join(os.path.dirname(__file__), "assets", "feda_rims_only.xml")


# ── Warp kernels ──────────────────────────────────────────────────────────────


@wp.kernel
def _fill_phi_from_joint_q_dw(
    joint_q: wp.array[float],
    axle_q_dof_arr: wp.array[wp.int32],
    rim_phi: wp.array[float],
):
    """dim=N_TIRES.  Read axle rotation angle from joint_q for each tire."""
    env = wp.tid()
    rim_phi[env] = joint_q[axle_q_dof_arr[env]]


@wp.kernel
def _update_mass_damp(
    node_xd: wp.array[wp.vec3],
    lm_flat: wp.array[float],
    alpha_m: float,
    f_pers: wp.array[wp.vec3],
):
    """Write mass-proportional Rayleigh damping into node_f_ext_persistent.

    f_damp[i] = -alpha_m * m_node[i] * xd[i].
    Bead nodes have xd=0 (prescribed), so their damping force is zero.
    """
    i = wp.tid()
    m = lm_flat[i * 6]
    xd = node_xd[i]
    f_pers[i] = wp.vec3(-alpha_m * m * xd[0], -alpha_m * m * xd[1], -alpha_m * m * xd[2])


@wp.kernel
def _prescribe_beads_gpu_dw(
    xpos: wp.array2d[wp.vec3],
    cvel: wp.array2d[wp.spatial_vector],
    spindle_mj_arr: wp.array[wp.int32],
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
    n_bead: int,
    n_nodes: int,
):
    """Prescribe bead nodes from spindle position and axle spin.

    Uses hub position from xpos[0, spindle_mj] and rim_phi for axle rotation.
    MuJoCo Z-up → ANCF Y-up: (x,y,z) → (y, z, x).
    """
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]

    pos_zu = xpos[0, spindle_mj_arr[env]]
    sv = cvel[0, spindle_mj_arr[env]]

    omega = sv[1]  # MuJoCo ang_y = ANCF ang_x (axle spin)
    v_fwd = sv[3]  # MuJoCo vx = forward velocity

    hub_x = pos_zu[1]  # MuJoCo Y (lateral) -> ANCF X
    hub_y = pos_zu[2]  # MuJoCo Z (vertical) -> ANCF Y
    hub_z = pos_zu[0]  # MuJoCo X (forward)  -> ANCF Z

    phi = rim_phi[env]
    r = bead_rest[i]
    d0 = bead_D0[i]

    cp = wp.cos(phi)
    sp = wp.sin(phi)
    y_loc = r[1] * cp - r[2] * sp
    z_new = r[1] * sp + r[2] * cp

    node_x[global_idx] = wp.vec3(hub_x + r[0], hub_y + y_loc, hub_z + z_new)
    node_xd[global_idx] = wp.vec3(0.0, -omega * z_new, omega * y_loc + v_fwd)
    node_xdd[global_idx] = wp.vec3(0.0, 0.0, 0.0)

    d_y = d0[1] * cp - d0[2] * sp
    d_z = d0[1] * sp + d0[2] * cp
    node_D[global_idx] = wp.vec3(d0[0], d_y, d_z)
    node_Dd[global_idx] = wp.vec3(0.0, -omega * d_z, omega * d_y)
    node_Ddd[global_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _accum_bead_wrench_dw(
    global_f_int: wp.array[float],
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    xpos: wp.array2d[wp.vec3],
    spindle_mj_arr: wp.array[wp.int32],
    staging: wp.array[wp.spatial_vector],
    n_bead: int,
    n_nodes: int,
):
    """Accumulate bead constraint reactions into per-env staging wrench (Newton 3rd law).

    global_f_int[bead] is the Lagrange multiplier lambda for the bilateral
    bead constraint (per Chrono ChLinkNodeFrame).  Newton 3rd law: reaction on
    spindle = -lambda.  Moment arm from hub centre to bead position.
    """
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]
    base = global_idx * 6
    fx = -global_f_int[base + 0]
    fy = -global_f_int[base + 1]
    fz = -global_f_int[base + 2]
    f = wp.vec3(fx, fy, fz)
    pos_zu = xpos[0, spindle_mj_arr[env]]
    hub_ancf = wp.vec3(pos_zu[1], pos_zu[2], pos_zu[0])  # Z-up -> Y-up
    r = node_x[global_idx] - hub_ancf
    tau = wp.cross(r, f)
    wp.atomic_add(staging, env, wp.spatial_vector(tau[0], tau[1], tau[2], fx, fy, fz))


@wp.kernel
def _staging_to_xfrc_dw(
    staging: wp.array[wp.spatial_vector],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj_arr: wp.array[wp.int32],
    alpha: float,
    tare_fz: float,
):
    """Map ANCF Y-up staging wrench -> MuJoCo Z-up xfrc_applied on each spindle.

    xfrc layout (mujoco_warp): [0:3]=force, [3:6]=torque  (FORCE FIRST).
    tare_fz cancels tire gravity so pre-contact xfrc = 0.
    1 MuJoCo world: xfrc_applied[0, spindle_mj_arr[env]].
    Each env writes to a different body index — no race condition.
    """
    env = wp.tid()
    w = staging[env]
    tau_zu = wp.vec3(w[2], w[0], w[1])  # ANCF Y-up tau -> Z-up
    f_zu = wp.vec3(w[5], w[3], w[4] + tare_fz)  # ANCF Y-up f -> Z-up + tare
    cur = xfrc_applied[0, spindle_mj_arr[env]]
    xfrc_applied[0, spindle_mj_arr[env]] = wp.spatial_vector(
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
    """ANCF Y-up -> Z-up for particle rendering: (x,y,z) -> (z,x,y)."""
    i = wp.tid()
    p = src[i]
    dst[i] = wp.vec3(p[2], p[0], p[1])


@wp.kernel
def _gather_zu(
    src: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    dst: wp.array[wp.vec3],
):
    """Gather indexed positions already in Z-up (no coordinate conversion)."""
    i = wp.tid()
    dst[i] = src[indices[i]]


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


def _find_dof_q(builder: newton.ModelBuilder, joint_name: str) -> int:
    """Return q coordinate index for the joint whose last path segment matches joint_name."""
    for j, label in enumerate(builder.joint_label):
        if label.split("/")[-1] == joint_name:
            return int(builder.joint_q_start[j])
    raise KeyError(f"joint '{joint_name}' not found in builder")


def _find_body(model: newton.Model, body_name: str) -> int:
    """Return Newton body index whose last path segment matches body_name."""
    for j, label in enumerate(model.body_label):
        if label.split("/")[-1] == body_name:
            return j
    raise KeyError(f"body '{body_name}' not found in model")


# ── Example class ─────────────────────────────────────────────────────────────


class Example:
    """FEDA double-wishbone rigid car + 4 ANCF FEM tires.

    1 MuJoCo world drives full suspension dynamics; 4 SolverANCFShell envs
    model tire deformation and ground contact.  Coupling per substep:
    spindle xpos -> bead Dirichlet -> ANCF NR -> bead reactions -> spindle xfrc.
    """

    def __init__(self, viewer=None, args=None):
        device = "cuda:0"
        if getattr(args, "fast_math", False):
            wp.config.fast_math = True
        wp.init()

        self._frame = 0
        self._t = 0.0
        self.viewer = viewer
        self._diag_period = int(getattr(args, "diag_period", 5))
        self._t_wall = time.perf_counter()
        self._t_step = 0.0
        self._t_render = 0.0
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
        if shell_tires and len(shell_tires) > 0:
            if isinstance(shell_tires, str):
                shell_tires = json.loads(shell_tires)
            t0 = shell_tires[0]  # use first entry for all 4 tires
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

        # Auto-derive mass-proportional Rayleigh damping coefficient
        _l_sw = math.sqrt((_R_OUTER - _R_INNER) ** 2 + (_WIDTH / 2.0) ** 2)
        _k_sw = e_tire * h_shell * 2.0 * math.pi * _R_OUTER / _l_sw
        _m_free = m_tire * (1.0 - n_bead / n_nodes)
        _omega_n = math.sqrt(_k_sw / max(_m_free, 1e-9))
        alpha_m_damp = float(getattr(args, "alpha_m_damp", 2.0 * 0.10 * _omega_n))

        self._n_bead = n_bead
        self._n_nodes = n_nodes
        self._n_bead_per_ring = n_bead_per_ring
        self._sec_divs = sec_divs
        self._n_ax_divs = n_ax_divs
        self._fz_tare = fz_tare
        self._m_tire = m_tire
        self._alpha_m_damp = alpha_m_damp

        # ── Bead ring node indices ────────────────────────────────────────────
        # sec_divs[0] rows pinned per side (left: j=0..k, right: j=n_ax_divs-k+1)
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
            n_envs=_N_TIRES,
            thickness_gp=thick_gp,
        )

        # Place each ANCF tire at its spindle's Y-up position.
        # Spindle (x_fwd, y_lat, z_up) in Z-up -> ANCF (y_lat, z_up, x_fwd).
        world_x = np.concatenate(
            [
                x0_np
                + np.array(
                    [
                        _SPINDLE_ZU[e, 1],  # y_lat  -> ANCF X
                        _SPINDLE_ZU[e, 2],  # z_up   -> ANCF Y
                        _SPINDLE_ZU[e, 0],  # x_fwd  -> ANCF Z
                    ],
                    dtype=np.float32,
                )
                for e in range(_N_TIRES)
            ],
            axis=0,
        )
        self.ancf_solver.node_x.assign(world_x)

        # Global Dirichlet: _step_batched zeros NR corrector at bead nodes.
        # Must be global indices (e * n_nodes + local) to cover all envs.
        bead_global_np = np.concatenate([bead_np + e * n_nodes for e in range(_N_TIRES)])
        self.ancf_solver.set_dirichlet_nodes(bead_global_np)

        self._pressure = pressure
        self._pressure_target = pressure
        self._pressure_current = pressure
        self.ancf_solver.set_cavity([pressure] * _N_TIRES, [pressure] * _N_TIRES)

        # ── MuJoCo car builder (Z-up, 1 world) ───────────────────────────────
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        car.default_shape_cfg.mu = mu
        car.add_mjcf(_MJCF_PATH, up_axis="Z")
        # Ground plane is defined in double_wishbone.xml worldbody — no add_ground_plane() needed.

        # Find DOF indices before finalize (particle additions don't shift joints).
        self._steer_qd_dof = _find_dof(car, "steer_motor")
        axle_q_dofs = [_find_dof_q(car, n) for n in _AXLE_NAMES]
        axle_qd_dofs_np = np.array([_find_dof(car, n) for n in _AXLE_NAMES], dtype=np.int32)
        self._axle_q_dof_arr = wp.array(axle_q_dofs, dtype=wp.int32, device=device)
        self._axle_qd_dofs = axle_qd_dofs_np  # CPU array for control writes

        print(f"[DW] steer_qd_dof={self._steer_qd_dof}")
        print(f"[DW] axle_q_dofs={axle_q_dofs}  axle_qd_dofs={axle_qd_dofs_np.tolist()}")

        # Add ANCF visualization particles (massless, Z-up coordinates).
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
        car.add_triangles(
            i=all_tris[:, 0].tolist(),
            j=all_tris[:, 1].tolist(),
            k=all_tris[:, 2].tolist(),
        )

        self.model = car.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.launch(
            _ancf_yup_to_zu,
            dim=_N_TIRES * n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device=device,
        )

        # ── MuJoCo solver (full DW dynamics) ─────────────────────────────────
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            iterations=50,
            ls_iterations=10,
            njmax=500,
            nconmax=128,
            update_data_interval=0,
        )
        self.solver.mjw_model.opt.graph_conditional = False
        print(f"[DW] MuJoCo opt.iterations={self.solver.mjw_model.opt.iterations}  njmax={self.solver.mjw_data.njmax}")

        # ── Map spindle names -> MuJoCo body indices ──────────────────────────
        btow = self.solver.mjc_body_to_newton.numpy()[0]  # world 0
        spindle_mj_list = []
        for name in _SPINDLE_NAMES:
            nidx = _find_body(self.model, name)
            m = np.where(btow == nidx)[0]
            assert len(m), (
                f"spindle '{name}' (newton={nidx}) not in MuJoCo body map. "
                f"body_labels={[l.split('/')[-1] for l in self.model.body_label]}"
            )
            spindle_mj_list.append(int(m[0]))
        self._spindle_mj_arr = wp.array(spindle_mj_list, dtype=wp.int32, device=device)
        print(f"[DW] spindle MuJoCo indices: {spindle_mj_list}")

        # ── Register per-wheel coupling in SolverANCFShellRigid ──────────────
        # Single-world layout: all 4 spindles in world 0 at their real MJCF positions.
        # world_idx=0 for all; lateral_offset=0 (spindle already at correct Y in xpos).
        for _w in range(_N_TIRES):
            bead_global_w = (bead_np + _w * n_nodes).astype(np.int32)
            self.ancf_solver.setup_wheel(
                tire_idx=_w,
                spindle_mj=spindle_mj_list[_w],
                bead_idx_np=bead_global_w,
                tare_fz=fz_tare,
                world_idx=0,  # single MuJoCo world
                lateral_offset=0.0,  # real spindle positions in xpos, no offset needed
                device=device,
            )
        # Sync coupling alpha from module constant to solver.
        # The DW FEDA spindles have huge Dirichlet bead reactions (k_eff≈2.9e8 N/m)
        # that must be scaled down to avoid exploding the vehicle dynamics.
        # SolverANCFShellRigid defaults to 1.0 — override to match original DW value.
        self.ancf_solver._coupling_alpha = _COUPLING_ALPHA
        print(
            f"[DW] m_tire={m_tire:.3f} kg  fz_tare={fz_tare:.2f} N"
            f"  coupling_alpha={_COUPLING_ALPHA}  alpha_m={alpha_m_damp:.1f} s^-1"
        )

        # ── GUI state ─────────────────────────────────────────────────────────
        self.steer_angle = float(getattr(args, "steer_angle", 0.0))
        self.wheel_speed = float(getattr(args, "wheel_speed", 0.0))
        self._target_wheel_speed = self.wheel_speed

        # ── Per-substep GPU arrays ────────────────────────────────────────────
        self._rim_phi = wp.zeros(_N_TIRES, dtype=float, device=device)
        self._xfrc_stg = wp.zeros(_N_TIRES, dtype=wp.spatial_vector, device=device)  # legacy display buffer

        # ── Bead ring visualization buffers ───────────────────────────────────
        N_r = n_bead_per_ring
        n_rings = 2 * sec_divs[0]
        seg_s_1 = np.array([i + k * N_r for k in range(n_rings) for i in range(N_r)], dtype=np.int32)
        seg_e_1 = np.array([(i + 1) % N_r + k * N_r for k in range(n_rings) for i in range(N_r)], dtype=np.int32)
        n_segs_1 = len(seg_s_1)
        seg_s_all = np.concatenate([seg_s_1 + e * n_bead for e in range(_N_TIRES)])
        seg_e_all = np.concatenate([seg_e_1 + e * n_bead for e in range(_N_TIRES)])
        n_segs = n_segs_1 * _N_TIRES

        self._bead_pos_zu = wp.zeros(_N_TIRES * n_bead, dtype=wp.vec3, device=device)
        self._spoke_start_zu = wp.zeros(_N_TIRES * n_bead, dtype=wp.vec3, device=device)
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=device)
        self._ring_line_s = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(n_segs, dtype=wp.vec3, device=device)
        self._n_ring_segs = n_segs

        # ── Graph capture ─────────────────────────────────────────────────────
        # Enable Dirichlet zeroing in _step_batched to prevent bead drift.
        self.ancf_solver._fix_dirichlet_in_batched = True
        self.ancf_solver.capture_graph(sim_dt)

        # Restore ANCF state consumed by capture_graph warmup.
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

        # Warm up kin step + prescribe so MuJoCo data structures are fully initialised.
        self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, sim_dt)
        self._prescribe_beads()

        # Seed visualization buffers.
        wp.launch(
            _ancf_yup_to_zu,
            dim=_N_TIRES * n_nodes,
            inputs=[self.ancf_solver.node_x, self.state_0.particle_q],
            device=device,
        )
        wp.launch(
            _gather_zu,
            dim=_N_TIRES * n_bead,
            inputs=[self.state_0.particle_q, self._bead_idx_all, self._bead_pos_zu],
            device=device,
        )
        wp.launch(
            _build_ring_lines,
            dim=n_segs,
            inputs=[self._bead_pos_zu, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=device,
        )
        wp.launch(
            _fill_spindle_positions_dw,
            dim=_N_TIRES * n_bead,
            inputs=[self.solver.xpos, self._spindle_mj_arr, self._spoke_start_zu, n_bead],
            device=device,
        )

        self._kin_graph = None
        self._dyn_graph = None
        self._substep_graph = None
        self._try_capture_mujoco_graphs()
        self._try_capture_substep_graph()

        if viewer is not None:
            viewer.set_model(self.model)
            viewer.set_camera(
                pos=wp.vec3(-12.0, -18.0, 9.0),
                pitch=-22.0,
                yaw=48.0,
            )

    # ── Graph capture ─────────────────────────────────────────────────────────

    def _try_capture_mujoco_graphs(self) -> None:
        """Capture step_kinematics and step_dynamics as standalone CUDA graphs.

        Pre-instantiates graph execs by launching once, so they are ready for
        use as child nodes in the combined substep graph without triggering
        CUDA error 900 (cudaGraphInstantiate during capture).
        """
        dev = "cuda:0"
        dt = self._sim_dt
        wp.synchronize_device(dev)

        try:
            wp.capture_begin(device=dev)
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            self._kin_graph = wp.capture_end(device=dev)
            print("[DW GRAPHS] Kinematics graph captured ✓")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._kin_graph = None
            print(f"[DW GRAPHS] Kinematics capture failed: {e!r}")

        wp.synchronize_device(dev)
        try:
            wp.capture_begin(device=dev)
            self.solver.step_dynamics(self.state_rigid)
            self._dyn_graph = wp.capture_end(device=dev)
            print("[DW GRAPHS] Dynamics graph captured ✓")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._dyn_graph = None
            print(f"[DW GRAPHS] Dynamics capture failed: {e!r}")

        wp.synchronize_device(dev)
        if self._kin_graph is not None:
            wp.capture_launch(self._kin_graph)
        if self._dyn_graph is not None:
            wp.capture_launch(self._dyn_graph)
        wp.synchronize_device(dev)
        print("[DW GRAPHS] Graph execs instantiated ✓")

    def _try_capture_substep_graph(self) -> None:
        """Capture one complete substep as a single CUDA graph (1 launch/frame).

        Uses ancf.step() (unrolled NR+PCG) instead of ancf.graph_step() because
        cudaGraphLaunch is not capturable (CUDA error 900 otherwise).
        Saves and restores all simulation state around the capture warmup.
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

        # Force ANCF graph exec instantiation outside any capture context.
        ancf.graph_step()
        wp.synchronize_device(dev)
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

        def _one_substep():
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            wp.launch(
                _fill_phi_from_joint_q_dw,
                dim=_N_TIRES,
                inputs=[self.state_0.joint_q, self._axle_q_dof_arr, self._rim_phi],
                device=dev,
            )
            self._prescribe_beads()
            if self._alpha_m_damp > 0.0:
                wp.launch(
                    _update_mass_damp,
                    dim=_N_TIRES * self._n_nodes,
                    inputs=[ancf.node_xd, ancf.lumped_mass_tiled, self._alpha_m_damp, ancf.node_f_ext_persistent],
                    device=dev,
                )
            ancf.step(None, None, None, None, dt)
            self._prescribe_beads()
            self._accumulate_wrenches()
            self.solver.step_dynamics(self.state_rigid)
            wp.copy(self.state_0.body_q, self.state_rigid.body_q)
            wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

        try:
            wp.capture_begin(device=dev)
            for _ in range(self._substeps):
                _one_substep()
            self._substep_graph = wp.capture_end(device=dev)
            print(f"[DW FRAME GRAPH] Captured {self._substeps} substeps — 1 launch/frame ✓")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._substep_graph = None
            print(f"[DW FRAME GRAPH] Capture failed ({e!r}) — falling back to loop mode")

        # Restore all state after warmup.
        wp.synchronize_device(dev)
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
        self.state_0.body_q.assign(s["s0_bq"])
        self.state_0.body_qd.assign(s["s0_bqd"])
        self.state_0.joint_q.assign(s["s0_jq"])
        self.state_0.joint_qd.assign(s["s0_jqd"])
        self.state_rigid.body_q.assign(s["sr_bq"])
        self.state_rigid.body_qd.assign(s["sr_bqd"])
        self.state_rigid.joint_q.assign(s["sr_jq"])
        self.state_rigid.joint_qd.assign(s["sr_jqd"])

    # ── Coupling helpers ───────────────────────────────────────────────────────

    def _prescribe_beads(self) -> None:
        wp.launch(
            _prescribe_beads_gpu_dw,
            dim=_N_TIRES * self._n_bead,
            inputs=[
                self.solver.xpos,
                self.solver.cvel,
                self._spindle_mj_arr,
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
                self._n_bead,
                self._n_nodes,
            ],
            device="cuda:0",
        )

    def _accumulate_wrenches(self) -> None:
        self.solver.xfrc_applied.zero_()
        # SolverANCFShellRigid.accumulate_wheel_wrenches() replaces the flat
        # _accum_bead_wrench_dw + _staging_to_xfrc_dw pair.  It loops over each
        # of the 4 wheels, accumulates bead reactions into per-wheel staging, and
        # writes to xfrc_applied[0, spindle_mj_arr[w]].  This is the bilateral
        # constraint insertion point: future upgrade replaces global_f_int with λ.
        self.ancf_solver.accumulate_wheel_wrenches(
            xfrc_applied=self.solver.xfrc_applied,
            xpos=self.solver.xpos,
            device="cuda:0",
        )

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self) -> None:
        """Run one frame (substeps steps).  Uses CUDA frame graph when available."""
        if self._substep_graph is not None:
            wp.capture_launch(self._substep_graph)
            wp.synchronize_device()
            return

        ancf = self.ancf_solver
        dev = "cuda:0"
        for _ in range(self._substeps):
            if self._kin_graph is not None:
                wp.capture_launch(self._kin_graph)
            else:
                self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
            wp.launch(
                _fill_phi_from_joint_q_dw,
                dim=_N_TIRES,
                inputs=[self.state_0.joint_q, self._axle_q_dof_arr, self._rim_phi],
                device=dev,
            )
            self._prescribe_beads()
            if self._alpha_m_damp > 0.0:
                wp.launch(
                    _update_mass_damp,
                    dim=_N_TIRES * self._n_nodes,
                    inputs=[ancf.node_xd, ancf.lumped_mass_tiled, self._alpha_m_damp, ancf.node_f_ext_persistent],
                    device=dev,
                )
            ancf.graph_step()
            self._prescribe_beads()
            self._accumulate_wrenches()
            if self._dyn_graph is not None:
                wp.capture_launch(self._dyn_graph)
            else:
                self.solver.step_dynamics(self.state_rigid)
            wp.copy(self.state_0.body_q, self.state_rigid.body_q)
            wp.copy(self.state_0.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_0.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_0.joint_qd, self.state_rigid.joint_qd)

    def _update_controls(self) -> None:
        """Write steering and throttle to control arrays (outside CUDA graph)."""
        jtp = self.control.joint_target_q.numpy().copy()
        jtp[self._steer_qd_dof] = self.steer_angle
        self.control.joint_target_q.assign(jtp)

        jtv = self.control.joint_target_qd.numpy().copy()
        for e in range(_N_TIRES):
            jtv[int(self._axle_qd_dofs[e])] = self.wheel_speed
        self.control.joint_target_qd.assign(jtv)

    def step(self) -> None:
        # CTIS pressure ramp (2000 Pa/frame toward GUI target)
        if self._pressure > 0.0:
            d = self._pressure_target - self._pressure_current
            ramp = min(abs(d), 2000.0) * (1.0 if d >= 0.0 else -1.0)
            self._pressure_current += ramp
            self.ancf_solver.set_cavity(
                [self._pressure_current] * _N_TIRES,
                [self._pressure] * _N_TIRES,
            )

        # Ramp wheel speed
        d = self._target_wheel_speed - self.wheel_speed
        if abs(d) <= _WHEEL_SPEED_RATE:
            self.wheel_speed = self._target_wheel_speed
        else:
            self.wheel_speed += _WHEEL_SPEED_RATE if d > 0.0 else -_WHEEL_SPEED_RATE

        self._update_controls()

        t0 = time.perf_counter()
        self.simulate()
        wp.synchronize_stream(wp.get_stream("cuda:0"))
        self._t_step += time.perf_counter() - t0
        self._frame += 1
        self._t += _FRAME_DT

        # Update ANCF visualization particles and bead ring / spoke buffers.
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

        if self._frame % self._diag_period == 0:
            _now = time.perf_counter()
            fps = self._diag_period / max(_now - self._t_wall, 1e-9)
            self._gui_fps = fps
            self._t_wall = _now
            self._print_diag(fps)
            self._t_step = self._t_render = 0.0

    def _print_diag(self, fps: float = 0.0) -> None:
        x_all = self.ancf_solver.node_x.numpy()
        # Read staging from per-wheel buffers in SolverANCFShellRigid
        stg_per_wheel = [self.ancf_solver._xfrc_stg_per_tire[w].numpy()[0] for w in range(_N_TIRES)]
        phi_all = self._rim_phi.numpy()
        xpos_all = self.solver.xpos.numpy()
        smj = self._spindle_mj_arr.numpy()
        nn = self._n_nodes
        rest_np = self._bead_rest_np

        print(f"\n[{self._frame:4d}] t={self._t:.2f}s  fps={fps:.1f}")
        labels = ["FL", "FR", "RL", "RR"]
        for e in range(_N_TIRES):
            x_e = x_all[e * nn : (e + 1) * nn]
            stg_e = stg_per_wheel[e]
            phi_e = float(phi_all[e])
            any_nan = bool(np.any(np.isnan(x_e)))
            fz = float(stg_e[4]) + self._fz_tare  # add tare back for display
            bead_x = x_e[self._bead_idx_np]
            pos_zu = xpos_all[0, int(smj[e])]
            hub = np.array([float(pos_zu[1]), float(pos_zu[2]), float(pos_zu[0])])
            cp, sp = math.cos(phi_e), math.sin(phi_e)
            expected = np.stack(
                [
                    hub[0] + rest_np[:, 0],
                    hub[1] + rest_np[:, 1] * cp - rest_np[:, 2] * sp,
                    hub[2] + rest_np[:, 1] * sp + rest_np[:, 2] * cp,
                ],
                axis=1,
            )
            drift_mm = float(np.max(np.linalg.norm(bead_x - expected, axis=1))) * 1e3
            self._gui_fz[e] = fz
            self._gui_drift[e] = drift_mm
            print(
                f"  {labels[e]}: {'NaN!' if any_nan else 'ok  '}"
                f"  hub=({hub[0]:.3f},{hub[1]:.3f},{hub[2]:.3f})"
                f"  drift={drift_mm:.2f}mm  Fz={fz:+.0f}N"
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
        fwd = self.wheel_speed * _R_OUTER
        ui.text(f"forward speed ~ {fwd:+.2f} m/s")
        if abs(self.steer_angle) > 1e-3:
            r = (2.0 * _WB_H) / math.tan(abs(self.steer_angle))
            ui.text(f"turn radius   ~ {r:.2f} m")
        else:
            ui.text("turn radius   ~ straight")

        if self._pressure > 0.0:
            ui.separator()
            ui.text("CTIS pressure")
            changed, val = ui.slider_float("nominal [Pa]", self._pressure_target, 0.0, self._pressure * 2.0)
            if changed:
                self._pressure_target = float(val)
            ui.text(f"  current {self._pressure_current:8.0f} Pa  ({self._pressure_current / 6894.76:.1f} psi)")

        ui.separator()
        ui.text("Live")
        ui.text(f"  fps   {self._gui_fps:6.1f}")
        labels = ["FL", "FR", "RL", "RR"]
        for e in range(_N_TIRES):
            ui.text(f"  {labels[e]}  Fz={self._gui_fz[e]:+.0f} N  drift={self._gui_drift[e]:.2f} mm")

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self) -> None:
        if self.viewer is None:
            return
        t0 = time.perf_counter()
        self.viewer.begin_frame(self._t)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, colors=(1.0, 0.45, 0.0))
        self.viewer.log_lines("bead_spokes", self._spoke_start_zu, self._bead_pos_zu, colors=(1.0, 0.90, 0.1))
        self.viewer.end_frame()
        self._t_render += time.perf_counter() - t0

    # ── Tests ──────────────────────────────────────────────────────────────────

    def test_post_step(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()[: self._n_nodes]
        if np.any(np.isnan(x_np)):
            raise AssertionError(f"NaN in ANCF node_x at frame {self._frame}, t={self._t:.3f}s")

    def test_final(self) -> None:
        x_np = self.ancf_solver.node_x.numpy()
        xd_np = self.ancf_solver.node_xd.numpy()
        assert not np.any(np.isnan(x_np)), "NaN in node_x at test_final"
        assert not np.any(np.isnan(xd_np)), "NaN in node_xd at test_final"
        assert not np.any(np.isinf(x_np)), "Inf in node_x at test_final"

        phi_all = self._rim_phi.numpy()
        stg_per_wheel = [self.ancf_solver._xfrc_stg_per_tire[w].numpy()[0] for w in range(_N_TIRES)]
        xpos_all = self.solver.xpos.numpy()
        smj = self._spindle_mj_arr.numpy()
        rest_np = self._bead_rest_np
        nn = self._n_nodes

        for e in range(_N_TIRES):
            x_e = x_np[e * nn : (e + 1) * nn]
            stg_e = stg_per_wheel[e]
            phi_e = float(phi_all[e])
            pos_zu = xpos_all[0, int(smj[e])]
            hub = np.array([float(pos_zu[1]), float(pos_zu[2]), float(pos_zu[0])])
            bead_x = x_e[self._bead_idx_np]
            cp, sp = math.cos(phi_e), math.sin(phi_e)
            expected = np.stack(
                [
                    hub[0] + rest_np[:, 0],
                    hub[1] + rest_np[:, 1] * cp - rest_np[:, 2] * sp,
                    hub[2] + rest_np[:, 1] * sp + rest_np[:, 2] * cp,
                ],
                axis=1,
            )
            max_drift = float(np.max(np.linalg.norm(bead_x - expected, axis=1)))
            assert max_drift < 1e-3, f"FAIL tire {e}: bead drift {max_drift * 1e3:.2f} mm > 1 mm"
            fz = abs(float(stg_e[4]))
            fz_exp = _M_RIGID * _GRAVITY
            rel = abs(fz - fz_exp) / max(fz_exp, 1.0)
            assert rel < 0.50, f"FAIL tire {e}: F_z={fz:.1f} N  expected~{fz_exp:.1f} N  err={rel * 100:.1f}% > 50%"

        fi_np = self.ancf_solver.global_f_int.numpy()
        assert np.all(np.isfinite(fi_np)), "Non-finite global_f_int at test_final"
        print(f"[PASS] frame={self._frame}  t={self._t:.1f}s  all 4 tires OK")

    # ── Parser ─────────────────────────────────────────────────────────────────

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--wheel-speed", type=float, default=0.0, help="Throttle angular speed [rad/s].  Default: 0."
        )
        parser.add_argument(
            "--steer-angle",
            type=float,
            default=0.0,
            help=f"Steering [rad].  Range ±{_MAX_STEER:.5f} (±27.5°).  Default: 0.",
        )
        parser.add_argument(
            "--substeps", type=int, default=_SIM_SUBSTEPS, help=f"Substeps per frame.  Default: {_SIM_SUBSTEPS}."
        )
        parser.add_argument(
            "--nr-iters",
            type=int,
            default=_NR_ITERS,
            help=f"Newton-Raphson iterations per substep.  Default: {_NR_ITERS}.",
        )
        parser.add_argument(
            "--pcg-iters", type=int, default=_PCG_ITERS, help=f"PCG iterations per NR step.  Default: {_PCG_ITERS}."
        )
        parser.add_argument("--kn", type=float, default=_KN, help=f"Contact normal stiffness [N/m].  Default: {_KN}.")
        parser.add_argument("--kd", type=float, default=_KD, help=f"Contact damping [N·s/m].  Default: {_KD}.")
        parser.add_argument("--mu", type=float, default=_MU, help=f"Friction coefficient.  Default: {_MU}.")
        parser.add_argument(
            "--shell-tires",
            type=str,
            default=None,
            help="JSON array of per-tire configs (same format as ancf_mujoco_tires). "
            "First entry used for all 4 tires. Overrides flat --e-tire etc. args.",
        )
        parser.add_argument(
            "--pressure", type=float, default=_PRESSURE, help=f"Gauge inflation pressure [Pa].  Default: {_PRESSURE}."
        )
        parser.add_argument(
            "--e-tire", type=float, default=_E_TIRE, help=f"Shell Young's modulus [Pa].  Default: {_E_TIRE:.1e}."
        )
        parser.add_argument("--nu-tire", type=float, default=_NU_TIRE, help=f"Poisson ratio.  Default: {_NU_TIRE}.")
        parser.add_argument(
            "--rho-tire", type=float, default=_RHO_TIRE, help=f"Density [kg/m^3].  Default: {_RHO_TIRE}."
        )
        parser.add_argument(
            "--h-shell", type=float, default=_H_SHELL, help=f"Shell thickness [m].  Default: {_H_SHELL}."
        )
        parser.add_argument(
            "--thickness-gp",
            type=int,
            default=3,
            choices=[3, 5],
            help="Through-thickness Gauss points (3=faster, 5=full).  Default: 3.",
        )
        parser.add_argument("--diag-period", type=int, default=5, help="Print diagnostics every N frames.  Default: 5.")
        parser.add_argument(
            "--fast-math", action="store_true", default=False, help="Enable Warp fast-math (~5-15%% speedup)."
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
