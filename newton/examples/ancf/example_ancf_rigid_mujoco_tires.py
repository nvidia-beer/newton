# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""N parallel ANCF FEM tires on MuJoCo rigid spindles — SolverANCFShellRigid variant.

Same as ancf_mujoco_tires but uses SolverANCFShellRigid, which exposes a
per-wheel wrench API as the insertion point for future bilateral rim-body
constraints (Chrono ChLinkNodeFrame style).  Same MJCF, mesh, and coupling
loop — only the solver class and _accumulate_wrenches differ.

Architecture
------------
  Rigid spindles - SolverMuJoCo (Z-up, MJCF ancf_single_tire.xml, N worlds)
  FEM tires x N  - SolverANCFShellRigid (Y-up, n_envs=N, default N=4)

Coupling (per substep):
  1. Sync rim angle from spin_y joint_q
  2. step_kinematics -> xpos on GPU
  3. Prescribe ANCF bead node positions from spindle xpos (rigid → soft)
  4. ANCF graph_step (HHT, NR, PCG)
  5. accumulate_wheel_wrenches() per-wheel → xfrc_applied on spindle (soft → rigid)
  6. step_dynamics

Coordinate systems
------------------
  MuJoCo Z-up : x_fwd, y_lat, z_up
  ANCF Y-up   : x_lat, y_up,  z_fwd  (axle along X, tread at Y=0)
  Z-up -> Y-up : (x,y,z) -> (y, z, x)
  Y-up -> Z-up : (x,y,z) -> (z, x, y)
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
    build_ancf_parabolic_mesh,
    build_ancf_tire_mesh,
    isotropic_ancf_material,
)
from newton._src.solvers.ancf_shell.solver_ancf_shell_rigid import SolverANCFShellRigid as SolverANCFShell

# ── Tire geometry (FEDA 335/65R22.5) ─────────────────────────────────────────
_R_OUTER = 0.499  # [m] tread crown radius
_R_INNER = 0.286  # [m] bead / rim seat radius
_WIDTH = 0.335  # [m] bead-to-bead width
_N_CIRC = 20  # circumferential divisions
_SEC_DIVS = (1, 2, 3)  # axial element divisions per half-section; 1 bead row per side (simpler than DW's 2)
_H_SHELL = 0.006  # [m] shell thickness

# ── Shell material defaults (overridden by --shell-tire JSON) ────────────────
_E_TIRE = 1.0e7  # [Pa]  10 MPa — stiffens sidewall 10× vs old 1 MPa default
_NU_TIRE = 0.45
_RHO_TIRE = 700.0  # [kg/m^3]
_ALPHA_D = 0.0  # Rayleigh stiffness damping — zeroed: bead nodes have node_xd=0
# but adjacent free crown nodes have non-zero velocity; off-diagonal K×v creates
# a spurious upward force that grows with hub speed → hub rockets.  HHT α=−0.2
# provides sufficient high-frequency numerical damping without Rayleigh damping.

_PRESSURE = 30_000.0  # [Pa] gauge inflation pressure

# ── FEDA masses — rim collapsed into spindle (DW topology) ───────────────────
_M_SPINDLE = 13.08  # [kg] FEDA_DoubleWishbone.cpp
_M_RIM = 18.80  # [kg] FEDA_Wheel.cpp
_M_RIGID = _M_SPINDLE + _M_RIM  # 31.88 kg

# ── Solver parameter defaults (overridden by CLI args) ───────────────────────
_SIM_SUBSTEPS = 20  # substeps=20: CFL=0.63<1 at E=10MPa; substeps=10 had CFL=1.27 → PCG ill-conditioned → NaN frame 5
_NR_ITERS = 16  # no GS loop; kn/K_eff=1.6%/iter at substeps=20
_PCG_ITERS = 25
_FRAME_DT = 1.0 / 60.0
_USE_GRAPH = True

# ── Coupling: tare + alpha ────────────────────────────────────────────────────
# Tare cancels tire self-weight from bead reactions so pre-contact xfrc≈0.
# alpha=1.0: at equilibrium 1.0*(stg[4]+tare)=M_rigid*g.
_M_TIRE = _RHO_TIRE * _H_SHELL * (2.0 * math.pi * _R_OUTER * _WIDTH + 2.0 * math.pi * (_R_OUTER**2 - _R_INNER**2))
_FZ_TARE = _M_TIRE * 9.81
_COUPLING_ALPHA = 1.0  # direct xfrc coupling does not need GS mass weighting
_B_SLIDE = 500.0  # [N·s/m] axle slide damping — stops spurious drift without RPM

# ── Rolling traction (Coulomb at spindle level) ───────────────────────────────
# FEM contact friction on crown nodes (~36N at kn=10k) is drowned out by the
# ~4000N elastic+pressure bead reactions.  We implement macro Coulomb traction
# directly at the coupling level, identical to vehicle dynamics models (Pacejka).
# F_traction = -mu_roll × F_normal × tanh(v_slip / v_reg)
# v_slip = v_hub_fwd - omega_axle × R_outer  (positive = braking slip)
# F_normal from staging wrench (already computed, zero cost).
_MU_ROLL = 0.9  # effective tire–road friction coefficient
_V_REG_ROLL = 0.05  # slip regularization velocity [m/s] (~5 cm/s dead-band)

# ── Vertical spindle load (simulates vehicle body weight) ─────────────────────
# SAFE RANGE depends on kn:
#   kn=10,000  N/m → max ~250 N  (δ_max≈25mm; beyond this shell inverts → NaN)
#   kn=300,000 N/m → max ~4000 N (δ=13mm at HMMWV quarter-car load)
# Default 0: no load applied; tune via --f-load or GUI slider.
# F_load acts downward (−Z in MuJoCo Z-up) on each spindle independently.
_F_LOAD = 0.0  # [N] per-spindle downward load (0 = tire weight only)

# ── Ground contact defaults (overridden by CLI args) ─────────────────────────
# With E=10 MPa, SEC_DIVS=(1,2,3): n_nodes=260, m_node=8.82/260=33.9g.
# HHT PREDICTOR STABILITY: kn < m_node / ((0.5−β) × dt²).
#   At substeps=20: dt=1/1200 → threshold = 0.0339/(0.14×(1/1200)²) = 347,600 N/m.
#   kn=10,000 << 347,600 → predictor always below ground → no chattering.
#
# CONTACT MODE: ω_contact = √(kn/m_node) = √(10k/0.0339) = 543 rad/s.
#   substeps=20: ω·dt = 543/1200 = 0.45 → HHT provides very strong damping ✓.
#   (substeps=10: ω·dt = 0.91, marginally stable; CFL=1.27 at E=10 MPa → NaN frame 5.)
#
# CFL at E=10 MPa: c=√(E/ρ)=119.5 m/s, l_circ=2π×0.499/20=0.157 m.
#   substeps=20: CFL = 119.5/(1200×0.157) = 0.63 < 1 ✓.
#   substeps=10: CFL = 1.27 → K_inertia/K_elastic ≈ 33.9k/480k → κ≈15 → PCG ill-conditioned.
#   substeps=20: K_inertia = 135.6 kN/m → κ≈4.5 → PCG=25 trivially converges.
#
# KD EXPLICIT STABILITY: kd_max = 2·m_node/dt = 2×0.0339×1200 = 81.4 N·s/m.
#   kd=20: kd·dt/m = 20/(1200×0.0339) = 0.49 < 2 ✓ (better margin than substeps=10).
_KN = 10_000.0  # [N/m] — ω·dt=0.91 at substeps=10 (well-damped); DW uses 20k at substeps=100
_KD = 20.0  # [N·s/m] — kd·dt/m=0.98 < 2 (explicit stability) ✓
_MU = 0.9

# ── Test rig geometry ─────────────────────────────────────────────────────────
_DROP_CLEARANCE = 0.000  # [m] quasi-static start (tread at ground, no impact)
_SIM_DURATION = 5.0  # [s]
_GRAVITY = 9.81  # [m/s^2]
_RPM_RATE = 4.0  # [rpm/frame] max RPM change per step() call

# ── Bead ring indices (default values — recomputed as instance vars in __init__) ─
# SEC_DIVS (1,2,3): 2*(1+2+3)=12 axial divisions -> 13 rings, 260 nodes.
# SEC_DIVS[0]=1 bead row pinned per side:
#   Left : j=0  → [0,   N_CIRC)
#   Right: j=12 → [12×N_CIRC, 13×N_CIRC)
_N_AX_DIVS = 2 * sum(_SEC_DIVS)  # 12
_N_BEAD_PER_RING = _N_CIRC  # 20
_N_BEAD = 2 * _SEC_DIVS[0] * _N_BEAD_PER_RING  # 40

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "ancf_single_tire.xml")


# ── Warp kernels ──────────────────────────────────────────────────────────────


@wp.kernel
def _update_mass_damp(
    node_xd: wp.array[wp.vec3],
    lm_flat: wp.array[float],  # lumped_mass [n_nodes*6]; lm[6*i]=pos-DOF mass
    alpha_m: float,  # mass-proportional Rayleigh coeff [s^-1]
    f_pers: wp.array[wp.vec3],  # node_f_ext_persistent (written, not added)
):
    """Write mass-proportional damping into node_f_ext_persistent each substep.

    f_damp[i] = -alpha_m * m_node[i] * xd[i]

    Bead nodes have xd=0 (prescribed in _prescribe_beads_gpu), so their
    damping force is automatically zero — no spurious hub force.  Free crown
    and sidewall nodes are damped, killing low-frequency free oscillations
    that HHT barely attenuates (ωΔt<<1 at E=10 MPa, substeps=20).

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
    """dim=N. Kinematic rim angle integration: phi += omega × dt.
    Identical to ancf_rim_shell._advance_phi_and_z_batched (phi part).
    No MuJoCo spin_y DOF needed — spin is self-tracked."""
    e = wp.tid()
    rim_phi[e] = rim_phi[e] + rim_omega[e] * dt


@wp.kernel
def _set_rolling_ctrl(
    rim_omega: wp.array[float],  # (N,) self-tracked, not from cvel
    joint_target_vel: wp.array[float],
    slide_x_qd_dof: int,
    n_qd_per_world: int,
    R_outer: float,
):
    """dim=N. No-slip rolling: lin_motor target = rim_omega × R_outer.
    Reads from self-tracked rim_omega (not MuJoCo cvel) — exact kinematic prescription.
    Matches Chrono ChLinkMotorLinearSpeed."""
    env = wp.tid()
    joint_target_vel[slide_x_qd_dof + env * n_qd_per_world] = rim_omega[env] * R_outer


@wp.kernel
def _fill_phi_from_joint_q(
    joint_q: wp.array[float],
    axle_dof: int,
    n_q_per_world: int,
    rim_phi: wp.array[float],  # shape (N,)
):
    env = wp.tid()
    rim_phi[env] = joint_q[axle_dof + env * n_q_per_world]


@wp.kernel
def _prescribe_slide_x_vel(
    joint_qd: wp.array[float],
    rim_omega: wp.array[float],  # (N,) rad/s
    slide_x_qd_dof: int,
    n_qd_per_world: int,
    r_outer: float,
):
    """dim=N.  Write joint_qd[slide_x] = omega × R BEFORE step_kinematics.

    step_kinematics then integrates joint_q[slide_x] += (omega × R) × dt,
    giving exact no-slip rolling — identical coupling to ancf_rim_shell which
    advances hub_z kinematically.  The lin_motor (kv=5000) stays in MJCF as a
    soft-constraint backup but contributes near-zero force in steady state.
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
    lateral_spacing: float,
    R_outer: float,
    vel_predict_dt: float,  # > 0: linearly extrapolate hub pos by vel × dt (reduces coupling lag O(dt²))
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

    hub_x = pos_zu[1] + float(env) * lateral_spacing
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
    # Bead velocity = rotational + chassis forward velocity (from slide_x DOF)
    node_xd[global_idx] = wp.vec3(0.0, -omega * z_new, omega * y_loc + v_fwd)
    node_xdd[global_idx] = wp.vec3(0.0, 0.0, 0.0)

    d_y = d0[1] * cp - d0[2] * sp
    d_z = d0[1] * sp + d0[2] * cp
    node_D[global_idx] = wp.vec3(d0[0], d_y, d_z)
    node_Dd[global_idx] = wp.vec3(0.0, -omega * d_z, omega * d_y)
    node_Ddd[global_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _accum_bead_wrench(
    global_f_int: wp.array[float],
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    xpos: wp.array2d[wp.vec3],
    spindle_mj: int,
    staging: wp.array[wp.spatial_vector],  # shape (N,)
    n_bead: int,
    n_nodes: int,
    lateral_spacing: float,  # must match _prescribe_beads_gpu
):
    """Accumulate bead constraint reactions into per-env staging wrench (Newton 3rd law).

    Dim = N * n_bead.  tid = env * n_bead + local_bead_index.
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
    pos_zu = xpos[env, spindle_mj]
    hub_ancf = wp.vec3(pos_zu[1] + float(env) * lateral_spacing, pos_zu[2], pos_zu[0])
    r = node_x[global_idx] - hub_ancf
    tau = wp.cross(r, f)
    wp.atomic_add(staging, env, wp.spatial_vector(tau[0], tau[1], tau[2], fx, fy, fz))


@wp.kernel
def _zero_xfrc_forward(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
):
    """Zero the forward (MuJoCo X) force component of xfrc_applied.

    slide_x is kinematically prescribed — ANCF forward forces must NOT reach
    step_dynamics.  At 300 RPM the forward force can be O(10 kN), creating a
    velocity impulse large enough to overflow joint_qd and produce NaN.
    slide_z (vertical) is free and remains two-way; all torques are already
    zeroed in _staging_to_xfrc_wheel.
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
def _xpos_shift_lateral(
    xpos: wp.array2d[wp.vec3],  # (nworld, nbody_mj)
    nbody: int,
    spacing: float,  # positive=add, negative=restore
):
    """Shift each world's body positions by world_index * spacing in Y (lateral)."""
    tid = wp.tid()
    world = tid // nbody
    body = tid % nbody
    p = xpos[world, body]
    xpos[world, body] = wp.vec3(p[0], p[1] + float(world) * spacing, p[2])


@wp.kernel
def _gather_zu(
    src: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    dst: wp.array[wp.vec3],
):
    """Gather indexed positions that are already in Z-up — no coordinate conversion."""
    i = wp.tid()
    dst[i] = src[indices[i]]


@wp.kernel
def _gather_yup_to_zu_env(
    src: wp.array[wp.vec3],  # (N*n_nodes,)
    indices: wp.array[wp.int32],  # (n_bead,) local bead indices
    dst: wp.array[wp.vec3],  # (N*n_bead,)
    n_bead: int,
    n_nodes: int,
):
    """Batched N-env version: gather bead nodes for all envs."""
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_node = env * n_nodes + indices[i]
    p = src[global_node]
    dst[tid] = wp.vec3(p[2], p[0], p[1])


@wp.kernel
def _offset_bead_radial_env(
    bead_pos_zu: wp.array[wp.vec3],  # (N*n_bead,) in-place
    xpos: wp.array2d[wp.vec3],
    spindle_mj: int,
    n_bead: int,
    delta: float,  # outward offset [m]
):
    """Push each bead node radially outward from its env's spindle axis (in Z-up XZ plane)."""
    tid = wp.tid()
    env = tid // n_bead
    hub = xpos[env, spindle_mj]  # Z-up hub centre
    p = bead_pos_zu[tid]
    rx = p[0] - hub[0]  # Z-up X offset from hub
    rz = p[2] - hub[2]  # Z-up Z (height) offset from hub
    n = wp.sqrt(rx * rx + rz * rz)
    if n > 1e-6:
        bead_pos_zu[tid] = wp.vec3(
            p[0] + delta * rx / n,
            p[1],
            p[2] + delta * rz / n,
        )


@wp.kernel
def _fill_spindle_positions_env(
    body_q: wp.array[wp.transform],  # state_0.body_q (Z-up transforms)
    spindle_idx: int,  # world-0 Newton body index
    n_bodies_world: int,  # bodies per world
    out: wp.array[wp.vec3],  # (N*n_bead,)
    n_bead: int,
    lateral_spacing: float,  # Y offset per env in Z-up (= ANCF X)
):
    """Fill spoke-start positions from body_q + per-env lateral display offset.
    MuJoCo spindles share Y=0; add lateral_spacing so spokes point to correct tire.
    """
    tid = wp.tid()
    env = tid // n_bead
    idx = env * n_bodies_world + spindle_idx
    p = wp.transform_get_translation(body_q[idx])
    out[tid] = wp.vec3(p[0], p[1] + float(env) * lateral_spacing, p[2])


@wp.kernel
def _offset_bead_radial(
    bead_pos_zu: wp.array[wp.vec3],
    hub_z: float,
    delta: float,
):
    """Push each bead node radially outward from hub axis (XZ plane in Z-up) by delta."""
    i = wp.tid()
    p = bead_pos_zu[i]
    rx = p[0]
    rz = p[2] - hub_z
    n = wp.sqrt(rx * rx + rz * rz)
    if n > 1e-6:
        bead_pos_zu[i] = wp.vec3(p[0] + delta * rx / n, p[1], p[2] + delta * rz / n)


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
def _apply_slide_damping(
    cvel: wp.array2d[wp.spatial_vector],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    b_slide: float,
):
    """Axle slide damping: F_damp = -b_slide * v_fwd on each spindle.

    Models rolling resistance + bearing friction.  Without this, the spindle's
    slide_x DOF has zero damping, so any perturbation (inflation pressure impulse,
    contact asymmetry) causes indefinite drift.  The ground contact friction on
    free FEM nodes (~36 N at kn=10k) is dwarfed by bead elastic reactions
    (~4000 N from pressure) and never reaches the spindle as a useful friction signal.

    xfrc layout (mujoco_warp FORCE FIRST): [0]=fx [1]=fy [2]=fz [3]=tx [4]=ty [5]=tz
    MuJoCo Z-up: vx=cvel[3]=forward, vz=cvel[5]=up.
    """
    env = wp.tid()
    sv = cvel[env, spindle_mj]
    v_fwd = sv[3]  # forward (MuJoCo x)
    cur = xfrc_applied[env, spindle_mj]
    xfrc_applied[env, spindle_mj] = wp.spatial_vector(
        cur[0] - b_slide * v_fwd,
        cur[1],
        cur[2],
        cur[3],
        cur[4],
        cur[5],
    )


@wp.kernel
def _apply_vertical_load(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    f_load: float,  # downward force per spindle [N], positive = down
):
    """Apply constant vertical (downward) preload to each spindle.

    Simulates vehicle body weight pressing the tyre onto the ground.
    F_friction_max = mu * (spindle_weight + f_load) — traction scales with load.

    SAFE RANGE: f_load < kn * delta_safe where delta_safe ≈ 0.025 m.
    At kn=10k: max ~250 N.   At kn=300k: max ~4000 N (HMMWV quarter-car).

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
def _apply_rolling_traction(
    cvel: wp.array2d[wp.spatial_vector],
    staging: wp.array[wp.spatial_vector],  # raw from _accum_bead_wrench (N,)
    xfrc_applied: wp.array2d[wp.spatial_vector],
    spindle_mj: int,
    r_outer: float,
    mu_roll: float,
    v_reg: float,  # slip regularisation [m/s]
    tare_fz: float,  # tire weight tare [N] — same as _staging_to_xfrc
):
    """Macro Coulomb rolling traction applied directly at spindle level.

    The FEM ground-contact friction on free crown nodes (~36 N at kn=10k) is
    swamped by the ~4000 N elastic+pressure bead reactions and does not
    propagate to the spindle.  This kernel implements the same physics at the
    rigid-body coupling boundary — identical in concept to Pacejka/Dugoff
    tyre models used in vehicle dynamics.

    F_normal = staging_Fz + tare (positive only when tyre is in contact).
    v_slip   = v_hub_fwd  – omega_axle × R_outer  (positive → braking/skid).
    F_trac   = –mu_roll × F_normal × tanh(v_slip / v_reg)  (opposes slip).

    xfrc layout (mujoco_warp FORCE FIRST): [0]=fx [1]=fy [2]=fz [3]=tx [4]=ty [5]=tz.
    MuJoCo Z-up: ang_y = cvel[1], vx = cvel[3] = forward.
    staging layout (ANCF Y-up → accumulation): [4] = Fz (vertical ANCF = MuJoCo Z).
    """
    env = wp.tid()
    sv = cvel[env, spindle_mj]
    omega = sv[1]  # axle spin [rad/s]
    v_fwd = sv[3]  # hub forward velocity [m/s]

    # Contact normal force (positive = tyre pressing on ground).
    # staging[4] is the raw Lagrange-multiplier vertical force before tare correction.
    F_norm = staging[env][4] + tare_fz
    if F_norm <= 0.0:
        return  # tyre airborne — no traction

    v_slip = v_fwd - omega * r_outer  # positive → hub outruns rotation
    F_trac = -mu_roll * F_norm * wp.tanh(v_slip / v_reg)

    cur = xfrc_applied[env, spindle_mj]
    xfrc_applied[env, spindle_mj] = wp.spatial_vector(
        cur[0] + F_trac,
        cur[1],
        cur[2],
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
    """GPU-only contact visualization — no D→H sync, zero FPS impact.

    One thread per node.  In-contact nodes (pen > 0) emit a spike above the
    ground plane whose height = pen * vis_scale, making sub-mm penetrations
    visible.  Non-contact nodes emit a zero-length segment (invisible).

    ANCF Y-up → Z-up: (x, y, z) → (z, x, y)
    Ground in Z-up = z-plane at z = ground_y (= 0).
    Spike goes upward in Z-up (+z direction).
    """
    i = wp.tid()
    p = node_x[i]  # ANCF Y-up
    pen = ground_y - p[1]  # positive = inside ground

    # Convert node position to Z-up: ANCF (x,y,z) → Z-up (z,x,y)
    base = wp.vec3(p[2], p[0], ground_y)  # clamp to ground surface for spike base

    if pen > 0.0:
        line_starts[i] = base
        line_ends[i] = wp.vec3(base[0], base[1], base[2] + pen * vis_scale)
    else:
        line_starts[i] = base
        line_ends[i] = base  # zero-length → invisible


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
    """Single ANCF FEM tire on MuJoCo rigid spindle — intermediate coupling test."""

    def __init__(self, viewer=None, args=None):
        device = "cuda:0"
        if getattr(args, "fast_math", False):
            wp.config.fast_math = True  # ~5-15% on stiffness; risks NaN with E<2MPa
        wp.init()

        self._frame = 0
        self._t = 0.0
        self.viewer = viewer
        self._diag_period = int(getattr(args, "diag_period", 5))
        self._debug_rpm = bool(getattr(args, "debug_rpm", False))
        self._t_wall = time.perf_counter()
        self._t_step = 0.0
        self._t_render = 0.0
        self._t_kin = 0.0
        self._t_ancf = 0.0
        self._t_dyn = 0.0
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
        drop_clearance = float(getattr(args, "drop_clearance", _DROP_CLEARANCE))

        # ── Tire configs from --shell-tires JSON list; n_envs = len(list) ───────
        # Each entry in shell-tires defines one parallel environment (tire).
        # All tires share the same mesh geometry (first entry drives geometry);
        # per-env pressure can differ but mesh/material is uniform for batching.
        tires_raw = getattr(args, "shell_tires", None) or getattr(args, "shell_tire", None)
        if isinstance(tires_raw, str):
            parsed = json.loads(tires_raw)
            tire_cfgs = parsed if isinstance(parsed, list) else [parsed]
        elif isinstance(tires_raw, list):
            tire_cfgs = tires_raw if tires_raw else [{}]
        elif isinstance(tires_raw, dict):
            tire_cfgs = [tires_raw]
        else:
            tire_cfgs = [{}]

        # n_envs from number of tire entries; override with --n-envs if provided
        n_envs_from_cfg = len(tire_cfgs)
        n_envs = int(getattr(args, "n_envs", n_envs_from_cfg))
        # Pad/trim tire_cfgs to match n_envs
        while len(tire_cfgs) < n_envs:
            tire_cfgs.append(tire_cfgs[-1])
        tire_cfgs = tire_cfgs[:n_envs]

        tire_cfg = tire_cfgs[0]  # geometry/material from first entry (shared across envs)
        pressures = [float(c.get("pressure", _PRESSURE)) for c in tire_cfgs]

        r_outer = float(tire_cfg.get("r-outer", _R_OUTER))
        self._r_outer = r_outer  # needed by _prescribe_beads and _advance_hub_z
        r_inner = float(tire_cfg.get("r-inner", _R_INNER))
        width = float(tire_cfg.get("width", _WIDTH))
        e_tire = float(tire_cfg.get("E", _E_TIRE))
        nu_tire = float(tire_cfg.get("nu", _NU_TIRE))
        rho_tire = float(tire_cfg.get("rho", _RHO_TIRE))
        h_shell = float(tire_cfg.get("thickness", _H_SHELL))
        alpha_d = float(tire_cfg.get("alpha-damp", _ALPHA_D))
        n_circ = int(tire_cfg.get("n-circ", _N_CIRC))
        sec_divs = tuple(int(x) for x in tire_cfg.get("sec-divs", list(_SEC_DIVS)))
        pressure = pressures[0]
        mesh_type = str(tire_cfg.get("mesh-type", "polaris"))
        n_ax = int(tire_cfg.get("n-ax", 4))

        drop_h = r_outer + drop_clearance
        env_spacing = float(getattr(args, "env_spacing", 1.2))  # [m] lateral gap between tires (default)
        # Per-tire positions from config (ANCF X = lateral); fallback to uniform spacing.
        tire_positions = [float(tire_cfgs[e].get("position", [e * env_spacing, 0.0, 0.0])[0]) for e in range(n_envs)]
        drop_heights = [drop_h] * n_envs  # all tires at same height; spaced laterally

        kn = float(getattr(args, "kn", _KN))
        kd = float(getattr(args, "kd", _KD))
        mu = float(getattr(args, "mu", _MU))
        nr_iters = int(getattr(args, "nr_iters", _NR_ITERS))
        pcg_iters = int(getattr(args, "pcg_iters", _PCG_ITERS))
        substeps = int(getattr(args, "substeps", _SIM_SUBSTEPS))
        thickness_gp = int(tire_cfg.get("thickness-gp", getattr(args, "thickness_gp", 3)))

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
            n_bead = 2 * n_circ  # one ring per side at j=0 and j=n_ax
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
        d0_np_tiled = np.tile(d0_np, (n_envs, 1)).astype(np.float32)

        m_tire = rho_tire * h_shell * (2.0 * math.pi * r_outer * width + 2.0 * math.pi * (r_outer**2 - r_inner**2))
        fz_tare = m_tire * 9.81

        # ── Mass-proportional Rayleigh damping (auto-derived) ─────────────────
        _l_sw = math.sqrt((r_outer - r_inner) ** 2 + (width / 2.0) ** 2)
        _k_sw = e_tire * h_shell * 2.0 * math.pi * r_outer / _l_sw
        _m_free = m_tire * (1.0 - n_bead / n_nodes)
        _omega_n = math.sqrt(_k_sw / max(_m_free, 1e-9))
        alpha_m_damp = float(getattr(args, "alpha_m_damp", 2.0 * 0.10 * _omega_n))

        self._b_slide = float(getattr(args, "b_slide", _B_SLIDE))
        self._mu_roll = float(getattr(args, "mu_roll", _MU_ROLL))
        self._v_reg_roll = float(getattr(args, "v_reg_roll", _V_REG_ROLL))
        _f_load_raw = float(getattr(args, "f_load", _F_LOAD))
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
        self._sec_divs = sec_divs
        self._n_ax_divs = n_ax_divs
        self._n_bead = n_bead
        self._fz_tare = fz_tare
        self._m_tire = m_tire
        self._alpha_m_damp = alpha_m_damp
        self._m_rigid = float(getattr(args, "m_rigid", _M_RIGID))

        # ── Bead ring node indices ────────────────────────────────────────────
        # parabolic: 1 ring per side (j=0 left, j=n_ax right)
        # polaris:   sec_divs[0] rows per side
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
        # Tiled bead indices into the N*n_nodes particle array (Z-up, all envs).
        # particle_q[e*n_nodes + bead_np[i]] = env e's bead i in Z-up (already converted).
        bead_idx_all_np = np.concatenate([bead_np + e * n_nodes for e in range(n_envs)])
        self._bead_idx_all = wp.array(bead_idx_all_np.astype(np.int32), device=device)
        self._bead_rest = wp.array(x0_np[bead_np].astype(np.float32), dtype=wp.vec3, device=device)
        self._bead_idx_np = self._bead_idx.numpy()  # constant — cache to avoid D→H each diag
        self._bead_rest_np = self._bead_rest.numpy()  # constant — cache to avoid D→H each diag
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
            n_envs=n_envs,
            thickness_gp=thickness_gp,
        )

        # Tile initial node positions for N envs, spacing them 2 m apart in ANCF X (= MuJoCo Y = lateral).
        # Place each ANCF tire at its config position (ANCF X = lateral in Z-up Y).
        # MuJoCo spindles remain at Y=0; lateral_spacing in coupling kernels provides
        # the per-env hub offset so physics coupling is correct.
        x_spacing = tire_positions[-1] / max(n_envs - 1, 1) if n_envs > 1 else 0.0  # uniform fallback
        world_x = np.concatenate(
            [x0_np + np.array([tire_positions[e], drop_heights[e], 0.0], dtype=np.float32) for e in range(n_envs)],
            axis=0,
        )  # shape (N*n_nodes, 3)
        self.ancf_solver.node_x.assign(world_x)
        # _step_batched requires GLOBAL flat indices (env * n_nodes + local_idx).
        # Passing only local indices (bead_np) zeros only env-0 beads; envs 1-3
        # accumulate unconstrained NR corrections → bead drift → explosion.
        bead_global_np = np.concatenate([bead_np + e * n_nodes for e in range(n_envs)])
        self.ancf_solver.set_dirichlet_nodes(bead_global_np)
        self._build_pressures = pressures
        self._pressure_targets = list(pressures)
        self._pressure_currents = list(pressures)
        self._build_pressure = pressures[0]  # kept for GUI compat
        self._pressure_target = pressures[0]
        self._pressure_current = pressures[0]
        self.ancf_solver.set_cavity(pressures, pressures)

        # ── MuJoCo rigid-body builder (Z-up) ─────────────────────────────────
        _mjcf_path = getattr(args, "mjcf", None)
        if _mjcf_path is None:
            _mjcf_path = _ASSETS
        elif not os.path.isabs(_mjcf_path):
            _mjcf_path = os.path.join(os.path.dirname(_ASSETS), _mjcf_path)

        # Build a single-world template to extract per-world DOF strides.
        car_template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car_template)
        car_template.add_mjcf(_mjcf_path, up_axis="Z")

        slide_z_q_dof_local = _find_dof_q(car_template, "slide_z")
        slide_x_qd_dof_local = _find_dof(car_template, "slide_x")  # for lin_motor
        # spin_y removed from MJCF — spin tracked in rim_phi/rim_omega GPU arrays
        n_q_per_world = int(car_template.joint_coord_count)
        n_qd_per_world = int(car_template.joint_dof_count)
        print(
            f"[STM] slide_z q_dof={slide_z_q_dof_local}"
            f"  slide_x qd_dof={slide_x_qd_dof_local}"
            f"  n_q_per_world={n_q_per_world}"
            f"  n_envs={n_envs}"
            f"  (spin_y removed — kinematic via rim_omega_wp)"
        )

        # Build the multi-world MuJoCo model (N identical spindle rigs).
        # Each world is offset in MuJoCo Y (= ANCF X) by x_spacing so that
        # xpos[env, spindle_mj] matches the ANCF tire's lateral position.
        car = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(car)
        for _e in range(n_envs):
            car.add_world(car_template)  # no xform: fwd_position ignores body_q init

        self._slide_z_q_dof = slide_z_q_dof_local
        self._slide_x_qd_dof = slide_x_qd_dof_local  # lin_motor: no-slip rolling
        self._n_q_per_world = n_q_per_world
        self._n_qd_per_world = n_qd_per_world

        # ── RPM control via MuJoCo velocity actuator ─────────────────────────
        # The MJCF has <velocity name="spin_motor" joint="spin_y" kv="500"/>.
        # Setting ctrl[env * n_ctrl_per_world + spin_ctrl_local] = omega [rad/s]
        # lets MuJoCo compute the actual driving torque = kv * (omega - actual_omega).
        # This is physically correct: the motor can stall under load, and free-rolling
        # (ctrl=0) emerges naturally from friction forces alone.
        default_rpm = float(getattr(args, "rpm", 0.0))
        self._target_rpm = [float(c.get("rpm", default_rpm)) for c in tire_cfgs]
        self._current_rpm = list(self._target_rpm)
        # RPM → rim_omega_wp[e] = current_rpm × 2π/60 (ramped in step())

        # ── Rendering model ───────────────────────────────────────────────────
        # Convert all N*n_nodes positions from ANCF Y-up to Z-up for the viewer.
        world_x_zu_all = []
        for _e in range(n_envs):
            wx_e = world_x[_e * n_nodes : (_e + 1) * n_nodes]
            wx_zu = np.stack([wx_e[:, 2], wx_e[:, 0], wx_e[:, 1]], axis=1).astype(np.float32)
            world_x_zu_all.append(wx_zu)
        world_x_zu = np.concatenate(world_x_zu_all, axis=0)  # (N*n_nodes, 3)

        # Add particles DIRECTLY to car (preserves N-world structure).
        # Ground plane comes from the MJCF (<geom type="plane"> in ancf_single_tire.xml).
        # builder.add_world(car) would FLATTEN N worlds → 1, making
        # SolverMuJoCo see nworld=1 and xpos[1..N-1, spindle_mj] unavailable.
        car.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x_zu],
            vel=[(0.0, 0.0, 0.0)] * (n_nodes * n_envs),
            mass=[0.0] * (n_nodes * n_envs),
            radius=[0.001] * (n_nodes * n_envs),
        )
        en_np = self.ancf_model.elem_nodes.numpy()
        tris = np.empty((len(en_np) * 2, 3), dtype=np.int32)
        tris[0::2] = en_np[:, [0, 1, 2]]
        tris[1::2] = en_np[:, [0, 2, 3]]
        all_tris = np.concatenate([tris + _e * n_nodes for _e in range(n_envs)], axis=0)
        car.add_triangles(
            i=all_tris[:, 0].tolist(),
            j=all_tris[:, 1].tolist(),
            k=all_tris[:, 2].tolist(),
        )
        self.model = car.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()

        # XML carrier at pos="0 0 0"; joint_q[slide_z] sets hub height.
        # Must sync both model.joint_q and state_0.joint_q so step_kinematics
        # reads the correct spindle position from the first substep.
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
        print(f"[STM] spindle newton={self._spindle_newton_idx}  mj={self._spindle_mj}")
        print(
            f"[STM] m_tire={self._m_tire:.3f} kg  fz_tare={self._fz_tare:.2f} N"
            f"  coupling_alpha={_COUPLING_ALPHA:.4f}"
            f"  alpha_m_damp={self._alpha_m_damp:.1f} s^-1"
        )

        # ── Register per-wheel coupling in SolverANCFShellRigid ──────────────
        # N-parallel-world layout: env e lives in MuJoCo world e (world_idx=e).
        # Spindle body index is the same in every world (same MJCF replicated).
        # lateral_offset = e * lateral_spacing so the moment arm is computed from
        # the correct ANCF hub centre (all MuJoCo spindles sit at Y=0 but ANCF
        # tires are spaced by lateral_spacing in ANCF X).
        _lat_spacing = tire_positions[1] if n_envs > 1 else 0.0  # same as _lateral_spacing below
        for _e in range(n_envs):
            bead_global_e = (bead_np + _e * n_nodes).astype(np.int32)
            self.ancf_solver.setup_wheel(
                tire_idx=_e,
                spindle_mj=self._spindle_mj,
                bead_idx_np=bead_global_e,
                tare_fz=fz_tare,
                world_idx=_e,  # parallel-world: env == world
                lateral_offset=float(_e) * _lat_spacing,
                device=device,
            )

        # ── GPU scalars / per-env arrays ─────────────────────────────────────
        self._rim_phi = wp.zeros(n_envs, dtype=float, device=device)  # kinematic, self-tracked
        self._rim_omega_wp = wp.zeros(n_envs, dtype=float, device=device)  # kinematic spin (not from MuJoCo cvel)
        # Host mirror of _rim_omega_wp; kept in sync by step() so the RPM ramp
        # never reads back from the device.  Starts zeroed to match the array above.
        self._rim_omega_np = np.zeros(n_envs, dtype=np.float32)
        self._xfrc_stg = wp.zeros(n_envs, dtype=wp.spatial_vector, device=device)
        self._n_nodes = n_nodes
        self._n_envs = n_envs
        self._tire_positions = tire_positions  # per-env ANCF X offsets from config
        self._lateral_spacing = tire_positions[1] if n_envs > 1 else 0.0  # step for coupling
        # Use car_template body count (before ground/particles added) to get
        # the exact per-world stride — model.body_count includes extra bodies.
        self._n_bodies_per_world = car_template.body_count
        print(
            f"[DBG] model.body_count={self.model.body_count}"
            f"  template.body_count={car_template.body_count}"
            f"  n_bodies_per_world={self._n_bodies_per_world}"
            f"  spindle_newton_idx={self._spindle_newton_idx}"
            f"  n_envs={n_envs}"
        )
        # Print initial body positions for all worlds to verify alignment
        wp.synchronize_device(device)
        bq_init = self.state_0.body_q.numpy()
        print(f"[DBG] body_q shape={bq_init.shape}  dtype={bq_init.dtype}")
        for _e in range(n_envs):
            sidx = self._spindle_newton_idx + _e * self._n_bodies_per_world
            if sidx < len(bq_init):
                sp = bq_init[sidx]
                print(
                    f"[DBG] env{_e} spindle body_q[{sidx}] = ({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})"
                    f"  ANCF_tire_X={_e * x_spacing:.3f}"
                )

        # ── Bead ring visualization buffers (N envs) ─────────────────────────
        N_r = self._n_bead_per_ring
        n_rings = 2 * self._sec_divs[0]
        # Per-env segment indices offset by n_bead*env in the flat bead_pos array
        seg_s_1env = np.array([i + k * N_r for k in range(n_rings) for i in range(N_r)], dtype=np.int32)
        seg_e_1env = np.array([(i + 1) % N_r + k * N_r for k in range(n_rings) for i in range(N_r)], dtype=np.int32)
        n_segs_1env = len(seg_s_1env)
        seg_s_all = np.concatenate([seg_s_1env + e * n_bead for e in range(n_envs)])
        seg_e_all = np.concatenate([seg_e_1env + e * n_bead for e in range(n_envs)])
        n_segs_total = n_segs_1env * n_envs
        self._bead_pos_zu = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)
        self._spoke_start_zu = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=device)
        self._ring_line_s = wp.zeros(n_segs_total, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(n_segs_total, dtype=wp.vec3, device=device)
        self._n_ring_segs = n_segs_total

        # ── Contact spike visualization buffers ───────────────────────────────
        # One line segment per FEM node: zero-length when not in contact,
        # spike of height = penetration * _contact_vis_scale when in contact.
        # Runs entirely on GPU in render() — no D→H sync, zero sim cost.
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
        # reference config has no internal force.  Matches wheel_mujoco pattern.
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
                self._lateral_spacing,
            ],
            device=device,
        )

        if not _USE_GRAPH:
            _anc = self.ancf_solver
            _dt_ref = self._sim_dt
            _anc.graph_step = lambda: _anc.step(None, None, None, None, _dt_ref)

        # Capture the full substep (kin + coupling + ANCF + accumulate + dyn)
        # into a single CUDA graph → 1 launch instead of 3, saving ~42ms/substep.
        # Uses ancf.step() (unrolled NR+PCG) because ancf.graph_step() cannot be
        # nested inside wp.capture_begin (cudaGraphLaunch is not a capturable op).
        # Previous attempt had incomplete state restore (missed state_rigid).
        self._kin_graph = None
        self._dyn_graph = None
        self._substep_graph = None
        # 1. First capture kin+dyn separately — this forces all MuJoCo kernels to compile.
        self._try_capture_mujoco_graphs()
        # Combined substep graph capture moved to after GS buffer allocation below.

        # Combined substep graph: kin/dyn/ancf graphs ready.
        self._try_capture_substep_graph()

        if viewer is not None:
            viewer.set_model(self.model)
            # Set per-world display offsets directly from tire positions so each
            # spindle appears centered on its tire.  viewer.world_offsets is a
            # wp.array[wp.vec3] of size model.world_count.
            if self._n_envs > 1:
                offsets = np.zeros((self._n_envs, 3), dtype=np.float32)
                for _e in range(self._n_envs):
                    offsets[_e, 1] = self._tire_positions[_e]  # ANCF X → Z-up Y
                viewer.world_offsets = wp.array(offsets, dtype=wp.vec3, device=device)

    # ── Substep graph capture ─────────────────────────────────────────────────

    def _try_capture_substep_graph(self) -> None:
        """Capture one complete substep as a single CUDA graph (1 launch vs 3).

        Saves and restores ALL simulation state — ANCF, state_0, and state_rigid —
        around the capture warmup.  ancf.step() is used (unrolled NR+PCG) because
        ancf.graph_step() cannot be nested inside wp.capture_begin.
        """
        dev = "cuda:0"
        ancf = self.ancf_solver
        dt = self._sim_dt

        # ── Save state for restore after warmup ────────────────────────────────
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

        # Pre-warmup: force ANCF graph exec instantiation before combined capture.
        # wp.capture_end() creates the graph but NOT the exec (lazy). Launching once
        # here forces instantiation OUTSIDE any capture context (error 900 otherwise).
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

        # ── Capture: one pass per substep using direct kernel launches ──────────
        # cudaGraphLaunch is NOT allowed during stream capture (error 900), so
        # ancf.step() (unrolled NR+PCG) is used instead of ancf.graph_step().
        # All kernels pre-compiled by _try_capture_mujoco_graphs().
        def _one_substep_direct():
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
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            # Predictor: extrapolate spindle pos by vel × dt → O(dt²) interface (vs O(dt) without).
            # Reduces coupling lag and stabilises the FEM-rigid feedback at larger substep dt.
            self._prescribe_beads(vel_predict_dt=dt)
            if self._alpha_m_damp > 0.0:
                wp.launch(
                    _update_mass_damp,
                    dim=self._n_envs * self._n_nodes,
                    inputs=[ancf.node_xd, ancf.lumped_mass_tiled, self._alpha_m_damp, ancf.node_f_ext_persistent],
                    device=dev,
                )
            ancf.step(None, None, None, None, dt)
            # Corrector: pin beads to current spindle pos (no extrapolation) after FEM solve.
            self._prescribe_beads(vel_predict_dt=0.0)
            self._accumulate_wrenches()
            self.solver.step_dynamics(self.state_rigid)
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

        try:
            wp.capture_begin(device=dev)
            _one_substep_direct()  # capture ONE substep only
            self._substep_graph = wp.capture_end(device=dev)
            print(f"[SUBSTEP GRAPH] Captured 1 substep — {self._substeps} launches/frame ✓")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._substep_graph = None
            print(f"[SUBSTEP GRAPH] Capture failed ({e!r}) — falling back to 3-graph mode")

        # ── Restore state ───────────────────────────────────────────────────────
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

    def _try_capture_mujoco_graphs(self) -> None:
        """Capture step_kinematics and step_dynamics as standalone CUDA graphs.

        These are captured SEPARATELY (not nested inside ANCF capture) which avoids
        the CUDA error 900 that occurs when trying to nest graph captures.  In the GS
        loop, graph launches replace the ~50-kernel Python call sequences, reducing
        per-call overhead from ~3-4ms to ~0.5ms each.
        """
        dev = "cuda:0"
        dt = self._sim_dt
        wp.synchronize_device(dev)
        try:
            wp.capture_begin(device=dev)
            self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, dt)
            self._kin_graph = wp.capture_end(device=dev)
            print("[MJ GRAPHS] Kinematics graph captured ✓")
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
            print("[MJ GRAPHS] Dynamics graph captured ✓")
        except Exception as e:
            try:
                wp.capture_end(device=dev)
            except Exception:
                pass
            self._dyn_graph = None
            print(f"[MJ GRAPHS] Dynamics capture failed: {e!r}")

        # ── Force exec instantiation by launching each graph once ─────────────
        # wp.capture_end() creates the graph but NOT the exec (lazy creation).
        # When used as child nodes inside the combined capture, Warp tries to
        # lazily create the exec during capture → cudaGraphInstantiate → error 900.
        # Launching once here forces instantiation OUTSIDE any capture context.
        wp.synchronize_device(dev)
        if self._kin_graph is not None:
            wp.capture_launch(self._kin_graph)  # forces _kin_graph.graph_exec creation
        if self._dyn_graph is not None:
            wp.capture_launch(self._dyn_graph)  # forces _dyn_graph.graph_exec creation
        wp.synchronize_device(dev)
        print("[MJ GRAPHS] Graph execs instantiated ✓ — ready for child graph composition")

    # ── Coupling helpers ───────────────────────────────────────────────────────

    def _prescribe_beads(self, vel_predict_dt: float = 0.0) -> None:
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
                self._lateral_spacing,
                self._r_outer,
                vel_predict_dt,
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

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self) -> None:
        # Per-substep graph strategy: one small graph per substep.
        # Each graph has ~140 nodes (1 NR block) → ~1 ms launch overhead vs 75 ms
        # for the old all-substeps frame graph (~2800 nodes). GPU executes while CPU
        # launches the next substep graph: max(CPU_launch, GPU_exec) ≈ GPU-bound.
        if self._substep_graph is not None:
            for _sub in range(self._substeps):
                wp.capture_launch(self._substep_graph)
            wp.synchronize_device()
            return

        # Fallback: per-substep loop using pre-captured small sub-graphs.
        _ta = time.perf_counter()
        ancf = self.ancf_solver
        dev = "cuda:0"

        for _sub in range(self._substeps):
            wp.launch(
                _advance_rim_phi_batched,
                dim=self._n_envs,
                inputs=[self._rim_phi, self._rim_omega_wp, self._sim_dt],
                device=dev,
            )
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
            if self._kin_graph is not None:
                wp.capture_launch(self._kin_graph)
            else:
                self.solver.step_kinematics(self.state_0, self.state_rigid, self.control, None, self._sim_dt)
            self._prescribe_beads(vel_predict_dt=self._sim_dt)
            if self._alpha_m_damp > 0.0:
                wp.launch(
                    _update_mass_damp,
                    dim=self._n_envs * self._n_nodes,
                    inputs=[ancf.node_xd, ancf.lumped_mass_tiled, self._alpha_m_damp, ancf.node_f_ext_persistent],
                    device=dev,
                )
            ancf.graph_step()
            self._prescribe_beads(vel_predict_dt=0.0)
            self._accumulate_wrenches()
            if self._dyn_graph is not None:
                wp.capture_launch(self._dyn_graph)
            else:
                self.solver.step_dynamics(self.state_rigid)
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

        self._t_kin += time.perf_counter() - _ta

    def step(self) -> None:
        # Ramp CTIS pressure per-env toward GUI targets (2000 Pa/frame each)
        if self._build_pressure > 0.0:
            for e in range(self._n_envs):
                d = self._pressure_targets[e] - self._pressure_currents[e]
                step = min(abs(d), 2000.0) * (1.0 if d >= 0.0 else -1.0)
                self._pressure_currents[e] += step
            # Keep single-env compat aliases in sync (env 0)
            self._pressure_current = self._pressure_currents[0]
            self._pressure_target = self._pressure_targets[0]
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

        # ── RPM debug (every 1 second real time) ─────────────────────────────
        # Off by default: the block below issues ~8 device-to-host copies per
        # print, which stalls the pipeline.  Enable with --debug-rpm.
        _now = time.perf_counter()
        if self._debug_rpm and _now - getattr(self, "_rpm_dbg_last", 0.0) >= 1.0:
            try:
                self.control.joint_target_vel.numpy()
                self.state_0.joint_qd.numpy()
                cvel_np = self.solver.cvel.numpy()
                phi_np = self._rim_phi.numpy()
                xd_np = self.ancf_solver.node_xd.numpy()
                print(f"\n[RPM dbg f={self._frame}]")
                for e in range(self._n_envs):
                    float("nan")  # no spin_motor ctrl — spin is kinematic
                    float("nan")  # no spin_y DOF in MuJoCo
                    omega_mj = float(rim_omega_np[e])  # kinematic rim_omega
                    phi_deg = float(phi_np[e]) * 180.0 / math.pi
                    # bead node (constrained, should have tangential velocity)
                    b0 = e * self._n_nodes + int(self._bead_idx_np[0])
                    bxd = xd_np[b0]
                    xd_np[b0]  # position already fetched as xd_np

                    # crown node — pick the node nearest the tread crown (furthest from hub)
                    # Approximate: all nodes, pick max Y in ANCF frame (= max distance from axle)
                    x_env = self.ancf_solver.node_x.numpy()[e * self._n_nodes : (e + 1) * self._n_nodes]
                    xd_env = xd_np[e * self._n_nodes : (e + 1) * self._n_nodes]
                    # Y-up: hub at hub_y, crown nodes furthest in Y
                    crown_i = int(np.argmax(x_env[:, 1]))  # highest Y = crown
                    x_env[crown_i]
                    crown_xd = xd_env[crown_i]
                    # Tangential speed of crown node (should be non-zero if tire spins)
                    float(np.linalg.norm(crown_xd))
                    float(np.linalg.norm(bxd))
                    # forward position + velocity from slide_x DOF
                    jq_e = self.state_0.joint_q.numpy()
                    jqd_e = self.state_0.joint_qd.numpy()
                    slide_x_q_i = self._slide_z_q_dof - 1  # slide_x q  = index 0
                    slide_x_qd_i = self._slide_x_qd_dof  # slide_x qd = direct
                    fwd_m = float(jq_e[slide_x_q_i + e * self._n_q_per_world]) if slide_x_q_i >= 0 else float("nan")
                    fwd_v = float(jqd_e[slide_x_qd_i + e * self._n_qd_per_world])
                    v_fwd_mj = float(cvel_np[e, self._spindle_mj][3])  # MuJoCo vx = forward
                    print(
                        f"  e{e}: rpm_target={self._target_rpm[e]:+.1f}  omega={omega_mj * 60 / 6.2832:+.1f}rpm"
                        f"  |  fwd={fwd_m:+.6f}m  v_fwd={fwd_v:+.4f}m/s  cvel_x={v_fwd_mj:+.4f}m/s"
                        f"  |  phi={phi_deg:+.1f}deg"
                    )
            except Exception as _e:
                print(f"[RPM dbg ERROR] {_e}")
            self._rpm_dbg_last = _now

        _t0 = time.perf_counter()
        self.simulate()
        wp.synchronize_stream(wp.get_stream("cuda:0"))  # drain GPU stream before viewer reads body_q
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
                self._lateral_spacing,
            ],
            device="cuda:0",
        )

        if self._frame % self._diag_period == 0:
            _now = time.perf_counter()
            _fps = self._diag_period / max(_now - self._t_wall, 1e-9)
            _ms = 1e3 / max(_fps, 1e-3)
            self._print_diag(_fps, _ms)
            self._t_wall = _now
            self._t_step = self._t_render = self._t_kin = self._t_ancf = self._t_dyn = 0.0

    def _print_diag(self, fps: float = 0.0, ms: float = 0.0) -> None:
        with wp.ScopedTimer("diag", use_nvtx=False, color="red"):
            x_all = self.ancf_solver.node_x.numpy()  # (N*n_nodes, 3)
            xd_all = self.ancf_solver.node_xd.numpy()  # (N*n_nodes, 3)
            stg_all = self._xfrc_stg.numpy()  # (N,)
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
            hub = np.array([float(sp_tf[1]) + e * self._lateral_spacing, float(sp_tf[2]), float(sp_tf[0])])
            expected = np.stack(
                [
                    hub[0] + rest_np[:, 0],
                    hub[1] + rest_np[:, 1] * cp - rest_np[:, 2] * sp_,
                    hub[2] + rest_np[:, 1] * sp_ + rest_np[:, 2] * cp,
                ],
                axis=1,
            )
            drift_mm = float(np.max(np.linalg.norm(bead_x - expected, axis=1))) * 1e3
            fz = float(stg_e[4])

            # What the coupling kernel actually uses as hub position
            xpos_e = xpos_all[e, self._spindle_mj]  # MuJoCo Z-up (x_fwd,y_lat,z_up)
            hub_x_mj = float(xpos_e[1]) + e * self._lateral_spacing  # → ANCF X
            hub_y_mj = float(xpos_e[2])  # → ANCF Y (height)

            # ── Rolling slip test ─────────────────────────────────────────────
            # No-slip rolling: v_hub_fwd == omega_axle * R_outer.
            # ang_y (MuJoCo cvel[1]) = axle spin; vx (cvel[3]) = hub forward velocity.
            # In Z-up: contact-patch x-vel = v_hub_x - omega_y * R_outer = 0 for pure roll.
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
        stg_np = stg_all[0]
        fz_contact = float(stg_np[4])
        fz_exp = self._m_rigid * _GRAVITY
        sp_z = float(sp_tf0[2])
        xpos_zu = xpos_all[0, self._spindle_mj]
        float(abs(xpos_zu[2] - sp_tf0[2])) * 1e3

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
        _kin_ms = 1e3 * self._t_kin / self._diag_period
        _ancf_ms = 1e3 * self._t_ancf / self._diag_period
        _dyn_ms = 1e3 * self._t_dyn / self._diag_period
        print(
            f"[{self._frame:4d}] {'NaN!' if any_nan else 'ok  '}"
            f"  t={self._t:.2f}s"
            f"  sp_Z={sp_z:+.4f}m"
            f"  F_z={fz_contact:+.0f}N (wt~{fz_exp:.0f}N)"
            f"  bead_drift={bead_drift_mm:.3f}mm"
            f"  v_max={node_v_max:.2f}m/s"
            f"  fi_max={fi_max:.2e}"
            f"  fps={fps:.1f} ({ms:.1f}ms/frame"
            f"  step={_step_ms:.1f}ms[kin={_kin_ms:.1f} ancf={_ancf_ms:.1f} dyn={_dyn_ms:.1f}]"
            f"  render={_render_ms:.1f}ms)"
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
        _rpm_changed = False
        for e in range(self._n_envs):
            changed, val = ui.slider_float(f"Env {e}##rpm{e}", self._target_rpm[e], -300.0, 300.0)
            if changed:
                self._target_rpm[e] = float(val)
                _rpm_changed = True
            omega = self._target_rpm[e] * (2.0 * math.pi / 60.0)
            ui.text(f"  env{e}  {self._target_rpm[e]:+7.1f} RPM  ({omega:+.2f} rad/s)")
        # rim_omega_wp is updated gradually in step(); no immediate write here.

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
        F_trac_max = self._mu_roll * F_total
        ui.text(f"  F_normal ~{F_total:.0f} N  →  max traction {F_trac_max:.0f} N")
        ui.text(f"  safe max {self._f_load_safe_max:.0f} N  (raise kn=300k → 7500 N)")

        ui.separator()
        ui.text("Axle slide damping")
        changed, val = ui.slider_float("b_slide [N·s/m]##bslide", self._b_slide, 0.0, 5000.0)
        if changed:
            self._b_slide = float(val)
        F_damp_preview = self._b_slide * abs(self._gui_v_fwd)
        ui.text(f"  F_damp = {F_damp_preview:.1f} N  at v_hub={self._gui_v_fwd:+.3f} m/s")

        ui.separator()
        ui.text("Contact spikes")
        changed, val = ui.slider_float("pen scale##cvis", self._contact_vis_scale, 1.0, 500.0)
        if changed:
            self._contact_vis_scale = float(val)
        ui.text("  cyan spikes = in-contact nodes, height = pen × scale")

        ui.separator()
        ui.text("Rolling traction")
        changed, val = ui.slider_float("mu_roll##mur", self._mu_roll, 0.0, 2.0)
        if changed:
            self._mu_roll = float(val)
        changed, val = ui.slider_float("v_reg [m/s]##vreg", self._v_reg_roll, 0.001, 0.5)
        if changed:
            self._v_reg_roll = float(val)
        _slip_now = self._gui_slip_vel
        _F_n_now = max(0.0, self._gui_fz)
        _F_trac = -self._mu_roll * _F_n_now * math.tanh(_slip_now / max(self._v_reg_roll, 1e-6))
        ui.text(f"  F_traction   {_F_trac:+.1f} N  (mu×Fn×tanh(slip/v_reg))")

        ui.separator()
        ui.text("Rolling slip  (env 0)")
        ui.text(f"  v_roll       {self._gui_v_roll:+.4f} m/s  (omega * R)")
        ui.text(f"  v_hub_fwd    {self._gui_v_fwd:+.4f} m/s  (spindle vx)")
        ui.text(f"  slip vel     {self._gui_slip_vel:+.4f} m/s  (v_hub - v_roll)")
        ui.text(f"  slip pos     {self._gui_pos_slip:+.4f} m   (hub_x - phi*R)")

    # ── Render ─────────────────────────────────────────────────────────────────

    # _update_spin_ctrl and _update_rolling_ctrl removed:
    # spin is now kinematic via rim_omega_wp GPU array (updated in step()).
    # lin_motor ctrl is set per-substep in _one_substep_direct / simulate() loop.

    def render(self) -> None:
        if self.viewer is None:
            return
        _t0 = time.perf_counter()
        self.viewer.begin_frame(self._t)

        # Display-only: shift each world's rigid bodies laterally so they align
        # with their ANCF tire.  Physics body_q is restored immediately after.
        # viewer.set_world_offsets() already handles per-world lateral offsets for bodies.
        self.viewer.log_state(self.state_0)
        # Orange rings: two closed-polygon outlines pushed 15 mm outside crown.
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
        # Cyan spikes: one per FEM node, height ∝ contact penetration.
        # Kernel runs entirely on GPU — no D→H copy, no FPS cost.
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

        stg_np = self._xfrc_stg.numpy()[0]
        fz = abs(float(stg_np[4]))
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
            default=4,
            help="Number of parallel environments (tires + spindles). Default: 4.",
        )
        parser.add_argument(
            "--m-rigid",
            type=float,
            default=_M_RIGID,
            help="Total rigid body mass [kg] used for contact force validation. "
            "Must match the spindle mass in the MJCF. Default: 31.88 kg (FEDA).",
        )
        parser.add_argument(
            "--mjcf",
            type=str,
            default=None,
            help="Path to MJCF XML (absolute or relative to assets/). "
            "Default: ancf_single_tire.xml. Must have bodies named "
            "'carrier/slide_z' and 'spindle/spin_y'.",
        )
        parser.add_argument(
            "--drop-clearance",
            type=float,
            default=_DROP_CLEARANCE,
            help="Initial hub clearance above ground [m]. Default 0 (quasi-static start).",
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
            "--debug-rpm",
            action="store_true",
            help="Print per-env RPM/velocity debug once per second.  Costs ~8 device-to-host copies per print.",
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
            help="Ground contact normal stiffness [N/m].  "
            "HHT stable when kn < m_node/((0.5-beta)*dt^2).  "
            "At substeps=10, kn<=10k keeps ω·dt<1.",
        )
        parser.add_argument(
            "--kd",
            type=float,
            default=_KD,
            help="Ground contact normal damping [N·s/m].  kd < 2*m_node/dt.",
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
            "--shell-tire",
            type=str,
            default=None,
            help="Single tire config JSON (legacy; use --shell-tires for multi-env).",
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
            "--mu-roll",
            type=float,
            default=_MU_ROLL,
            help="Macro Coulomb rolling friction coefficient at spindle level. "
            "Implements vehicle-dynamics-style traction (Pacejka concept). "
            "Set 0 to disable. Default: 0.9.",
        )
        parser.add_argument(
            "--v-reg-roll",
            type=float,
            default=_V_REG_ROLL,
            help="Slip velocity regularisation for rolling traction [m/s]. tanh smoothing width. Default: 0.05 m/s.",
        )
        parser.add_argument(
            "--b-slide",
            type=float,
            default=_B_SLIDE,
            help="Axle slide damping [N·s/m] applied to spindle forward DOF. "
            "Stops spurious drift caused by pressure impulse during drop. "
            "Set 0 to disable. Default: 500.",
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
