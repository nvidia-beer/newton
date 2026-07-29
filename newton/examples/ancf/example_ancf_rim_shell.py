# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Example ANCF Rim Shell
#
# FEDA balloon ANCF3423 FEM tire carcass mounted on a rigid rim with
# inflation pressure applied.  Geometry, material, and solver parameters
# are identical to example_double_wishbone_tires_rims so that stability
# changes can be validated here before being applied to the full vehicle.
#
# Two rim-coupling modes:
#   Mode A  (driven)  — rim spins at prescribed RPM (GUI slider).
#   Mode B  (free)    — rim has moment of inertia; responds to bead torque.
#
# The hub starts at R_outer + drop_height and falls under gravity until
# the tread touches the ground (Y=0).  The rim is visualised as orange
# spokes from the hub centre to each bead-ring node.
#
# Multi-tire: pass --shell-tires as a JSON array (see ancf_rim_shell.json).
# N tires are batched in a single solver; mutable per-tire state lives in
# N-element wp.arrays written before each graph replay.
#
# The full substep loop is captured as a CUDA graph (outer graph wraps
# prescribe + advance-phi + solver.step + post-prescribe each substep).
# Per-frame CPU work (gravity drop, pressure ramp, RPM slider) runs
# outside the graph; mutable state is passed via N-element wp.arrays.
#
# Command: python -m newton.examples ancf.example_ancf_rim_shell
###########################################################################

import json
import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.ancf_shell import (
    SolverANCFShell,
    build_ancf_tire_mesh,
    isotropic_ancf_material,
)
from newton._src.solvers.coupling import _pack_ancf as _pack_ancf_kernel
from newton._src.solvers.coupling import _unpack_ancf as _unpack_ancf_kernel

# ── FEDA tire geometry ────────────────────────────────────────────────────────
_R_OUTER = 0.499  # tread crown radius    [m]
_R_INNER = 0.286  # bead / rim seat       [m]
_WIDTH = 0.335  # bead-to-bead width    [m]
_SEC_DIVS = (1, 2, 3)

# Default material — overridden by --shell-tires JSON (e.g. ancf_rim_shell.json)
_E_BALLOON = 1.0e6  # [Pa]  balloon default; DW uses 5e7
_NU_BALLOON = 0.45
_RHO_BALLOON = 700.0  # [kg/m³]
_THICK = 0.006  # [m]

# Pressure defaults — overridden by per-tire JSON pressure key
_BUILD_PRESSURE = 30_000.0  # [Pa]
_MAX_GAUGE = 8_000.0  # [Pa]
_GAUGE_RATE = 500.0  # [Pa/frame]
_RPM_RATE = 4.0  # rpm per frame (~1.3 s for full +/-300 rpm travel)

_G = 9.81  # [m/s²]


# ── Warp kernels ──────────────────────────────────────────────────────────
# All kernels are batched: dim = N * n_bead (or N).
# env = tid // n_bead  selects the tire;  i = tid % n_bead selects the bead.
# N-element hub arrays are written by Python before each graph replay and
# read inside the CUDA graph — no host/device sync inside the loop.


@wp.kernel
def _prescribe_bead_batched(
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_xdd: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    bead_rest: wp.array[wp.vec3],
    bead_D0: wp.array[wp.vec3],
    hub_x: wp.array[float],
    hub_y: wp.array[float],
    hub_vy: wp.array[float],
    hub_z: wp.array[float],
    rim_phi: wp.array[float],
    rim_omega: wp.array[float],
    R_outer: float,
    n_bead: int,
    n_nodes: int,
):
    """dim = N * n_bead. Prescribe bead kinematics for every tire env."""
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead

    local_idx = bead_idx[i]
    global_idx = env * n_nodes + local_idx

    r = bead_rest[i]
    d0 = bead_D0[i]

    phi = rim_phi[env]
    omega = rim_omega[env]
    hy = hub_y[env]
    hvy = hub_vy[env]
    hz = hub_z[env]
    hx = hub_x[env]
    hvz = omega * R_outer

    cp = wp.cos(phi)
    sp = wp.sin(phi)

    y_loc = r[1] * cp - r[2] * sp
    z_new = r[1] * sp + r[2] * cp

    node_x[global_idx] = wp.vec3(hx + r[0], hy + y_loc, hz + z_new)
    node_xd[global_idx] = wp.vec3(0.0, hvy - omega * z_new, hvz + omega * y_loc)
    node_xdd[global_idx] = wp.vec3(0.0, 0.0, 0.0)

    d_y = d0[1] * cp - d0[2] * sp
    d_z = d0[1] * sp + d0[2] * cp
    node_D[global_idx] = wp.vec3(d0[0], d_y, d_z)
    node_Dd[global_idx] = wp.vec3(0.0, -omega * d_z, omega * d_y)
    node_Ddd[global_idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _advance_phi_and_z_batched(
    rim_phi: wp.array[float],
    rim_omega: wp.array[float],
    hub_z: wp.array[float],
    dt: float,
    R_outer: float,
):
    """dim = N. Advance rim angle and hub Z by no-slip rolling for each tire."""
    e = wp.tid()
    omega = rim_omega[e]
    rim_phi[e] = rim_phi[e] + omega * dt
    hub_z[e] = hub_z[e] + omega * R_outer * dt


@wp.kernel
def _accum_bead_torque_batched(
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    bead_rest: wp.array[wp.vec3],
    bead_mass: wp.array[float],
    hub_x: wp.array[float],
    hub_y: wp.array[float],
    hub_z: wp.array[float],
    rim_phi: wp.array[float],
    inv_dt2: float,
    tau_buf: wp.array[float],
    n_bead: int,
    n_nodes: int,
):
    """dim = N * n_bead. Accumulate rim torque from bead constraint residual."""
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead

    local_idx = bead_idx[i]
    global_idx = env * n_nodes + local_idx

    r = bead_rest[i]
    m = bead_mass[i]

    phi = rim_phi[env]
    cp = wp.cos(phi)
    sp = wp.sin(phi)

    y_loc = r[1] * cp - r[2] * sp
    z_loc = r[1] * sp + r[2] * cp

    tgt = wp.vec3(hub_x[env] + r[0], hub_y[env] + y_loc, hub_z[env] + z_loc)
    delta = node_x[global_idx] - tgt

    tau = (y_loc * m * delta[2] - z_loc * m * delta[1]) * inv_dt2
    wp.atomic_add(tau_buf, env, tau)


@wp.kernel
def _update_rim_omega_batched(
    rim_omega: wp.array[float],
    tau_buf: wp.array[float],
    dt: float,
    I_rim: float,
):
    """dim = N. Integrate rim angular velocity and reset torque accumulator."""
    e = wp.tid()
    rim_omega[e] = rim_omega[e] + tau_buf[e] * dt / I_rim
    tau_buf[e] = float(0.0)


@wp.kernel
def _accum_bead_fy_batched(
    global_f_int: wp.array[float],
    bead_idx: wp.array[wp.int32],
    fy_buf: wp.array[float],
    n_bead: int,
    n_nodes: int,
):
    """dim = N * n_bead. Accumulate hub-Y reaction from ANCF bead internal forces.

    Newton 3rd law: force on hub = −global_f_int at bead nodes.
    """
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]
    wp.atomic_add(fy_buf, env, -global_f_int[global_idx * 6 + 1])


@wp.kernel
def _integrate_hub_y(
    hub_y: wp.array[float],
    hub_vy: wp.array[float],
    fy_buf: wp.array[float],
    total_mass: float,
    g_load: float,
    dt: float,
):
    """dim = N. Symplectic-Euler integration of hub Y: F_net = fy_bead − g_load.

    Clears fy_buf after use so the buffer is ready for the next GS iteration.
    """
    e = wp.tid()
    f_net = fy_buf[e] - g_load
    hub_vy[e] = hub_vy[e] + (f_net / total_mass) * dt
    hub_y[e] = hub_y[e] + hub_vy[e] * dt
    fy_buf[e] = float(0.0)


@wp.kernel
def _build_rim_disc_batched(
    hub_x_wp: wp.array[float],
    hub_y_wp: wp.array[float],
    hub_z_wp: wp.array[float],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    rim_radius: float,
    rim_half_width: float,
    flange_width: float,
    n_pts: int,
    n_struts: int,
    segs_per_env: int,
):
    """dim = N * segs_per_env.

    Builds rim geometry as line segments:
      - n_pts segments → left-side flange circle  (x = hub_x - (rim_half_width + flange_width))
      - n_pts segments → right-side flange circle (x = hub_x + (rim_half_width + flange_width))
      - n_struts segs  → axial struts at rim seat, connecting left to right

    flange_width > 0 pushes the circles axially beyond the tire sidewall so
    they are visible from typical side-on viewing angles.
    """
    tid = wp.tid()
    env = tid // segs_per_env
    seg = tid % segs_per_env

    hx = hub_x_wp[env]
    hy = hub_y_wp[env]
    hz = hub_z_wp[env]

    two_pi = float(6.283185307)
    x_flange = rim_half_width + flange_width

    if seg < n_pts:
        # Left flange circle (extends beyond left sidewall)
        a0 = two_pi * float(seg) / float(n_pts)
        a1 = two_pi * float(seg + 1) / float(n_pts)
        starts[tid] = wp.vec3(hx - x_flange, hy + rim_radius * wp.cos(a0), hz + rim_radius * wp.sin(a0))
        ends[tid] = wp.vec3(hx - x_flange, hy + rim_radius * wp.cos(a1), hz + rim_radius * wp.sin(a1))
    elif seg < 2 * n_pts:
        # Right flange circle (extends beyond right sidewall)
        k = seg - n_pts
        a0 = two_pi * float(k) / float(n_pts)
        a1 = two_pi * float(k + 1) / float(n_pts)
        starts[tid] = wp.vec3(hx + x_flange, hy + rim_radius * wp.cos(a0), hz + rim_radius * wp.sin(a0))
        ends[tid] = wp.vec3(hx + x_flange, hy + rim_radius * wp.cos(a1), hz + rim_radius * wp.sin(a1))
    else:
        # Axial struts at rim seat radius, spanning full flange-to-flange width
        k = seg - 2 * n_pts
        a = two_pi * float(k) / float(n_struts)
        cy = hy + rim_radius * wp.cos(a)
        cz = hz + rim_radius * wp.sin(a)
        starts[tid] = wp.vec3(hx - x_flange, cy, cz)
        ends[tid] = wp.vec3(hx + x_flange, cy, cz)


@wp.kernel
def _update_spindle_body_q(
    hub_x: wp.array[float],  # (N,) — static hub X positions
    hub_y: wp.array[float],  # (N,) — current hub Y (updated before graph replay)
    hub_z: wp.array[float],  # (N,) — current hub Z (updated inside graph)
    rim_phi: wp.array[float],  # (N,) — current rim angle [rad]
    body_idx: wp.array[wp.int32],  # (N,) — Newton body indices for spindles
    body_q: wp.array[wp.transform],  # all body transforms in the Newton model
):
    """dim = N.  Write each spindle's world transform into body_q.

    Rotation: around ANCF X-axis by rim_phi (so spokes visibly spin).
    No CPU round-trip — pure GPU kernel.
    """
    e = wp.tid()
    bidx = body_idx[e]
    pos = wp.vec3(hub_x[e], hub_y[e], hub_z[e])
    phi = rim_phi[e]
    half = phi * 0.5
    q = wp.quat(wp.sin(half), 0.0, 0.0, wp.cos(half))  # quat_from_axis_angle(X, phi)
    body_q[bidx] = wp.transform(pos, q)


@wp.kernel
def _build_spoke_lines_batched(
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    hub_x: wp.array[float],
    hub_y: wp.array[float],
    hub_z: wp.array[float],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    n_bead: int,
    n_nodes: int,
):
    """dim = N * n_bead. Lines from hub centre to bead nodes for rim viz."""
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    global_idx = env * n_nodes + bead_idx[i]
    starts[tid] = wp.vec3(hub_x[env], hub_y[env], hub_z[env])
    ends[tid] = node_x[global_idx]


@wp.kernel
def _accum_bead_wrench_kinematic(
    global_f_int: wp.array[float],
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    hub_x: wp.array[float],
    hub_y: wp.array[float],
    hub_z: wp.array[float],
    wrench_buf: wp.array[wp.spatial_vector],  # (N,) [tau_x,tau_y,tau_z, fx,fy,fz] Y-up
    n_bead: int,
    n_nodes: int,
):
    """dim = N * n_bead.  Full 6-DOF bead wrench from global_f_int (Lagrange λ).

    Replaces _accum_bead_torque_batched (position residual, spin only) and
    _accum_bead_fy_batched (vertical only).  Uses the exact constraint reaction,
    matching SolverANCFShellRigid._accum_bead_wrench_wheel in ancf_rigid_mujoco_tires.

    ANCF Y-up: X=lateral, Y=up, Z=forward.
      tau_x  → spin torque  (updates rim_omega via I_rim)
      fy     → vertical     (updates hub_vy via hub_mass, includes tire weight)
      fz     → longitudinal (available for future slip physics)
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

    hub = wp.vec3(hub_x[env], hub_y[env], hub_z[env])
    r = node_x[global_idx] - hub
    tau = wp.cross(r, f)

    wp.atomic_add(wrench_buf, env, wp.spatial_vector(tau[0], tau[1], tau[2], fx, fy, fz))


@wp.kernel
def _integrate_hub_dof(
    hub_y: wp.array[float],
    hub_vy: wp.array[float],
    rim_omega: wp.array[float],
    wrench_buf: wp.array[wp.spatial_vector],
    hub_mass: float,
    I_rim: float,
    g_load: float,  # downward gravitational + corner load [N]
    tare_fy: float,  # tire self-weight tare [N] (subtracted from fy_bead)
    dt: float,
    update_vy: int,  # 1 = integrate vertical DOF, 0 = gravity-only
    update_om: int,  # 1 = integrate spin DOF, 0 = prescribed
):
    """dim = N.  Integrate hub vertical (Fy) and spin (tau_x) from bead wrench.

    Unified replacement for _integrate_hub_y + _update_rim_omega_batched.
    Clears wrench_buf after use.

    Vertical: a_y = (fy_bead + tare_fy − g_load) / hub_mass
    Spin:     alpha = tau_x / I_rim
    """
    e = wp.tid()
    w = wrench_buf[e]

    if update_vy == 1:
        fy_bead = w[4] + tare_fy  # Y-up force on hub, corrected for tire weight
        f_net = fy_bead - g_load  # subtract downward load
        hub_vy[e] = hub_vy[e] + (f_net / hub_mass) * dt
        hub_y[e] = hub_y[e] + hub_vy[e] * dt

    if update_om == 1:
        rim_omega[e] = rim_omega[e] + w[0] * dt / I_rim  # tau_x

    wrench_buf[e] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _gather_bead_positions(
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],
    bead_pos: wp.array[wp.vec3],  # (N*n_bead,) output
    n_bead: int,
    n_nodes: int,
):
    """dim = N * n_bead. Gather bead node positions (Y-up, no conversion needed)."""
    tid = wp.tid()
    env = tid // n_bead
    i = tid % n_bead
    bead_pos[tid] = node_x[env * n_nodes + bead_idx[i]]


@wp.kernel
def _build_ring_lines(
    bead_pos: wp.array[wp.vec3],
    seg_s: wp.array[wp.int32],
    seg_e: wp.array[wp.int32],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    """dim = n_segs. Build closed bead-ring line segments."""
    i = wp.tid()
    starts[i] = bead_pos[seg_s[i]]
    ends[i] = bead_pos[seg_e[i]]


@wp.kernel
def _gather_contact_spikes_yup(
    node_x: wp.array[wp.vec3],  # ANCF Y-up (ground at Y=0)
    ground_y: float,
    vis_scale: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    """Per-node contact spike. Y-up: spike rises in +Y above the ground plane.
    Zero-length segment when not in contact (invisible in viewer)."""
    i = wp.tid()
    p = node_x[i]
    pen = ground_y - p[1]  # positive = penetrating
    base = wp.vec3(p[0], ground_y, p[2])
    if pen > 0.0:
        starts[i] = base
        ends[i] = wp.vec3(base[0], base[1] + pen * vis_scale, base[2])
    else:
        starts[i] = base
        ends[i] = base


# ── Example class ─────────────────────────────────────────────────────────


class Example:
    """FEDA balloon ANCF tire(s) on a rigid rim — driven or free-revolute.

    Pass ``--shell-tires`` as a JSON array to run N tires side by side.
    All tires share the same ANCF mesh topology; materials and RPMs are
    per-tire.  The full substep loop is captured as one CUDA graph.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(getattr(args, "substeps", 10))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        device = "cuda:0"
        self._mode = str(getattr(args, "mode", "driven"))

        # ── Parse per-tire config ─────────────────────────────────────────
        tires_raw = getattr(args, "shell_tires", None)
        if isinstance(tires_raw, str):
            tires_raw = json.loads(tires_raw)
        if tires_raw:
            tire_cfgs = [t for t in tires_raw if t.get("active", True)]
        else:
            tire_cfgs = [
                {
                    "name": "balloon",
                    "position": [0.0, 0.0, 0.0],
                    "rpm": float(getattr(args, "rpm", 0.0)),
                }
            ]
        n_envs = len(tire_cfgs)
        self._n_envs = n_envs

        # ── Build ANCF mesh using first tire's material ───────────────────
        n_circ = int(tire_cfgs[0].get("n-circ", int(getattr(args, "n_circ", 20))))
        thick = float(tire_cfgs[0].get("thickness", _THICK))

        E0 = tire_cfgs[0].get("E", _E_BALLOON)
        nu0 = tire_cfgs[0].get("nu", _NU_BALLOON)
        rho0 = tire_cfgs[0].get("rho", _RHO_BALLOON)
        if E0 is None:
            E0, nu0, rho0 = _E_BALLOON, _NU_BALLOON, _RHO_BALLOON
        alpha_d0 = float(tire_cfgs[0].get("alpha-damp", 0.15))
        mat0 = isotropic_ancf_material(E=float(E0), nu=float(nu0), rho=float(rho0), alpha_damp=alpha_d0)

        self.ancf_model = build_ancf_tire_mesh(
            R_outer=_R_OUTER,
            R_inner=_R_INNER,
            width=_WIDTH,
            n_circ=n_circ,
            section_divs=_SEC_DIVS,
            section_mats=(mat0, mat0, mat0),
            section_h=(thick, thick, thick),
            pressure=float(tire_cfgs[0].get("pressure", _BUILD_PRESSURE)),
            device=device,
        )
        n_nodes = self.ancf_model.n_nodes
        n_ax_divs = 2 * sum(_SEC_DIVS)  # 12 for (1, 2, 3)
        self._n_nodes = n_nodes

        # For N > 1 tile elem_mat across envs (supports per-tire material)
        if n_envs > 1:
            base_mat_np = self.ancf_model.elem_mat.numpy()  # (n_elems, 11)
            ne = self.ancf_model.n_elems
            per_env_mats = []
            for cfg in tire_cfgs:
                E_c = cfg.get("E", None)
                nu_c = cfg.get("nu", None)
                rho_c = cfg.get("rho", None)
                alpha_dc = float(cfg.get("alpha-damp", 0.15))
                if E_c is not None:
                    m = isotropic_ancf_material(
                        E=float(E_c),
                        nu=float(nu_c or _NU_BALLOON),
                        rho=float(rho_c or _RHO_BALLOON),
                        alpha_damp=alpha_dc,
                    )
                    row = np.array(
                        [
                            m.C11,
                            m.C22,
                            m.C33,
                            m.C12,
                            m.C13,
                            m.C23,
                            m.G23,
                            m.G13,
                            m.G12,
                            m.rho,
                            m.alpha_damp,
                        ],
                        dtype=np.float32,
                    )
                    per_env_mats.append(np.tile(row[np.newaxis, :], (ne, 1)))
                else:
                    per_env_mats.append(base_mat_np.copy())
            self.ancf_model.elem_mat = wp.array(np.concatenate(per_env_mats, axis=0), dtype=float, device=device)

        # ── Bead ring nodes (same for all envs) ───────────────────────────
        left_bead = np.arange(0, n_circ, dtype=np.int32)
        right_bead = np.arange(n_ax_divs * n_circ, (n_ax_divs + 1) * n_circ, dtype=np.int32)
        all_bead = np.concatenate([left_bead, right_bead])
        n_bead = len(all_bead)
        self._n_bead = n_bead

        x0_np = self.ancf_model.node_x0.numpy()
        D0_np = self.ancf_model.node_D0.numpy()
        bead_rest_np = x0_np[all_bead].astype(np.float32)
        bead_D0_np = D0_np[all_bead].astype(np.float32)
        self._bead_rest_np = bead_rest_np

        self._bead_idx = wp.array(all_bead, dtype=wp.int32, device=device)
        self._bead_rest = wp.array(bead_rest_np, dtype=wp.vec3, device=device)
        self._bead_D0 = wp.array(bead_D0_np, dtype=wp.vec3, device=device)

        # ── Per-tire rigid-rim config ─────────────────────────────────────
        # Each tire can specify a "rim" object in its JSON entry:
        #   "rim": { "visible": true, "kn": 50000.0, "kd": 20.0,
        #            "n-pts": 32, "n-struts": 8, "flange": 0.03,
        #            "hub-mass": 74.2, "inertia": 0.654 }
        # "visible"  (default true): show the rim disc in the viewer.
        # "kn"       (default 0): inner-surface contact stiffness; 0 = no rim contact.
        # "kd"       (default = ground kd): rim contact damping.
        # "flange"   (default 0.03 m): axial overhang beyond tire sidewall so the
        #   rim circles protrude past the opaque mesh and are visible side-on.
        # "hub-mass" (default 74.2 kg): hub + axle translational mass [kg].
        # "inertia"  (default 8*R_inner² kg·m²): rim rotational inertia [kg·m²]
        #   (Mode B / free only; baked into CUDA graph — needs restart to change).
        rim_cfgs = [cfg.get("rim", {}) for cfg in tire_cfgs]
        self._rim_visible = [bool(rc.get("visible", True)) for rc in rim_cfgs]
        # Use the maximum rim-kn across all tires (solver has one value).
        rim_kn_per = [float(rc.get("kn", 0.0)) for rc in rim_cfgs]
        rim_kd_per = [float(rc.get("kd", float(getattr(args, "kd", 20.0)))) for rc in rim_cfgs]
        self._rim_kn = max(rim_kn_per)
        self._rim_kd = float(np.mean(rim_kd_per))

        # Rim disc rendering: 2 circles + struts
        # "flange": axial overhang beyond tire sidewall so circles are visible
        self._rim_flange = float(rim_cfgs[0].get("flange", 0.03))
        n_pts_rim = int(rim_cfgs[0].get("n-pts", 32))
        n_struts_rim = int(rim_cfgs[0].get("n-struts", 8))
        self._rim_n_pts = n_pts_rim
        self._rim_n_struts = n_struts_rim
        self._rim_segs_per_env = 2 * n_pts_rim + n_struts_rim
        self._rim_disc_starts = wp.zeros(n_envs * self._rim_segs_per_env, dtype=wp.vec3, device=device)
        self._rim_disc_ends = wp.zeros(n_envs * self._rim_segs_per_env, dtype=wp.vec3, device=device)

        # Spoke rendering buffers (N * n_bead line segments) — yellow
        self._spoke_starts = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)
        self._spoke_ends = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)

        # ── Bead ring + contact spike visualization ───────────────────────
        # Bead ring: closed polygon connecting adjacent bead nodes — orange
        _N_r = n_circ  # nodes per ring = circumferential divisions
        _n_rings = 2 * _SEC_DIVS[0]  # bead rows per side
        seg_s_np = np.array([i + k * _N_r for k in range(_n_rings) for i in range(_N_r)], dtype=np.int32)
        seg_e_np = np.array([(i + 1) % _N_r + k * _N_r for k in range(_n_rings) for i in range(_N_r)], dtype=np.int32)
        _n_segs_1 = len(seg_s_np)
        seg_s_all = np.concatenate([seg_s_np + e * n_bead for e in range(n_envs)])
        seg_e_all = np.concatenate([seg_e_np + e * n_bead for e in range(n_envs)])
        self._ring_seg_s = wp.array(seg_s_all, dtype=wp.int32, device=device)
        self._ring_seg_e = wp.array(seg_e_all, dtype=wp.int32, device=device)
        self._n_ring_segs = _n_segs_1 * n_envs
        self._bead_pos_yup = wp.zeros(n_envs * n_bead, dtype=wp.vec3, device=device)
        self._ring_line_s = wp.zeros(self._n_ring_segs, dtype=wp.vec3, device=device)
        self._ring_line_e = wp.zeros(self._n_ring_segs, dtype=wp.vec3, device=device)
        # Contact spikes: cyan vertical spikes at penetrating nodes — GPU only
        self._contact_line_s = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=device)
        self._contact_line_e = wp.zeros(n_envs * n_nodes, dtype=wp.vec3, device=device)
        self._contact_vis_scale = 100.0  # pen × scale → visible spike height

        # ── Per-tire hub/rim Python state ─────────────────────────────────
        drop_height = float(getattr(args, "drop_height", 0.5))
        hub_y_init = float(_R_OUTER) + drop_height
        self._hub_rest = float(_R_OUTER)
        self._hub_y_init = hub_y_init
        # hub-mass / inertia: read from rim JSON first, then CLI arg, then default.
        _default_hub_mass = float(getattr(args, "hub_mass", 74.2))
        _default_I_rim = float(getattr(args, "inertia", 8.0 * _R_INNER**2))
        self._hub_mass = float(rim_cfgs[0].get("hub-mass", _default_hub_mass))
        self._I_rim = float(rim_cfgs[0].get("inertia", _default_I_rim))

        self._hub_x = [float(cfg.get("position", [0.0])[0]) for cfg in tire_cfgs]
        self._hub_y = [hub_y_init] * n_envs
        self._hub_vy = [0.0] * n_envs
        self._hub_z = [0.0] * n_envs
        self._rim_phi = [0.0] * n_envs
        self._rim_omega = [0.0] * n_envs

        default_rpm = float(getattr(args, "rpm", 0.0))
        self._target_rpm = [float(cfg.get("rpm", default_rpm)) for cfg in tire_cfgs]
        if self._mode == "driven":
            for e in range(n_envs):
                self._rim_omega[e] = self._target_rpm[e] * (2.0 * np.pi / 60.0)

        self._tire_names = [cfg.get("name", f"tire{e}") for e, cfg in enumerate(tire_cfgs)]
        self._corner_mass = float(getattr(args, "corner_mass", 0.0))
        self._gs_iters_hub = int(getattr(args, "gs_iters", 2))

        # ── World node positions: reference + X offset + Y drop height ────
        world_x = np.empty((n_envs * n_nodes, 3), dtype=np.float32)
        for e in range(n_envs):
            off = np.array([self._hub_x[e], hub_y_init, 0.0], dtype=np.float32)
            world_x[e * n_nodes : (e + 1) * n_nodes] = x0_np.astype(np.float32) + off

        # ── Newton model: ground + all tire particles + triangle topology ──
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane(color=(0.65, 0.65, 0.65))
        builder.add_particles(
            pos=[(float(p[0]), float(p[1]), float(p[2])) for p in world_x],
            vel=[(0.0, 0.0, 0.0)] * (n_envs * n_nodes),
            mass=[1.0] * (n_envs * n_nodes),
            radius=[0.001] * (n_envs * n_nodes),
        )
        en_np = self.ancf_model.elem_nodes.numpy()
        tris_1 = np.empty((len(en_np) * 2, 3), dtype=np.int32)
        tris_1[0::2] = en_np[:, [0, 1, 2]]
        tris_1[1::2] = en_np[:, [0, 2, 3]]
        tris_all = np.concatenate([tris_1 + e * n_nodes for e in range(n_envs)], axis=0)
        builder.add_triangles(
            i=tris_all[:, 0].tolist(),
            j=tris_all[:, 1].tolist(),
            k=tris_all[:, 2].tolist(),
        )

        # ── Kinematic spindle bodies — rendered as solid MuJoCo-style geoms ──
        # One body per tire: spokes + axle capsules rendered by viewer.log_state.
        # is_kinematic=True means Newton won't simulate them; body_q is written
        # from Python hub tracking each frame before log_state.
        _vis_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False,
            has_particle_collision=False,
        )
        _SPOKE_RADIUS = 0.018  # capsule radius [m]
        _SPOKE_ANGLES = (0.0, 120.0, 240.0)  # degrees, in the Y-Z plane
        _half_w = float(_WIDTH * 0.5)
        self._spindle_body_idx = []
        for e in range(n_envs):
            body = builder.add_body(
                xform=wp.transform((self._hub_x[e], hub_y_init, 0.0), wp.quat_identity()),
                mass=self._hub_mass,
                is_kinematic=True,
                label=f"spindle_{e}",
            )
            builder.add_joint_free(body)
            # Spoke and axle capsule geoms removed — bead rings + spokes
            # rendered via log_lines (orange/yellow) replace them.
            self._spindle_body_idx.append(body)

        self.model = builder.finalize(device=device)
        self.state_0 = self.model.state()

        # ── ANCF solver ───────────────────────────────────────────────────
        kn = float(getattr(args, "kn", 2e4))
        kd = float(getattr(args, "kd", 20.0))
        mu = float(getattr(args, "mu", 0.9))
        self.solver = SolverANCFShell(
            model=self.model,
            ancf_model=self.ancf_model,
            ground_z=0.0,
            kn=kn,
            kd=kd,
            mu=mu,
            v_reg=1e-3,
            nr_max_iter=int(getattr(args, "nr_iters", 5)),
            pcg_max_iter=int(getattr(args, "pcg_iters", 20)),
            n_envs=n_envs,
            rim_radius=_R_INNER if self._rim_kn > 0.0 else 0.0,
            rim_kn=self._rim_kn,
            rim_kd=self._rim_kd,
        )
        self.solver.node_x.assign(world_x)

        # Seed initial velocities from rim rotation
        d0_np = self.ancf_model.node_D0.numpy().astype(np.float32)
        xd_init = np.zeros((n_envs * n_nodes, 3), dtype=np.float32)
        Dd_init = np.zeros((n_envs * n_nodes, 3), dtype=np.float32)
        for e in range(n_envs):
            omega_e = self._rim_omega[e]
            if abs(omega_e) < 1e-12:
                continue
            s = slice(e * n_nodes, (e + 1) * n_nodes)
            wx = world_x[s]
            xd_init[s, 1] = -omega_e * wx[:, 2]
            xd_init[s, 2] = omega_e * (wx[:, 1] - self._hub_y[e])
            Dd_init[s, 1] = -omega_e * d0_np[:, 2]
            Dd_init[s, 2] = omega_e * d0_np[:, 1]
        self.solver.node_xd.assign(xd_init)
        self.solver.node_Dd.assign(Dd_init)

        # Save initial state for restore after graph-capture warm-up
        self._ancf_x0_world = wp.clone(self.solver.node_x)
        self._ancf_xd0 = wp.clone(self.solver.node_xd)
        self._ancf_d0_init = wp.clone(self.solver.node_D)
        self._ancf_Dd0 = wp.clone(self.solver.node_Dd)

        # ── Mutable hub/rim state as N-element wp.arrays ──────────────────
        # Written each frame before graph replay; read by kernels inside graph.
        def _f32(lst):
            return np.array(lst, dtype=np.float32)

        self._hub_x_wp = wp.array(_f32(self._hub_x), dtype=float, device=device)
        self._hub_y_wp = wp.array(_f32(self._hub_y), dtype=float, device=device)
        self._hub_vy_wp = wp.array(_f32(self._hub_vy), dtype=float, device=device)
        self._hub_z_wp = wp.zeros(n_envs, dtype=float, device=device)
        self._rim_phi_wp = wp.zeros(n_envs, dtype=float, device=device)
        self._rim_omega_wp = wp.array(_f32(self._rim_omega), dtype=float, device=device)
        self._tau_buf = wp.zeros(n_envs, dtype=float, device=device)
        # Unified 6-DOF wrench buffer — replaces _tau_buf (spin only) and _fy_buf (vertical only).
        # [tau_x, tau_y, tau_z, fx, fy, fz] in ANCF Y-up per env.
        # Same pattern as SolverANCFShellRigid._xfrc_stg_per_tire in ancf_rigid_mujoco_tires.
        self._wrench_buf = wp.zeros(n_envs, dtype=wp.spatial_vector, device=device)
        # Spindle body indices as a GPU array for _update_spindle_body_q kernel
        self._spindle_body_idx_wp = wp.array(
            np.array(self._spindle_body_idx, dtype=np.int32), dtype=wp.int32, device=device
        )

        # ── Loaded-mode GS coupling buffers (requires hub wp.arrays above) ──
        if self._mode == "loaded":
            n_total = n_envs * n_nodes
            self._total_mass = self._hub_mass + self._corner_mass
            self._g_load = self._total_mass * _G
            self._ancf_pack_buf = wp.zeros(30 * n_total, dtype=float, device=device)
            self._fy_buf = wp.zeros(n_envs, dtype=float, device=device)
            self._hub_y_n_wp = wp.clone(self._hub_y_wp)
            self._hub_vy_n_wp = wp.clone(self._hub_vy_wp)
            self._rim_phi_n_wp = wp.clone(self._rim_phi_wp)
            self._hub_z_n_wp = wp.clone(self._hub_z_wp)

        # ── CTIS pressure per tire ────────────────────────────────────────
        self._build_pressure = [float(cfg.get("pressure", _BUILD_PRESSURE)) for cfg in tire_cfgs]
        self._max_gauge = [float(cfg.get("max-gauge-pressure", _MAX_GAUGE)) for cfg in tire_cfgs]
        self._gauge_rate = [float(cfg.get("gauge-rate", _GAUGE_RATE)) for cfg in tire_cfgs]
        # RPM slew limit [rpm/frame].  The slider sets a *target*; the commanded
        # speed walks toward it so prescribed bead nodes never take a velocity
        # step (which would shear against the still-at-rest crown).
        self._rpm_rate = [float(cfg.get("rpm-rate", _RPM_RATE)) for cfg in tire_cfgs]
        # Start already at target so a non-zero "rpm" in JSON behaves as before.
        self._current_rpm = list(self._target_rpm)
        self._target_pressure = list(self._build_pressure)
        self._current_pressure = [0.0] * n_envs
        self.solver.set_cavity(self._build_pressure, self._build_pressure)

        # Tire self-weight tare: total FEM tire mass × g.
        # Subtracted from fy_bead in _integrate_hub_dof so pre-contact wrench ≈ 0.
        m_lump = self.solver.lumped_mass.numpy()
        m_tire = float(m_lump[::6].sum())  # position-DOF mass summed over all nodes
        self._tare_fy = m_tire * _G  # upward tare [N]

        # Bead masses for free-mode torque integration (legacy, kept for _tau_buf path)
        bead_m = m_lump[all_bead * 6].astype(np.float64).clip(1e-9)
        self._bead_mass = wp.array(bead_m.astype(np.float32), dtype=float, device=device)

        # Cache bead indices on CPU once — the array never changes.
        self._bead_idx_np = all_bead

        # ── Initial bead prescribe + Dirichlet setup ──────────────────────
        self._launch_prescribe(device)
        all_bead_global = np.concatenate([all_bead + e * n_nodes for e in range(n_envs)])
        self.solver.set_dirichlet_nodes(all_bead_global)

        # Bind hub arrays to solver for rim contact (must be before capture_graph)
        if self._rim_kn > 0.0:
            self.solver.set_rim(self._hub_y_wp, self._hub_z_wp)

        # Inner graph (fills lumped_mass_scaled), then outer CUDA graph
        self.solver.capture_graph(self.sim_dt)
        self._device = device
        self._build_graph()

        wp.copy(self.state_0.particle_q, self.solver.node_x)

        if viewer is not None:
            viewer.set_model(self.model)
            mid_x = float(np.mean(self._hub_x))
            viewer.set_camera(
                pos=wp.vec3(mid_x, hub_y_init + 0.3, 4.5),
                pitch=-10.0,
                yaw=-90.0,
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _launch_prescribe(self, device):
        """Prescribe all bead nodes (all envs) from current Python hub state."""

        def _f32(lst):
            return np.array(lst, dtype=np.float32)

        self._hub_x_wp.assign(_f32(self._hub_x))
        self._hub_y_wp.assign(_f32(self._hub_y))
        self._hub_vy_wp.assign(_f32(self._hub_vy))
        self._hub_z_wp.assign(_f32(self._hub_z))
        self._rim_phi_wp.assign(_f32(self._rim_phi))
        self._rim_omega_wp.assign(_f32(self._rim_omega))
        wp.launch(
            _prescribe_bead_batched,
            dim=self._n_envs * self._n_bead,
            inputs=[
                self.solver.node_x,
                self.solver.node_xd,
                self.solver.node_xdd,
                self.solver.node_D,
                self.solver.node_Dd,
                self.solver.node_Ddd,
                self._bead_idx,
                self._bead_rest,
                self._bead_D0,
                self._hub_x_wp,
                self._hub_y_wp,
                self._hub_vy_wp,
                self._hub_z_wp,
                self._rim_phi_wp,
                self._rim_omega_wp,
                _R_OUTER,
                self._n_bead,
                self._n_nodes,
            ],
            device=device,
        )

    # ── Graph-capturable substep loop ─────────────────────────────────────

    def simulate(self):
        """Run sim_substeps of ANCF + batched prescribe.

        All modes use solver.graph_step() (pre-captured ANCF inner graph).
        Called directly from step() each frame — no outer CUDA graph.
        """
        dev = self._device
        dt = self.sim_dt
        N = self._n_envs
        n_bead = self._n_bead
        n_nodes = self._n_nodes

        prescribe_inputs = [
            self.solver.node_x,
            self.solver.node_xd,
            self.solver.node_xdd,
            self.solver.node_D,
            self.solver.node_Dd,
            self.solver.node_Ddd,
            self._bead_idx,
            self._bead_rest,
            self._bead_D0,
            self._hub_x_wp,
            self._hub_y_wp,
            self._hub_vy_wp,
            self._hub_z_wp,
            self._rim_phi_wp,
            self._rim_omega_wp,
            _R_OUTER,
            n_bead,
            n_nodes,
        ]

        if self._mode == "loaded":
            # Mode C: GS hub-Y coupling loop.  Uses graph_step() like modes A/B.
            gs_iters = self._gs_iters_hub
            n_total = N * n_nodes

            for _ in range(self.sim_substeps):
                if gs_iters > 1:
                    # Save t_n state into double buffer
                    wp.launch(
                        _pack_ancf_kernel,
                        dim=n_total,
                        inputs=[
                            self.solver.node_x,
                            self.solver.node_xd,
                            self.solver.node_xdd,
                            self.solver.node_D,
                            self.solver.node_Dd,
                            self.solver.node_Ddd,
                            self.solver.global_f_int,
                            self.solver.global_f_int0,
                            self._ancf_pack_buf,
                            n_total,
                        ],
                        device=dev,
                    )
                    wp.copy(self._hub_y_n_wp, self._hub_y_wp)
                    wp.copy(self._hub_vy_n_wp, self._hub_vy_wp)
                    wp.copy(self._rim_phi_n_wp, self._rim_phi_wp)
                    wp.copy(self._hub_z_n_wp, self._hub_z_wp)

                for k in range(gs_iters):
                    if k > 0:
                        # Restore to t_n
                        wp.launch(
                            _unpack_ancf_kernel,
                            dim=n_total,
                            inputs=[
                                self._ancf_pack_buf,
                                self.solver.node_x,
                                self.solver.node_xd,
                                self.solver.node_xdd,
                                self.solver.node_D,
                                self.solver.node_Dd,
                                self.solver.node_Ddd,
                                self.solver.global_f_int,
                                self.solver.global_f_int0,
                                n_total,
                            ],
                            device=dev,
                        )
                        wp.copy(self._hub_y_wp, self._hub_y_n_wp)
                        wp.copy(self._hub_vy_wp, self._hub_vy_n_wp)
                        wp.copy(self._rim_phi_wp, self._rim_phi_n_wp)
                        wp.copy(self._hub_z_wp, self._hub_z_n_wp)

                    wp.launch(_prescribe_bead_batched, dim=N * n_bead, inputs=prescribe_inputs, device=dev)
                    wp.launch(
                        _advance_phi_and_z_batched,
                        dim=N,
                        inputs=[
                            self._rim_phi_wp,
                            self._rim_omega_wp,
                            self._hub_z_wp,
                            dt,
                            _R_OUTER,
                        ],
                        device=dev,
                    )
                    self.solver.graph_step()
                    wp.launch(_prescribe_bead_batched, dim=N * n_bead, inputs=prescribe_inputs, device=dev)
                    # Unified wrench → Fy (vertical) + tau_x (spin)
                    wp.launch(
                        _accum_bead_wrench_kinematic,
                        dim=N * n_bead,
                        inputs=[
                            self.solver.global_f_int,
                            self.solver.node_x,
                            self._bead_idx,
                            self._hub_x_wp,
                            self._hub_y_wp,
                            self._hub_z_wp,
                            self._wrench_buf,
                            n_bead,
                            n_nodes,
                        ],
                        device=dev,
                    )
                    wp.launch(
                        _integrate_hub_dof,
                        dim=N,
                        inputs=[
                            self._hub_y_wp,
                            self._hub_vy_wp,
                            self._rim_omega_wp,
                            self._wrench_buf,
                            self._total_mass,
                            self._I_rim,
                            self._g_load,
                            self._tare_fy,
                            dt,
                            1,
                            0,  # update_vy=1 (Fy), update_om=0 (spin prescribed in driven/loaded)
                        ],
                        device=dev,
                    )
            return

        for _ in range(self.sim_substeps):
            # Pre-step: prescribe beads at phi_n
            wp.launch(_prescribe_bead_batched, dim=N * n_bead, inputs=prescribe_inputs, device=dev)

            # Advance phi_n → phi_{n+1} and hub_z
            wp.launch(
                _advance_phi_and_z_batched,
                dim=N,
                inputs=[
                    self._rim_phi_wp,
                    self._rim_omega_wp,
                    self._hub_z_wp,
                    dt,
                    _R_OUTER,
                ],
                device=dev,
            )

            self.solver.graph_step()

            # Post-step: snap beads to exact phi_{n+1} target
            wp.launch(_prescribe_bead_batched, dim=N * n_bead, inputs=prescribe_inputs, device=dev)

            # All modes: unified 6-DOF wrench from global_f_int (exact Lagrange λ).
            # Replaces _accum_bead_torque_batched (position residual) and _accum_bead_fy_batched.
            # free  → update_vy=1 (Fy) + update_om=1 (tau_x spin)
            # driven/loaded → update_vy=1 (Fy) + update_om=0 (spin prescribed)
            _update_vy = 1
            _update_om = 1 if self._mode == "free" else 0
            _g_load = self._g_load if self._mode == "loaded" else self._hub_mass * _G
            wp.launch(
                _accum_bead_wrench_kinematic,
                dim=N * n_bead,
                inputs=[
                    self.solver.global_f_int,
                    self.solver.node_x,
                    self._bead_idx,
                    self._hub_x_wp,
                    self._hub_y_wp,
                    self._hub_z_wp,
                    self._wrench_buf,
                    n_bead,
                    n_nodes,
                ],
                device=dev,
            )
            wp.launch(
                _integrate_hub_dof,
                dim=N,
                inputs=[
                    self._hub_y_wp,
                    self._hub_vy_wp,
                    self._rim_omega_wp,
                    self._wrench_buf,
                    self._hub_mass,
                    self._I_rim,
                    _g_load,
                    self._tare_fy,
                    dt,
                    _update_vy,
                    _update_om,
                ],
                device=dev,
            )

    def _restore_initial_state(self, dev):
        """Restore ANCF + hub state to t=0 (used after warmup in _build_graph)."""
        wp.copy(self.solver.node_x, self._ancf_x0_world)
        wp.copy(self.solver.node_xd, self._ancf_xd0)
        self.solver.node_xdd.zero_()
        wp.copy(self.solver.node_D, self._ancf_d0_init)
        wp.copy(self.solver.node_Dd, self._ancf_Dd0)
        self.solver.node_Ddd.zero_()
        self.solver.global_f_int.zero_()
        self.solver.global_f_int0.zero_()
        self._launch_prescribe(dev)

    def _build_graph(self):
        """Warm-up simulate() for Warp JIT, then restore initial state.

        All modes use a Python substep loop with solver.graph_step() (the
        pre-captured ANCF inner graph, captured in __init__ via capture_graph()).
        No outer CUDA graph is used: replaying a flat 3500-node graph is slower
        than 10 × inner-graph launches at the small N and node counts here.
        """
        dev = self._device

        self.simulate()
        wp.synchronize_device(dev)

        self._restore_initial_state(dev)

        if self._mode == "loaded":
            N = self._n_envs
            init_y = np.full(N, self._hub_y_init, dtype=np.float32)
            zeros_n = np.zeros(N, dtype=np.float32)
            self._hub_y_wp.assign(init_y)
            self._hub_y = list(init_y)
            self._hub_vy_wp.assign(zeros_n)
            self._hub_vy = [0.0] * N
            self._hub_z_wp.assign(zeros_n)
            self._hub_z = [0.0] * N
            self._rim_phi_wp.assign(zeros_n)
            self._rim_phi = [0.0] * N

        wp.synchronize_device(dev)
        self.graph = None

    # ── Simulation step ───────────────────────────────────────────────────

    _dbg_frame = 0

    def step(self):
        frame = Example._dbg_frame
        Example._dbg_frame += 1
        N = self._n_envs

        # Hub gravity drop (modes A/B only — mode C hub Y is driven by ANCF forces)
        if self._mode != "loaded":
            for e in range(N):
                if self._hub_y[e] > self._hub_rest:
                    self._hub_vy[e] -= _G * self.frame_dt
                    self._hub_y[e] += self._hub_vy[e] * self.frame_dt
                    if self._hub_y[e] <= self._hub_rest:
                        self._hub_y[e] = self._hub_rest
                        self._hub_vy[e] = 0.0
            self._hub_y_wp.assign(np.array(self._hub_y, dtype=np.float32))
            self._hub_vy_wp.assign(np.array(self._hub_vy, dtype=np.float32))

        # Gauge pressure ramp toward target
        for e in range(N):
            d = self._target_pressure[e] - self._current_pressure[e]
            r = self._gauge_rate[e]
            if abs(d) <= r:
                self._current_pressure[e] = self._target_pressure[e]
            else:
                self._current_pressure[e] += r if d > 0.0 else -r
        nominal = [self._build_pressure[e] + self._current_pressure[e] for e in range(N)]
        self.solver.set_cavity(nominal, self._build_pressure)

        # Driven / loaded mode: ramp slider RPM → omega → wp.array.
        # Mirrors the gauge-pressure ramp above: the slider is a target, not an
        # instantaneous command.  A step in omega would step the prescribed bead
        # velocity while the crown is still at rest, shearing the sidewall.
        if self._mode in ("driven", "loaded"):
            for e in range(N):
                d = self._target_rpm[e] - self._current_rpm[e]
                r = self._rpm_rate[e]
                if abs(d) <= r:
                    self._current_rpm[e] = self._target_rpm[e]
                else:
                    self._current_rpm[e] += r if d > 0.0 else -r
                self._rim_omega[e] = self._current_rpm[e] * (2.0 * np.pi / 60.0)
            self._rim_omega_wp.assign(np.array(self._rim_omega, dtype=np.float32))

        # Run all substeps
        _t0 = time.perf_counter()
        self.simulate()
        wp.synchronize_device(self._device)
        _t1 = time.perf_counter()
        if frame % 30 == 0:
            print(f"  [perf] simulate={(_t1 - _t0) * 1000:.1f}ms  frame={frame}")

        # Sync Python hub/rim state from GPU.
        # Loaded mode: hub Y changed inside simulate() GS loop — must read back.
        # Free mode: omega changed inside the graph — read back for diagnostics.
        # Driven mode: omega is constant — approximate advance is exact.
        if self._mode == "loaded":
            self._hub_y = list(self._hub_y_wp.numpy())
            self._hub_vy = list(self._hub_vy_wp.numpy())
            self._hub_z = list(self._hub_z_wp.numpy())
            self._rim_phi = list(self._rim_phi_wp.numpy())
        else:
            if self._mode == "free":
                omega_np = self._rim_omega_wp.numpy()
                self._rim_omega = [float(omega_np[e]) for e in range(N)]
            for e in range(N):
                o = self._rim_omega[e]
                self._hub_z[e] += o * _R_OUTER * self.frame_dt
                self._rim_phi[e] += o * self.frame_dt

        wp.copy(self.state_0.particle_q, self.solver.node_x)
        self.sim_time += self.frame_dt

        # Diagnostics (every 30 frames, env-0 only) — full node D2H is expensive.
        if frame % 30 == 0:
            x_np = self.solver.node_x.numpy()
            xd_np = self.solver.node_xd.numpy()
            n = self._n_nodes
            x0 = x_np[:n]
            xd0 = xd_np[:n]
            nan = bool(np.any(np.isnan(x0)))
            vmax = float(np.max(np.linalg.norm(xd0, axis=1)))
            bi = self._bead_idx_np
            bead_r = float(
                np.mean(np.linalg.norm(x0[bi] - np.array([self._hub_x[0], self._hub_y[0], self._hub_z[0]]), axis=1))
            )
            y_vals = x0[:, 1]
            vz_all = xd0[:, 2]
            in_contact = y_vals < 0.0
            n_contact = int(in_contact.sum())
            depth_max = float(-y_vals[in_contact].min()) if n_contact > 0 else 0.0
            v_contact_z = float(vz_all[in_contact].mean()) if n_contact > 0 else float("nan")
            v_expected = -self._rim_omega[0] * _R_OUTER
            hub_vz = self._rim_omega[0] * _R_OUTER
            if n_contact > 0:
                contact_str = (
                    f"  contact: {n_contact:3d}  depth={depth_max * 1e3:.2f} mm"
                    f"  v_slip_z={v_contact_z:+.3f} (pure={v_expected:+.3f})"
                )
            else:
                contact_str = "  no contact"
            prefix = f"[{frame:4d}]  hub_y={self._hub_y[0]:.4f}  hub_z={self._hub_z[0]:+.3f}"
            if self._mode == "loaded":
                prefix += f"  hub_vy={self._hub_vy[0]:+.3f}"
            print(
                f"{prefix}  hub_vz={hub_vz:+.3f}"
                f"  gauge={self._current_pressure[0]:.0f} Pa"
                + ("  *** NaN ***" if nan else "")
                + f"  vmax={vmax:.2f} m/s  bead_r={bead_r:.4f}"
                + contact_str
            )

    # ── GUI ───────────────────────────────────────────────────────────────

    def gui(self, ui):
        labels = {
            "driven": "Mode A — driven",
            "free": "Mode B — free revolute",
            "loaded": f"Mode C — loaded GS  corner={self._corner_mass:.0f} kg  gs_iters={self._gs_iters_hub}",
        }
        label = labels.get(self._mode, self._mode)
        ui.text(label)
        ui.separator()

        ui.text("CTIS gauge pressure")
        for e, name in enumerate(self._tire_names):
            p_min = 0.0
            p_max = self._build_pressure[e] + self._max_gauge[e]
            changed, val = ui.slider_float(f"{name} [Pa]##{e}", self._target_pressure[e], p_min, p_max)
            if changed:
                self._target_pressure[e] = float(val)
            ui.text(f"  {self._current_pressure[e]:8.0f} Pa  ({self._current_pressure[e] / 6894.76:.1f} psi)")

        ui.separator()
        if self._mode == "driven":
            ui.text("RPM per tire")
            for e, name in enumerate(self._tire_names):
                changed, val = ui.slider_float(f"{name}##{e}", self._target_rpm[e], -300.0, 300.0)
                if changed:
                    self._target_rpm[e] = float(val)
        else:
            for e in range(self._n_envs):
                rpm = self._rim_omega[e] * 60.0 / (2.0 * np.pi)
                ui.text(f"  {self._tire_names[e]}  {rpm:+7.1f} RPM")

        ui.separator()
        ui.text(f"hub Y = {self._hub_y[0]:.3f} m")

    # ── Render ────────────────────────────────────────────────────────────

    def render(self):
        if self.viewer is None:
            return

        # Build rigid rim disc geometry (circles + axial struts)
        if any(self._rim_visible):
            wp.launch(
                _build_rim_disc_batched,
                dim=self._n_envs * self._rim_segs_per_env,
                inputs=[
                    self._hub_x_wp,
                    self._hub_y_wp,
                    self._hub_z_wp,
                    self._rim_disc_starts,
                    self._rim_disc_ends,
                    float(_R_INNER),
                    float(_WIDTH * 0.5),
                    self._rim_flange,
                    self._rim_n_pts,
                    self._rim_n_struts,
                    self._rim_segs_per_env,
                ],
                device=self._device,
            )

        # ── Update kinematic spindle body_q from live hub pos + rim angle ──
        # Pure GPU kernel — no CPU round-trip.  _hub_y_wp is written by step()
        # before graph replay so it already reflects the current drop position.
        if self._spindle_body_idx:
            wp.launch(
                _update_spindle_body_q,
                dim=self._n_envs,
                inputs=[
                    self._hub_x_wp,
                    self._hub_y_wp,
                    self._hub_z_wp,
                    self._rim_phi_wp,
                    self._spindle_body_idx_wp,
                    self.state_0.body_q,
                ],
                device=self._device,
            )

        # ── Bead rings (orange) ───────────────────────────────────────────
        wp.launch(
            _gather_bead_positions,
            dim=self._n_envs * self._n_bead,
            inputs=[self.solver.node_x, self._bead_idx, self._bead_pos_yup, self._n_bead, self._n_nodes],
            device=self._device,
        )
        wp.launch(
            _build_ring_lines,
            dim=self._n_ring_segs,
            inputs=[self._bead_pos_yup, self._ring_seg_s, self._ring_seg_e, self._ring_line_s, self._ring_line_e],
            device=self._device,
        )

        # ── Spokes: hub → bead nodes (yellow) ────────────────────────────
        wp.launch(
            _build_spoke_lines_batched,
            dim=self._n_envs * self._n_bead,
            inputs=[
                self.solver.node_x,
                self._bead_idx,
                self._hub_x_wp,
                self._hub_y_wp,
                self._hub_z_wp,
                self._spoke_starts,
                self._spoke_ends,
                self._n_bead,
                self._n_nodes,
            ],
            device=self._device,
        )

        # ── Contact spikes (cyan) — GPU only, no FPS cost ─────────────────
        wp.launch(
            _gather_contact_spikes_yup,
            dim=self._n_envs * self._n_nodes,
            inputs=[self.solver.node_x, 0.0, self._contact_vis_scale, self._contact_line_s, self._contact_line_e],
            device=self._device,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Rim disc lines and capsule geoms removed — bead rings + spokes replace them
        self.viewer.log_lines("bead_rings", self._ring_line_s, self._ring_line_e, (1.0, 0.45, 0.0))  # orange
        self.viewer.log_lines("bead_spokes", self._spoke_starts, self._spoke_ends, (1.0, 0.90, 0.1))  # yellow
        self.viewer.log_lines("contact_spikes", self._contact_line_s, self._contact_line_e, (0.0, 1.0, 1.0))  # cyan
        self.viewer.end_frame()

    # ── Tests ─────────────────────────────────────────────────────────────

    def test_post_step(self):
        """Per-step NaN guard — catches instability before it cascades."""
        x_np = self.solver.node_x.numpy()
        assert not np.any(np.isnan(x_np)), "NaN in node positions during simulation"

    def test_final(self):
        x_np = self.solver.node_x.numpy()
        assert not np.any(np.isnan(x_np)), "NaN in node positions"
        assert not np.any(np.isinf(x_np)), "Inf in node positions"

        # Sync exact phi and hub_z from GPU — Python-tracked values are
        # approximate in free mode (omega varies within the frame).
        phi_gpu = self._rim_phi_wp.numpy()
        hub_z_gpu = self._hub_z_wp.numpy()

        bead_idx = self._bead_idx_np
        r = self._bead_rest_np
        for e in range(self._n_envs):
            x_e = x_np[e * self._n_nodes : (e + 1) * self._n_nodes]
            bead_actual = x_e[bead_idx]
            cp = float(np.cos(phi_gpu[e]))
            sp = float(np.sin(phi_gpu[e]))
            bead_tgt = np.stack(
                [
                    self._hub_x[e] + r[:, 0],
                    self._hub_y[e] + r[:, 1] * cp - r[:, 2] * sp,
                    float(hub_z_gpu[e]) + r[:, 1] * sp + r[:, 2] * cp,
                ],
                axis=1,
            )
            err = float(np.max(np.linalg.norm(bead_actual - bead_tgt, axis=1)))
            assert err < 1e-3, f"Bead constraint error env={e} ({self._tire_names[e]}): {err * 1e3:.2f} mm > 1 mm"

        # K-proportional Rayleigh damping wired from JSON → elem_mat[:,10]
        em = self.ancf_model.elem_mat.numpy()
        alpha_col = em[:, 10]
        assert np.all(np.isfinite(alpha_col)), "elem_mat alpha_damp contains non-finite values"
        assert np.all(alpha_col > 0.0), "elem_mat alpha_damp must be positive"
        # All elements in a given tire share the same isotropic material → uniform.
        ne_per = em.shape[0] // self._n_envs
        for e in range(self._n_envs):
            a = float(alpha_col[e * ne_per])
            assert abs(a - 0.15) < 0.01, f"alpha_damp env={e} ({self._tire_names[e]}): {a:.4f}, expected ≈0.15"

        # Velocity bound — max node speed is a stability proxy.
        xd_np = self.solver.node_xd.numpy().reshape(-1, 3)
        max_v = float(np.max(np.linalg.norm(xd_np, axis=1)))
        assert max_v < 50.0, f"Excessive node speed {max_v:.1f} m/s — possible instability"

        # Inflation check — mean radial distance from hub must exceed R_inner,
        # confirming the tire is pressurised and has not collapsed.
        for e in range(self._n_envs):
            x_e = x_np[e * self._n_nodes : (e + 1) * self._n_nodes]
            hub_c = np.array([self._hub_x[e], self._hub_y[e], float(hub_z_gpu[e])], dtype=np.float32)
            radii = np.linalg.norm(x_e - hub_c, axis=1)
            r_mean = float(np.mean(radii))
            assert r_mean > _R_INNER, (
                f"Tire {self._tire_names[e]}: mean radius {r_mean:.4f} m ≤ "
                f"R_inner={_R_INNER:.4f} m — tire may have collapsed"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--mode",
            choices=["driven", "free", "loaded"],
            default="driven",
            help="driven = prescribed RPM;  free = free revolute;  "
            "loaded = hub Y driven by ANCF bead forces + corner load (GS coupling).",
        )
        parser.add_argument(
            "--n-circ", type=int, default=20, help="Circumferential elements (overridden by --shell-tires JSON)."
        )
        parser.add_argument("--rpm", type=float, default=0.0, help="Initial rim speed [rev/min] (GUI-controlled).")
        parser.add_argument("--drop-height", type=float, default=0.5, help="Hub drop height above ground-tangency [m].")
        parser.add_argument(
            "--hub-mass",
            type=float,
            default=74.2,
            help="Hub translational mass [kg] (overridden by rim.hub-mass in JSON).",
        )
        parser.add_argument(
            "--inertia",
            type=float,
            default=8.0 * _R_INNER**2,
            help="Rim moment of inertia [kg·m²] (overridden by rim.inertia in JSON).",
        )
        parser.add_argument("--substeps", type=int, default=10, help="Substeps per frame.  dt = 1/60/substeps.")
        parser.add_argument("--kn", type=float, default=2e4, help="Ground contact normal stiffness [N/m].")
        parser.add_argument("--kd", type=float, default=20.0, help="Ground contact normal damping [N·s/m].")
        parser.add_argument("--mu", type=float, default=0.9, help="Ground contact friction coefficient.")
        parser.add_argument("--nr-iters", type=int, default=3, help="Newton-Raphson iterations per substep.")
        parser.add_argument("--pcg-iters", type=int, default=20, help="PCG iterations per NR step.")
        parser.add_argument(
            "--shell-tires",
            type=str,
            default=None,
            help="JSON array of per-tire configs (position, material, rpm, pressure, …).",
        )
        parser.add_argument(
            "--corner-mass",
            type=float,
            default=0.0,
            help="Vehicle corner load [kg] added to hub mass for mode 'loaded'.",
        )
        parser.add_argument(
            "--gs-iters",
            type=int,
            default=2,
            help="GS iterations per substep for mode 'loaded'.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
