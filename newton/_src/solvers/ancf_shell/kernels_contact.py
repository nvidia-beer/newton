# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Ground-plane penalty contact for ANCF shell nodes.

Flat rigid plane at height ``ground_z`` (Newton is Y-up, so the plane is y = ground_z).
Each node whose position penetrates the plane receives a normal penalty force and a
regularised (tanh) Coulomb friction force.

The contact iteration matrix adds normal stiffness/damping and a positive tangential
majorizer to K_eff. Velocity terms use the HHT factor c_v. The physical force law
remains regularised Coulomb friction (see :func:`_contact_tangent_diag`).
Heightfield / soil terrain contact lives in :mod:`terrain_scm`.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def _contact_tangent_diag(
    kn: float,
    kd: float,
    mu: float,
    v_reg: float,
    fn: float,
    vn: float,
    vt_x: float,
    vt_z: float,
    vt_mag: float,
    g: float,  # tanh(|vt|/v_reg)
    c_v: float,  # gamma/(beta*dt): d(velocity)/d(displacement) for the HHT corrector
) -> wp.vec3:
    """Positive contact iteration diagonal for frozen normal load, (x, y, z).

    The friction derivative along sliding tends to zero; using that slope with
    a fixed Newton budget can overshoot sticking and reverse the slip repeatedly.
    The lagged slope tanh(s)/s majorizes sech(s)^2, retains the correct sticking
    limit, and leaves the normal solve and residual forces unchanged.
    """
    k_yy = kn
    if vn > 0.0:
        k_yy = k_yy + kd * c_v
    k_t = mu * fn * c_v * g / vt_mag
    return wp.vec3(k_t, k_yy, k_t)


@wp.kernel
def apply_ground_contact(
    node_x: wp.array[wp.vec3],  # current positions
    node_xd: wp.array[wp.vec3],  # current velocities (for friction)
    ground_z: float,
    kn: float,  # normal penalty stiffness  [N/m]
    kd: float,  # normal damping            [N·s/m]
    mu: float,  # Coulomb friction coefficient
    v_reg: float,  # friction regularisation velocity [m/s]
    c_v: float,  # gamma/(beta*dt) — velocity->displacement factor for the tangent
    # output: add contact forces to global_f
    global_f: wp.array[float],  # (n_nodes * 6,) flat DOF vector
    # output: diagonal contact stiffness (n_nodes * 6,) — only pos DOFs modified
    K_contact_diag: wp.array[float],
):
    """One thread per node.  Adds penalty normal + Coulomb tangential forces."""
    i = wp.tid()

    xi = node_x[i]
    # Newton is Y-up: ground_z is actually the ground level on the Y axis
    pen = ground_z - xi[1]  # penetration depth (positive = inside ground)

    if pen <= 0.0:
        return

    vi = node_xd[i]

    # Normal force (penalty + damping) in Y direction
    vn = -vi[1]  # normal velocity (positive = moving into ground)
    fn = kn * pen + kd * wp.max(vn, 0.0)

    # Tangential velocity in XZ plane
    vt_x = vi[0]
    vt_z = vi[2]
    vt_mag = wp.sqrt(vt_x * vt_x + vt_z * vt_z + 1.0e-12)
    g = wp.tanh(vt_mag / v_reg)
    slip_scale = g / vt_mag

    # Coulomb friction (opposes tangential motion)
    ft_x = -mu * fn * vt_x * slip_scale
    ft_z = -mu * fn * vt_z * slip_scale

    base = i * 6
    wp.atomic_add(global_f, base + 0, ft_x)  # fx (tangential)
    wp.atomic_add(global_f, base + 1, fn)  # fy (normal, outward = +y)
    wp.atomic_add(global_f, base + 2, ft_z)  # fz (tangential)

    k = _contact_tangent_diag(kn, kd, mu, v_reg, fn, vn, vt_x, vt_z, vt_mag, g, c_v)
    wp.atomic_add(K_contact_diag, base + 0, k[0])
    wp.atomic_add(K_contact_diag, base + 1, k[1])
    wp.atomic_add(K_contact_diag, base + 2, k[2])


@wp.kernel
def zero_contact_diag(K_contact_diag: wp.array[float]):
    """Clear the contact diagonal before each step."""
    i = wp.tid()
    K_contact_diag[i] = float(0.0)


@wp.kernel
def apply_ground_contact_batched(
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    ground_z: float,
    kn: float,
    kd: float,
    mu: float,
    v_reg: float,
    c_v: float,  # gamma/(beta*dt) — velocity->displacement factor for the tangent
    global_f: wp.array[float],
    K_contact_diag: wp.array[float],
    n_nodes: int,
):
    """dim = N*n_nodes.  env = tid // n_nodes;  i = tid % n_nodes.

    node_x/xd are flat [N*n_nodes]; global_f/K_contact_diag are flat [N*n_dof].
    """
    tid = wp.tid()
    i = tid % n_nodes
    dof_base = (tid // n_nodes) * n_nodes * 6 + i * 6

    xi = node_x[tid]
    pen = ground_z - xi[1]
    if pen <= 0.0:
        return

    vi = node_xd[tid]
    vn = -vi[1]
    fn = kn * pen + kd * wp.max(vn, 0.0)

    vt_x = vi[0]
    vt_z = vi[2]
    vt_mag = wp.sqrt(vt_x * vt_x + vt_z * vt_z + 1.0e-12)
    g = wp.tanh(vt_mag / v_reg)
    slip_scale = g / vt_mag
    ft_x = -mu * fn * vt_x * slip_scale
    ft_z = -mu * fn * vt_z * slip_scale

    wp.atomic_add(global_f, dof_base + 0, ft_x)
    wp.atomic_add(global_f, dof_base + 1, fn)
    wp.atomic_add(global_f, dof_base + 2, ft_z)

    k = _contact_tangent_diag(kn, kd, mu, v_reg, fn, vn, vt_x, vt_z, vt_mag, g, c_v)
    wp.atomic_add(K_contact_diag, dof_base + 0, k[0])
    wp.atomic_add(K_contact_diag, dof_base + 1, k[1])
    wp.atomic_add(K_contact_diag, dof_base + 2, k[2])
