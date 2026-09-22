# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Ground-plane penalty contact for ANCF shell nodes.

Flat rigid plane at height ``ground_z`` (Newton is Y-up, so the plane is y = ground_z).
Each node whose position penetrates the plane receives a normal penalty force and a
regularised (tanh) Coulomb friction force.

The contact tangent is added to K_eff as a per-DOF diagonal: the normal stiffness and
damping term plus the diagonal of the friction Jacobian, both converted from velocity to
displacement derivatives with the HHT factor c_v (see :func:`_contact_tangent_diag`).
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
    """Diagonal of -∂f_contact/∂u for one node, (x, y, z).

    Every contact force term is differentiated, including the velocity-dependent
    ones, so Newton sees the same model the residual evaluates:
      normal   : ∂fn/∂y  = -kn,  ∂fn/∂vy = -kd (only while closing)
      friction : f_t = -mu·fn·g(|v|)·v̂,  ∂f_t/∂v = -mu·fn·[ g/|v|·(I - v̂v̂ᵀ) + g'·v̂v̂ᵀ ]
                 with g' = sech²(|v|/v_reg)/v_reg  (its |v|→0 limit is 1/v_reg).
    Velocity derivatives are converted to displacement derivatives with c_v.
    The regularised-Coulomb slope mu·fn/v_reg is far stiffer than the shell, so
    leaving it out of the tangent (explicit friction) makes NR unable to converge.
    """
    k_yy = kn
    if vn > 0.0:
        k_yy = k_yy + kd * c_v
    sech2 = 1.0 - g * g
    g_prime = sech2 / v_reg
    g_over_v = g / vt_mag
    inv_v2 = 1.0 / (vt_mag * vt_mag)
    ux2 = vt_x * vt_x * inv_v2
    uz2 = vt_z * vt_z * inv_v2
    k_xx = mu * fn * c_v * (g_over_v * (1.0 - ux2) + g_prime * ux2)
    k_zz = mu * fn * c_v * (g_over_v * (1.0 - uz2) + g_prime * uz2)
    return wp.vec3(k_xx, k_yy, k_zz)


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
