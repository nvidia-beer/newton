# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Warp kernels for crawlable solver: ground contact with stick-slip (paper friction).

import warp as wp

# Same as soft/kernels.py for ground contact
PARTICLE_FLAG_ACTIVE = 1


@wp.kernel
def eval_particle_ground_contacts_crawl(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_in_left: wp.array(dtype=wp.int32),
    particle_in_right: wp.array(dtype=wp.int32),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    ground: wp.array(dtype=float),
    gravity_magnitude: float,
    slip_leg: int,
    ft_magnitude: float,
    slip_direction_sign: float,
    n_left: int,
    n_right: int,
    crawl_axis: int,
    slip_force_scale: float,
    f: wp.array(dtype=wp.vec3),
):
    """
    Ground contact with paper stick-slip: two groups (left/right).
    slip_leg: 0 = left slips, 1 = right slips, 2 = both stick (Coulomb for all).
    When slipping, apply f_t in slip direction to that group; when sticking, normal + Coulomb.
    """
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[tid]
    v = particle_v[tid]
    radius = particle_radius[tid]
    inv_m = particle_inv_mass[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)

    vn = wp.dot(n, v)
    jn = c * ke
    if c >= 0.0:
        return
    jd = wp.min(vn, 0.0) * kd
    fn = jn + jd

    in_left = particle_in_left[tid] != 0
    in_right = particle_in_right[tid] != 0

    # Tangential: slip group gets f_t in slip direction; stick group gets Coulomb
    vt = v - n * vn
    vs = wp.length(vt)
    if vs > 1e-9:
        vt = vt / vs
    else:
        vt = wp.vec3(0.0, 0.0, 0.0)

    inv_m_safe = wp.max(inv_m, 1e-9)
    m = wp.where(inv_m > 0.0, 1.0 / inv_m_safe, 0.0)
    n_cap = m * gravity_magnitude
    fn_eff = wp.min(wp.abs(fn), n_cap)

    # Slip direction vector in tangent plane (along crawl_axis: 0=x, 1=y)
    slip_vec = wp.vec3(0.0, 0.0, 0.0)
    if crawl_axis == 0:
        slip_vec = wp.vec3(slip_direction_sign, 0.0, 0.0)
    elif crawl_axis == 1:
        slip_vec = wp.vec3(0.0, slip_direction_sign, 0.0)
    else:
        slip_vec = wp.vec3(slip_direction_sign, 0.0, 0.0)
    slip_vec = slip_vec - n * wp.dot(n, slip_vec)
    slip_len = wp.length(slip_vec)
    if slip_len > 1e-9:
        slip_vec = slip_vec / slip_len

    force = -n * fn

    # Slip force: apply in slip_direction_sign so body moves. Cap at Coulomb limit per particle
    # so we don't create a moment that lifts the leg off the ground (paper: |f_t| = μ f_n at slip).
    if in_left:
        if slip_leg == 0 and n_left > 0:
            ft_per = ft_magnitude * slip_force_scale / float(n_left)
            ft_cap = mu * fn_eff  # per-particle Coulomb limit
            ft_per = wp.min(ft_per, ft_cap)
            force = force + slip_vec * ft_per
        else:
            ft = wp.min(vs * kf, mu * fn_eff)
            friction = -vt * ft
            if slip_leg == 1 and slip_len > 1e-9:
                crawl_comp = wp.dot(friction, slip_vec) * slip_vec
                friction = friction - crawl_comp
            force = force + friction
    elif in_right:
        if slip_leg == 1 and n_right > 0:
            ft_per = ft_magnitude * slip_force_scale / float(n_right)
            ft_cap = mu * fn_eff
            ft_per = wp.min(ft_per, ft_cap)
            force = force + slip_vec * ft_per
        else:
            ft = wp.min(vs * kf, mu * fn_eff)
            friction = -vt * ft
            if slip_leg == 0 and slip_len > 1e-9:
                crawl_comp = wp.dot(friction, slip_vec) * slip_vec
                friction = friction - crawl_comp
            force = force + friction
    else:
        ft = wp.min(vs * kf, mu * fn_eff)
        force = force - vt * ft

    f[tid] = force


@wp.kernel
def apply_crawl_kinematic_displacement(
    particle_q: wp.array(dtype=wp.vec3),
    crawl_axis: int,
    displacement: float,
):
    """
    Displace all particles along the crawl axis by `displacement` (meters).
    From the paper (Gamus et al.): when one foot slips, contact positions are updated
    so the slipping contact moves by Δd; the whole body therefore displaces by ±Δd.
    This applies that kinematic result to the FEM mesh (no velocity kick).
    """
    tid = wp.tid()
    if wp.abs(displacement) < 1e-12:
        return
    q = particle_q[tid]
    if crawl_axis == 0:
        particle_q[tid] = wp.vec3(q[0] + displacement, q[1], q[2])
    elif crawl_axis == 1:
        particle_q[tid] = wp.vec3(q[0], q[1] + displacement, q[2])
    else:
        particle_q[tid] = wp.vec3(q[0] + displacement, q[1], q[2])


