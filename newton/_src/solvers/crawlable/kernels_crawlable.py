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
def eval_particle_ground_contacts_crawl_from_buf(
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
    crawl_state_buf: wp.array(dtype=float),
    n_left: int,
    n_right: int,
    crawl_axis: int,
    slip_force_scale: float,
    f: wp.array(dtype=wp.vec3),
):
    """Same as eval_particle_ground_contacts_crawl but reads slip_leg, ft_magnitude, slip_direction_sign from crawl_state_buf[0,1,2] (graph-capture safe, no host sync)."""
    slip_leg_val = crawl_state_buf[0]
    slip_leg = 0
    if slip_leg_val >= 0.5 and slip_leg_val < 1.5:
        slip_leg = 1
    elif slip_leg_val >= 1.5:
        slip_leg = 2
    ft_magnitude = crawl_state_buf[1]
    slip_direction_sign = crawl_state_buf[2]
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
    if in_left:
        if slip_leg == 0 and n_left > 0:
            ft_per = ft_magnitude * slip_force_scale / float(n_left)
            ft_cap = mu * fn_eff
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


# -----------------------------------------------------------------------------
# Device-side stick-slip (graph-capture safe): reduce positions -> phi1/phi2 -> paper state
# Layout of reduce_buf (16 floats): [sum_lc_x,y,z, cnt_lc, sum_rc_x,y,z, cnt_rc, sum_lj_x,y,z, cnt_lj, sum_rj_x,y,z, cnt_rj]
# Layout of state_buf (5 floats): [slip_leg, ft_mag, slip_direction_sign, d_current, d_prev]
# Layout of persistent_buf (5 floats): [d_prev, phi1_prev, phi2_prev, last_slip_dir, last_flip_phase]
# -----------------------------------------------------------------------------

@wp.kernel(enable_backward=False)
def crawl_reduce_init(reduce_buf: wp.array(dtype=float)):
    """Zero the 16-float reduce buffer."""
    tid = wp.tid()
    if tid < 16:
        reduce_buf[tid] = 0.0


@wp.kernel(enable_backward=False)
def crawl_reduce_positions(
    particle_q: wp.array(dtype=wp.vec3),
    in_left_contact: wp.array(dtype=wp.int32),
    in_right_contact: wp.array(dtype=wp.int32),
    in_left_joint: wp.array(dtype=wp.int32),
    in_right_joint: wp.array(dtype=wp.int32),
    reduce_buf: wp.array(dtype=float),
):
    """Atomic sum of positions per group. reduce_buf layout: 4 groups × (3 sum + 1 count)."""
    tid = wp.tid()
    q = particle_q[tid]
    if in_left_contact[tid] != 0:
        wp.atomic_add(reduce_buf, 0, q[0])
        wp.atomic_add(reduce_buf, 1, q[1])
        wp.atomic_add(reduce_buf, 2, q[2])
        wp.atomic_add(reduce_buf, 3, 1.0)
    if in_right_contact[tid] != 0:
        wp.atomic_add(reduce_buf, 4, q[0])
        wp.atomic_add(reduce_buf, 5, q[1])
        wp.atomic_add(reduce_buf, 6, q[2])
        wp.atomic_add(reduce_buf, 7, 1.0)
    if in_left_joint[tid] != 0:
        wp.atomic_add(reduce_buf, 8, q[0])
        wp.atomic_add(reduce_buf, 9, q[1])
        wp.atomic_add(reduce_buf, 10, q[2])
        wp.atomic_add(reduce_buf, 11, 1.0)
    if in_right_joint[tid] != 0:
        wp.atomic_add(reduce_buf, 12, q[0])
        wp.atomic_add(reduce_buf, 13, q[1])
        wp.atomic_add(reduce_buf, 14, q[2])
        wp.atomic_add(reduce_buf, 15, 1.0)


@wp.kernel(enable_backward=False)
def crawl_joint_angles_from_means(
    reduce_buf: wp.array(dtype=float),
    angles_buf: wp.array(dtype=float),
    crawl_axis: int,
):
    """
    Compute phi1, phi2 from 4 mean positions (joint_angles_from_positions).
    crawl_axis 0=x, 1=y; vertical_axis fixed 2=z.
    Writes angles_buf[0]=phi1, angles_buf[1]=phi2.
    """
    cnt_lc = reduce_buf[3]
    cnt_rc = reduce_buf[7]
    cnt_lj = reduce_buf[11]
    cnt_rj = reduce_buf[15]
    if cnt_lc < 1e-9 or cnt_rc < 1e-9 or cnt_lj < 1e-9 or cnt_rj < 1e-9:
        angles_buf[0] = 3.14159265
        angles_buf[1] = 3.14159265
        return
    # left_contact=0-2, right_contact=4-6, left_joint=8-10, right_joint=12-14
    # joint_angles_from_positions(left_contact, left_joint, right_joint, right_contact)
    ax = crawl_axis
    p1_0 = reduce_buf[0] / cnt_lc
    p1_1 = reduce_buf[1] / cnt_lc
    p1_2 = reduce_buf[2] / cnt_lc
    p2_0 = reduce_buf[8] / cnt_lj
    p2_1 = reduce_buf[9] / cnt_lj
    p2_2 = reduce_buf[10] / cnt_lj
    p3_0 = reduce_buf[12] / cnt_rj
    p3_1 = reduce_buf[13] / cnt_rj
    p3_2 = reduce_buf[14] / cnt_rj
    p4_0 = reduce_buf[4] / cnt_rc
    p4_1 = reduce_buf[5] / cnt_rc
    p4_2 = reduce_buf[6] / cnt_rc
    left_contact_ax = p1_0 if ax == 0 else p1_1
    left_contact_vz = p1_2
    left_joint_ax = p2_0 if ax == 0 else p2_1
    left_joint_vz = p2_2
    right_joint_ax = p3_0 if ax == 0 else p3_1
    right_joint_vz = p3_2
    right_contact_ax = p4_0 if ax == 0 else p4_1
    right_contact_vz = p4_2
    v1_0 = left_joint_ax - left_contact_ax
    v1_1 = left_joint_vz - left_contact_vz
    n1 = wp.sqrt(v1_0 * v1_0 + v1_1 * v1_1) + 1e-12
    v1_0 = v1_0 / n1
    v1_1 = v1_1 / n1
    v2_0 = right_joint_ax - left_joint_ax
    v2_1 = right_joint_vz - left_joint_vz
    n2 = wp.sqrt(v2_0 * v2_0 + v2_1 * v2_1) + 1e-12
    v2_0 = v2_0 / n2
    v2_1 = v2_1 / n2
    dot1 = v1_0 * v2_0 + v1_1 * v2_1
    dot1 = wp.max(-1.0, wp.min(1.0, dot1))
    phi1 = 3.14159265 - wp.acos(dot1)
    w1_0 = right_joint_ax - left_joint_ax
    w1_1 = right_joint_vz - left_joint_vz
    nw1 = wp.sqrt(w1_0 * w1_0 + w1_1 * w1_1) + 1e-12
    w1_0 = w1_0 / nw1
    w1_1 = w1_1 / nw1
    w2_0 = right_contact_ax - right_joint_ax
    w2_1 = right_contact_vz - right_joint_vz
    nw2 = wp.sqrt(w2_0 * w2_0 + w2_1 * w2_1) + 1e-12
    w2_0 = w2_0 / nw2
    w2_1 = w2_1 / nw2
    dot2 = w1_0 * w2_0 + w1_1 * w2_1
    dot2 = wp.max(-1.0, wp.min(1.0, dot2))
    phi2 = 3.14159265 - wp.acos(dot2)
    angles_buf[0] = phi1
    angles_buf[1] = phi2


@wp.kernel(enable_backward=False)
def crawl_paper_state_step(
    angles_buf: wp.array(dtype=float),
    persistent_buf: wp.array(dtype=float),
    params_buf: wp.array(dtype=float),
    state_buf: wp.array(dtype=float),
    M: float,
    L: float,
    beta: float,
    mu: float,
    g: float,
    dt: float,
    min_dwell_phase: float,
    min_bend: float,
    crawl_direction: float,
):
    """
    One step of paper stick-slip state machine (device, no host sync).
    params_buf[0]=period, [1]=phase_in_cycle, [2]=last_flip_phase.
    Reads phi1, phi2 from angles_buf; d_prev, phi1_prev, phi2_prev, last_slip_dir from persistent_buf.
    Writes slip_leg, ft_mag, slip_direction_sign, d_current, d_prev to state_buf; updates persistent_buf.
    """
    phi1 = angles_buf[0]
    phi2 = angles_buf[1]
    d_prev = persistent_buf[0]
    phi1_prev = persistent_buf[1]
    phi2_prev = persistent_buf[2]
    prev_slip_dir = persistent_buf[3]
    if prev_slip_dir < -0.5:
        prev_slip_dir = -1.0
    elif prev_slip_dir > 0.5:
        prev_slip_dir = 1.0
    else:
        prev_slip_dir = 1.0
    last_flip_phase = persistent_buf[4]
    period = params_buf[0]
    phase_in_cycle = params_buf[1]

    l = L / (2.0 + beta)
    num_theta = wp.sin(phi1) - wp.sin(phi2)
    den_theta = wp.cos(phi1) + wp.cos(phi2) - beta
    theta = wp.atan2(num_theta, den_theta)
    d = l * (beta * wp.cos(theta) - wp.cos(phi1 - theta) - wp.cos(phi2 + theta))
    xc = l / (2.0 * (2.0 + beta)) * (
        (2.0 + beta) * beta * wp.cos(theta)
        - (3.0 + 2.0 * beta) * wp.cos(phi1 - theta)
        - wp.cos(phi2 + theta)
    )
    d_safe = wp.max(d, 1e-12)
    fn1 = (1.0 - xc / d_safe) * M * g
    fn2 = (xc / d_safe) * M * g
    fn1 = wp.max(0.0, fn1)
    fn2 = wp.max(0.0, fn2)
    delta = (1.0 + beta) / (2.0 * (2.0 + beta)) * l * (wp.cos(phi2 + theta) - wp.cos(phi1 - theta))

    phi1_dot = 0.0
    phi2_dot = 0.0
    if dt > 1e-12:
        phi1_dot = (phi1 - phi1_prev) / dt
        phi2_dot = (phi2 - phi2_prev) / dt
    u = wp.sin(phi1) - wp.sin(phi2)
    v = wp.cos(phi1) + wp.cos(phi2) - beta
    denom = u * u + v * v
    theta_dot = 0.0
    if denom >= 1e-18:
        d_theta_d_phi1 = (v * wp.cos(phi1) + u * wp.sin(phi1)) / denom
        d_theta_d_phi2 = (-v * wp.cos(phi2) + u * wp.sin(phi2)) / denom
        theta_dot = d_theta_d_phi1 * phi1_dot + d_theta_d_phi2 * phi2_dot
    d_dot = l * (
        wp.sin(phi2 + theta) * (phi2_dot + theta_dot)
        + wp.sin(phi1 - theta) * (phi1_dot - theta_dot)
        - beta * wp.sin(theta) * theta_dot
    )

    slip_leg = 0.0
    ft_mag = mu * fn1
    if delta <= 0.0:
        slip_leg = 1.0
        ft_mag = mu * fn2

    d_dot_eps = 5e-2
    if period > 1e-9:
        alpha = 0.35
        d_dot_eps = alpha * l * (2.0 * 3.14159265) / period
        if d_dot_eps < 1e-5:
            d_dot_eps = 1e-5
    slip_dir = prev_slip_dir
    flip_occurred = False
    new_slip = 1.0
    if d_dot < 0.0:
        new_slip = -1.0
    if wp.abs(d_dot) >= d_dot_eps:
        allow_flip = True
        if min_dwell_phase > 0.0 and phase_in_cycle >= 0.0:
            if last_flip_phase >= 0.0:
                delta_phase = phase_in_cycle - last_flip_phase
                if delta_phase < 0.0:
                    delta_phase = delta_phase + 1.0
                allow_flip = delta_phase >= min_dwell_phase
            else:
                allow_flip = (phase_in_cycle >= min_dwell_phase) or (new_slip >= 0.5)
            if new_slip != prev_slip_dir and not allow_flip:
                slip_dir = prev_slip_dir
            else:
                slip_dir = new_slip
                flip_occurred = new_slip != prev_slip_dir
        else:
            slip_dir = new_slip
            flip_occurred = slip_dir != prev_slip_dir

    slip_direction_sign = slip_dir * crawl_direction
    bend = wp.max(wp.abs(phi1 - 3.14159265), wp.abs(phi2 - 3.14159265))
    if bend < min_bend:
        slip_leg = 2.0
        ft_mag = 0.0

    state_buf[0] = slip_leg
    state_buf[1] = ft_mag
    state_buf[2] = slip_direction_sign
    state_buf[3] = d
    state_buf[4] = d_prev

    persistent_buf[0] = d
    persistent_buf[1] = phi1
    persistent_buf[2] = phi2
    persistent_buf[3] = slip_dir
    if flip_occurred and phase_in_cycle >= 0.0:
        persistent_buf[4] = phase_in_cycle


@wp.kernel(enable_backward=False)
def apply_crawl_kinematic_from_buf(
    particle_q: wp.array(dtype=wp.vec3),
    crawl_axis: int,
    crawl_state_buf: wp.array(dtype=float),
    crawl_direction: float,
):
    """Apply paper kinematic displacement from state_buf: body_disp = (d_current - d_prev) * sign; no host sync."""
    slip_leg = crawl_state_buf[0]
    d_current = crawl_state_buf[3]
    d_prev = crawl_state_buf[4]
    delta_d = d_current - d_prev
    if wp.abs(delta_d) < 1e-12:
        return
    body_disp = 0.0
    if slip_leg < 0.5:
        body_disp = -delta_d * crawl_direction
    elif slip_leg < 1.5:
        body_disp = delta_d * crawl_direction
    if wp.abs(body_disp) < 1e-12:
        return
    tid = wp.tid()
    q = particle_q[tid]
    if crawl_axis == 0:
        particle_q[tid] = wp.vec3(q[0] + body_disp, q[1], q[2])
    elif crawl_axis == 1:
        particle_q[tid] = wp.vec3(q[0], q[1] + body_disp, q[2])
    else:
        particle_q[tid] = wp.vec3(q[0] + body_disp, q[1], q[2])


