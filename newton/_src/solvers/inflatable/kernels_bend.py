# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Bend/torque kernel: linear spring + torque for springs with non-zero rest direction.
# τ = -(k*θ + kd*ω)*axis, F = (τ × d)/l at endpoints.

import warp as wp


@wp.kernel
def eval_springs_linear_and_torque(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    spring_rest_direction: wp.array(dtype=wp.vec3),
    torque_stiffness: wp.float32,
    torque_damping: wp.float32,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    if i == -1 or j == -1:
        return

    # Linear spring
    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]
    xi, xj = x[i], x[j]
    vi, vj = v[i], v[j]
    xij = xi - xj
    vij = vi - vj
    l = wp.length(xij)
    if l < 1.0e-6:
        return
    d = xij / l
    c = l - rest
    dcdt = wp.dot(d, vij)
    fs = d * (ke * c + kd * dcdt)
    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)

    # Torque (zero rest direction = skip)
    d0 = spring_rest_direction[tid]
    if wp.length(d0) < 0.5:
        return
    dot_d_d0 = wp.clamp(wp.dot(d, d0), -1.0, 1.0)
    theta = wp.acos(dot_d_d0)
    rot_axis = wp.cross(d0, d)
    rot_len = wp.length(rot_axis)
    if rot_len < 1.0e-6:
        return
    rot_axis = rot_axis / rot_len
    d_dot = (vij - d * wp.dot(d, vij)) / l
    omega_along = wp.dot(wp.cross(d, d_dot), rot_axis)
    tau_mag = -(torque_stiffness * theta + torque_damping * omega_along) * l
    force_torque = wp.cross(rot_axis * tau_mag, d) / l
    wp.atomic_sub(f, i, force_torque)
    wp.atomic_add(f, j, force_torque)
