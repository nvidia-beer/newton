# Copyright (c) 2026, The Newton Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Device kernels shared by partitioned and coupled-Newton ANCF interfaces."""

from __future__ import annotations

import warp as wp

from ...sim import JointType
from ..featherstone.kernels import jcalc_integrate


@wp.func
def _scaled_error(residual: float, guess: float, response: float, atol: float, rtol: float) -> float:
    """Residual relative to the per-coordinate tolerance ``atol + rtol * max(|guess|, |response|)``."""
    return wp.abs(residual) / (atol + rtol * wp.max(wp.abs(guess), wp.abs(response)))


@wp.func
def _aitken_weight(
    weight: float, numerator: wp.float64, denominator: wp.float64, iteration: int, initial: float
) -> float:
    """Vector Aitken delta-squared update of the relaxation weight; the first trial uses ``initial``."""
    if iteration == 0:
        return initial
    if denominator > wp.float64(1.0e-30):
        return float(-wp.float64(weight) * numerator / denominator)
    return weight


@wp.kernel
def _integrate_interface_joints(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    dof_dim: wp.array2d[int],
    joint_X_c: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    q: wp.array[float],
    qd: wp.array[float],
    qdd: wp.array[float],
    dt: float,
    q_new: wp.array[float],
    qd_new: wp.array[float],
):
    joint = wp.tid()
    kind = joint_type[joint]
    a, b = q_start[joint], qd_start[joint]
    child = joint_child[joint]
    if kind == JointType.FREE or kind == JointType.DISTANCE:
        # Public joint_qd is child-COM velocity in the parent joint frame.
        # Featherstone's internal integrator instead expects an origin twist.
        position = wp.vec3(q[a], q[a + 1], q[a + 2])
        rotation = wp.quat(q[a + 3], q[a + 4], q[a + 5], q[a + 6])
        offset = wp.transform_point(wp.transform_inverse(joint_X_c[joint]), body_com[child])
        centre = position + wp.quat_rotate(rotation, offset)
        linear = wp.vec3(qd[b], qd[b + 1], qd[b + 2])
        angular = wp.vec3(qd[b + 3], qd[b + 4], qd[b + 5])
        speed = wp.length(angular)
        next_rotation = rotation
        if speed > 0.0:
            next_rotation = wp.normalize(wp.quat_from_axis_angle(angular / speed, dt * speed) * rotation)
        next_position = centre + dt * linear - wp.quat_rotate(next_rotation, offset)
        for i in range(3):
            q_new[a + i] = next_position[i]
        for i in range(4):
            q_new[a + 3 + i] = next_rotation[i]
        for i in range(6):
            qd_new[b + i] = qd[b + i]
    else:
        jcalc_integrate(
            joint_parent[joint],
            joint_X_c[joint],
            body_com[child],
            kind,
            q,
            qd,
            qdd,
            a,
            b,
            dof_dim[joint, 0],
            dof_dim[joint, 1],
            dt,
            q_new,
            qd_new,
        )


# ── ANCF state pack / unpack kernels ────────────────────────────────────────
# Pack all eight dynamic ANCF arrays (6 × vec3 node arrays + global_f_int +
# global_f_int0) into a single contiguous float buffer.  Using one kernel
# dispatch instead of eight separate wp.copy calls reduces CUDA launch
# overhead by 8×.
#
# global_f_int0 MUST be packed alongside global_f_int.  After each
# graph_step() the solver writes ``global_f_int0 = global_f_int`` (HHT
# history update).  Without restoring global_f_int0, GS iteration k=1
# would see the future (k=0) forces as its "previous-step" history, causing
# the HHT predictor to overshoot on every second iteration → cascade NaN.
#
# Buffer layout (n = total node count across all envs):
#   [0        … 3n)    node_x   (3 floats per node)
#   [3n       … 6n)    node_xd
#   [6n       … 9n)    node_xdd
#   [9n       … 12n)   node_D
#   [12n      … 15n)   node_Dd
#   [15n      … 18n)   node_Ddd
#   [18n      … 24n)   global_f_int   (6 floats per node)
#   [24n      … 30n)   global_f_int0  (6 floats per node)
# Total: 30n floats.


@wp.kernel
def _pack_ancf(
    nx: wp.array[wp.vec3],
    nxd: wp.array[wp.vec3],
    nxdd: wp.array[wp.vec3],
    nd: wp.array[wp.vec3],
    ndd: wp.array[wp.vec3],
    nddd: wp.array[wp.vec3],
    fint: wp.array[float],
    fint0: wp.array[float],
    buf: wp.array[float],
    n: int,
):
    i = wp.tid()
    b3 = i * 3
    v = nx[i]
    buf[b3] = v[0]
    buf[b3 + 1] = v[1]
    buf[b3 + 2] = v[2]
    v = nxd[i]
    buf[3 * n + b3] = v[0]
    buf[3 * n + b3 + 1] = v[1]
    buf[3 * n + b3 + 2] = v[2]
    v = nxdd[i]
    buf[6 * n + b3] = v[0]
    buf[6 * n + b3 + 1] = v[1]
    buf[6 * n + b3 + 2] = v[2]
    v = nd[i]
    buf[9 * n + b3] = v[0]
    buf[9 * n + b3 + 1] = v[1]
    buf[9 * n + b3 + 2] = v[2]
    v = ndd[i]
    buf[12 * n + b3] = v[0]
    buf[12 * n + b3 + 1] = v[1]
    buf[12 * n + b3 + 2] = v[2]
    v = nddd[i]
    buf[15 * n + b3] = v[0]
    buf[15 * n + b3 + 1] = v[1]
    buf[15 * n + b3 + 2] = v[2]
    base_f = 18 * n + i * 6
    base_f0 = 24 * n + i * 6
    base_s = i * 6
    buf[base_f] = fint[base_s]
    buf[base_f + 1] = fint[base_s + 1]
    buf[base_f + 2] = fint[base_s + 2]
    buf[base_f + 3] = fint[base_s + 3]
    buf[base_f + 4] = fint[base_s + 4]
    buf[base_f + 5] = fint[base_s + 5]
    buf[base_f0] = fint0[base_s]
    buf[base_f0 + 1] = fint0[base_s + 1]
    buf[base_f0 + 2] = fint0[base_s + 2]
    buf[base_f0 + 3] = fint0[base_s + 3]
    buf[base_f0 + 4] = fint0[base_s + 4]
    buf[base_f0 + 5] = fint0[base_s + 5]


@wp.kernel
def _unpack_ancf(
    buf: wp.array[float],
    nx: wp.array[wp.vec3],
    nxd: wp.array[wp.vec3],
    nxdd: wp.array[wp.vec3],
    nd: wp.array[wp.vec3],
    ndd: wp.array[wp.vec3],
    nddd: wp.array[wp.vec3],
    fint: wp.array[float],
    fint0: wp.array[float],
    n: int,
):
    i = wp.tid()
    b3 = i * 3
    nx[i] = wp.vec3(buf[b3], buf[b3 + 1], buf[b3 + 2])
    nxd[i] = wp.vec3(buf[3 * n + b3], buf[3 * n + b3 + 1], buf[3 * n + b3 + 2])
    nxdd[i] = wp.vec3(buf[6 * n + b3], buf[6 * n + b3 + 1], buf[6 * n + b3 + 2])
    nd[i] = wp.vec3(buf[9 * n + b3], buf[9 * n + b3 + 1], buf[9 * n + b3 + 2])
    ndd[i] = wp.vec3(buf[12 * n + b3], buf[12 * n + b3 + 1], buf[12 * n + b3 + 2])
    nddd[i] = wp.vec3(buf[15 * n + b3], buf[15 * n + b3 + 1], buf[15 * n + b3 + 2])
    base_f = 18 * n + i * 6
    base_f0 = 24 * n + i * 6
    base_s = i * 6
    fint[base_s] = buf[base_f]
    fint[base_s + 1] = buf[base_f + 1]
    fint[base_s + 2] = buf[base_f + 2]
    fint[base_s + 3] = buf[base_f + 3]
    fint[base_s + 4] = buf[base_f + 4]
    fint[base_s + 5] = buf[base_f + 5]
    fint0[base_s] = buf[base_f0]
    fint0[base_s + 1] = buf[base_f0 + 1]
    fint0[base_s + 2] = buf[base_f0 + 2]
    fint0[base_s + 3] = buf[base_f0 + 3]
    fint0[base_s + 4] = buf[base_f0 + 4]
    fint0[base_s + 5] = buf[base_f0 + 5]


@wp.kernel
def _aitken_coefficient(
    guess: wp.array[float],
    response: wp.array[float],
    previous: wp.array[float],
    weight: wp.array[float],
    residual_norm: wp.array[float],
    iteration: int,
    initial_weight: float,
):
    # Vector Aitken delta-squared on the interface velocity fixed point.
    # https://precice.org/configuration-acceleration.html
    numerator = wp.float64(0.0)
    denominator = wp.float64(0.0)
    norm = wp.float64(0.0)
    for i in range(guess.shape[0]):
        r = response[i] - guess[i]
        change = wp.float64(r - previous[i])
        numerator += wp.float64(previous[i]) * change
        denominator += change * change
        norm += wp.float64(r) * wp.float64(r)
        previous[i] = r
    weight[0] = _aitken_weight(weight[0], numerator, denominator, iteration, initial_weight)
    residual_norm[iteration] = float(wp.sqrt(norm))


@wp.kernel
def _relax_coordinates(
    guess: wp.array[float],
    response: wp.array[float],
    weight: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    out[i] = guess[i] + weight[0] * (response[i] - guess[i])


@wp.kernel
def _predict_interface_velocity(velocity: wp.array[float], increment: wp.array[float], converged: wp.array[int]):
    i = wp.tid()
    # Do not extrapolate a remaining interface error as physical acceleration.
    if converged[0] != 0:
        velocity[i] += increment[i]


@wp.kernel
def _save_interface_increment(
    current: wp.array[float],
    previous: wp.array[float],
    increment: wp.array[float],
    iterations: wp.array[int],
    totals: wp.array[int],
):
    i = wp.tid()
    increment[i] = current[i] - previous[i]
    if i == 0:
        totals[0] += 1
        totals[1] += iterations[0]


@wp.kernel
def _adaptive_interface_step(
    guess: wp.array[float],
    response: wp.array[float],
    previous_residual: wp.array[float],
    relaxation: wp.array[float],
    output: wp.array[float],
    residual_norm: wp.array[float],
    active: wp.array[int],
    converged: wp.array[int],
    iterations: wp.array[int],
    iteration: int,
    max_iterations: int,
    atol: float,
    rtol: float,
    initial_weight: float,
    allow_early_exit: bool,
):
    numerator = wp.float64(0.0)
    denominator = wp.float64(0.0)
    norm = wp.float64(0.0)
    scaled = float(0.0)
    for j in range(guess.shape[0]):
        r = response[j] - guess[j]
        change = wp.float64(r - previous_residual[j])
        numerator += wp.float64(previous_residual[j]) * change
        denominator += change * change
        norm += wp.float64(r) * wp.float64(r)
        scaled = wp.max(scaled, _scaled_error(r, guess[j], response[j], atol, rtol))
        previous_residual[j] = r
    weight = _aitken_weight(relaxation[0], numerator, denominator, iteration, initial_weight)
    relaxation[0] = weight
    residual_norm[iteration] = float(wp.sqrt(norm))
    iterations[0] = iteration + 1
    # Form a secant and evaluate the Aitken-corrected trial before accepting.
    # An extrapolated velocity can have a small residual but still amplify
    # added-inertia error if accepted without this correction.
    converged[0] = int(iteration >= 2 and scaled <= 1.0)
    active[0] = int(iteration + 1 < max_iterations and (not allow_early_exit or converged[0] == 0))
    for i in range(guess.shape[0]):
        output[i] = guess[i] + weight * (response[i] - guess[i])


@wp.kernel(enable_backward=False)
def _finish_shell_linearization(
    ready: wp.array[int],
    age: wp.array[int],
    refreshed: wp.array[int],
    count: wp.array[int],
    failed: wp.array[int],
):
    count[0] += 1
    failed[0] = int(ready[0] == 0)
    if ready[0] != 0:
        age[0] = 0
        refreshed[0] = 1


@wp.kernel(enable_backward=False)
def _save_corrected_increment(
    raw: wp.array[float],
    corrected: wp.array[float],
    previous: wp.array[float],
    increment: wp.array[float],
    converged: wp.array[int],
    iterations: wp.array[int],
    totals: wp.array[int],
):
    i = wp.tid()
    increment[i] = 0.0
    solution = raw[i]
    if converged[0] != 0:
        solution = corrected[i]
        # Predict from the corrected fixed point, not the remaining split error
        # in the raw rigid response. Extrapolating that error excites added inertia.
        increment[i] = 2.0 * solution - previous[i] - raw[i]
    previous[i] = solution
    if i == 0:
        totals[0] += 1
        totals[1] += iterations[0]
