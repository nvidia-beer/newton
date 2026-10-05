# Copyright (c) 2026, The Newton Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Device kernels shared by partitioned and coupled-Newton ANCF interfaces."""

from __future__ import annotations

import functools

import warp as wp

from ...sim import JointType
from ..featherstone.kernels import jcalc_integrate


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
    if iteration == 0:
        weight[0] = initial_weight
    elif denominator > wp.float64(1.0e-30):
        weight[0] = float(-wp.float64(weight[0]) * numerator / denominator)
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
        scale = atol + rtol * wp.max(wp.abs(guess[j]), wp.abs(response[j]))
        scaled = wp.max(scaled, wp.abs(r) / scale)
        previous_residual[j] = r
    weight = relaxation[0]
    if iteration == 0:
        weight = initial_weight
    elif denominator > wp.float64(1.0e-30):
        weight = float(-wp.float64(weight) * numerator / denominator)
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
def _probe_guess(base: wp.array[float], out: wp.array[float], column: wp.array[int], epsilon: float):
    i = wp.tid()
    out[i] = base[i]
    if i == column[0]:
        out[i] += epsilon


@wp.kernel(enable_backward=False)
def _response_column(
    base: wp.array[float],
    response: wp.array[float],
    matrix: wp.array2d[wp.float64],
    column: wp.array[int],
    epsilon: float,
):
    i = wp.tid()
    identity = wp.float64(0.0)
    if i == column[0]:
        identity = wp.float64(1.0)
    matrix[i, column[0]] = identity - wp.float64(response[i] - base[i]) / wp.float64(epsilon)


@wp.kernel(enable_backward=False)
def _advance_interface_iteration(iteration: wp.array[int]):
    iteration[0] += 1


@wp.kernel(enable_backward=False)
def _advance_response_probe(column: wp.array[int], active: wp.array[int], n: int):
    column[0] += 1
    active[0] = int(column[0] < n)


@wp.kernel(enable_backward=False)
def _choose_response_refresh(
    guess: wp.array[float],
    response: wp.array[float],
    previous: wp.array[float],
    ready: wp.array[int],
    age: wp.array[int],
    totals: wp.array[int],
    refreshed: wp.array[int],
    active: wp.array[int],
    iteration_index: wp.array[int],
    atol: float,
    rtol: float,
    warmup_steps: int,
    refresh_interval: int,
    minimum_interval: int,
    probe_count: wp.array[int],
):
    iteration = iteration_index[0]
    norm = wp.float64(0.0)
    old_norm = wp.float64(0.0)
    scaled = float(0.0)
    for i in range(guess.shape[0]):
        r = wp.float64(response[i] - guess[i])
        scaled = wp.max(
            scaled,
            float(wp.abs(r)) / (atol + rtol * wp.max(wp.abs(guess[i]), wp.abs(response[i]))),
        )
        norm += r * r
        old_norm += wp.float64(previous[i]) * wp.float64(previous[i])
    if iteration == 0:
        refreshed[0] = 0
        age[0] += 1
    slow = iteration > 0 and norm > wp.float64(0.81) * old_norm and scaled > 1.0
    # Invalid/noisy estimates must not trigger a full Jacobian rebuild each
    # substep. Use the existing Aitken fallback while the refresh budget recovers.
    active[0] = int(
        totals[0] >= warmup_steps
        and refreshed[0] == 0
        and (probe_count[0] == 0 or age[0] >= minimum_interval)
        and (ready[0] == 0 or age[0] >= refresh_interval or slow)
    )


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


@functools.cache
def _make_response_inverse(n: int):
    """Partial-pivot inverse of I - d(rigid response)/d(interface velocity)."""
    mat = wp.types.matrix(shape=(n, n), dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def invert(
        matrix: wp.array2d[wp.float64],
        inverse: wp.array2d[wp.float64],
        ready: wp.array[int],
        age: wp.array[int],
        refreshed: wp.array[int],
        count: wp.array[int],
    ):
        a = mat()
        b = mat()
        for i in range(n):
            b[i, i] = wp.float64(1.0)
            for j in range(n):
                a[i, j] = matrix[i, j]
        valid = bool(True)
        for col in range(n):
            pivot = col
            for row in range(col + 1, n):
                if wp.abs(a[row, col]) > wp.abs(a[pivot, col]):
                    pivot = row
            if wp.abs(a[pivot, col]) < wp.float64(1.0e-8):
                valid = False
            if valid:
                for j in range(n):
                    temp = a[col, j]
                    a[col, j] = a[pivot, j]
                    a[pivot, j] = temp
                    temp = b[col, j]
                    b[col, j] = b[pivot, j]
                    b[pivot, j] = temp
                d = a[col, col]
                for j in range(n):
                    a[col, j] /= d
                    b[col, j] /= d
                for i in range(n):
                    if i != col:
                        factor = a[i, col]
                        for j in range(n):
                            a[i, j] -= factor * a[col, j]
                            b[i, j] -= factor * b[col, j]
        for i in range(n):
            for j in range(n):
                if not wp.isfinite(b[i, j]) or wp.abs(b[i, j]) > wp.float64(10.0):
                    valid = False
        ready[0] = int(valid)
        if valid:
            for i in range(n):
                for j in range(n):
                    inverse[i, j] = b[i, j]
        refreshed[0] = 1
        age[0] = 0
        count[0] += n

    return invert


@wp.kernel(enable_backward=False)
def _update_response_secant(
    guess: wp.array[float],
    response: wp.array[float],
    previous_guess: wp.array[float],
    previous_residual: wp.array[float],
    inverse: wp.array2d[wp.float64],
    ready: wp.array[int],
    refresh: wp.array[int],
    hy: wp.array[wp.float64],
    sh: wp.array[wp.float64],
    iteration_index: wp.array[int],
):
    iteration = iteration_index[0]
    # Good Broyden: H += (s-Hy)(s^T H)/(s^T H y), with y = -delta(residual).
    # Reject tiny/noisy secants and near-zero denominators; do not compare
    # residuals across substeps, whose force and inertia history have changed.
    # Float32 bead poses make smaller velocity secants unreliable during motion.
    n = guess.shape[0]
    if iteration > 0 and ready[0] != 0 and refresh[0] == 0:
        curvature = wp.float64(0.0)
        snorm = wp.float64(0.0)
        hynorm = wp.float64(0.0)
        ynorm = wp.float64(0.0)
        for i in range(n):
            a = wp.float64(0.0)
            b = wp.float64(0.0)
            for j in range(n):
                yj = wp.float64(previous_residual[j] - (response[j] - guess[j]))
                sj = wp.float64(guess[j] - previous_guess[j])
                a += inverse[i, j] * yj
                b += sj * inverse[j, i]
            hy[i] = a
            sh[i] = b
            si = wp.float64(guess[i] - previous_guess[i])
            yi = wp.float64(previous_residual[i] - (response[i] - guess[i]))
            curvature += si * a
            snorm += si * si
            hynorm += a * a
            ynorm += yi * yi
        if (
            snorm > wp.float64(1.0e-6)
            and ynorm > wp.float64(1.0e-4)
            and curvature > wp.float64(0.05) * wp.sqrt(snorm * hynorm)
        ):
            valid = bool(True)
            for i in range(n):
                error = wp.float64(guess[i] - previous_guess[i]) - hy[i]
                for j in range(n):
                    inverse[i, j] += error * sh[j] / curvature
                    if not wp.isfinite(inverse[i, j]) or wp.abs(inverse[i, j]) > wp.float64(10.0):
                        valid = False
            if not valid:
                ready[0] = 0
    for i in range(n):
        previous_guess[i] = guess[i]


@wp.kernel(enable_backward=False)
def _response_secant_products(
    guess: wp.array[float],
    response: wp.array[float],
    previous_guess: wp.array[float],
    previous_residual: wp.array[float],
    inverse: wp.array2d[wp.float64],
    ready: wp.array[int],
    refresh: wp.array[int],
    hy: wp.array[wp.float64],
    sh: wp.array[wp.float64],
    iteration: wp.array[int],
):
    i = wp.tid()
    if iteration[0] > 0 and ready[0] != 0 and refresh[0] == 0:
        a = wp.float64(0.0)
        b = wp.float64(0.0)
        for j in range(guess.shape[0]):
            a += inverse[i, j] * wp.float64(previous_residual[j] - (response[j] - guess[j]))
            b += wp.float64(guess[j] - previous_guess[j]) * inverse[j, i]
        hy[i] = a
        sh[i] = b


@wp.kernel(enable_backward=False)
def _response_secant_curvature(
    guess: wp.array[float],
    response: wp.array[float],
    previous_guess: wp.array[float],
    previous_residual: wp.array[float],
    ready: wp.array[int],
    refresh: wp.array[int],
    hy: wp.array[wp.float64],
    iteration: wp.array[int],
    accepted_curvature: wp.array[wp.float64],
):
    accepted_curvature[0] = wp.float64(0.0)
    if iteration[0] > 0 and ready[0] != 0 and refresh[0] == 0:
        curvature = wp.float64(0.0)
        snorm = wp.float64(0.0)
        hynorm = wp.float64(0.0)
        ynorm = wp.float64(0.0)
        for i in range(guess.shape[0]):
            si = wp.float64(guess[i] - previous_guess[i])
            yi = wp.float64(previous_residual[i] - (response[i] - guess[i]))
            curvature += si * hy[i]
            snorm += si * si
            hynorm += hy[i] * hy[i]
            ynorm += yi * yi
        if (
            snorm > wp.float64(1.0e-6)
            and ynorm > wp.float64(1.0e-4)
            and curvature > wp.float64(0.05) * wp.sqrt(snorm * hynorm)
        ):
            accepted_curvature[0] = curvature


@wp.kernel(enable_backward=False)
def _response_secant_apply(
    guess: wp.array[float],
    previous_guess: wp.array[float],
    inverse: wp.array2d[wp.float64],
    ready: wp.array[int],
    hy: wp.array[wp.float64],
    sh: wp.array[wp.float64],
    accepted_curvature: wp.array[wp.float64],
):
    i = wp.tid()
    curvature = accepted_curvature[0]
    # Gate on the previous kernel's decision, not ready: another row may
    # invalidate ready while this row is still updating the matrix.
    if curvature > wp.float64(0.0):
        error = wp.float64(guess[i] - previous_guess[i]) - hy[i]
        for j in range(guess.shape[0]):
            value = inverse[i, j] + error * sh[j] / curvature
            inverse[i, j] = value
            if not wp.isfinite(value) or wp.abs(value) > wp.float64(10.0):
                wp.atomic_min(ready, 0, 0)
    previous_guess[i] = guess[i]


@wp.kernel(enable_backward=False)
def _response_correction(
    guess: wp.array[float],
    response: wp.array[float],
    inverse: wp.array2d[wp.float64],
    ready: wp.array[int],
    correction: wp.array[wp.float64],
):
    i = wp.tid()
    if ready[0] != 0:
        value = wp.float64(0.0)
        for j in range(guess.shape[0]):
            value += inverse[i, j] * wp.float64(response[j] - guess[j])
        correction[i] = value


@wp.kernel(enable_backward=False)
def _newton_interface_step(
    guess: wp.array[float],
    response: wp.array[float],
    previous: wp.array[float],
    weight: wp.array[float],
    inverse: wp.array2d[wp.float64],
    response_correction: wp.array[wp.float64],
    ready: wp.array[int],
    output: wp.array[float],
    residual_norm: wp.array[float],
    active: wp.array[int],
    converged: wp.array[int],
    iterations: wp.array[int],
    iteration_index: wp.array[int],
    max_iterations: int,
    atol: float,
    rtol: float,
    initial_weight: float,
    allow_exit: bool,
):
    iteration = iteration_index[0]
    norm = wp.float64(0.0)
    numerator = wp.float64(0.0)
    denominator = wp.float64(0.0)
    scaled = float(0.0)
    for i in range(guess.shape[0]):
        r = response[i] - guess[i]
        change = wp.float64(r - previous[i])
        numerator += wp.float64(previous[i]) * change
        denominator += change * change
        norm += wp.float64(r) * wp.float64(r)
        scaled = wp.max(
            scaled,
            wp.abs(r) / (atol + rtol * wp.max(wp.abs(guess[i]), wp.abs(response[i]))),
        )
    omega = weight[0]
    if iteration == 0:
        omega = initial_weight
    elif denominator > wp.float64(1.0e-30):
        omega = float(-wp.float64(omega) * numerator / denominator)
    weight[0] = omega
    for i in range(guess.shape[0]):
        correction = omega * (response[i] - guess[i])
        if ready[0] != 0:
            if response_correction.shape[0] != 0:
                correction = float(response_correction[i])
            else:
                acc = wp.float64(0.0)
                for j in range(guess.shape[0]):
                    acc += inverse[i, j] * wp.float64(response[j] - guess[j])
                correction = float(acc)
        output[i] = guess[i] + correction
    for i in range(guess.shape[0]):
        previous[i] = response[i] - guess[i]
    residual_norm[iteration] = float(wp.sqrt(norm))
    iterations[0] = iteration + 1
    minimum_met = iteration >= 2 or ready[0] != 0
    converged[0] = int(minimum_met and scaled <= 1.0)
    active[0] = int(iteration + 1 < max_iterations and (not allow_exit or converged[0] == 0))


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
