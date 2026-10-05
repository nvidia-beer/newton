# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU reductions and small solves for joint shell/interface corrections."""

import functools

import warp as wp


@wp.kernel(enable_backward=False)
def aa_gather(
    a: wp.array[wp.vec3],
    da: wp.array[wp.vec3],
    v: wp.array[float],
    free: wp.array[float],
    mass: wp.array[float],
    Mr: wp.array3d[float],
    arm: wp.array[float],
    scale: float,
    x: wp.array[float],
    weights: wp.array[float],
):
    i = wp.tid()
    n = mass.shape[0]
    if i < n:
        j = i % 6
        value = float(0.0)
        if j < 3:
            value = a[i // 6][j]
        else:
            value = da[i // 6][j - 3]
        x[i] = value * scale * free[i]
        weights[i] = mass[i] * free[i]
    else:
        row = i - n
        x[i] = v[row]
        weights[i] = Mr[0, row, row] + arm[row]


@wp.func
def triple(a: float, b: float, c: float):
    return wp.float64(a) * wp.float64(b) * wp.float64(c)


@wp.kernel(enable_backward=False)
def aa_products(
    current: wp.array[float],
    output: wp.array[float],
    previous: wp.array[float],
    weights: wp.array[float],
    nums: wp.array[wp.float64],
):
    group = wp.tid()
    offset = group * 128
    c = wp.tile_load(current, shape=128, offset=offset)
    out = wp.tile_load(output, shape=128, offset=offset)
    old = wp.tile_load(previous, shape=128, offset=offset)
    w = wp.tile_load(weights, shape=128, offset=offset)
    residual = out - c
    diff = residual - old
    num = wp.tile_sum(wp.tile_map(triple, diff, residual, w))
    den = wp.tile_sum(wp.tile_map(triple, diff, diff, w))
    wp.tile_atomic_add(nums, num, offset=0)
    wp.tile_atomic_add(nums, den, offset=1)


@wp.kernel(enable_backward=False)
def aa_update(
    current: wp.array[float],
    output: wp.array[float],
    prev_x: wp.array[float],
    prev_r: wp.array[float],
    nums: wp.array[wp.float64],
    iteration: int,
    a: wp.array[wp.vec3],
    da: wp.array[wp.vec3],
    v: wp.array[float],
    free: wp.array[float],
    scale: float,
):
    node = wp.tid()
    n = free.shape[0]
    gamma = float(0.0)
    if iteration > 0 and nums[1] > wp.float64(1e-12):
        gamma = wp.clamp(float(nums[0] / nums[1]), -2.0, 1.0)
    if node < n // 6:
        av = a[node]
        dv = da[node]
        for j in range(6):
            i = node * 6 + j
            res = output[i] - current[i]
            mixed = output[i] - gamma * (current[i] - prev_x[i] + res - prev_r[i])
            prev_x[i] = current[i]
            prev_r[i] = res
            if free[i] != 0.0:
                if j < 3:
                    av[j] = mixed / scale
                else:
                    dv[j - 3] = mixed / scale
        a[node] = av
        da[node] = dv
    elif node < n // 6 + v.shape[0]:
        row = node - n // 6
        i = n + row
        res = output[i] - current[i]
        v[row] = output[i] - gamma * (current[i] - prev_x[i] + res - prev_r[i])
        prev_x[i] = current[i]
        prev_r[i] = res


@wp.func
def product_double(a: float, b: float):
    return wp.float64(a) * wp.float64(b)


@wp.kernel(enable_backward=False)
def project(
    B: wp.array2d[float],
    X: wp.array2d[float],
    V: wp.array2d[float],
    M: wp.array2d[float],
    C: wp.array2d[float],
    k: int,
    n_chunks: int,
    scale: float,
    refresh: wp.array[int],
    Z: wp.array[wp.float64],
):
    group = wp.tid()
    env = group // (k * k)
    i = group // k % k
    j = group % k
    # The cached reduced tangent only consumes these columns on refresh.
    if j != k - 1 and refresh[0] == 0:
        return
    total = wp.tile_zeros(shape=1, dtype=wp.float64)
    for chunk in range(n_chunks):
        b = wp.tile_load(B, shape=(1, 128), offset=(env * k + i, chunk * 128))
        x = wp.tile_load(X, shape=(1, 128), offset=(env * k + j, chunk * 128))
        v = wp.tile_load(V, shape=(1, 128), offset=(env * k + j, chunk * 128))
        mass = wp.tile_load(M, shape=(1, 128), offset=(env, chunk * 128))
        contact = wp.tile_load(C, shape=(1, 128), offset=(env, chunk * 128))
        value = mass * v + scale * contact * x
        total += wp.tile_sum(wp.tile_map(product_double, b, value))
    wp.tile_store(Z, total, offset=group)


@functools.cache
def make_inverse(n):
    size = wp.constant(n * n)

    @wp.kernel(enable_backward=False, module="unique")
    def inverse(
        S: wp.array2d[wp.float64],
        out: wp.array2d[wp.float64],
        ready: wp.array[int],
        age: wp.array[int],
        refreshed: wp.array[int],
        count: wp.array[int],
    ):
        _, lane = wp.tid()
        i = lane // n
        j = lane % n
        active = lane < n * n
        A = wp.tile_zeros(shape=size, dtype=wp.float64, storage="shared")
        X = wp.tile_zeros(shape=size, dtype=wp.float64, storage="shared")
        An = wp.tile_zeros(shape=size, dtype=wp.float64, storage="shared")
        Xn = wp.tile_zeros(shape=size, dtype=wp.float64, storage="shared")
        a = wp.float64(0.0)
        x = wp.float64(0.0)
        if active:
            a = S[i, j]
            if i == j:
                x = wp.float64(1.0)
        wp.tile_scatter_masked(A, lane, a, active)
        wp.tile_scatter_masked(X, lane, x, active)
        for k in range(n):
            a = wp.float64(0.0)
            x = wp.float64(0.0)
            if active:
                pivot = k
                for row in range(k + 1, n):
                    if wp.abs(A[row * n + k]) > wp.abs(A[pivot * n + k]):
                        pivot = row
                row = i
                if i == k:
                    row = pivot
                elif i == pivot:
                    row = k
                a = A[row * n + j]
                x = X[row * n + j]
                piv = A[pivot * n + k]
                if i == k:
                    a /= piv
                    x /= piv
                else:
                    ratio = A[row * n + k] / piv
                    a -= ratio * A[pivot * n + j]
                    x -= ratio * X[pivot * n + j]
            wp.tile_scatter_masked(An, lane, a, active)
            wp.tile_scatter_masked(Xn, lane, x, active)
            wp.tile_assign(A, An)
            wp.tile_assign(X, Xn)
        if active:
            out[i, j] = X[lane]
        if lane == 0:
            ready[0] = 1
            age[0] = 0
            refreshed[0] = 1
            count[0] += 1

    return inverse


@wp.kernel(enable_backward=False)
def rigid_mass(
    start: wp.array[int],
    end: wp.array[int],
    children: wp.array[int],
    inertia: wp.array[wp.spatial_matrix],
    jacobian: wp.array3d[float],
    mass: wp.array3d[float],
):
    """Assemble each entry independently, retaining the serial sum order over bodies."""
    articulation, i, j = wp.tid()
    total = float(0.0)
    for joint in range(start[articulation], end[articulation]):
        spatial = inertia[children[joint]]
        row = (joint - start[articulation]) * 6
        value = float(0.0)
        for k in range(6):
            for l in range(6):
                value += jacobian[articulation, row + k, i] * spatial[k, l] * jacobian[articulation, row + l, j]
        total += value
    mass[articulation, i, j] = total


@wp.kernel(enable_backward=False)
def copy_kinematics(
    a0: wp.array[float],
    a1: wp.array[float],
    a2: wp.array[float],
    a3: wp.array[float],
    a4: wp.array[float],
    a5: wp.array[float],
    b0: wp.array[float],
    b1: wp.array[float],
    b2: wp.array[float],
    b3: wp.array[float],
    b4: wp.array[float],
    b5: wp.array[float],
):
    i = wp.tid()
    if i < a0.shape[0]:
        b0[i] = a0[i]
    if i < a1.shape[0]:
        b1[i] = a1[i]
    if i < a2.shape[0]:
        b2[i] = a2[i]
    if i < a3.shape[0]:
        b3[i] = a3[i]
    if i < a4.shape[0]:
        b4[i] = a4[i]
    if i < a5.shape[0]:
        b5[i] = a5[i]


@wp.kernel(enable_backward=False)
def copy_rigid(
    pose: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    q: wp.array[float],
    qd: wp.array[float],
    pose_out: wp.array[wp.transform],
    velocity_out: wp.array[wp.spatial_vector],
    q_out: wp.array[float],
    qd_out: wp.array[float],
):
    i = wp.tid()
    if i < pose.shape[0]:
        pose_out[i] = pose[i]
        velocity_out[i] = velocity[i]
    if i < q.shape[0]:
        q_out[i] = q[i]
    if i < qd.shape[0]:
        qd_out[i] = qd[i]
