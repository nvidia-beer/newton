# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Galerkin correction for the smooth modes missed by node-block Jacobi."""

import warp as wp

wp.set_module_options({"enable_backward": False})

_mat99d = wp.types.matrix(shape=(9, 9), dtype=wp.float64)
TILE = wp.constant(128)

_vec9d = wp.types.vector(length=9, dtype=wp.float64)


@wp.func
def _to_double(value: float) -> wp.float64:
    return wp.float64(value)


@wp.func
def _mode(x: wp.vec3, director: wp.vec3, component: int, mode: int) -> float:
    sub = component % 3
    value = float(0.0)
    if mode < 3:
        if component < 3 and sub == mode:
            value = 1.0
    else:
        v = x
        if component >= 3:
            v = director
        if mode < 6:
            axis = wp.vec3(0.0)
            axis[mode - 3] = 1.0
            value = wp.cross(axis, v)[sub]
        elif sub == mode - 6:
            value = v[sub]
    return value


@wp.kernel
def build_modes(
    x: wp.array[wp.vec3],
    director: wp.array[wp.vec3],
    free: wp.array[float],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    n: int,
    nnz: int,
    padded: int,
    q: wp.array[float],
    aq: wp.array[float],
):
    # Adjacent lanes write adjacent rows within one mode, matching q/aq storage.
    mode, row = wp.tid()
    env = row // n
    local = row % n
    node = local // 6
    component = local % 6
    node_base = env * (n // 6)
    # A nearby origin avoids loss of precision far from the world origin.
    origin = x[node_base]
    q[(env * 9 + mode) * padded + local] = free[row] * _mode(
        x[node_base + node] - origin, director[node_base + node], component, mode
    )
    value = wp.float64(0.0)
    for b in range(offsets[node], offsets[node + 1]):
        other = columns[b]
        p = x[node_base + other] - origin
        d = director[node_base + other]
        # The nine modes have only 1, 4 or 2 nonzero entries per node.
        # Form A*Q without six repeated cross products for every matrix row.
        row_base = env * nnz + b * 36 + component * 6
        free_base = env * n + 6 * other
        if mode < 3:
            value += wp.float64(values[row_base + mode]) * wp.float64(free[free_base + mode])
        elif mode < 6:
            axis = mode - 3
            u, v = (axis + 1) % 3, (axis + 2) % 3
            value -= wp.float64(values[row_base + u]) * wp.float64(free[free_base + u] * p[v])
            value += wp.float64(values[row_base + v]) * wp.float64(free[free_base + v] * p[u])
            value -= wp.float64(values[row_base + u + 3]) * wp.float64(free[free_base + u + 3] * d[v])
            value += wp.float64(values[row_base + v + 3]) * wp.float64(free[free_base + v + 3] * d[u])
        else:
            c = mode - 6
            value += wp.float64(values[row_base + c]) * wp.float64(free[free_base + c] * p[c])
            value += wp.float64(values[row_base + c + 3]) * wp.float64(free[free_base + c + 3] * d[c])
    aq[(env * 9 + mode) * padded + local] = float(value)


@wp.kernel
def gram(q: wp.array[float], aq: wp.array[float], padded: int, chunks: int, matrix: wp.array[wp.float64]):
    tid = wp.tid()
    env = tid // (81 * chunks)
    entry = (tid // chunks) % 81
    chunk = tid % chunks
    lhs = wp.tile_load(q, shape=TILE, offset=(env * 9 + entry // 9) * padded + chunk * TILE)
    rhs = wp.tile_load(aq, shape=TILE, offset=(env * 9 + entry % 9) * padded + chunk * TILE)
    wp.tile_atomic_add(matrix, wp.tile_map(_to_double, wp.tile_sum(lhs * rhs)), offset=env * 81 + entry)


@wp.kernel
def project(q: wp.array[float], residual: wp.array[float], padded: int, chunks: int, rhs: wp.array[wp.float64]):
    tid = wp.tid()
    mode = tid // chunks
    env = mode // 9
    chunk = tid % chunks
    a = wp.tile_load(q, shape=TILE, offset=mode * padded + chunk * TILE)
    b = wp.tile_load(residual, shape=TILE, offset=env * padded + chunk * TILE)
    wp.tile_atomic_add(rhs, wp.tile_map(_to_double, wp.tile_sum(a * b)), offset=mode)


@wp.kernel
def factor_coarse(
    matrix: wp.array[wp.float64],
    factors: wp.array[wp.float64],
    scales: wp.array[wp.float64],
):
    """Equilibrate and factor one coarse matrix per block, once per PCG solve."""
    env, lane = wp.tid()
    a = wp.tile_zeros(shape=81, dtype=wp.float64, storage="shared")
    next_a = wp.tile_zeros(shape=81, dtype=wp.float64, storage="shared")
    row, col = lane // 9, lane % 9
    value = wp.float64(0.0)
    active = lane < 81
    if active:
        sr, sc = wp.float64(0.0), wp.float64(0.0)
        dr, dc = matrix[env * 81 + row * 10], matrix[env * 81 + col * 10]
        if dr > wp.float64(0.0):
            sr = wp.float64(1.0) / wp.sqrt(dr)
        if dc > wp.float64(0.0):
            sc = wp.float64(1.0) / wp.sqrt(dc)
        value = (matrix[env * 81 + row * 9 + col] + matrix[env * 81 + col * 9 + row]) * wp.float64(0.5)
        value *= sr * sc
        if col == 0:
            scales[env * 9 + row] = sr
    wp.tile_scatter_masked(a, lane, value, active)
    for k in range(9):
        if active:
            value = a[lane]
            if row > k:
                multiplier = wp.float64(0.0)
                pivot = a[k * 10]
                if pivot > wp.float64(1.0e-12):
                    multiplier = a[row * 9 + k] / pivot
                if col > k:
                    value -= multiplier * a[k * 9 + col]
                elif col == k:
                    value = multiplier
        # Finish all reads of the old pivot before overwriting any entry of a.
        wp.tile_scatter_masked(next_a, lane, value, active)
        wp.tile_assign(a, next_a)
    wp.tile_store(factors, a, offset=env * 81)


@wp.kernel
def solve_coarse_factored(
    factors: wp.array[wp.float64],
    scales: wp.array[wp.float64],
    projected: wp.array[wp.float64],
    coefficients: wp.array2d[float],
):
    """Apply the same coarse factorization to the initial and final residuals."""
    env = wp.tid()
    rhs = _vec9d()
    for i in range(9):
        value = projected[env * 9 + i] * scales[env * 9 + i]
        for j in range(i):
            value -= factors[env * 81 + i * 9 + j] * rhs[j]
        rhs[i] = value
    result = _vec9d()
    for reverse in range(9):
        i = 8 - reverse
        value = rhs[i]
        for j in range(i + 1, 9):
            value -= factors[env * 81 + i * 9 + j] * result[j]
        pivot = factors[env * 81 + i * 10]
        if pivot > wp.float64(1.0e-12):
            result[i] = value / pivot
        coefficients[env, i] = float(result[i] * scales[env * 9 + i])


@wp.kernel
def solve_coarse(
    matrix: wp.array[wp.float64],
    projected: wp.array[wp.float64],
    coefficients: wp.array2d[float],
):
    env = wp.tid()
    a = _mat99d()
    rhs = _vec9d()
    for i in range(9):
        for j in range(9):
            a[i, j] = wp.float64(matrix[env * 81 + i * 9 + j] + matrix[env * 81 + j * 9 + i]) * wp.float64(0.5)
        rhs[i] = wp.float64(projected[env * 9 + i])
    # Diagonal equilibration removes the different units of translations and rotations.
    scale = _vec9d()
    for i in range(9):
        scale[i] = wp.float64(0.0)
        if a[i, i] > wp.float64(0.0):
            scale[i] = wp.float64(1.0) / wp.sqrt(a[i, i])
    for i in range(9):
        rhs[i] *= scale[i]
        for j in range(9):
            a[i, j] *= scale[i] * scale[j]
    # A zero mode (e.g. all nodes pinned) contributes no correction.
    for k in range(9):
        if a[k, k] > wp.float64(1.0e-12):
            for i in range(k + 1, 9):
                factor = a[i, k] / a[k, k]
                for j in range(k + 1, 9):
                    a[i, j] -= factor * a[k, j]
                rhs[i] -= factor * rhs[k]
    result = _vec9d()
    for reverse in range(9):
        i = 8 - reverse
        value = rhs[i]
        for j in range(i + 1, 9):
            value -= a[i, j] * result[j]
        if a[i, i] > wp.float64(1.0e-12):
            result[i] = value / a[i, i]
        coefficients[env, i] = float(result[i] * scale[i])


@wp.kernel
def correct(
    q: wp.array[float],
    aq: wp.array[float],
    coefficients: wp.array2d[float],
    n: int,
    padded: int,
    x: wp.array[float],
    residual: wp.array[float],
):
    row = wp.tid()
    env = row // n
    index = env * padded + row % n
    dx = float(0.0)
    dr = float(0.0)
    for k in range(9):
        dx += q[(env * 9 + k) * padded + row % n] * coefficients[env, k]
        dr += aq[(env * 9 + k) * padded + row % n] * coefficients[env, k]
    x[index] += dx
    residual[index] -= dr
