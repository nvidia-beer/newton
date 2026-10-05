# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Ordered reductions for independent shell environments.

Element/node incidence is fixed. Gather in that order so GPU scheduling cannot
seed different tipping motions in identical translated environments.
"""

import functools

import numpy as np
import warp as wp

from .kernels_assembly import _pressure_element

wp.set_module_options({"enable_backward": False})


def incidence_maps(nodes, block_map, n_nodes, n_blocks, device):
    node_entries = [[] for _ in range(n_nodes)]
    block_entries = [[] for _ in range(n_blocks)]
    for e, quad in enumerate(nodes):
        for a, node in enumerate(quad):
            node_entries[node].append(4 * e + a)
            for b in range(4):
                block_entries[block_map[e, 4 * a + b]].append(16 * e + 4 * a + b)

    def pack(entries):
        offsets = np.cumsum([0] + [len(row) for row in entries], dtype=np.int32)
        indices = np.array([i for row in entries for i in row], dtype=np.int32)
        return wp.array(offsets, dtype=int, device=device), wp.array(indices, dtype=int, device=device)

    return *pack(node_entries), *pack(block_entries)


@wp.kernel
def forces(
    offsets: wp.array[int],
    indices: wp.array[int],
    elem_f: wp.array2d[float],
    output: wp.array[float],
    n_nodes: int,
    n_elems: int,
):
    tid = wp.tid()
    env = tid // (n_nodes * 6)
    node = (tid // 6) % n_nodes
    sub = tid % 6
    total = wp.float64(0.0)
    for j in range(offsets[node], offsets[node + 1]):
        entry = indices[j]
        total += wp.float64(elem_f[env * n_elems + entry // 4, (entry % 4) * 6 + sub])
    output[tid] += float(total)


@wp.kernel
def stiffness(
    offsets: wp.array[int],
    indices: wp.array[int],
    elem_k: wp.array3d[float],
    scale: float,
    output: wp.array[float],
    n_blocks: int,
    n_elems: int,
):
    tid = wp.tid()
    env = tid // (n_blocks * 36)
    block = (tid // 36) % n_blocks
    row = (tid % 36) // 6
    col = tid % 6
    total = wp.float64(0.0)
    for j in range(offsets[block], offsets[block + 1]):
        entry = indices[j]
        e = env * n_elems + entry // 16
        a = (entry % 16) // 4
        b = entry % 4
        total += wp.float64(elem_k[e, a * 6 + row, b * 6 + col])
    output[tid] = scale * float(total)


CAVITY_CHUNK = wp.constant(32)


@wp.kernel
def cavity_centroid_parts(
    x: wp.array[wp.vec3],
    n: int,
    parts: wp.array2d[wp.vec3d],
):
    env, chunk = wp.tid()
    origin = x[env * n]
    total = wp.vec3d(0.0)
    for j in range(CAVITY_CHUNK):
        i = chunk * CAVITY_CHUNK + j
        if i < n:
            total += wp.vec3d(x[env * n + i] - origin)
    parts[env, chunk] = total


@wp.kernel
def cavity_centroid_finish(parts: wp.array2d[wp.vec3d], n: int, centre: wp.array[wp.vec3]):
    env = wp.tid()
    total = wp.vec3d(0.0)
    for chunk in range(parts.shape[1]):
        total += parts[env, chunk]
    centre[env] = wp.vec3(total / wp.float64(n))


@wp.kernel
def cavity_volume_parts(
    x: wp.array[wp.vec3],
    nodes: wp.array2d[int],
    psign: wp.array[float],
    centre: wp.array[wp.vec3],
    n: int,
    ne: int,
    parts: wp.array2d[wp.float64],
):
    env, chunk = wp.tid()
    base = env * n
    origin = x[base]
    c = centre[env]
    total = wp.float64(0.0)
    for j in range(CAVITY_CHUNK):
        e = chunk * CAVITY_CHUNK + j
        if e < ne:
            x0 = x[base + nodes[e, 0]] - origin
            x1 = x[base + nodes[e, 1]] - origin
            x2 = x[base + nodes[e, 2]] - origin
            x3 = x[base + nodes[e, 3]] - origin
            area = (0.5 * psign[e]) * wp.cross(x2 - x0, x3 - x1)
            total += wp.float64(wp.dot(0.25 * (x0 + x1 + x2 + x3) - c, area) / 3.0)
    parts[env, chunk] = total


@wp.kernel
def cavity_volume_finish(parts: wp.array2d[wp.float64], volume: wp.array[float]):
    env = wp.tid()
    total = wp.float64(0.0)
    for chunk in range(parts.shape[1]):
        total += parts[env, chunk]
    volume[env] = float(total)


@wp.kernel
def pressure(
    node_x: wp.array[wp.vec3],
    nodes: wp.array2d[int],
    psign: wp.array[float],
    p: wp.array[float],
    output: wp.array2d[float],
    n_elems: int,
    n_nodes: int,
):
    tid = wp.tid()
    e = tid % n_elems
    env = tid // n_elems
    base = env * n_nodes
    _pressure_element(
        node_x[base + nodes[e, 0]],
        node_x[base + nodes[e, 1]],
        node_x[base + nodes[e, 2]],
        node_x[base + nodes[e, 3]],
        p[env] * psign[e],
        tid,
        output,
    )


@functools.cache
def make_cavity_reduction(n, ne, block):
    """Reduce the closed-surface volume in one block, using a nearby origin."""
    inverse_n = 1.0 / n

    @wp.kernel(enable_backward=False, module="unique")
    def cavity(
        x: wp.array[wp.vec3],
        nodes: wp.array2d[int],
        sign: wp.array[float],
        centre: wp.array[wp.vec3],
        volume: wp.array[float],
    ):
        env, lane = wp.tid()
        base = env * n
        origin = x[base]
        t = wp.vec3d(0.0)
        for part in range(wp.static((n + block - 1) // block)):
            i = part * block + lane
            if i < n:
                t += wp.vec3d(x[base + i] - origin)
        sx = wp.tile_sum(wp.tile(t[0]))
        sy = wp.tile_sum(wp.tile(t[1]))
        sz = wp.tile_sum(wp.tile(t[2]))
        c = wp.vec3(
            float(sx[0] * wp.float64(inverse_n)),
            float(sy[0] * wp.float64(inverse_n)),
            float(sz[0] * wp.float64(inverse_n)),
        )
        total = wp.float64(0.0)
        for part in range(wp.static((ne + block - 1) // block)):
            e = part * block + lane
            if e < ne:
                x0 = x[base + nodes[e, 0]] - origin
                x1 = x[base + nodes[e, 1]] - origin
                x2 = x[base + nodes[e, 2]] - origin
                x3 = x[base + nodes[e, 3]] - origin
                area = 0.5 * sign[e] * wp.cross(x2 - x0, x3 - x1)
                total += wp.float64(wp.dot(0.25 * (x0 + x1 + x2 + x3) - c, area) / 3.0)
        result = wp.tile_sum(wp.tile(total))
        if lane == 0:
            centre[env] = c
            volume[env] = float(result[0])

    return cavity
