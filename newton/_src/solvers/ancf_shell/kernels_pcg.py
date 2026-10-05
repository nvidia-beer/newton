# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-local symmetric Gauss-Seidel for small ANCF systems."""

import functools

import numpy as np
import warp as wp


@functools.cache
def make_colored_sweep(padded: int, n_colors: int):
    """Sweep preselected lower/upper block neighbors in one CUDA block per tire."""

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        starts: wp.array2d[int],
        blocks: wp.array[int],
        columns: wp.array[int],
        scaled: wp.array[float],
        nodes: wp.array2d[int],
        counts: wp.array[int],
        output: wp.array[float],
        nnz: int,
    ):
        env, lane = wp.tid()
        z = wp.tile_zeros(shape=padded, dtype=float, storage="shared")
        # The highest-color backward pass has no upper neighbors.
        for pass_index in range(wp.static(2 * n_colors - 1)):
            forward = pass_index < n_colors
            color = pass_index
            direction = int(0)
            if not forward:
                color = 2 * n_colors - pass_index - 2
                direction = 1
            active = lane < 6 * counts[color]
            node = int(0)
            row = lane % 6
            value = float(0.0)
            if active:
                node = nodes[color, lane // 6]
                if forward:
                    value = output[env * padded + node * 6 + row]
                else:
                    value = z[node * 6 + row]
                for edge in range(starts[direction, node], starts[direction, node + 1]):
                    block = blocks[edge]
                    other = columns[block]
                    for col in range(6):
                        value -= scaled[env * nnz + block * 36 + row * 6 + col] * z[other * 6 + col]
            # Every lane reaches the barrier, including inactive padding.
            wp.tile_scatter_masked(z, node * 6 + row, value, active)
        wp.tile_store(output, z, offset=env * padded)

    return sweep


def packed_sweep_layout(starts, blocks, columns, nodes, counts, nnz):
    """Construct a lane-contiguous layout, keeping each node's CSR edge order."""
    nc = len(counts)
    width = ((int(max(counts)) * 6 + 31) // 32) * 32
    maxedges = int(np.diff(starts, axis=1).max())
    size = 2 * nc * maxedges * 6 * width
    # Dense graphs with many small colors can waste excessive padding.
    if size == 0 or size > 4 * nnz:
        return None
    source = np.full((2, nc, maxedges, 6, width), -1, np.int32)
    other = np.zeros((2, nc, maxedges, nodes.shape[1]), np.int32)
    degree = np.zeros((2, nc, nodes.shape[1]), np.int32)
    for direction in range(2):
        for color in range(nc):
            for slot in range(counts[color]):
                node = nodes[color, slot]
                edges = blocks[starts[direction, node] : starts[direction, node + 1]]
                degree[direction, color, slot] = len(edges)
                for edge, block in enumerate(edges):
                    other[direction, color, edge, slot] = columns[block]
                    for row in range(6):
                        for col in range(6):
                            source[direction, color, edge, col, slot * 6 + row] = block * 36 + row * 6 + col
    return source.reshape(-1), other, degree, width, maxedges


@wp.kernel(enable_backward=False)
def pack_sweep_values(
    values: wp.array[float],
    inverse: wp.array[float],
    rows: wp.array[int],
    source: wp.array[int],
    n_nodes: int,
    nnz: int,
    size: int,
    out: wp.array[float],
):
    """Scale only the off-diagonal entries used by SGS, directly into packed storage."""
    i = wp.tid()
    index = source[i % size]
    value = float(0.0)
    if index >= 0:
        env = i // size
        block = index // 36
        row, col = (index % 36) // 6, index % 6
        inv_base = (env * n_nodes + rows[block]) * 36 + row * 6
        for k in range(6):
            value += inverse[inv_base + k] * values[env * nnz + block * 36 + k * 6 + col]
    out[i] = value


@functools.cache
def make_packed_colored_sweep(padded: int, ncolors: int, width: int, maxedges: int):
    """Coalesce matrix reads without changing the CSR summation order."""

    @wp.kernel(enable_backward=False, module="unique")
    def sweep(
        packed: wp.array[float],
        other: wp.array4d[int],
        degree: wp.array3d[int],
        nodes: wp.array2d[int],
        counts: wp.array[int],
        output: wp.array[float],
    ):
        env, lane = wp.tid()
        z = wp.tile_zeros(shape=padded, dtype=float, storage="shared")
        for phase in range(wp.static(2 * ncolors - 1)):
            forward = phase < ncolors
            color = phase
            direction = int(0)
            if not forward:
                color = 2 * ncolors - phase - 2
                direction = 1
            active = lane < 6 * counts[color]
            node = int(0)
            row = lane % 6
            value = float(0.0)
            if active:
                slot = lane // 6
                node = nodes[color, slot]
                if forward:
                    value = output[env * padded + node * 6 + row]
                else:
                    value = z[node * 6 + row]
                for edge in range(degree[direction, color, slot]):
                    neighbor = other[direction, color, edge, slot]
                    for col in range(6):
                        index = ((((env * 2 + direction) * ncolors + color) * maxedges + edge) * 6 + col) * width + lane
                        value -= packed[index] * z[neighbor * 6 + col]
            wp.tile_scatter_masked(z, node * 6 + row, value, active)
        wp.tile_store(output, z, offset=env * padded)

    return sweep
