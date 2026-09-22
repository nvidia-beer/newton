# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Mesh bookkeeping and render-buffer kernels shared by the ANCF tire examples.

The tire mesh is a quad grid of ``n_ax_divs + 1`` axial node rows with ``n_circ`` nodes each
(row-major, axial-major), loaded from a baked USD asset (``load_ancf_tire_usd``). The helpers
here derive the index sets the examples need from that layout (bead rows, ring segments,
render triangles) and convert ANCF Y-up node data into the Z-up viewer buffers.

ANCF frame: x_lat, y_up, z_fwd (axle along X).  Viewer frame (Z-up): x_fwd, y_lat, z_up.
Y-up -> Z-up: (x, y, z) -> (z, x, y).
"""

from __future__ import annotations

import numpy as np
import warp as wp

# ── Kernels ───────────────────────────────────────────────────────────────────


@wp.kernel
def _ancf_yup_to_zu(src: wp.array[wp.vec3], dst: wp.array[wp.vec3]):
    """ANCF Y-up -> Z-up for particle rendering: (x,y,z) -> (z,x,y)."""
    i = wp.tid()
    p = src[i]
    dst[i] = wp.vec3(p[2], p[0], p[1])


@wp.kernel
def _gather_zu(src: wp.array[wp.vec3], indices: wp.array[wp.int32], dst: wp.array[wp.vec3]):
    """Gather indexed positions already in Z-up."""
    i = wp.tid()
    dst[i] = src[indices[i]]


@wp.kernel
def _build_ring_lines(
    bead_pos_zu: wp.array[wp.vec3],
    seg_s: wp.array[wp.int32],
    seg_e: wp.array[wp.int32],
    line_starts: wp.array[wp.vec3],
    line_ends: wp.array[wp.vec3],
):
    """Line-segment start/end arrays from gathered bead positions."""
    i = wp.tid()
    line_starts[i] = bead_pos_zu[seg_s[i]]
    line_ends[i] = bead_pos_zu[seg_e[i]]


# ── Host-side index / material builders ───────────────────────────────────────


def material_row(mat) -> np.ndarray:
    """The 11 ``elem_mat`` coefficients of an ANCF material, in the solver's column order."""
    return np.array(
        [mat.C11, mat.C22, mat.C33, mat.C12, mat.C13, mat.C23, mat.G23, mat.G13, mat.G12, mat.rho, mat.alpha_damp],
        dtype=np.float32,
    )


def bead_row_indices(n_bead_per_ring: int, n_bead_rows: int, n_ax_divs: int) -> np.ndarray:
    """Local node indices of the ``n_bead_rows`` outermost rows on each side (left rows first)."""
    left_rows = [np.arange(k * n_bead_per_ring, (k + 1) * n_bead_per_ring, dtype=np.int32) for k in range(n_bead_rows)]
    right_rows = [
        np.arange((n_ax_divs - k) * n_bead_per_ring, (n_ax_divs - k + 1) * n_bead_per_ring, dtype=np.int32)
        for k in range(n_bead_rows)
    ]
    return np.concatenate(left_rows + right_rows)


def ring_segments(n_bead_per_ring: int, n_bead_rows: int, n_bead: int, n_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) indices into the flat ``n_envs * n_bead`` bead array closing each bead ring."""
    n_rings = 2 * n_bead_rows
    seg_s_1 = np.array(
        [i + k * n_bead_per_ring for k in range(n_rings) for i in range(n_bead_per_ring)], dtype=np.int32
    )
    seg_e_1 = np.array(
        [(i + 1) % n_bead_per_ring + k * n_bead_per_ring for k in range(n_rings) for i in range(n_bead_per_ring)],
        dtype=np.int32,
    )
    seg_s_all = np.concatenate([seg_s_1 + e * n_bead for e in range(n_envs)])
    seg_e_all = np.concatenate([seg_e_1 + e * n_bead for e in range(n_envs)])
    return seg_s_all, seg_e_all


def quad_triangles(elem_nodes: np.ndarray, n_nodes: int, n_envs: int) -> np.ndarray:
    """Two render triangles per quad element, tiled over ``n_envs`` tires of ``n_nodes`` nodes."""
    tris = np.empty((len(elem_nodes) * 2, 3), dtype=np.int32)
    tris[0::2] = elem_nodes[:, [0, 1, 2]]
    tris[1::2] = elem_nodes[:, [0, 2, 3]]
    return np.concatenate([tris + e * n_nodes for e in range(n_envs)], axis=0)
