# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Scatter element-level forces and stiffness blocks into global arrays.

Global DOF layout: flat float array of length n_nodes * 6.
    DOF index = node_i * 6 + sub   (sub 0-5: px,py,pz,Dx,Dy,Dz)

Global stiffness: 6×6 node-block CSR, one block per (node_i, node_j) pair that
    appears in any element (see `build_block_structure`).

Graph-capture path
------------------
Since the mesh never changes, K_eff has a fixed sparsity pattern every step.
`assemble_sparse_stiffness` (one-time, at init) builds a scalar CSR via
warp.sparse.bsr_set_from_triplets only to fix that pattern; `build_block_structure`
derives the node-block CSR and the element->block scatter map from it; the
solver then refills the blocked values in place (`add_diag_to_blk_values_batched`
and the gather kernels) without any memory allocation, so the step can be
captured as a CUDA graph.
"""

import numpy as np
import warp as wp
import warp.sparse as wps

wp.set_module_options({"enable_backward": False})


@wp.func
def _pressure_element(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    p: float,
    e: int,
    elem_fp: wp.array2d[float],  # (.,24) zeroed, write block
):
    """Follower pressure on the deformed bilinear quad mid-surface.

    Force   f_a = p ∫ N_a (x_,ξ × x_,η) dξ dη          (acts normal to current surface)

    2×2 Gauss quadrature on ξ,η ∈ [−1,1].  ``p`` already carries the per-element
    orientation sign so positive p pushes outward.  The follower-load tangent is
    deliberately not assembled (as in Chrono's ChANCFTire, SetStiff(false)):
    the pressure enters the residual only.
    """
    g = 0.5773502691896258  # 1/sqrt(3)
    gpxi = wp.vec4(-g, g, g, -g)
    gpeta = wp.vec4(-g, -g, g, g)
    cxi = wp.vec4(-1.0, 1.0, 1.0, -1.0)  # corner natural coords
    ceta = wp.vec4(-1.0, -1.0, 1.0, 1.0)

    for igp in range(4):
        xi = gpxi[igp]
        eta = gpeta[igp]
        # Bilinear Q4 shape functions + derivatives at this Gauss point.
        N0 = 0.25 * (1.0 + cxi[0] * xi) * (1.0 + ceta[0] * eta)
        N1 = 0.25 * (1.0 + cxi[1] * xi) * (1.0 + ceta[1] * eta)
        N2 = 0.25 * (1.0 + cxi[2] * xi) * (1.0 + ceta[2] * eta)
        N3 = 0.25 * (1.0 + cxi[3] * xi) * (1.0 + ceta[3] * eta)
        Nv = wp.vec4(N0, N1, N2, N3)

        dxi0 = 0.25 * cxi[0] * (1.0 + ceta[0] * eta)
        dxi1 = 0.25 * cxi[1] * (1.0 + ceta[1] * eta)
        dxi2 = 0.25 * cxi[2] * (1.0 + ceta[2] * eta)
        dxi3 = 0.25 * cxi[3] * (1.0 + ceta[3] * eta)

        det0 = 0.25 * ceta[0] * (1.0 + cxi[0] * xi)
        det1 = 0.25 * ceta[1] * (1.0 + cxi[1] * xi)
        det2 = 0.25 * ceta[2] * (1.0 + cxi[2] * xi)
        det3 = 0.25 * ceta[3] * (1.0 + cxi[3] * xi)

        # Surface tangents at the Gauss point (weight = 1).
        x_xi = dxi0 * x0 + dxi1 * x1 + dxi2 * x2 + dxi3 * x3
        x_eta = det0 * x0 + det1 * x1 + det2 * x2 + det3 * x3
        nvec = wp.cross(x_xi, x_eta)  # area-weighted current normal

        for a in range(4):
            fa = (p * Nv[a]) * nvec
            elem_fp[e, a * 6 + 0] = elem_fp[e, a * 6 + 0] + fa[0]
            elem_fp[e, a * 6 + 1] = elem_fp[e, a * 6 + 1] + fa[1]
            elem_fp[e, a * 6 + 2] = elem_fp[e, a * 6 + 2] + fa[2]


# ---------------------------------------------------------------------------
# Sealed-gas cavity (CTIS) — enclosed volume + ideal-gas pressure.
#
# The tire air is a closed compressible cavity: p = K_gas / V, where V is the
# enclosed volume (a function of the node positions) and K_gas = m·R·T is set
# by the CTIS control (how much air is pumped in).  As the tire expands V grows
# and p drops — self-limiting negative feedback that keeps the inflate/deflate
# sweep stable.  The gauge load applied to the shell is (p − p_build), so at the
# build pressure the net load is zero and the tire holds its built shape.
# ---------------------------------------------------------------------------


@wp.kernel
def cavity_gas_law(
    Kgas: wp.array[float],  # (N,) m·R·T  (CTIS control)
    V: wp.array[float],  # (N,) enclosed volume
    pbuild: wp.array[float],  # (N,) build pressure (rest shape)
    p_out: wp.array[float],  # (N,) absolute pressure  = K_gas / V
    gauge_out: wp.array[float],  # (N,) gauge load = p − p_build  (== solver.pressure)
):
    """dim = N.  Ideal-gas pressure and the gauge load applied to the shell."""
    env = wp.tid()
    v = V[env]
    p = float(0.0)
    if v > 1.0e-9:
        p = Kgas[env] / v
    p_out[env] = p
    gauge_out[env] = p - pbuild[env]


@wp.kernel
def cavity_gas_law_circuit(
    Kgas: wp.array[float],  # (N,) m·R·T per tire; the circuit holds their sum
    V: wp.array[float],  # (N,) enclosed volume per tire
    pbuild: wp.array[float],  # (N,) build pressure (rest shape)
    p_out: wp.array[float],  # (N,) absolute pressure, the same for every tire
    gauge_out: wp.array[float],  # (N,) gauge load = p − p_build
    n_envs: int,
):
    """dim = 1.  Ideal gas in ONE cavity: all N tires are joined by an air line, so
    p = Σ K_gas / Σ V.  A tire squeezed by the ground pushes its air into the other
    three instead of stiffening on its own (SHERP's pneumocirculating suspension)."""
    k_sum = float(0.0)
    v_sum = float(0.0)
    for e in range(n_envs):
        k_sum += Kgas[e]
        v_sum += V[e]
    p = float(0.0)
    if v_sum > 1.0e-9:
        p = k_sum / v_sum
    for e in range(n_envs):
        p_out[e] = p
        gauge_out[e] = p - pbuild[e]


@wp.kernel
def compute_pressure_force_stiffness(
    node_x: wp.array[wp.vec3],  # (n_nodes,) deformed positions
    elem_nodes: wp.array2d[wp.int32],  # (n_elem, 4)
    psign: wp.array[float],  # (n_elem,) orientation +1/−1
    pressure: wp.array[float],  # [1] gauge pressure [Pa]
    elem_fp: wp.array2d[float],  # (n_elem, 24)   zeroed by caller
):
    """Single-env follower pressure force."""
    e = wp.tid()
    p = pressure[0] * psign[e]
    x0 = node_x[elem_nodes[e, 0]]
    x1 = node_x[elem_nodes[e, 1]]
    x2 = node_x[elem_nodes[e, 2]]
    x3 = node_x[elem_nodes[e, 3]]
    _pressure_element(x0, x1, x2, x3, p, e, elem_fp)


# ---------------------------------------------------------------------------
# COO triplet extraction — element K blocks → (row, col, val) arrays
# ---------------------------------------------------------------------------


@wp.kernel
def extract_coo_triplets(
    elem_nodes: wp.array2d[wp.int32],  # (n_elem, 4)
    elem_K: wp.array3d[float],  # (n_elem, 24, 24)
    # Outputs: one triplet per (elem, node_k, node_j, sub_k, sub_j) combination
    # Stride: n_elem * 4 * 4 * 6 * 6  total entries
    # We pack as (elem, k, j) → linear index = e*(24*24) + k*24 + j
    out_row: wp.array[wp.int32],  # (n_elem * 576,)
    out_col: wp.array[wp.int32],  # (n_elem * 576,)
    out_val: wp.array[float],  # (n_elem * 576,)
):
    """Unpack each element 24×24 K into scalar COO triplets (global DOF indices)."""
    e = wp.tid()
    base = e * 576

    for node_k in range(4):
        na_k = elem_nodes[e, node_k]
        for sk in range(6):
            k_local = node_k * 6 + sk
            k_global = na_k * 6 + sk

            for node_j in range(4):
                na_j = elem_nodes[e, node_j]
                for sj in range(6):
                    j_local = node_j * 6 + sj
                    j_global = na_j * 6 + sj

                    idx = base + k_local * 24 + j_local
                    out_row[idx] = k_global
                    out_col[idx] = j_global
                    out_val[idx] = elem_K[e, k_local, j_local]


# ---------------------------------------------------------------------------
# Python-level assembly helpers
# ---------------------------------------------------------------------------


def assemble_sparse_stiffness(
    elem_nodes: wp.array2d[wp.int32],
    elem_K: wp.array3d[float],
    n_nodes: int,
    device: str,
) -> wps.BsrMatrix:
    """Assemble the global stiffness as a scalar CSR (BsrMatrix with 1×1 blocks).

    Duplicate (row,col) entries are summed by bsr_set_from_triplets.
    """
    n_dof = n_nodes * 6
    n_elem = elem_K.shape[0]
    n_triplets = n_elem * 576  # 24*24 per element
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=float, device=device)
    wp.launch(extract_coo_triplets, dim=n_elem, inputs=[elem_nodes, elem_K, rows, cols, vals], device=device)
    K_bsr = wps.bsr_zeros(n_dof, n_dof, block_type=float, device=device)
    wps.bsr_set_from_triplets(K_bsr, rows, cols, vals)
    return K_bsr


# ---------------------------------------------------------------------------
# Blocked (6x6 per node pair) storage of K_eff — used by the PCG path.
# Node-block CSR: blk_offsets (n_nodes+1,), blk_columns (nnzb,) shared across envs;
# values flat (N * nnzb * 36,), block b of env e at e*nnz + b*36, row-major 6x6.
# Every node pair of an element receives a full 6x6 block, so this layout holds
# exactly the real non-zeros (no triplet-capacity padding) and the SpMV reads one
# column index per 36 values instead of one per value.
# ---------------------------------------------------------------------------


def build_block_structure(
    elem_nodes_wp: wp.array2d[wp.int32],
    K_eff: wps.BsrMatrix,
    n_nodes: int,
    device,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array2d[wp.int32], int]:
    """Derive the node-block CSR and the element->block scatter map from the scalar CSR.

    Returns ``(blk_offsets, blk_columns, blk_scatter_map, nnzb)`` where
    ``blk_scatter_map[e, a*4 + b]`` is the block index of node pair
    ``(elem_nodes[e, a], elem_nodes[e, b])``.
    """
    offsets = K_eff.offsets.numpy()  # (n_dof + 1,)
    columns = K_eff.columns.numpy()  # capacity-sized; valid up to offsets[-1]
    elem_nodes = elem_nodes_wp.numpy()

    blk_offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    blk_cols: list[np.ndarray] = []
    for a in range(n_nodes):
        rs, re = int(offsets[6 * a]), int(offsets[6 * a + 1])
        nb = np.unique(columns[rs:re] // 6)
        blk_cols.append(nb.astype(np.int32))
        blk_offsets[a + 1] = blk_offsets[a] + len(nb)
    blk_columns = np.concatenate(blk_cols) if blk_cols else np.zeros(0, dtype=np.int32)
    nnzb = int(blk_offsets[-1])

    n_elem = elem_nodes.shape[0]
    scatter = np.full((n_elem, 16), -1, dtype=np.int32)
    for e in range(n_elem):
        nodes = elem_nodes[e]
        for a in range(4):
            row = int(nodes[a])
            rs, re = int(blk_offsets[row]), int(blk_offsets[row + 1])
            row_cols = blk_columns[rs:re]
            for b in range(4):
                pos = int(np.searchsorted(row_cols, int(nodes[b])))
                if pos < len(row_cols) and row_cols[pos] == nodes[b]:
                    scatter[e, a * 4 + b] = rs + pos
    if np.any(scatter < 0):
        raise RuntimeError("build_block_structure: element node pair missing from the assembled sparsity")

    return (
        wp.array(blk_offsets, dtype=wp.int32, device=device),
        wp.array(blk_columns, dtype=wp.int32, device=device),
        wp.array(scatter, dtype=wp.int32, device=device),
        nnzb,
    )


@wp.kernel
def add_diag_to_blk_values_batched(
    K_diag: wp.array[float],  # (N*n_dof,)
    blk_offsets: wp.array[wp.int32],  # (n_nodes+1,) — shared
    blk_columns: wp.array[wp.int32],  # (nnzb,) — shared
    vals: wp.array[float],  # (N*nnz,) flat
    n_dof: int,
    nnz: int,
):
    """dim = N*n_dof.  Add K_diag onto the diagonal of the blocked matrix."""
    tid = wp.tid()
    env = tid // n_dof
    i = tid % n_dof
    node = i // 6
    s = i % 6
    diag_val = K_diag[tid]
    for b in range(blk_offsets[node], blk_offsets[node + 1]):
        if blk_columns[b] == node:
            wp.atomic_add(vals, env * nnz + b * 36 + s * 7, diag_val)
            break
