# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Scatter element-level forces and stiffness blocks into global arrays.

Global DOF layout: flat float array of length n_nodes * 6.
    DOF index = node_i * 6 + sub   (sub 0-5: px,py,pz,Dx,Dy,Dz)

Global stiffness: sparse BSR matrix with 6×6 blocks, one block per
    (node_i, node_j) pair that appears in any element.  Built via
    warp.sparse.bsr_set_from_triplets which sums duplicate (row,col) entries.

Graph-capture path
------------------
Since the mesh never changes, K_eff has a fixed sparsity pattern every step.
`build_scatter_map` (CPU, one-time) maps each element (row, col) triplet to
its absolute index in BsrMatrix.values.  `scatter_elem_to_bsr` then zeros
and fills K_eff.values in-place without any memory allocation — the only path
that can be captured as a CUDA graph.
"""

import numpy as np
import warp as wp
import warp.sparse as wps

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------
# Force scatter — element forces → global flat DOF vector
# ---------------------------------------------------------------------------


@wp.kernel
def scatter_forces(
    elem_nodes: wp.array2d[wp.int32],  # (n_elem, 4)
    elem_f: wp.array2d[float],  # (n_elem, 24)
    global_f: wp.array[float],  # (n_nodes * 6,) output
):
    """Atomically accumulate element internal forces into the global DOF vector."""
    e = wp.tid()
    for node_local in range(4):
        na = elem_nodes[e, node_local]
        base = na * 6
        for s in range(6):
            wp.atomic_add(global_f, base + s, elem_f[e, node_local * 6 + s])


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
def accumulate_centroid(
    node_x: wp.array[wp.vec3],  # (N*n_nodes,)
    centroid_sum: wp.array[float],  # (N*3,) zeroed by caller
    n_nodes: int,
):
    """dim = N*n_nodes.  Sum node positions per env (caller divides by n_nodes).

    A co-moving centroid makes the enclosed-volume integral translation
    invariant — the tire can drop/roll without spuriously changing its volume.
    """
    tid = wp.tid()
    env = tid // n_nodes
    x = node_x[tid]
    wp.atomic_add(centroid_sum, env * 3 + 0, x[0])
    wp.atomic_add(centroid_sum, env * 3 + 1, x[1])
    wp.atomic_add(centroid_sum, env * 3 + 2, x[2])


@wp.kernel
def accumulate_cavity_volume(
    node_x: wp.array[wp.vec3],  # (N*n_nodes,) deformed
    elem_nodes: wp.array2d[wp.int32],  # (n_elems, 4) shared topology
    psign: wp.array[float],  # (n_elems,) outward orientation
    centroid_sum: wp.array[float],  # (N*3,)
    inv_n_nodes: float,
    V_out: wp.array[float],  # (N,) zeroed by caller
    n_elems: int,
    n_nodes: int,
):
    """dim = N*n_elems.  Enclosed volume via divergence theorem, cone-closed to
    the per-env centroid:  V = (1/3) Σ_elem (x̄ − c) · (area-normal)."""
    tid = wp.tid()
    env = tid // n_elems
    e = tid % n_elems
    nb = env * n_nodes
    cx = wp.vec3(
        centroid_sum[env * 3 + 0] * inv_n_nodes,
        centroid_sum[env * 3 + 1] * inv_n_nodes,
        centroid_sum[env * 3 + 2] * inv_n_nodes,
    )
    x0 = node_x[nb + elem_nodes[e, 0]]
    x1 = node_x[nb + elem_nodes[e, 1]]
    x2 = node_x[nb + elem_nodes[e, 2]]
    x3 = node_x[nb + elem_nodes[e, 3]]
    xbar = 0.25 * (x0 + x1 + x2 + x3)
    nrm = (0.5 * psign[e]) * wp.cross(x2 - x0, x3 - x1)  # outward area-normal
    wp.atomic_add(V_out, env, (1.0 / 3.0) * wp.dot(xbar - cx, nrm))


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


@wp.kernel
def compute_pressure_force_stiffness_batched(
    node_x: wp.array[wp.vec3],  # (N*n_nodes,) deformed
    elem_nodes: wp.array2d[wp.int32],  # (n_elems, 4) — shared topology
    psign: wp.array[float],  # (n_elems,) — shared orientation
    pressure: wp.array[float],  # (N,) per-env gauge pressure [Pa]
    elem_fp: wp.array2d[float],  # (N*n_elems, 24)   zeroed by caller
    n_elems: int,
    n_nodes: int,
):
    """dim = N*n_elems.  Per-env follower pressure force."""
    tid = wp.tid()
    env = tid // n_elems
    e = tid % n_elems
    p = pressure[env] * psign[e]
    nb = env * n_nodes
    x0 = node_x[nb + elem_nodes[e, 0]]
    x1 = node_x[nb + elem_nodes[e, 1]]
    x2 = node_x[nb + elem_nodes[e, 2]]
    x3 = node_x[nb + elem_nodes[e, 3]]
    _pressure_element(x0, x1, x2, x3, p, tid, elem_fp)


@wp.kernel
def compute_pressure_force_stiffness_batched_gp(
    node_x: wp.array[wp.vec3],
    elem_nodes: wp.array2d[wp.int32],
    psign: wp.array[float],
    pressure: wp.array[float],
    elem_fp: wp.array2d[float],  # (N*n_elems, 24)  zeroed by caller
    n_elems: int,
    n_nodes: int,
):
    """dim = N*n_elems*4.  One thread per (element, Gauss point).

    Same physics as compute_pressure_force_stiffness_batched, one thread per
    in-plane Gauss point; forces accumulate via atomic_add.
    """
    tid = wp.tid()
    e_global = tid // 4
    gp_flat = tid % 4

    env = e_global // n_elems
    e = e_global % n_elems
    p = pressure[env] * psign[e]
    nb = env * n_nodes

    x0 = node_x[nb + elem_nodes[e, 0]]
    x1 = node_x[nb + elem_nodes[e, 1]]
    x2 = node_x[nb + elem_nodes[e, 2]]
    x3 = node_x[nb + elem_nodes[e, 3]]

    # 2×2 Gauss point natural coordinates (hardcoded by gp_flat 0..3)
    g = float(0.5773502691896258)
    xi = float(0.0)
    eta = float(0.0)
    if gp_flat == 0:
        xi = -g
        eta = -g
    elif gp_flat == 1:
        xi = g
        eta = -g
    elif gp_flat == 2:
        xi = g
        eta = g
    else:
        xi = -g
        eta = g

    cxi0 = float(-1.0)
    cxi1 = float(1.0)
    cxi2 = float(1.0)
    cxi3 = float(-1.0)
    ceta0 = float(-1.0)
    ceta1 = float(-1.0)
    ceta2 = float(1.0)
    ceta3 = float(1.0)

    N0 = 0.25 * (1.0 + cxi0 * xi) * (1.0 + ceta0 * eta)
    N1 = 0.25 * (1.0 + cxi1 * xi) * (1.0 + ceta1 * eta)
    N2 = 0.25 * (1.0 + cxi2 * xi) * (1.0 + ceta2 * eta)
    N3 = 0.25 * (1.0 + cxi3 * xi) * (1.0 + ceta3 * eta)
    Nv = wp.vec4(N0, N1, N2, N3)

    dxi0 = 0.25 * cxi0 * (1.0 + ceta0 * eta)
    dxi1 = 0.25 * cxi1 * (1.0 + ceta1 * eta)
    dxi2 = 0.25 * cxi2 * (1.0 + ceta2 * eta)
    dxi3 = 0.25 * cxi3 * (1.0 + ceta3 * eta)

    det0 = 0.25 * ceta0 * (1.0 + cxi0 * xi)
    det1 = 0.25 * ceta1 * (1.0 + cxi1 * xi)
    det2 = 0.25 * ceta2 * (1.0 + cxi2 * xi)
    det3 = 0.25 * ceta3 * (1.0 + cxi3 * xi)

    x_xi = dxi0 * x0 + dxi1 * x1 + dxi2 * x2 + dxi3 * x3
    x_eta = det0 * x0 + det1 * x1 + det2 * x2 + det3 * x3
    nvec = wp.cross(x_xi, x_eta)

    for a in range(4):
        fa = (p * Nv[a]) * nvec
        wp.atomic_add(elem_fp, e_global, a * 6 + 0, fa[0])
        wp.atomic_add(elem_fp, e_global, a * 6 + 1, fa[1])
        wp.atomic_add(elem_fp, e_global, a * 6 + 2, fa[2])


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
    elem_nodes: wp.array,
    elem_K: wp.array,
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
# Effective stiffness K_eff = M/beta/dt^2 + (1+alpha)*K_t
# (updated in-place without allocation — see below)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graph-capture path: in-place K_eff update without memory allocation
# ---------------------------------------------------------------------------


@wp.kernel
def zero_bsr_values(values: wp.array[float]):
    """Zero all non-zero values in a BSR matrix (one thread per stored entry)."""
    i = wp.tid()
    values[i] = float(0.0)


@wp.kernel
def scatter_elem_to_bsr(
    elem_K: wp.array3d[float],  # (n_elem, 24, 24)
    scatter_map: wp.array2d[wp.int32],  # (n_elem, 576)
    scale: float,
    out_vals: wp.array[float],  # BsrMatrix.values (nnz,)
):
    """Atomically add scale * elem_K into BSR values using a precomputed scatter map.

    One thread per element.  scatter_map[e, k*24+j] holds the absolute index
    into out_vals for global DOF pair (row_k, col_j); -1 means absent (impossible
    for a well-formed mesh but guarded for safety).
    """
    e = wp.tid()
    for k in range(24):
        for j in range(24):
            idx = scatter_map[e, k * 24 + j]
            if idx >= 0:
                wp.atomic_add(out_vals, idx, scale * elem_K[e, k, j])


def build_scatter_map(
    elem_nodes_wp: wp.array,
    K_eff: wps.BsrMatrix,
    device,
) -> wp.array:
    """Build element→BSR-values scatter map (CPU numpy, one-time in __init__).

    Returns an int32 Warp array of shape (n_elem, 576) where entry [e, k*24+j]
    is the absolute index into K_eff.values for global DOF pair
    (elem_nodes[e, k//6]*6 + k%6,  elem_nodes[e, j//6]*6 + j%6).

    M and K share the same sparsity (both assembled from the same mesh), so a
    single scatter_map works for both.
    """
    offsets = K_eff.offsets.numpy()  # (n_dof + 1,)
    columns = K_eff.columns.numpy()  # (nnz,)
    elem_nodes = elem_nodes_wp.numpy()  # (n_elem, 4)
    n_elem = elem_nodes.shape[0]

    scatter = np.full((n_elem, 576), -1, dtype=np.int32)
    for e in range(n_elem):
        nodes = elem_nodes[e]
        for k_loc in range(24):
            node_k = k_loc // 6
            sk = k_loc % 6
            row = int(nodes[node_k]) * 6 + sk
            rs, re = int(offsets[row]), int(offsets[row + 1])
            row_cols = columns[rs:re]
            for j_loc in range(24):
                node_j = j_loc // 6
                sj = j_loc % 6
                col = int(nodes[node_j]) * 6 + sj
                pos = int(np.searchsorted(row_cols, col))
                if pos < len(row_cols) and row_cols[pos] == col:
                    scatter[e, k_loc * 24 + j_loc] = rs + pos

    return wp.array(scatter, dtype=wp.int32, device=device)


# ---------------------------------------------------------------------------
# N-env batched variants (flat arrays: env * per_env + local index)
# Reference / topology arrays (elem_nodes, scatter_map) are shared across envs.
# ---------------------------------------------------------------------------


@wp.kernel
def scatter_forces_batched(
    elem_nodes: wp.array2d[wp.int32],  # (n_elems, 4) — shared
    elem_f: wp.array2d[float],  # (N*n_elems, 24)
    global_f: wp.array[float],  # (N*n_nodes*6,)
    n_elems: int,
    n_nodes: int,
):
    """dim = N*n_elems.  env = tid // n_elems;  e = tid % n_elems."""
    tid = wp.tid()
    env = tid // n_elems
    e = tid % n_elems
    node_base = env * n_nodes * 6
    for node_local in range(4):
        na = elem_nodes[e, node_local]
        base = node_base + na * 6
        for s in range(6):
            wp.atomic_add(global_f, base + s, elem_f[tid, node_local * 6 + s])


@wp.kernel
def scatter_elem_to_bsr_batched(
    elem_K: wp.array3d[float],  # (N*n_elems, 24, 24)
    scatter_map: wp.array2d[wp.int32],  # (n_elems, 576) — shared
    scale: float,
    out_vals: wp.array[float],  # (N*nnz,) flat
    n_elems: int,
    nnz: int,
):
    """dim = N*n_elems*24.  One thread per (env, element, K-row k)."""
    tid = wp.tid()
    n_per_env = n_elems * 24
    env = tid // n_per_env
    e_row = tid % n_per_env
    e = e_row // 24
    k = e_row % 24
    e_global = env * n_elems + e
    base = env * nnz
    for j in range(24):
        idx = scatter_map[e, k * 24 + j]
        if idx >= 0:
            wp.atomic_add(out_vals, base + idx, scale * elem_K[e_global, k, j])


@wp.kernel
def add_diag_to_bsr_values_batched(
    K_diag: wp.array[float],  # (N*n_dof,)
    bsr_offsets: wp.array[wp.int32],  # (n_dof+1,) — shared
    bsr_columns: wp.array[wp.int32],  # (nnz,)     — shared
    bsr_vals: wp.array[float],  # (N*nnz,) flat
    n_dof: int,
    nnz: int,
):
    """dim = N*n_dof.  env = tid // n_dof;  i = tid % n_dof."""
    tid = wp.tid()
    env = tid // n_dof
    i = tid % n_dof
    base = env * nnz
    diag_val = K_diag[tid]
    for ptr in range(bsr_offsets[i], bsr_offsets[i + 1]):
        if bsr_columns[ptr] == i:
            wp.atomic_add(bsr_vals, base + ptr, diag_val)
            break
            break
