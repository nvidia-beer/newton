# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
ANCF3423 element forces and tangent stiffness kernel.

Kept in a separate module from kernels_element.py so that the lumped-mass
kernel compiles independently (Warp compiles all kernels in a module together
on first launch).

B-row precomputation strategy
------------------------------
Phase 1 (24-DOF loop per GP):
  For each DOF k, compute and store into per-GP scratch:
    gp_bd[e, k, 0..2]  — diagonal B components in material frame
    gp_bs0[e, k]        — in-plane shear component in material frame (2E12m)
    gp_b13[e, k]        — transverse shear b13 in material frame
    gp_b23[e, k]        — transverse shear b23 in material frame
  Then accumulate force f[k] (inline arithmetic, no function calls).

Phase 2 (24×24 DOF-DOF loop per GP):
  Read from scratch — zero function calls, pure scalar multiply-adds.
  This is what matters for nvcc compile time: the inner j-loop body is
  ~25 flops with no inlined device-function calls.

EAS (Enhanced Assumed Strain) + β-transform locking remedies
--------------------------------------------------------------
  - 5 EAS modes: ξ·T0c0, η·T0c1, ζ·T0c2, ξ·T0c3, η·T0c3
    where T0c* are β-transformed unit strains evaluated at the element center.
  - ANS ε_zz: replaced via bilinear interpolation from 4 corner tying points.
  - ANS γ_13, γ_23: replaced via bilinear interpolation from 4 mid-edge tying points.
  - All strains are transformed from natural to material frame via β.
  - EAS α is warm-started and updated by one Newton step per element per call.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})

# Import shared helpers — Warp traces them into this module's CUDA unit.
from .kernels_element import (  # noqa: E402
    _b13_at,
    _b23_at,
    _b33_at,
    _b_diag,
    _b_shear,
    _beta_transform_diag,
    _beta_transform_shear,
    _compute_beta,
    _dshape_deta,
    _dshape_dxi,
    _e13_at,
    _e23_at,
    _e33_at,
    _g_grad,
    _g_pos,
    _gl_shear,
    _gl_strain,
    _gp2,
    _gpw,
    _gpz,
    _jacobian,
    _matmul33,
    _shape,
    _w2,
)


@wp.kernel
def compute_rest_jacobians(
    # Reference state
    node_x0: wp.array[wp.vec3],
    node_D0: wp.array[wp.vec3],
    # Mesh
    elem_nodes: wp.array2d[wp.int32],
    elem_h: wp.array[float],
    elem_fiber_cos: wp.array[float],
    elem_fiber_sin: wp.array[float],
    # Outputs — one per element, computed once at solver init
    out_det_J0c: wp.array[float],
    out_T0c0_d: wp.array[wp.vec3],
    out_T0c0_s: wp.array[wp.vec3],
    out_T0c1_d: wp.array[wp.vec3],
    out_T0c1_s: wp.array[wp.vec3],
    out_T0c2_d: wp.array[wp.vec3],
    out_T0c2_s: wp.array[wp.vec3],
    out_T0c3_d: wp.array[wp.vec3],
    out_T0c3_s: wp.array[wp.vec3],
    out_J0inv_a: wp.array[wp.mat33],
    out_J0inv_b: wp.array[wp.mat33],
    out_J0inv_cc: wp.array[wp.mat33],
    out_J0inv_d: wp.array[wp.mat33],
    out_J0inv_tA: wp.array[wp.mat33],
    out_J0inv_tB: wp.array[wp.mat33],
    out_J0inv_tC: wp.array[wp.mat33],
    out_J0inv_tD: wp.array[wp.mat33],
):
    """One thread per element.  Precomputes the rest-configuration Jacobian
    inverses and EAS T0 basis that ``compute_element_forces_stiffness[_batched_gp]``
    used to recompute from scratch every NR iteration (and, in the GP-parallel
    batched kernel, redundantly again per Gauss-point thread of the same
    element).  Everything here depends only on node_x0/node_D0 (rest
    configuration) and the fiber angle — never on the current deformed state —
    so it's a true per-element constant, computed once at solver init instead
    of up to ~n_gp_total x nr_max_iter times per substep.

    Must be called once after node_x0/node_D0/elem_fiber_cos/elem_fiber_sin
    are finalized, before the first step() — see _precompute_rest_jacobians().
    """
    e = wp.tid()

    na0 = elem_nodes[e, 0]
    na1 = elem_nodes[e, 1]
    na2 = elem_nodes[e, 2]
    na3 = elem_nodes[e, 3]

    X0 = node_x0[na0]
    X1 = node_x0[na1]
    X2 = node_x0[na2]
    X3 = node_x0[na3]
    R0 = node_D0[na0]
    R1 = node_D0[na1]
    R2 = node_D0[na2]
    R3 = node_D0[na3]

    h = elem_h[e]
    cos_t = elem_fiber_cos[e]
    sin_t = elem_fiber_sin[e]

    # --- beta at element center (xi=eta=zeta=0) for EAS T0 columns ---
    N_c = _shape(0.0, 0.0)
    dNxi_c = _dshape_dxi(0.0)
    dNeta_c = _dshape_deta(0.0)
    J0c = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_c, dNeta_c, N_c, 0.0, h)
    det_J0c = wp.abs(wp.determinant(J0c))
    beta_c = _compute_beta(J0c, cos_t, sin_t)

    T0c0_d = _beta_transform_diag(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), beta_c)
    T0c0_s = _beta_transform_shear(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), beta_c)
    T0c1_d = _beta_transform_diag(wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 0.0), beta_c)
    T0c1_s = _beta_transform_shear(wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 0.0), beta_c)
    T0c2_d = _beta_transform_diag(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), beta_c)
    T0c2_s = _beta_transform_shear(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), beta_c)
    T0c3_d = _beta_transform_diag(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), beta_c)
    T0c3_s = _beta_transform_shear(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), beta_c)

    # --- F/J0inv at 4 corners (ANS eps_zz) and 4 mid-edge tying points (ANS shear) ---
    N_a = _shape(-1.0, -1.0)
    dNxi_a = _dshape_dxi(-1.0)
    dNeta_a = _dshape_deta(-1.0)
    J0_a = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_a, dNeta_a, N_a, 0.0, h)
    J0inv_a = wp.inverse(J0_a)

    N_b = _shape(+1.0, -1.0)
    dNxi_b = _dshape_dxi(-1.0)
    dNeta_b = _dshape_deta(+1.0)
    J0_b = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_b, dNeta_b, N_b, 0.0, h)
    J0inv_b = wp.inverse(J0_b)

    N_cc = _shape(+1.0, +1.0)
    dNxi_cc = _dshape_dxi(+1.0)
    dNeta_cc = _dshape_deta(+1.0)
    J0_cc = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_cc, dNeta_cc, N_cc, 0.0, h)
    J0inv_cc = wp.inverse(J0_cc)

    N_d = _shape(-1.0, +1.0)
    dNxi_d = _dshape_dxi(+1.0)
    dNeta_d = _dshape_deta(-1.0)
    J0_d = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_d, dNeta_d, N_d, 0.0, h)
    J0inv_d = wp.inverse(J0_d)

    N_tA = _shape(0.0, -1.0)
    dNxi_tA = _dshape_dxi(-1.0)
    dNeta_tA = _dshape_deta(0.0)
    J0_tA = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_tA, dNeta_tA, N_tA, 0.0, h)
    J0inv_tA = wp.inverse(J0_tA)

    N_tC = _shape(0.0, +1.0)
    dNxi_tC = _dshape_dxi(+1.0)
    dNeta_tC = _dshape_deta(0.0)
    J0_tC = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_tC, dNeta_tC, N_tC, 0.0, h)
    J0inv_tC = wp.inverse(J0_tC)

    N_tB = _shape(+1.0, 0.0)
    dNxi_tB = _dshape_dxi(0.0)
    dNeta_tB = _dshape_deta(+1.0)
    J0_tB = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_tB, dNeta_tB, N_tB, 0.0, h)
    J0inv_tB = wp.inverse(J0_tB)

    N_tD = _shape(-1.0, 0.0)
    dNxi_tD = _dshape_dxi(0.0)
    dNeta_tD = _dshape_deta(-1.0)
    J0_tD = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi_tD, dNeta_tD, N_tD, 0.0, h)
    J0inv_tD = wp.inverse(J0_tD)

    out_det_J0c[e] = det_J0c
    out_T0c0_d[e] = T0c0_d
    out_T0c0_s[e] = T0c0_s
    out_T0c1_d[e] = T0c1_d
    out_T0c1_s[e] = T0c1_s
    out_T0c2_d[e] = T0c2_d
    out_T0c2_s[e] = T0c2_s
    out_T0c3_d[e] = T0c3_d
    out_T0c3_s[e] = T0c3_s
    out_J0inv_a[e] = J0inv_a
    out_J0inv_b[e] = J0inv_b
    out_J0inv_cc[e] = J0inv_cc
    out_J0inv_d[e] = J0inv_d
    out_J0inv_tA[e] = J0inv_tA
    out_J0inv_tB[e] = J0inv_tB
    out_J0inv_tC[e] = J0inv_tC
    out_J0inv_tD[e] = J0inv_tD


@wp.kernel
def compute_element_forces_stiffness(
    # Current state
    node_x: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    # Reference state
    node_x0: wp.array[wp.vec3],
    node_D0: wp.array[wp.vec3],
    # Mesh
    elem_nodes: wp.array2d[wp.int32],
    elem_h: wp.array[float],
    elem_mat: wp.array2d[float],  # (n_elem, 11)
    # Outputs
    elem_f: wp.array2d[float],  # (n_elem, 24)
    elem_K: wp.array3d[float],  # (n_elem, 24, 24)
    # Per-GP B-row scratch (written phase-1, read phase-2)
    gp_bd: wp.array3d[float],  # (n_elem, 24, 3)  b_diag components (material frame)
    gp_bs0: wp.array2d[float],  # (n_elem, 24)     b_shear[0] = 2E12m (material frame)
    gp_b13: wp.array2d[float],  # (n_elem, 24)     b13 (material frame)
    gp_b23: wp.array2d[float],  # (n_elem, 24)     b23 (material frame)
    # EAS + fiber angle (locking remedies)
    elem_eas_alpha: wp.array2d[float],  # (n_elem, 5) in/out EAS state
    elem_fiber_cos: wp.array[float],  # cos(theta) per element
    elem_fiber_sin: wp.array[float],  # sin(theta) per element
    n_gp_z: int,  # through-thickness points: 3 (fast) or 5 (accurate)
    # Rest-configuration Jacobian inverses + EAS T0 basis — precomputed once
    # by compute_rest_jacobians (see its docstring); depends only on rest
    # config + fiber angle, never the current deformed state.
    elem_det_J0c: wp.array[float],
    elem_T0c0_d: wp.array[wp.vec3],
    elem_T0c0_s: wp.array[wp.vec3],
    elem_T0c1_d: wp.array[wp.vec3],
    elem_T0c1_s: wp.array[wp.vec3],
    elem_T0c2_d: wp.array[wp.vec3],
    elem_T0c2_s: wp.array[wp.vec3],
    elem_T0c3_d: wp.array[wp.vec3],
    elem_T0c3_s: wp.array[wp.vec3],
    elem_J0inv_a: wp.array[wp.mat33],
    elem_J0inv_b: wp.array[wp.mat33],
    elem_J0inv_cc: wp.array[wp.mat33],
    elem_J0inv_d: wp.array[wp.mat33],
    elem_J0inv_tA: wp.array[wp.mat33],
    elem_J0inv_tB: wp.array[wp.mat33],
    elem_J0inv_tC: wp.array[wp.mat33],
    elem_J0inv_tD: wp.array[wp.mat33],
):
    """One GPU thread per element.  Accumulates f_int and K_mat."""
    e = wp.tid()

    na0 = elem_nodes[e, 0]
    na1 = elem_nodes[e, 1]
    na2 = elem_nodes[e, 2]
    na3 = elem_nodes[e, 3]

    x0 = node_x[na0]
    x1 = node_x[na1]
    x2 = node_x[na2]
    x3 = node_x[na3]
    D0 = node_D[na0]
    D1 = node_D[na1]
    D2 = node_D[na2]
    D3 = node_D[na3]

    X0 = node_x0[na0]
    X1 = node_x0[na1]
    X2 = node_x0[na2]
    X3 = node_x0[na3]
    R0 = node_D0[na0]
    R1 = node_D0[na1]
    R2 = node_D0[na2]
    R3 = node_D0[na3]

    h = elem_h[e]

    C11 = elem_mat[e, 0]
    C22 = elem_mat[e, 1]
    C33 = elem_mat[e, 2]
    C12 = elem_mat[e, 3]
    C13 = elem_mat[e, 4]
    C23 = elem_mat[e, 5]
    G23 = elem_mat[e, 6]
    G13 = elem_mat[e, 7]
    G12 = elem_mat[e, 8]

    cos_t = elem_fiber_cos[e]
    sin_t = elem_fiber_sin[e]

    # EAS state (warm-started)
    alpha0 = elem_eas_alpha[e, 0]
    alpha1 = elem_eas_alpha[e, 1]
    alpha2 = elem_eas_alpha[e, 2]
    alpha3 = elem_eas_alpha[e, 3]
    alpha4 = elem_eas_alpha[e, 4]

    # --- rest-config Jacobian inverses + EAS T0 basis: precomputed once by
    # compute_rest_jacobians (depends only on rest config + fiber angle) ---
    det_J0c = elem_det_J0c[e]
    T0c0_d = elem_T0c0_d[e]
    T0c0_s = elem_T0c0_s[e]
    T0c1_d = elem_T0c1_d[e]
    T0c1_s = elem_T0c1_s[e]
    T0c2_d = elem_T0c2_d[e]
    T0c2_s = elem_T0c2_s[e]
    T0c3_d = elem_T0c3_d[e]
    T0c3_s = elem_T0c3_s[e]
    J0inv_a = elem_J0inv_a[e]
    J0inv_b = elem_J0inv_b[e]
    J0inv_cc = elem_J0inv_cc[e]
    J0inv_d = elem_J0inv_d[e]
    J0inv_tA = elem_J0inv_tA[e]
    J0inv_tB = elem_J0inv_tB[e]
    J0inv_tC = elem_J0inv_tC[e]
    J0inv_tD = elem_J0inv_tD[e]

    # --- ANS: ε_zz at 4 corners (ζ=0) ---
    e33_c0 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, -1.0, -1.0, h)
    e33_c1 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, +1.0, -1.0, h)
    e33_c2 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, +1.0, +1.0, h)
    e33_c3 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, -1.0, +1.0, h)

    # F at each corner (for ε_zz ANS B-rows) — uses the CURRENT deformed
    # config, so this must stay in the hot loop; only J0inv is precomputed.
    N_a = _shape(-1.0, -1.0)
    dNxi_a = _dshape_dxi(-1.0)
    dNeta_a = _dshape_deta(-1.0)
    J_a = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_a, dNeta_a, N_a, 0.0, h)
    F_a = _matmul33(J_a, J0inv_a)

    N_b = _shape(+1.0, -1.0)
    dNxi_b = _dshape_dxi(-1.0)
    dNeta_b = _dshape_deta(+1.0)
    J_b = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_b, dNeta_b, N_b, 0.0, h)
    F_b = _matmul33(J_b, J0inv_b)

    N_cc = _shape(+1.0, +1.0)
    dNxi_cc = _dshape_dxi(+1.0)
    dNeta_cc = _dshape_deta(+1.0)
    J_cc = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_cc, dNeta_cc, N_cc, 0.0, h)
    F_cc = _matmul33(J_cc, J0inv_cc)

    N_d = _shape(-1.0, +1.0)
    dNxi_d = _dshape_dxi(+1.0)
    dNeta_d = _dshape_deta(-1.0)
    J_d = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_d, dNeta_d, N_d, 0.0, h)
    F_d = _matmul33(J_d, J0inv_d)

    # --- ANS shear tying-point F (ζ=0) — same note as above, J0inv precomputed.
    # Needed so _b13_at / _b23_at linearise the ANS shear strain correctly:
    # the reference gradient g and deformation gradient F must be evaluated at
    # the same tying point used for the strain (not at the Gauss point).
    # tA: (ξ=0, η=-1)  tC: (ξ=0, η=+1)  for γ_13
    # tB: (ξ=+1, η=0)  tD: (ξ=-1, η=0)  for γ_23
    N_tA = _shape(0.0, -1.0)
    dNxi_tA = _dshape_dxi(-1.0)
    dNeta_tA = _dshape_deta(0.0)
    J_tA = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tA, dNeta_tA, N_tA, 0.0, h)
    F_tA = _matmul33(J_tA, J0inv_tA)

    N_tC = _shape(0.0, +1.0)
    dNxi_tC = _dshape_dxi(+1.0)
    dNeta_tC = _dshape_deta(0.0)
    J_tC = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tC, dNeta_tC, N_tC, 0.0, h)
    F_tC = _matmul33(J_tC, J0inv_tC)

    N_tB = _shape(+1.0, 0.0)
    dNxi_tB = _dshape_dxi(0.0)
    dNeta_tB = _dshape_deta(+1.0)
    J_tB = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tB, dNeta_tB, N_tB, 0.0, h)
    F_tB = _matmul33(J_tB, J0inv_tB)

    N_tD = _shape(-1.0, 0.0)
    dNxi_tD = _dshape_dxi(0.0)
    dNeta_tD = _dshape_deta(-1.0)
    J_tD = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tD, dNeta_tD, N_tD, 0.0, h)
    F_tD = _matmul33(J_tD, J0inv_tD)

    # --- Existing ANS for γ_13, γ_23 (4 mid-edge tying points, ζ=0) ---
    e13_A = _e13_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, 0.0, -1.0, h)
    e13_C = _e13_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, 0.0, +1.0, h)
    e23_B = _e23_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, +1.0, 0.0, h)
    e23_D = _e23_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, -1.0, 0.0, h)

    # Zero outputs
    for i in range(24):
        elem_f[e, i] = float(0.0)
        for j in range(24):
            elem_K[e, i, j] = float(0.0)

    # EAS accumulation (for one Newton step)
    HE0 = float(0.0)
    HE1 = float(0.0)
    HE2 = float(0.0)
    HE3 = float(0.0)
    HE4 = float(0.0)
    KA00 = float(0.0)
    KA01 = float(0.0)
    KA02 = float(0.0)
    KA03 = float(0.0)
    KA04 = float(0.0)
    KA11 = float(0.0)
    KA12 = float(0.0)
    KA13 = float(0.0)
    KA14 = float(0.0)
    KA22 = float(0.0)
    KA23 = float(0.0)
    KA24 = float(0.0)
    KA33 = float(0.0)
    KA34 = float(0.0)
    KA44 = float(0.0)

    # === Gauss integration: 2x2 in-plane × n_gp_z through-thickness ===
    for gi in range(2):
        xi = _gp2(gi)
        w_xi = _w2(gi)
        for gj in range(2):
            eta = _gp2(gj)
            w_eta = _w2(gj)

            w_A = 0.5 * (1.0 - eta)
            w_C = 0.5 * (1.0 + eta)
            w_D = 0.5 * (1.0 - xi)
            w_B = 0.5 * (1.0 + xi)

            for gk in range(n_gp_z):
                zeta = _gpz(gk, n_gp_z)
                w_zeta = _gpw(gk, n_gp_z)
                w = w_xi * w_eta * w_zeta

                N = _shape(xi, eta)
                dNxi = _dshape_dxi(eta)
                dNeta = _dshape_deta(xi)

                J0 = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi, dNeta, N, zeta, h)
                J = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi, dNeta, N, zeta, h)

                det_J0 = wp.abs(wp.determinant(J0))
                J0inv = wp.inverse(J0)
                w_detJ0 = w * det_J0

                F = wp.mat33(
                    J[0, 0] * J0inv[0, 0] + J[0, 1] * J0inv[1, 0] + J[0, 2] * J0inv[2, 0],
                    J[0, 0] * J0inv[0, 1] + J[0, 1] * J0inv[1, 1] + J[0, 2] * J0inv[2, 1],
                    J[0, 0] * J0inv[0, 2] + J[0, 1] * J0inv[1, 2] + J[0, 2] * J0inv[2, 2],
                    J[1, 0] * J0inv[0, 0] + J[1, 1] * J0inv[1, 0] + J[1, 2] * J0inv[2, 0],
                    J[1, 0] * J0inv[0, 1] + J[1, 1] * J0inv[1, 1] + J[1, 2] * J0inv[2, 1],
                    J[1, 0] * J0inv[0, 2] + J[1, 1] * J0inv[1, 2] + J[1, 2] * J0inv[2, 2],
                    J[2, 0] * J0inv[0, 0] + J[2, 1] * J0inv[1, 0] + J[2, 2] * J0inv[2, 0],
                    J[2, 0] * J0inv[0, 1] + J[2, 1] * J0inv[1, 1] + J[2, 2] * J0inv[2, 1],
                    J[2, 0] * J0inv[0, 2] + J[2, 1] * J0inv[1, 2] + J[2, 2] * J0inv[2, 2],
                )

                # Natural strains
                e_diag_nat = _gl_strain(F)
                e_shear_nat = _gl_shear(F)

                # ANS: replace ε_zz with corner interpolation
                e33_ans = N[0] * e33_c0 + N[1] * e33_c1 + N[2] * e33_c2 + N[3] * e33_c3
                e_diag_nat = wp.vec3(e_diag_nat[0], e_diag_nat[1], e33_ans)

                # ANS: replace γ_13, γ_23 with mid-edge interpolation
                e13_ans = w_A * e13_A + w_C * e13_C
                e23_ans = w_D * e23_D + w_B * e23_B
                e_shear_nat = wp.vec3(e_shear_nat[0], e13_ans, e23_ans)

                # β at this Gauss point
                beta_gp = _compute_beta(J0, cos_t, sin_t)

                # Transform natural → material strains
                e_d = _beta_transform_diag(e_diag_nat, e_shear_nat, beta_gp)
                e_s = _beta_transform_shear(e_diag_nat, e_shear_nat, beta_gp)

                # EAS enhancement
                eas_scale = det_J0c / wp.max(det_J0, 1.0e-20)
                eas_xi_s = eas_scale * xi
                eas_eta_s = eas_scale * eta
                eas_zeta_s = eas_scale * zeta
                e_d = (
                    e_d
                    + alpha0 * eas_xi_s * T0c0_d
                    + alpha1 * eas_eta_s * T0c1_d
                    + alpha2 * eas_zeta_s * T0c2_d
                    + (alpha3 * eas_xi_s + alpha4 * eas_eta_s) * T0c3_d
                )
                e_s = (
                    e_s
                    + alpha0 * eas_xi_s * T0c0_s
                    + alpha1 * eas_eta_s * T0c1_s
                    + alpha2 * eas_zeta_s * T0c2_s
                    + (alpha3 * eas_xi_s + alpha4 * eas_eta_s) * T0c3_s
                )

                # Material stress S = C·ε  (in material frame)
                S11 = C11 * e_d[0] + C12 * e_d[1] + C13 * e_d[2]
                S22 = C12 * e_d[0] + C22 * e_d[1] + C23 * e_d[2]
                S33 = C13 * e_d[0] + C23 * e_d[1] + C33 * e_d[2]
                S12_2 = G12 * e_s[0]
                S13_2 = G13 * e_s[1]
                S23_2 = G23 * e_s[2]

                S_d = wp.vec3(S11, S22, S33)
                S_s = wp.vec3(S12_2, S13_2, S23_2)

                # EAS G-columns at this GP (scaled T0 columns × natural coordinate × scale)
                G0_d = eas_xi_s * T0c0_d
                G0_s = eas_xi_s * T0c0_s
                G1_d = eas_eta_s * T0c1_d
                G1_s = eas_eta_s * T0c1_s
                G2_d = eas_zeta_s * T0c2_d
                G2_s = eas_zeta_s * T0c2_s
                G3_d = eas_xi_s * T0c3_d
                G3_s = eas_xi_s * T0c3_s
                G4_d = eas_eta_s * T0c3_d
                G4_s = eas_eta_s * T0c3_s

                # HE[k] = G[:,k] · S * w_detJ0
                HE0 = HE0 + (wp.dot(G0_d, S_d) + wp.dot(G0_s, S_s)) * w_detJ0
                HE1 = HE1 + (wp.dot(G1_d, S_d) + wp.dot(G1_s, S_s)) * w_detJ0
                HE2 = HE2 + (wp.dot(G2_d, S_d) + wp.dot(G2_s, S_s)) * w_detJ0
                HE3 = HE3 + (wp.dot(G3_d, S_d) + wp.dot(G3_s, S_s)) * w_detJ0
                HE4 = HE4 + (wp.dot(G4_d, S_d) + wp.dot(G4_s, S_s)) * w_detJ0

                # K_alpha[k,j] = G[:,k] · C · G[:,j] * w_detJ0
                CG0_d = wp.vec3(
                    C11 * G0_d[0] + C12 * G0_d[1] + C13 * G0_d[2],
                    C12 * G0_d[0] + C22 * G0_d[1] + C23 * G0_d[2],
                    C13 * G0_d[0] + C23 * G0_d[1] + C33 * G0_d[2],
                )
                CG0_s = wp.vec3(G12 * G0_s[0], G13 * G0_s[1], G23 * G0_s[2])
                CG1_d = wp.vec3(
                    C11 * G1_d[0] + C12 * G1_d[1] + C13 * G1_d[2],
                    C12 * G1_d[0] + C22 * G1_d[1] + C23 * G1_d[2],
                    C13 * G1_d[0] + C23 * G1_d[1] + C33 * G1_d[2],
                )
                CG1_s = wp.vec3(G12 * G1_s[0], G13 * G1_s[1], G23 * G1_s[2])
                CG2_d = wp.vec3(
                    C11 * G2_d[0] + C12 * G2_d[1] + C13 * G2_d[2],
                    C12 * G2_d[0] + C22 * G2_d[1] + C23 * G2_d[2],
                    C13 * G2_d[0] + C23 * G2_d[1] + C33 * G2_d[2],
                )
                CG2_s = wp.vec3(G12 * G2_s[0], G13 * G2_s[1], G23 * G2_s[2])
                CG3_d = wp.vec3(
                    C11 * G3_d[0] + C12 * G3_d[1] + C13 * G3_d[2],
                    C12 * G3_d[0] + C22 * G3_d[1] + C23 * G3_d[2],
                    C13 * G3_d[0] + C23 * G3_d[1] + C33 * G3_d[2],
                )
                CG3_s = wp.vec3(G12 * G3_s[0], G13 * G3_s[1], G23 * G3_s[2])
                CG4_d = wp.vec3(
                    C11 * G4_d[0] + C12 * G4_d[1] + C13 * G4_d[2],
                    C12 * G4_d[0] + C22 * G4_d[1] + C23 * G4_d[2],
                    C13 * G4_d[0] + C23 * G4_d[1] + C33 * G4_d[2],
                )
                CG4_s = wp.vec3(G12 * G4_s[0], G13 * G4_s[1], G23 * G4_s[2])

                w_dJ = w_detJ0
                KA00 = KA00 + (wp.dot(G0_d, CG0_d) + wp.dot(G0_s, CG0_s)) * w_dJ
                KA01 = KA01 + (wp.dot(G0_d, CG1_d) + wp.dot(G0_s, CG1_s)) * w_dJ
                KA02 = KA02 + (wp.dot(G0_d, CG2_d) + wp.dot(G0_s, CG2_s)) * w_dJ
                KA03 = KA03 + (wp.dot(G0_d, CG3_d) + wp.dot(G0_s, CG3_s)) * w_dJ
                KA04 = KA04 + (wp.dot(G0_d, CG4_d) + wp.dot(G0_s, CG4_s)) * w_dJ
                KA11 = KA11 + (wp.dot(G1_d, CG1_d) + wp.dot(G1_s, CG1_s)) * w_dJ
                KA12 = KA12 + (wp.dot(G1_d, CG2_d) + wp.dot(G1_s, CG2_s)) * w_dJ
                KA13 = KA13 + (wp.dot(G1_d, CG3_d) + wp.dot(G1_s, CG3_s)) * w_dJ
                KA14 = KA14 + (wp.dot(G1_d, CG4_d) + wp.dot(G1_s, CG4_s)) * w_dJ
                KA22 = KA22 + (wp.dot(G2_d, CG2_d) + wp.dot(G2_s, CG2_s)) * w_dJ
                KA23 = KA23 + (wp.dot(G2_d, CG3_d) + wp.dot(G2_s, CG3_s)) * w_dJ
                KA24 = KA24 + (wp.dot(G2_d, CG4_d) + wp.dot(G2_s, CG4_s)) * w_dJ
                KA33 = KA33 + (wp.dot(G3_d, CG3_d) + wp.dot(G3_s, CG3_s)) * w_dJ
                KA34 = KA34 + (wp.dot(G3_d, CG4_d) + wp.dot(G3_s, CG4_s)) * w_dJ
                KA44 = KA44 + (wp.dot(G4_d, CG4_d) + wp.dot(G4_s, CG4_s)) * w_dJ

                # === Phase 1: compute B-rows for all 24 DOFs, store material-frame ===
                for node_k in range(4):
                    Na_k = N[node_k]
                    dNxi_k = dNxi[node_k]
                    dNeta_k = dNeta[node_k]
                    g_p_k = _g_pos(dNxi_k, dNeta_k, J0inv)
                    g_d_k = _g_grad(dNxi_k, dNeta_k, Na_k, h, zeta, J0inv)

                    for is_grad_k in range(2):
                        if is_grad_k == 0:
                            g_k = g_p_k
                        else:
                            g_k = g_d_k

                        for alpha_k in range(3):
                            k = node_k * 6 + is_grad_k * 3 + alpha_k

                            bd_k = _b_diag(g_k, F, alpha_k)
                            bs_k = _b_shear(g_k, F, alpha_k)

                            # ANS transverse shear B-rows: use F and J0inv at each
                            # tying point, not at the current Gauss point — the
                            # linearisation of the ANS-interpolated strain must
                            # be consistent with where the strain was sampled.
                            b13_k = w_A * _b13_at(
                                X0, X1, X2, X3, R0, R1, R2, R3, F_tA, J0inv_tA, 0.0, -1.0, h, node_k, is_grad_k, alpha_k
                            ) + w_C * _b13_at(
                                X0, X1, X2, X3, R0, R1, R2, R3, F_tC, J0inv_tC, 0.0, +1.0, h, node_k, is_grad_k, alpha_k
                            )
                            b23_k = w_D * _b23_at(
                                X0, X1, X2, X3, R0, R1, R2, R3, F_tD, J0inv_tD, -1.0, 0.0, h, node_k, is_grad_k, alpha_k
                            ) + w_B * _b23_at(
                                X0, X1, X2, X3, R0, R1, R2, R3, F_tB, J0inv_tB, +1.0, 0.0, h, node_k, is_grad_k, alpha_k
                            )

                            # ANS ε_zz B-row: interpolate from 4 corners
                            b33_k = (
                                N[0]
                                * _b33_at(
                                    X0,
                                    X1,
                                    X2,
                                    X3,
                                    R0,
                                    R1,
                                    R2,
                                    R3,
                                    F_a,
                                    J0inv_a,
                                    -1.0,
                                    -1.0,
                                    h,
                                    node_k,
                                    is_grad_k,
                                    alpha_k,
                                )
                                + N[1]
                                * _b33_at(
                                    X0,
                                    X1,
                                    X2,
                                    X3,
                                    R0,
                                    R1,
                                    R2,
                                    R3,
                                    F_b,
                                    J0inv_b,
                                    +1.0,
                                    -1.0,
                                    h,
                                    node_k,
                                    is_grad_k,
                                    alpha_k,
                                )
                                + N[2]
                                * _b33_at(
                                    X0,
                                    X1,
                                    X2,
                                    X3,
                                    R0,
                                    R1,
                                    R2,
                                    R3,
                                    F_cc,
                                    J0inv_cc,
                                    +1.0,
                                    +1.0,
                                    h,
                                    node_k,
                                    is_grad_k,
                                    alpha_k,
                                )
                                + N[3]
                                * _b33_at(
                                    X0,
                                    X1,
                                    X2,
                                    X3,
                                    R0,
                                    R1,
                                    R2,
                                    R3,
                                    F_d,
                                    J0inv_d,
                                    -1.0,
                                    +1.0,
                                    h,
                                    node_k,
                                    is_grad_k,
                                    alpha_k,
                                )
                            )

                            # Natural 6-vector B-row: [bd[0], bd[1], b33_ans, bs[0], b13_ans, b23_ans]
                            b_d_nat = wp.vec3(bd_k[0], bd_k[1], b33_k)
                            b_s_nat = wp.vec3(bs_k[0], b13_k, b23_k)

                            # β-transform to material frame
                            b_d_mat = _beta_transform_diag(b_d_nat, b_s_nat, beta_gp)
                            b_s_mat = _beta_transform_shear(b_d_nat, b_s_nat, beta_gp)

                            # Store material-frame B-rows for phase-2 K assembly
                            gp_bd[e, k, 0] = b_d_mat[0]
                            gp_bd[e, k, 1] = b_d_mat[1]
                            gp_bd[e, k, 2] = b_d_mat[2]
                            gp_bs0[e, k] = b_s_mat[0]
                            gp_b13[e, k] = b_s_mat[1]
                            gp_b23[e, k] = b_s_mat[2]

                            # Force contribution (material-frame B · material-frame S)
                            fk = (
                                b_d_mat[0] * S_d[0]
                                + b_d_mat[1] * S_d[1]
                                + b_d_mat[2] * S_d[2]
                                + b_s_mat[0] * S_s[0]
                                + b_s_mat[1] * S_s[1]
                                + b_s_mat[2] * S_s[2]
                            ) * w_detJ0
                            elem_f[e, k] = elem_f[e, k] + fk

                # === Phase 2: K assembly using material-frame B-rows + original C ===
                for k in range(24):
                    bdk0 = gp_bd[e, k, 0]
                    bdk1 = gp_bd[e, k, 1]
                    bdk2 = gp_bd[e, k, 2]
                    bsk0 = gp_bs0[e, k]
                    b13k = gp_b13[e, k]
                    b23k = gp_b23[e, k]
                    for j in range(24):
                        bdj0 = gp_bd[e, j, 0]
                        bdj1 = gp_bd[e, j, 1]
                        bdj2 = gp_bd[e, j, 2]
                        bsj0 = gp_bs0[e, j]
                        b13j = gp_b13[e, j]
                        b23j = gp_b23[e, j]
                        cbj0 = C11 * bdj0 + C12 * bdj1 + C13 * bdj2
                        cbj1 = C12 * bdj0 + C22 * bdj1 + C23 * bdj2
                        cbj2 = C13 * bdj0 + C23 * bdj1 + C33 * bdj2
                        cbj3 = G12 * bsj0
                        cbj4 = G13 * b13j
                        cbj5 = G23 * b23j
                        kval = (
                            bdk0 * cbj0 + bdk1 * cbj1 + bdk2 * cbj2 + bsk0 * cbj3 + b13k * cbj4 + b23k * cbj5
                        ) * w_detJ0
                        elem_K[e, k, j] = elem_K[e, k, j] + kval

    # === EAS one Newton step: solve K_alpha · Δα = HE ===
    # Forward elimination (Gaussian, exploiting symmetry)
    # Pivot on row 0
    inv00 = float(1.0) / wp.max(KA00, 1.0e-30)
    m10 = KA01 * inv00
    m20 = KA02 * inv00
    m30 = KA03 * inv00
    m40 = KA04 * inv00
    KA11 = KA11 - m10 * KA01
    KA12 = KA12 - m10 * KA02
    KA13 = KA13 - m10 * KA03
    KA14 = KA14 - m10 * KA04
    HE1 = HE1 - m10 * HE0
    KA22 = KA22 - m20 * KA02
    KA23 = KA23 - m20 * KA03
    KA24 = KA24 - m20 * KA04
    HE2 = HE2 - m20 * HE0
    KA33 = KA33 - m30 * KA03
    KA34 = KA34 - m30 * KA04
    HE3 = HE3 - m30 * HE0
    KA44 = KA44 - m40 * KA04
    HE4 = HE4 - m40 * HE0
    # Pivot on row 1
    inv11 = float(1.0) / wp.max(KA11, 1.0e-30)
    m21 = KA12 * inv11
    m31 = KA13 * inv11
    m41 = KA14 * inv11
    KA22 = KA22 - m21 * KA12
    KA23 = KA23 - m21 * KA13
    KA24 = KA24 - m21 * KA14
    HE2 = HE2 - m21 * HE1
    KA33 = KA33 - m31 * KA13
    KA34 = KA34 - m31 * KA14
    HE3 = HE3 - m31 * HE1
    KA44 = KA44 - m41 * KA14
    HE4 = HE4 - m41 * HE1
    # Pivot on row 2
    inv22 = float(1.0) / wp.max(KA22, 1.0e-30)
    m32 = KA23 * inv22
    m42 = KA24 * inv22
    KA33 = KA33 - m32 * KA23
    KA34 = KA34 - m32 * KA24
    HE3 = HE3 - m32 * HE2
    KA44 = KA44 - m42 * KA24
    HE4 = HE4 - m42 * HE2
    # Pivot on row 3
    inv33 = float(1.0) / wp.max(KA33, 1.0e-30)
    m43 = KA34 * inv33
    KA44 = KA44 - m43 * KA34
    HE4 = HE4 - m43 * HE3
    # Back-substitution
    da4 = HE4 / wp.max(KA44, 1.0e-30)
    da3 = (HE3 - KA34 * da4) * inv33
    da2 = (HE2 - KA24 * da4 - KA23 * da3) * inv22
    da1 = (HE1 - KA14 * da4 - KA13 * da3 - KA12 * da2) * inv11
    da0 = (HE0 - KA04 * da4 - KA03 * da3 - KA02 * da2 - KA01 * da1) * inv00

    # Update α (warm-started within a timestep, reset to 0 at step start by
    # the solver).  Clamp to ±0.1: the inflated Polaris tyre has ~6% hoop
    # strain so |α| ≈ O(0.06) at peak; 0.1 gives headroom while capping the
    # ill-conditioned-pivot fallout (Gaussian elimination without partial
    # pivoting can produce |Δα| ≫ 1 when a reduced pivot is near zero due to
    # catastrophic cancellation in the forward-elimination row operations).
    _EAS_MAX = float(1.0e-1)
    elem_eas_alpha[e, 0] = wp.clamp(alpha0 - da0, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e, 1] = wp.clamp(alpha1 - da1, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e, 2] = wp.clamp(alpha2 - da2, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e, 3] = wp.clamp(alpha3 - da3, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e, 4] = wp.clamp(alpha4 - da4, -_EAS_MAX, _EAS_MAX)

    # Stiffness-proportional Rayleigh damping: f_damp = alpha_damp * K_t * v
    alpha_d = elem_mat[e, 10]
    xd0 = node_xd[na0]
    xd1 = node_xd[na1]
    xd2 = node_xd[na2]
    xd3 = node_xd[na3]
    Dd0 = node_Dd[na0]
    Dd1 = node_Dd[na1]
    Dd2 = node_Dd[na2]
    Dd3 = node_Dd[na3]
    for i in range(24):
        kv = (
            elem_K[e, i, 0] * xd0[0]
            + elem_K[e, i, 1] * xd0[1]
            + elem_K[e, i, 2] * xd0[2]
            + elem_K[e, i, 3] * Dd0[0]
            + elem_K[e, i, 4] * Dd0[1]
            + elem_K[e, i, 5] * Dd0[2]
            + elem_K[e, i, 6] * xd1[0]
            + elem_K[e, i, 7] * xd1[1]
            + elem_K[e, i, 8] * xd1[2]
            + elem_K[e, i, 9] * Dd1[0]
            + elem_K[e, i, 10] * Dd1[1]
            + elem_K[e, i, 11] * Dd1[2]
            + elem_K[e, i, 12] * xd2[0]
            + elem_K[e, i, 13] * xd2[1]
            + elem_K[e, i, 14] * xd2[2]
            + elem_K[e, i, 15] * Dd2[0]
            + elem_K[e, i, 16] * Dd2[1]
            + elem_K[e, i, 17] * Dd2[2]
            + elem_K[e, i, 18] * xd3[0]
            + elem_K[e, i, 19] * xd3[1]
            + elem_K[e, i, 20] * xd3[2]
            + elem_K[e, i, 21] * Dd3[0]
            + elem_K[e, i, 22] * Dd3[1]
            + elem_K[e, i, 23] * Dd3[2]
        )
        elem_f[e, i] = elem_f[e, i] + alpha_d * kv


# ---------------------------------------------------------------------------
# GP-parallel stiffness kernels
# Restructured for "1 thread per (element, Gauss point)" to improve occupancy.
# Three-kernel sequence:
#   1. _zero_eas_and_accum_batched  — zero EAS alpha + HE/KA accumulators
#   2. compute_element_forces_stiffness_batched_gp — GP body (atomic accum)
#   3. _eas_solve_damping_batched   — EAS Newton step + Rayleigh damping
# ---------------------------------------------------------------------------


@wp.kernel
def _zero_eas_and_accum_batched(
    elem_eas_alpha: wp.array2d[float],  # (Nne, 5)
    elem_HE: wp.array2d[float],  # (Nne, 5)
    elem_KA: wp.array2d[float],  # (Nne, 15)
):
    """dim = Nne.  Zero EAS alpha and HE/KA accumulators before GP kernel.

    KNOWN ISSUE (audit 2026-09-09, AUDIT_ANCF_JEEP.md §1.1): zeroing alpha
    here every NR iteration discards the update written by
    ``_eas_solve_damping_batched``, so the enhanced strain is always 0 and the
    batched element effectively runs without EAS.  Removing the reset was
    tried and made ancf_rigid_mujoco_tires explode: the EAS modes are built in
    the global rather than the natural frame (§1.7) and the 5x5 elimination has
    no pivoting, so enabling it as-is injects clamped ±0.1 strain.  Keep alpha
    at 0 until §1.7 is fixed; then drop the alpha reset.
    """
    e = wp.tid()
    for i in range(5):
        elem_eas_alpha[e, i] = float(0.0)
        elem_HE[e, i] = float(0.0)
    for i in range(15):
        elem_KA[e, i] = float(0.0)


@wp.kernel
def compute_element_forces_stiffness_batched_gp(
    # Current state — flat [N*n_nodes]
    node_x: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    # Reference state — shared [n_nodes]
    node_x0: wp.array[wp.vec3],
    node_D0: wp.array[wp.vec3],
    # Mesh — shared
    elem_nodes: wp.array2d[wp.int32],
    elem_h: wp.array[float],
    elem_mat: wp.array2d[float],
    # Outputs — flat first-dim [N*n_elems]
    elem_f: wp.array2d[float],
    elem_K: wp.array3d[float],
    gp_bd: wp.array3d[float],  # (Nne*n_gp_total, 24, 3)
    gp_bs0: wp.array2d[float],  # (Nne*n_gp_total, 24)
    gp_b13: wp.array2d[float],  # (Nne*n_gp_total, 24)
    gp_b23: wp.array2d[float],  # (Nne*n_gp_total, 24)
    gp_w: wp.array[float],  # (Nne*n_gp_total,) GP quadrature weight * det_J0
    # EAS + fiber — flat first-dim [N*n_elems] / shared
    elem_eas_alpha: wp.array2d[float],
    elem_fiber_cos: wp.array[float],
    elem_fiber_sin: wp.array[float],
    # EAS accumulators — atomic accumulation targets
    elem_HE: wp.array2d[float],  # (Nne, 5)
    elem_KA: wp.array2d[float],  # (Nne, 15)
    # Batch params
    n_elems_per_env: int,
    n_nodes_per_env: int,
    n_gp_total: int,  # = 4 * n_gp_z (total GP count per element)
    n_gp_z: int,  # through-thickness points: 3 (fast) or 5 (accurate)
    # Rest-configuration Jacobian inverses + EAS T0 basis — precomputed once
    # by compute_rest_jacobians (see its docstring); shared across envs (like
    # elem_h/elem_fiber_cos above), indexed by local element index e, NOT
    # e_global.  Depends only on rest config + fiber angle, never the current
    # deformed state — and never on which env, since the rest mesh is shared.
    elem_det_J0c: wp.array[float],
    elem_T0c0_d: wp.array[wp.vec3],
    elem_T0c0_s: wp.array[wp.vec3],
    elem_T0c1_d: wp.array[wp.vec3],
    elem_T0c1_s: wp.array[wp.vec3],
    elem_T0c2_d: wp.array[wp.vec3],
    elem_T0c2_s: wp.array[wp.vec3],
    elem_T0c3_d: wp.array[wp.vec3],
    elem_T0c3_s: wp.array[wp.vec3],
    elem_J0inv_a: wp.array[wp.mat33],
    elem_J0inv_b: wp.array[wp.mat33],
    elem_J0inv_cc: wp.array[wp.mat33],
    elem_J0inv_d: wp.array[wp.mat33],
    elem_J0inv_tA: wp.array[wp.mat33],
    elem_J0inv_tB: wp.array[wp.mat33],
    elem_J0inv_tC: wp.array[wp.mat33],
    elem_J0inv_tD: wp.array[wp.mat33],
):
    """dim = N * n_elems_per_env * n_gp_total. One thread per (element, Gauss point)."""
    tid = wp.tid()
    e_global = tid // n_gp_total
    gp_flat = tid % n_gp_total

    # Decode GP index: flat = gi * 2*n_gp_z + gj * n_gp_z + gk
    gk = gp_flat % n_gp_z
    gj = (gp_flat // n_gp_z) % 2
    gi = gp_flat // (2 * n_gp_z)

    env = e_global // n_elems_per_env
    e = e_global % n_elems_per_env
    node_off = env * n_nodes_per_env

    na0 = elem_nodes[e, 0]
    na1 = elem_nodes[e, 1]
    na2 = elem_nodes[e, 2]
    na3 = elem_nodes[e, 3]

    x0 = node_x[node_off + na0]
    x1 = node_x[node_off + na1]
    x2 = node_x[node_off + na2]
    x3 = node_x[node_off + na3]
    D0 = node_D[node_off + na0]
    D1 = node_D[node_off + na1]
    D2 = node_D[node_off + na2]
    D3 = node_D[node_off + na3]

    X0 = node_x0[na0]
    X1 = node_x0[na1]
    X2 = node_x0[na2]
    X3 = node_x0[na3]
    R0 = node_D0[na0]
    R1 = node_D0[na1]
    R2 = node_D0[na2]
    R3 = node_D0[na3]

    h = elem_h[e]

    C11 = elem_mat[e_global, 0]
    C22 = elem_mat[e_global, 1]
    C33 = elem_mat[e_global, 2]
    C12 = elem_mat[e_global, 3]
    C13 = elem_mat[e_global, 4]
    C23 = elem_mat[e_global, 5]
    G23 = elem_mat[e_global, 6]
    G13 = elem_mat[e_global, 7]
    G12 = elem_mat[e_global, 8]

    cos_t = elem_fiber_cos[e]
    sin_t = elem_fiber_sin[e]

    alpha0 = elem_eas_alpha[e_global, 0]
    alpha1 = elem_eas_alpha[e_global, 1]
    alpha2 = elem_eas_alpha[e_global, 2]
    alpha3 = elem_eas_alpha[e_global, 3]
    alpha4 = elem_eas_alpha[e_global, 4]

    # --- rest-config Jacobian inverses + EAS T0 basis: precomputed once by
    # compute_rest_jacobians (depends only on rest config + fiber angle),
    # shared across envs and across every GP-thread of this same element. ---
    det_J0c = elem_det_J0c[e]
    T0c0_d = elem_T0c0_d[e]
    T0c0_s = elem_T0c0_s[e]
    T0c1_d = elem_T0c1_d[e]
    T0c1_s = elem_T0c1_s[e]
    T0c2_d = elem_T0c2_d[e]
    T0c2_s = elem_T0c2_s[e]
    T0c3_d = elem_T0c3_d[e]
    T0c3_s = elem_T0c3_s[e]
    J0inv_a = elem_J0inv_a[e]
    J0inv_b = elem_J0inv_b[e]
    J0inv_cc = elem_J0inv_cc[e]
    J0inv_d = elem_J0inv_d[e]
    J0inv_tA = elem_J0inv_tA[e]
    J0inv_tB = elem_J0inv_tB[e]
    J0inv_tC = elem_J0inv_tC[e]
    J0inv_tD = elem_J0inv_tD[e]

    e33_c0 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, -1.0, -1.0, h)
    e33_c1 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, +1.0, -1.0, h)
    e33_c2 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, +1.0, +1.0, h)
    e33_c3 = _e33_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, -1.0, +1.0, h)

    # F at each corner/tying point -- uses the CURRENT deformed config, so this
    # must stay in the hot loop; only J0inv is precomputed.
    N_a = _shape(-1.0, -1.0)
    dNxi_a = _dshape_dxi(-1.0)
    dNeta_a = _dshape_deta(-1.0)
    J_a = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_a, dNeta_a, N_a, 0.0, h)
    F_a = _matmul33(J_a, J0inv_a)

    N_b = _shape(+1.0, -1.0)
    dNxi_b = _dshape_dxi(-1.0)
    dNeta_b = _dshape_deta(+1.0)
    J_b = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_b, dNeta_b, N_b, 0.0, h)
    F_b = _matmul33(J_b, J0inv_b)

    N_cc = _shape(+1.0, +1.0)
    dNxi_cc = _dshape_dxi(+1.0)
    dNeta_cc = _dshape_deta(+1.0)
    J_cc = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_cc, dNeta_cc, N_cc, 0.0, h)
    F_cc = _matmul33(J_cc, J0inv_cc)

    N_d = _shape(-1.0, +1.0)
    dNxi_d = _dshape_dxi(+1.0)
    dNeta_d = _dshape_deta(-1.0)
    J_d = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_d, dNeta_d, N_d, 0.0, h)
    F_d = _matmul33(J_d, J0inv_d)

    N_tA = _shape(0.0, -1.0)
    dNxi_tA = _dshape_dxi(-1.0)
    dNeta_tA = _dshape_deta(0.0)
    J_tA = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tA, dNeta_tA, N_tA, 0.0, h)
    F_tA = _matmul33(J_tA, J0inv_tA)

    N_tC = _shape(0.0, +1.0)
    dNxi_tC = _dshape_dxi(+1.0)
    dNeta_tC = _dshape_deta(0.0)
    J_tC = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tC, dNeta_tC, N_tC, 0.0, h)
    F_tC = _matmul33(J_tC, J0inv_tC)

    N_tB = _shape(+1.0, 0.0)
    dNxi_tB = _dshape_dxi(0.0)
    dNeta_tB = _dshape_deta(+1.0)
    J_tB = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tB, dNeta_tB, N_tB, 0.0, h)
    F_tB = _matmul33(J_tB, J0inv_tB)

    N_tD = _shape(-1.0, 0.0)
    dNxi_tD = _dshape_dxi(0.0)
    dNeta_tD = _dshape_deta(-1.0)
    J_tD = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi_tD, dNeta_tD, N_tD, 0.0, h)
    F_tD = _matmul33(J_tD, J0inv_tD)

    e13_A = _e13_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, 0.0, -1.0, h)
    e13_C = _e13_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, 0.0, +1.0, h)
    e23_B = _e23_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, +1.0, 0.0, h)
    e23_D = _e23_at(x0, x1, x2, x3, D0, D1, D2, D3, X0, X1, X2, X3, R0, R1, R2, R3, -1.0, 0.0, h)

    # === Single GP body — no loop, coordinates decoded from tid ===
    xi = _gp2(gi)
    w_xi = _w2(gi)
    eta = _gp2(gj)
    w_eta = _w2(gj)
    zeta = _gpz(gk, n_gp_z)
    w_zeta = _gpw(gk, n_gp_z)
    w = w_xi * w_eta * w_zeta

    w_A = 0.5 * (1.0 - eta)
    w_C = 0.5 * (1.0 + eta)
    w_D = 0.5 * (1.0 - xi)
    w_B = 0.5 * (1.0 + xi)

    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)

    J0 = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi, dNeta, N, zeta, h)
    J = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi, dNeta, N, zeta, h)

    det_J0 = wp.abs(wp.determinant(J0))
    J0inv = wp.inverse(J0)
    w_detJ0 = w * det_J0
    gp_w[tid] = w_detJ0

    F = wp.mat33(
        J[0, 0] * J0inv[0, 0] + J[0, 1] * J0inv[1, 0] + J[0, 2] * J0inv[2, 0],
        J[0, 0] * J0inv[0, 1] + J[0, 1] * J0inv[1, 1] + J[0, 2] * J0inv[2, 1],
        J[0, 0] * J0inv[0, 2] + J[0, 1] * J0inv[1, 2] + J[0, 2] * J0inv[2, 2],
        J[1, 0] * J0inv[0, 0] + J[1, 1] * J0inv[1, 0] + J[1, 2] * J0inv[2, 0],
        J[1, 0] * J0inv[0, 1] + J[1, 1] * J0inv[1, 1] + J[1, 2] * J0inv[2, 1],
        J[1, 0] * J0inv[0, 2] + J[1, 1] * J0inv[1, 2] + J[1, 2] * J0inv[2, 2],
        J[2, 0] * J0inv[0, 0] + J[2, 1] * J0inv[1, 0] + J[2, 2] * J0inv[2, 0],
        J[2, 0] * J0inv[0, 1] + J[2, 1] * J0inv[1, 1] + J[2, 2] * J0inv[2, 1],
        J[2, 0] * J0inv[0, 2] + J[2, 1] * J0inv[1, 2] + J[2, 2] * J0inv[2, 2],
    )

    e_diag_nat = _gl_strain(F)
    e_shear_nat = _gl_shear(F)

    e33_ans = N[0] * e33_c0 + N[1] * e33_c1 + N[2] * e33_c2 + N[3] * e33_c3
    e_diag_nat = wp.vec3(e_diag_nat[0], e_diag_nat[1], e33_ans)

    e13_ans = w_A * e13_A + w_C * e13_C
    e23_ans = w_D * e23_D + w_B * e23_B
    e_shear_nat = wp.vec3(e_shear_nat[0], e13_ans, e23_ans)

    beta_gp = _compute_beta(J0, cos_t, sin_t)

    e_d = _beta_transform_diag(e_diag_nat, e_shear_nat, beta_gp)
    e_s = _beta_transform_shear(e_diag_nat, e_shear_nat, beta_gp)

    eas_scale = det_J0c / wp.max(det_J0, 1.0e-20)
    eas_xi_s = eas_scale * xi
    eas_eta_s = eas_scale * eta
    eas_zeta_s = eas_scale * zeta
    e_d = (
        e_d
        + alpha0 * eas_xi_s * T0c0_d
        + alpha1 * eas_eta_s * T0c1_d
        + alpha2 * eas_zeta_s * T0c2_d
        + (alpha3 * eas_xi_s + alpha4 * eas_eta_s) * T0c3_d
    )
    e_s = (
        e_s
        + alpha0 * eas_xi_s * T0c0_s
        + alpha1 * eas_eta_s * T0c1_s
        + alpha2 * eas_zeta_s * T0c2_s
        + (alpha3 * eas_xi_s + alpha4 * eas_eta_s) * T0c3_s
    )

    S11 = C11 * e_d[0] + C12 * e_d[1] + C13 * e_d[2]
    S22 = C12 * e_d[0] + C22 * e_d[1] + C23 * e_d[2]
    S33 = C13 * e_d[0] + C23 * e_d[1] + C33 * e_d[2]
    S12_2 = G12 * e_s[0]
    S13_2 = G13 * e_s[1]
    S23_2 = G23 * e_s[2]

    S_d = wp.vec3(S11, S22, S33)
    S_s = wp.vec3(S12_2, S13_2, S23_2)

    G0_d = eas_xi_s * T0c0_d
    G0_s = eas_xi_s * T0c0_s
    G1_d = eas_eta_s * T0c1_d
    G1_s = eas_eta_s * T0c1_s
    G2_d = eas_zeta_s * T0c2_d
    G2_s = eas_zeta_s * T0c2_s
    G3_d = eas_xi_s * T0c3_d
    G3_s = eas_xi_s * T0c3_s
    G4_d = eas_eta_s * T0c3_d
    G4_s = eas_eta_s * T0c3_s

    # HE atomic accumulation
    wp.atomic_add(elem_HE, e_global, 0, (wp.dot(G0_d, S_d) + wp.dot(G0_s, S_s)) * w_detJ0)
    wp.atomic_add(elem_HE, e_global, 1, (wp.dot(G1_d, S_d) + wp.dot(G1_s, S_s)) * w_detJ0)
    wp.atomic_add(elem_HE, e_global, 2, (wp.dot(G2_d, S_d) + wp.dot(G2_s, S_s)) * w_detJ0)
    wp.atomic_add(elem_HE, e_global, 3, (wp.dot(G3_d, S_d) + wp.dot(G3_s, S_s)) * w_detJ0)
    wp.atomic_add(elem_HE, e_global, 4, (wp.dot(G4_d, S_d) + wp.dot(G4_s, S_s)) * w_detJ0)

    CG0_d = wp.vec3(
        C11 * G0_d[0] + C12 * G0_d[1] + C13 * G0_d[2],
        C12 * G0_d[0] + C22 * G0_d[1] + C23 * G0_d[2],
        C13 * G0_d[0] + C23 * G0_d[1] + C33 * G0_d[2],
    )
    CG0_s = wp.vec3(G12 * G0_s[0], G13 * G0_s[1], G23 * G0_s[2])
    CG1_d = wp.vec3(
        C11 * G1_d[0] + C12 * G1_d[1] + C13 * G1_d[2],
        C12 * G1_d[0] + C22 * G1_d[1] + C23 * G1_d[2],
        C13 * G1_d[0] + C23 * G1_d[1] + C33 * G1_d[2],
    )
    CG1_s = wp.vec3(G12 * G1_s[0], G13 * G1_s[1], G23 * G1_s[2])
    CG2_d = wp.vec3(
        C11 * G2_d[0] + C12 * G2_d[1] + C13 * G2_d[2],
        C12 * G2_d[0] + C22 * G2_d[1] + C23 * G2_d[2],
        C13 * G2_d[0] + C23 * G2_d[1] + C33 * G2_d[2],
    )
    CG2_s = wp.vec3(G12 * G2_s[0], G13 * G2_s[1], G23 * G2_s[2])
    CG3_d = wp.vec3(
        C11 * G3_d[0] + C12 * G3_d[1] + C13 * G3_d[2],
        C12 * G3_d[0] + C22 * G3_d[1] + C23 * G3_d[2],
        C13 * G3_d[0] + C23 * G3_d[1] + C33 * G3_d[2],
    )
    CG3_s = wp.vec3(G12 * G3_s[0], G13 * G3_s[1], G23 * G3_s[2])
    CG4_d = wp.vec3(
        C11 * G4_d[0] + C12 * G4_d[1] + C13 * G4_d[2],
        C12 * G4_d[0] + C22 * G4_d[1] + C23 * G4_d[2],
        C13 * G4_d[0] + C23 * G4_d[1] + C33 * G4_d[2],
    )
    CG4_s = wp.vec3(G12 * G4_s[0], G13 * G4_s[1], G23 * G4_s[2])

    w_dJ = w_detJ0
    # KA atomic accumulation — flat index mapping: 00→0,01→1,02→2,03→3,04→4,
    #   11→5,12→6,13→7,14→8, 22→9,23→10,24→11, 33→12,34→13, 44→14
    wp.atomic_add(elem_KA, e_global, 0, (wp.dot(G0_d, CG0_d) + wp.dot(G0_s, CG0_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 1, (wp.dot(G0_d, CG1_d) + wp.dot(G0_s, CG1_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 2, (wp.dot(G0_d, CG2_d) + wp.dot(G0_s, CG2_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 3, (wp.dot(G0_d, CG3_d) + wp.dot(G0_s, CG3_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 4, (wp.dot(G0_d, CG4_d) + wp.dot(G0_s, CG4_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 5, (wp.dot(G1_d, CG1_d) + wp.dot(G1_s, CG1_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 6, (wp.dot(G1_d, CG2_d) + wp.dot(G1_s, CG2_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 7, (wp.dot(G1_d, CG3_d) + wp.dot(G1_s, CG3_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 8, (wp.dot(G1_d, CG4_d) + wp.dot(G1_s, CG4_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 9, (wp.dot(G2_d, CG2_d) + wp.dot(G2_s, CG2_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 10, (wp.dot(G2_d, CG3_d) + wp.dot(G2_s, CG3_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 11, (wp.dot(G2_d, CG4_d) + wp.dot(G2_s, CG4_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 12, (wp.dot(G3_d, CG3_d) + wp.dot(G3_s, CG3_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 13, (wp.dot(G3_d, CG4_d) + wp.dot(G3_s, CG4_s)) * w_dJ)
    wp.atomic_add(elem_KA, e_global, 14, (wp.dot(G4_d, CG4_d) + wp.dot(G4_s, CG4_s)) * w_dJ)

    # === Phase 1: compute B-rows for all 24 DOFs, store to tid slot ===
    for node_k in range(4):
        Na_k = N[node_k]
        dNxi_k = dNxi[node_k]
        dNeta_k = dNeta[node_k]
        g_p_k = _g_pos(dNxi_k, dNeta_k, J0inv)
        g_d_k = _g_grad(dNxi_k, dNeta_k, Na_k, h, zeta, J0inv)

        for is_grad_k in range(2):
            if is_grad_k == 0:
                g_k = g_p_k
            else:
                g_k = g_d_k

            for alpha_k in range(3):
                k = node_k * 6 + is_grad_k * 3 + alpha_k

                bd_k = _b_diag(g_k, F, alpha_k)
                bs_k = _b_shear(g_k, F, alpha_k)

                b13_k = w_A * _b13_at(
                    X0, X1, X2, X3, R0, R1, R2, R3, F_tA, J0inv_tA, 0.0, -1.0, h, node_k, is_grad_k, alpha_k
                ) + w_C * _b13_at(
                    X0, X1, X2, X3, R0, R1, R2, R3, F_tC, J0inv_tC, 0.0, +1.0, h, node_k, is_grad_k, alpha_k
                )
                b23_k = w_D * _b23_at(
                    X0, X1, X2, X3, R0, R1, R2, R3, F_tD, J0inv_tD, -1.0, 0.0, h, node_k, is_grad_k, alpha_k
                ) + w_B * _b23_at(
                    X0, X1, X2, X3, R0, R1, R2, R3, F_tB, J0inv_tB, +1.0, 0.0, h, node_k, is_grad_k, alpha_k
                )

                b33_k = (
                    N[0]
                    * _b33_at(X0, X1, X2, X3, R0, R1, R2, R3, F_a, J0inv_a, -1.0, -1.0, h, node_k, is_grad_k, alpha_k)
                    + N[1]
                    * _b33_at(X0, X1, X2, X3, R0, R1, R2, R3, F_b, J0inv_b, +1.0, -1.0, h, node_k, is_grad_k, alpha_k)
                    + N[2]
                    * _b33_at(X0, X1, X2, X3, R0, R1, R2, R3, F_cc, J0inv_cc, +1.0, +1.0, h, node_k, is_grad_k, alpha_k)
                    + N[3]
                    * _b33_at(X0, X1, X2, X3, R0, R1, R2, R3, F_d, J0inv_d, -1.0, +1.0, h, node_k, is_grad_k, alpha_k)
                )

                b_d_nat = wp.vec3(bd_k[0], bd_k[1], b33_k)
                b_s_nat = wp.vec3(bs_k[0], b13_k, b23_k)

                b_d_mat = _beta_transform_diag(b_d_nat, b_s_nat, beta_gp)
                b_s_mat = _beta_transform_shear(b_d_nat, b_s_nat, beta_gp)

                # Store to this thread's tid slot (each GP has its own row)
                gp_bd[tid, k, 0] = b_d_mat[0]
                gp_bd[tid, k, 1] = b_d_mat[1]
                gp_bd[tid, k, 2] = b_d_mat[2]
                gp_bs0[tid, k] = b_s_mat[0]
                gp_b13[tid, k] = b_s_mat[1]
                gp_b23[tid, k] = b_s_mat[2]

                # Force contribution — atomic add across GP threads for same element
                fk = (
                    b_d_mat[0] * S_d[0]
                    + b_d_mat[1] * S_d[1]
                    + b_d_mat[2] * S_d[2]
                    + b_s_mat[0] * S_s[0]
                    + b_s_mat[1] * S_s[1]
                    + b_s_mat[2] * S_s[2]
                ) * w_detJ0
                wp.atomic_add(elem_f, e_global, k, fk)


@wp.kernel
def compute_element_K_from_B(
    gp_bd: wp.array3d[float],  # (Nne*n_gp_total, 24, 3)
    gp_bs0: wp.array2d[float],  # (Nne*n_gp_total, 24)
    gp_b13: wp.array2d[float],  # (Nne*n_gp_total, 24)
    gp_b23: wp.array2d[float],  # (Nne*n_gp_total, 24)
    gp_w: wp.array[float],  # (Nne*n_gp_total,)
    elem_mat: wp.array2d[float],  # (Nne, n_mat)
    elem_K: wp.array3d[float],  # (Nne, 24, 24) — written directly (no atomic)
    n_gp_total: int,
):
    """dim = Nne*576. One thread per (element, k, j) K entry. No atomics.

    Eliminates the 6912-atomic-per-element contention in Phase 2 by giving
    each K[e,k,j] its own thread that loops over GPs sequentially.
    SM fill: 0.6x (old thread-per-GP) → 27x (460K threads).
    """
    tid = wp.tid()
    e_global = tid // 576
    kj = tid % 576
    k = kj // 24
    j = kj % 24

    C11 = elem_mat[e_global, 0]
    C22 = elem_mat[e_global, 1]
    C33 = elem_mat[e_global, 2]
    C12 = elem_mat[e_global, 3]
    C13 = elem_mat[e_global, 4]
    C23 = elem_mat[e_global, 5]
    G23 = elem_mat[e_global, 6]
    G13 = elem_mat[e_global, 7]
    G12 = elem_mat[e_global, 8]

    acc = float(0.0)
    gp_base = e_global * n_gp_total
    for gp in range(n_gp_total):
        gp_tid = gp_base + gp
        w = gp_w[gp_tid]

        bdk0 = gp_bd[gp_tid, k, 0]
        bdk1 = gp_bd[gp_tid, k, 1]
        bdk2 = gp_bd[gp_tid, k, 2]
        bsk0 = gp_bs0[gp_tid, k]
        b13k = gp_b13[gp_tid, k]
        b23k = gp_b23[gp_tid, k]

        bdj0 = gp_bd[gp_tid, j, 0]
        bdj1 = gp_bd[gp_tid, j, 1]
        bdj2 = gp_bd[gp_tid, j, 2]
        bsj0 = gp_bs0[gp_tid, j]
        b13j = gp_b13[gp_tid, j]
        b23j = gp_b23[gp_tid, j]

        cbj0 = C11 * bdj0 + C12 * bdj1 + C13 * bdj2
        cbj1 = C12 * bdj0 + C22 * bdj1 + C23 * bdj2
        cbj2 = C13 * bdj0 + C23 * bdj1 + C33 * bdj2
        cbj3 = G12 * bsj0
        cbj4 = G13 * b13j
        cbj5 = G23 * b23j

        acc = acc + (bdk0 * cbj0 + bdk1 * cbj1 + bdk2 * cbj2 + bsk0 * cbj3 + b13k * cbj4 + b23k * cbj5) * w

    elem_K[e_global, k, j] = acc


@wp.kernel
def _eas_solve_damping_batched(
    elem_HE: wp.array2d[float],  # (Nne, 5) — accumulated per GP
    elem_KA: wp.array2d[float],  # (Nne, 15) — accumulated per GP
    elem_eas_alpha: wp.array2d[float],  # (Nne, 5) — in=0, out=updated
    elem_mat: wp.array2d[float],  # (Nne, 11) for alpha_damp
    node_xd: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    elem_nodes: wp.array2d[wp.int32],
    elem_K: wp.array3d[float],  # assembled K (read)
    elem_f: wp.array2d[float],  # in/out forces
    n_elems_per_env: int,
    n_nodes_per_env: int,
):
    """dim = Nne.  EAS Newton step + Rayleigh damping after GP accumulation."""
    e_global = wp.tid()
    env = e_global // n_elems_per_env
    e = e_global % n_elems_per_env
    node_off = env * n_nodes_per_env

    # Read EAS accumulators
    HE0 = elem_HE[e_global, 0]
    HE1 = elem_HE[e_global, 1]
    HE2 = elem_HE[e_global, 2]
    HE3 = elem_HE[e_global, 3]
    HE4 = elem_HE[e_global, 4]
    KA00 = elem_KA[e_global, 0]
    KA01 = elem_KA[e_global, 1]
    KA02 = elem_KA[e_global, 2]
    KA03 = elem_KA[e_global, 3]
    KA04 = elem_KA[e_global, 4]
    KA11 = elem_KA[e_global, 5]
    KA12 = elem_KA[e_global, 6]
    KA13 = elem_KA[e_global, 7]
    KA14 = elem_KA[e_global, 8]
    KA22 = elem_KA[e_global, 9]
    KA23 = elem_KA[e_global, 10]
    KA24 = elem_KA[e_global, 11]
    KA33 = elem_KA[e_global, 12]
    KA34 = elem_KA[e_global, 13]
    KA44 = elem_KA[e_global, 14]

    alpha0 = elem_eas_alpha[e_global, 0]
    alpha1 = elem_eas_alpha[e_global, 1]
    alpha2 = elem_eas_alpha[e_global, 2]
    alpha3 = elem_eas_alpha[e_global, 3]
    alpha4 = elem_eas_alpha[e_global, 4]

    # === EAS one Newton step: solve K_alpha · Δα = HE ===
    # Forward elimination (Gaussian, exploiting symmetry)
    inv00 = float(1.0) / wp.max(KA00, 1.0e-30)
    m10 = KA01 * inv00
    m20 = KA02 * inv00
    m30 = KA03 * inv00
    m40 = KA04 * inv00
    KA11 = KA11 - m10 * KA01
    KA12 = KA12 - m10 * KA02
    KA13 = KA13 - m10 * KA03
    KA14 = KA14 - m10 * KA04
    HE1 = HE1 - m10 * HE0
    KA22 = KA22 - m20 * KA02
    KA23 = KA23 - m20 * KA03
    KA24 = KA24 - m20 * KA04
    HE2 = HE2 - m20 * HE0
    KA33 = KA33 - m30 * KA03
    KA34 = KA34 - m30 * KA04
    HE3 = HE3 - m30 * HE0
    KA44 = KA44 - m40 * KA04
    HE4 = HE4 - m40 * HE0
    inv11 = float(1.0) / wp.max(KA11, 1.0e-30)
    m21 = KA12 * inv11
    m31 = KA13 * inv11
    m41 = KA14 * inv11
    KA22 = KA22 - m21 * KA12
    KA23 = KA23 - m21 * KA13
    KA24 = KA24 - m21 * KA14
    HE2 = HE2 - m21 * HE1
    KA33 = KA33 - m31 * KA13
    KA34 = KA34 - m31 * KA14
    HE3 = HE3 - m31 * HE1
    KA44 = KA44 - m41 * KA14
    HE4 = HE4 - m41 * HE1
    inv22 = float(1.0) / wp.max(KA22, 1.0e-30)
    m32 = KA23 * inv22
    m42 = KA24 * inv22
    KA33 = KA33 - m32 * KA23
    KA34 = KA34 - m32 * KA24
    HE3 = HE3 - m32 * HE2
    KA44 = KA44 - m42 * KA24
    HE4 = HE4 - m42 * HE2
    inv33 = float(1.0) / wp.max(KA33, 1.0e-30)
    m43 = KA34 * inv33
    KA44 = KA44 - m43 * KA34
    HE4 = HE4 - m43 * HE3
    da4 = HE4 / wp.max(KA44, 1.0e-30)
    da3 = (HE3 - KA34 * da4) * inv33
    da2 = (HE2 - KA24 * da4 - KA23 * da3) * inv22
    da1 = (HE1 - KA14 * da4 - KA13 * da3 - KA12 * da2) * inv11
    da0 = (HE0 - KA04 * da4 - KA03 * da3 - KA02 * da2 - KA01 * da1) * inv00

    _EAS_MAX = float(1.0e-1)
    elem_eas_alpha[e_global, 0] = wp.clamp(alpha0 - da0, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e_global, 1] = wp.clamp(alpha1 - da1, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e_global, 2] = wp.clamp(alpha2 - da2, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e_global, 3] = wp.clamp(alpha3 - da3, -_EAS_MAX, _EAS_MAX)
    elem_eas_alpha[e_global, 4] = wp.clamp(alpha4 - da4, -_EAS_MAX, _EAS_MAX)

    # Stiffness-proportional Rayleigh damping: f_damp = alpha_damp * K_t * v
    na0 = elem_nodes[e, 0]
    na1 = elem_nodes[e, 1]
    na2 = elem_nodes[e, 2]
    na3 = elem_nodes[e, 3]
    alpha_d = elem_mat[e_global, 10]
    xd0 = node_xd[node_off + na0]
    xd1 = node_xd[node_off + na1]
    xd2 = node_xd[node_off + na2]
    xd3 = node_xd[node_off + na3]
    Dd0 = node_Dd[node_off + na0]
    Dd1 = node_Dd[node_off + na1]
    Dd2 = node_Dd[node_off + na2]
    Dd3 = node_Dd[node_off + na3]
    for i in range(24):
        kv = (
            elem_K[e_global, i, 0] * xd0[0]
            + elem_K[e_global, i, 1] * xd0[1]
            + elem_K[e_global, i, 2] * xd0[2]
            + elem_K[e_global, i, 3] * Dd0[0]
            + elem_K[e_global, i, 4] * Dd0[1]
            + elem_K[e_global, i, 5] * Dd0[2]
            + elem_K[e_global, i, 6] * xd1[0]
            + elem_K[e_global, i, 7] * xd1[1]
            + elem_K[e_global, i, 8] * xd1[2]
            + elem_K[e_global, i, 9] * Dd1[0]
            + elem_K[e_global, i, 10] * Dd1[1]
            + elem_K[e_global, i, 11] * Dd1[2]
            + elem_K[e_global, i, 12] * xd2[0]
            + elem_K[e_global, i, 13] * xd2[1]
            + elem_K[e_global, i, 14] * xd2[2]
            + elem_K[e_global, i, 15] * Dd2[0]
            + elem_K[e_global, i, 16] * Dd2[1]
            + elem_K[e_global, i, 17] * Dd2[2]
            + elem_K[e_global, i, 18] * xd3[0]
            + elem_K[e_global, i, 19] * xd3[1]
            + elem_K[e_global, i, 20] * xd3[2]
            + elem_K[e_global, i, 21] * Dd3[0]
            + elem_K[e_global, i, 22] * Dd3[1]
            + elem_K[e_global, i, 23] * Dd3[2]
        )
        elem_f[e_global, i] = elem_f[e_global, i] + alpha_d * kv
