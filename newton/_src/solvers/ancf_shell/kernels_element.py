# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
ANCF3423 element kernels: shape functions, Green-Lagrange strain, ANS shear
correction, internal force vector, and analytical tangent stiffness matrix.

Element DOF layout (24 per element):
    q_e = [p0, D0, p1, D1, p2, D2, p3, D3]
    DOF k = node_a * 6 + sub_dof
    sub_dof ∈ {0,1,2} → position component  (p_{a,alpha})
    sub_dof ∈ {3,4,5} → gradient component  (D_{a,alpha-3})

Voigt strain ordering used throughout:
    e = [E11, E22, E33, 2·E12, 2·E13, 2·E23]
    (indices: 1=ξ, 2=η, 3=ζ in element parameter space)

References:
    Gruber et al. (2017) "A novel director-based Bernoulli-Euler beam finite
    element in absolute nodal coordinate formulation free of geometric
    singularities." Mech. Sci.

    Chrono fea/ChElementShellANCF_3423.cpp — ANS tying points A/B/C/D and
    through-thickness Gauss quadrature (5-point rule).
"""

import warp as wp

# Disable autodiff backward pass — we don't use gradients and generating it
# roughly doubles the CUDA code that nvcc has to optimise.
wp.set_module_options({"enable_backward": False})

# ---------------------------------------------------------------------------
# Gauss quadrature tables — exposed as @wp.func because Warp cannot index
# Python tuples with runtime variables inside kernels.
# ---------------------------------------------------------------------------


@wp.func
def _gp2(i: int) -> float:
    """In-plane 2-point Gauss abscissae on [-1,1]."""
    if i == 0:
        return float(-0.5773502692)
    return float(0.5773502692)


@wp.func
def _w2(i: int) -> float:
    """In-plane 2-point Gauss weights (both = 1.0)."""
    return float(1.0)


@wp.func
def _gp5z(i: int) -> float:
    """Through-thickness 5-point Gauss-Legendre abscissae on [-1,1]."""
    if i == 0:
        return float(-0.9061798459)
    if i == 1:
        return float(-0.5384693101)
    if i == 2:
        return float(0.0)
    if i == 3:
        return float(0.5384693101)
    return float(0.9061798459)


@wp.func
def _gp5w(i: int) -> float:
    """Through-thickness 5-point Gauss-Legendre weights."""
    if i == 0:
        return float(0.2369268851)
    if i == 1:
        return float(0.4786286705)
    if i == 2:
        return float(0.5688888889)
    if i == 3:
        return float(0.4786286705)
    return float(0.2369268851)


@wp.func
def _gp3z(i: int) -> float:
    """Through-thickness 3-point Gauss-Legendre abscissae on [-1,1].
    ~33% faster than 5-point; sufficient for thin shells (h/L < 0.05).
    """
    if i == 0:
        return float(-0.7745966692)
    if i == 1:
        return float(0.0)
    return float(0.7745966692)


@wp.func
def _gp3w(i: int) -> float:
    """Through-thickness 3-point Gauss-Legendre weights."""
    if i == 0:
        return float(0.5555555556)
    if i == 1:
        return float(0.8888888889)
    return float(0.5555555556)


@wp.func
def _gpz(i: int, n: int) -> float:
    """Dispatch abscissa for n=3 or n=5 through-thickness Gauss rule."""
    if n == 3:
        return _gp3z(i)
    return _gp5z(i)


@wp.func
def _gpw(i: int, n: int) -> float:
    """Dispatch weight for n=3 or n=5 through-thickness Gauss rule."""
    if n == 3:
        return _gp3w(i)
    return _gp5w(i)


# ---------------------------------------------------------------------------
# @wp.func helpers — shape functions and Jacobian
# ---------------------------------------------------------------------------


@wp.func
def _shape(xi: float, eta: float) -> wp.vec4:
    """Bilinear shape functions N0..N3 at (xi, eta)."""
    return wp.vec4(
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta),
    )


@wp.func
def _dshape_dxi(eta: float) -> wp.vec4:
    """∂N/∂ξ at given eta."""
    return wp.vec4(
        -0.25 * (1.0 - eta),
        0.25 * (1.0 - eta),
        0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta),
    )


@wp.func
def _dshape_deta(xi: float) -> wp.vec4:
    """∂N/∂η at given xi."""
    return wp.vec4(
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
        0.25 * (1.0 + xi),
        0.25 * (1.0 - xi),
    )


@wp.func
def _jacobian(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    D0: wp.vec3,
    D1: wp.vec3,
    D2: wp.vec3,
    D3: wp.vec3,
    dNxi: wp.vec4,
    dNeta: wp.vec4,
    N: wp.vec4,
    zeta: float,
    h: float,
) -> wp.mat33:
    """Covariant Jacobian matrix J with columns [∂r/∂ξ, ∂r/∂η, ∂r/∂ζ].

    r(ξ,η,ζ) = Σ Ni·(xi + ζ·h/2·Di)
    """
    hz = 0.5 * h * zeta
    r_xi = dNxi[0] * (x0 + hz * D0) + dNxi[1] * (x1 + hz * D1) + dNxi[2] * (x2 + hz * D2) + dNxi[3] * (x3 + hz * D3)
    r_eta = (
        dNeta[0] * (x0 + hz * D0) + dNeta[1] * (x1 + hz * D1) + dNeta[2] * (x2 + hz * D2) + dNeta[3] * (x3 + hz * D3)
    )
    r_zeta = 0.5 * h * (N[0] * D0 + N[1] * D1 + N[2] * D2 + N[3] * D3)
    return wp.mat33(
        r_xi[0],
        r_eta[0],
        r_zeta[0],
        r_xi[1],
        r_eta[1],
        r_zeta[1],
        r_xi[2],
        r_eta[2],
        r_zeta[2],
    )


@wp.func
def _matmul33(A: wp.mat33, B: wp.mat33) -> wp.mat33:
    """Explicit mat33 × mat33 product (avoids ambiguity with element-wise *)."""
    return wp.mat33(
        A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0] + A[0, 2] * B[2, 0],
        A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1] + A[0, 2] * B[2, 1],
        A[0, 0] * B[0, 2] + A[0, 1] * B[1, 2] + A[0, 2] * B[2, 2],
        A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0] + A[1, 2] * B[2, 0],
        A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1] + A[1, 2] * B[2, 1],
        A[1, 0] * B[0, 2] + A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2],
        A[2, 0] * B[0, 0] + A[2, 1] * B[1, 0] + A[2, 2] * B[2, 0],
        A[2, 0] * B[0, 1] + A[2, 1] * B[1, 1] + A[2, 2] * B[2, 1],
        A[2, 0] * B[0, 2] + A[2, 1] * B[1, 2] + A[2, 2] * B[2, 2],
    )


# ---------------------------------------------------------------------------
# Green-Lagrange strain in Voigt notation
# ---------------------------------------------------------------------------


@wp.func
def _gl_strain(F: wp.mat33) -> wp.vec3:
    """Diagonal Green-Lagrange components: [E11, E22, E33]."""
    return wp.vec3(
        0.5 * (F[0, 0] * F[0, 0] + F[1, 0] * F[1, 0] + F[2, 0] * F[2, 0] - 1.0),
        0.5 * (F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1] + F[2, 1] * F[2, 1] - 1.0),
        0.5 * (F[0, 2] * F[0, 2] + F[1, 2] * F[1, 2] + F[2, 2] * F[2, 2] - 1.0),
    )


@wp.func
def _gl_shear(F: wp.mat33) -> wp.vec3:
    """Off-diagonal Green-Lagrange: [2·E12, 2·E13, 2·E23]."""
    return wp.vec3(
        F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1] + F[2, 0] * F[2, 1],  # 2·E12
        F[0, 0] * F[0, 2] + F[1, 0] * F[1, 2] + F[2, 0] * F[2, 2],  # 2·E13
        F[0, 1] * F[0, 2] + F[1, 1] * F[1, 2] + F[2, 1] * F[2, 2],  # 2·E23
    )


# ---------------------------------------------------------------------------
# Orthotropic stress S = C·e  (2nd Piola-Kirchhoff, Voigt)
# Material is read from a flat float array: [C11,C22,C33,C12,C13,C23,G23,G13,G12,rho,alpha]
# ---------------------------------------------------------------------------


@wp.func
def _stress(
    e_diag: wp.vec3,
    e_shear: wp.vec3,
    C11: float,
    C22: float,
    C33: float,
    C12: float,
    C13: float,
    C23: float,
    G23: float,
    G13: float,
    G12: float,
) -> wp.mat33:
    """Return S packed as 3×3: diag=[S11,S22,S33], off=[2S12,2S13,2S23] in rows 0..2.

    We return a mat33 to avoid needing a vec6 type:
        row 0 → [S11,  S22,  S33 ]
        row 1 → [2S12, 2S13, 2S23]
    (row 2 unused)
    """
    S11 = C11 * e_diag[0] + C12 * e_diag[1] + C13 * e_diag[2]
    S22 = C12 * e_diag[0] + C22 * e_diag[1] + C23 * e_diag[2]
    S33 = C13 * e_diag[0] + C23 * e_diag[1] + C33 * e_diag[2]
    S12_2 = G12 * e_shear[0]
    S13_2 = G13 * e_shear[1]
    S23_2 = G23 * e_shear[2]
    return wp.mat33(
        S11,
        S22,
        S33,
        S12_2,
        S13_2,
        S23_2,
        0.0,
        0.0,
        0.0,
    )


# ---------------------------------------------------------------------------
# B-column helpers
# ---------------------------------------------------------------------------
# For DOF k = node_a*6 + sub (sub 0-2 → pos, sub 3-5 → grad):
#   g_a = G_a (position) or H_a (gradient) — the reference-space gradient vector
#   B_diag[k] = [∂E11/∂q_k, ∂E22/∂q_k, ∂E33/∂q_k]
#   B_shear[k] = [∂(2E12)/∂q_k, ∂(2E13)/∂q_k, ∂(2E23)/∂q_k]
#
# Derivation (index notation, α = DOF direction):
#   ∂F[i,j]/∂q_k = δ_{i,α} · g_a[j]
#   ∂E[p,q]/∂q_k = 0.5·(g_a[p]·F[α,q] + F[α,p]·g_a[q])
#
# Diagonal Voigt:
#   ∂E[j,j]/∂q_k = g_a[j]·F[α,j]
#
# Off-diagonal Voigt (factor 2 already absorbed):
#   ∂(2E[p,q])/∂q_k = g_a[p]·F[α,q] + F[α,p]·g_a[q]   (p≠q)


@wp.func
def _b_diag(g: wp.vec3, F: wp.mat33, alpha: int) -> wp.vec3:
    """Diagonal B sub-vector [∂E11, ∂E22, ∂E33] for one DOF direction alpha."""
    Fa = wp.vec3(F[alpha, 0], F[alpha, 1], F[alpha, 2])
    return wp.vec3(g[0] * Fa[0], g[1] * Fa[1], g[2] * Fa[2])


@wp.func
def _b_shear(g: wp.vec3, F: wp.mat33, alpha: int) -> wp.vec3:
    """Shear B sub-vector [∂(2E12), ∂(2E13), ∂(2E23)] for direction alpha."""
    Fa = wp.vec3(F[alpha, 0], F[alpha, 1], F[alpha, 2])
    return wp.vec3(
        g[0] * Fa[1] + Fa[0] * g[1],  # 2·∂E12
        g[0] * Fa[2] + Fa[0] * g[2],  # 2·∂E13
        g[1] * Fa[2] + Fa[1] * g[2],  # 2·∂E23
    )


@wp.func
def _g_pos(dNa_xi: float, dNa_eta: float, J0inv: wp.mat33) -> wp.vec3:
    """Reference-space gradient G_a for position DOFs of node a.

    G_a = [∂Na/∂ξ, ∂Na/∂η, 0] · J0^{-1}
    """
    v = wp.vec3(dNa_xi, dNa_eta, 0.0)
    return wp.vec3(
        J0inv[0, 0] * v[0] + J0inv[1, 0] * v[1],
        J0inv[0, 1] * v[0] + J0inv[1, 1] * v[1],
        J0inv[0, 2] * v[0] + J0inv[1, 2] * v[1],
    )


@wp.func
def _g_grad(dNa_xi: float, dNa_eta: float, Na: float, h: float, zeta: float, J0inv: wp.mat33) -> wp.vec3:
    """Reference-space gradient H_a for gradient (D) DOFs of node a.

    H_a = [ζ·h/2·∂Na/∂ξ, ζ·h/2·∂Na/∂η, Na·h/2] · J0^{-1}
    """
    hz2 = 0.5 * h
    v = wp.vec3(zeta * hz2 * dNa_xi, zeta * hz2 * dNa_eta, Na * hz2)
    return wp.vec3(
        J0inv[0, 0] * v[0] + J0inv[1, 0] * v[1] + J0inv[2, 0] * v[2],
        J0inv[0, 1] * v[0] + J0inv[1, 1] * v[1] + J0inv[2, 1] * v[2],
        J0inv[0, 2] * v[0] + J0inv[1, 2] * v[1] + J0inv[2, 2] * v[2],
    )


# ---------------------------------------------------------------------------
# ANS tying-point shear strain evaluation
# ---------------------------------------------------------------------------


@wp.func
def _e13_at(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    D0: wp.vec3,
    D1: wp.vec3,
    D2: wp.vec3,
    D3: wp.vec3,
    x0_ref: wp.vec3,
    x1_ref: wp.vec3,
    x2_ref: wp.vec3,
    x3_ref: wp.vec3,
    D0_ref: wp.vec3,
    D1_ref: wp.vec3,
    D2_ref: wp.vec3,
    D3_ref: wp.vec3,
    xi: float,
    eta: float,
    h: float,
) -> float:
    """Transverse shear 2·E13 at a tying point (ζ=0 always for ANS)."""
    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)
    J0 = _jacobian(x0_ref, x1_ref, x2_ref, x3_ref, D0_ref, D1_ref, D2_ref, D3_ref, dNxi, dNeta, N, 0.0, h)
    J = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi, dNeta, N, 0.0, h)
    J0inv = wp.inverse(J0)
    F = _matmul33(J, J0inv)
    sh = _gl_shear(F)
    return sh[1]  # 2·E13


@wp.func
def _e23_at(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    D0: wp.vec3,
    D1: wp.vec3,
    D2: wp.vec3,
    D3: wp.vec3,
    x0_ref: wp.vec3,
    x1_ref: wp.vec3,
    x2_ref: wp.vec3,
    x3_ref: wp.vec3,
    D0_ref: wp.vec3,
    D1_ref: wp.vec3,
    D2_ref: wp.vec3,
    D3_ref: wp.vec3,
    xi: float,
    eta: float,
    h: float,
) -> float:
    """Transverse shear 2·E23 at a tying point (ζ=0)."""
    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)
    J0 = _jacobian(x0_ref, x1_ref, x2_ref, x3_ref, D0_ref, D1_ref, D2_ref, D3_ref, dNxi, dNeta, N, 0.0, h)
    J = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi, dNeta, N, 0.0, h)
    J0inv = wp.inverse(J0)
    F = _matmul33(J, J0inv)
    sh = _gl_shear(F)
    return sh[2]  # 2·E23


@wp.func
def _b13_at(
    x0_ref: wp.vec3,
    x1_ref: wp.vec3,
    x2_ref: wp.vec3,
    x3_ref: wp.vec3,
    D0_ref: wp.vec3,
    D1_ref: wp.vec3,
    D2_ref: wp.vec3,
    D3_ref: wp.vec3,
    F: wp.mat33,
    J0inv: wp.mat33,
    xi: float,
    eta: float,
    h: float,
    node: int,
    is_grad: int,
    alpha: int,
) -> float:
    """B[2·E13, DOF] at a tying point — single DOF contribution."""
    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)
    Na = N[node]
    dNa_xi = dNxi[node]
    dNa_eta = dNeta[node]
    if is_grad == 0:
        g = _g_pos(dNa_xi, dNa_eta, J0inv)
    else:
        g = _g_grad(dNa_xi, dNa_eta, Na, h, 0.0, J0inv)
    b_sh = _b_shear(g, F, alpha)
    return b_sh[1]  # index 1 = 2·E13 contribution


@wp.func
def _b23_at(
    x0_ref: wp.vec3,
    x1_ref: wp.vec3,
    x2_ref: wp.vec3,
    x3_ref: wp.vec3,
    D0_ref: wp.vec3,
    D1_ref: wp.vec3,
    D2_ref: wp.vec3,
    D3_ref: wp.vec3,
    F: wp.mat33,
    J0inv: wp.mat33,
    xi: float,
    eta: float,
    h: float,
    node: int,
    is_grad: int,
    alpha: int,
) -> float:
    """B[2·E23, DOF] at a tying point — single DOF contribution."""
    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)
    Na = N[node]
    dNa_xi = dNxi[node]
    dNa_eta = dNeta[node]
    if is_grad == 0:
        g = _g_pos(dNa_xi, dNa_eta, J0inv)
    else:
        g = _g_grad(dNa_xi, dNa_eta, Na, h, 0.0, J0inv)
    b_sh = _b_shear(g, F, alpha)
    return b_sh[2]  # index 2 = 2·E23 contribution


# ---------------------------------------------------------------------------
# ANS ε_zz tying-point helpers
# ---------------------------------------------------------------------------


@wp.func
def _e33_at(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    D0: wp.vec3,
    D1: wp.vec3,
    D2: wp.vec3,
    D3: wp.vec3,
    x0_ref: wp.vec3,
    x1_ref: wp.vec3,
    x2_ref: wp.vec3,
    x3_ref: wp.vec3,
    D0_ref: wp.vec3,
    D1_ref: wp.vec3,
    D2_ref: wp.vec3,
    D3_ref: wp.vec3,
    xi: float,
    eta: float,
    h: float,
) -> float:
    """Thickness normal strain E33 at a corner tying point (zeta=0 for ANS)."""
    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)
    J0 = _jacobian(x0_ref, x1_ref, x2_ref, x3_ref, D0_ref, D1_ref, D2_ref, D3_ref, dNxi, dNeta, N, 0.0, h)
    J = _jacobian(x0, x1, x2, x3, D0, D1, D2, D3, dNxi, dNeta, N, 0.0, h)
    J0inv = wp.inverse(J0)
    F = _matmul33(J, J0inv)
    return _gl_strain(F)[2]


@wp.func
def _b33_at(
    x0_ref: wp.vec3,
    x1_ref: wp.vec3,
    x2_ref: wp.vec3,
    x3_ref: wp.vec3,
    D0_ref: wp.vec3,
    D1_ref: wp.vec3,
    D2_ref: wp.vec3,
    D3_ref: wp.vec3,
    F: wp.mat33,
    J0inv: wp.mat33,
    xi: float,
    eta: float,
    h: float,
    node: int,
    is_grad: int,
    alpha: int,
) -> float:
    """B[E33, DOF] at a corner tying point (zeta=0) for ANS linearization."""
    N = _shape(xi, eta)
    dNxi = _dshape_dxi(eta)
    dNeta = _dshape_deta(xi)
    Na = N[node]
    dNa_xi = dNxi[node]
    dNa_eta = dNeta[node]
    if is_grad == 0:
        g = _g_pos(dNa_xi, dNa_eta, J0inv)
    else:
        g = _g_grad(dNa_xi, dNa_eta, Na, h, 0.0, J0inv)
    return _b_diag(g, F, alpha)[2]


# ---------------------------------------------------------------------------
# β matrix and strain/B-row transforms for fiber-angle locking remedies
# ---------------------------------------------------------------------------


@wp.func
def _compute_beta(J0: wp.mat33, cos_theta: float, sin_theta: float) -> wp.mat33:
    """Rotation matrix β[α,i] = AA_α[i] mapping global Cartesian → material frame.

    F = J·J₀⁻¹ is a Cartesian deformation gradient, so E = ½(FᵀF−I) has global
    Cartesian components.  β is therefore a pure rotation (entries ≤ 1) — no J₀⁻¹
    involved.  Using J₀⁻¹ columns instead would inflate β by ~1/h ≈ 400 m⁻¹ for
    thin shells and blow up the float32 EAS K_alpha to ~10¹⁴.
    """
    G1 = wp.vec3(J0[0, 0], J0[1, 0], J0[2, 0])
    G2 = wp.vec3(J0[0, 1], J0[1, 1], J0[2, 1])
    A3 = wp.normalize(wp.cross(G1, G2))
    A1 = wp.normalize(G1)
    A2 = wp.cross(A3, A1)
    AA1 = cos_theta * A1 + sin_theta * A2
    AA2 = -sin_theta * A1 + cos_theta * A2
    return wp.mat33(
        AA1[0],
        AA1[1],
        AA1[2],
        AA2[0],
        AA2[1],
        AA2[2],
        A3[0],
        A3[1],
        A3[2],
    )


@wp.func
def _beta_transform_diag(e_d: wp.vec3, e_s: wp.vec3, b: wp.mat33) -> wp.vec3:
    """Apply 6x6 Voigt β transform to a natural 6-vector; return material diagonal [E11m,E22m,E33m].
    Input e_d=[E11,E22,E33] natural, e_s=[2E12,2E13,2E23] natural."""
    b00 = b[0, 0]
    b01 = b[0, 1]
    b02 = b[0, 2]
    b10 = b[1, 0]
    b11 = b[1, 1]
    b12 = b[1, 2]
    b20 = b[2, 0]
    b21 = b[2, 1]
    b22 = b[2, 2]
    e11 = e_d[0]
    e22 = e_d[1]
    e33 = e_d[2]
    g12 = e_s[0]
    g13 = e_s[1]
    g23 = e_s[2]
    E11m = b00 * b00 * e11 + b01 * b01 * e22 + b02 * b02 * e33 + b00 * b01 * g12 + b00 * b02 * g13 + b01 * b02 * g23
    E22m = b10 * b10 * e11 + b11 * b11 * e22 + b12 * b12 * e33 + b10 * b11 * g12 + b10 * b12 * g13 + b11 * b12 * g23
    E33m = b20 * b20 * e11 + b21 * b21 * e22 + b22 * b22 * e33 + b20 * b21 * g12 + b20 * b22 * g13 + b21 * b22 * g23
    return wp.vec3(E11m, E22m, E33m)


@wp.func
def _beta_transform_shear(e_d: wp.vec3, e_s: wp.vec3, b: wp.mat33) -> wp.vec3:
    """Apply 6x6 Voigt β transform; return material shear [2E12m,2E13m,2E23m]."""
    b00 = b[0, 0]
    b01 = b[0, 1]
    b02 = b[0, 2]
    b10 = b[1, 0]
    b11 = b[1, 1]
    b12 = b[1, 2]
    b20 = b[2, 0]
    b21 = b[2, 1]
    b22 = b[2, 2]
    e11 = e_d[0]
    e22 = e_d[1]
    e33 = e_d[2]
    g12 = e_s[0]
    g13 = e_s[1]
    g23 = e_s[2]
    G12m = (
        2.0 * (b00 * b10 * e11 + b01 * b11 * e22 + b02 * b12 * e33)
        + (b00 * b11 + b10 * b01) * g12
        + (b00 * b12 + b10 * b02) * g13
        + (b01 * b12 + b11 * b02) * g23
    )
    G13m = (
        2.0 * (b00 * b20 * e11 + b01 * b21 * e22 + b02 * b22 * e33)
        + (b00 * b21 + b20 * b01) * g12
        + (b00 * b22 + b20 * b02) * g13
        + (b01 * b22 + b21 * b02) * g23
    )
    G23m = (
        2.0 * (b10 * b20 * e11 + b11 * b21 * e22 + b12 * b22 * e33)
        + (b10 * b21 + b20 * b11) * g12
        + (b10 * b22 + b20 * b12) * g13
        + (b11 * b22 + b21 * b12) * g23
    )
    return wp.vec3(G12m, G13m, G23m)


# ---------------------------------------------------------------------------
# One Gauss-point contribution to elem_f and elem_K
# ---------------------------------------------------------------------------
# Each GP accumulates 24 force scalars and 24×24 stiffness scalars.
# The loops are unrolled by iterating (node a, is_grad a, dir a) for k, and
# (node b, is_grad b, dir b) for j.  Re-computing g/B per (k,j) pair avoids
# needing 24-element local arrays.

# compute_element_forces_stiffness lives in kernels_stiffness.py (separate
# compile unit so the lumped-mass kernel here doesn't block its compilation).

# ---------------------------------------------------------------------------
# Lumped mass kernel  (replaces the old consistent-mass 24×24 kernel)
# ---------------------------------------------------------------------------


@wp.kernel
def compute_lumped_mass(
    node_x0: wp.array[wp.vec3],
    node_D0: wp.array[wp.vec3],
    elem_nodes: wp.array2d[wp.int32],
    elem_h: wp.array[float],
    elem_mat: wp.array2d[float],
    lumped_mass: wp.array[float],  # (n_nodes * 6,) accumulated
):
    """Diagonal (lumped) mass per DOF via element-volume distribution.

    Integrates element mass over the reference configuration and distributes:
      - Position DOFs  (sub 0-2): m_elem / (4 * 3) per DOF
      - Gradient DOFs  (sub 3-5): m_pos * (h/2)^2 / 3  (thickness inertia)

    This replaces the full 24x24 consistent-mass kernel. The consistent mass
    would be row-summed into a lumped diagonal anyway, and the 24x24 loop is
    the bottleneck for nvcc compile time.
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
    rho = elem_mat[e, 9]

    # Integrate element mass: m = ∫ ρ dV
    m_elem = float(0.0)
    for gi in range(2):
        xi = _gp2(gi)
        w_xi = _w2(gi)
        for gj in range(2):
            eta = _gp2(gj)
            w_eta = _w2(gj)
            for gk in range(5):
                zeta = _gp5z(gk)
                w_z = _gp5w(gk)
                w = w_xi * w_eta * w_z
                N = _shape(xi, eta)
                dNxi = _dshape_dxi(eta)
                dNeta = _dshape_deta(xi)
                J0 = _jacobian(X0, X1, X2, X3, R0, R1, R2, R3, dNxi, dNeta, N, zeta, h)
                m_elem = m_elem + rho * w * wp.abs(wp.determinant(J0))

    # Distribute uniformly to all 4 nodes × 3 position directions
    m_per_pos = m_elem / float(4 * 3)
    # Director DOFs get the same lumped mass as position DOFs.
    # Physical rotational inertia m_pos*(h/2)^2/3 ≈ 5e-6*m_pos gives
    # K_eff[dir] ≈ 38 N/m vs K_eff[contact-pos] ≈ 2.3e6 N/m at impact →
    # κ ≈ 60 000, requiring ~2000 PCG iters to converge (we only have 300).
    # Equal mass collapses κ to ≈ 11; 300 iters is then ample.
    m_per_grad = m_per_pos

    for ni in range(4):
        na = elem_nodes[e, ni]
        for alpha in range(3):
            wp.atomic_add(lumped_mass, na * 6 + alpha, m_per_pos)
            wp.atomic_add(lumped_mass, na * 6 + 3 + alpha, m_per_grad)
