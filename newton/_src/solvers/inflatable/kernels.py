# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp kernels for the inflatable soft-body solver.

This module is the GPU half of :class:`SolverInflatable`. Every kernel falls
into one of the following physical roles:

* **Time integration** — apply the solved ``Δv`` to ``(x, v)`` and the
  Dirichlet pin RHS override.
* **System-matrix assembly** — write per-element 3×3 blocks into the
  triplet arrays consumed by Warp's ``bsr_set_from_triplets``. The
  resulting ``A`` is the Backward-Euler Jacobian.
* **Force evaluators** — compute per-particle ``vec3`` forces ``[N]``
  used as the BE right-hand side ``h·f``.
* **Inflation** — rescale per-tet rest pose ``Dm`` and per-spring rest
  length ``L₀`` to drive the FEM toward a target inflated volume.

Mathematical conventions
========================

Backward-Euler step (one Newton iteration). With ``Δv = v_{n+1} − v_n``,

    A · Δv = h · f_n
    A = M + h² · H_pot + h · C            (PSD Hessian + PD damping)

The position update is ``x_{n+1} = x_n + h · (v_n + Δv)``
(``update_state`` kernel below).

Per-element BSR triplet layout
==============================

System-matrix kernels write disjoint regions of the triplet arrays so
``bsr_set_from_triplets`` can sum duplicates into ``A``.  Assembly order
(matches ``SolverImplicitSoft._build_constant_matrix``)::

    [0 .. P)                       — per-particle diagonal (mass + spring sum)
    [P .. P+2·S)                   — per-spring (i,j) and (j,i) off-diagonals
    [P+2·S .. P+2·S+3·R)          — per-tri 3 lumped diagonal blocks
    [P+2·S+3·R .. P+2·S+3·R+16·T) — per-tet 4×4 nodal Hessian blocks (rest-state)
    [... +4·E)                     — per-edge 4 lumped diagonal blocks (bending)

where P = particle_count, S = spring_count, R = tri_count, T = tet_count, E = edge_count.

Sign convention for forces
==========================

Force kernels return forces in physical convention: ``f`` points in the
direction the particle is pushed. Internal-force kernels for elastic
elements compute ``∂ψ/∂x_a`` first then subtract from the force buffer
(so ``f -= ∂ψ/∂x_a`` ⇒ stored ``f`` is ``−∂ψ/∂x_a``, the restoring force).

Stable Neo-Hookean tetrahedra
=============================

Per-element strain energy density (Smith, De Goes & Kim 2018):

    ψ(F) = ½·μ·(I_C − 3) − ½·μ·log(I_C + 1) + ½·λ·(J − α)²
    α    = 1 + μ/λ − μ/(4·λ)            (rest correction)
    I_C  = ‖F‖²_F                       (squared Frobenius norm)
    J    = det(F)
    F    = Ds · Dm                      (deformation gradient)
    Ds   = [x₁−x₀, x₂−x₀, x₃−x₀]        (current shape, columns)
    Dm   = (rest shape)⁻¹               (cached on the model as ``tet_poses``)

The force kernel evaluates ``f_a = -∂ψ/∂x_a · V_e`` per node from the first
PK1 stress ``P = ∂ψ/∂F`` via ``f_a = P · Dm^T``. The matrix kernel computes
the 4×4 = 16 blocks of ``+h²·∂²ψ/∂x_a∂x_b`` using the chain rule
``H[a,b] = (dF/dx_a)^T · (∂²ψ/∂F²) · (dF/dx_b) + λ · (dJ/dx_a) ⊗ (dJ/dx_b)``.

Inflation by rest-configuration scaling
=======================================

Pressure ``p`` (volume-ratio target ≥ 1.0) drives the rest configuration:

    Dm  ← Dm  / cbrt(p)        (per tet ⇒ rest shape grows by cbrt(p))
    L₀  ← L₀ · cbrt(p)         (per spring)

Linear scale cbrt(p) ⇒ volumetric scale p. The FEM force then drives the
body toward F = I in the new rest frame (i.e. expanded by cbrt(p) per side,
volume × p). Per-chamber pressures use the same scaling restricted by a
chamber-id mask.

Dirichlet pin
=============

When the kinematic glue is configured, two kernels enforce
``Δv[p] = target_dv[p]`` exactly at masked particles (Sifakis SIGGRAPH
2012 §3 / Baraff–Witkin SIGGRAPH '98 §5):

    apply_dirichlet_pin_kernel        — per-particle: capture reaction
        particle_f[p] := target_dv[p]                     (mask[p] == 1)
        reaction[p]   := particle_f_old[p] / dt − mass · g
        (inertial term −mass·target_dv/dt is intentionally omitted;
         see kernel docstring for the added-mass stability argument)

    filter_dirichlet_pin_in_bsr_kernel — per-BSR-row: filter A, Schur RHS
        A[p, p]   := I,  A[p, c≠p] := 0,  A[r≠p, p] := 0
        particle_f[r] -= A[r, p] · target_dv[p]           (Schur step)

The reaction is forwarded by the glue to the rigid solver via
``state.body_f``; including the inertial reaction
``mass · target_dv / dt`` keeps momentum conserved at the interface.

File layout
===========

1. Constants and helper types
2. Time integration  — ``update_state``, ``apply_dirichlet_pin_kernel``
3. Mass diagonal    — ``build_system_matrix_diagonal_mass_kernel``
4. Linear springs   — force + matrix tangent
5. Tetrahedral FEM  — Stable Neo-Hookean force + tangent
6. Triangle FEM     — membrane + tangent + edge bending (force + implicit tangent)
7. Particle-particle (hash grid)
8. Particle-rigid contact — ground plane, force-based, constraint-based
9. Gravity
10. Inflation       — rest-pose scaling + volume integration
"""

import warp as wp

from newton import ParticleFlags

# =====================================================================
# 1. Constants and helper types
# =====================================================================

# Active-particle bitmask (matches ``ParticleFlags.ACTIVE``).
PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


# =====================================================================
# 2. Time integration
# =====================================================================
#
# Final integration step ``x_{n+1} = x_n + h · v_{n+1}`` and the Dirichlet
# pin's RHS override. The pin lives here (not under "system-matrix") because
# it edits the RHS, not A.
# =====================================================================


@wp.kernel
def update_state(
    dv: wp.array[wp.vec3],
    dt: wp.float32,
    positions_in: wp.array[wp.vec3],
    velocities_in: wp.array[wp.vec3],
    positions_out: wp.array[wp.vec3],
    velocities_out: wp.array[wp.vec3],
):
    """Integrate the BE solution: ``v_{n+1} = v_n + Δv``, ``x_{n+1} = x_n + h·v_{n+1}``."""
    tid = wp.tid()
    vel = velocities_in[tid] + dv[tid]
    positions_out[tid] = positions_in[tid] + vel * dt
    velocities_out[tid] = vel


@wp.kernel
def apply_dirichlet_pin_kernel(
    mask: wp.array[wp.int32],
    target_dv: wp.array[wp.vec3],
    gravity: wp.vec3,
    mass: wp.float32,
    target_scale: wp.float32,
    dt: float,
    # in/out
    particle_f: wp.array[wp.vec3],
    # out
    reaction: wp.array[wp.vec3],
):
    """Dirichlet RHS override and reaction capture for kinematic-pinned particles.

    For mask[p] == 1 (pinned):

        reaction[p]   := particle_f[p] / dt − mass · gravity − mass · target_dv[p] / dt
        particle_f[p] := target_scale · target_dv[p]

    ``target_scale`` is ``1.0`` when paired with
    :func:`filter_dirichlet_pin_in_bsr_kernel` (matrix has been row/col-
    eliminated to ``A[p, p] = I``, so ``Δv[p] = target_dv[p]`` exactly).
    Set it to ``mass`` when the filter is disabled (legacy RHS-only path:
    A's diagonal is still ``m + h²·k_pp``, so ``Δv[p] ≈ target_dv[p]``
    with the soft-pin compliance the FEM tangent introduces).

    The reaction is the elastic / contact / damping force the FEM wanted to
    apply at this vertex, minus gravity (already an external force on the soft
    particle).  By Newton's third law the rigid body feels ``+reaction``; the
    glue forwards it onto ``state.body_f`` or ``mjw_data.xfrc_applied``.

    The inertial term ``m · target_dv / dt`` is intentionally omitted.
    Including it is correct when the rigid body uses a position-based (XPBD)
    integrator where ``body_f`` enters as a positional displacement, but it
    creates an added-mass instability when forwarded as a continuous force to
    an explicit-Euler integrator (MuJoCo, Newton's impulse integrator): the
    stability condition ``N_pin · m_particle / M_rigid < 1`` is easily violated
    (Causin et al. 2005, Förster et al. 2007).  Elastic-only reaction is
    bounded by the FEM stiffness and unconditionally stable.

    The RHS override is paired with :func:`filter_dirichlet_pin_in_bsr_kernel`,
    which sets ``A[p, p] = I`` and zeros the rest of row/column ``p``. The
    combined effect is ``Δv[p] = target_dv[p]`` exactly — a hard pin, not a
    soft one — and rigid translations of the pin no longer leak into the
    elastic neighbours through stiffness off-diagonals.

    For mask[p] == 0 (free): reaction[p] is zeroed and particle_f[p] is left
    untouched.

    Args:
        mask: Per-particle ``int32`` flag, ``1`` for pinned and ``0`` for free.
        target_dv: Per-particle target velocity change ``[m/s]``.
        gravity: World gravity vector ``[m/s²]``.
        mass: Uniform particle mass ``[kg]``.
        target_scale: Multiplier on ``target_dv`` when writing the RHS;
            ``1.0`` with the BSR filter (hard pin), ``mass`` without
            (legacy soft pin).
        dt: Substep size ``[s]``.
        particle_f: Implicit-solver RHS, ``dt · sum_of_forces`` ``[N·s]``.
            Overwritten in place at pinned indices.
        reaction: Output buffer for the per-particle non-gravity reaction ``[N]``.
    """
    p = wp.tid()
    if mask[p] == 0:
        reaction[p] = wp.vec3(0.0, 0.0, 0.0)
        return

    f_total = particle_f[p]
    # Elastic-only: omit the inertial term (mass·target_dv/dt) to avoid
    # added-mass instability when forwarding to an explicit-Euler rigid solver.
    reaction[p] = (f_total / dt) - mass * gravity

    particle_f[p] = target_scale * target_dv[p]


@wp.kernel
def filter_dirichlet_pin_in_bsr_kernel(
    mask: wp.array[wp.int32],
    target_dv: wp.array[wp.vec3],
    bsr_offsets: wp.array[int],
    bsr_columns: wp.array[int],
    bsr_values: wp.array[wp.mat33f],
    # in/out
    particle_f: wp.array[wp.vec3],
):
    """Row/column elimination of pinned DOFs in the assembled BSR matrix.

    Implements the standard treatment of essential boundary conditions
    (Sifakis SIGGRAPH 2012 §3 / Baraff–Witkin SIGGRAPH '98 §5): for each
    pinned particle ``p``, replace row and column ``p`` of ``A`` with the
    constraint ``Δv[p] = target_dv[p]`` and Schur-condense the prescribed
    motion into the RHS at the unpinned neighbours.

    Per-row behaviour (one thread per BSR row):

        if mask[r] == 1 (pinned):
            for each block (r, c) in row r:
                values[k] = I        if c == r
                values[k] = 0        if c != r
            (RHS at r is overwritten to target_dv[r] by
            ``apply_dirichlet_pin_kernel``.)

        if mask[r] == 0 (free):
            for each block (r, c) in row r with mask[c] == 1:
                particle_f[r] -= values[k] · target_dv[c]    (Schur step)
                values[k] = 0

    The Schur step preserves the constrained problem exactly: removing
    column ``p`` and accumulating ``A[r, p] · target_dv[p]`` into the RHS
    is mathematically equivalent to keeping ``A[r, p]`` and constraining
    ``Δv[p] = target_dv[p]``. Without it the unpinned neighbours would feel
    a phantom rigid translation through the zeroed stiffness coupling.

    Run AFTER ``bsr_set_from_triplets`` and AFTER the per-particle pin RHS
    override, BEFORE the linear solve. Modifies ``A`` in place —
    ``apply_dirichlet_filter`` restores ``A`` from a clean snapshot before
    each call so the Schur step sees the original off-diagonals every substep.

    Args:
        mask: Per-particle ``int32`` flag, ``1`` for pinned and ``0`` for free.
        target_dv: Per-particle target velocity change ``[m/s]``.
        bsr_offsets: BSR row offsets, length ``particle_count + 1``.
        bsr_columns: BSR column indices for each block.
        bsr_values: BSR block values (3×3 ``mat33f``); modified in place.
        particle_f: Implicit-solver RHS ``[N·s]``; Schur correction added
            in place at unpinned rows that have a pinned column.
    """
    r = wp.tid()
    start = bsr_offsets[r]
    end = bsr_offsets[r + 1]

    identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    zero = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    is_pinned = mask[r] == 1
    rhs_correction = wp.vec3(0.0, 0.0, 0.0)

    for k in range(start, end):
        c = bsr_columns[k]
        if is_pinned:
            # Pinned row: zero everything except diagonal, which becomes I.
            if c == r:
                bsr_values[k] = identity
            else:
                bsr_values[k] = zero
        elif mask[c] == 1:
            # Unpinned row, pinned column: Schur-condense and zero A[r, c].
            rhs_correction = rhs_correction + bsr_values[k] * target_dv[c]
            bsr_values[k] = zero

    if not is_pinned:
        # Only this thread writes to particle_f[r]; no atomic needed.
        particle_f[r] = particle_f[r] - rhs_correction


# =====================================================================
# 3. Mass diagonal (always present)
# =====================================================================
#
# Spring-less variant of the diagonal block writer: just M·I per particle.
# When springs are present, the spring kernel below subsumes this and
# writes M·I plus the spring contributions in one pass.
# =====================================================================


@wp.kernel
def build_system_matrix_diagonal_mass_kernel(
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    particle_mass: wp.array[wp.float32],
):
    """Write ``A_ii = m_i·I`` for each particle (no springs path)."""
    tid = wp.tid()
    rows[tid] = tid
    cols[tid] = tid
    m = particle_mass[tid]
    values[tid] = wp.mat33f(m, 0.0, 0.0, 0.0, m, 0.0, 0.0, 0.0, m)


# =====================================================================
# 4. Linear springs (force + matrix tangent)
# =====================================================================
#
# Spring force on particle i from spring (i,j):
#
#     f_i = − k · (l − L₀) · dir − k_d · ((v_i − v_j) · dir) · dir
#     dir = (x_i − x_j) / l,    l = ‖x_i − x_j‖
#
# Backward-Euler tangent contribution (lumped isotropic):
#
#     A_ii  +=  +(h² · k + h · k_d) · I
#     A_ij  +=  −(h² · k + h · k_d) · I    (one block per (i,j), one per (j,i))
#
# Together this is a graph-Laplacian-style contribution: the constant mode
# (rigid translation) is in the null space of the spring contribution, so
# pure translations don't see spurious spring forces in the implicit step.
# =====================================================================


@wp.kernel
def eval_springs(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_lengths: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    f: wp.array[wp.vec3],
):
    """Linear spring restoring + damping force on each endpoint."""
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    if i == -1 or j == -1:
        return

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]
    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj
    l = wp.length(xij)

    # Protect against division by zero on coincident endpoints.
    if l < 1.0e-6:
        return

    dir = xij / l
    c = l - rest
    dcdt = wp.dot(dir, vij)

    # f_spring = (k·c + k_d·ċ) · dir, applied with opposite signs at i and j.
    fs = dir * (ke * c + kd * dcdt)
    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def eval_springs_linear_and_torque(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    spring_indices: wp.array[int],
    spring_rest_lengths: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    spring_rest_direction: wp.array[wp.vec3],
    torque_stiffness: wp.float32,
    torque_damping: wp.float32,
    f: wp.array[wp.vec3],
):
    """Linear spring + torsional restoring force toward a per-spring rest direction.

    Linear part: identical to :func:`eval_springs`.

    Torque: a unit rest direction ``d₀`` defines a target axis. Let ``θ`` be
    the angle between ``d₀`` and the current direction ``d = (x_i−x_j)/l``,
    and ``ω`` the rate of change projected onto the rotation axis
    ``d₀ × d``. Torque magnitude is

        τ = -(k_τ · θ + k_ωτ · ω) · l

    converted to a transverse force ``F_t = (axis · τ) × d / l`` applied
    with opposite signs at the two endpoints. Springs with ``‖d₀‖ < 0.5``
    are treated as torque-free.
    """
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    if i == -1 or j == -1:
        return

    # --- Linear part (same as eval_springs) -----------------------
    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]
    xi, xj = x[i], x[j]
    vi, vj = v[i], v[j]
    xij = xi - xj
    vij = vi - vj
    l = wp.length(xij)
    if l < 1.0e-6:
        return
    d = xij / l
    c = l - rest
    dcdt = wp.dot(d, vij)
    fs = d * (ke * c + kd * dcdt)
    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)

    # --- Torque part (skip for zero rest direction) --------------
    # spring_rest_direction is baked as (V[hi] - V[lo])/|..| (lo→hi).
    # d above is (xi - xj) = (x[lo] - x[hi])/l (hi→lo). Negate to align.
    d0 = -spring_rest_direction[tid]
    if wp.length(d0) < 0.5:
        return
    dot_d_d0 = wp.clamp(wp.dot(d, d0), -1.0, 1.0)
    theta = wp.acos(dot_d_d0)
    rot_axis = wp.cross(d0, d)
    rot_len = wp.length(rot_axis)
    if rot_len < 1.0e-6:
        return
    rot_axis = rot_axis / rot_len
    d_dot = (vij - d * wp.dot(d, vij)) / l
    omega_along = wp.dot(wp.cross(d, d_dot), rot_axis)
    tau_mag = -(torque_stiffness * theta + torque_damping * omega_along) * l
    force_torque = wp.cross(rot_axis * tau_mag, d) / l
    wp.atomic_sub(f, i, force_torque)
    wp.atomic_add(f, j, force_torque)


@wp.kernel
def build_system_matrix_diagonal_kernel(
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    spring_indices: wp.array[int],
    spring_stiffness: wp.array[wp.float32],
    spring_damping: wp.array[wp.float32],
    dt: wp.float32,
    particle_mass: wp.array[wp.float32],
    n_springs: wp.int32,
):
    """Per-particle diagonal of A: ``A_ii = (m_i + Σ_springs (h·d + h²·k))·I``.

    The off-diagonal counterpart ``A_ij = −(h·d + h²·k)·I`` lives in
    :func:`build_system_matrix_sparse_kernel`.
    """
    i = wp.tid()
    dt2 = dt * dt
    diag = particle_mass[i]
    for s in range(n_springs):
        ia = spring_indices[s * 2 + 0]
        ja = spring_indices[s * 2 + 1]
        if ia == i or ja == i:
            k = spring_stiffness[s]
            d = spring_damping[s]
            diag = diag + dt * d + dt2 * k
    rows[i] = i
    cols[i] = i
    values[i] = wp.mat33f(diag, 0.0, 0.0, 0.0, diag, 0.0, 0.0, 0.0, diag)


@wp.kernel
def build_system_matrix_sparse_kernel(
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    indices: wp.array[int],
    spring_stiffness: wp.array[wp.float32],
    spring_damping: wp.array[wp.float32],
    dt: wp.float32,
    block_offset: wp.int32,
):
    """Per-spring symmetric off-diagonal: ``A_ij = A_ji = −(h·d + h²·k)·I``.

    Pairs the ``+(h·d + h²·k)·I`` diagonal contribution above into a
    Laplacian-form block whose null space contains rigid translations
    (so free-fall is unimpeded by spring damping).
    """
    tid = wp.tid()
    i = indices[tid * 2 + 0]
    j = indices[tid * 2 + 1]
    k = spring_stiffness[tid]
    d = spring_damping[tid]
    neg_off = -(dt * d + dt * dt * k)
    block_ij = wp.mat33f(
        neg_off,
        0.0,
        0.0,
        0.0,
        neg_off,
        0.0,
        0.0,
        0.0,
        neg_off,
    )
    block_idx = block_offset + tid * 2
    rows[block_idx] = i
    cols[block_idx] = j
    values[block_idx] = block_ij
    rows[block_idx + 1] = j
    cols[block_idx + 1] = i
    values[block_idx + 1] = block_ij


# =====================================================================
# 5. Tetrahedral FEM — Stable Neo-Hookean (force + matrix tangent)
# =====================================================================
#
# Per-element strain energy density (Smith, De Goes & Kim 2018):
#
#     ψ(F) = ½·μ·(I_C − 3) − ½·μ·log(I_C + 1) + ½·λ·(J − α)²
#     α    = 1 + μ/λ − μ/(4·λ)              (rest correction → ψ(I) = 0)
#     I_C  = ‖F‖²_F = trace(F^T F)
#     J    = det(F)
#     F    = Ds · Dm                         (deformation gradient)
#     Ds   = [x₁ − x₀, x₂ − x₀, x₃ − x₀]    (current shape, columns)
#     Dm   = (rest shape)⁻¹                 (cached on the model)
#
# First Piola-Kirchhoff stress (regularised deviatoric form used here):
#
#     P = ∂ψ/∂F = μ · F · (1 − 1/(I_C + 1)) + λ · (J − α) · (∂J/∂F)
#
# Per-node force from PK1 stress times Dm^T:
#
#     f_a = −V_e · ∂ψ/∂x_a            (V_e = rest volume)
#
# implemented via ``H = P · Dm^T`` whose columns give f_1, f_2, f_3 and
# f_0 = −(f_1+f_2+f_3) by Newton's third law.
#
# Hessian block at node-pair (a,b) (12×12 element Hessian, 4×4 = 16 blocks):
#
#     H[a,b] = (dF/dx_a)^T · (∂²ψ/∂F²) · (dF/dx_b)        (deviatoric)
#            + λ · (dJ/dx_a) ⊗ (dJ/dx_b)                   (volumetric)
#
# The matrix kernel writes ``+h²·H[a,b]`` to ``A_{node_a, node_b}``. Per
# Smith 2018 a per-element PSD projection of ``H`` would prevent indefinite
# blocks under large deformation; not yet applied (planned).
# =====================================================================


@wp.func
def skew3(v: wp.vec3) -> wp.mat33:
    """3×3 skew-symmetric matrix: skew3(v) · u == v × u."""
    return wp.mat33(
        0.0,
        -v[2],
        v[1],
        v[2],
        0.0,
        -v[0],
        -v[1],
        v[0],
        0.0,
    )


@wp.func
def clamp_deformation_stretch(F: wp.mat33, min_stretch: float, max_stretch: float):
    """Principal-stretch clamp on the deformation gradient.

    Bounds the singular values (principal stretches) of ``F`` to
    ``[min_stretch, max_stretch]`` and rebuilds ``F`` from the clamped values:
    ``F ← U · diag(clamp(Σ)) · Vᵀ``. A positive ``min_stretch`` floors the
    smallest stretch above zero, so an element can never invert (``det F ≤ 0``)
    or collapse, and ``max_stretch`` caps run-away extension — the two states
    that make the Neo-Hookean stress and its tangent blow up. The clamped ``F``
    is then used for both the force and the Hessian so they stay consistent.

    NOTE: this borrows the SVD-of-F framework popularised by invertible FEM
    (Irving et al. 2004) but is NOT that method. Irving *admits* inversion (it
    lets the smallest singular value go negative via a deliberate sign choice
    and extrapolates the stress to push the element back out); this clamp
    instead floors the stretches positive and forbids inversion outright —
    simpler and more aggressive, trading physical fidelity for a hard
    ``det F > 0`` guarantee.

    Disabled (returns ``F`` unchanged) when either bound is negative. The
    solver enables it by default with loose bounds that normal deformation
    never reaches, so the clamp only engages on near-degenerate elements.
    """
    if min_stretch < 0.0 or max_stretch < 0.0:
        return F
    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, S, V)
    S = wp.max(wp.min(S, wp.vec3(max_stretch)), wp.vec3(min_stretch))
    return U * wp.diag(S) * wp.transpose(V)


@wp.kernel
def eval_tetrahedra(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    indices: wp.array2d[int],
    pose: wp.array[wp.mat33],
    activation: wp.array[float],
    materials: wp.array2d[float],
    min_stretch: float,
    max_stretch: float,
    f: wp.array[wp.vec3],
):
    """Stable Neo-Hookean tet force per node ``[N]``.

    Activation channel ``act`` adds to the volumetric strain ``(J − α + act)``
    so a contractile / expanding muscle can be driven by control without
    changing the rest pose. Rayleigh damping is applied via the deformation
    gradient time derivative ``dF/dt`` and ``dJ/dt``.

    When ``min_stretch``/``max_stretch`` are non-negative, the deformation
    gradient is passed through :func:`clamp_deformation_stretch` first, so an
    inverted or over-stretched tet can never drive the stress to blow up
    (a principal-stretch clamp; see that function for how it relates to and
    differs from invertible FEM, Irving et al. 2004). ``J`` is additionally
    clamped at ``J_MIN = 0.01`` as a backstop when the stretch clamp is disabled.
    """
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # α = 1 + μ/λ − μ/(4λ): rest correction so ψ(I) = 0 (Smith 2018).
    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # Fold V into the Lamé parameters so subsequent products are per-element
    # quantities, not per-unit-volume.
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Ds · Dm   and  dF/dt = (velocity-gradient-shape) · Dm.
    F = Ds * Dm
    # Invertible-FEM stretch clamp (no-op unless both bounds are non-negative);
    # keeps an inverted / over-stretched tet from blowing up the stress.
    F = clamp_deformation_stretch(F, min_stretch, max_stretch)
    dFdt = wp.matrix_from_cols(v10, v20, v30) * Dm

    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # I_C = ‖F‖²_F.
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)

    # Deviatoric PK1: P_dev = μ · F · (1 − 1/(I_C+1)) (regularised) plus
    # a Rayleigh-damping term proportional to dF/dt.
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * wp.transpose(Dm)

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    # Volumetric: f_volume = λ · (J − α + act) along ∂J/∂x_a (== ½·(x_b×x_c)
    # for opposite-face pair).
    J = wp.determinant(F)
    J_MIN = 0.01
    J = wp.max(J, J_MIN)

    s = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s
    dJdx2 = wp.cross(x30, x10) * s
    dJdx3 = wp.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (wp.dot(dJdx1, v10) + wp.dot(dJdx2, v20) + wp.dot(dJdx3, v30)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = -(f1 + f2 + f3)  # Newton's third law on the element

    # f_a (variable) = +V·∂ψ/∂x_a; subtracting from buffer applies the
    # restoring force −V·∂ψ/∂x_a.
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


@wp.kernel
def build_system_matrix_tet_kernel(
    x: wp.array[wp.vec3],
    indices: wp.array2d[int],
    pose: wp.array[wp.mat33],
    materials: wp.array2d[float],
    min_stretch: float,
    max_stretch: float,
    dt: wp.float32,
    block_offset: wp.int32,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    dirty: wp.array[wp.int32],
):
    """Per-tet Hessian: 4×4 = 16 nodal blocks of ``+h² · ∂²ψ/∂x_a∂x_b``.

    Deviatoric (exact closed-form, not the incorrect dFdx matrix approximation):

    .. code-block:: none

        K_dev[a,b][α,β] = k_mu·s·dot(Dm_row_a, Dm_row_b)·δ_{αβ}
                         + (2·k_mu/(I_C+1)²)·(F·Dm_row_a)[α]·(F·Dm_row_b)[β]

    where ``Dm_row_k = Dm[k-1, :]`` (k-th row of Dm) for nodes k=1,2,3, and
    ``Dm_row_0 = −(Dm_row_1 + Dm_row_2 + Dm_row_3)`` (Newton's law: Σ=0).

    Volumetric: ``H_vol[a,b] = λ · (dJ/dx_a) ⊗ (dJ/dx_b)`` (rank-1 PSD).

    The geometric stiffness ``λ(J−α)·d²J/∂xₐ∂x_b = geo·skew3(w_ab)`` is
    intentionally omitted.  Although its coefficient can be clamped to ≥0,
    the skew-symmetric 3×3 blocks it contributes to off-diagonal entries
    break the PSD property of the assembled 12×12 element Hessian and cause
    BiCGStab/CG divergence for stiff or nearly-incompressible materials.
    ``K_dev + K_vol`` alone is unconditionally PSD (proof: both quadratic
    forms are non-negative for all displacement fields).

    Degenerate / inverted rest poses (``inv_rest_volume ≤ 0``) write zero
    blocks instead of NaN-propagating.

    ``dirty[tid] == 0`` means the deformation gradient has not changed
    enough since the last frame; the kernel returns early, leaving the
    stale (but correct) triplet values in place.
    """
    tid = wp.tid()
    if dirty[tid] == 0:
        return
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    det_Dm = wp.determinant(Dm)
    inv_rest_volume = det_Dm * 6.0
    if inv_rest_volume <= 0.0:
        # Degenerate / inverted rest pose ⇒ write zero blocks.
        zero = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for a in range(4):
            row_a = i if a == 0 else (j if a == 1 else (k if a == 2 else l))
            for b in range(4):
                col_b = i if b == 0 else (j if b == 1 else (k if b == 2 else l))
                blk = block_offset + tid * 16 + a * 4 + b
                rows[blk] = row_a
                cols[blk] = col_b
                values[blk] = zero
        return
    rest_volume = 1.0 / inv_rest_volume

    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume

    F = Ds * Dm
    # Same principal-stretch clamp as the force kernel, so the tangent is
    # built from the same (clamped) F and stays consistent with the force.
    F = clamp_deformation_stretch(F, min_stretch, max_stretch)
    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)
    Ic1 = Ic + 1.0
    Ic1 = wp.max(Ic1, 1e-6)
    s = 1.0 - 1.0 / Ic1
    coef = k_mu * 2.0 / (Ic1 * Ic1)

    # ∂J/∂x_a: per-node gradient of det(F).
    s_vol = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s_vol
    dJdx2 = wp.cross(x30, x10) * s_vol
    dJdx3 = wp.cross(x10, x20) * s_vol
    dJdx0 = -(dJdx1 + dJdx2 + dJdx3)

    # Per-node Dm row vectors for the correct deviatoric Hessian.
    #
    # From the chain rule, ∂F[i,j]/∂(x_a)_α = δ_{i,α} · Dm_row_a[j], so the
    # exact 3×3 deviatoric block is:
    #
    #   K_dev[a,b][α,β] = k_mu·s·dot(Dm_row_a, Dm_row_b)·δ_{αβ}
    #                    + coef·(F·Dm_row_a)[α]·(F·Dm_row_b)[β]
    #
    # Node 0 is the reference: its Dm_row is minus the sum of the other three
    # (Newton's law: Σ_a ∂F/∂x_a = 0).
    dm_row_1 = wp.vec3(Dm[0, 0], Dm[0, 1], Dm[0, 2])
    dm_row_2 = wp.vec3(Dm[1, 0], Dm[1, 1], Dm[1, 2])
    dm_row_3 = wp.vec3(Dm[2, 0], Dm[2, 1], Dm[2, 2])
    dm_row_0 = -(dm_row_1 + dm_row_2 + dm_row_3)

    # F·Dm_row_a — one matrix-vector product per node (vec3).
    F_dm_0 = F * dm_row_0
    F_dm_1 = F * dm_row_1
    F_dm_2 = F * dm_row_2
    F_dm_3 = F * dm_row_3

    dJdx = wp.vec3()
    dm_row_a = wp.vec3()
    F_dm_a = wp.vec3()

    # Per-tet rest volume is already folded into k_mu / k_lambda above, so
    # the global scale is just +h². Sign is + because we want to add
    # ``+h²·H_pot`` to ``A``.
    scale = dt * dt

    # The geometric stiffness λ(J−α)·d²J/∂xₐ∂x_b = geo·skew3(w_ab) is dropped.
    # skew3 is antisymmetric, so those off-diagonal blocks break the PSD property
    # of the full 12×12 element Hessian even when the coefficient is positive.
    # K_dev + K_vol alone is provably PSD for all F (proof in the module docstring).
    for a in range(4):
        if a == 0:
            dJdx = dJdx0
            dm_row_a = dm_row_0
            F_dm_a = F_dm_0
        elif a == 1:
            dJdx = dJdx1
            dm_row_a = dm_row_1
            F_dm_a = F_dm_1
        elif a == 2:
            dJdx = dJdx2
            dm_row_a = dm_row_2
            F_dm_a = F_dm_2
        else:
            dJdx = dJdx3
            dm_row_a = dm_row_3
            F_dm_a = F_dm_3
        for b in range(4):
            dJdx_b = wp.vec3()
            dm_row_b = wp.vec3()
            F_dm_b = wp.vec3()
            if b == 0:
                dJdx_b = dJdx0
                dm_row_b = dm_row_0
                F_dm_b = F_dm_0
            elif b == 1:
                dJdx_b = dJdx1
                dm_row_b = dm_row_1
                F_dm_b = F_dm_1
            elif b == 2:
                dJdx_b = dJdx2
                dm_row_b = dm_row_2
                F_dm_b = F_dm_2
            else:
                dJdx_b = dJdx3
                dm_row_b = dm_row_3
                F_dm_b = F_dm_3

            # Deviatoric: K_dev[a,b] = k_mu·s·dot(Dm_row_a,Dm_row_b)·I
            #                         + coef·(F·Dm_row_a)⊗(F·Dm_row_b).
            dot_ab = wp.dot(dm_row_a, dm_row_b)
            K_dev_ab = wp.identity(n=3, dtype=float) * (k_mu * s * dot_ab) + wp.outer(F_dm_a, F_dm_b) * coef

            # Volumetric rank-1: H_vol[a,b] = λ · (dJ/dx_a) ⊗ (dJ/dx_b).
            K_vol_ab = wp.outer(dJdx, dJdx_b) * k_lambda

            # geo_coeff == 0: K_geo_ab is dropped (see comment above).
            K_ab = (K_dev_ab + K_vol_ab) * scale
            row_a = i if a == 0 else (j if a == 1 else (k if a == 2 else l))
            col_b = i if b == 0 else (j if b == 1 else (k if b == 2 else l))
            blk = block_offset + tid * 16 + a * 4 + b
            rows[blk] = row_a
            cols[blk] = col_b
            values[blk] = wp.mat33f(
                K_ab[0, 0],
                K_ab[0, 1],
                K_ab[0, 2],
                K_ab[1, 0],
                K_ab[1, 1],
                K_ab[1, 2],
                K_ab[2, 0],
                K_ab[2, 1],
                K_ab[2, 2],
            )


# =====================================================================
# 6. Triangle FEM (membrane + tangent + edge bending)
# =====================================================================
#
# Triangle membrane uses a 2D analogue of the tet model. For each triangle:
#
#     F   = (in-plane) Ds · Dm,    Ds = [x_1 − x_0, x_2 − x_0]
#     I_C = ‖F‖²
#     P   = μ · F · (I_C − 2)/I_C + k_d · dF/dt    (deviatoric)
#     plus an area-preservation constraint  c = area/area_rest − 1 + act
#
# The matrix kernel uses the lumped Baraff–Witkin tangent
# ``+h² · (μ + λ) · area_rest / 3`` per vertex on the diagonal (rest area, as
# the Lamé parameters are folded with it) — coarse but PSD by construction.
#
# Edge bending uses a dihedral-angle force (Bridson-style discrete bending):
#
#     f = −e_length · (k_e · (θ − θ_rest) + k_d · θ̇) · ∂θ/∂x
#
# The bending tangent is not in ``A``: bending is treated explicitly.
# =====================================================================


@wp.kernel
def eval_triangles(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    indices: wp.array2d[int],
    pose: wp.array[wp.mat22],
    activation: wp.array[float],
    materials: wp.array2d[float],
    f: wp.array[wp.vec3],
):
    """Triangle membrane force ``[N]`` (deviatoric + area + drag/lift)."""
    tid = wp.tid()

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]
    k_drag = materials[tid, 3]
    k_lift = materials[tid, 4]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]

    x10 = x1 - x0
    x20 = x2 - x0

    v10 = v1 - v0
    v20 = v2 - v0

    Dm = pose[tid]

    inv_rest_area = wp.determinant(Dm) * 2.0
    rest_area = 1.0 / inv_rest_area

    # Fold area into Lamé params (per-element rather than per-unit-area).
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F (2D) = Ds · Dm where Ds is built from edge vectors.
    F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]
    F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]

    dFdt1 = v10 * Dm[0, 0] + v20 * Dm[1, 0]
    dFdt2 = v10 * Dm[0, 1] + v20 * Dm[1, 1]

    Ic = wp.dot(F1, F1) + wp.dot(F2, F2)

    # Deviatoric: P = μ · F · (I_C − 2)/I_C + k_d · dF/dt.
    deviatoric_scale = (Ic - 2.0) / Ic
    P1 = F1 * k_mu * deviatoric_scale + dFdt1 * k_damp
    P2 = F2 * k_mu * deviatoric_scale + dFdt2 * k_damp

    f1 = P1 * Dm[0, 0] + P2 * Dm[0, 1]
    f2 = P1 * Dm[1, 0] + P2 * Dm[1, 1]

    # Area preservation: c = area/area_rest − 1 + act.
    n = wp.cross(x10, x20)
    area = wp.length(n) * 0.5

    act = activation[tid]
    c = area * inv_rest_area - 1.0 + act

    n = wp.normalize(n)
    dcdq = wp.cross(x20, n) * inv_rest_area * 0.5
    dcdr = wp.cross(n, x10) * inv_rest_area * 0.5

    f_area = k_lambda * c
    dcdt = wp.dot(dcdq, v1) + wp.dot(dcdr, v2) - wp.dot(dcdq + dcdr, v0)
    f_damp = k_damp * dcdt

    f1 = f1 + dcdq * (f_area + f_damp)
    f2 = f2 + dcdr * (f_area + f_damp)
    f0 = f1 + f2

    # Aerodynamic drag + lift (cloth-like).
    vmid = (v0 + v1 + v2) * 0.3333
    vdir = wp.normalize(vmid)
    f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))
    f_lift = n * (k_lift * area * (wp.HALF_PI - wp.acos(wp.dot(n, vdir)))) * wp.dot(vmid, vmid)

    f0 = f0 - f_drag - f_lift
    f1 = f1 + f_drag + f_lift
    f2 = f2 + f_drag + f_lift

    wp.atomic_add(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


@wp.kernel
def build_system_matrix_tri_kernel(
    indices: wp.array2d[int],
    pose: wp.array[wp.mat22],
    materials: wp.array2d[float],
    dt: wp.float32,
    block_offset: wp.int32,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
):
    """Lumped triangle tangent: ``A_ii += +h²·(μ + λ)·area_rest/3·I`` per vertex.

    Baraff–Witkin lumping: a coarse approximation that is diagonal-PSD by
    construction (no per-element eigendecomposition needed). Depends only on
    rest geometry (``pose``) and material constants — position-independent, so
    this kernel runs once at construction, not every frame. Skips degenerate
    triangles by writing zero blocks.
    """
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]

    Dm = pose[tid]
    det_Dm = wp.determinant(Dm)
    inv_rest_area = det_Dm * 2.0
    if inv_rest_area <= 0.0:
        # Degenerate triangle ⇒ zero blocks, skip.
        zero = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rows[block_offset + tid * 3 + 0] = i
        cols[block_offset + tid * 3 + 0] = i
        values[block_offset + tid * 3 + 0] = zero
        rows[block_offset + tid * 3 + 1] = j
        cols[block_offset + tid * 3 + 1] = j
        values[block_offset + tid * 3 + 1] = zero
        rows[block_offset + tid * 3 + 2] = k
        cols[block_offset + tid * 3 + 2] = k
        values[block_offset + tid * 3 + 2] = zero
        return
    rest_area = 1.0 / inv_rest_area

    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    diag_val = dt * dt * (k_mu + k_lambda) / 3.0
    block = wp.mat33f(diag_val, 0.0, 0.0, 0.0, diag_val, 0.0, 0.0, 0.0, diag_val)

    rows[block_offset + tid * 3 + 0] = i
    cols[block_offset + tid * 3 + 0] = i
    values[block_offset + tid * 3 + 0] = block
    rows[block_offset + tid * 3 + 1] = j
    cols[block_offset + tid * 3 + 1] = j
    values[block_offset + tid * 3 + 1] = block
    rows[block_offset + tid * 3 + 2] = k
    cols[block_offset + tid * 3 + 2] = k
    values[block_offset + tid * 3 + 2] = block


# 7. Particle-particle interactions (hash grid)
# =====================================================================
#
# For granular / fluid-like particle ensembles. Per particle, walks neighbour
# cells in the hash grid and applies a penalty + Coulomb-friction force.
# Skipped entirely for FEM-only meshes (where ``model.particle_grid is None``).
# =====================================================================


@wp.func
def particle_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    """Pairwise particle interaction: penalty + damping + Coulomb friction.

    ``n`` is the contact normal (i → j), ``v`` the relative velocity, ``c``
    the signed gap (negative when overlapping). Returns the force on
    particle i; particle j gets the negative.
    """
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d

    fn = jn + jd

    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Coulomb cap: |f_t| ≤ μ · |f_n|.
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))

    return -n * fn - vt * ft


@wp.kernel
def eval_particle_forces(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    # outputs
    particle_f: wp.array[wp.vec3],
):
    """Aggregate per-particle force from hash-grid neighbour pairs ``[N]``."""
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # Hash grid not built yet.
        return
    if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]

    f = wp.vec3()
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:
            n = x - particle_x[index]
            d = wp.length(n)
            if d < 1.0e-10:  # co-located vertices (e.g. glued seam) — skip
                continue
            err = d - radius - particle_radius[index]
            if err <= k_cohesion:
                n = n / d
                vrel = v - particle_v[index]
                f = f + particle_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f


# =====================================================================
# 8. Particle-rigid contact
# =====================================================================
#
# Three flavours, picked per-solver-config:
#
# * Analytic ground plane n·x + d = 0 — penalty + Coulomb.
# * Force-based soft-rigid contact from ``model.collide(state)`` output.
# * Constraint-based PBD-style post-integration position projection (relaxed
#   Gauss–Seidel; no XPBD compliance term or running Lagrange multiplier)
#   with per-particle / per-body delta accumulation; the
#   ``apply_particle_corrections`` kernel applies the deltas with clamps.
# =====================================================================


@wp.kernel
def eval_particle_ground_contacts(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    ground: wp.array[float],
    gravity_world0: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
):
    """Analytic ground-plane contact: penalty + Coulomb friction ``[N]``.

    The plane is ``n · x + d = 0`` (passed as a 4-float array). Penetration
    ``c = n·x + d − r`` is clamped at zero for "no jump up". Friction obeys
    ``|f_t| ≤ μ · N`` with ``N = min(|f_n|, m·‖g‖)`` — capping by weight so
    a stiff penalty doesn't make a particle infinitely sticky.

    ``gravity_world0`` is a length-≥1 ``vec3`` array so ``‖g‖`` is read on
    the device (avoids ``model.gravity.numpy()`` host-sync that breaks CUDA-
    graph capture, cuda 906).
    """
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[tid]
    v = particle_v[tid]
    radius = particle_radius[tid]
    inv_m = particle_inv_mass[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)

    vn = wp.dot(n, v)
    jn = c * ke
    if c >= 0.0:
        return

    jd = wp.min(vn, 0.0) * kd
    fn = jn + jd

    vt = v - n * vn
    vs = wp.length(vt)
    if vs > 0.0:
        vt = vt / vs

    # ‖g‖ for world 0, read on-device.
    gravity_magnitude = float(9.81)
    if gravity_world0.shape[0] > 0:
        gravity_magnitude = wp.length(gravity_world0[0])

    # Coulomb with weight-supported normal cap: N_eff = min(|f_n|, m·g).
    inv_m_safe = wp.max(inv_m, 1e-9)  # avoid div0 in either wp.where branch
    m = wp.where(inv_m > 0.0, 1.0 / inv_m_safe, 0.0)
    n_cap = m * gravity_magnitude
    fn_eff = wp.min(wp.abs(fn), n_cap)
    ft = wp.min(vs * kf, mu * fn_eff)

    f[tid] = f[tid] - n * fn - vt * ft


@wp.kernel
def eval_soft_contacts(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    soft_contact_count: wp.array[wp.int32],
    soft_contact_particle: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    particle_radius: wp.array[float],
    f: wp.array[wp.vec3],
):
    """Force-based soft-rigid contact from the collision pipeline output ``[N]``.

    Penalty + damping + Coulomb friction. Reads contact pairs from
    ``contacts.soft_contact_*``; the kernel returns early past ``count`` so
    the launch dim can safely be ``soft_contact_max`` (no host sync needed).
    """
    tid = wp.tid()

    count = soft_contact_count[0]
    if tid >= count:
        return

    particle_idx = soft_contact_particle[tid]
    if particle_idx < 0:
        return

    body_pos = soft_contact_body_pos[tid]
    body_vel = soft_contact_body_vel[tid]
    n = soft_contact_normal[tid]

    x = particle_x[particle_idx]
    v = particle_v[particle_idx]
    radius = particle_radius[particle_idx]

    # Penetration: contact normal points body → particle.
    d = wp.dot(x - body_pos, n)
    penetration = radius - d

    if penetration > 0.0:
        rel_v = v - body_vel
        vn = wp.dot(rel_v, n)
        vt = rel_v - vn * n

        # Normal: spring + dissipative damping (only opposing approach).
        fn = ke * penetration - kd * wp.min(vn, 0.0)

        # Coulomb cap with non-negative normal (damping shouldn't make
        # friction reverse direction).
        vt_mag = wp.length(vt)
        if vt_mag > 1.0e-6:
            fn_safe = wp.max(0.0, fn)
            ft_max = mu * fn_safe
            ft = wp.min(kf * vt_mag, ft_max)
            force = fn * n - ft * (vt / vt_mag)
        else:
            force = fn * n

        wp.atomic_add(f, particle_idx, force)


# =====================================================================
# 9. Gravity
# =====================================================================
#
# Reads gravity from a length-1 device array so there's no host sync inside
# CUDA-graph capture (``model.gravity.numpy()`` would break capture).
# =====================================================================


@wp.kernel
def eval_gravity_from_array(
    gravity: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    forces: wp.array[wp.vec3],
):
    """Per-particle gravity ``f = m_i · g`` ``[N]`` for active particles."""
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    g = gravity[0]
    m = particle_mass[tid]
    forces[tid] = wp.vec3(g[0] * m, g[1] * m, g[2] * m)


@wp.kernel
def eval_linear_damping_kernel(
    particle_v: wp.array[wp.vec3],
    particle_mass: wp.array[wp.float32],
    particle_flags: wp.array[wp.int32],
    alpha: wp.float32,
    forces: wp.array[wp.vec3],
):
    """Mass-proportional Rayleigh damping ``f = −α · m_i · v_i`` [N].

    Provides rigid-body velocity damping (translation + rotation) that the
    FEM volumetric term cannot supply — the latter only damps deformation-rate,
    not rigid motion.  The explicit force ``−α·m·v`` is unconditionally stable
    for ``h·α ≪ 1`` (typically α ≤ 5 s⁻¹ at h = 1.67 ms).
    """
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    forces[tid] = -alpha * particle_mass[tid] * particle_v[tid]


# =====================================================================
# 10. Inflation: rest-pose scaling
# =====================================================================
#
# Pressure ``p`` (volume ratio target) is applied by scaling the cached
# original rest configuration:
#
#     Dm  ←  Dm_orig  / cbrt(p)         (per tet — Dm is rest⁻¹)
#     L₀  ←  L₀_orig  · cbrt(p)         (per spring)
#
# For per-chamber control, each tet / spring carries an int32 chamber id
# (``-1`` ⇒ excluded, kept at original rest config). Pressure is clamped
# in-kernel to ``[1e-6, 100]`` to defend against bad inputs.
#
# The volume kernel ``compute_volume_kernel`` reduces tet volumes for the
# host-side telemetry methods on ``SolverInflatable``.
# =====================================================================


@wp.kernel
def scale_spring_rest_lengths_kernel(
    original_rest_lengths: wp.array[wp.float32],
    scale: wp.float32,
    scaled_rest_lengths: wp.array[wp.float32],
):
    """Per-spring scalar scale: ``L₀ ← L₀_orig · cbrt(p)``."""
    sid = wp.tid()
    scaled_rest_lengths[sid] = original_rest_lengths[sid] * scale


@wp.kernel
def scale_tet_poses_kernel(
    original_poses: wp.array[wp.mat33],
    scale: wp.float32,
    scaled_poses: wp.array[wp.mat33],
):
    """Per-tet inverse-rest scale: ``Dm ← Dm_orig / cbrt(p)``.

    The model stores ``Dm = rest⁻¹``, so to grow the rest shape by linear
    factor ``s = cbrt(p)``, we divide ``Dm`` by ``s``.
    """
    tid = wp.tid()
    inv_scale = 1.0 / scale
    orig = original_poses[tid]
    scaled_poses[tid] = wp.mat33(
        orig[0, 0] * inv_scale,
        orig[0, 1] * inv_scale,
        orig[0, 2] * inv_scale,
        orig[1, 0] * inv_scale,
        orig[1, 1] * inv_scale,
        orig[1, 2] * inv_scale,
        orig[2, 0] * inv_scale,
        orig[2, 1] * inv_scale,
        orig[2, 2] * inv_scale,
    )


@wp.kernel
def scale_spring_rest_lengths_per_chamber_kernel(
    original_rest_lengths: wp.array[wp.float32],
    spring_chamber_mask: wp.array[wp.int32],
    chamber_pressures: wp.array[wp.float32],
    num_chambers: int,
    scaled_rest_lengths: wp.array[wp.float32],
):
    """Per-spring chamber-specific scale ``L₀ ← L₀_orig · cbrt(p_chamber)``.

    Mask entry ``-1`` ⇒ spring is in the rigid base, keep original length.
    """
    sid = wp.tid()
    c = spring_chamber_mask[sid]
    if c < 0:
        scaled_rest_lengths[sid] = original_rest_lengths[sid]
        return
    c = wp.min(c, num_chambers - 1)
    pressure = chamber_pressures[c]
    pressure = wp.max(1.0e-6, wp.min(pressure, 100.0))
    scale = wp.cbrt(pressure)
    scaled_rest_lengths[sid] = original_rest_lengths[sid] * scale


@wp.kernel
def scale_tet_poses_per_chamber_kernel(
    original_poses: wp.array[wp.mat33],
    tet_chamber_mask: wp.array[wp.int32],
    chamber_pressures: wp.array[wp.float32],
    num_chambers: int,
    scaled_poses: wp.array[wp.mat33],
):
    """Per-tet chamber-specific inverse-rest scale ``Dm ← Dm_orig / cbrt(p_chamber)``.

    Mask entry ``-1`` ⇒ tet is in the rigid base, keep original ``Dm``.
    """
    tid = wp.tid()
    c = tet_chamber_mask[tid]
    orig = original_poses[tid]
    if c < 0:
        scaled_poses[tid] = orig
        return
    c = wp.min(c, num_chambers - 1)
    pressure = chamber_pressures[c]
    pressure = wp.max(1.0e-6, wp.min(pressure, 100.0))
    linear_scale = wp.cbrt(pressure)
    inv_scale = 1.0 / linear_scale
    scaled_poses[tid] = wp.mat33(
        orig[0, 0] * inv_scale,
        orig[0, 1] * inv_scale,
        orig[0, 2] * inv_scale,
        orig[1, 0] * inv_scale,
        orig[1, 1] * inv_scale,
        orig[1, 2] * inv_scale,
        orig[2, 0] * inv_scale,
        orig[2, 1] * inv_scale,
        orig[2, 2] * inv_scale,
    )


@wp.kernel
def compute_volume_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_volumes: wp.array[wp.float32],
):
    """Per-tet volume ``V_e = ⅙ · |det([e₁, e₂, e₃])|`` ``[m³]``.

    Edge vectors ``e_i = x_i − x_0``. Used by the host-side
    :meth:`SolverInflatable.compute_volume` to sum into a single volume
    measurement.
    """
    tid = wp.tid()

    i0 = tet_indices[tid, 0]
    i1 = tet_indices[tid, 1]
    i2 = tet_indices[tid, 2]
    i3 = tet_indices[tid, 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    cross = wp.cross(e2, e3)
    det = wp.dot(e1, cross)

    tet_volumes[tid] = wp.abs(det) / 6.0


# =====================================================================
# 11. Hexahedral FEM kernels (Q1 trilinear element, 2×2×2 Gauss rule)
# =====================================================================

# 1. Constants and small helpers
# ---------------------------------------------------------------------------

# Gauss-point parametric coordinate ±1/√3
_HEX_GP_FLOAT = 0.5773502691896258

HEX_GP = wp.constant(wp.float32(_HEX_GP_FLOAT))


@wp.func
def _hex_gauss_xi(g: int) -> float:
    """ξ-coordinate of 2×2×2 Gauss point g (0–7)."""
    if g == 1 or g == 2 or g == 5 or g == 6:
        return HEX_GP
    return -HEX_GP


@wp.func
def _hex_gauss_eta(g: int) -> float:
    """η-coordinate of 2×2×2 Gauss point g (0–7)."""
    if g == 2 or g == 3 or g == 6 or g == 7:
        return HEX_GP
    return -HEX_GP


@wp.func
def _hex_gauss_zeta(g: int) -> float:
    """ζ-coordinate of 2×2×2 Gauss point g (0–7)."""
    if g == 4 or g == 5 or g == 6 or g == 7:
        return HEX_GP
    return -HEX_GP


@wp.func
def _hex_node_xi(a: int) -> float:
    """ξ-sign of node a (0–7)."""
    if a == 1 or a == 2 or a == 5 or a == 6:
        return wp.float32(1.0)
    return wp.float32(-1.0)


@wp.func
def _hex_node_eta(a: int) -> float:
    """η-sign of node a (0–7)."""
    if a == 2 or a == 3 or a == 6 or a == 7:
        return wp.float32(1.0)
    return wp.float32(-1.0)


@wp.func
def _hex_node_zeta(a: int) -> float:
    """ζ-sign of node a (0–7)."""
    if a == 4 or a == 5 or a == 6 or a == 7:
        return wp.float32(1.0)
    return wp.float32(-1.0)


@wp.func
def _hex_dN_dxi(a: int, xi: float, eta: float, zeta: float) -> wp.vec3:
    """Shape-function gradient ∂N_a/∂(ξ, η, ζ) for trilinear hex node a."""
    xa = _hex_node_xi(a)
    ya = _hex_node_eta(a)
    za = _hex_node_zeta(a)
    dNdxi = xa * (1.0 + ya * eta) * (1.0 + za * zeta) * 0.125
    dNdeta = (1.0 + xa * xi) * ya * (1.0 + za * zeta) * 0.125
    dNdzeta = (1.0 + xa * xi) * (1.0 + ya * eta) * za * 0.125
    return wp.vec3(dNdxi, dNdeta, dNdzeta)


@wp.func
def _select_vec3(
    a: int, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, v4: wp.vec3, v5: wp.vec3, v6: wp.vec3, v7: wp.vec3
) -> wp.vec3:
    """Select v{a} for a ∈ [0, 7] via an if-chain (no dynamic indexing needed)."""
    if a == 0:
        return v0
    elif a == 1:
        return v1
    elif a == 2:
        return v2
    elif a == 3:
        return v3
    elif a == 4:
        return v4
    elif a == 5:
        return v5
    elif a == 6:
        return v6
    else:
        return v7


@wp.func
def _select_int(a: int, n0: int, n1: int, n2: int, n3: int, n4: int, n5: int, n6: int, n7: int) -> int:
    """Select n{a} for a ∈ [0, 7]."""
    if a == 0:
        return n0
    elif a == 1:
        return n1
    elif a == 2:
        return n2
    elif a == 3:
        return n3
    elif a == 4:
        return n4
    elif a == 5:
        return n5
    elif a == 6:
        return n6
    else:
        return n7


@wp.func
def _clamp_stretch(F: wp.mat33f, min_s: float, max_s: float) -> wp.mat33f:
    """Clamp principal stretches of F to [min_s, max_s] via SVD.

    Identical in purpose to the tetrahedral kernel's ``clamp_deformation_stretch``.
    Disabled (F returned unchanged) when either bound is negative.
    """
    if min_s < 0.0 or max_s < 0.0:
        return F
    U = wp.mat33f()
    S = wp.vec3f()
    V = wp.mat33f()
    wp.svd3(F, U, S, V)
    S = wp.max(wp.min(S, wp.vec3f(max_s)), wp.vec3f(min_s))
    return U * wp.diag(S) * wp.transpose(V)


# ---------------------------------------------------------------------------
# 2. Force kernel
# ---------------------------------------------------------------------------


@wp.kernel
def eval_hexahedra(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    inv_J0: wp.array2d[wp.mat33f],
    det_J0_w: wp.array2d[wp.float32],
    activation: wp.array[wp.float32],
    materials: wp.array2d[wp.float32],
    min_stretch: float,
    max_stretch: float,
    f: wp.array[wp.vec3],
):
    """Stable Neo-Hookean hexahedral force per node [N].

    One thread per element.  Integrates PK1 stress over the 2×2×2 Gauss rule.
    ``activation[e]`` shifts the volumetric strain (J − α + act), identical to
    the tetrahedral kernel convention.  Rayleigh damping uses dF/dt.
    """
    eid = wp.tid()

    k_mu = materials[eid, 0]
    k_lambda = materials[eid, 1]
    k_damp = materials[eid, 2]
    act = activation[eid]
    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    n0 = indices[eid, 0]
    n1 = indices[eid, 1]
    n2 = indices[eid, 2]
    n3 = indices[eid, 3]
    n4 = indices[eid, 4]
    n5 = indices[eid, 5]
    n6 = indices[eid, 6]
    n7 = indices[eid, 7]

    x0 = x[n0]
    x1 = x[n1]
    x2 = x[n2]
    x3 = x[n3]
    x4 = x[n4]
    x5 = x[n5]
    x6 = x[n6]
    x7 = x[n7]
    v0 = v[n0]
    v1 = v[n1]
    v2 = v[n2]
    v3 = v[n3]
    v4 = v[n4]
    v5 = v[n5]
    v6 = v[n6]
    v7 = v[n7]

    for g in range(8):
        xi_g = _hex_gauss_xi(g)
        eta_g = _hex_gauss_eta(g)
        zeta_g = _hex_gauss_zeta(g)

        inv_J0_g = inv_J0[eid, g]
        det_J0_g = det_J0_w[eid, g]
        inv_J0_gT = wp.transpose(inv_J0_g)

        # Reference-config shape gradients ∇_X N_a = J0^{-T} · ∇_ξ N_a
        dN0 = inv_J0_gT * _hex_dN_dxi(0, xi_g, eta_g, zeta_g)
        dN1 = inv_J0_gT * _hex_dN_dxi(1, xi_g, eta_g, zeta_g)
        dN2 = inv_J0_gT * _hex_dN_dxi(2, xi_g, eta_g, zeta_g)
        dN3 = inv_J0_gT * _hex_dN_dxi(3, xi_g, eta_g, zeta_g)
        dN4 = inv_J0_gT * _hex_dN_dxi(4, xi_g, eta_g, zeta_g)
        dN5 = inv_J0_gT * _hex_dN_dxi(5, xi_g, eta_g, zeta_g)
        dN6 = inv_J0_gT * _hex_dN_dxi(6, xi_g, eta_g, zeta_g)
        dN7 = inv_J0_gT * _hex_dN_dxi(7, xi_g, eta_g, zeta_g)

        # Deformation gradient F = Σ_a x_a^curr ⊗ ∇_X N_a
        F = (
            wp.outer(x0, dN0)
            + wp.outer(x1, dN1)
            + wp.outer(x2, dN2)
            + wp.outer(x3, dN3)
            + wp.outer(x4, dN4)
            + wp.outer(x5, dN5)
            + wp.outer(x6, dN6)
            + wp.outer(x7, dN7)
        )
        F = _clamp_stretch(F, min_stretch, max_stretch)

        # Velocity gradient dF/dt (for Rayleigh damping)
        dFdt = (
            wp.outer(v0, dN0)
            + wp.outer(v1, dN1)
            + wp.outer(v2, dN2)
            + wp.outer(v3, dN3)
            + wp.outer(v4, dN4)
            + wp.outer(v5, dN5)
            + wp.outer(v6, dN6)
            + wp.outer(v7, dN7)
        )

        # I_C = ‖F‖²_F
        fc0 = wp.vec3f(F[0, 0], F[1, 0], F[2, 0])
        fc1 = wp.vec3f(F[0, 1], F[1, 1], F[2, 1])
        fc2 = wp.vec3f(F[0, 2], F[1, 2], F[2, 2])
        Ic = wp.dot(fc0, fc0) + wp.dot(fc1, fc1) + wp.dot(fc2, fc2)

        # Deviatoric PK1 + Rayleigh damping
        P = F * (k_mu * (1.0 - 1.0 / (Ic + 1.0))) + dFdt * k_damp

        # Volumetric PK1: P_vol = λ·(J−α+act) · cof(F)
        J = wp.max(wp.determinant(F), wp.float32(0.01))
        cof_col0 = wp.cross(fc1, fc2)
        cof_col1 = wp.cross(fc2, fc0)
        cof_col2 = wp.cross(fc0, fc1)
        cof_F = wp.matrix_from_cols(cof_col0, cof_col1, cof_col2)
        P = P + cof_F * (k_lambda * (J - alpha + act))

        # Nodal forces: f_a −= det_J0_g · P · ∇_X N_a
        w = det_J0_g
        wp.atomic_sub(f, n0, w * (P * dN0))
        wp.atomic_sub(f, n1, w * (P * dN1))
        wp.atomic_sub(f, n2, w * (P * dN2))
        wp.atomic_sub(f, n3, w * (P * dN3))
        wp.atomic_sub(f, n4, w * (P * dN4))
        wp.atomic_sub(f, n5, w * (P * dN5))
        wp.atomic_sub(f, n6, w * (P * dN6))
        wp.atomic_sub(f, n7, w * (P * dN7))


# ---------------------------------------------------------------------------
# 3. System-matrix (Hessian) kernel
# ---------------------------------------------------------------------------


@wp.kernel
def build_system_matrix_hex_kernel(
    x: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    inv_J0: wp.array2d[wp.mat33f],
    det_J0_w: wp.array2d[wp.float32],
    materials: wp.array2d[wp.float32],
    min_stretch: float,
    max_stretch: float,
    dt: wp.float32,
    block_offset: wp.int32,
    rows: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    dirty: wp.array[wp.int32],
):
    """Per-hex PSD tangent: one thread per element × node-pair (a, b).

    Launch with ``dim = hex_count * 64``.  Thread ``tid`` handles element
    ``eid = tid // 64`` and block pair ``(a, b) = (tid % 64 // 8, tid % 8)``.

    Each thread accumulates its single 3×3 block K[a,b] over 8 Gauss points,
    then writes one BSR triplet.  This keeps register pressure low (one mat33f
    accumulator) at the cost of recomputing the Gauss data 64× per element;
    the ``inv_J0`` / ``det_J0_w`` rows for a given element are accessed by all
    64 threads of that element and should reside in L1 after the first hit.

    The geometric stiffness λ(J−α)·d²J/(∂x_a ∂x_b) is omitted (same reason
    as in the tetrahedral solver: its skew-antisymmetric 3×3 off-diagonal
    contributions break the BSR positive-semidefiniteness for stiff materials).

    K[a,b] = Σ_g det_J0_g · [ k_mu·s · dot(∇N_a,∇N_b)·I
                              + 2·k_mu/(I_C+1)² · (F·∇N_a)⊗(F·∇N_b)
                              + k_lambda · (cof(F)·∇N_a)⊗(cof(F)·∇N_b) ]

    ``dirty[eid] == 0`` skips assembly (stale-but-correct triplets remain).
    """
    tid = wp.tid()
    eid = tid // 64
    ab = tid % 64
    a = ab // 8
    b = ab % 8

    if dirty[eid] == 0:
        return

    k_mu = materials[eid, 0]
    k_lambda = materials[eid, 1]

    n0 = indices[eid, 0]
    n1 = indices[eid, 1]
    n2 = indices[eid, 2]
    n3 = indices[eid, 3]
    n4 = indices[eid, 4]
    n5 = indices[eid, 5]
    n6 = indices[eid, 6]
    n7 = indices[eid, 7]

    x0 = x[n0]
    x1 = x[n1]
    x2 = x[n2]
    x3 = x[n3]
    x4 = x[n4]
    x5 = x[n5]
    x6 = x[n6]
    x7 = x[n7]

    na = _select_int(a, n0, n1, n2, n3, n4, n5, n6, n7)
    nb = _select_int(b, n0, n1, n2, n3, n4, n5, n6, n7)

    Kab = wp.mat33f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for g in range(8):
        xi_g = _hex_gauss_xi(g)
        eta_g = _hex_gauss_eta(g)
        zeta_g = _hex_gauss_zeta(g)

        inv_J0_g = inv_J0[eid, g]
        det_J0_g = det_J0_w[eid, g]
        inv_J0_gT = wp.transpose(inv_J0_g)

        # Reference-config shape gradients for all 8 nodes
        dN0g = inv_J0_gT * _hex_dN_dxi(0, xi_g, eta_g, zeta_g)
        dN1g = inv_J0_gT * _hex_dN_dxi(1, xi_g, eta_g, zeta_g)
        dN2g = inv_J0_gT * _hex_dN_dxi(2, xi_g, eta_g, zeta_g)
        dN3g = inv_J0_gT * _hex_dN_dxi(3, xi_g, eta_g, zeta_g)
        dN4g = inv_J0_gT * _hex_dN_dxi(4, xi_g, eta_g, zeta_g)
        dN5g = inv_J0_gT * _hex_dN_dxi(5, xi_g, eta_g, zeta_g)
        dN6g = inv_J0_gT * _hex_dN_dxi(6, xi_g, eta_g, zeta_g)
        dN7g = inv_J0_gT * _hex_dN_dxi(7, xi_g, eta_g, zeta_g)

        # Deformation gradient
        F = (
            wp.outer(x0, dN0g)
            + wp.outer(x1, dN1g)
            + wp.outer(x2, dN2g)
            + wp.outer(x3, dN3g)
            + wp.outer(x4, dN4g)
            + wp.outer(x5, dN5g)
            + wp.outer(x6, dN6g)
            + wp.outer(x7, dN7g)
        )
        F = _clamp_stretch(F, min_stretch, max_stretch)

        fc0 = wp.vec3f(F[0, 0], F[1, 0], F[2, 0])
        fc1 = wp.vec3f(F[0, 1], F[1, 1], F[2, 1])
        fc2 = wp.vec3f(F[0, 2], F[1, 2], F[2, 2])
        Ic = wp.dot(fc0, fc0) + wp.dot(fc1, fc1) + wp.dot(fc2, fc2)
        Ic1 = wp.max(Ic + 1.0, wp.float32(1.0e-6))
        s = 1.0 - 1.0 / Ic1
        coef = k_mu * 2.0 / (Ic1 * Ic1)

        # Cofactor matrix cof(F) = [fc1×fc2, fc2×fc0, fc0×fc1]
        cof_col0 = wp.cross(fc1, fc2)
        cof_col1 = wp.cross(fc2, fc0)
        cof_col2 = wp.cross(fc0, fc1)
        cof_F = wp.matrix_from_cols(cof_col0, cof_col1, cof_col2)

        # Shape gradients for the specific (a, b) pair
        dN_Xa = _select_vec3(a, dN0g, dN1g, dN2g, dN3g, dN4g, dN5g, dN6g, dN7g)
        dN_Xb = _select_vec3(b, dN0g, dN1g, dN2g, dN3g, dN4g, dN5g, dN6g, dN7g)

        Fd_a = F * dN_Xa  # F · ∇_X N_a
        Fd_b = F * dN_Xb  # F · ∇_X N_b
        dJ_a = cof_F * dN_Xa  # ∂J/∂x_a
        dJ_b = cof_F * dN_Xb  # ∂J/∂x_b

        dot_ab = wp.dot(dN_Xa, dN_Xb)
        I3 = wp.identity(n=3, dtype=wp.float32)

        K_dev = I3 * (k_mu * s * dot_ab) + wp.outer(Fd_a, Fd_b) * coef
        K_vol = wp.outer(dJ_a, dJ_b) * k_lambda
        Kab = Kab + (K_dev + K_vol) * det_J0_g

    scale = dt * dt
    blk = block_offset + tid
    rows[blk] = na
    cols[blk] = nb
    values[blk] = Kab * scale


# ---------------------------------------------------------------------------
# 4. Volume kernel
# ---------------------------------------------------------------------------


@wp.kernel
def compute_hex_volume_kernel(
    det_J0_w: wp.array2d[wp.float32],
    volumes: wp.array[wp.float32],
):
    """Element volume V_e = Σ_g det_J0_g (Gauss weight = 1 already folded in)."""
    eid = wp.tid()
    v = wp.float32(0.0)
    for g in range(8):
        v = v + det_J0_w[eid, g]
    volumes[eid] = v


# ---------------------------------------------------------------------------
# 5. Inflation scaling kernels (used by SolverInflatableHex)
# ---------------------------------------------------------------------------


@wp.kernel
def scale_hex_gauss_kernel(
    original_inv_J0: wp.array2d[wp.mat33f],
    original_det_J0_w: wp.array2d[wp.float32],
    linear_scale: wp.float32,
    inv_J0_out: wp.array2d[wp.mat33f],
    det_J0_w_out: wp.array2d[wp.float32],
):
    """Uniform pressure scale for all hex elements.

    When the rest config expands by ``s = cbrt(p)``:

    - ``J0 → s · J0``  so  ``inv_J0 → inv_J0 / s``
    - ``det_J0_w → s³ · det_J0_w = p · det_J0_w``

    Launch with ``dim = hex_count * 8`` (one thread per element × Gauss point).
    """
    tid = wp.tid()
    eid = tid // 8
    g = tid % 8
    inv_s = wp.float32(1.0) / linear_scale
    vol_s = linear_scale * linear_scale * linear_scale  # = p
    inv_J0_out[eid, g] = original_inv_J0[eid, g] * inv_s
    det_J0_w_out[eid, g] = original_det_J0_w[eid, g] * vol_s


@wp.kernel
def scale_hex_gauss_per_chamber_kernel(
    original_inv_J0: wp.array2d[wp.mat33f],
    original_det_J0_w: wp.array2d[wp.float32],
    hex_chamber_mask: wp.array[wp.int32],
    chamber_pressures: wp.array[wp.float32],
    num_chambers: int,
    inv_J0_out: wp.array2d[wp.mat33f],
    det_J0_w_out: wp.array2d[wp.float32],
):
    """Per-chamber pressure scale for hex elements.

    Mask entry ``-1`` means the element belongs to the rigid base; its Gauss
    data is copied unchanged.

    Launch with ``dim = hex_count * 8``.
    """
    tid = wp.tid()
    eid = tid // 8
    g = tid % 8
    c = hex_chamber_mask[eid]
    if c < 0:
        inv_J0_out[eid, g] = original_inv_J0[eid, g]
        det_J0_w_out[eid, g] = original_det_J0_w[eid, g]
        return
    c = wp.min(c, num_chambers - 1)
    pressure = chamber_pressures[c]
    pressure = wp.max(wp.float32(1.0e-6), wp.min(pressure, wp.float32(100.0)))
    s = wp.cbrt(pressure)
    inv_s = wp.float32(1.0) / s
    vol_s = pressure  # = s³
    inv_J0_out[eid, g] = original_inv_J0[eid, g] * inv_s
    det_J0_w_out[eid, g] = original_det_J0_w[eid, g] * vol_s


# ---------------------------------------------------------------------------
# 12. Block-Jacobi (3x3) preconditioner
#
# Warp's built-in ``preconditioner(ptype="diag")`` keeps only the three
# diagonal scalars of each 3x3 block (see ``_extract_inverse_diagonal_blocked``
# in warp/optim/linear.py), discarding the intra-particle x/y/z coupling.  For
# Neo-Hookean elasticity that coupling is strong, so inverting the full 3x3
# block markedly improves CG conditioning.
# ---------------------------------------------------------------------------


@wp.kernel
def invert_block_diagonal_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33f],
    inv_diag: wp.array[wp.mat33f],
):
    """Invert each 3x3 diagonal block of a BSR matrix in place.

    Falls back to point-Jacobi (per-component reciprocal) when a block is
    numerically singular, so the result is never worse-conditioned than
    Warp's built-in ``"diag"`` preconditioner.
    """
    i = wp.tid()

    D = wp.identity(n=3, dtype=wp.float32)
    for k in range(offsets[i], offsets[i + 1]):
        if columns[k] == i:
            D = values[k]

    # Scale-aware singularity test: compare |det| against the cube of the mean
    # diagonal, so the threshold tracks the block's magnitude.
    tr = (D[0, 0] + D[1, 1] + D[2, 2]) / 3.0
    det = wp.determinant(D)

    if tr > 0.0 and wp.abs(det) > 1.0e-9 * tr * tr * tr:
        inv_diag[i] = wp.inverse(D)
    else:
        r0 = float(1.0)
        r1 = float(1.0)
        r2 = float(1.0)
        if wp.abs(D[0, 0]) > 1.0e-12:
            r0 = 1.0 / D[0, 0]
        if wp.abs(D[1, 1]) > 1.0e-12:
            r1 = 1.0 / D[1, 1]
        if wp.abs(D[2, 2]) > 1.0e-12:
            r2 = 1.0 / D[2, 2]
        inv_diag[i] = wp.mat33f(r0, 0.0, 0.0, 0.0, r1, 0.0, 0.0, 0.0, r2)


@wp.kernel
def block_jacobi_mv_kernel(
    Minv: wp.array[wp.mat33f],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    alpha: wp.float32,
    beta: wp.float32,
):
    """Generalised block-diagonal matvec ``z = alpha * (Minv @ x) + beta * y``."""
    i = wp.tid()
    s = wp.vec3(0.0, 0.0, 0.0)
    if alpha != 0.0:
        s = s + alpha * (Minv[i] * x[i])
    if beta != 0.0:
        s = s + beta * y[i]
    z[i] = s
