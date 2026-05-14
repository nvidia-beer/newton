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
``bsr_set_from_triplets`` can sum duplicates into ``A``. Per-step layout::

    [0 .. P)              — per-particle diagonal (mass + spring sum)
    [P .. P+2·S)          — per-spring (i,j) and (j,i) off-diagonals
    [P+2·S .. P+2·S+16·T) — per-tet 4×4 nodal Hessian blocks
    [... +3·R)            — per-tri 3 lumped diagonal blocks

where P = particle_count, S = spring_count, T = tet_count, R = tri_count.

Sign convention for forces
==========================

Force kernels return forces in physical convention: ``f`` points in the
direction the particle is pushed. Internal-force kernels for elastic
elements compute ``∂ψ/∂x_a`` first then subtract from the force buffer
(so ``f -= ∂ψ/∂x_a`` ⇒ stored ``f`` is ``−∂ψ/∂x_a``, the restoring force).

Stable Neo-Hookean tetrahedra
=============================

Per-element strain energy density (Smith, De Goes & Kim 2018):

    ψ(F) = ½·μ·(I_C − 3) − μ·log(I_C + 1) + ½·λ·(J − α)²
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
        reaction[p]   := particle_f_old[p] / dt
                         − mass · g − mass · target_dv[p] / dt

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
6. Triangle FEM     — membrane + tangent + edge bending
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
    dv: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    positions_in: wp.array(dtype=wp.vec3),
    velocities_in: wp.array(dtype=wp.vec3),
    positions_out: wp.array(dtype=wp.vec3),
    velocities_out: wp.array(dtype=wp.vec3),
):
    """Integrate the BE solution: ``v_{n+1} = v_n + Δv``, ``x_{n+1} = x_n + h·v_{n+1}``."""
    tid = wp.tid()
    vel = velocities_in[tid] + dv[tid]
    positions_out[tid] = positions_in[tid] + vel * dt
    velocities_out[tid] = vel


@wp.kernel
def apply_dirichlet_pin_kernel(
    mask: wp.array(dtype=wp.int32),
    target_dv: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    mass: wp.float32,
    target_scale: wp.float32,
    dt: float,
    # in/out
    particle_f: wp.array(dtype=wp.vec3),
    # out
    reaction: wp.array(dtype=wp.vec3),
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

    The reaction is the constraint force the rigid body must supply at the
    attach point: elastic / contact / damping forces the FEM wanted to apply
    at this vertex, minus gravity (already an external force on the soft
    particle), minus the inertial reaction ``m · target_dv / dt`` needed to
    accelerate the soft particle onto the rigid kinematic target. By Newton's
    third law the rigid body feels ``+reaction``; the glue forwards it onto
    ``state.body_f``. Including the inertial term is necessary for momentum
    conservation at the interface — without it the rigid body never feels the
    soft mass and the coupling oscillates under fast rigid motion (Macklin et
    al., XPBD rigid bodies, SCA 2020).

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
    reaction[p] = (f_total / dt) - mass * gravity - mass * target_dv[p] / dt

    particle_f[p] = target_scale * target_dv[p]


@wp.kernel
def filter_dirichlet_pin_in_bsr_kernel(
    mask: wp.array(dtype=wp.int32),
    target_dv: wp.array(dtype=wp.vec3),
    bsr_offsets: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_values: wp.array(dtype=wp.mat33f),
    # in/out
    particle_f: wp.array(dtype=wp.vec3),
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
    override, BEFORE the preconditioner is rebuilt and BEFORE the linear
    solve. Modifies ``A`` in place — ``A`` must be rebuilt from scratch on
    the next substep when a pin is active (the solver enforces this).

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
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
    mass: wp.float32,
):
    """Write ``A_ii = mass·I`` for each particle (no springs path)."""
    tid = wp.tid()
    rows[tid] = tid
    cols[tid] = tid
    values[tid] = wp.mat33f(mass, 0.0, 0.0, 0.0, mass, 0.0, 0.0, 0.0, mass)


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
#     A_ii  +=  +h² · k · I + h · k_d · I
#     A_ij  +=  −h² · k · I       (one block per (i,j), one per (j,i))
#
# Together this is a graph-Laplacian-style contribution: the constant mode
# (rigid translation) is in the null space of the spring contribution, so
# pure translations don't see spurious spring forces in the implicit step.
# =====================================================================

@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
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
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    spring_rest_direction: wp.array(dtype=wp.vec3),
    torque_stiffness: wp.float32,
    torque_damping: wp.float32,
    f: wp.array(dtype=wp.vec3),
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
    d0 = spring_rest_direction[tid]
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
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
    spring_indices: wp.array(dtype=int),
    spring_stiffness: wp.array(dtype=wp.float32),
    spring_damping: wp.array(dtype=wp.float32),
    dt: wp.float32,
    mass: wp.float32,
    n_springs: wp.int32,
):
    """Per-particle diagonal of A: ``A_ii = (mass + Σ_springs (h·d + h²·k))·I``.

    The off-diagonal counterpart ``A_ij = −(h·d + h²·k)·I`` lives in
    :func:`build_system_matrix_sparse_kernel`.
    """
    i = wp.tid()
    dt2 = dt * dt
    diag = mass
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
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
    indices: wp.array(dtype=int),
    spring_stiffness: wp.array(dtype=wp.float32),
    spring_damping: wp.array(dtype=wp.float32),
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
        neg_off, 0.0, 0.0,
        0.0, neg_off, 0.0,
        0.0, 0.0, neg_off,
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
#     ψ(F) = ½·μ·(I_C − 3) − μ·log(I_C + 1) + ½·λ·(J − α)²
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

@wp.kernel
def eval_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Stable Neo-Hookean tet force per node ``[N]``.

    Activation channel ``act`` adds to the volumetric strain ``(J − α + act)``
    so a contractile / expanding muscle can be driven by control without
    changing the rest pose. Rayleigh damping is applied via the deformation
    gradient time derivative ``dF/dt`` and ``dJ/dt``.

    ``J`` is clamped at ``J_MIN = 0.01`` to avoid the volumetric force
    exploding when a tet inverts (``J < 0``) or collapses (``J → 0``).
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
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp

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
    x: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat33),
    materials: wp.array2d(dtype=float),
    dt: wp.float32,
    block_offset: wp.int32,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
):
    """Per-tet Hessian: 4×4 = 16 nodal blocks of ``+h² · ∂²ψ/∂x_a∂x_b``.

    Deviatoric: ``H_dev[a,b] = (dF/dx_a)^T · (∂²ψ/∂F²) · (dF/dx_b)`` with
    ``∂²ψ/∂F²(B) = μ·s·B + (2μ/(I_C+1)²)·F·(F:B)`` and ``s = 1 − 1/(I_C+1)``.

    Volumetric: ``H_vol[a,b] = λ · (dJ/dx_a) ⊗ (dJ/dx_b)`` (the ``(J−α)`` term
    that would also appear is dropped here; near rest its contribution is
    small. PSD projection of ``H`` is the planned next iteration).

    ``∂F/∂x_a`` packing: ``∂F/∂x_0 = −Dm`` (full matrix), ``∂F/∂x_1 = [Dm[:,0],
    0, 0]``, ``∂F/∂x_2 = [0, Dm[:,1], 0]``, ``∂F/∂x_3 = [0, 0, Dm[:,2]]``.

    Degenerate / inverted rest poses (``inv_rest_volume ≤ 0``) write zero
    blocks instead of NaN-propagating.
    """
    tid = wp.tid()
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

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume

    F = Ds * Dm
    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)
    Ic1 = Ic + 1.0
    Ic1 = wp.max(Ic1, 1e-6)
    s = 1.0 - 1.0 / Ic1
    coef = k_mu * 2.0 / (Ic1 * Ic1)

    # ∂F/∂x_a: per-node 3×3 derivatives.
    dFdx0 = wp.mat33(
        -Dm[0, 0], -Dm[0, 1], -Dm[0, 2],
        -Dm[1, 0], -Dm[1, 1], -Dm[1, 2],
        -Dm[2, 0], -Dm[2, 1], -Dm[2, 2],
    )
    dFdx1 = wp.mat33(Dm[0, 0], 0.0, 0.0, Dm[1, 0], 0.0, 0.0, Dm[2, 0], 0.0, 0.0)
    dFdx2 = wp.mat33(0.0, Dm[0, 1], 0.0, 0.0, Dm[1, 1], 0.0, 0.0, Dm[2, 1], 0.0)
    dFdx3 = wp.mat33(0.0, 0.0, Dm[0, 2], 0.0, 0.0, Dm[1, 2], 0.0, 0.0, Dm[2, 2])

    # ∂J/∂x_a: per-node gradient of det(F).
    J = wp.determinant(F)
    J_MIN = 0.01
    J = wp.max(J, J_MIN)
    s_vol = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s_vol
    dJdx2 = wp.cross(x30, x10) * s_vol
    dJdx3 = wp.cross(x10, x20) * s_vol
    dJdx0 = -(dJdx1 + dJdx2 + dJdx3)

    dFdx = wp.mat33()
    dJdx = wp.vec3()

    # Per-tet rest volume is already folded into k_mu / k_lambda above, so
    # the global scale is just +h². Sign is + because we want to add
    # ``+h²·H_pot`` to ``A``.
    scale = dt * dt

    for a in range(4):
        if a == 0:
            dFdx = dFdx0
            dJdx = dJdx0
        elif a == 1:
            dFdx = dFdx1
            dJdx = dJdx1
        elif a == 2:
            dFdx = dFdx2
            dJdx = dJdx2
        else:
            dFdx = dFdx3
            dJdx = dJdx3
        for b in range(4):
            if b == 0:
                dFdx_b = dFdx0
                dJdx_b = dJdx0
            elif b == 1:
                dFdx_b = dFdx1
                dJdx_b = dJdx1
            elif b == 2:
                dFdx_b = dFdx2
                dJdx_b = dJdx2
            else:
                dFdx_b = dFdx3
                dJdx_b = dJdx3

            # Deviatoric: H_dev[a,b] = (dFdx_a)^T · (∂²ψ/∂F²)(dFdx_b).
            #   ∂²ψ/∂F²(B) = μ·s·B + coef·F·(F:B), with coef = 2μ/(I_C+1)².
            F_dot_dF_b = F[0, 0] * dFdx_b[0, 0] + F[0, 1] * dFdx_b[0, 1] + F[0, 2] * dFdx_b[0, 2]
            F_dot_dF_b += F[1, 0] * dFdx_b[1, 0] + F[1, 1] * dFdx_b[1, 1] + F[1, 2] * dFdx_b[1, 2]
            F_dot_dF_b += F[2, 0] * dFdx_b[2, 0] + F[2, 1] * dFdx_b[2, 1] + F[2, 2] * dFdx_b[2, 2]
            dP_dF_dFdx_b = wp.mat33(
                k_mu * s * dFdx_b[0, 0] + coef * F[0, 0] * F_dot_dF_b, k_mu * s * dFdx_b[0, 1] + coef * F[0, 1] * F_dot_dF_b, k_mu * s * dFdx_b[0, 2] + coef * F[0, 2] * F_dot_dF_b,
                k_mu * s * dFdx_b[1, 0] + coef * F[1, 0] * F_dot_dF_b, k_mu * s * dFdx_b[1, 1] + coef * F[1, 1] * F_dot_dF_b, k_mu * s * dFdx_b[1, 2] + coef * F[1, 2] * F_dot_dF_b,
                k_mu * s * dFdx_b[2, 0] + coef * F[2, 0] * F_dot_dF_b, k_mu * s * dFdx_b[2, 1] + coef * F[2, 1] * F_dot_dF_b, k_mu * s * dFdx_b[2, 2] + coef * F[2, 2] * F_dot_dF_b,
            )
            K_dev_ab = wp.transpose(dFdx) * dP_dF_dFdx_b

            # Volumetric: H_vol[a,b] = λ · (dJ/dx_a) ⊗ (dJ/dx_b).
            K_vol_ab = wp.outer(dJdx, dJdx) * k_lambda

            K_ab = (K_dev_ab + K_vol_ab) * scale
            row_a = i if a == 0 else (j if a == 1 else (k if a == 2 else l))
            col_b = i if b == 0 else (j if b == 1 else (k if b == 2 else l))
            blk = block_offset + tid * 16 + a * 4 + b
            rows[blk] = row_a
            cols[blk] = col_b
            values[blk] = wp.mat33f(K_ab[0, 0], K_ab[0, 1], K_ab[0, 2], K_ab[1, 0], K_ab[1, 1], K_ab[1, 2], K_ab[2, 0], K_ab[2, 1], K_ab[2, 2])


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
# ``+h² · (μ + λ) · area / 3`` per vertex on the diagonal — coarse but PSD
# by construction.
#
# Edge bending uses a dihedral-angle force (Bridson-style discrete bending):
#
#     f = −e_length · (k_e · (θ − θ_rest) + k_d · θ̇) · ∂θ/∂x
#
# The bending tangent is not in ``A``: bending is treated explicitly.
# =====================================================================

@wp.kernel
def eval_triangles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
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
    x: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    materials: wp.array2d(dtype=float),
    dt: wp.float32,
    block_offset: wp.int32,
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
):
    """Lumped triangle tangent: ``A_ii += +h²·(μ + λ)·area/3·I`` per vertex.

    Baraff–Witkin lumping: a coarse approximation that is diagonal-PSD by
    construction (no per-element eigendecomposition needed). Skips
    degenerate triangles by writing zero blocks.
    """
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x10 = x1 - x0
    x20 = x2 - x0
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


@wp.kernel
def eval_bending(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Edge dihedral-angle bending force ``[N]``.

    For each shared edge (vertices 3-4 with adjacent triangles 1-3-4 and
    2-3-4) computes the dihedral angle ``θ`` between the two face normals,
    its rate ``θ̇`` from velocities, and applies

        f_a = −e_length · (k_e · (θ − θ_rest) + k_d · θ̇) · ∂θ/∂x_a

    on the four involved vertices. Treats degenerate triangles or edges as
    inactive.
    """
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    n1 = wp.cross(x3 - x1, x4 - x1)
    n2 = wp.cross(x4 - x2, x3 - x2)
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1 = n1 / n1_length
    n2 = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1, n2)
    sin_theta = wp.dot(wp.cross(n1, n2), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    d1 = n1 * e_length
    d2 = n2 * e_length
    d3 = n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)
    d4 = n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)

    f_elastic = ke * (theta - rest_angle)
    f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))

    f_total = -e_length * (f_elastic + f_damp)

    wp.atomic_add(f, i, d1 * f_total)
    wp.atomic_add(f, j, d2 * f_total)
    wp.atomic_add(f, k, d3 * f_total)
    wp.atomic_add(f, l, d4 * f_total)


# =====================================================================
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
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
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
# * Constraint-based (XPBD-style) post-integration position projection
#   with per-particle / per-body delta accumulation; the
#   ``apply_particle_corrections`` kernel applies the deltas with clamps.
# =====================================================================

@wp.kernel
def eval_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    ground: wp.array(dtype=float),
    gravity_world0: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
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
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    particle_radius: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
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


@wp.kernel
def solve_soft_contacts_constraint(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    particle_friction: wp.array(dtype=float),
    particle_ka: float,
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    """XPBD-style constraint solve for soft-rigid contacts.

    Computes per-particle position deltas (and rigid-body spatial deltas)
    that prevent penetration. Preventive: kicks in when ``c < particle_ka``
    rather than waiting for deep penetration like the force-based path.
    Apply with :func:`apply_particle_corrections`.

    Coulomb friction in constraint form: tangential correction magnitude
    is bounded by ``μ · |normal_correction|``.
    """
    tid = wp.tid()

    count = min(contact_max, soft_contact_count[0])
    if tid >= count:
        return

    shape_index = soft_contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = soft_contact_particle[tid]

    if (particle_flags[particle_index] & 1) == 0:  # PARTICLE_FLAG_ACTIVE = 1
        return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    # Body transform (identity if static / ground).
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    bx = wp.transform_point(X_wb, soft_contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = soft_contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    # Preventive engagement: act before deep penetration.
    if c > particle_ka:
        return

    # Use shape friction so per-shape μ from builder is honored.
    mu = shape_material_mu[shape_index]

    # Body velocity at the contact point.
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]
    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, soft_contact_body_vel[tid])

    v = pv - bv

    # Normal correction (capped to 3·radius to avoid catastrophic step).
    max_n = 3.0 * particle_radius[particle_index]
    lambda_n = wp.max(c, -max_n)
    delta_n = n * lambda_n

    vn = wp.dot(n, v)
    vt = v - n * vn

    # Effective inverse mass (particle + lever-arm-projected body inertia).
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return

    # Coulomb cap: |λ_f| ≤ μ · |λ_n|.
    penetration = wp.max(-lambda_n, 0.0)
    friction_cap = mu * penetration
    lambda_f = wp.max(-friction_cap, -wp.length(vt) * dt)
    if wp.length(vt) > 1e-6:
        delta_f = wp.normalize(vt) * lambda_f
    else:
        delta_f = wp.vec3(0.0)

    # Total correction; relax to avoid overshoot.
    delta_total = (delta_f - delta_n) / denom * relaxation

    # Particle correction weighted by inverse mass.
    wp.atomic_add(delta, particle_index, w1 * delta_total)

    # Reaction onto the body (Newton's third law) as a spatial vector.
    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_total, delta_t))


@wp.kernel
def accumulate_body_force_from_constraint_delta(
    body_deltas: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    inv_dt2: float,
    # in/out (one thread per body, no race):
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Convert XPBD-style body position-correction deltas into forces and
    accumulate into the input state's ``body_f``.

    ``body_deltas[b]`` is what :func:`solve_soft_contacts_constraint` writes
    via ``atomic_sub``; each spatial-vector entry has units of ``m·kg``
    (top: linear Lagrange-multiplier × inverse-mass-weighted, bottom:
    angular cross-arm-projected). Dividing by ``dt²`` converts to Newtons /
    Newton-metres. Accumulating across the 48 contact iterations gives the
    integrated reaction force the rigid solver should see.

    Skipped for static / kinematic bodies (``inv_m == 0``) so the kinematic
    stem stays prescribed and the ground stays at infinity.

    Why force, not direct ``body_q`` write: in the dual-solver simulate
    loop the rigid solver (XPBD or MuJoCo) reads ``state_in.body_q`` and
    writes ``state_rigid.body_q``; whatever the inflatable solver writes
    to ``state_out.body_q`` is then *overwritten* by the merge step. So
    the soft-rigid reaction has to flow through ``body_f`` (a force the
    rigid solver consumes during its own step), the same channel the
    glue's :meth:`apply_reaction_to_body` uses.
    """
    bid = wp.tid()
    if body_inv_mass[bid] == 0.0:
        return
    delta = body_deltas[bid]
    f_lin = wp.spatial_top(delta) * inv_dt2
    f_ang = wp.spatial_bottom(delta) * inv_dt2
    cur = body_f[bid]
    cur_lin = wp.spatial_top(cur)
    cur_ang = wp.spatial_bottom(cur)
    body_f[bid] = wp.spatial_vector(cur_lin + f_lin, cur_ang + f_ang)


@wp.kernel
def apply_body_corrections(
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_deltas: wp.array(dtype=wp.spatial_vector),
    dt: float,
    # in/out (one thread per body, no race):
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Apply ``body_deltas`` from :func:`solve_soft_contacts_constraint` to
    each dynamic rigid body's pose and velocity (Newton's 3rd-law reaction
    from soft-rigid contacts).

    ``body_deltas[b]`` is a spatial vector with ``top = -delta_total`` and
    ``bottom = -cross(r, delta_total)`` accumulated atomically per contact.
    Multiplying by ``inv_m`` / applying ``inv_I`` recovers the linear /
    angular position correction; the velocity update is ``correction / dt``
    per the PBD position-to-velocity convention. Skipped for static or
    kinematic bodies (``inv_m == 0``) so the kinematic stem stays prescribed.

    Without this pass, the constraint-contact projection updates only the
    soft particles — the rigid body never feels the equal-and-opposite
    force, and free dynamic bodies (e.g. a ball squeezed by a soft gripper)
    don't move regardless of grip force.
    """
    bid = wp.tid()
    inv_m = body_inv_mass[bid]
    if inv_m == 0.0:
        return
    inv_I = body_inv_inertia[bid]

    delta = body_deltas[bid]
    dlin = wp.spatial_top(delta) * inv_m
    dang_world = wp.spatial_bottom(delta)

    tf = body_q[bid]
    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    # Angular: rotate dang into body frame, apply inv_I, rotate back to world.
    dang_body = wp.quat_rotate_inv(q0, dang_world)
    dw_body = inv_I * dang_body
    dw_world = wp.quat_rotate(q0, dw_body)

    # Position update.
    p1 = p0 + dlin

    # Quaternion update: q_new = normalize(q + 0.5 · (dw·dt, 0) · q).
    dq = wp.quat(dw_world * dt, 0.0)
    q1 = wp.normalize(q0 + 0.5 * dq * q0)

    body_q[bid] = wp.transform(p1, q1)

    # Velocity update: v += dlin / dt, w += dw.
    v0 = wp.spatial_top(body_qd[bid])
    w0 = wp.spatial_bottom(body_qd[bid])
    v1 = v0 + dlin / dt
    w1 = w0 + dw_world
    body_qd[bid] = wp.spatial_vector(v1, w1)


@wp.kernel
def apply_particle_corrections(
    x_current: wp.array(dtype=wp.vec3),
    v_current: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    v_max: float,
    max_correction: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    """Apply XPBD-style position deltas with stability clamps.

    ``x_new = x + clamp(δ, max_correction)``,
    ``v_new = clamp(v + δ/dt, v_max)``.

    The position-clamp prevents single-step explosions on stiff penalties;
    the velocity-clamp keeps post-correction velocity bounded.
    """
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    xp = x_current[tid]
    vp = v_current[tid]
    d = delta[tid]

    d_mag = wp.length(d)
    if d_mag > max_correction and d_mag > 1.0e-9:
        d = d * (max_correction / d_mag)

    x_new = xp + d
    v_new = vp + d / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new = v_new * (v_max / v_new_mag)

    x_out[tid] = x_new
    v_out[tid] = v_new




# =====================================================================
# 9. Gravity
# =====================================================================
#
# Reads gravity from a length-1 device array so there's no host sync inside
# CUDA-graph capture (``model.gravity.numpy()`` would break capture).
# =====================================================================

@wp.kernel
def eval_gravity_from_array(
    gravity: wp.array(dtype=wp.vec3),
    mass: wp.float32,
    particle_flags: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
):
    """Per-particle gravity ``f = mass · g`` ``[N]`` for active particles."""
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    g = gravity[0]
    forces[tid] = wp.vec3(g[0] * mass, g[1] * mass, g[2] * mass)


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
    original_rest_lengths: wp.array(dtype=wp.float32),
    scale: wp.float32,
    scaled_rest_lengths: wp.array(dtype=wp.float32),
):
    """Per-spring scalar scale: ``L₀ ← L₀_orig · cbrt(p)``."""
    sid = wp.tid()
    scaled_rest_lengths[sid] = original_rest_lengths[sid] * scale


@wp.kernel
def scale_tet_poses_kernel(
    original_poses: wp.array(dtype=wp.mat33),
    scale: wp.float32,
    scaled_poses: wp.array(dtype=wp.mat33),
):
    """Per-tet inverse-rest scale: ``Dm ← Dm_orig / cbrt(p)``.

    The model stores ``Dm = rest⁻¹``, so to grow the rest shape by linear
    factor ``s = cbrt(p)``, we divide ``Dm`` by ``s``.
    """
    tid = wp.tid()
    inv_scale = 1.0 / scale
    orig = original_poses[tid]
    scaled_poses[tid] = wp.mat33(
        orig[0, 0] * inv_scale, orig[0, 1] * inv_scale, orig[0, 2] * inv_scale,
        orig[1, 0] * inv_scale, orig[1, 1] * inv_scale, orig[1, 2] * inv_scale,
        orig[2, 0] * inv_scale, orig[2, 1] * inv_scale, orig[2, 2] * inv_scale,
    )


@wp.kernel
def scale_spring_rest_lengths_per_chamber_kernel(
    original_rest_lengths: wp.array(dtype=wp.float32),
    spring_chamber_mask: wp.array(dtype=wp.int32),
    chamber_pressures: wp.array(dtype=wp.float32),
    num_chambers: int,
    scaled_rest_lengths: wp.array(dtype=wp.float32),
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
    original_poses: wp.array(dtype=wp.mat33),
    tet_chamber_mask: wp.array(dtype=wp.int32),
    chamber_pressures: wp.array(dtype=wp.float32),
    num_chambers: int,
    scaled_poses: wp.array(dtype=wp.mat33),
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
        orig[0, 0] * inv_scale, orig[0, 1] * inv_scale, orig[0, 2] * inv_scale,
        orig[1, 0] * inv_scale, orig[1, 1] * inv_scale, orig[1, 2] * inv_scale,
        orig[2, 0] * inv_scale, orig[2, 1] * inv_scale, orig[2, 2] * inv_scale,
    )


@wp.kernel
def compute_volume_kernel(
    positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array2d(dtype=wp.int32),
    tet_volumes: wp.array(dtype=wp.float32),
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
