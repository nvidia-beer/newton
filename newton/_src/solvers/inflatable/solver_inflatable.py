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

"""Inflatable soft-body solver: implicit Backward-Euler FEM with rest-pose pressure control.

Mathematical overview
=====================

Each substep does one Newton iteration of Backward Euler on the soft particles:

    M · (v_{n+1} − v_n) = h · f(x_{n+1}, v_{n+1})           (continuous BE)
    x_{n+1} = x_n + h · v_{n+1}

Linearising ``f`` around the current state ``(x_n, v_n)`` and writing
``Δv = v_{n+1} − v_n`` produces a sparse linear system

    A · Δv = h · f_n
    A = M − h² · (∂f/∂x) − h · (∂f/∂v)

For elastic forces ``f_int = −∇ψ`` (ψ is the strain-energy potential), the
energy Hessian ``H_pot = ∂²ψ/∂x²`` is PSD, so ``∂f_int/∂x = −H_pot``. With
dissipative damping ``f_damp = −C·v`` (``C`` PSD), this gives

    A = M + h² · H_pot + h · C        (PSD plus mass ⇒ PD)

``A`` is rebuilt every substep by ``_build_system_matrix`` from the current
state and solved by ``implicit_integration``. The position update is the
symplectic-Euler form ``x_{n+1} = x_n + h · (v_n + Δv)``.

Stable Neo-Hookean
==================

Tetrahedral elements use the Smith et al. 2018 stable Neo-Hookean model
with the rest correction ``α = 1 + μ/λ − μ/(4·λ)``. The deviatoric piece
is regularised by ``s(I_C) = 1 − 1/(I_C + 1)``; the volumetric piece is
``½·k_λ·(J − α)²``. Per-element Hessians are assembled raw — no PSD
projection — so for stiff materials or large deformations ``A`` can drift
indefinite and require many substeps. Per-element absolute-eigenvalue
filtering (Chen et al. 2024) is the planned next iteration.

Inflation by rest-configuration scaling
=======================================

Pressure ``p`` (a dimensionless volume ratio in ``[1.0, max_volume_ratio]``)
is applied by scaling the per-tet rest pose and per-spring rest length:

    Dm  ←  Dm  / cbrt(p)        (per tet; Dm is the inverse rest-shape matrix)
    L₀  ←  L₀ · cbrt(p)         (per spring)

The deformation gradient ``F = Ds · Dm`` then scales by ``1/cbrt(p)``, so at
new equilibrium (F = I) the body has expanded linearly by ``cbrt(p)``,
i.e. final volume / original volume = ``p``. Per-chamber pressures use the
same scaling restricted to masked tets / springs.

Dirichlet pin (kinematic-glue hook)
===================================

When :meth:`set_dirichlet_pin` is configured, each substep enforces
``Δv[p] = target_dv[p]`` exactly via two phases (Sifakis SIGGRAPH 2012
§3 / Baraff–Witkin SIGGRAPH '98 §5):

1. :meth:`_apply_dirichlet_override` captures the constraint reaction
   ``reaction[p] := f_total[p]/h − m·g − m·target_dv[p]/h`` and overwrites
   ``particle_f[p] := target_dv[p]``. The inertial term is required for
   momentum conservation when the reaction is forwarded to the rigid body
   via ``state.body_f``.

2. :meth:`_apply_dirichlet_filter` row/column-eliminates the pinned DOFs
   in the BSR matrix and Schur-condenses the prescribed motion into the
   unpinned RHS, closing the leak path through stiffness off-diagonals.

File layout
===========

1. ``__init__`` — buffer allocation, validation
2. Public API — :meth:`step`, inflation control, telemetry, Dirichlet pin
3. Internals — system-matrix assembly, linear solve, constraint projection
4. Force evaluators — one method per force kind, called from :meth:`step`
5. Helpers — surface caching, gravity array
"""

import numpy as np
import warp as wp

from warp.optim.linear import bicgstab, cg, cr, gmres, preconditioner
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from newton import ParticleFlags
from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.solver import SolverBase

from .kernels import (
    accumulate_body_force_from_constraint_delta,
    apply_dirichlet_pin_kernel,
    apply_particle_corrections,
    build_system_matrix_diagonal_kernel,
    build_system_matrix_diagonal_mass_kernel,
    build_system_matrix_sparse_kernel,
    build_system_matrix_tet_kernel,
    build_system_matrix_tri_kernel,
    compute_volume_kernel,
    eval_bending,
    eval_gravity_from_array,
    eval_particle_forces,
    eval_particle_ground_contacts,
    eval_soft_contacts,
    eval_springs,
    eval_springs_linear_and_torque,
    eval_tetrahedra,
    eval_triangles,
    filter_dirichlet_pin_in_bsr_kernel,
    scale_spring_rest_lengths_kernel,
    scale_spring_rest_lengths_per_chamber_kernel,
    scale_tet_poses_kernel,
    scale_tet_poses_per_chamber_kernel,
    solve_soft_contacts_constraint,
    update_state,
)

PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


class SolverInflatable(SolverBase):
    """Inflatable soft-body solver with implicit FEM, self-contact, and pressure control.

    Each substep solves ``A · Δv = h · f`` with ``A = M + h²·H_pot + h·C``
    rebuilt from the current state. Inflation is applied by scaling the
    per-element rest configuration (``model.tet_poses``,
    ``model.spring_rest_length``); the FEM forces then drive the body
    toward the new rest state.

    Modes:
        Single pressure: call :meth:`set_pressure` for the whole body.
        Multi-chamber: call :meth:`set_chamber_mask` once and
        :meth:`set_chamber_pressures` to set per-chamber pressures (e.g.
        a bending actuator with two side-by-side chambers at different
        pressures).

    """

    # ==================================================================
    # 1. Construction
    # ==================================================================

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        max_volume_ratio: float = 3.0,
        preconditioner_type: str = "diag",
        solver_type: str = "bicgstab",
        linear_solver_maxiter: int = 50,
        use_constraint_contacts: bool = False,
        contact_relaxation: float = 0.5,
        contact_max_velocity: float = 20.0,
        contact_max_correction: float = 0.02,
        contact_iterations: int = 2,
        ground_plane: "tuple[float, float, float, float] | None" = None,
        ground_ke: float = 1.0e5,
        ground_kd: float = 1.0e2,
        ground_kf: float = 1.0e3,
        ground_mu: float = 0.5,
        torque_stiffness: float = 0.0,
        torque_damping: float = 0.0,
        spring_rest_direction: "np.ndarray | None" = None,
        dirichlet_pin_filter_A: bool = True,
    ):
        super().__init__(model=model)

        # Bisection knob for the BSR row/col elimination at pinned DOFs
        # (Baraff–Witkin). When False, the Dirichlet pin reverts to the
        # legacy RHS-only override (with the inertial-term reaction fix
        # still applied). Off-by-default would erase the #1b stability fix
        # for everyone — leave on unless debugging.
        self._dirichlet_pin_filter_A = bool(dirichlet_pin_filter_A)

        # Uniform per-particle mass used in the implicit matrix and the
        # gravity / Dirichlet-pin force evaluators. Kept as a scalar for now;
        # see the discussion in the inflatable kernels for the path to
        # per-particle physical mass.
        self.mass = float(mass)
        self.preconditioner_type = preconditioner_type
        self.solver_type = solver_type
        self.linear_solver_maxiter = linear_solver_maxiter

        # Constraint-style soft-rigid contact (XPBD-like position projection)
        self.use_constraint_contacts = use_constraint_contacts
        self.contact_relaxation = contact_relaxation
        self.contact_max_velocity = contact_max_velocity
        self.contact_max_correction = contact_max_correction
        self.contact_iterations = contact_iterations

        # Optional analytic ground plane: n·x + d = 0
        self._ground_plane = None
        if ground_plane is not None:
            self._ground_plane = wp.array(ground_plane, dtype=wp.float32, device=model.device)
        self._ground_ke = float(ground_ke)
        self._ground_kd = float(ground_kd)
        self._ground_kf = float(ground_kf)
        self._ground_mu = float(ground_mu)

        # BSR system matrix A. Triplet layout (row, col, mat33f value):
        #   [0 .. P)                 — per-particle diagonal (mass + spring sum)
        #   [P .. P+2·S)             — per-spring (i,j) and (j,i) off-diagonals
        #   [P+2·S .. P+2·S+16·T)    — per-tet 4×4 nodal blocks (FEM tangent)
        #   [... +3·R)               — per-tri 3 lumped diagonal blocks
        # where P = particle_count, S = spring_count, T = tet_count, R = tri_count.
        extra_matrix_blocks = 0
        base_blocks = (
            model.particle_count
            + model.spring_count * 2
            + model.tet_count * 16
            + model.tri_count * 3
        )
        num_blocks = base_blocks + extra_matrix_blocks
        self.bsr_rows = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_cols = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_values = wp.zeros(num_blocks, dtype=wp.mat33f, device=model.device)
        self.A_bsr = bsr_zeros(
            rows_of_blocks=model.particle_count,
            cols_of_blocks=model.particle_count,
            block_type=wp.mat33f,
            device=model.device,
        )
        # Initial assembly with state=None: produces a mass+spring matrix
        # (FEM tangent skipped). The first call to step() rebuilds with the
        # actual state.
        self._build_system_matrix(model, None, dt)
        self.M_bsr = preconditioner(self.A_bsr, ptype=preconditioner_type)
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)

        # Dirichlet pin buffers (configured by set_dirichlet_pin)
        self._dirichlet_mask: "wp.array | None" = None
        self._dirichlet_target_dv: "wp.array | None" = None
        self._dirichlet_reaction: "wp.array | None" = None
        self._dirichlet_gravity: wp.vec3 = wp.vec3(0.0, 0.0, 0.0)

        # Inflation: original rest config is cached so set_pressure scales
        # from a stable baseline rather than compounding per call.
        self.max_volume_ratio = max_volume_ratio
        self.current_pressure = 1.0
        self.target_pressure = 1.0
        self.tet_chamber_mask = None
        self.spring_chamber_mask = None
        self.num_chambers = 0
        self._chamber_pressures_array = None

        if model.tet_count > 0:
            self.original_tet_poses = wp.clone(model.tet_poses)
            self.tet_volumes = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device)
        else:
            self.original_tet_poses = None
            self.tet_volumes = None
        if model.spring_count > 0:
            self.original_spring_rest_length = wp.clone(model.spring_rest_length)
        else:
            self.original_spring_rest_length = None
        self._initial_volume = None

        # Optional bend torque for springs with a non-zero rest direction
        # (used by bending actuators built on top of spring networks).
        self.torque_stiffness = float(torque_stiffness)
        self.torque_damping = float(torque_damping)
        dr = np.asarray(
            spring_rest_direction
            if spring_rest_direction is not None
            else np.zeros((model.spring_count, 3), dtype=np.float32),
            dtype=np.float32,
        )
        if dr.shape != (model.spring_count, 3):
            raise ValueError(
                f"spring_rest_direction shape {dr.shape} != (spring_count={model.spring_count}, 3)"
            )
        self.spring_rest_direction = wp.array(dr, dtype=wp.vec3, device=model.device)

    # ==================================================================
    # 2. Public API: per-substep entry point
    # ==================================================================

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        """Advance one implicit substep.

        Order of operations:

        1. Rebuild ``A`` and the diagonal preconditioner when a state-
           dependent tangent exists (tets, tris) or when the Dirichlet
           pin is active (the pin filter mutates ``A`` in place, so it
           must be regenerated from scratch each substep).
        3. Evaluate every force kind into ``state_in.particle_f``
           (multiplied by ``dt`` so the linear system is ``A · Δv = particle_f``).
        4. Apply the Dirichlet pin (no-op if not configured): capture the
           reaction force, override the RHS at pinned rows, then row/column-
           eliminate the pinned DOFs in ``A`` and Schur-condense the
           prescribed motion into the unpinned RHS. The preconditioner is
           not rebuilt here — using the slightly stale Jacobi values at
           pinned rows costs at most a few extra CG iterations and keeps
           the substep cost flat.
        5. Solve ``A · Δv = particle_f`` and integrate
           ``v_{n+1} = v_n + Δv``, ``x_{n+1} = x_n + dt · v_{n+1}``.
        6. Optional XPBD-style position projection on rigid contacts
           (``use_constraint_contacts``).

        Rigid bodies in ``model`` (when present) are not touched: a separate
        rigid solver handles them, coupled via :class:`GlueAttachments`.
        """
        model = self.model
        if control is None:
            control = model.control()

        # Cache initial volume on first step (for get_volume_ratio etc.)
        if self._initial_volume is None:
            self._initial_volume = self.compute_volume(state_in)

        # 1. Rebuild A whenever a state-dependent tangent is present, or when
        # the Dirichlet pin is active (the pin filter mutates A in place, so
        # the matrix must be regenerated from scratch each substep).
        pin_active = self._dirichlet_mask is not None
        rebuild_A = bool(model.tet_count or model.tri_count or pin_active)
        if rebuild_A:
            self._build_system_matrix(model, state_in, dt)
            self.M_bsr = preconditioner(self.A_bsr, ptype=self.preconditioner_type)

        # 2. Evaluate every force kind into per-particle vec3 buffers.
        spring_forces = self.eval_spring_forces(model, state_in)
        triangle_forces = self.eval_triangle_forces(model, control, state_in)
        bending_forces = self.eval_bending_forces(model, control, state_in)
        tetrahedral_forces = self.eval_tetrahedral_forces(model, control, state_in)
        particle_forces = self.eval_particle_particle_forces(model, control, state_in)
        particle_ground_contact_forces = self.eval_particle_ground_contact_forces(model, control, state_in)
        soft_contact_forces = self.eval_soft_contact_forces(model, state_in, contacts)
        gravity_forces = self.eval_gravity_forces(model)

        # When using constraint-style contacts (XPBD position projection),
        # zero the force-based soft contact term to avoid double-counting.
        if self.use_constraint_contacts:
            soft_contact_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)

        # particle_f = h · sum(forces) is the RHS of A · Δv = particle_f.
        state_in.particle_f = dt * (
            spring_forces
            + tetrahedral_forces
            + triangle_forces
            + bending_forces
            + particle_forces
            + particle_ground_contact_forces
            + soft_contact_forces
            + gravity_forces
        )

        # 4. Dirichlet pin: capture reaction, override RHS at pinned rows,
        # then row/col-eliminate ``A`` and Schur-condense the prescribed
        # motion into the unpinned RHS. The diagonal preconditioner built
        # above is used as-is — for pinned rows it is stale (inverts the
        # pre-filter ``m + h²·k_pp`` rather than ``1.0``), which gives a
        # mildly worse rate on those rows but does not affect correctness:
        # the linear solve recovers ``Δv[p] = target_dv[p]`` exactly because
        # ``A[p, p] = I`` and ``rhs[p] = target_dv[p]`` after the filter.
        self._apply_dirichlet_override(model, state_in, dt)
        self._apply_dirichlet_filter(model, state_in)

        # 5. Linear solve + integrate.
        self.implicit_integration(model, state_in, state_out, dt)

        # 6. Optional XPBD-style position projection on rigid contacts.
        if self.use_constraint_contacts and contacts is not None:
            self.apply_constraint_contact_corrections(model, state_in, state_out, contacts, dt)

        # Rigid-body integration is intentionally NOT done here. The inflatable
        # solver only owns the soft particles. Rigid bodies (when present) are
        # either driven by a separate rigid solver (XPBD / MuJoCo / Featherstone)
        # — coupled via :class:`GlueAttachments` — or held kinematic
        # by the caller. Double-integrating them here would corrupt the rigid
        # solver's state in the dual-solver case and apply spurious gravity in
        # the kinematic case.
        return state_out

    # ==================================================================
    # 3. Public API: inflation control
    # ==================================================================
    #
    # Pressure ``p`` is a unitless volume ratio target. The rest configuration
    # is scaled so the FEM equilibrium is at volume ``p × initial_volume``:
    #
    #     Dm  ←  Dm  / cbrt(p)        ⇒ each tet's rest size grows by cbrt(p)
    #     L₀  ←  L₀ · cbrt(p)         ⇒ each spring's rest length grows by cbrt(p)
    #
    # Linear scale cbrt(p) → volumetric scale p (in the bulk; surface
    # corrections are handled by the contact and self-contact terms).
    # ==================================================================

    def set_pressure(self, pressure: float):
        """Set inflation volume ratio for the whole body (single-pressure mode).

        Forwards to :meth:`set_chamber_pressures` if a chamber mask has been
        configured. Pressure is clamped to ``[1.0, max_volume_ratio]``.
        """
        if self.tet_chamber_mask is not None and self.num_chambers > 0:
            self.set_chamber_pressures([pressure] * self.num_chambers)
            self.current_pressure = pressure
            return
        model = self.model
        pressure = float(np.clip(pressure, 1.0, self.max_volume_ratio))
        if abs(pressure - self.current_pressure) < 1e-6:
            return
        self.current_pressure = pressure
        linear_scale = float(np.cbrt(pressure))
        if model.spring_count > 0 and self.original_spring_rest_length is not None:
            wp.launch(
                kernel=scale_spring_rest_lengths_kernel,
                dim=model.spring_count,
                inputs=[self.original_spring_rest_length, linear_scale],
                outputs=[model.spring_rest_length],
                device=model.device,
            )
        if model.tet_count > 0 and self.original_tet_poses is not None:
            wp.launch(
                kernel=scale_tet_poses_kernel,
                dim=model.tet_count,
                inputs=[self.original_tet_poses, linear_scale],
                outputs=[model.tet_poses],
                device=model.device,
            )

    def set_target_pressure(self, target: float):
        """Store a target pressure used by :meth:`apply_pressure_control`.

        Clamped to ``[1.0, max_volume_ratio]``. Does not change the current
        pressure on its own — call :meth:`apply_pressure_control` per frame
        to chase the target.
        """
        self.target_pressure = float(np.clip(target, 1.0, self.max_volume_ratio))

    def apply_pressure_control(self, rate: float = 0.1):
        """Step current pressure toward :attr:`target_pressure` at ``rate`` per call.

        Linear interpolation: ``p ← p + rate · (target − p)``. Used by
        examples that expose a slider — keeps the per-substep change in the
        rest configuration small enough that the FEM linearisation stays
        accurate.
        """
        new_pressure = self.current_pressure + rate * (self.target_pressure - self.current_pressure)
        self.set_pressure(new_pressure)

    def set_chamber_mask(
        self,
        tet_chamber_mask: "wp.array",
        spring_chamber_mask: "wp.array | None" = None,
        num_chambers: int = 0,
    ):
        """Mark spatially separate inflation regions for per-chamber control.

        Each entry of ``tet_chamber_mask`` is the chamber id for that tet
        (``-1`` to exclude). Same for ``spring_chamber_mask``. After this
        call, use :meth:`set_chamber_pressures` to set each chamber's
        pressure independently. Used for bending actuators with two
        side-by-side chambers at different pressures.
        """
        self.tet_chamber_mask = tet_chamber_mask
        self.spring_chamber_mask = spring_chamber_mask
        self.num_chambers = int(num_chambers)
        if self.num_chambers > 0:
            self._chamber_pressures_array = wp.array(
                np.full(self.num_chambers, 1.0, dtype=np.float32),
                dtype=wp.float32,
                device=self.model.device,
            )

    def set_chamber_pressures(self, pressure_list: list[float]):
        """Set per-chamber pressure (clamped to ``[1.0, max_volume_ratio]``).

        Each chamber's tets get rest-pose scaling ``cbrt(p_chamber)``; springs
        with a chamber assignment do the same. Springs without a chamber
        assignment are scaled by ``cbrt(mean(pressures))``.
        """
        if self.tet_chamber_mask is None or self.num_chambers <= 0:
            return
        model = self.model
        pressures = np.array(
            [float(np.clip(p, 1.0, self.max_volume_ratio)) for p in pressure_list],
            dtype=np.float32,
        )
        if len(pressures) < self.num_chambers:
            pressures = np.resize(pressures, self.num_chambers)
            pressures[len(pressure_list):] = 1.0
        pressures = pressures[: self.num_chambers]
        self._chamber_pressures_array.assign(pressures)
        if model.spring_count > 0 and self.original_spring_rest_length is not None:
            if self.spring_chamber_mask is not None:
                wp.launch(
                    kernel=scale_spring_rest_lengths_per_chamber_kernel,
                    dim=model.spring_count,
                    inputs=[
                        self.original_spring_rest_length,
                        self.spring_chamber_mask,
                        self._chamber_pressures_array,
                        self.num_chambers,
                    ],
                    outputs=[model.spring_rest_length],
                    device=model.device,
                )
            else:
                # No spring-side mask: use the mean chamber pressure.
                avg_p = float(np.cbrt(np.mean(pressures)))
                wp.launch(
                    kernel=scale_spring_rest_lengths_kernel,
                    dim=model.spring_count,
                    inputs=[self.original_spring_rest_length, avg_p],
                    outputs=[model.spring_rest_length],
                    device=model.device,
                )
        if model.tet_count > 0 and self.original_tet_poses is not None:
            wp.launch(
                kernel=scale_tet_poses_per_chamber_kernel,
                dim=model.tet_count,
                inputs=[
                    self.original_tet_poses,
                    self.tet_chamber_mask,
                    self._chamber_pressures_array,
                    self.num_chambers,
                ],
                outputs=[model.tet_poses],
                device=model.device,
            )

    # ==================================================================
    # 4. Public API: inflation telemetry
    # ==================================================================

    def compute_volume(self, state: State) -> float:
        """Sum of per-tet volumes from current particle positions ``[m³]``.

        Computes ``V_e = ⅙ · |det(Ds)|`` for each tetrahedron and reduces on
        the host. Returns ``0.0`` for tet-less models.
        """
        model = self.model
        if model.tet_count == 0 or self.tet_volumes is None:
            return 0.0
        wp.launch(
            kernel=compute_volume_kernel,
            dim=model.tet_count,
            inputs=[state.particle_q, model.tet_indices],
            outputs=[self.tet_volumes],
            device=model.device,
        )
        return float(np.sum(self.tet_volumes.numpy()))

    def get_initial_volume(self, state: State) -> float:
        """Initial volume (cached on first sensible call) ``[m³]``.

        Returns ``1.0`` as a fallback while the cached value is unset and the
        first sample is non-finite or zero — keeps :meth:`get_volume_ratio`
        sane during model warm-up.
        """
        if self._initial_volume is None:
            vol = self.compute_volume(state)
            if np.isfinite(vol) and vol > 1.0e-12:
                self._initial_volume = vol
            else:
                return 1.0
        return self._initial_volume

    def get_volume_ratio(self, state: State) -> float:
        """current_volume / initial_volume.

        Clamped to ``[0, max(10, 2·max_volume_ratio)]`` to defend against
        degenerate states where a denormal initial volume would otherwise
        produce a ratio of ``1e18``.
        """
        initial = self.get_initial_volume(state)
        if initial <= 1.0e-12:
            return 1.0
        current = self.compute_volume(state)
        if not np.isfinite(current) or current <= 0.0:
            return 1.0
        ratio = current / initial
        if not np.isfinite(ratio):
            return 1.0
        return float(np.clip(ratio, 0.0, max(10.0, self.max_volume_ratio * 2.0)))

    def get_inflation_info(self, state: State) -> dict:
        """Dictionary of inflation state for telemetry / UI overlays.

        Keys:
            initial_volume, current_volume, max_volume — volumes ``[m³]``;
            current_ratio — ``current_volume / initial_volume``;
            target_ratio — :attr:`target_pressure`;
            max_ratio — :attr:`max_volume_ratio`;
            pressure — current applied pressure.
        """
        initial = self.get_initial_volume(state)
        current = self.compute_volume(state)
        if not np.isfinite(current):
            current_ratio = float("nan")
        elif initial > 1.0e-12 and np.isfinite(initial):
            current_ratio = float(current / initial)
            if not np.isfinite(current_ratio):
                current_ratio = float("nan")
        else:
            current_ratio = 1.0
        return {
            "initial_volume": initial,
            "current_volume": current,
            "max_volume": initial * self.max_volume_ratio,
            "current_ratio": current_ratio,
            "target_ratio": self.target_pressure,
            "max_ratio": self.max_volume_ratio,
            "pressure": self.current_pressure,
        }

    # ==================================================================
    # 5. Public API: Dirichlet pin (kinematic-glue hook)
    # ==================================================================
    #
    # The pin lets an external solver-agnostic glue
    # (GlueAttachments) drive a subset of soft particles to follow
    # a rigid body. The override is purely on the implicit RHS:
    #
    #     particle_f[p]  :=  mass · target_dv[p]            for mask[p] == 1
    #     reaction[p]    :=  particle_f[p] / dt − mass · g  (non-gravity reaction, captured before override)
    #
    # The reaction is forwarded by the glue onto the rigid body's body_f.
    # ==================================================================

    def set_dirichlet_pin(
        self,
        mask: "wp.array | None",
        target_dv: "wp.array | None",
        reaction: "wp.array | None",
    ) -> None:
        """Configure a Dirichlet boundary condition on a subset of particles.

        Args:
            mask: Per-particle ``int32`` flag, ``1`` for pinned and ``0``
                for free, shape ``[particle_count]``. Caller owns the buffer.
            target_dv: Per-particle target velocity change ``[m/s]``,
                shape ``[particle_count]``, ``vec3``. Caller refreshes each
                substep before calling :meth:`step`.
            reaction: Output buffer for the per-particle reaction force
                ``[N]``, shape ``[particle_count]``, ``vec3``. Solver writes
                each substep; entries for free particles are zeroed.

        Pass ``None`` for all three arguments to disable the pin. Snapshots
        ``model.gravity`` once into a Python ``vec3`` — call again or
        :meth:`refresh_dirichlet_gravity` after changing gravity, always
        outside any captured CUDA-graph region.
        """
        if mask is None and target_dv is None and reaction is None:
            self._dirichlet_mask = None
            self._dirichlet_target_dv = None
            self._dirichlet_reaction = None
            return
        if mask is None or target_dv is None or reaction is None:
            raise ValueError(
                "set_dirichlet_pin: pass all three of (mask, target_dv, reaction) "
                "to enable, or all three as None to disable."
            )
        n = self.model.particle_count
        for name, arr in (("mask", mask), ("target_dv", target_dv), ("reaction", reaction)):
            if arr.shape[0] != n:
                raise ValueError(
                    f"set_dirichlet_pin: {name} length {arr.shape[0]} != particle_count {n}"
                )
        self._dirichlet_mask = mask
        self._dirichlet_target_dv = target_dv
        self._dirichlet_reaction = reaction
        self.refresh_dirichlet_gravity()

    def refresh_dirichlet_gravity(self) -> None:
        """Re-snapshot ``model.gravity`` into a Python ``vec3``.

        Triggers a device→host memcpy when ``model.gravity`` is a
        ``wp.array``, so it must be called outside any captured CUDA-graph
        region. Re-call after changing gravity at runtime.
        """
        g = self.model.gravity
        if isinstance(g, wp.array):
            arr = g.numpy().reshape(-1, 3)[0]
            self._dirichlet_gravity = wp.vec3(float(arr[0]), float(arr[1]), float(arr[2]))
        elif g is None:
            self._dirichlet_gravity = wp.vec3(0.0, 0.0, 0.0)
        else:
            self._dirichlet_gravity = wp.vec3(float(g[0]), float(g[1]), float(g[2]))

    # ==================================================================
    # 6. Internals: implicit integration
    # ==================================================================
    #
    # Solve  A · Δv = h · f_n   then   v_{n+1} = v_n + Δv,  x_{n+1} = x_n + h · v_{n+1}
    # Iterative linear solver (CG / BiCGStab / GMRES / CR), preconditioner
    # rebuilt every substep when A changes.
    # ==================================================================

    def implicit_integration(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ):
        """Solve ``A · Δv = particle_f`` and integrate state.

        ``state_in.particle_f`` has been pre-multiplied by ``dt`` by
        :meth:`step`, so the linear solve already produces ``Δv`` (not
        ``Δv / dt``). The Dirichlet pin override and BSR row/column
        elimination have already been applied by :meth:`step` before this
        method is called, so the linear system here is the constrained one.
        """
        # Iterative linear solve. ``check_every=0`` keeps convergence checks
        # device-side (no host sync). ``use_cuda_graph=False`` is required
        # because examples wrap step() in ``wp.ScopedCapture`` and Warp's
        # internal graph capture would conflict.
        maxiter = self.linear_solver_maxiter
        if self.solver_type == "cg":
            cg(self.A_bsr, state_in.particle_f, self.dv, tol=1e-2, maxiter=maxiter,
               M=self.M_bsr, check_every=0, use_cuda_graph=False)
        elif self.solver_type == "bicgstab":
            bicgstab(self.A_bsr, state_in.particle_f, self.dv, tol=1e-2, maxiter=maxiter,
                     M=self.M_bsr, check_every=0, use_cuda_graph=False)
        elif self.solver_type == "gmres":
            gmres(self.A_bsr, state_in.particle_f, self.dv, tol=1e-2, maxiter=maxiter,
                  M=self.M_bsr, check_every=0, use_cuda_graph=False)
        elif self.solver_type == "cr":
            cr(self.A_bsr, state_in.particle_f, self.dv, tol=1e-2, maxiter=maxiter,
               M=self.M_bsr, check_every=0, use_cuda_graph=False)
        else:
            raise ValueError(f"Invalid solver type: {self.solver_type}")

        # x_{n+1} = x_n + dt · (v_n + Δv).
        wp.launch(
            kernel=update_state,
            dim=model.particle_count,
            inputs=[
                self.dv, dt,
                state_in.particle_q, state_in.particle_qd,
                state_out.particle_q, state_out.particle_qd,
            ],
            device=model.device,
        )

    def _build_system_matrix(self, model: Model, state: State | None, dt: float):
        """Assemble ``A = M + h²·H_pot + h·C`` from triplet writers.

        Each per-element kernel writes its blocks into a region of the
        triplet arrays at a known offset; ``bsr_set_from_triplets`` then
        sums duplicate (row, col) entries into the BSR matrix. Tet and tri
        FEM tangents need the current state; on construction (``state=None``)
        they are skipped and only mass + spring blocks are populated.
        """
        offset = 0

        # Diagonal block (per particle): M·I plus the spring sum
        # (h·d + h²·k)·I lumped per attached spring. When there are no
        # springs the simpler mass-only kernel writes just M·I.
        if model.spring_count > 0:
            wp.launch(
                kernel=build_system_matrix_diagonal_kernel,
                dim=model.particle_count,
                inputs=[
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    model.spring_indices,
                    model.spring_stiffness,
                    model.spring_damping,
                    wp.float32(dt),
                    wp.float32(self.mass),
                    wp.int32(model.spring_count),
                ],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=build_system_matrix_diagonal_mass_kernel,
                dim=model.particle_count,
                inputs=[self.bsr_rows, self.bsr_cols, self.bsr_values, wp.float32(self.mass)],
                device=model.device,
            )
        offset += model.particle_count

        # Spring off-diagonals: A_ij = A_ji = -(h·d + h²·k)·I (graph-Laplacian
        # form — together with the diagonal contribution above this gives a
        # PSD spring sub-block whose null space contains rigid translations).
        # The damping term must be subtracted off-diagonal too; otherwise it
        # acts like a Dirichlet-to-ground velocity drag and resists free-fall.
        if model.spring_count > 0:
            wp.launch(
                kernel=build_system_matrix_sparse_kernel,
                dim=model.spring_count,
                inputs=[
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    model.spring_indices,
                    model.spring_stiffness,
                    model.spring_damping,
                    wp.float32(dt),
                    wp.int32(offset),
                ],
                device=model.device,
            )
            offset += model.spring_count * 2

        # Tetrahedral FEM tangent: 4×4 nodal blocks per element of the
        # 12×12 element Hessian +h²·∂²ψ/∂x² (Stable Neo-Hookean).
        if model.tet_count > 0 and state is not None:
            wp.launch(
                kernel=build_system_matrix_tet_kernel,
                dim=model.tet_count,
                inputs=[
                    state.particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(dt),
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                ],
                device=model.device,
            )
            offset += model.tet_count * 16

        # Triangle FEM lumped tangent: per-triangle, write +h²·(k_μ+k_λ)·area/3
        # to each of the three diagonal blocks (Baraff-Witkin lumping).
        if model.tri_count > 0 and state is not None:
            wp.launch(
                kernel=build_system_matrix_tri_kernel,
                dim=model.tri_count,
                inputs=[
                    state.particle_q,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_materials,
                    wp.float32(dt),
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                ],
                device=model.device,
            )
            offset += model.tri_count * 3

        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True,
        )

    def _apply_dirichlet_override(self, model: Model, state_in: State, dt: float) -> None:
        """Capture the pin reaction and override the RHS at pinned particles.

        For each pinned ``p``: writes the constraint reaction force (elastic
        + contact, minus gravity, minus the inertial reaction
        ``m · target_dv / dt``) into ``reaction``, then overwrites
        ``particle_f[p] := target_dv[p]`` so that the post-filter system
        ``A[p, p] = I`` yields ``Δv[p] = target_dv[p]`` exactly.

        Pairs with :meth:`_apply_dirichlet_filter`, which mutates ``A`` in
        place — that is what makes this a hard pin rather than the soft pin
        the previous RHS-only implementation provided. No-op when the pin is
        not configured.
        """
        if self._dirichlet_mask is None:
            return
        target_scale = 1.0 if self._dirichlet_pin_filter_A else self.mass
        wp.launch(
            kernel=apply_dirichlet_pin_kernel,
            dim=model.particle_count,
            inputs=[
                self._dirichlet_mask,
                self._dirichlet_target_dv,
                self._dirichlet_gravity,
                wp.float32(self.mass),
                wp.float32(target_scale),
                float(dt),
                state_in.particle_f,
                self._dirichlet_reaction,
            ],
            device=model.device,
        )

    def _apply_dirichlet_filter(self, model: Model, state_in: State) -> None:
        """Row/column-eliminate pinned DOFs in ``A`` and Schur-condense the RHS.

        For each pinned particle ``p``: zeros row ``p`` and column ``p`` of
        the assembled BSR matrix, sets ``A[p, p] = I``, and accumulates
        ``-A[r, p] · target_dv[p]`` into ``particle_f[r]`` at every unpinned
        neighbour ``r`` (Schur condensation of the prescribed motion).

        Together with :meth:`_apply_dirichlet_override` this is the standard
        treatment of essential boundary conditions in implicit FEM (Sifakis
        SIGGRAPH 2012 §3 / Baraff–Witkin SIGGRAPH '98 §5). It removes the
        leak path through which neighbour off-diagonals biased the achieved
        ``Δv[p]`` away from ``target_dv[p]`` in the previous RHS-only pin.

        Mutates ``A_bsr.values`` in place — :meth:`step` rebuilds ``A`` from
        scratch each substep when the pin is active so the modifications do
        not accumulate. No-op when the pin is not configured or when the
        constructor flag ``dirichlet_pin_filter_A`` is False (bisection knob).
        """
        if self._dirichlet_mask is None or not self._dirichlet_pin_filter_A:
            return
        wp.launch(
            kernel=filter_dirichlet_pin_in_bsr_kernel,
            dim=model.particle_count,
            inputs=[
                self._dirichlet_mask,
                self._dirichlet_target_dv,
                self.A_bsr.offsets,
                self.A_bsr.columns,
                self.A_bsr.values,
                state_in.particle_f,
            ],
            device=model.device,
        )

    def apply_constraint_contact_corrections(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        dt: float,
    ):
        """XPBD-style position projection on rigid contacts (post-integration).

        Iterates ``contact_iterations`` times, each iteration running a
        Gauss-Seidel sweep over ``contacts.soft_contact_*`` and clamping the
        per-particle delta and resulting velocity to keep stiff contacts
        stable. Used when ``use_constraint_contacts=True`` to prevent
        penetration that the force-based contact alone cannot.
        """
        if not hasattr(contacts, "soft_contact_count"):
            return

        # Lazily allocate the per-particle friction array.
        particle_friction = getattr(model, "particle_friction", None)
        if (
            particle_friction is None
            or not hasattr(particle_friction, "shape")
            or particle_friction.shape[0] != model.particle_count
        ):
            particle_friction = wp.full(
                (model.particle_count,),
                float(model.soft_contact_mu),
                dtype=wp.float32,
                device=model.device,
            )
            model.particle_friction = particle_friction

        particle_deltas = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        body_deltas = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        for _ in range(self.contact_iterations):
            particle_deltas.zero_()
            body_deltas.zero_()
            wp.launch(
                kernel=solve_soft_contacts_constraint,
                dim=contacts.soft_contact_max,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    model.particle_inv_mass,
                    model.particle_radius,
                    model.particle_flags,
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.shape_body,
                    model.shape_material_mu,
                    particle_friction,
                    model.particle_adhesion,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_max,
                    dt,
                    self.contact_relaxation,
                ],
                outputs=[particle_deltas, body_deltas],
                device=model.device,
            )
            wp.launch(
                kernel=apply_particle_corrections,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    particle_deltas,
                    model.particle_flags,
                    dt,
                    self.contact_max_velocity,
                    self.contact_max_correction,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )
            # Newton's-3rd-law reaction onto the rigid body. The constraint
            # kernel atomically writes ``body_deltas`` (m·kg, position-shape
            # XPBD-style) per contact pair; we convert per-iteration to a
            # force ``= delta / dt²`` and accumulate into ``state_in.body_f``
            # so the *downstream* rigid solver (XPBD or MuJoCo) integrates
            # the ball under the soft-contact reaction. ``state_in`` is the
            # state the rigid solver reads (``state_0`` in the dual-solver
            # simulate loop); writing to ``state_out.body_f`` would be
            # discarded by the merge that takes ``body_q`` from the rigid
            # solver's output. Without this, free dynamic bodies see no
            # reaction and stay still no matter how hard the gripper grips.
            wp.launch(
                kernel=accumulate_body_force_from_constraint_delta,
                dim=model.body_count,
                inputs=[
                    body_deltas,
                    model.body_inv_mass,
                    wp.float32(1.0 / (float(dt) * float(dt))),
                ],
                outputs=[state_in.body_f],
                device=model.device,
            )

    # ==================================================================
    # 7. Force evaluators (called from step())
    # ==================================================================
    #
    # Each evaluator returns a per-particle ``vec3`` of forces ``[N]``. The
    # sign convention is the physical one: ``f`` points in the direction the
    # particle should be pushed. ``step`` sums them, multiplies by ``dt``, and
    # stores in ``state_in.particle_f`` as the BE RHS.
    # ==================================================================

    def eval_tetrahedral_forces(self, model: Model, control: Control, state: State):
        """Tet FEM elastic force from Stable Neo-Hookean ``[N]``.

        Computes ``f = −V · ∂ψ/∂x`` per node where
        ``ψ = ½·k_μ·(I_C − 3 − 2·log(...)) + ½·k_λ·(J − α)²`` (deviatoric +
        volumetric, with the rest correction ``α = 1 + μ/λ − μ/(4·λ)``).
        Activation channels modulate the volumetric term per element.
        """
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.tet_count:
            wp.launch(
                kernel=eval_tetrahedra,
                dim=model.tet_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.tet_indices,
                    model.tet_poses,
                    control.tet_activations if hasattr(control, "tet_activations") else model.tet_activations,
                    model.tet_materials,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_triangle_forces(self, model: Model, control: Control, state: State):
        """Tri membrane (in-plane stretch + area-preservation) force ``[N]``.

        Activation channels modulate the area-preservation target.
        """
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.tri_count:
            wp.launch(
                kernel=eval_triangles,
                dim=model.tri_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.tri_indices,
                    model.tri_poses,
                    control.tri_activations if hasattr(control, "tri_activations") else model.tri_activations,
                    model.tri_materials,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_bending_forces(self, model: Model, control: Control, state: State):
        """Edge-bending force ``[N]`` on each shared-edge dihedral.

        Pulls the dihedral angle toward ``model.edge_rest_angle`` via
        ``f_elastic = ke · (θ − θ_rest)`` and dissipates with ``kd``.
        """
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.edge_count:
            wp.launch(
                kernel=eval_bending,
                dim=model.edge_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.edge_indices,
                    model.edge_rest_angle,
                    model.edge_bending_properties,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_spring_forces(self, model: Model, state: State):
        """Linear spring force ``[N]`` (with optional bend torque).

        Linear: ``f = −k · (l − L₀) · dir − k_d · ((v_i − v_j) · dir) · dir``.
        When ``torque_stiffness > 0`` and any spring has a non-zero rest
        direction in :attr:`spring_rest_direction`, an additional torque-as-
        force term opposes deviation from that rest direction.
        """
        if model.spring_count == 0:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if self.torque_stiffness > 0.0:
            wp.launch(
                kernel=eval_springs_linear_and_torque,
                dim=model.spring_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.spring_indices,
                    model.spring_rest_length,
                    model.spring_stiffness,
                    model.spring_damping,
                    self.spring_rest_direction,
                    wp.float32(self.torque_stiffness),
                    wp.float32(self.torque_damping),
                ],
                outputs=[forces],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=eval_springs,
                dim=model.spring_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    model.spring_indices,
                    model.spring_rest_length,
                    model.spring_stiffness,
                    model.spring_damping,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_particle_particle_forces(self, model: Model, control: Control, state: State):
        """Hash-grid particle-particle penalty + Coulomb friction ``[N]``.

        Used by granular / fluid-like particle ensembles when
        ``model.particle_grid`` is populated. The hash grid is rebuilt here
        from ``state.particle_q`` so the query is valid regardless of which
        solver last touched it — :class:`SolverXPBD` reserves the grid in
        its constructor (allocates buffers but does not build a spatial
        index), and querying that reserved-but-unbuilt grid causes an
        illegal-memory-access on the device. Building locally each call
        also keeps the result consistent with the soft solver's current
        positions rather than the rigid solver's.
        """
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.particle_count > 1 and model.particle_max_radius > 0.0 and model.particle_grid is not None:
            search_radius = model.particle_max_radius * 2.0 + model.particle_cohesion
            with wp.ScopedDevice(model.device):
                model.particle_grid.build(state.particle_q, radius=search_radius)
            wp.launch(
                kernel=eval_particle_forces,
                dim=model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    state.particle_q,
                    state.particle_qd,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_ke,
                    model.particle_kd,
                    model.particle_kf,
                    model.particle_mu,
                    model.particle_cohesion,
                    model.particle_max_radius,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_particle_ground_contact_forces(self, model: Model, control: Control, state: State):
        """Particle vs analytic ground plane (penalty + Coulomb friction) ``[N]``.

        Active only when ``ground_plane=(nx, ny, nz, d)`` was passed at
        construction; otherwise returns zero (use the collision pipeline +
        ``use_constraint_contacts=True`` for non-planar ground).
        """
        if self._ground_plane is None:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=eval_particle_ground_contacts,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_inv_mass,
                model.particle_flags,
                self._ground_ke,
                self._ground_kd,
                self._ground_kf,
                self._ground_mu,
                self._ground_plane,
                self._get_gravity_array(model),
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def eval_soft_contact_forces(self, model: Model, state: State, contacts: Contacts):
        """Force-based soft-rigid contact from ``model.collide(state)`` output ``[N]``.

        Penalty + damping + Coulomb friction. Zeroed by :meth:`step` when
        ``use_constraint_contacts=True`` to avoid double-counting with the
        XPBD-style projection.
        """
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if contacts is None or not hasattr(contacts, "soft_contact_count"):
            return forces
        wp.launch(
            kernel=eval_soft_contacts,
            dim=contacts.soft_contact_max,
            inputs=[
                state.particle_q,
                state.particle_qd,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_kf,
                model.soft_contact_mu,
                model.particle_radius,
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def eval_gravity_forces(self, model: Model):
        """Per-particle gravity ``f = m · g`` ``[N]``.

        Reads ``model.gravity`` from a device array (graph-capture safe; no
        host sync). Zeroed for particles that don't have ``ParticleFlags.ACTIVE``.
        """
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.particle_count:
            gravity_arr = self._get_gravity_array(model)
            wp.launch(
                kernel=eval_gravity_from_array,
                dim=model.particle_count,
                inputs=[gravity_arr, wp.float32(self.mass), model.particle_flags],
                outputs=[forces],
                device=model.device,
            )
        return forces

    # ==================================================================
    # 8. Internal helpers
    # ==================================================================

    def _get_gravity_array(self, model: Model) -> wp.array:
        """Return ``model.gravity`` as a ``wp.array(dtype=wp.vec3)``.

        Wraps a Python ``vec3`` in a length-1 array when needed; otherwise
        returns the existing array unchanged. Used by force evaluators that
        want graph-capture-safe access to gravity.
        """
        g = model.gravity
        if isinstance(g, wp.array):
            return g
        return wp.array([g], dtype=wp.vec3, device=model.device)
