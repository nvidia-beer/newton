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

"""Fully-implicit Backward-Euler soft-body solver base.

Handles tetrahedral and/or hexahedral FEM elements based on what is present in
the model (``model.tet_count`` and ``model.hex_count``), plus springs, triangle
membranes, gravity, particle-particle interactions, and soft-rigid contacts.

Design
======

The entire solver reduces to three ideas:

1. **Scalar energies.** Each elastic element defines a scalar strain-energy
   density ``ψ(x) ≥ 0``, minimised at rest.  The force is ``f = −∇ψ``; the
   tangent stiffness is the Hessian ``∂²ψ/∂x²``.

2. **Implicit time step.** Advancing the state is one linear solve of an SPD
   system ``A Δv = h f``.  ``A`` is mass plus a weighted sum of those Hessians,
   so CG converges unconditionally.

3. **Inequality constraints.** Contact is enforced either by a penalty
   (a steep added energy) or by projecting positions to satisfy it.

Mathematical overview
=====================

Each substep performs one Newton iteration of Backward Euler::

    A · Δv = h · f_n
    A = M + h² · H_pot + h · C        (PSD)

``M = m I`` is the lumped mass matrix, ``H_pot = ∂²ψ/∂x²`` the elastic
Hessian, and ``C`` the Rayleigh damping matrix (also PSD).

BSR triplet layout (built once at construction, never rebuilt)::

    [0 .. P)                           — per-particle diagonal (mass + spring sum)
    [P .. P+2·S)                       — per-spring (i,j) and (j,i) off-diagonals
    [P+2·S .. P+2·S+3·R)              — per-tri 3 lumped diagonal blocks
    [P+2·S+3·R .. P+2·S+3·R+16·T)    — per-tet 4×4 Hessian (rest-state, F=I)
    [... +64·H)                        — per-hex 8×8 Hessian blocks

where P, S, R, T, H = particle/spring/tri/tet/hex counts.
"""

import warp as wp
from warp.optim.linear import LinearOperator, bicgstab, cg, cr, gmres, preconditioner
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from newton import ParticleFlags
from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.solver import SolverBase

from .kernels import (
    block_jacobi_mv_kernel,
    build_system_matrix_diagonal_kernel,
    build_system_matrix_diagonal_mass_kernel,
    build_system_matrix_hex_kernel,
    build_system_matrix_sparse_kernel,
    build_system_matrix_tet_kernel,
    build_system_matrix_tri_kernel,
    compute_hex_volume_kernel,
    compute_volume_kernel,
    eval_gravity_from_array,
    eval_hexahedra,
    eval_linear_damping_kernel,
    eval_particle_forces,
    eval_particle_ground_contacts,
    eval_soft_contacts,
    eval_springs,
    eval_springs_linear_and_torque,
    eval_tetrahedra,
    eval_triangles,
    invert_block_diagonal_kernel,
    update_state,
)
from .soft_surface_contacts import SoftBodySurfaceContacts

PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


@wp.kernel
def accumulate_scaled_kernel(
    src: wp.array[wp.vec3],
    scale: wp.float32,
    dst: wp.array[wp.vec3],
):
    tid = wp.tid()
    dst[tid] = dst[tid] + scale * src[tid]


class SolverImplicitSoft(SolverBase):
    """Fully-implicit Backward-Euler FEM soft-body solver.

    Handles tetrahedral and/or hexahedral Neo-Hookean FEM, springs, triangle
    membrane FEM, gravity, particle-particle interactions, soft-rigid contacts,
    and an optional analytic ground plane.  Which element types are active is
    determined by ``model.tet_count`` and ``model.hex_count``; both may be
    non-zero simultaneously.

    Args:
        model: The Newton model owning the particles.
        dt: Default timestep [s] used for the initial matrix assembly.
        mass: Unused — kept for API compatibility. Per-particle masses are read
            from ``model.particle_mass`` (set by density during
            :meth:`~newton.ModelBuilder.add_soft_mesh`).
        preconditioner_type: Jacobi preconditioner type (``"diag"``).
        solver_type: Iterative solver — ``"cg"``, ``"bicgstab"``,
            ``"gmres"``, or ``"cr"``.
        linear_solver_maxiter: Maximum solver iterations per substep.
        min_stretch: Lower element principal-stretch clamp (dimensionless).
            Pass a negative value to disable.
        max_stretch: Upper element principal-stretch clamp (dimensionless).
            Pass a negative value to disable.
        ground_plane: ``(nx, ny, nz, d)`` coefficients of an analytic
            ground plane ``n·x + d = 0``. ``None`` disables it.
        ground_ke: Ground normal stiffness [N/m].
        ground_kd: Ground normal damping [N·s/m].
        ground_kf: Ground tangential friction stiffness [N/m].
        ground_mu: Ground Coulomb friction coefficient.
        torque_stiffness: Spring bend-torque stiffness [N·m/rad].
            Active only when ``> 0`` and ``spring_rest_direction`` is set.
        torque_damping: Spring bend-torque damping [N·m·s/rad].
        spring_rest_direction: Per-spring rest-frame direction vectors,
            shape ``[spring_count, 3]``. ``None`` means no torque.
        self_contact_ke: Particle-particle self-contact penalty stiffness [N/m].
            Pass ``0.0`` to disable; the hash-grid build is skipped entirely.
        self_contact_kd: Particle-particle self-contact damping [N·s/m].
        linear_damping: Mass-proportional Rayleigh damping coefficient [1/s].
    """

    # ==================================================================
    # 1. Construction
    # ==================================================================

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        preconditioner_type: str = "diag",
        solver_type: str = "bicgstab",
        linear_solver_maxiter: int = 50,
        min_stretch: float = 0.05,
        max_stretch: float = 20.0,
        ground_plane: "tuple[float, float, float, float] | None" = None,
        ground_ke: float = 1.0e5,
        ground_kd: float = 1.0e2,
        ground_kf: float = 1.0e3,
        ground_mu: float = 0.5,
        torque_stiffness: float = 0.0,
        torque_damping: float = 0.0,
        spring_rest_direction: "object | None" = None,
        self_contact_ke: float = 1.0e3,
        self_contact_kd: float = 1.0e2,
        linear_damping: float = 0.0,
    ):
        super().__init__(model=model)

        model.particle_ke = float(self_contact_ke)
        model.particle_kd = float(self_contact_kd)

        self.mass = float(mass)
        self.linear_damping = float(linear_damping)
        self.preconditioner_type = preconditioner_type
        self.solver_type = solver_type
        self.linear_solver_maxiter = linear_solver_maxiter
        # 0 = device-side fixed iteration count (CUDA-graph capturable, always
        # runs `linear_solver_maxiter` iterations).  >0 = host-side residual
        # checks with early exit, but NOT graph-capturable.  Use >0 only to
        # diagnose how many iterations the solve actually needs.
        self.linear_solver_check_every = 0
        self.last_cg_iters: int | None = None
        self.min_stretch = float(min_stretch)
        self.max_stretch = float(max_stretch)
        self.dt = float(dt)

        # ------------------------------------------------------------------
        # Hexahedral element data — read from model (set by ModelBuilder)
        # ------------------------------------------------------------------
        H = model.hex_count
        self.hex_count = H
        if H > 0:
            self.hex_indices = model.hex_indices.reshape((H, 8))
            self.hex_materials = model.hex_materials
            self.hex_activations = model.hex_activations
            self.hex_inv_J0 = model.hex_inv_J0.reshape((H, 8))
            self.hex_det_J0_w = model.hex_det_J0_w.reshape((H, 8))
        else:
            self.hex_indices = None
            self.hex_materials = None
            self.hex_activations = None
            self.hex_inv_J0 = None
            self.hex_det_J0_w = None

        # BSR triplet buffers.  Layout (all constant, built once at init):
        #   [0..P)                   diagonal (mass + spring)
        #   [P..P+2S)                spring off-diagonals
        #   [P+2S..P+2S+3R)          tri lumped diagonal  (rest-geometry only)
        #   [P+2S+3R..P+2S+3R+16T)  tet 4×4 Hessian (rest-state F=I)
        #   [... +64H)               hex 8×8 Hessian
        num_blocks = model.particle_count + model.spring_count * 2 + model.tri_count * 3 + model.tet_count * 16 + H * 64
        self.bsr_rows = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_cols = wp.zeros(num_blocks, dtype=wp.int32, device=model.device)
        self.bsr_values = wp.zeros(num_blocks, dtype=wp.mat33f, device=model.device)
        self.A_bsr = bsr_zeros(
            rows_of_blocks=model.particle_count,
            cols_of_blocks=model.particle_count,
            block_type=wp.mat33f,
            device=model.device,
        )
        # Build the system matrix once at construction; it is never rebuilt.
        self._build_constant_matrix(model, dt)
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True,
        )
        self._pc_inv_diag = None
        self.M_bsr = self._build_preconditioner()
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        # Scratch reused across steps so nothing allocates inside a captured graph.
        self._force_bufs: dict[str, wp.array] = {}
        for key in (
            "spring",
            "tet",
            "hex",
            "tri",
            "particle",
            "ground",
            "soft_contact",
            "gravity",
            "damping",
            "surface",
        ):
            self._forces(key)
        self._gravity_arr: wp.array | None = None
        self._lin_solver = None
        self._lin_solver_key = None

        # Initial volume: sum contributions from tet and/or hex elements.
        self._initial_volume = 0.0
        if model.tet_count > 0:
            _vols = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=compute_volume_kernel,
                dim=model.tet_count,
                inputs=[model.particle_q, model.tet_indices],
                outputs=[_vols],
                device=model.device,
            )
            self._initial_volume += float(_vols.numpy().sum())
        if H > 0:
            _hvols = wp.zeros(H, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=compute_hex_volume_kernel,
                dim=H,
                inputs=[self.hex_det_J0_w],
                outputs=[_hvols],
                device=model.device,
            )
            self._initial_volume += float(_hvols.numpy().sum())

        # Spring bend-torque (optional).
        self.torque_stiffness = float(torque_stiffness)
        self.torque_damping = float(torque_damping)
        if spring_rest_direction is None:
            self.spring_rest_direction = wp.zeros(model.spring_count, dtype=wp.vec3, device=model.device)
        else:
            if hasattr(spring_rest_direction, "shape"):
                if spring_rest_direction.shape != (model.spring_count, 3):
                    raise ValueError(
                        f"spring_rest_direction shape {spring_rest_direction.shape} != (spring_count={model.spring_count}, 3)"
                    )
            self.spring_rest_direction = wp.array(spring_rest_direction, dtype=wp.vec3, device=model.device)

        # Optional analytic ground plane  n·x + d = 0.
        self._ground_plane = None
        if ground_plane is not None:
            self._ground_plane = wp.array(ground_plane, dtype=wp.float32, device=model.device)
        self._ground_ke = float(ground_ke)
        self._ground_kd = float(ground_kd)
        self._ground_kf = float(ground_kf)
        self._ground_mu = float(ground_mu)

        self._soft_surface_contacts: SoftBodySurfaceContacts | None = None

    # ==================================================================
    # 2. Step
    # ==================================================================

    def set_soft_surface_contacts(
        self,
        particle_ranges: "list[tuple[int, int]]",
        surface_triangles_list: "list[np.ndarray | None]",
        ke: float = 1.0e3,
        kd: float = 1.0e2,
    ) -> None:
        """Register per-body surface meshes for deformed soft-soft (and soft-rigid) contacts.

        Must be called after :meth:`newton.ModelBuilder.finalize` and before the
        first :meth:`step`.  The meshes are built from rest-pose positions and
        refitted every time :meth:`_rebuild_element_blocks` is called.

        Args:
            particle_ranges: List of ``(start, count)`` slices into
                ``model.particle_q`` — one entry per soft body.
            surface_triangles_list: Per-body surface triangle arrays, shape
                ``(T_i, 3)``, indices local to the body (0 .. count_i−1).
                Pass ``None`` for bodies without surface data.
            ke: Surface penalty stiffness [N/m].
            kd: Surface contact damping [N·s/m].
        """
        model = self.model
        self._soft_surface_contacts = SoftBodySurfaceContacts(
            particle_q=model.particle_q,
            particle_ranges=particle_ranges,
            surface_triangles_list=surface_triangles_list,
            ke=ke,
            kd=kd,
            device=model.device,
        )

    def _accumulate_forces(
        self,
        model: Model,
        state_in: State,
        contacts: Contacts,
        control: Control,
        dt: float,
    ) -> None:
        """Write ``state_in.particle_f = h · Σf`` from all force terms."""
        if self._soft_surface_contacts is not None:
            # update() (BVH refit) must be called by the caller before step()
            # when using CUDA graphs — mesh.refit() goes through the C++ runtime
            # and may not be captured.  _accumulate_forces is called inside the
            # graph, so we only do the pure-GPU force evaluation here.
            surface_contact_forces = self._soft_surface_contacts.eval_forces(
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_radius,
                model.particle_flags,
                model.particle_count,
            )
        else:
            surface_contact_forces = self._forces("surface")

        # Accumulate in place: wp.array arithmetic allocates a fresh array per
        # operand, and any allocation inside a captured CUDA graph is freed after
        # capture while the graph keeps writing to it.
        particle_f = state_in.particle_f
        particle_f.zero_()
        for term in (
            self.eval_spring_forces(model, state_in),
            self.eval_tetrahedral_forces(model, control, state_in),
            self.eval_hexahedral_forces(model, control, state_in),
            self.eval_triangle_forces(model, control, state_in),
            self.eval_particle_particle_forces(model, control, state_in),
            self.eval_particle_ground_contact_forces(model, control, state_in),
            self.eval_soft_contact_forces(model, state_in, contacts),
            self.eval_gravity_forces(model),
            self.eval_linear_damping_forces(model, state_in),
            surface_contact_forces,
        ):
            wp.launch(
                kernel=accumulate_scaled_kernel,
                dim=model.particle_count,
                inputs=[term, wp.float32(dt)],
                outputs=[particle_f],
                device=model.device,
            )

    def _rebuild_element_blocks(
        self,
        particle_q: wp.array,
        dt: float,
        update_preconditioner: bool = True,
    ) -> None:
        """Recompute tet and/or hex Hessian BSR blocks from the current particle positions.

        Rebuilds every active element type's Hessian from the current deformation
        gradient F.  Must be called outside a CUDA graph: ``bsr_set_from_triplets``
        internally calls CUB ``DeviceRadixSort``, which allocates temporary
        storage and cannot be captured.  Invoke from the frame loop before
        ``wp.capture_launch``.

        When a CUDA graph is active, pass ``update_preconditioner=False``.
        ``bsr_set_from_triplets`` reuses the same ``A_bsr`` buffer (same GPU
        pointer), so the captured graph reads the freshly written values.
        ``preconditioner()`` allocates a *new* inv_diag buffer; reassigning
        ``self.M_bsr`` drops the old object's refcount to zero, Python frees
        it, and the graph accesses freed GPU memory.  Skipping the M update
        keeps the captured pointer alive and valid.

        Also refits the surface-contact BVHs if they were registered via
        :meth:`set_soft_surface_contacts`.

        Args:
            particle_q: Current particle positions [m].
            dt: Substep size [s].
            update_preconditioner: Whether to rebuild ``M_bsr`` after assembly.
        """
        # BVH refit for soft-surface contacts
        if self._soft_surface_contacts is not None:
            self._soft_surface_contacts.update(particle_q)

        model = self.model
        if model.tet_count > 0:
            wp.launch(
                kernel=build_system_matrix_tet_kernel,
                dim=model.tet_count,
                inputs=[
                    particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    wp.float32(dt),
                    wp.int32(self._tet_bsr_offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._tet_all_dirty,
                ],
                device=model.device,
            )
        if self.hex_count > 0:
            wp.launch(
                kernel=build_system_matrix_hex_kernel,
                dim=self.hex_count * 64,
                inputs=[
                    particle_q,
                    self.hex_indices,
                    self.hex_inv_J0,
                    self.hex_det_J0_w,
                    self.hex_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    wp.float32(dt),
                    wp.int32(self._hex_bsr_offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._hex_all_dirty,
                ],
                device=model.device,
            )
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=False,
        )
        # block_diag refreshes in place (pointers stable) so it is safe even
        # while a CUDA graph is active; other types reallocate M_bsr.
        if update_preconditioner or self.preconditioner_type == "block_diag":
            self._refresh_preconditioner()

    def _rebuild_tet_blocks(self, *args, **kwargs) -> None:
        """Deprecated alias for :meth:`_rebuild_element_blocks`."""
        return self._rebuild_element_blocks(*args, **kwargs)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        """Advance one fully-implicit substep.

        Evaluates all forces into ``state_in.particle_f = h · Σf``, then
        solves ``A · Δv = particle_f``.  The system matrix ``A`` is constant
        (built once at construction from rest-pose element Hessians); call
        :meth:`_rebuild_element_blocks` before ``step()`` if you need the
        tangent refreshed from the current deformed configuration.

        Surface-contact BVH refitting must be done by calling
        :meth:`_rebuild_element_blocks` from outside a CUDA graph, rather than
        relying on ``step()`` itself.
        """
        model = self.model
        if control is None:
            control = model.control()

        self._accumulate_forces(model, state_in, contacts, control, dt)
        self.implicit_integration(model, state_in, state_out, dt)

        return state_out

    # ==================================================================
    # 3. Internals: system matrix + linear solve
    # ==================================================================

    def _build_constant_matrix(self, model: Model, dt: float) -> None:
        """Write all BSR triplets once at construction.

        Mass, spring, and tri blocks are state-independent.  The tet/hex
        Hessian blocks are written here for the rest pose (F=I) and then
        refreshed per-step via :meth:`_rebuild_element_blocks`.
        """
        offset = 0

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
                    model.particle_mass,
                    wp.int32(model.spring_count),
                ],
                device=model.device,
            )
        else:
            wp.launch(
                kernel=build_system_matrix_diagonal_mass_kernel,
                dim=model.particle_count,
                inputs=[self.bsr_rows, self.bsr_cols, self.bsr_values, model.particle_mass],
                device=model.device,
            )
        offset += model.particle_count

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

        # Tri lumped diagonal — depends only on rest geometry, not positions.
        if model.tri_count > 0:
            wp.launch(
                kernel=build_system_matrix_tri_kernel,
                dim=model.tri_count,
                inputs=[
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

        # Tet 4×4 Hessian — written here at rest (F=I); refreshed per-step
        # from the deformed configuration in _rebuild_element_blocks().
        self._tet_bsr_offset = offset
        if model.tet_count > 0:
            self._tet_all_dirty = wp.ones(model.tet_count, dtype=wp.int32, device=model.device)
            wp.launch(
                kernel=build_system_matrix_tet_kernel,
                dim=model.tet_count,
                inputs=[
                    model.particle_q,
                    model.tet_indices,
                    model.tet_poses,
                    model.tet_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    wp.float32(dt),
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._tet_all_dirty,
                ],
                device=model.device,
            )
            offset += model.tet_count * 16

        # Hex 8×8 Hessian (rest state: dirty-flag all elements)
        self._hex_bsr_offset = offset
        if self.hex_count > 0:
            self._hex_all_dirty = wp.ones(self.hex_count, dtype=wp.int32, device=model.device)
            wp.launch(
                kernel=build_system_matrix_hex_kernel,
                dim=self.hex_count * 64,
                inputs=[
                    model.particle_q,
                    self.hex_indices,
                    self.hex_inv_J0,
                    self.hex_det_J0_w,
                    self.hex_materials,
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                    wp.float32(dt),
                    wp.int32(offset),
                    self.bsr_rows,
                    self.bsr_cols,
                    self.bsr_values,
                    self._hex_all_dirty,
                ],
                device=model.device,
            )

    # ------------------------------------------------------------------
    # Preconditioner
    # ------------------------------------------------------------------

    def _build_preconditioner(self):
        """Construct the preconditioner operator for :attr:`A_bsr`.

        ``"block_diag"`` inverts the full 3x3 diagonal block per particle.
        Warp's built-in ``"diag"`` keeps only the three diagonal scalars of
        that block, discarding the intra-particle x/y/z coupling that
        Neo-Hookean elasticity produces; retaining it markedly reduces the
        CG iteration count.

        The inverse-block buffer is allocated **once** and refreshed in place,
        so the operator's device pointers stay valid for the lifetime of any
        CUDA graph that captures a solve.
        """
        if self.preconditioner_type != "block_diag":
            return preconditioner(self.A_bsr, ptype=self.preconditioner_type)

        n = self.model.particle_count
        device = self.model.device
        self._pc_inv_diag = wp.empty(n, dtype=wp.mat33f, device=device)
        self._refresh_preconditioner()

        inv_diag = self._pc_inv_diag

        def block_jacobi_mv(x, y, z, alpha, beta):
            wp.launch(
                kernel=block_jacobi_mv_kernel,
                dim=n,
                inputs=[inv_diag, x, y, z, wp.float32(alpha), wp.float32(beta)],
                device=device,
            )

        return LinearOperator(shape=(n, n), dtype=wp.vec3, device=device, matvec=block_jacobi_mv)

    def _refresh_preconditioner(self) -> None:
        """Recompute the preconditioner from the current :attr:`A_bsr`.

        For ``"block_diag"`` this writes in place, leaving ``M_bsr`` and its
        device pointers untouched — safe to call every frame even while a CUDA
        graph holds references to them.  Other types reallocate, so callers
        must skip this while a graph is active.
        """
        if self.preconditioner_type != "block_diag":
            self.M_bsr = preconditioner(self.A_bsr, ptype=self.preconditioner_type)
            return

        wp.launch(
            kernel=invert_block_diagonal_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.A_bsr.offsets,
                self.A_bsr.columns,
                self.A_bsr.values,
                self._pc_inv_diag,
            ],
            device=self.model.device,
        )

    def implicit_integration(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ) -> None:
        """Solve ``A · Δv = particle_f`` and integrate ``v``, ``x``.

        ``state_in.particle_f`` must already be pre-multiplied by ``dt``.
        """
        sg = getattr(self, "_solver_glue", None)
        if sg is not None and sg._mask is not None:
            sg.apply_override(model, state_in, dt)
            sg.apply_filter(model, state_in)

        maxiter = self.linear_solver_maxiter
        # check_every=0 keeps the solve graph-capturable but always runs the
        # full `maxiter` iterations (warp/optim/linear.py: the conditional
        # early-exit path allocates its condition buffer per call, which cannot
        # be baked into a persistent graph).  Set linear_solver_check_every > 0
        # to trade capture for a real early exit.
        check_every = self.linear_solver_check_every
        solvers = {"cg": cg, "bicgstab": bicgstab, "gmres": gmres, "cr": cr}
        solve = solvers.get(self.solver_type)
        if solve is None:
            raise ValueError(f"Invalid solver type: {self.solver_type}")

        # Persistent functor: the one-shot solver functions allocate their scratch
        # buffers on every call, which is not safe inside a captured CUDA graph.
        key = (self.solver_type, maxiter, check_every)
        if self._lin_solver is None or self._lin_solver_key != key:
            self._lin_solver = solve(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=maxiter,
                M=self.M_bsr,
                check_every=check_every,
                use_cuda_graph=False,
                run=False,
            )
            self._lin_solver_key = key
        result = self._lin_solver(b=state_in.particle_f, M=self.M_bsr)

        if check_every > 0 and result is not None:
            # Host-check path returns (iterations, residual, tolerance).
            self.last_cg_iters = int(result[0])

        wp.launch(
            kernel=update_state,
            dim=model.particle_count,
            inputs=[
                self.dv,
                dt,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=model.device,
        )

    # ==================================================================
    # 4. Force evaluators
    # ==================================================================

    def eval_tetrahedral_forces(self, model: Model, control: Control, state: State):
        """Tet FEM elastic force from Stable Neo-Hookean [N]."""
        forces = self._forces("tet")
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
                    wp.float32(self.min_stretch),
                    wp.float32(self.max_stretch),
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_hexahedral_forces(self, model: Model, control: Control, state: State) -> wp.array:
        """Q1 hex FEM elastic force from Stable Neo-Hookean [N].

        Returns zeros when ``hex_count == 0``.
        """
        forces = self._forces("hex")
        if self.hex_count == 0:
            return forces
        act = (
            control.hex_activations
            if (control is not None and hasattr(control, "hex_activations"))
            else self.hex_activations
        )
        wp.launch(
            kernel=eval_hexahedra,
            dim=self.hex_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.hex_indices,
                self.hex_inv_J0,
                self.hex_det_J0_w,
                act,
                self.hex_materials,
                wp.float32(self.min_stretch),
                wp.float32(self.max_stretch),
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def eval_triangle_forces(self, model: Model, control: Control, state: State):
        """Triangle membrane (in-plane stretch + area-preservation) force [N]."""
        forces = self._forces("tri")
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

    def eval_spring_forces(self, model: Model, state: State):
        """Linear spring force [N] with optional bend torque."""
        forces = self._forces("spring")
        if model.spring_count == 0:
            return forces
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
        """Hash-grid particle-particle penalty + Coulomb friction [N]."""
        forces = self._forces("particle")
        if (
            model.particle_ke <= 0.0
            or model.particle_count <= 1
            or model.particle_max_radius <= 0.0
            or model.particle_grid is None
        ):
            return forces
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
        """Particle vs analytic ground plane (penalty + Coulomb friction) [N].

        Returns zeros when no ``ground_plane`` was configured.
        """
        forces = self._forces("ground")
        if self._ground_plane is None:
            return forces
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
        """Force-based soft-rigid contact from ``model.collide(state)`` [N]."""
        forces = self._forces("soft_contact")
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
        """Per-particle gravity ``f = m · g`` [N]."""
        forces = self._forces("gravity")
        if model.particle_count:
            wp.launch(
                kernel=eval_gravity_from_array,
                dim=model.particle_count,
                inputs=[self._get_gravity_array(model), model.particle_mass, model.particle_flags],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_linear_damping_forces(self, model: Model, state: State) -> wp.array:
        """Mass-proportional Rayleigh damping ``f = −α·m·v`` [N].

        Damps rigid-body drift and rolling without affecting rest-state
        equilibrium.  Zero-cost when ``linear_damping == 0``.
        """
        forces = self._forces("damping")
        if self.linear_damping > 0.0 and model.particle_count:
            wp.launch(
                kernel=eval_linear_damping_kernel,
                dim=model.particle_count,
                inputs=[
                    state.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    wp.float32(self.linear_damping),
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    # ==================================================================
    # 5. Helpers
    # ==================================================================

    def _get_gravity_array(self, model: Model) -> wp.array:
        """Return ``model.gravity`` as a ``wp.array[wp.vec3]``."""
        g = model.gravity
        if isinstance(g, wp.array):
            return g
        if self._gravity_arr is None:
            self._gravity_arr = wp.array([g], dtype=wp.vec3, device=model.device)
        return self._gravity_arr

    def _forces(self, key: str) -> wp.array:
        """Return the zeroed persistent force scratch buffer for ``key``."""
        buf = self._force_bufs.get(key)
        if buf is None:
            buf = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.model.device)
            self._force_bufs[key] = buf
        else:
            buf.zero_()
        return buf
