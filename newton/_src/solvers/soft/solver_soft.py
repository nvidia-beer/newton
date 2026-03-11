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

"""Soft body solver with implicit integration using sparse matrix solvers."""

import warp as wp
from warp.optim.linear import cg, preconditioner, bicgstab, gmres, cr
from warp.sparse import bsr_zeros, bsr_set_from_triplets

from .kernels import (
    build_system_matrix_diagonal_kernel,
    build_system_matrix_diagonal_mass_kernel,
    build_system_matrix_sparse_kernel,
    build_system_matrix_tet_kernel,
    build_system_matrix_tri_kernel,
    eval_springs,
    eval_tetrahedra,
    eval_triangles,
    eval_bending,
    eval_triangles_contact,
    eval_gravity,
    eval_gravity_from_array,
    eval_soft_contacts,
    eval_particle_ground_contacts,
    solve_soft_contacts_constraint,
    apply_particle_corrections,
    update_state,
    eval_barycentric_constraints,
    clear_barycentric_constraints,
    set_barycentric_constraint,
)
from .particles import eval_particle_forces

from newton import ParticleFlags
from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.solver import SolverBase

PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


class SolverSoft(SolverBase):
    """
    Soft body solver with implicit integration.
    
    Uses sparse matrix solvers (BiCGStab, CG, GMRES, CR) for stable simulation
    of stiff materials. The key equation solved each timestep is:
    
        A · Δv = f
    
    Where:
        A = M - h*D - h²*K is the system matrix
        M is the mass matrix
        D is the damping matrix  
        K is the stiffness matrix
        h is the timestep
        f is the force vector
    
    Parameters
    ----------
    model : Model
        The Newton physics model
    dt : float
        Physics timestep (default: 1/60)
    mass : float
        Particle mass (default: 1.0)
    preconditioner_type : str
        Preconditioner type: "id", "diag", or "diag_abs" (default: "id")
    solver_type : str
        Linear solver: "bicgstab", "cg", "gmres", or "cr" (default: "bicgstab")
    linear_solver_maxiter : int
        Max iterations for the linear solver (default: 50). Large meshes need more;
        too low (e.g. 3) can cause non-convergence and NaN.
    use_fem_tangent_in_matrix : bool
        If True, include tetrahedral and triangle FEM tangent in the system matrix
        and rebuild each step (more stable for stiff materials). If False, only
        mass and springs are in the matrix (default). Set False to avoid NaN
        when using the new FEM-in-matrix path until it is fully validated.
    """

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        preconditioner_type: str = "id",
        solver_type: str = "bicgstab",
        use_fem_tangent_in_matrix: bool = False,
        use_constraint_contacts: bool = False,
        contact_relaxation: float = 0.5,
        contact_max_velocity: float = 20.0,
        contact_max_correction: float = 0.02,
        contact_iterations: int = 2,
        linear_solver_maxiter: int = 50,
        extra_matrix_blocks: int = 0,
        ground_plane: "tuple[float, float, float, float] | None" = None,
        ground_ke: float = 1.0e5,
        ground_kd: float = 1.0e2,
        ground_kf: float = 1.0e3,
        ground_mu: float = 0.5,
    ):
        super().__init__(model=model)
        
        self.friction_smoothing = 2.0
        self.mass = mass
        self.Minv = 1.0 / self.mass
        
        # Optional force-based ground plane (n·x + d = 0)
        self._ground_plane = None
        if ground_plane is not None:
            self._ground_plane = wp.array(
                ground_plane, dtype=wp.float32, device=model.device
            )
        self._ground_ke = float(ground_ke)
        self._ground_kd = float(ground_kd)
        self._ground_kf = float(ground_kf)
        self._ground_mu = float(ground_mu)
        
        # Constraint-based contact settings (like XPBD)
        self.use_constraint_contacts = use_constraint_contacts
        self.contact_relaxation = contact_relaxation
        self.contact_max_velocity = contact_max_velocity  # clamp velocity after correction to avoid explosion
        self.contact_max_correction = contact_max_correction  # max position correction per application
        self.contact_iterations = contact_iterations  # more iterations = better friction resolution
        self.linear_solver_maxiter = linear_solver_maxiter
        self.use_fem_tangent_in_matrix = use_fem_tangent_in_matrix
        
        # Pre-allocate arrays for BSR matrix: base blocks + optional extra (e.g. self-contact in Deformable)
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
        
        # Initialize BSR system matrix
        self.A_bsr = bsr_zeros(
            rows_of_blocks=model.particle_count,
            cols_of_blocks=model.particle_count,
            block_type=wp.mat33f,
            device=model.device,
        )
        
        # Build and assemble the system matrix (with state for FEM tangent; done each step)
        self._build_system_matrix(model, None, dt)
        
        # Preconditioner for the sparse matrix (rebuilt each step when FEM is used)
        self.preconditioner_type = preconditioner_type
        self.M_bsr = preconditioner(self.A_bsr, ptype=preconditioner_type)
        self.solver_type = solver_type
        
        # Use vec3f arrays for dv in the sparse path
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        
        # Barycentric constraint arrays
        self.max_constraints = model.particle_count * 2
        
        self.constraint_is_active = wp.zeros(self.max_constraints, dtype=wp.bool, device=model.device)
        self.constraint_positions = wp.zeros(self.max_constraints, dtype=wp.vec3, device=model.device)
        self.constraint_types = wp.zeros(self.max_constraints, dtype=wp.int32, device=model.device)
        self.constraint_indices = wp.zeros((self.max_constraints, 4), dtype=wp.int32, device=model.device)
        self.constraint_weights = wp.zeros((self.max_constraints, 4), dtype=wp.float32, device=model.device)
        self.constraint_vertex_count = wp.zeros(self.max_constraints, dtype=wp.int32, device=model.device)
        
        self.constraint_count = 0

    def _build_system_matrix(self, model: Model, state: State | None, dt: float):
        """Build the sparse system matrix A = M - h*D - h²*K (mass, springs, tet FEM, tri FEM)."""
        offset = 0
        # 1) Diagonal: M + sum of spring (dt*D - dt²*K) per particle (no duplicate blocks)
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
                    self.mass,
                    self.Minv,
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
        # 2) Spring off-diagonals only (i,j) and (j,i)
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
                    wp.float32(dt),
                    wp.int32(offset),
                ],
                device=model.device,
            )
            offset += model.spring_count * 2
        # 3) Tetrahedral FEM tangent (requires current state)
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
        # 4) Triangle FEM lumped tangent (requires current state)
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
        # Subclass hook: add extra blocks (e.g. self-contact tangent in SolverDeformable)
        self._add_extra_matrix_blocks(model, state, dt, offset)

        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True
        )

    def _add_extra_matrix_blocks(
        self, model: Model, state: State | None, dt: float, block_offset: int
    ) -> None:
        """Override in subclasses to add blocks to the system matrix (e.g. self-contact tangent).
        Write into self.bsr_rows[block_offset:], self.bsr_cols[block_offset:], self.bsr_values[block_offset:].
        """
        pass

    def eval_tetrahedral_forces(self, model: Model, control: Control, state: State):
        """Evaluate tetrahedral FEM forces."""
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
                    control.tet_activations if hasattr(control, 'tet_activations') else model.tet_activations,
                    model.tet_materials,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_spring_forces(self, model: Model, state: State):
        """Evaluate spring forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.spring_count:
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

    def eval_triangle_forces(self, model: Model, control: Control, state: State):
        """Evaluate triangle membrane forces."""
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
                    control.tri_activations if hasattr(control, 'tri_activations') else model.tri_activations,
                    model.tri_materials,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_bending_forces(self, model: Model, control: Control, state: State):
        """Evaluate edge bending forces."""
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

    def eval_triangle_contact_forces(self, model: Model, control: Control, state: State):
        """Evaluate triangle-particle contact forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if getattr(model, 'enable_tri_collisions', False) and model.tri_count:
            wp.launch(
                kernel=eval_triangles_contact,
                dim=model.tri_count * model.particle_count,
                inputs=[
                    model.particle_count,
                    state.particle_q,
                    state.particle_qd,
                    model.tri_indices,
                    model.tri_materials,
                    model.particle_radius,
                ],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_particle_particle_forces(self, model: Model, control: Control, state: State):
        """Evaluate particle-particle interaction forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.particle_count > 1 and model.particle_max_radius > 0.0 and model.particle_grid is not None:
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
        """Particle–ground contact forces.

        If ground_plane=(nx,ny,nz,d) was passed at construction, applies force-based
        ground contact and Coulomb friction. Otherwise returns zero (ground via
        collision pipeline: use_constraint_contacts=True and model.collide(state)).
        """
        if self._ground_plane is None:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        forces = wp.zeros(
            model.particle_count, dtype=wp.vec3, device=model.device
        )
        wp.launch(
            kernel=eval_particle_ground_contacts,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_flags,
                self._ground_ke,
                self._ground_kd,
                self._ground_kf,
                self._ground_mu,
                self._ground_plane,
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces
    
    def eval_soft_contact_forces(self, model: Model, state: State, contacts: Contacts):
        """Evaluate contact forces from collision detection results."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        
        if contacts is None or not hasattr(contacts, 'soft_contact_count'):
            return forces
        
        # Launch with soft_contact_max; kernel uses soft_contact_count[0] and returns early for tid >= count (no host sync, graph-capture safe).
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

    def _get_gravity_vec3(self, model: Model) -> wp.vec3:
        """Get gravity as wp.vec3, handling both array and vec3 formats."""
        g = model.gravity
        # If gravity is a warp array, extract the first element
        if isinstance(g, wp.array):
            g_np = g.numpy()
            return wp.vec3(float(g_np[0][0]), float(g_np[0][1]), float(g_np[0][2]))
        # If it's already a vec3, use it directly
        return g

    def _get_gravity_array(self, model: Model) -> wp.array:
        """Get gravity as wp.array, handling both array and vec3 formats."""
        g = model.gravity
        # If gravity is already a warp array, return it
        if isinstance(g, wp.array):
            return g
        # If it's a vec3, wrap it in an array
        return wp.array([g], dtype=wp.vec3, device=model.device)

    def eval_gravity_forces(self, model: Model):
        """Evaluate gravity forces for all particles. Uses device array for gravity (no host sync, graph-capture safe)."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.particle_count:
            gravity_arr = self._get_gravity_array(model)
            wp.launch(
                kernel=eval_gravity_from_array,
                dim=model.particle_count,
                inputs=[gravity_arr, self.mass, model.particle_flags],
                outputs=[forces],
                device=model.device,
            )
        return forces

    def eval_constraints(self, model: Model, control: Control, state: State):
        """Evaluate barycentric constraint forces."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if self.constraint_count > 0:
            wp.launch(
                kernel=eval_barycentric_constraints,
                dim=self.constraint_count,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.constraint_positions,
                    self.constraint_is_active,
                    self.constraint_types,
                    self.constraint_indices,
                    self.constraint_weights,
                    self.constraint_vertex_count,
                    model.particle_ke,
                    model.particle_kd,
                    forces
                ],
                device=model.device,
            )
        return forces

    def clear_constraints(self, model=None):
        """Clear all constraints."""
        if self.constraint_count > 0:
            wp.launch(
                kernel=clear_barycentric_constraints,
                dim=self.constraint_count,
                inputs=[
                    self.constraint_is_active,
                    self.constraint_types,
                    self.constraint_indices,
                    self.constraint_weights,
                    self.constraint_vertex_count
                ],
                device=self.model.device,
            )
            self.constraint_count = 0
        
        if hasattr(self, 'initialized_constraint_data'):
            self.initialized_constraint_data = []

    def add_constraint(self, constraint_type_id, vertex_indices, barycentric_weights, target_position, vertex_count, model=None):
        """Add a constraint to the solver."""
        if self.constraint_count >= self.max_constraints:
            raise ValueError(f"Maximum number of constraints ({self.max_constraints}) reached")
        
        constraint_idx = self.constraint_count
        
        wp.launch(
            kernel=set_barycentric_constraint,
            dim=1,
            inputs=[
                constraint_idx,
                constraint_type_id,
                wp.vec3(*target_position),
                vertex_indices,
                barycentric_weights,
                vertex_count,
                self.constraint_is_active,
                self.constraint_positions,
                self.constraint_types,
                self.constraint_indices,
                self.constraint_weights,
                self.constraint_vertex_count
            ],
            device=self.model.device,
        )
        
        self.constraint_count += 1

    def implicit_integration(self, model: Model, state_in: State, state_out: State, dt: float):
        """Perform implicit integration step using sparse matrix solver."""
        maxiter = self.linear_solver_maxiter
        # check_every=0: device-side convergence only, no host sync during solve (graph-capture safe)
        if self.solver_type == "cg":
            iterations, residual, _ = cg(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=maxiter,
                M=self.M_bsr,
                check_every=0,
                use_cuda_graph=True,
            )
        elif self.solver_type == "bicgstab":
            iterations, residual, _ = bicgstab(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=maxiter,
                M=self.M_bsr,
                check_every=0,
                use_cuda_graph=True,
            )
        elif self.solver_type == "gmres":
            iterations, residual, _ = gmres(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=maxiter,
                M=self.M_bsr,
                check_every=0,
                use_cuda_graph=True,
            )
        elif self.solver_type == "cr":
            iterations, residual, _ = cr(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=maxiter,
                M=self.M_bsr,
                check_every=0,
                use_cuda_graph=True,
            )
        else:
            raise ValueError(f"Invalid solver type: {self.solver_type}")
        
        wp.launch(
            kernel=update_state,
            dim=model.particle_count,
            inputs=[
                self.dv, dt,
                state_in.particle_q, state_in.particle_qd,
                state_out.particle_q, state_out.particle_qd
            ],
            device=model.device,
        )

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """
        Simulate the model for a given time step.
        
        Args:
            state_in: The input state
            state_out: The output state
            control: The control input
            contacts: The contact information (generated by model.collide())
            dt: The time step
        """
        model = self.model
        
        if control is None:
            control = model.control()

        # Rebuild system matrix (and preconditioner) only when FEM tangent is in the matrix
        if self.use_fem_tangent_in_matrix:
            self._build_system_matrix(model, state_in, dt)
            self.M_bsr = preconditioner(self.A_bsr, ptype=self.preconditioner_type)

        # Evaluate all forces
        spring_forces = self.eval_spring_forces(model, state_in)
        triangle_forces = self.eval_triangle_forces(model, control, state_in)
        triangle_contact_forces = self.eval_triangle_contact_forces(model, control, state_in)
        bending_forces = self.eval_bending_forces(model, control, state_in)
        tetrahedral_forces = self.eval_tetrahedral_forces(model, control, state_in)
        particle_forces = self.eval_particle_particle_forces(model, control, state_in)
        particle_ground_contact_forces = self.eval_particle_ground_contact_forces(model, control, state_in)
        soft_contact_forces = self.eval_soft_contact_forces(model, state_in, contacts)
        constraint_forces = self.eval_constraints(model, control, state_in)
        gravity_forces = self.eval_gravity_forces(model)

        # Combine all forces (skip force-based soft contact when using constraint contacts to avoid double penalty)
        if self.use_constraint_contacts:
            soft_contact_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        state_in.particle_f = dt * (
            spring_forces + 
            tetrahedral_forces + 
            triangle_forces + 
            triangle_contact_forces + 
            bending_forces + 
            particle_forces + 
            particle_ground_contact_forces + 
            soft_contact_forces +
            gravity_forces + 
            constraint_forces
        )
        
        # Implicit integration
        self.implicit_integration(model, state_in, state_out, dt)
        
        # Apply constraint-based contact corrections (like XPBD) if enabled
        if self.use_constraint_contacts and contacts is not None:
            self.apply_constraint_contact_corrections(model, state_in, state_out, contacts, dt)

        # Integrate rigid bodies if present
        # Use gravity array format for integrate_bodies kernel compatibility
        if model.body_count:
            gravity_array = self._get_gravity_array(model)
            original_gravity = model.gravity
            model.gravity = gravity_array
            self.integrate_bodies(model, state_in, state_out, dt, 0.0)
            model.gravity = original_gravity
        
        return state_out
    
    def apply_constraint_contact_corrections(
        self, model: Model, state_in: State, state_out: State, contacts: Contacts, dt: float
    ):
        """Apply constraint-based contact corrections (like XPBD) to prevent penetration.
        
        This method applies position corrections after integration to prevent particles
        from penetrating rigid shapes. It's similar to XPBD's constraint-based approach
        but works with SolverSoft's force-based integration.
        
        Args:
            model: The physics model
            state_in: Input state (before integration)
            state_out: Output state (after integration, will be corrected)
            contacts: Contact information from collision detection
            dt: Timestep
        """
        if not hasattr(contacts, 'soft_contact_count'):
            return
        
        # Kernel uses soft_contact_count[0] internally; launch with soft_contact_max so no host sync (graph-capture safe).
        particle_friction = getattr(model, "particle_friction", None)
        if particle_friction is None or not hasattr(particle_friction, "shape") or particle_friction.shape[0] != model.particle_count:
            particle_friction = wp.full(
                (model.particle_count,),
                float(model.soft_contact_mu),
                dtype=wp.float32,
                device=model.device,
            )
            model.particle_friction = particle_friction
        
        # Allocate arrays for constraint corrections (kernel always expects 2 outputs: delta, body_delta)
        particle_deltas = wp.zeros(
            model.particle_count, dtype=wp.vec3, device=model.device
        )
        body_deltas = wp.zeros(
            model.body_count, dtype=wp.spatial_vector, device=model.device
        )
        
        # Apply constraint corrections iteratively (like XPBD)
        for iteration in range(self.contact_iterations):
            particle_deltas.zero_()
            body_deltas.zero_()
            
            # Solve constraint corrections
            wp.launch(
                kernel=solve_soft_contacts_constraint,
                dim=contacts.soft_contact_max,
                inputs=[
                    state_out.particle_q,  # Current integrated state
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
            
            # Apply position corrections (additive velocity update + clamps for stability)
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

