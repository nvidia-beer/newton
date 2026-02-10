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
    build_system_matrix_sparse_kernel,
    eval_springs,
    eval_tetrahedra,
    eval_triangles,
    eval_bending,
    eval_triangles_contact,
    eval_gravity,
    eval_soft_contacts,
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
    """

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        preconditioner_type: str = "id",
        solver_type: str = "bicgstab",
        use_constraint_contacts: bool = False,
        contact_relaxation: float = 0.9,
    ):
        super().__init__(model=model)
        
        self.friction_smoothing = 2.0
        self.mass = mass
        self.Minv = 1.0 / self.mass
        
        # Constraint-based contact settings (like XPBD)
        self.use_constraint_contacts = use_constraint_contacts
        self.contact_relaxation = contact_relaxation
        
        # Pre-allocate arrays for BSR matrix construction
        num_blocks = model.spring_count * 4  # Each edge contributes 4 blocks
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
        
        # Build and assemble the system matrix
        self.initialize_system_matrix_sparse(model, dt)
        
        # Preconditioner for the sparse matrix
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

    def initialize_system_matrix_sparse(self, model: Model, dt: float):
        """Build the sparse system matrix for implicit integration."""
        if model.spring_count == 0:
            return
            
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
                self.mass,
                self.Minv
            ],
            device=model.device,
        )
        
        bsr_set_from_triplets(
            dest=self.A_bsr,
            rows=self.bsr_rows,
            columns=self.bsr_cols,
            values=self.bsr_values,
            prune_numerical_zeros=True
        )

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
        """Evaluate particle-ground contact forces.
        
        Note: In the current newton version, ground contact is handled through
        the collision pipeline (model.collide()). This method is kept for
        backwards compatibility but returns zero forces.
        """
        return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
    
    def eval_soft_contact_forces(self, model: Model, state: State, contacts: Contacts):
        """Evaluate contact forces from collision detection results."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        
        if contacts is None or not hasattr(contacts, 'soft_contact_count'):
            return forces
        
        # Get contact count (convert to Python int)
        contact_count = int(contacts.soft_contact_count.numpy()[0])
        if contact_count == 0:
            return forces
        
        # Launch kernel to compute contact forces
        wp.launch(
            kernel=eval_soft_contacts,
            dim=contact_count,
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
        """Evaluate gravity forces for all particles."""
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if model.particle_count:
            # Get gravity as vec3, scale by mass
            g = self._get_gravity_vec3(model)
            gravity_force_wp = wp.vec3(g[0] * self.mass, g[1] * self.mass, g[2] * self.mass)
            wp.launch(
                kernel=eval_gravity,
                dim=model.particle_count,
                inputs=[gravity_force_wp, model.particle_flags],
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
        if self.solver_type == "cg":
            iterations, residual, _ = cg(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=3,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        elif self.solver_type == "bicgstab":
            iterations, residual, _ = bicgstab(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=3,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        elif self.solver_type == "gmres":
            iterations, residual, _ = gmres(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=3,
                M=self.M_bsr,
                use_cuda_graph=True
            )
        elif self.solver_type == "cr":
            iterations, residual, _ = cr(
                self.A_bsr,
                state_in.particle_f,
                self.dv,
                tol=1e-2,
                maxiter=3,
                M=self.M_bsr,
                use_cuda_graph=True
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
        
        # Debug: Check input state
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        
        if self._debug_step_count <= 3:
            import numpy as np
            pos = state_in.particle_q.numpy()
            vel = state_in.particle_qd.numpy()
            print(f"[DEBUG] Step {self._debug_step_count}: pos_mean={pos.mean(axis=0)}, vel_mean={vel.mean(axis=0)}", flush=True)
            print(f"[DEBUG] Step {self._debug_step_count}: pos_max={np.abs(pos).max():.2e}, vel_max={np.abs(vel).max():.2e}", flush=True)
        
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
        
        if self._debug_step_count <= 3:
            import numpy as np
            print(f"[DEBUG] Step {self._debug_step_count}: spring_max={np.abs(spring_forces.numpy()).max():.2e}", flush=True)
            print(f"[DEBUG] Step {self._debug_step_count}: tet_max={np.abs(tetrahedral_forces.numpy()).max():.2e}", flush=True)
            print(f"[DEBUG] Step {self._debug_step_count}: gravity_max={np.abs(gravity_forces.numpy()).max():.2e}", flush=True)
        
        # Combine all forces
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
        
        if self._debug_step_count <= 3:
            import numpy as np
            pos_out = state_out.particle_q.numpy()
            vel_out = state_out.particle_qd.numpy()
            print(f"[DEBUG] Step {self._debug_step_count} AFTER: pos_mean={pos_out.mean(axis=0)}, vel_mean={vel_out.mean(axis=0)}", flush=True)
            print(f"[DEBUG] Step {self._debug_step_count} AFTER: pos_max={np.abs(pos_out).max():.2e}, vel_max={np.abs(vel_out).max():.2e}", flush=True)
        
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
        
        contact_count = int(contacts.soft_contact_count.numpy()[0])
        if contact_count == 0:
            return
        
        # Allocate arrays for constraint corrections
        particle_deltas = wp.zeros(
            model.particle_count, dtype=wp.vec3, device=model.device
        )
        body_deltas = None
        if model.body_count > 0:
            body_deltas = wp.zeros(
                model.body_count, dtype=wp.spatial_vector, device=model.device
            )
        
        # Apply constraint corrections iteratively (like XPBD)
        for iteration in range(2):  # Standard 2 iterations
            particle_deltas.zero_()
            if body_deltas is not None:
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
                    model.soft_contact_mu,
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
                outputs=[particle_deltas, body_deltas] if body_deltas is not None else [particle_deltas],
                device=model.device,
            )
            
            # Apply position corrections directly
            # Simple correction: x_new = x_old + delta, v_new = (x_new - x_orig) / dt
            wp.launch(
                kernel=apply_particle_corrections,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,  # Original state before integration
                    state_out.particle_q,  # Current integrated state
                    particle_deltas,
                    model.particle_flags,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )

