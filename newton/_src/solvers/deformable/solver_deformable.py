# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Deformable solver: SolverSoft + self-collision (vertex–triangle and edge–edge
# repulsion on surface mesh, BVH-based detection, RHS-only forces).

import numpy as np
import warp as wp

from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.soft import SolverSoft
from newton._src.utils.mesh import get_surface_triangles_and_edges
from warp.optim.linear import preconditioner

from .kernels_self_contact import (
    build_system_matrix_self_contact_diagonal_kernel,
    eval_self_contact_edge_edge_from_collision_info,
    eval_self_contact_vertex_triangle_from_collision_info,
)
from .tri_mesh_collision import TriMeshCollisionDetector


class SolverDeformable(SolverSoft):
    """
    Deformable soft body solver with self-collision.

    Core is unchanged from SolverSoft: implicit integration, one step per call.
    When handle_self_contact is True, a lumped self-contact tangent (dt² * k_eff * I
    per vertex in contact) is added to the system matrix A so repulsion is implicit;
    vertex–triangle contact count is used for k_eff. Edge–edge remains RHS-only.
    Detection uses a margin (>= radius); force law uses radius/stiffness/force_cap.
    """

    def __init__(
        self,
        model: Model,
        dt: float = 1.0 / 60.0,
        mass: float = 1.0,
        preconditioner_type: str = "id",
        solver_type: str = "bicgstab",
        use_constraint_contacts: bool = False,
        contact_relaxation: float = 0.5,
        contact_max_velocity: float = 20.0,
        contact_max_correction: float = 0.02,
        contact_iterations: int = 2,
        linear_solver_maxiter: int = 50,
        ground_plane: "tuple[float, float, float, float] | None" = None,
        ground_ke: float = 1.0e5,
        ground_kd: float = 1.0e2,
        ground_kf: float = 1.0e3,
        ground_mu: float = 0.5,
        handle_self_contact: bool = False,
        self_contact_radius: float = 0.02,
        self_contact_margin: float | None = None,
        self_contact_stiffness: float = 2.0e4,
        self_contact_force_cap: float = 2.0,
        self_contact_edge_edge: bool = True,
        self_contact_friction_mu: float = 0.5,
        self_contact_friction_epsilon: float = 1.0e-2,
        vertex_collision_buffer_pre_alloc: int = 8,
        edge_collision_buffer_pre_alloc: int = 8,
        edge_edge_parallel_epsilon: float = 1.0e-5,
        extra_matrix_blocks: int | None = None,
    ):
        """
        Args:
            handle_self_contact: Enable vertex–triangle and edge–edge self-contact.
            self_contact_radius: Distance at which pairs start to generate repulsion (force law).
            self_contact_margin: BVH query radius for collision detection; must be >= radius.
                If None, set to 1.5 * self_contact_radius.
            self_contact_stiffness: Repulsion stiffness.
            self_contact_force_cap: Cap on repulsion force magnitude (0 = no cap).
            self_contact_edge_edge: Include edge–edge repulsion (in addition to vertex–triangle).
            self_contact_friction_mu: Friction coefficient for self-contact (VBD-style).
            self_contact_friction_epsilon: Friction smoothing threshold for small slip (VBD-style).
            vertex_collision_buffer_pre_alloc: Per-vertex collision buffer size for BVH detection.
            edge_collision_buffer_pre_alloc: Per-edge collision buffer size for BVH detection.
            edge_edge_parallel_epsilon: Epsilon for near-parallel edge handling in edge–edge.
        """
        if extra_matrix_blocks is None:
            extra_matrix_blocks = model.particle_count if handle_self_contact else 0
        super().__init__(
            model=model,
            dt=dt,
            mass=mass,
            preconditioner_type=preconditioner_type,
            solver_type=solver_type,
            use_constraint_contacts=use_constraint_contacts,
            contact_relaxation=contact_relaxation,
            contact_max_velocity=contact_max_velocity,
            contact_max_correction=contact_max_correction,
            contact_iterations=contact_iterations,
            linear_solver_maxiter=linear_solver_maxiter,
            extra_matrix_blocks=extra_matrix_blocks,
            ground_plane=ground_plane,
            ground_ke=ground_ke,
            ground_kd=ground_kd,
            ground_kf=ground_kf,
            ground_mu=ground_mu,
        )
        self.handle_self_contact = bool(handle_self_contact)
        self.self_contact_radius = float(self_contact_radius)
        self.self_contact_margin = (
            float(self_contact_margin)
            if self_contact_margin is not None
            else 1.5 * self.self_contact_radius
        )
        self.self_contact_stiffness = float(self_contact_stiffness)
        self.self_contact_force_cap = float(self_contact_force_cap)
        self.self_contact_edge_edge = bool(self_contact_edge_edge)
        self.self_contact_friction_mu = float(self_contact_friction_mu)
        self.self_contact_friction_epsilon = float(self_contact_friction_epsilon)

        self._surface_tri_indices = None
        self._surface_edge_indices = None
        self._num_surface_tris = 0
        self._num_surface_edges = 0
        self._trimesh_collision_detector = None

        if self.handle_self_contact:
            if self.self_contact_margin < self.self_contact_radius:
                raise ValueError(
                    "self_contact_margin must be >= self_contact_radius to avoid missing contacts. "
                    "Use self_contact_margin ~1.5–2× self_contact_radius."
                )
            self._build_surface_arrays(model)
            if self._num_surface_tris > 0:
                self._trimesh_collision_detector = TriMeshCollisionDetector(
                    model=model,
                    record_triangle_contacting_vertices=False,
                    vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                    edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                    edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
                )

    def _build_surface_arrays(self, model: Model):
        """Build surface triangle and edge arrays from model (tri or tet)."""
        tri_indices = model.tri_indices.numpy() if model.tri_indices is not None else None
        tet_indices = model.tet_indices.numpy() if model.tet_indices is not None else None
        surface_tris, surface_edges = get_surface_triangles_and_edges(
            tri_indices,
            tet_indices,
            model.tri_count,
            model.tet_count,
        )
        self._num_surface_tris = int(surface_tris.shape[0])
        self._num_surface_edges = int(surface_edges.shape[0])
        self._surface_tris_np = np.asarray(surface_tris, dtype=np.int32)
        self._surface_edges_np = np.asarray(surface_edges, dtype=np.int32)
        if self._num_surface_tris > 0:
            self._surface_tri_indices = wp.array(
                surface_tris.ravel(),
                dtype=wp.int32,
                device=model.device,
            )
        else:
            self._surface_tri_indices = wp.array(
                np.zeros(0, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
        if self._num_surface_edges > 0:
            self._surface_edge_indices = wp.array(
                surface_edges.ravel(),
                dtype=wp.int32,
                device=model.device,
            )
        else:
            self._surface_edge_indices = wp.array(
                np.zeros(0, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )

    def eval_self_contact_forces(self, model: Model, state: State, dt: float):
        """Evaluate self-contact repulsion and friction from BVH collision info (vertex–triangle and edge–edge)."""
        if (
            not self.handle_self_contact
            or self._num_surface_tris == 0
            or self._trimesh_collision_detector is None
        ):
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        col = self._trimesh_collision_detector.collision_info
        wp.launch(
            kernel=eval_self_contact_vertex_triangle_from_collision_info,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                forces,
                col.vertex_colliding_triangles,
                col.vertex_colliding_triangles_offsets,
                col.vertex_colliding_triangles_buffer_sizes,
                col.vertex_colliding_triangles_count,
                self._surface_tri_indices,
                self._num_surface_tris,
                self.self_contact_radius,
                self.self_contact_stiffness,
                self.self_contact_force_cap,
                self.self_contact_friction_mu,
                self.self_contact_friction_epsilon,
                dt,
            ],
            device=model.device,
        )
        if self.self_contact_edge_edge and self._num_surface_edges > 0:
            wp.launch(
                kernel=eval_self_contact_edge_edge_from_collision_info,
                dim=self._num_surface_edges,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    forces,
                    col.edge_colliding_edges,
                    col.edge_colliding_edges_offsets,
                    col.edge_colliding_edges_buffer_sizes,
                    col.edge_colliding_edges_count,
                    self._surface_edge_indices,
                    self._num_surface_edges,
                    self.self_contact_radius,
                    self.self_contact_stiffness,
                    self.self_contact_force_cap,
                    self.self_contact_friction_mu,
                    self.self_contact_friction_epsilon,
                    dt,
                ],
                device=model.device,
            )
        return forces

    def _add_extra_matrix_blocks(
        self, model: Model, state: State | None, dt: float, block_offset: int
    ) -> None:
        """Add lumped self-contact tangent (dt² * k_eff * I) to diagonal for vertices in contact."""
        if (
            state is None
            or not self.handle_self_contact
            or self._trimesh_collision_detector is None
            or self._num_surface_tris == 0
        ):
            return
        col = self._trimesh_collision_detector.collision_info
        wp.launch(
            kernel=build_system_matrix_self_contact_diagonal_kernel,
            dim=model.particle_count,
            inputs=[
                self.bsr_rows,
                self.bsr_cols,
                self.bsr_values,
                col.vertex_colliding_triangles_count,
                col.vertex_colliding_triangles_buffer_sizes,
                wp.float32(self.self_contact_stiffness),
                wp.float32(dt),
                wp.int32(block_offset),
            ],
            device=model.device,
        )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        """Step with optional self-contact forces added to the force sum."""
        model = self.model
        if control is None:
            control = model.control()

        # When self-contact is on, rebuild system matrix (including self-contact tangent) and preconditioner each step
        if self.handle_self_contact and self._trimesh_collision_detector is not None:
            self._trimesh_collision_detector.refit(state_in.particle_q)
            self._trimesh_collision_detector.vertex_triangle_collision_detection(
                self.self_contact_margin
            )
            if self.self_contact_edge_edge and self._num_surface_edges > 0:
                self._trimesh_collision_detector.edge_edge_collision_detection(
                    self.self_contact_margin
                )
            self._build_system_matrix(model, state_in, dt)
            self.M_bsr = preconditioner(self.A_bsr, ptype=self.preconditioner_type)

        spring_forces = self.eval_spring_forces(model, state_in)
        triangle_forces = self.eval_triangle_forces(model, control, state_in)
        triangle_contact_forces = self.eval_triangle_contact_forces(model, control, state_in)
        bending_forces = self.eval_bending_forces(model, control, state_in)
        tetrahedral_forces = self.eval_tetrahedral_forces(model, control, state_in)
        particle_forces = self.eval_particle_particle_forces(model, control, state_in)
        particle_ground_contact_forces = self.eval_particle_ground_contact_forces(
            model, control, state_in
        )
        soft_contact_forces = self.eval_soft_contact_forces(model, state_in, contacts)
        constraint_forces = self.eval_constraints(model, control, state_in)
        gravity_forces = self.eval_gravity_forces(model)
        # Self-contact forces (refit/detection already done above when building matrix)
        self_contact_forces = self.eval_self_contact_forces(model, state_in, dt)

        if self.use_constraint_contacts:
            soft_contact_forces = wp.zeros(
                model.particle_count, dtype=wp.vec3, device=model.device
            )
        # RHS = dt * (all forces); implicit solve in SolverSoft.implicit_integration.
        state_in.particle_f = dt * (
            spring_forces
            + tetrahedral_forces
            + triangle_forces
            + triangle_contact_forces
            + bending_forces
            + particle_forces
            + particle_ground_contact_forces
            + soft_contact_forces
            + gravity_forces
            + constraint_forces
            + self_contact_forces
        )

        self.implicit_integration(model, state_in, state_out, dt)

        if self.use_constraint_contacts and contacts is not None:
            self.apply_constraint_contact_corrections(
                model, state_in, state_out, contacts, dt
            )

        if model.body_count:
            gravity_array = self._get_gravity_array(model)
            original_gravity = model.gravity
            model.gravity = gravity_array
            self.integrate_bodies(model, state_in, state_out, dt, 0.0)
            model.gravity = original_gravity

        return state_out

    def rebuild_bvh(self, state: State):
        """Rebuild BVHs for self-contact from state.particle_q. Call after large deformation."""
        if self.handle_self_contact and self._trimesh_collision_detector is not None:
            self._trimesh_collision_detector.rebuild(state.particle_q)
