# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Rigid-Soft Body Interaction Example

Demonstrates a rigid ball interacting with a soft (FEM) ball.
Choose between MuJoCo or XPBD for the rigid body solver.

Usage:
    python -m newton.examples.soft.example_rigid_soft_interaction --solver xpbd
    python -m newton.examples.soft.example_rigid_soft_interaction --solver mujoco
"""

import warp as wp
import numpy as np
import argparse
import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverSoft, SolverXPBD, FEMSphere


# Colors
COLOR_GREEN = (0.2, 0.9, 0.3)    # Rigid ball
COLOR_BLUE = (0.3, 0.5, 1.0)     # Soft ball


class RigidSoftInteractionExample:
    """Rigid-soft interaction with selectable rigid body solver."""
    
    def __init__(
        self,
        viewer,
        solver_type: str = "xpbd",  # "mujoco" or "xpbd"
        ball_radius: float = 0.3,
        drop_height: float = 1.5,
        ball_subdivisions: int = 2,
        ball_interior_layers: int = 2,
        soft_ball_mass: float = 1.0,
        rigid_ball_mass: float = 2.0,
        k_mu: float = 4e4,
        k_lambda: float = 4e4,
        k_damp: float = 3.0,
        gravity: float = 9.81,
        substeps: int = 16,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.solver_type = solver_type.lower()
        
        x_spacing = ball_radius * 2.5
        
        print("=" * 70)
        print(f"  Rigid-Soft Interaction ({self.solver_type.upper()} solver)")
        print("=" * 70)
        print(f"\n  GREEN ball: Rigid ({self.solver_type.upper()})")
        print(f"  BLUE ball: Soft/FEM")
        
        # Generate FEM mesh
        print(f"\nGenerating FEM mesh...")
        self.sphere = FEMSphere(
            radius=ball_radius,
            subdivisions=ball_subdivisions,
            interior_layers=ball_interior_layers,
            verbose=True,
        )
        mesh_data = self.sphere.get_mesh_data()
        
        # Build model
        builder = newton.ModelBuilder()
        if self.solver_type == "mujoco":
            SolverMuJoCo.register_custom_attributes(builder)
        
        # Ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=5e5, kd=1e3, kf=1e4, mu=0.5)
        )
        
        # Soft ball (BLUE) - on ground at center
        print(f"Adding soft ball (BLUE)...")
        vertices = mesh_data["vertices"]
        indices = mesh_data["indices"]
        
        # Position soft ball at center, just above ground
        soft_height = ball_radius * 1.1  # Slightly above ground
        vertices = [(v[0], v[1], v[2] + soft_height) for v in vertices]
        
        # Rigid ball (GREEN) - falling from above, directly over soft ball
        print(f"\nAdding rigid ball (GREEN)...")
        self.rigid_body_id = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, drop_height), wp.quat_identity())
        )
        joint_id = builder.add_joint_free(self.rigid_body_id)
        builder.add_articulation([joint_id], key="rigid_ball")
        self.rigid_shape_id = builder.add_shape_sphere(
            body=self.rigid_body_id,
            radius=ball_radius,
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5, kd=100.0, kf=1e4, mu=0.5,  # Bouncy
                density=rigid_ball_mass / ((4/3) * np.pi * ball_radius**3),
            )
        )
        
        start_particle = builder.particle_count
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            scale=1.0,
            density=soft_ball_mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        
        # Add springs for stability
        spring_ke = k_mu * 0.5
        spring_kd = k_damp * 0.5
        num_tets = len(indices) // 4
        vertex_positions = np.array(vertices)
        added_springs = set()
        
        for tet_idx in range(num_tets):
            base = tet_idx * 4
            tet_verts = [indices[base + i] for i in range(4)]
            for i_local in range(4):
                for j_local in range(i_local + 1, 4):
                    a, b = tet_verts[i_local], tet_verts[j_local]
                    if a > b:
                        a, b = b, a
                    spring_key = (a, b)
                    if spring_key not in added_springs:
                        added_springs.add(spring_key)
                        p0, p1 = vertex_positions[a], vertex_positions[b]
                        rest_len = float(np.linalg.norm(p1 - p0))
                        builder.add_spring(start_particle + a, start_particle + b,
                                           spring_ke, spring_kd, 0.0)
        
        # Finalize model
        self.model = builder.finalize()
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        
        # Soft contact parameters (controls force from soft body onto rigid body)
        self.model.soft_contact_ke = 5e4
        self.model.soft_contact_kd = 2000.0
        self.model.soft_contact_kf = 1e4
        self.model.soft_contact_mu = 0.8
        
        # Particle rendering
        if self.model.particle_count > 0:
            self.model.particle_radius = wp.array(
                np.full(self.model.particle_count, 0.015),
                dtype=wp.float32, device=self.model.device
            )
        
        print(f"\nModel: {self.model.body_count} bodies, {self.model.particle_count} particles")
        
        # Create solvers
        print(f"\n--- Creating Solvers ---")
        
        print(f"Creating SolverSoft...")
        self.soft_solver = SolverSoft(
            model=self.model, dt=self.sim_dt, mass=soft_ball_mass, solver_type="bicgstab"
        )
        
        if self.solver_type == "mujoco":
            print(f"Creating SolverMuJoCo...")
            self.rigid_solver = SolverMuJoCo(
                self.model,
                use_mujoco_contacts=True,  # Use MuJoCo's own collision detection
                ls_parallel=True,
                njmax=50,
            )
        else:
            print(f"Creating SolverXPBD...")
            self.rigid_solver = SolverXPBD(self.model)
        
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_soft = self.model.state()
        self.state_rigid = self.model.state()
        self.control = self.model.control()
        
        # Initialize
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.collide(self.state_0)
        
        # Viewer setup
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True  # Enable soft body visualization
            self.viewer.update_shape_colors({
                self.rigid_shape_id: COLOR_GREEN,
            })
        
        print(f"\n  Solver: {self.solver_type.upper()}")
        print(f"  RED:  Rigid ball")
        print(f"  BLUE: Soft ball")
    
    def step(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(state=self.state_0)
            
            # Soft solver for particles
            self.soft_solver.step(
                self.state_0, self.state_soft, self.control, self.contacts, self.sim_dt
            )
            
            # Rigid solver for bodies
            self.rigid_solver.step(
                self.state_0, self.state_rigid, self.control, self.contacts, self.sim_dt
            )
            
            # Combine: particles from soft, bodies from rigid
            wp.copy(self.state_1.particle_q, self.state_soft.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_soft.particle_qd)
            wp.copy(self.state_1.body_q, self.state_rigid.body_q)
            wp.copy(self.state_1.body_qd, self.state_rigid.body_qd)
            wp.copy(self.state_1.joint_q, self.state_rigid.joint_q)
            wp.copy(self.state_1.joint_qd, self.state_rigid.joint_qd)
            
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
    
    def render(self):
        if self.viewer:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            if self.contacts:
                self.viewer.log_contacts(self.contacts, self.state_0)
            self.viewer.end_frame()
    
    def run(self, num_frames: int = 600):
        print(f"\nRunning {num_frames} frames...")
        for frame in range(num_frames):
            self.step()
            self.render()
            
            if frame % 100 == 0:
                print(f"  Frame {frame}/{num_frames}, sim_time={self.sim_time:.2f}s")
        
        print(f"\nSimulation complete!")


def main():
    parser = argparse.ArgumentParser(description="Rigid-Soft Interaction")
    parser.add_argument("--solver", type=str, default="xpbd", choices=["mujoco", "xpbd"],
                        help="Rigid body solver: 'mujoco' or 'xpbd' (default: xpbd)")
    parser.add_argument("--ball-radius", type=float, default=0.3)
    parser.add_argument("--drop-height", type=float, default=1.5)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    # Initialize Warp
    wp.init()
    
    # Create viewer
    if args.headless:
        viewer = None
    else:
        viewer = newton.viewer.ViewerGL()
    
    # Create and run example
    example = RigidSoftInteractionExample(
        viewer=viewer,
        solver_type=args.solver,
        ball_radius=args.ball_radius,
        drop_height=args.drop_height,
        substeps=args.substeps,
    )
    
    example.run(num_frames=args.num_frames)
    
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
