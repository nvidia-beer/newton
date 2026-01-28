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

"""
Bouncing Ball Example

A simple soft sphere dropped from height - bounces like a basketball.
Uses FEM tetrahedral mesh with implicit integration.

Usage:
    python -m newton.examples.soft.example_bouncing_ball
    python -m newton.examples.soft.example_bouncing_ball --initial_height 3.0 --k_mu 5e5
"""

import warp as wp
import numpy as np
import argparse

import newton
from newton.solvers import SolverSoft, FEMSphere


class Example:
    """
    Simple bouncing ball - drop and watch it bounce!
    
    Uses FEMSphere to generate a tetrahedral mesh and SolverSoft
    for implicit integration with sparse matrix solvers.
    """
    
    def __init__(
        self,
        viewer,
        radius: float = 0.3,
        subdivisions: int = 2,
        interior_layers: int = 2,
        initial_height: float = 2.5,
        mass: float = 0.5,
        k_mu: float = 2e5,
        k_lambda: float = 2e5,
        k_damp: float = 0.5,
        gravity: float = 9.81,
        substeps: int = 8,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.radius = radius
        self.initial_height = initial_height
        self.mass = mass
        
        self.viewer = viewer
        
        # Generate sphere mesh
        print(f"Creating bouncing ball...", flush=True)
        self.sphere = FEMSphere(
            radius=radius,
            subdivisions=subdivisions,
            interior_layers=interior_layers
        )
        self.sphere.info()
        
        # Get mesh data
        mesh_data = self.sphere.get_mesh_data()
        vertices = mesh_data['vertices']
        indices = mesh_data['indices']
        
        # Build Newton model
        builder = newton.ModelBuilder()
        
        # Add bouncy ground plane at Z=0
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,   # High stiffness for bounce
                kd=1e3,   # Some damping
                kf=1e4,   # Friction stiffness
                mu=0.5    # Friction coefficient
            )
        )
        
        # Track particle start index for springs
        start_particle = builder.particle_count
        
        # Add soft ball - positioned above ground (Z is up)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, initial_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            scale=1.0,
            density=mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        
        # Add springs between mesh vertices for stability
        spring_ke = k_mu * 0.5
        spring_kd = k_damp * 0.5
        num_tets = len(indices) // 4
        
        vertex_positions = np.array(vertices)
        added_springs = set()
        
        for t in range(num_tets):
            tet_indices = [indices[t * 4 + k] for k in range(4)]
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for ei, ej in edges:
                i_local, j_local = tet_indices[ei], tet_indices[ej]
                if i_local > j_local:
                    i_local, j_local = j_local, i_local
                
                spring_key = (i_local, j_local)
                if spring_key not in added_springs:
                    added_springs.add(spring_key)
                    
                    p0 = vertex_positions[i_local]
                    p1 = vertex_positions[j_local]
                    rest_length = float(np.linalg.norm(p1 - p0))
                    
                    builder.add_spring(
                        start_particle + i_local,
                        start_particle + j_local,
                        spring_ke,
                        spring_kd,
                        rest_length
                    )
        
        print(f"Added {len(added_springs)} unique springs")
        
        self.model = builder.finalize()
        
        # Set gravity (Z is up, gravity pulls down)
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        
        # Contact parameters tuned for realistic bounce
        self.model.soft_contact_ke = 5e4
        self.model.soft_contact_kd = 2000.0
        self.model.soft_contact_kf = 1e4
        self.model.soft_contact_mu = 0.8
        
        # Particle rendering radius
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.015),
            dtype=wp.float32,
            device=self.model.device
        )
        
        # Create solver with implicit integration
        self.solver = SolverSoft(
            model=self.model,
            dt=self.sim_dt,
            mass=mass,
            solver_type="bicgstab"
        )
        
        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        
        # Initialize FK
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0
        )
        
        # Set up viewer
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True  # Enable soft body visualization
        
        # Tracking variables
        self.bounce_count = 0
        self.was_falling = True
        self.max_height = 0.0
        
        print(f"\n🏀 Bouncing Ball Ready!", flush=True)
        print(f"   Dropping from height: {initial_height}m", flush=True)
        print(f"   Ball radius: {radius}m", flush=True)
        print(f"   Stiffness: μ={k_mu:.0e}, λ={k_lambda:.0e}", flush=True)
    
    def step(self):
        """Run one frame of simulation."""
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            
            # Collision detection
            self.contacts = self.model.collide(state=self.state_0)
            
            # Physics step
            self.solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
    
    def render(self):
        """Render current frame."""
        if self.viewer is None:
            return
            
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
    
    def run(self, num_frames: int = 600):
        """Run simulation loop."""
        print(f"\n🏀 Dropping the ball...", flush=True)
        
        for frame in range(num_frames):
            self.step()
            self.render()
            
            # Track ball position (Z is up)
            positions = self.state_0.particle_q.numpy()
            center = positions.mean(axis=0)
            height = center[2]
            
            # Detect bounces
            velocities = self.state_0.particle_qd.numpy()
            avg_vel_z = velocities.mean(axis=0)[2]
            
            is_falling = avg_vel_z < 0
            if self.was_falling and not is_falling and height < self.initial_height * 0.8:
                self.bounce_count += 1
                print(f"   🏀 Bounce #{self.bounce_count} at height {height:.2f}m", flush=True)
            self.was_falling = is_falling
            
            self.max_height = max(self.max_height, height)
            
            if frame % 60 == 0:
                print(f"   Frame {frame}: height={height:.2f}m, vel={avg_vel_z:.2f}m/s", flush=True)
        
        print(f"\n🏀 Simulation complete!", flush=True)
        print(f"   Total bounces: {self.bounce_count}", flush=True)
        print(f"   Max height reached: {self.max_height:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Bouncing Ball Simulation')
    
    # Ball parameters
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Ball radius (default: 0.3)')
    parser.add_argument('--subdivisions', type=int, default=2,
                        help='Mesh detail 1-3 (default: 2)')
    parser.add_argument('--interior_layers', type=int, default=2,
                        help='Interior layers (default: 2)')
    
    # Physics parameters
    parser.add_argument('--initial_height', type=float, default=2.5,
                        help='Drop height (default: 2.5)')
    parser.add_argument('--mass', type=float, default=0.5,
                        help='Ball mass (default: 0.5)')
    parser.add_argument('--k_mu', type=float, default=2e5,
                        help='Shear modulus - higher = stiffer (default: 2e5)')
    parser.add_argument('--k_lambda', type=float, default=2e5,
                        help='Bulk modulus - higher = less compressible (default: 2e5)')
    parser.add_argument('--k_damp', type=float, default=0.5,
                        help='Damping - higher = less bouncy (default: 0.5)')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravity (default: 9.81)')
    
    # Simulation parameters
    parser.add_argument('--substeps', type=int, default=8,
                        help='Substeps per frame (default: 8)')
    parser.add_argument('--num_frames', type=int, default=600,
                        help='Number of frames (default: 600)')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization')
    
    args = parser.parse_args()
    
    wp.init()
    
    with wp.ScopedDevice(args.device):
        # Create viewer - use OpenGL viewer without explicit size (use defaults like other examples)
        if args.headless:
            viewer = None
        else:
            try:
                # Use ViewerGL with default size (matches other examples)
                viewer = newton.viewer.ViewerGL()
            except Exception as e:
                print(f"Could not create OpenGL viewer: {e}")
                try:
                    # Fall back to Rerun
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                except Exception as e2:
                    print(f"Could not create Rerun viewer: {e2}")
                    print("Running headless...")
                    viewer = None
        
        example = Example(
            viewer=viewer,
            radius=args.radius,
            subdivisions=args.subdivisions,
            interior_layers=args.interior_layers,
            initial_height=args.initial_height,
            mass=args.mass,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            gravity=args.gravity,
            substeps=args.substeps,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
