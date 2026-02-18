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
Bouncing Box Example

A simple soft box dropped from height - bounces and tumbles.
Uses FEM tetrahedral mesh with implicit integration.

Usage:
    python -m newton.examples.soft.example_bouncing_box
    python -m newton.examples.soft.example_bouncing_box --initial_height 3.0 --k_mu 5e5
"""

import warp as wp
import numpy as np
import argparse

import newton
from newton.solvers import SolverSoft, TetraBox


class Example:
    """
    Simple bouncing box - drop and watch it bounce and tumble!
    
    Uses TetraBox to generate a tetrahedral mesh and SolverSoft
    for implicit integration with sparse matrix solvers.
    """
    
    def __init__(
        self,
        viewer,
        size=(0.6, 0.8, 1.0),  # Larger non-uniform box: width=0.6, height=0.8, depth=1.0
        subdivisions=(3, 5, 7),  # Subdivisions per axis (non-uniform: width=3, height=5, depth=7)
        initial_height: float = 2.5,
        mass: float = 0.5,
        k_mu: float = 2e5,
        k_lambda: float = 2e5,
        k_damp: float = 0.5,
        gravity: float = 9.81,
        substeps: int = 8,
    ):
        self.fps = 60
        # Auto-adjust parameters for coarse meshes and non-uniform meshes
        # Coarse meshes and non-uniform meshes need lower stiffness and higher damping for stability
        total_elements = subdivisions[0] * subdivisions[1] * subdivisions[2]
        max_segment = max(subdivisions)
        min_segment = min(subdivisions)
        is_uniform = subdivisions[0] == subdivisions[1] == subdivisions[2]
        aspect_ratio = max_segment / min_segment if min_segment > 0 else 1.0
        
        if subdivisions == (1, 1, 1):
            if k_mu >= 1e5:  # Only adjust if using default high stiffness
                k_mu = k_mu / 4  # Reduce stiffness 4x
                k_lambda = k_lambda / 4
                k_damp = k_damp * 4  # Increase damping 4x
                if substeps < 16:
                    substeps = 16  # More substeps for stability
                print(f"  Auto-adjusted parameters for coarse mesh (subdivisions=1,1,1):")
                print(f"    k_mu={k_mu:.0e}, k_lambda={k_lambda:.0e}, k_damp={k_damp:.1f}, substeps={substeps}")
        elif min_segment <= 2 and max_segment <= 4:
            # For coarse meshes (any dimension <= 2, max <= 4), adjust parameters
            # Non-uniform meshes get adjusted based on coarseness
            if k_mu >= 1e5:  # Only adjust if using default high stiffness
                # More aggressive adjustment for very coarse meshes
                if min_segment == 1:
                    k_mu = k_mu / 3  # Reduce stiffness 3x
                    k_lambda = k_lambda / 3
                    k_damp = k_damp * 3  # Increase damping 3x
                    if substeps < 14:
                        substeps = 14
                else:  # min_segment == 2
                    k_mu = k_mu / 2  # Reduce stiffness 2x
                    k_lambda = k_lambda / 2
                    k_damp = k_damp * 2  # Increase damping 2x
                    if substeps < 12:
                        substeps = 12
                print(f"  Auto-adjusted parameters for coarse mesh (subdivisions={subdivisions[0]},{subdivisions[1]},{subdivisions[2]}):")
                print(f"    k_mu={k_mu:.0e}, k_lambda={k_lambda:.0e}, k_damp={k_damp:.1f}, substeps={substeps}")
        elif not is_uniform and aspect_ratio > 1.5:
            # For non-uniform meshes with significant aspect ratio, adjust parameters
            # Non-uniform hexahedra create tetrahedra with varying aspect ratios, causing instability
            if k_mu >= 1e5:  # Only adjust if using default high stiffness
                # Adjust based on aspect ratio - more non-uniform = more adjustment needed
                if aspect_ratio >= 2.0:
                    # Very non-uniform: reduce stiffness more aggressively
                    k_mu = k_mu / 2.0  # Reduce stiffness 2x
                    k_lambda = k_lambda / 2.0
                    k_damp = k_damp * 2.0  # Increase damping 2x
                    if substeps < 12:
                        substeps = 12
                else:
                    # Moderately non-uniform
                    k_mu = k_mu / 1.5  # Reduce stiffness 1.5x
                    k_lambda = k_lambda / 1.5
                    k_damp = k_damp * 1.5  # Increase damping 1.5x
                    if substeps < 10:
                        substeps = 10
                print(f"  Auto-adjusted parameters for non-uniform mesh (subdivisions={subdivisions[0]},{subdivisions[1]},{subdivisions[2]}, aspect_ratio={aspect_ratio:.2f}):")
                print(f"    k_mu={k_mu:.0e}, k_lambda={k_lambda:.0e}, k_damp={k_damp:.1f}, substeps={substeps}")
        
        # Print mesh info
        is_uniform = subdivisions[0] == subdivisions[1] == subdivisions[2]
        mesh_type = "uniform" if is_uniform else "non-uniform"
        print(f"  Using {mesh_type} subdivided mesh: {subdivisions[0]}×{subdivisions[1]}×{subdivisions[2]} = "
              f"{total_elements} cubic elements ({total_elements * 6} tetrahedra)")
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        if isinstance(size, (int, float)):
            self.size = (float(size), float(size), float(size))
        else:
            self.size = tuple(float(s) for s in size)
        self.initial_height = initial_height
        self.mass = mass
        
        self.viewer = viewer
        
        # Generate box mesh
        print(f"Creating bouncing box...", flush=True)
        self.box = TetraBox(
            size=self.size,
            subdivisions=subdivisions,
        )
        self.box.info()
        
        # Get mesh data
        mesh_data = self.box.get_mesh_data()
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
        # Add soft box - positioned above ground (Z is up)
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
        
        print(f"\n📦 Bouncing Box Ready!", flush=True)
        print(f"   Dropping from height: {initial_height}m", flush=True)
        print(f"   Box size: {self.size}", flush=True)
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
        print(f"\n📦 Dropping the box...", flush=True)
        
        for frame in range(num_frames):
            self.step()
            self.render()
            
            # Track box position (Z is up)
            positions = self.state_0.particle_q.numpy()
            center = positions.mean(axis=0)
            height = center[2]
            
            # Detect bounces
            velocities = self.state_0.particle_qd.numpy()
            avg_vel_z = velocities.mean(axis=0)[2]
            
            is_falling = avg_vel_z < 0
            if self.was_falling and not is_falling and height < self.initial_height * 0.8:
                self.bounce_count += 1
                print(f"   📦 Bounce #{self.bounce_count} at height {height:.2f}m", flush=True)
            self.was_falling = is_falling
            
            self.max_height = max(self.max_height, height)
            
            if frame % 60 == 0:
                print(f"   Frame {frame}: height={height:.2f}m, vel={avg_vel_z:.2f}m/s", flush=True)
        
        print(f"\n📦 Simulation complete!", flush=True)
        print(f"   Total bounces: {self.bounce_count}", flush=True)
        print(f"   Max height reached: {self.max_height:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Bouncing Box Simulation')
    
    # Box parameters
    parser.add_argument('--size', type=float, nargs=3, default=[0.6, 0.8, 1.0],
                        help='Box size (width, height, depth) (default: 0.6 0.8 1.0 - larger non-uniform)')
    parser.add_argument('--subdivisions', type=int, nargs=3, default=[3, 5, 7],
                        help='Subdivisions per axis (default: 3 5 7 - non-uniform: width×height×depth)')
    
    # Physics parameters
    parser.add_argument('--initial_height', type=float, default=2.5,
                        help='Drop height (default: 2.5)')
    parser.add_argument('--mass', type=float, default=0.5,
                        help='Box mass (default: 0.5)')
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
        # Create viewer
        if args.headless:
            viewer = None
        else:
            try:
                viewer = newton.viewer.ViewerGL(
                    width=1920,
                    height=1080,
                )
            except Exception as e:
                print(f"Could not create OpenGL viewer: {e}")
                try:
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                except Exception as e2:
                    print(f"Could not create Rerun viewer: {e2}")
                    print("Running headless...")
                    viewer = None
        
        example = Example(
            viewer=viewer,
            size=tuple(args.size),
            subdivisions=tuple(args.subdivisions),
            initial_height=args.initial_height,
            mass=args.mass,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            gravity=args.gravity,
            substeps=args.substeps,
        )
        
        # Render initial frame
        if viewer:
            example.render()
        
        # Run simulation
        example.run(num_frames=args.num_frames)
        
        # Keep viewer open
        if viewer:
            print("\nSimulation complete - viewer window will stay open")
            print("Press ESC or close the window to exit")
            while viewer.is_running():
                viewer.end_frame()
            viewer.close()


if __name__ == "__main__":
    main()
