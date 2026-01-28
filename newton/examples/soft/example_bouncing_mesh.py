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
Bouncing Mesh Example

A soft body mesh dropped from height - uses FEM tetrahedral simulation.
Loads tetrahedral meshes from .mesh (Medit) format files.

Usage:
    python -m newton.examples.soft.example_bouncing_mesh --mesh_file tutorial/spot.mesh
    python -m newton.examples.soft.example_bouncing_mesh --mesh_file tutorial/spot.mesh --scale 0.5 --k_mu 1e5
"""

import warp as wp
import numpy as np
import argparse
import os

import newton
from newton.solvers import SolverSoft


def load_mesh(filename):
    """Load mesh file (.mesh Medit format)."""
    print(f"Loading mesh from: {filename}", flush=True)
    
    vertices = []
    tetras = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "Vertices":
            i += 1
            num_verts = int(lines[i].strip())
            i += 1
            for _ in range(num_verts):
                parts = lines[i].strip().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
        elif line == "Tetrahedra":
            i += 1
            num_tets = int(lines[i].strip())
            i += 1
            for _ in range(num_tets):
                parts = lines[i].strip().split()
                tetras.append([int(parts[0])-1, int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                i += 1
        else:
            i += 1
    
    vertices = np.array(vertices)
    tetras = np.array(tetras)
    
    if len(tetras) == 0:
        raise ValueError("No tetrahedra found in mesh file!")
    return vertices, tetras


def normalize_mesh(vertices):
    """Normalize mesh vertices by scaling to unit size."""
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    max_extent = np.max(extents)
    scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0
    # Only scale, don't center - centering is handled by add_soft_mesh position
    vertices_scaled = vertices * scale_factor
    return vertices_scaled, scale_factor, extents


class Example:
    """
    Bouncing mesh - drop a soft body mesh and watch it bounce!
    
    Loads tetrahedral mesh from file and uses SolverSoft
    for implicit integration with sparse matrix solvers.
    """
    
    def __init__(
        self,
        viewer,
        mesh_file: str,
        scale: float = 0.3,
        initial_height: float = 1.0,
        mass: float = 2.0,
        k_mu: float = 1.0e6,         # Tetrahedral shear modulus
        k_lambda: float = 1.0e6,     # Tetrahedral bulk modulus
        k_damp: float = 1.0,         # Tetrahedral damping
        spring_ke: float = 1.0e5,    # Spring stiffness
        spring_kd: float = 1.0,      # Spring damping
        gravity: float = 9.81,
        substeps: int = 5,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.scale = scale
        self.initial_height = initial_height
        self.mass = mass
        self.mesh_file = mesh_file
        
        self.viewer = viewer
        
        # Load mesh from file
        vertices_raw, tetras = load_mesh(mesh_file)
        print(f"Raw mesh: {len(vertices_raw)} vertices, {len(tetras)} tetrahedra", flush=True)
        
        # Normalize and scale the mesh
        vertices_normalized, auto_scale, extents = normalize_mesh(vertices_raw)
        vertices = (vertices_normalized * scale).astype(np.float32)
        
        # Flatten tetrahedra indices for Newton
        indices = tetras.flatten().astype(np.int32)
        
        print(f"Original extents: {extents}", flush=True)
        print(f"Scaled size: {scale:.2f}m", flush=True)
        
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
        
        # Add soft mesh - positioned above ground (Z is up)
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
        num_tets = len(tetras)
        
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
        
        print(f"Added {len(added_springs)} unique springs", flush=True)
        
        self.model = builder.finalize()
        
        print(f"\nModel created:", flush=True)
        print(f"  Particles: {self.model.particle_count}", flush=True)
        print(f"  Springs: {self.model.spring_count}", flush=True)
        print(f"  Tetrahedra: {self.model.tet_count}", flush=True)
        print(f"  Triangles: {self.model.tri_count}", flush=True)
        
        # Set gravity (Z is up, gravity pulls down)
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        
        # Contact parameters (from working example)
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 500.0
        self.model.soft_contact_kf = 5.0e4
        self.model.soft_contact_mu = 0.9
        
        # Particle constraint parameters
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        
        # Particle rendering radius
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.008),
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
        
        mesh_name = os.path.basename(mesh_file)
        print(f"\n🎎 Bouncing Mesh Ready!", flush=True)
        print(f"   Mesh: {mesh_name}", flush=True)
        print(f"   Dropping from height: {initial_height}m", flush=True)
        print(f"   Scale: {scale}m", flush=True)
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
    
    def run(self, num_frames: int = 300):
        """Run simulation loop."""
        print(f"\n🎎 Dropping the mesh...", flush=True)
        
        for frame in range(num_frames):
            self.step()
            self.render()
            
            # Track mesh position (Z is up)
            positions = self.state_0.particle_q.numpy()
            center = positions.mean(axis=0)
            height = center[2]
            
            # Detect bounces
            velocities = self.state_0.particle_qd.numpy()
            avg_vel_z = velocities.mean(axis=0)[2]
            
            is_falling = avg_vel_z < 0
            if self.was_falling and not is_falling and height < self.initial_height * 0.8:
                self.bounce_count += 1
                print(f"   🎎 Bounce #{self.bounce_count} at height {height:.2f}m", flush=True)
            self.was_falling = is_falling
            
            self.max_height = max(self.max_height, height)
            
            if frame % 60 == 0:
                print(f"   Frame {frame}: height={height:.2f}m, vel={avg_vel_z:.2f}m/s", flush=True)
        
        print(f"\n🎎 Simulation complete!", flush=True)
        print(f"   Total bounces: {self.bounce_count}", flush=True)
        print(f"   Max height reached: {self.max_height:.2f}m", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Bouncing Mesh Simulation')
    
    # Mesh parameters
    parser.add_argument('--mesh_file', type=str, default='tutorial/spot.mesh',
                        help='Path to .mesh file (Medit format)')
    parser.add_argument('--scale', type=float, default=0.3,
                        help='Mesh scale factor (default: 0.3)')
    
    # Physics parameters
    parser.add_argument('--initial_height', type=float, default=0.5,
                        help='Drop height (default: 0.5)')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Mesh mass (default: 1.0)')
    parser.add_argument('--k_mu', type=float, default=1.0e6,
                        help='Shear modulus (default: 1.0e6)')
    parser.add_argument('--k_lambda', type=float, default=1.0e6,
                        help='Bulk modulus (default: 1.0e6)')
    parser.add_argument('--k_damp', type=float, default=1.0,
                        help='Damping (default: 1.0)')
    parser.add_argument('--spring_ke', type=float, default=1.0e5,
                        help='Spring stiffness (default: 1.0e5)')
    parser.add_argument('--spring_kd', type=float, default=1.0,
                        help='Spring damping (default: 1.0)')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravity (default: 9.81)')
    
    # Simulation parameters
    parser.add_argument('--substeps', type=int, default=5,
                        help='Substeps per frame (default: 12)')
    parser.add_argument('--num_frames', type=int, default=300,
                        help='Number of frames (default: 300)')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization')
    
    args = parser.parse_args()
    
    # Check mesh file exists
    if not os.path.exists(args.mesh_file):
        print(f"Error: Mesh file not found: {args.mesh_file}")
        return
    
    wp.init()
    
    with wp.ScopedDevice(args.device):
        # Create viewer - use OpenGL for native window display
        if args.headless:
            viewer = None
        else:
            try:
                # Try OpenGL viewer first (shows in DCV/X11)
                viewer = newton.viewer.ViewerGL(
                    width=1024,
                    height=768,
                )
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
            mesh_file=args.mesh_file,
            scale=args.scale,
            initial_height=args.initial_height,
            mass=args.mass,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            gravity=args.gravity,
            substeps=args.substeps,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()

