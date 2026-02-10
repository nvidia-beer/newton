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
    python -m newton.examples.soft.example_bouncing_mesh
    python -m newton.examples.soft.example_bouncing_mesh --mesh_file examples/assets/spot.mesh --scale 0.5 --k_mu 1e5
"""

import warp as wp
import numpy as np
import argparse
import os

import newton
import newton.examples
from newton.solvers import SolverSoft
from newton.utils import load_tetrahedral_mesh


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
        scale: float = 1.0,  # Scale parameter for add_soft_mesh (default 1.0 for pre-processed meshes)
        initial_height: float = 0.2,
        mass: float = 1.0,  # Match old working example
        k_mu: float = 5.0,        # Tetrahedral shear modulus (match old working example)
        k_lambda: float = 5.0,    # Tetrahedral bulk modulus (match old working example)
        k_damp: float = 40.0,     # Tetrahedral damping (match old working example)
        spring_ke: float = 50.0,  # Spring stiffness (match old working example)
        spring_kd: float = 40.0,  # Spring damping (match old working example)
        gravity: float = 9.81,
        substeps: int = 20,        # Increased for stability with complex meshes
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
        self.k_mu = k_mu
        self.k_lambda = k_lambda
        self.spring_ke = spring_ke
        self.spring_kd = spring_kd
        
        self.viewer = viewer
        
        # Load mesh from file - assume it's already pre-processed (centered, fixed winding)
        # Use create-spot-fixed-mesh.sh to pre-process meshes
        print(f"Loading tetrahedral mesh from: {mesh_file}", flush=True)
        vertices, tetras = load_tetrahedral_mesh(mesh_file)
        print(f"Loaded mesh: {len(vertices)} vertices, {len(tetras)} tetrahedra", flush=True)
        
        # Validate mesh data
        if len(vertices) == 0:
            raise ValueError("Mesh has no vertices!")
        if len(tetras) == 0:
            raise ValueError("Mesh has no tetrahedra!")
        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
            raise ValueError("Mesh vertices contain NaN/Inf values!")
        
        # Check for valid tetrahedra indices
        max_idx = len(vertices) - 1
        if np.any(tetras < 0) or np.any(tetras > max_idx):
            raise ValueError(f"Tetrahedra indices out of range! Max index: {max_idx}")
        
        # CRITICAL DEBUG: Check mesh size BEFORE any processing
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = np.max(extents)
        min_z = vertices[:, 2].min()
        max_z = vertices[:, 2].max()
        final_size = max_extent * scale
        
        print(f"\n🔍 MESH SIZE DEBUG:", flush=True)
        print(f"   Mesh file: {mesh_file}", flush=True)
        print(f"   Extents: {extents}, max: {max_extent:.3f}m", flush=True)
        print(f"   Z range: [{min_z:.3f}, {max_z:.3f}]", flush=True)
        print(f"   Scale parameter: {scale}", flush=True)
        print(f"   Expected final size: {final_size:.3f}m", flush=True)
        
        # Check mesh size - spot_fixed.mesh preserves original scale (~4254m), which is OK
        # Use scale parameter to reduce mesh size in simulation (preserves stable rest volumes)
        if max_extent > 10000.0:
            print(f"\n❌ ERROR: Mesh is extremely large ({max_extent:.1f}m)!", flush=True)
            print(f"   This mesh likely needs preprocessing.", flush=True)
            print(f"\n   SOLUTION:", flush=True)
            print(f"   1. Create spot_fixed.mesh by running:", flush=True)
            print(f"      ./newton/.devcontainer/create-spot-fixed-mesh.sh", flush=True)
            print(f"   2. Or use an already-processed mesh like ball.mesh", flush=True)
            print(f"\n   Current mesh file: {mesh_file}", flush=True)
            raise ValueError(f"Mesh too large ({max_extent:.1f}m) - likely not pre-processed")
        
        if final_size > 100.0:
            print(f"   ℹ INFO: Final mesh size ({final_size:.2f}m) - using scale {scale:.2f}", flush=True)
            print(f"   This preserves stable rest volumes for FEM simulation", flush=True)
        elif final_size > 10.0:
            print(f"   ℹ INFO: Final mesh size ({final_size:.2f}m) - using scale {scale:.2f}", flush=True)
        
        # Convert to float32 and use directly (mesh should be pre-processed)
        vertices = vertices.astype(np.float32)
        
        # Flatten tetrahedra indices for Newton
        indices = tetras.flatten().astype(np.int32)
        print(f"✓ Using all {len(tetras)} tetrahedra", flush=True)
        
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
        
        # Add soft mesh - use loaded vertices directly with user's scale parameter
        # spot_fixed.mesh is normalized to 10m, so use scale=0.1 to get 1m mesh
        # No need to scale stiffness - rest configuration is computed from scaled vertices
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, initial_height),  # Initial height above ground
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,  # Pre-processed mesh vertices (normalized to 10m)
            indices=indices,
            scale=scale,  # User's scale parameter (use 0.1 for 1m mesh)
            density=mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )
        
        # Add springs between mesh vertices for stability (match old working example)
        # Use independent spring parameters, not derived from tetrahedral parameters
        num_tets = len(indices) // 4
        
        vertex_positions = np.array(vertices)  # Use original vertices for rest lengths (matches bouncing_ball.py)
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
                    # Scale rest length by scale factor to match scaled particle positions
                    # vertex_positions are from normalized mesh (10m), so scale by user's scale
                    rest_length = float(np.linalg.norm(p1 - p0)) * scale
                    
                    builder.add_spring(
                        start_particle + i_local,
                        start_particle + j_local,
                        self.spring_ke,
                        self.spring_kd,
                        rest_length
                    )
        
        print(f"Added {len(added_springs)} unique springs", flush=True)
        
        self.model = builder.finalize()
        
        print(f"\nModel created:", flush=True)
        print(f"  Particles: {self.model.particle_count}", flush=True)
        print(f"  Springs: {self.model.spring_count}", flush=True)
        print(f"  Tetrahedra: {self.model.tet_count}", flush=True)
        print(f"  Triangles: {self.model.tri_count}", flush=True)
        
        # Debug: Check rest configuration of tetrahedra
        if self.model.tet_count > 0:
            print(f"\n🔍 Checking tetrahedra rest configuration:", flush=True)
            tet_poses = self.model.tet_poses.numpy()
            rest_volumes = []
            for i in range(min(10, len(tet_poses))):  # First 10
                Dm = tet_poses[i]
                det = np.linalg.det(Dm)
                rest_vol = abs(1.0 / (det * 6.0)) if abs(det) > 1e-20 else 0.0
                rest_volumes.append(rest_vol)
            
            if rest_volumes:
                print(f"  Sample rest volumes (first {len(rest_volumes)}): {[f'{v:.2e}' for v in rest_volumes]}", flush=True)
                min_rest_vol = min(rest_volumes)
                max_rest_vol = max(rest_volumes)
                print(f"  Rest volume range: [{min_rest_vol:.2e}, {max_rest_vol:.2e}]", flush=True)
                
                if min_rest_vol < 1e-10:
                    print(f"  ⚠ Warning: Very small rest volumes detected - may cause instability!", flush=True)
                    print(f"  This can happen if mesh was normalized and rest config is too small.", flush=True)
        
        # Set gravity (Z is up, gravity pulls down)
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        
        # Contact parameters (match old working example)
        self.model.soft_contact_ke = 1.0e3  # Match old working example
        self.model.soft_contact_kd = 10.0   # Match old working example
        self.model.soft_contact_kf = 1.0e3  # Match old working example
        self.model.soft_contact_mu = 0.5    # Match old working example
        
        # Particle rendering radius (match bouncing_ball.py)
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.015),  # Match bouncing_ball.py (was 0.008)
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
        
        # Initialize FK (for rigid bodies/joints)
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0
        )
        
        # CRITICAL: Ensure particle states are initialized from model
        # (eval_fk only handles rigid bodies, not particles)
        if self.model.particle_count > 0:
            wp.copy(self.state_0.particle_q, self.model.particle_q)
            wp.copy(self.state_0.particle_qd, self.model.particle_qd)
            
            # Verify initialization
            positions = self.state_0.particle_q.numpy()
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                raise ValueError("Particle positions contain NaN/Inf after initialization! Check mesh data.")
            
            # Debug: Check initial positions
            pos_min = positions.min(axis=0)
            pos_max = positions.max(axis=0)
            pos_mean = positions.mean(axis=0)
            print(f"✓ Particle states initialized: {len(positions)} particles", flush=True)
            print(f"  Initial positions - min: {pos_min}, max: {pos_max}, mean: {pos_mean}", flush=True)
            
            # Check if particles are at expected height (using original vertices)
            expected_z_min = initial_height + vertices[:, 2].min()
            expected_z_max = initial_height + vertices[:, 2].max()
            actual_z_min = positions[:, 2].min()
            actual_z_max = positions[:, 2].max()
            print(f"  Expected Z range: [{expected_z_min:.3f}, {expected_z_max:.3f}], Actual: [{actual_z_min:.3f}, {actual_z_max:.3f}]", flush=True)
        
        # Set up viewer
        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True  # Enable soft body visualization
        
        # Set particle colors to white
        self.particle_colors = wp.full(
            shape=self.model.particle_count,
            value=wp.vec3(1.0, 1.0, 1.0),  # White color
            device=self.model.device
        )
        
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
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1
        
        for substep in range(self.substeps):
            # Debug: Check state BEFORE solver step
            if self._debug_frame_count <= 3 and substep <= 2:
                pos_before = self.state_0.particle_q.numpy()
                vel_before = self.state_0.particle_qd.numpy()
                
                pos_mean = pos_before.mean(axis=0)
                pos_max = np.abs(pos_before).max()
                vel_mean = vel_before.mean(axis=0)
                vel_max = np.abs(vel_before).max()
                
                print(f"\n[DEBUG] Frame {self._debug_frame_count}, Substep {substep} BEFORE solver:", flush=True)
                print(f"   Positions: mean={pos_mean}, max={pos_max:.2e}m", flush=True)
                print(f"   Velocities: mean={vel_mean}, max={vel_max:.2e}m/s", flush=True)
                
                # Check for any suspicious values
                if pos_max > 10.0:
                    print(f"   ⚠⚠⚠ WARNING: Positions are very large ({pos_max:.1f}m) - mesh may be wrong scale!", flush=True)
                if vel_max > 100.0:
                    print(f"   ⚠⚠⚠ WARNING: Velocities are very large ({vel_max:.1f}m/s) - forces may be too strong!", flush=True)
            
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
            
            # Debug: Check state AFTER solver step
            if self._debug_frame_count <= 3 and substep <= 2:
                pos_after = self.state_1.particle_q.numpy()
                vel_after = self.state_1.particle_qd.numpy()
                
                pos_mean_after = pos_after.mean(axis=0)
                pos_max_after = np.abs(pos_after).max()
                vel_mean_after = vel_after.mean(axis=0)
                vel_max_after = np.abs(vel_after).max()
                
                pos_delta = pos_max_after - pos_max if substep > 0 else 0
                vel_delta = vel_max_after - vel_max if substep > 0 else 0
                
                print(f"[DEBUG] Frame {self._debug_frame_count}, Substep {substep} AFTER solver:", flush=True)
                print(f"   Positions: mean={pos_mean_after}, max={pos_max_after:.2e}m (Δ={pos_delta:.2e})", flush=True)
                print(f"   Velocities: mean={vel_mean_after}, max={vel_max_after:.2e}m/s (Δ={vel_delta:.2e})", flush=True)
            
            # Check for NaN immediately after solver step
            positions = self.state_1.particle_q.numpy()
            velocities = self.state_1.particle_qd.numpy()
            
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                print(f"\n❌ ERROR: NaN detected after solver step {substep}!", flush=True)
                print(f"   Frame: {self._debug_frame_count}, Substep: {substep}", flush=True)
                print(f"   Sim time: {self.sim_time:.6f}s", flush=True)
                print(f"   Timestep: {self.sim_dt:.6f}s", flush=True)
                print(f"   Stiffness: k_mu={self.k_mu:.0e}, k_lambda={self.k_lambda:.0e}", flush=True)
                
                # Find which particles have NaN
                nan_mask = np.isnan(positions).any(axis=1) | np.isinf(positions).any(axis=1)
                nan_count = nan_mask.sum()
                if nan_count > 0:
                    nan_indices = np.where(nan_mask)[0][:10]  # First 10 NaN particles
                    print(f"   NaN particles: {nan_count} (showing first {len(nan_indices)}):", flush=True)
                    for idx in nan_indices:
                        print(f"     Particle {idx}: pos={positions[idx]}, vel={velocities[idx]}", flush=True)
                
                raise RuntimeError("Simulation became unstable - solver produced NaN values")
            
            # Check for explosion (positions growing too fast)
            if substep > 0:
                pos_max = np.abs(positions).max()
                vel_max_check = np.abs(velocities).max()
                
                if pos_max > 10.0:  # More than 10m is suspicious for normalized mesh
                    print(f"\n⚠⚠⚠ WARNING: Mesh exploding! Max position: {pos_max:.2f}m at substep {substep}", flush=True)
                    print(f"   Max velocity: {vel_max_check:.2f}m/s", flush=True)
                    print(f"   This suggests:", flush=True)
                    print(f"     - Rest configuration may be wrong (check tetrahedra rest volumes)", flush=True)
                    print(f"     - Stiffness parameters may be wrong for mesh scale", flush=True)
                    print(f"     - Time step may be too large (current: {self.sim_dt:.6f}s)", flush=True)
                    print(f"     - Mesh may have degenerate tetrahedra", flush=True)
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
    
    def render(self):
        """Render current frame."""
        if self.viewer is None:
            return
            
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        
        # Override particle colors to white
        if self.model.particle_count:
            self.viewer.log_points(
                name="/model/particles",
                points=self.state_0.particle_q,
                radii=self.model.particle_radius,
                colors=self.particle_colors,
                hidden=not self.viewer.show_particles,
            )
        
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
            
            # Check for NaN values (indicates simulation instability)
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                print(f"\n❌ ERROR: Simulation became unstable at frame {frame}!", flush=True)
                print(f"   NaN/Inf detected in particle positions", flush=True)
                print(f"   Try reducing stiffness (--k_mu, --k_lambda) or increasing substeps (--substeps)", flush=True)
                break
            
            center = positions.mean(axis=0)
            height = center[2]
            
            # Detect bounces
            velocities = self.state_0.particle_qd.numpy()
            
            # Check velocities for NaN
            if np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)):
                print(f"\n❌ ERROR: Simulation became unstable at frame {frame}!", flush=True)
                print(f"   NaN/Inf detected in particle velocities", flush=True)
                print(f"   Try reducing stiffness (--k_mu, --k_lambda) or increasing substeps (--substeps)", flush=True)
                break
            
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
    parser.add_argument('--mesh_file', type=str, default=None,
                        help='Path to .mesh file (Medit format). Defaults to examples/assets/spot_fixed.mesh')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Mesh scale factor for add_soft_mesh (default: 1.0 for pre-processed meshes)')
    
    # Physics parameters
    parser.add_argument('--initial_height', type=float, default=0.2,
                        help='Drop height (default: 0.2)')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Mesh mass (default: 1.0, match old working example)')
    parser.add_argument('--k_mu', type=float, default=5.0,
                        help='Shear modulus (default: 5.0, match old working example)')
    parser.add_argument('--k_lambda', type=float, default=5.0,
                        help='Bulk modulus (default: 5.0, match old working example)')
    parser.add_argument('--k_damp', type=float, default=40.0,
                        help='Tetrahedral damping (default: 40.0, match old working example)')
    parser.add_argument('--spring_ke', type=float, default=50.0,
                        help='Spring stiffness (default: 50.0, match old working example)')
    parser.add_argument('--spring_kd', type=float, default=40.0,
                        help='Spring damping (default: 40.0, match old working example)')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravity (default: 9.81)')
    
    # Simulation parameters
    parser.add_argument('--substeps', type=int, default=20,
                        help='Substeps per frame (default: 20, increased for stability with complex meshes)')
    parser.add_argument('--num_frames', type=int, default=300,
                        help='Number of frames (default: 300)')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization')
    
    args = parser.parse_args()
    
    # Resolve mesh file path - use default if not provided
    if args.mesh_file is None:
        # Try spot_fixed.mesh first, then fall back to spot.mesh
        spot_fixed = newton.examples.get_asset("spot_fixed.mesh")
        if os.path.exists(spot_fixed):
            args.mesh_file = spot_fixed
            print(f"Using pre-processed mesh: spot_fixed.mesh")
        else:
            args.mesh_file = newton.examples.get_asset("spot.mesh")
            print(f"⚠ Using spot.mesh (consider creating spot_fixed.mesh for better performance)")
            print(f"  Run: python -m newton.examples.soft.create_spot_fixed_mesh")
    
    # Check mesh file exists
    if not os.path.exists(args.mesh_file):
        print(f"Error: Mesh file not found: {args.mesh_file}")
        print(f"Please provide a valid path to a .mesh file")
        print(f"\nTo generate pre-processed meshes, run:")
        print(f"  python -m newton.examples.soft.create_ball_mesh")
        print(f"  python -m newton.examples.soft.create_spot_fixed_mesh")
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
                    width=1920,
                    height=1080,
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

