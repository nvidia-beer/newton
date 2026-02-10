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
Tetrahedral Mesh Stability Tester

This utility iteratively adds tetrahedra to a soft body simulation one by one,
testing stability after each addition. It helps identify problematic tetrahedra
that cause simulation instability.

This is a HEADLESS utility - no GUI or display required. Since it uses Warp/CUDA,
it should be run via Docker using the provided shell script wrapper.

Usage:
    # Recommended: Use the shell script wrapper (handles Docker setup)
    ./newton/.devcontainer/test-tetra-stability.sh examples/assets/spot.mesh
    ./newton/.devcontainer/test-tetra-stability.sh examples/assets/spot.mesh --test_steps 100 --max_position 10.0
    
    # Or run directly in Docker (if you have the environment set up)
    docker run ... python -m newton.examples.utils.test_tetra_stability \\
        --mesh_file newton/newton/examples/assets/spot.mesh \\
        --test_steps 100 \\
        --max_position 10.0
"""

import warp as wp
import numpy as np
import argparse
import os
import sys

import newton
from newton.solvers import SolverSoft
from newton.utils import load_tetrahedral_mesh


class TetraStabilityTester:
    """
    Tests tetrahedral mesh stability by adding tetrahedra incrementally.
    """
    
    def __init__(
        self,
        mesh_file: str,
        scale: float = 1.0,
        initial_height: float = 0.2,
        mass: float = 1.0,
        k_mu: float = 5.0,
        k_lambda: float = 5.0,
        k_damp: float = 40.0,
        spring_ke: float = 50.0,
        spring_kd: float = 40.0,
        gravity: float = 9.81,
        substeps: int = 20,
        test_steps: int = 50,
        max_position: float = 10.0,
        stability_threshold: float = 1e-6,
    ):
        """
        Initialize the stability tester.
        
        Args:
            mesh_file: Path to .mesh file
            scale: Mesh scale factor
            initial_height: Initial height above ground
            mass: Mesh mass
            k_mu: Shear modulus
            k_lambda: Bulk modulus
            k_damp: Damping
            spring_ke: Spring stiffness
            spring_kd: Spring damping
            gravity: Gravity magnitude
            substeps: Substeps per frame
            test_steps: Number of simulation steps to test after each tetra addition
            max_position: Maximum allowed position before considering unstable (meters)
            stability_threshold: Minimum rest volume to consider tetra valid
        """
        self.mesh_file = mesh_file
        self.scale = scale
        self.mass = mass
        self.k_mu = k_mu
        self.k_lambda = k_lambda
        self.k_damp = k_damp
        self.spring_ke = spring_ke
        self.spring_kd = spring_kd
        self.gravity = gravity
        self.substeps = substeps
        self.test_steps = test_steps
        self.max_position = max_position
        self.stability_threshold = stability_threshold
        
        # Load mesh
        print(f"Loading tetrahedral mesh from: {mesh_file}", flush=True)
        self.vertices, self.tetras = load_tetrahedral_mesh(mesh_file)
        print(f"Loaded mesh: {len(self.vertices)} vertices, {len(self.tetras)} tetrahedra", flush=True)
        
        # Validate mesh
        if len(self.vertices) == 0:
            raise ValueError("Mesh has no vertices!")
        if len(self.tetras) == 0:
            raise ValueError("Mesh has no tetrahedra!")
        if np.any(np.isnan(self.vertices)) or np.any(np.isinf(self.vertices)):
            raise ValueError("Mesh vertices contain NaN/Inf values!")
        
        # Check tetrahedra indices
        max_idx = len(self.vertices) - 1
        if np.any(self.tetras < 0) or np.any(self.tetras > max_idx):
            raise ValueError(f"Tetrahedra indices out of range! Max index: {max_idx}")
        
        # Convert to float32
        self.vertices = self.vertices.astype(np.float32)
        self.tetras = self.tetras.astype(np.int32)
        
        # Check mesh size and suggest scale if needed
        min_bounds = self.vertices.min(axis=0)
        max_bounds = self.vertices.max(axis=0)
        extents = max_bounds - min_bounds
        max_extent = np.max(extents)
        final_size = max_extent * scale
        
        # Calculate scaled mesh properties (scale is applied in add_soft_mesh)
        scaled_min_z = min_bounds[2] * scale
        scaled_max_z = max_bounds[2] * scale
        scaled_mesh_height = extents[2] * scale  # Scaled Z extent
        
        print(f"\n🔍 Mesh size analysis:", flush=True)
        print(f"  Original bounds: min={min_bounds}, max={max_bounds}", flush=True)
        print(f"  Original extents: {extents}, max: {max_extent:.3f}m", flush=True)
        print(f"  Scale parameter: {scale}", flush=True)
        print(f"  Scaled Z range: [{scaled_min_z:.3f}, {scaled_max_z:.3f}], height: {scaled_mesh_height:.3f}m", flush=True)
        print(f"  Expected final size: {final_size:.3f}m", flush=True)
        
        # CRITICAL: Verify mesh bottom is at Z=0 (as it should be from create_spot_fixed_mesh.py)
        if abs(min_bounds[2]) > 1e-3:
            print(f"  ⚠ WARNING: Mesh bottom is NOT at Z=0! min_bounds[2]={min_bounds[2]:.6f}", flush=True)
            print(f"  The mesh should have been created with bottom at Z=0.", flush=True)
            print(f"  This will cause the mesh to start under ground!", flush=True)
        
        # Calculate initial_height automatically based on SCALED mesh geometry
        # CRITICAL: add_soft_mesh adds vertices as-is, so if mesh bottom is at min_bounds[2],
        # the actual bottom in simulation will be at: initial_height + min_bounds[2] * scale
        # We want: initial_height + min_bounds[2] * scale + clearance >= 0
        # So: initial_height >= -min_bounds[2] * scale - clearance
        
        # Clearance is 10% of scaled mesh height (minimum 0.1m, maximum 1.0m)
        clearance = max(0.1, min(1.0, scaled_mesh_height * 0.1))
        
        # Calculate height needed to lift mesh bottom above ground with clearance
        # If mesh bottom is at Z=0 (min_bounds[2] = 0), then scaled_min_z = 0
        # So: calculated_height = -0 + clearance = clearance
        # This ensures mesh bottom is at 'clearance' above ground
        calculated_height = -scaled_min_z + clearance
        
        # Ensure minimum clearance above ground (in case mesh bottom is already at or above origin)
        calculated_height = max(calculated_height, clearance)
        
        # Verify calculation
        actual_bottom_z = calculated_height + scaled_min_z
        print(f"  Calculated initial_height: {calculated_height:.3f}m", flush=True)
        print(f"  Mesh bottom will be at: {actual_bottom_z:.3f}m (should be >= {clearance:.3f}m)", flush=True)
        if actual_bottom_z < clearance - 1e-3:
            print(f"  ⚠ WARNING: Mesh bottom will be below clearance level!", flush=True)
        
        # Use provided initial_height if > 0, otherwise use calculated value
        if initial_height > 0:
            self.initial_height = initial_height
            print(f"  Using provided initial_height: {initial_height:.3f}m", flush=True)
        else:
            self.initial_height = calculated_height
            print(f"  Calculated initial_height: {calculated_height:.3f}m", flush=True)
            print(f"    (scaled mesh Z range: [{scaled_min_z:.3f}, {scaled_max_z:.3f}], clearance: {clearance:.3f}m)", flush=True)
        
        # Warn if mesh is very large and suggest scale
        if max_extent > 100.0 and scale >= 1.0:
            suggested_scale = 0.1  # Like bouncing_mesh uses for spot.mesh
            print(f"\n⚠ WARNING: Mesh is very large ({max_extent:.1f}m)!", flush=True)
            print(f"  Consider using --scale {suggested_scale} for stability", flush=True)
            print(f"  This will result in a {max_extent * suggested_scale:.2f}m mesh", flush=True)
        elif final_size > 100.0:
            print(f"\n⚠ WARNING: Final mesh size ({final_size:.2f}m) is very large!", flush=True)
            print(f"  This may cause simulation instability", flush=True)
            print(f"  Consider using a smaller --scale parameter", flush=True)
        
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / substeps
        
        print(f"\nStability test parameters:", flush=True)
        print(f"  Test steps per tetra: {test_steps}", flush=True)
        print(f"  Max allowed position: {max_position}m", flush=True)
        print(f"  Initial height: {self.initial_height:.3f}m", flush=True)
        print(f"  Timestep: {self.sim_dt:.6f}s", flush=True)
        print(f"  Substeps: {substeps}", flush=True)
    
    def _check_tetra_volume(self, tet_indices, scaled=True):
        """Check if a tetrahedron has valid volume.
        
        Args:
            tet_indices: Vertex indices of the tetrahedron
            scaled: If True, account for scale factor (volumes scale by scale^3)
        """
        v0 = self.vertices[tet_indices[0]]
        v1 = self.vertices[tet_indices[1]]
        v2 = self.vertices[tet_indices[2]]
        v3 = self.vertices[tet_indices[3]]
        
        # Compute volume using determinant
        Dm = np.array([
            [v1[0] - v0[0], v2[0] - v0[0], v3[0] - v0[0]],
            [v1[1] - v0[1], v2[1] - v0[1], v3[1] - v0[1]],
            [v1[2] - v0[2], v2[2] - v0[2], v3[2] - v0[2]]
        ])
        
        det = np.linalg.det(Dm)
        volume = abs(det / 6.0)
        
        # Account for scale factor if requested (volumes scale by scale^3)
        if scaled and hasattr(self, 'scale'):
            volume_scaled = volume * (self.scale ** 3)
            return volume_scaled, det
        else:
            return volume, det
    
    def _compute_adaptive_threshold(self):
        """Compute an adaptive threshold based on actual volume distribution."""
        # Sample volumes from first 1000 tetrahedra to estimate distribution
        sample_size = min(1000, len(self.tetras))
        volumes = []
        
        for i in range(sample_size):
            volume, _ = self._check_tetra_volume(self.tetras[i])
            if volume > 0:
                volumes.append(volume)
        
        if len(volumes) > 0:
            volumes_array = np.array(volumes)
            min_volume = np.min(volumes_array)
            median_volume = np.median(volumes_array)
            
            # Use 1e-4 of minimum positive volume as threshold (very conservative)
            # But don't force it to be >= self.stability_threshold - use the computed value
            computed_threshold = min_volume * 1e-4
            
            # Only use base threshold if computed is unreasonably small (less than machine epsilon)
            if computed_threshold < 1e-20:
                adaptive_threshold = self.stability_threshold
                print(f"  Volume statistics (from {sample_size} sample):", flush=True)
                print(f"    Min positive: {min_volume:.2e}", flush=True)
                print(f"    Median: {median_volume:.2e}", flush=True)
                print(f"    Computed threshold too small, using base: {adaptive_threshold:.2e}", flush=True)
            else:
                adaptive_threshold = computed_threshold
                print(f"  Volume statistics (from {sample_size} sample):", flush=True)
                print(f"    Min positive: {min_volume:.2e}", flush=True)
                print(f"    Median: {median_volume:.2e}", flush=True)
                print(f"    Adaptive threshold: {adaptive_threshold:.2e} (1e-4 of min, base was {self.stability_threshold:.2e})", flush=True)
            
            return adaptive_threshold
        else:
            # Fallback to default
            print(f"  No positive volumes found in sample, using base threshold: {self.stability_threshold:.2e}", flush=True)
            return self.stability_threshold
    
    def _build_model_with_tetra_list(self, tetra_indices: list[int]):
        """Build a model with a specific list of tetrahedra indices.
        
        This is more precise than _build_model_with_tetras for incremental testing.
        """
        builder = newton.ModelBuilder()
        
        # Add ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,
                kd=1e3,
                kf=1e4,
                mu=0.5
            )
        )
        
        # Track particle start index
        start_particle = builder.particle_count
        
        # Get the actual tetrahedra to include
        tetras_subset = self.tetras[tetra_indices].copy()
        
        # Validate vertex indices are within bounds
        max_vertex_idx = len(self.vertices) - 1
        invalid_indices = []
        for t_idx, tet_idx in enumerate(tetra_indices):
            tet = self.tetras[tet_idx]
            for v_idx in tet:
                if v_idx < 0 or v_idx > max_vertex_idx:
                    invalid_indices.append((tet_idx, v_idx))
        
        if invalid_indices:
            raise ValueError(f"Invalid vertex indices in tetrahedra: {invalid_indices[:10]}")
        
        # Flatten tetrahedra indices for add_soft_mesh
        indices = tetras_subset.flatten().astype(np.int32)
        
        # Verify indices are valid
        if np.any(indices < 0) or np.any(indices >= len(self.vertices)):
            raise ValueError(f"Invalid vertex indices: min={indices.min()}, max={indices.max()}, "
                           f"but vertices range is [0, {len(self.vertices)-1}]")
        
        # Add soft mesh with subset of tetrahedra
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, self.initial_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.vertices,  # All vertices
            indices=indices,  # Only the specified tetrahedra
            scale=self.scale,
            density=self.mass,
            k_mu=self.k_mu,
            k_lambda=self.k_lambda,
            k_damp=self.k_damp,
        )
        
        # Add springs between mesh vertices for stability
        vertex_positions = np.array(self.vertices)
        added_springs = set()
        
        # Add springs only for edges in the current tetrahedra subset
        for tet in tetras_subset:
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for ei, ej in edges:
                i_local, j_local = tet[ei], tet[ej]
                if i_local > j_local:
                    i_local, j_local = j_local, i_local
                
                spring_key = (i_local, j_local)
                if spring_key not in added_springs:
                    added_springs.add(spring_key)
                    
                    p0 = vertex_positions[i_local]
                    p1 = vertex_positions[j_local]
                    rest_length = float(np.linalg.norm(p1 - p0)) * self.scale
                    
                    builder.add_spring(
                        start_particle + i_local,
                        start_particle + j_local,
                        self.spring_ke,
                        self.spring_kd,
                        rest_length
                    )
        
        model = builder.finalize()
        
        # Set gravity
        model.gravity = wp.array([wp.vec3(0.0, 0.0, -self.gravity)], dtype=wp.vec3, device=model.device)
        
        # Contact parameters
        model.soft_contact_ke = 1.0e3
        model.soft_contact_kd = 10.0
        model.soft_contact_kf = 1.0e3
        model.soft_contact_mu = 0.5
        
        # Particle radius
        model.particle_radius = wp.array(
            np.full(model.particle_count, 0.015),
            dtype=wp.float32,
            device=model.device
        )
        
        return model
    
    def _build_model_with_tetras(self, num_tetras: int):
        """Build a model with the first num_tetras tetrahedra.
        
        IMPORTANT: This carefully handles incremental addition:
        - All vertices are added first (add_soft_mesh does this)
        - Only the first num_tetras tetrahedra are added
        - Vertex indices in tetrahedra must be valid (0 to len(vertices)-1)
        """
        builder = newton.ModelBuilder()
        
        # Add ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,
                kd=1e3,
                kf=1e4,
                mu=0.5
            )
        )
        
        # Track particle start index
        start_particle = builder.particle_count
        
        # Validate: ensure we don't exceed available tetrahedra
        if num_tetras > len(self.tetras):
            raise ValueError(f"Requested {num_tetras} tetrahedra but only {len(self.tetras)} available")
        
        # Get subset of tetrahedra to add
        tetras_subset = self.tetras[:num_tetras].copy()
        
        # Validate vertex indices are within bounds
        max_vertex_idx = len(self.vertices) - 1
        invalid_indices = []
        for t_idx, tet in enumerate(tetras_subset):
            for v_idx in tet:
                if v_idx < 0 or v_idx > max_vertex_idx:
                    invalid_indices.append((t_idx, v_idx))
        
        if invalid_indices:
            raise ValueError(f"Invalid vertex indices in tetrahedra: {invalid_indices[:10]}")
        
        # Flatten tetrahedra indices for add_soft_mesh
        # Note: indices are already 0-based and reference self.vertices
        indices = tetras_subset.flatten().astype(np.int32)
        
        # Verify indices are valid
        if np.any(indices < 0) or np.any(indices >= len(self.vertices)):
            raise ValueError(f"Invalid vertex indices: min={indices.min()}, max={indices.max()}, "
                           f"but vertices range is [0, {len(self.vertices)-1}]")
        
        # Add soft mesh with subset of tetrahedra
        # Note: add_soft_mesh adds ALL vertices first, then adds tetrahedra
        # The indices array tells it which tetrahedra to add (4 indices per tetrahedron)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, self.initial_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.vertices,  # All vertices (required - add_soft_mesh adds all of them)
            indices=indices,  # Only first num_tetras tetrahedra (4 indices per tetrahedron)
            scale=self.scale,
            density=self.mass,
            k_mu=self.k_mu,
            k_lambda=self.k_lambda,
            k_damp=self.k_damp,
        )
        
        # Add springs between mesh vertices for stability
        # Only add springs for vertices used by the current tetrahedra subset
        vertex_positions = np.array(self.vertices)
        added_springs = set()
        
        # Get all vertices used by current tetrahedra
        used_vertices = set()
        for t in range(num_tetras):
            for k in range(4):
                used_vertices.add(tetras_subset[t][k])
        
        # Add springs only for edges in the current tetrahedra subset
        for t in range(num_tetras):
            tet_indices = tetras_subset[t]
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
                    rest_length = float(np.linalg.norm(p1 - p0)) * self.scale
                    
                    builder.add_spring(
                        start_particle + i_local,
                        start_particle + j_local,
                        self.spring_ke,
                        self.spring_kd,
                        rest_length
                    )
        
        model = builder.finalize()
        
        # Set gravity
        model.gravity = wp.array([wp.vec3(0.0, 0.0, -self.gravity)], dtype=wp.vec3, device=model.device)
        
        # Contact parameters
        model.soft_contact_ke = 1.0e3
        model.soft_contact_kd = 10.0
        model.soft_contact_kf = 1.0e3
        model.soft_contact_mu = 0.5
        
        # Particle radius
        model.particle_radius = wp.array(
            np.full(model.particle_count, 0.015),
            dtype=wp.float32,
            device=model.device
        )
        
        return model
    
    def _test_stability(self, model, num_tetras: int):
        """Test if simulation is stable with given number of tetrahedra."""
        # Create solver
        solver = SolverSoft(
            model=model,
            dt=self.sim_dt,
            mass=self.mass,
            solver_type="bicgstab"
        )
        
        # Create states
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = None
        
        # Initialize FK
        newton.eval_fk(
            model,
            model.joint_q,
            model.joint_qd,
            state_0
        )
        
        # Initialize particle states
        if model.particle_count > 0:
            wp.copy(state_0.particle_q, model.particle_q)
            wp.copy(state_0.particle_qd, model.particle_qd)
        
        # Run test simulation
        for step in range(self.test_steps):
            state_0.clear_forces()
            
            # Collision detection
            contacts = model.collide(state=state_0)
            
            # Physics step
            try:
                solver.step(
                    state_in=state_0,
                    state_out=state_1,
                    control=control,
                    contacts=contacts,
                    dt=self.sim_dt
                )
            except Exception as e:
                return False, f"Solver exception: {e}"
            
            # Check for NaN/Inf
            positions = state_1.particle_q.numpy()
            velocities = state_1.particle_qd.numpy()
            
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                return False, "NaN/Inf detected in positions"
            
            if np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)):
                return False, "NaN/Inf detected in velocities"
            
            # Check for explosion (positions growing too fast)
            pos_max = np.abs(positions).max()
            if pos_max > self.max_position:
                return False, f"Position explosion: max position = {pos_max:.2f}m"
            
            # Swap states
            state_0, state_1 = state_1, state_0
        
        # Final check
        final_positions = state_0.particle_q.numpy()
        final_velocities = state_0.particle_qd.numpy()
        
        if np.any(np.isnan(final_positions)) or np.any(np.isinf(final_positions)):
            return False, "NaN/Inf in final positions"
        
        if np.any(np.isnan(final_velocities)) or np.any(np.isinf(final_velocities)):
            return False, "NaN/Inf in final velocities"
        
        final_pos_max = np.abs(final_positions).max()
        if final_pos_max > self.max_position:
            return False, f"Final position explosion: max = {final_pos_max:.2f}m"
        
        return True, "Stable"
    
    def test_all_tetras(self, start_from: int = 0, skip_degenerate: bool = True):
        """Test all tetrahedra incrementally.
        
        Args:
            start_from: Index to start testing from (useful if first N are known stable)
            skip_degenerate: If True, skip tetrahedra with invalid volumes and continue testing
        """
        print(f"\n{'='*80}", flush=True)
        print(f"Starting incremental tetrahedral stability test", flush=True)
        if start_from > 0:
            print(f"Starting from tetrahedron index: {start_from}", flush=True)
        if skip_degenerate:
            print(f"Mode: Will stop immediately on first unstable tetrahedron", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        total_tetras = len(self.tetras)
        last_stable = start_from - 1
        degenerate_tetras = []
        unstable_tetras = []
        
        # Check if mesh appears to be unfiltered
        # spot_fixed.mesh should have ~2965 tetrahedra after filtering (down from 16471)
        mesh_basename = os.path.basename(self.mesh_file).lower()
        if "spot_fixed" in mesh_basename and total_tetras > 10000:
            print(f"\n{'='*80}", flush=True)
            print(f"⚠⚠⚠ WARNING: Mesh appears UNFILTERED! ⚠⚠⚠", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"  Mesh file: {self.mesh_file}", flush=True)
            print(f"  Total tetrahedra: {total_tetras}", flush=True)
            print(f"  Expected after filtering: ~2965 (down from 16471)", flush=True)
            print(f"", flush=True)
            print(f"  This mesh needs to be regenerated with proper filtering!", flush=True)
            print(f"  Run: ./newton/.devcontainer/create-spot-fixed-mesh.sh", flush=True)
            print(f"", flush=True)
            print(f"  The test will continue, but will likely fail on first tetrahedron.", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        # Pre-check tetrahedra volumes for informational purposes only
        # If mesh was created with create_spot_fixed_mesh.py, it's already filtered
        # The simulation itself will be the real test - volume checks are just warnings
        print("Pre-checking tetrahedra volumes (informational only - mesh should already be filtered)...", flush=True)
        
        # Compute volume statistics to set adaptive threshold
        # IMPORTANT: Check volumes AFTER scaling (as they will be in simulation)
        # Volumes scale by scale^3, so we need to account for this
        sample_volumes = []
        sample_volumes_unscaled = []
        for i in range(min(1000, total_tetras)):
            volume_scaled, det = self._check_tetra_volume(self.tetras[i], scaled=True)
            volume_unscaled, _ = self._check_tetra_volume(self.tetras[i], scaled=False)
            if volume_scaled > 0:
                sample_volumes.append(volume_scaled)
                sample_volumes_unscaled.append(volume_unscaled)
        
        if len(sample_volumes) > 0:
            volumes_array = np.array(sample_volumes)
            volumes_unscaled_array = np.array(sample_volumes_unscaled)
            min_volume = np.min(volumes_array)
            median_volume = np.median(volumes_array)
            min_volume_unscaled = np.min(volumes_unscaled_array)
            median_volume_unscaled = np.median(volumes_unscaled_array)
            
            # For normalized meshes with very small volumes, use minimum-based threshold
            # A tetrahedron is degenerate if it's much smaller than the smallest valid one
            # Use 1e-2 of minimum volume (catches volumes 100x smaller than smallest valid)
            min_based_threshold = min_volume * 1e-2
            
            # Also use relative threshold: 1e-4 of median
            relative_threshold = median_volume * 1e-4
            
            # Use the more lenient threshold (larger value = less filtering)
            # This ensures we don't filter out all tetrahedra in normalized meshes
            adaptive_threshold = max(min_based_threshold, relative_threshold)
            
            # For non-normalized meshes, also enforce absolute minimum
            if median_volume > 1e-6:
                absolute_threshold = 1e-10
                adaptive_threshold = max(adaptive_threshold, absolute_threshold)
            
            print(f"  Volume statistics (SCALED, scale={self.scale:.3f}):", flush=True)
            print(f"    Min: {min_volume:.2e}, Median: {median_volume:.2e}", flush=True)
            print(f"  Volume statistics (UNSCALED):", flush=True)
            print(f"    Min: {min_volume_unscaled:.2e}, Median: {median_volume_unscaled:.2e}", flush=True)
            
            # Check if all volumes are extremely small (suggests normalized mesh)
            if median_volume < 1e-10:
                print(f"  ⚠ WARNING: All volumes are extremely small (median={median_volume:.2e})!", flush=True)
                print(f"    This suggests the mesh was normalized but may not have been properly filtered.", flush=True)
                print(f"    Consider regenerating with: ./newton/.devcontainer/create-spot-fixed-mesh.sh", flush=True)
                print(f"    Using very lenient threshold to avoid filtering all tetrahedra.", flush=True)
            
            print(f"  Using threshold: {adaptive_threshold:.2e}", flush=True)
            print(f"    (min-based: {min_based_threshold:.2e}, relative: {relative_threshold:.2e})", flush=True)
            print(f"  This will filter SCALED volumes < {adaptive_threshold:.2e}", flush=True)
        else:
            adaptive_threshold = 1e-10  # Match mesh creation threshold
            print(f"  Using fallback threshold: {adaptive_threshold:.2e} (matches mesh creation threshold)", flush=True)
        
        clearly_degenerate = []
        for i in range(start_from, total_tetras):
            tet = self.tetras[i]
            volume_scaled, det = self._check_tetra_volume(tet, scaled=True)
            # Skip if scaled volume is too small (will cause explosion) or determinant is at machine epsilon
            # CRITICAL: Check SCALED volume (as it will be in simulation)
            if abs(det) < 1e-20 or volume_scaled < adaptive_threshold:
                clearly_degenerate.append((i, volume_scaled, det))
        
        if clearly_degenerate:
            print(f"  ⚠ Found {len(clearly_degenerate)} tetrahedra with very small volumes:", flush=True)
            for idx, vol, det in clearly_degenerate[:10]:  # Show first 10
                print(f"    Tetra {idx}: volume={vol:.2e}, det={det:.2e}", flush=True)
            if len(clearly_degenerate) > 10:
                print(f"    ... and {len(clearly_degenerate) - 10} more", flush=True)
            print(f"  ⚠ WARNING: Mesh may not have been properly filtered!", flush=True)
            print(f"  ⚠ Consider regenerating with: ./newton/.devcontainer/create-spot-fixed-mesh.sh", flush=True)
            print(f"  ⚠ However, will test ALL tetrahedra via simulation (simulation is the real test)", flush=True)
        else:
            print(f"  ✓ No clearly degenerate tetrahedra found (mesh appears properly filtered)", flush=True)
            print(f"  ✓ All {total_tetras - start_from} tetrahedra will be tested via simulation", flush=True)
        
        print(f"\nTesting stability with incremental tetrahedra addition...", flush=True)
        print(f"  Simulation 1: 1 tetrahedron", flush=True)
        print(f"  Simulation 2: 2 tetrahedra", flush=True)
        print(f"  Simulation 3: 3 tetrahedra", flush=True)
        print(f"  ... continuing until instability detected", flush=True)
        print(f"", flush=True)
        
        # Test ALL tetrahedra via simulation - don't pre-filter
        # If mesh was created properly, it's already filtered
        # The simulation itself will identify any problematic ones
        valid_tetra_list = list(range(start_from, total_tetras))
        
        if len(clearly_degenerate) > 0:
            print(f"\n  ⚠ Note: Found {len(clearly_degenerate)} tetrahedra with small volumes", flush=True)
            print(f"  ⚠ But will test ALL {len(valid_tetra_list)} tetrahedra via simulation", flush=True)
            print(f"  ⚠ Simulation is the real test - volume checks are just warnings", flush=True)
        else:
            print(f"\n  ✓ Mesh appears properly filtered - testing all {len(valid_tetra_list)} tetrahedra", flush=True)
        
        for sim_num, tet_idx in enumerate(valid_tetra_list, start=1):
            num_tetras_in_sim = sim_num  # Number of valid tetrahedra in this simulation
            
            print(f"Simulation {sim_num}: Testing with {num_tetras_in_sim} tetrahedra (adding tetra {tet_idx})...", flush=True)
            
            # Get list of tetrahedra to include (all valid ones up to and including this one)
            tetras_to_include = valid_tetra_list[:sim_num]
            
            # Validate tetrahedron indices before building
            tet = self.tetras[tet_idx]
            max_vertex_idx = len(self.vertices) - 1
            if np.any(tet < 0) or np.any(tet > max_vertex_idx):
                invalid_vtx = [v for v in tet if v < 0 or v > max_vertex_idx]
                error_msg = f"Invalid vertex indices in tetra {tet_idx}: {invalid_vtx} (max valid: {max_vertex_idx})"
                print(f"❌ FAILED ({error_msg})", flush=True)
                print(f"\n{'='*80}", flush=True)
                print(f"❌ INDEXING ERROR!", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"Tetrahedron index: {tet_idx}", flush=True)
                print(f"Tetrahedron vertex indices: {tet}", flush=True)
                print(f"Valid vertex index range: [0, {max_vertex_idx}]", flush=True)
                return tet_idx, error_msg
            
            # Build model with the list of tetrahedra to include
            # Need to pass the actual count (including skipped degenerate ones in sequence)
            # But only include the valid ones
            try:
                # Build model with only the valid tetrahedra up to this point
                model = self._build_model_with_tetra_list(tetras_to_include)
            except Exception as e:
                print(f"❌ FAILED (model build error: {e})", flush=True)
                print(f"\n{'='*80}", flush=True)
                print(f"❌ INSTABILITY DETECTED!", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"Problematic tetrahedron index: {tet_idx}", flush=True)
                print(f"Tetrahedron vertices: {self.tetras[tet_idx]}", flush=True)
                print(f"Reason: Model build error: {e}", flush=True)
                return tet_idx, f"Model build error: {e}"
            
            # Test stability by running simulation
            is_stable, reason = self._test_stability(model, len(tetras_to_include))
            
            if is_stable:
                last_stable = tet_idx
                if (sim_num % 100 == 0) or (sim_num == len(valid_tetra_list)):
                    print(f"  ✓ STABLE - Simulation with {num_tetras_in_sim} tetrahedra completed successfully", flush=True)
                elif sim_num <= 10:  # Show first 10 explicitly
                    print(f"  ✓ STABLE", flush=True)
            else:
                # INSTABILITY DETECTED - STOP IMMEDIATELY
                print(f"  ❌ UNSTABLE - {reason}", flush=True)
                print(f"\n{'='*80}", flush=True)
                print(f"❌ INSTABILITY DETECTED!", flush=True)
                print(f"{'='*80}", flush=True)
                print(f"Simulation {sim_num} failed with {num_tetras_in_sim} tetrahedra", flush=True)
                print(f"Problematic tetrahedron index: {tet_idx} (the {sim_num}th valid tetrahedron added)", flush=True)
                print(f"Tetrahedron vertices: {self.tetras[tet_idx]}", flush=True)
                print(f"Reason: {reason}", flush=True)
                
                # Check volume for this unstable tetrahedron
                volume, det = self._check_tetra_volume(self.tetras[tet_idx])
                print(f"Volume: {volume:.2e}, Determinant: {det:.2e}", flush=True)
                
                print(f"\nLast stable simulation: Simulation {sim_num - 1} with {num_tetras_in_sim - 1} tetrahedra", flush=True)
                print(f"Last stable tetrahedron index: {last_stable}", flush=True)
                print(f"Total valid tetrahedra tested before failure: {sim_num}", flush=True)
                
                # Show vertex positions
                print(f"\nVertex positions of problematic tetrahedron:", flush=True)
                for v_idx in self.tetras[tet_idx]:
                    v_pos = self.vertices[v_idx]
                    print(f"  Vertex {v_idx}: ({v_pos[0]:.6f}, {v_pos[1]:.6f}, {v_pos[2]:.6f})", flush=True)
                
                unstable_tetras.append((tet_idx, reason))
                if volume < 1e-20 or abs(det) < 1e-20:
                    degenerate_tetras.append((tet_idx, volume, det))
                
                # STOP IMMEDIATELY - don't continue testing
                return tet_idx, reason
        
        # Report summary
        print(f"\n{'='*80}", flush=True)
        if len(unstable_tetras) == 0:
            print(f"✓ ALL TETRAHEDRA STABLE!", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"Successfully tested all {total_tetras} tetrahedra via simulation", flush=True)
            if degenerate_tetras:
                print(f"\nNote: {len(degenerate_tetras)} tetrahedra had very small volumes but were stable in simulation", flush=True)
            return None, "All stable"
        else:
            print(f"TEST COMPLETE - Summary:", flush=True)
            print(f"{'='*80}", flush=True)
            print(f"\n❌ Unstable tetrahedra ({len(unstable_tetras)} - failed simulation test):", flush=True)
            for idx, reason in unstable_tetras[:20]:  # Show first 20
                volume, det = self._check_tetra_volume(self.tetras[idx])
                print(f"  Tetra {idx}: {reason} (volume={volume:.2e})", flush=True)
            if len(unstable_tetras) > 20:
                print(f"  ... and {len(unstable_tetras) - 20} more", flush=True)
            
            if degenerate_tetras:
                print(f"\n⚠ Note: {len(degenerate_tetras)} unstable tetrahedra also had very small volumes", flush=True)
            
            return len(unstable_tetras), f"{len(unstable_tetras)} unstable (tested via simulation)"
    
    def run(self, skip_degenerate: bool = True):
        """Run the stability test.
        
        Args:
            skip_degenerate: If True, skip degenerate tetrahedra and continue testing
        """
        result_idx, reason = self.test_all_tetras(skip_degenerate=skip_degenerate)
        
        print(f"\n{'='*80}", flush=True)
        print(f"TEST SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Mesh file: {self.mesh_file}", flush=True)
        print(f"Total vertices: {len(self.vertices)}", flush=True)
        print(f"Total tetrahedra: {len(self.tetras)}", flush=True)
        
        if result_idx is not None:
            print(f"\n❌ FAILED at tetrahedron index: {result_idx}", flush=True)
            print(f"Reason: {reason}", flush=True)
            print(f"\nTo fix this mesh:", flush=True)
            print(f"  1. Check tetrahedron {result_idx} for degenerate geometry", flush=True)
            print(f"  2. Verify vertex positions are valid", flush=True)
            print(f"  3. Consider remeshing the problematic region", flush=True)
            return False
        else:
            print(f"\n✓ All tetrahedra are stable!", flush=True)
            return True


def main():
    parser = argparse.ArgumentParser(description='Tetrahedral Mesh Stability Tester')
    
    # Mesh parameters
    parser.add_argument('--mesh_file', type=str, default=None,
                        help='Path to .mesh file (Medit format). Defaults to spot_fixed.mesh if available, else spot.mesh')
    parser.add_argument('--scale', type=float, default=None,
                        help='Mesh scale factor (default: auto-detect, use 0.1 for spot.mesh, 1.0 for spot_fixed.mesh)')
    
    # Physics parameters
    parser.add_argument('--initial_height', type=float, default=-1.0,
                        help='Initial height above ground (default: -1.0 = auto-calculate from mesh geometry)')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Mesh mass (default: 1.0)')
    parser.add_argument('--k_mu', type=float, default=5.0,
                        help='Shear modulus (default: 5.0)')
    parser.add_argument('--k_lambda', type=float, default=5.0,
                        help='Bulk modulus (default: 5.0)')
    parser.add_argument('--k_damp', type=float, default=40.0,
                        help='Tetrahedral damping (default: 40.0)')
    parser.add_argument('--spring_ke', type=float, default=50.0,
                        help='Spring stiffness (default: 50.0)')
    parser.add_argument('--spring_kd', type=float, default=40.0,
                        help='Spring damping (default: 40.0)')
    parser.add_argument('--gravity', type=float, default=9.81,
                        help='Gravity (default: 9.81)')
    
    # Test parameters
    parser.add_argument('--substeps', type=int, default=20,
                        help='Substeps per frame (default: 20)')
    parser.add_argument('--test_steps', type=int, default=50,
                        help='Number of simulation steps to test after each tetra addition (default: 50)')
    parser.add_argument('--max_position', type=float, default=10.0,
                        help='Maximum allowed position before considering unstable in meters (default: 10.0)')
    parser.add_argument('--stability_threshold', type=float, default=1e-6,
                        help='Minimum rest volume to consider tetra valid (default: 1e-6)')
    parser.add_argument('--skip-degenerate', action='store_true', default=True,
                        help='Skip degenerate tetrahedra and continue testing (default: True)')
    parser.add_argument('--fail-on-degenerate', action='store_true',
                        help='Fail immediately on degenerate tetrahedra (opposite of --skip-degenerate)')
    parser.add_argument('--device', type=str, default=None,
                        help='Compute device')
    
    args = parser.parse_args()
    
    # Resolve mesh file path - use default if not provided
    mesh_file = args.mesh_file
    if mesh_file is None:
        # Try spot_fixed.mesh first, then fall back to spot.mesh
        try:
            import newton.examples
            spot_fixed = newton.examples.get_asset("spot_fixed.mesh")
            if os.path.exists(spot_fixed):
                mesh_file = spot_fixed
                print(f"Using pre-processed mesh: spot_fixed.mesh", flush=True)
            else:
                mesh_file = newton.examples.get_asset("spot.mesh")
                print(f"⚠ Using spot.mesh (consider creating spot_fixed.mesh for better stability)", flush=True)
                print(f"  Run: ./newton/.devcontainer/create-spot-fixed-mesh.sh", flush=True)
        except Exception as e:
            print(f"Warning: Could not find default mesh files: {e}", flush=True)
            print(f"Please specify --mesh_file explicitly", flush=True)
            sys.exit(1)
    
    # Check mesh file exists - try multiple locations
    if not os.path.exists(mesh_file):
        # Try relative to current directory
        if os.path.exists(os.path.join(os.getcwd(), mesh_file)):
            mesh_file = os.path.join(os.getcwd(), mesh_file)
        # Try relative to examples directory
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "..", "assets", os.path.basename(mesh_file))):
            mesh_file = os.path.join(os.path.dirname(__file__), "..", "assets", os.path.basename(mesh_file))
        # Try using newton.examples.get_asset
        else:
            try:
                import newton.examples
                asset_path = newton.examples.get_asset(os.path.basename(mesh_file))
                if os.path.exists(asset_path):
                    mesh_file = asset_path
            except:
                pass
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file not found: {args.mesh_file if args.mesh_file else 'default'}")
        print(f"Tried: {mesh_file}")
        print(f"\nTo generate pre-processed meshes, run:")
        print(f"  ./newton/.devcontainer/create-spot-fixed-mesh.sh")
        print(f"  ./newton/.devcontainer/create-ball-mesh.sh")
        sys.exit(1)
    
    # Update args with resolved path
    args.mesh_file = os.path.abspath(mesh_file)
    
    # Auto-detect scale if not provided
    scale = args.scale
    if scale is None:
        # Load mesh temporarily to check size
        try:
            vertices, _ = load_tetrahedral_mesh(mesh_file)
            extents = vertices.max(axis=0) - vertices.min(axis=0)
            max_extent = np.max(extents)
            
            # Check if it's spot_fixed.mesh (normalized to ~10m) or spot.mesh (very large)
            mesh_basename = os.path.basename(mesh_file).lower()
            if "spot_fixed" in mesh_basename or (max_extent < 20.0 and max_extent > 5.0):
                # spot_fixed.mesh is normalized to ~10m, use scale=1.0
                scale = 1.0
                print(f"Auto-detected scale: {scale} (pre-processed mesh, size: {max_extent:.1f}m)", flush=True)
            elif max_extent > 100.0:
                # Large unprocessed mesh (like spot.mesh), use scale=0.1
                scale = 0.1
                print(f"Auto-detected scale: {scale} (large mesh, size: {max_extent:.1f}m)", flush=True)
            else:
                # Small mesh, use scale=1.0
                scale = 1.0
                print(f"Auto-detected scale: {scale} (mesh size: {max_extent:.1f}m)", flush=True)
        except Exception as e:
            print(f"Warning: Could not auto-detect scale: {e}", flush=True)
            scale = 1.0
    
    wp.init()
    
    with wp.ScopedDevice(args.device):
        tester = TetraStabilityTester(
            mesh_file=args.mesh_file,
            scale=scale,
            initial_height=args.initial_height,
            mass=args.mass,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            gravity=args.gravity,
            substeps=args.substeps,
            test_steps=args.test_steps,
            max_position=args.max_position,
            stability_threshold=args.stability_threshold,
        )
        
        # Determine skip_degenerate flag
        skip_degenerate = args.skip_degenerate and not args.fail_on_degenerate
        
        success = tester.run(skip_degenerate=skip_degenerate)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
