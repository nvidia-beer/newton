"""
Model Order Reduction utilities for Warp tutorials.

This module provides simplified MOR components for the tutorial notebook:
1. SnapshotCollector - Collect simulation snapshots for training
2. PODReducer - Compute reduced basis using SVD
3. ReducedBasis - Container for POD basis with projection operations
4. SolverReduced - Reduced-order solver using pre-computed basis

Usage:
    from reduction import SnapshotCollector, PODReducer, SolverReduced

Based on SOFA's ModelOrderReduction plugin:
- Goury & Duriez, "Fast, generic and reliable control and simulation of soft robots"
"""

import numpy as np
import warp as wp
from typing import Optional, Tuple, Any
from scipy import linalg as scipy_linalg


# =============================================================================
# SNAPSHOT COLLECTOR
# =============================================================================

class SnapshotCollector:
    """
    Collect simulation snapshots for POD training.
    
    The quality of the reduced basis depends on snapshot diversity:
    - Use varied conditions (gravity, external forces)
    - Collect at multiple timesteps during simulation
    - Include both transient and steady-state behavior
    
    Example:
        >>> collector = SnapshotCollector()
        >>> for step in range(n_steps):
        >>>     solver.step(state_in, state_out, dt)
        >>>     if step % 5 == 0:  # Every 5 steps
        >>>         collector.add_snapshot(state_out.particle_q.numpy())
        >>> snapshots = collector.get_snapshots()
    """
    
    def __init__(self, max_snapshots: int = 5000):
        """
        Initialize snapshot collector.
        
        Args:
            max_snapshots: Maximum number of snapshots to store
        """
        self.max_snapshots = max_snapshots
        self.rest_position = None
        self._snapshots = []
        self._step_count = 0
    
    @property
    def n_snapshots(self) -> int:
        """Number of collected snapshots."""
        return len(self._snapshots)
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (2 * n_particles for 2D)."""
        if len(self._snapshots) > 0:
            return len(self._snapshots[0])
        return 0
    
    def set_rest_position(self, positions: np.ndarray):
        """Set rest position from initial configuration."""
        self.rest_position = positions.flatten().astype(np.float32)
    
    def add_snapshot(self, positions: np.ndarray):
        """
        Add a single snapshot.
        
        Args:
            positions: Current positions, shape (n_particles, 2) or (n_dof,)
        """
        pos_flat = positions.flatten().astype(np.float32)
        
        # Set rest position from first snapshot if not provided
        if self.rest_position is None:
            self.rest_position = pos_flat.copy()
        
        # Store snapshot (circular buffer if exceeds max)
        if len(self._snapshots) >= self.max_snapshots:
            idx = self._step_count % self.max_snapshots
            self._snapshots[idx] = pos_flat
        else:
            self._snapshots.append(pos_flat)
        
        self._step_count += 1
    
    def get_snapshots(self) -> np.ndarray:
        """
        Get collected snapshots as matrix.
        
        Returns:
            Snapshot matrix, shape (n_dof, n_snapshots)
        """
        if len(self._snapshots) == 0:
            raise ValueError("No snapshots collected")
        return np.array(self._snapshots).T
    
    def get_displacement_snapshots(self) -> np.ndarray:
        """Get displacement snapshots (relative to rest position)."""
        snapshots = self.get_snapshots()
        return snapshots - self.rest_position.reshape(-1, 1)
    
    def clear(self):
        """Clear all collected snapshots."""
        self._snapshots = []
        self._step_count = 0


# =============================================================================
# REDUCED BASIS
# =============================================================================

class ReducedBasis:
    """
    Container for POD reduced basis.
    
    Stores:
    - POD modes (V_r matrix)
    - Rest position
    - Singular values (energy content)
    
    Provides projection operations:
    - project_to_reduced: q_r = V^T @ (x - x0)
    - project_to_full: x = V @ q_r + x0
    """
    
    def __init__(
        self,
        modes: np.ndarray,
        rest_position: np.ndarray,
        singular_values: Optional[np.ndarray] = None,
        n_particles: Optional[int] = None,
    ):
        """
        Initialize reduced basis.
        
        Args:
            modes: Mode matrix V_r, shape (n_dof, n_modes)
            rest_position: Rest configuration x0, shape (n_dof,)
            singular_values: SVD singular values (for analysis)
            n_particles: Number of particles (n_dof = 2 * n_particles)
        """
        self.modes = modes.astype(np.float32)
        self.rest_position = rest_position.flatten().astype(np.float32)
        self.singular_values = singular_values
        
        self.n_full = modes.shape[0]
        self.n_modes = modes.shape[1]
        self.n_particles = n_particles if n_particles else self.n_full // 2
        
        # Precompute transpose for efficient projection
        self._modes_T = self.modes.T
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio: n_full / n_modes."""
        return self.n_full / self.n_modes
    
    @property
    def speedup_estimate(self) -> float:
        """Estimated linear solve speedup: (n_full/n_modes)^2."""
        return self.compression_ratio ** 2
    
    def project_to_reduced(self, x_full: np.ndarray) -> np.ndarray:
        """Project full state to reduced coordinates."""
        x_diff = x_full.flatten() - self.rest_position
        return self._modes_T @ x_diff
    
    def project_to_full(self, q_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct full state from reduced coordinates."""
        return self.modes @ q_reduced + self.rest_position
    
    def reconstruction_error(self, x_full: np.ndarray) -> float:
        """Compute relative reconstruction error."""
        x_diff = x_full.flatten() - self.rest_position
        x_reconstructed = self.modes @ (self._modes_T @ x_diff)
        error = np.linalg.norm(x_diff - x_reconstructed)
        norm = np.linalg.norm(x_diff)
        return error / norm if norm > 1e-10 else 0.0
    
    def save(self, filepath: str):
        """Save reduced basis to file."""
        np.savez(
            filepath,
            modes=self.modes,
            rest_position=self.rest_position,
            singular_values=self.singular_values,
            n_particles=self.n_particles,
        )
        print(f"Saved reduced basis to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ReducedBasis":
        """Load reduced basis from file."""
        data = np.load(filepath)
        return cls(
            modes=data['modes'],
            rest_position=data['rest_position'],
            singular_values=data.get('singular_values'),
            n_particles=int(data.get('n_particles', data['modes'].shape[0] // 2)),
        )
    
    def __repr__(self) -> str:
        return (f"ReducedBasis(n_full={self.n_full}, n_modes={self.n_modes}, "
                f"compression={self.compression_ratio:.1f}x)")


# =============================================================================
# POD REDUCER
# =============================================================================

class PODReducer:
    """
    Proper Orthogonal Decomposition for computing reduced basis.
    
    POD finds an optimal low-dimensional basis from simulation snapshots
    using Singular Value Decomposition (SVD).
    
    Mathematical Foundation:
    -----------------------
    Given N snapshots: X = [x₁ - x₀, x₂ - x₀, ..., xₙ - x₀]
    
    SVD: X = U Σ V^T
    
    The columns of U are the POD modes (principal directions).
    The singular values σᵢ indicate energy content of each mode.
    
    Mode Selection:
        Keep r modes such that √(Σᵢ>ᵣ σᵢ² / Σᵢ σᵢ²) < tolerance
    
    Example:
        >>> reducer = PODReducer(tolerance=1e-3)
        >>> basis, info = reducer.fit(snapshots, rest_position)
        >>> print(f"Reduced from {basis.n_full} to {basis.n_modes} modes")
    """
    
    def __init__(
        self,
        tolerance: float = 1e-3,
        max_modes: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize the POD reducer.
        
        Args:
            tolerance: Energy tolerance for mode selection.
                       Lower = more modes, higher accuracy.
                       1e-3 = 99.9% energy, 1e-2 = 99% energy
            max_modes: Maximum number of modes to keep (optional cap)
            verbose: Print progress information
        """
        self.tolerance = tolerance
        self.max_modes = max_modes
        self.verbose = verbose
        
        # Results
        self.singular_values = None
        self.energy_captured = None
        self.n_modes = None
    
    def fit(
        self,
        snapshots: np.ndarray,
        rest_position: np.ndarray,
    ) -> Tuple[ReducedBasis, dict]:
        """
        Compute POD basis from snapshots.
        
        Args:
            snapshots: Shape (n_dof, n_snapshots) or (n_snapshots, n_dof)
            rest_position: Rest configuration, shape (n_dof,)
        
        Returns:
            basis: ReducedBasis object containing the modes
            info: Dictionary with SVD information
        """
        if self.verbose:
            print("=" * 50)
            print("POD Reduction: Computing Reduced Basis")
            print("=" * 50)
        
        rest_position = rest_position.flatten()
        expected_n_dof = len(rest_position)
        
        # Ensure correct orientation: (n_dof, n_snapshots)
        if snapshots.shape[0] == expected_n_dof:
            pass  # Already correct
        elif snapshots.shape[1] == expected_n_dof:
            snapshots = snapshots.T
        else:
            raise ValueError(f"Snapshots shape {snapshots.shape} doesn't match DOF {expected_n_dof}")
        
        n_dof, n_snapshots = snapshots.shape
        
        if self.verbose:
            print(f"  DOF: {n_dof}, Snapshots: {n_snapshots}")
        
        # Compute displacement snapshots
        snapshot_diff = snapshots - rest_position.reshape(-1, 1)
        
        # Check for NaN
        if np.any(np.isnan(snapshot_diff)):
            raise ValueError("NaN detected in snapshots!")
        
        # Perform SVD (economy SVD)
        if self.verbose:
            print("  Computing SVD...")
        
        U, s, Vt = np.linalg.svd(snapshot_diff, full_matrices=False)
        self.singular_values = s
        
        # Compute energy ratios
        s_squared = s ** 2
        total_energy = np.sum(s_squared)
        
        # Determine number of modes
        n_modes = 1
        remaining_energy_ratio = np.sqrt(np.sum(s_squared[n_modes:]) / total_energy)
        
        while remaining_energy_ratio > self.tolerance and n_modes < len(s):
            n_modes += 1
            if n_modes < len(s):
                remaining_energy_ratio = np.sqrt(np.sum(s_squared[n_modes:]) / total_energy)
            else:
                remaining_energy_ratio = 0.0
        
        # Apply max_modes cap
        if self.max_modes is not None:
            n_modes = min(n_modes, self.max_modes)
        
        self.n_modes = n_modes
        self.energy_captured = 1.0 - remaining_energy_ratio ** 2
        
        if self.verbose:
            print(f"  Singular values: {len(s)}")
            print(f"  Tolerance: {self.tolerance}")
            print(f"  Selected modes: {n_modes}")
            print(f"  Energy captured: {self.energy_captured * 100:.4f}%")
        
        # Extract basis modes
        modes = U[:, :n_modes]
        
        # Create ReducedBasis
        basis = ReducedBasis(
            modes=modes,
            rest_position=rest_position,
            singular_values=s[:n_modes],
            n_particles=n_dof // 2,
        )
        
        info = {
            'singular_values': s,
            'energy_ratio': s_squared / total_energy,
            'n_modes': n_modes,
            'energy_captured': self.energy_captured,
        }
        
        if self.verbose:
            print("=" * 50)
            print(f"POD Complete: {n_dof} DOF → {n_modes} modes")
            print(f"Compression: {basis.compression_ratio:.1f}x")
            print("=" * 50)
        
        return basis, info


# =============================================================================
# WARP PROJECTION KERNELS
# =============================================================================

@wp.kernel
def project_positions_to_reduced_kernel(
    positions: wp.array(dtype=wp.vec2),
    rest_positions: wp.array(dtype=wp.vec2),
    modes: wp.array2d(dtype=float),
    q_reduced: wp.array(dtype=float),
    n_particles: int,
    n_modes: int,
):
    """Project full positions to reduced coordinates: q_r = V^T @ (x - x0)"""
    mode_idx = wp.tid()
    if mode_idx >= n_modes:
        return
    
    accum = float(0.0)
    for i in range(n_particles):
        diff = positions[i] - rest_positions[i]
        accum += modes[i * 2, mode_idx] * diff[0]
        accum += modes[i * 2 + 1, mode_idx] * diff[1]
    
    q_reduced[mode_idx] = accum


@wp.kernel
def project_reduced_to_full_kernel(
    q_reduced: wp.array(dtype=float),
    rest_positions: wp.array(dtype=wp.vec2),
    modes: wp.array2d(dtype=float),
    positions: wp.array(dtype=wp.vec2),
    n_particles: int,
    n_modes: int,
):
    """Reconstruct full positions from reduced: x = V @ q_r + x0"""
    particle_idx = wp.tid()
    if particle_idx >= n_particles:
        return
    
    dx = float(0.0)
    dy = float(0.0)
    
    for m in range(n_modes):
        coeff = q_reduced[m]
        dx += modes[particle_idx * 2, m] * coeff
        dy += modes[particle_idx * 2 + 1, m] * coeff
    
    rest = rest_positions[particle_idx]
    positions[particle_idx] = wp.vec2(rest[0] + dx, rest[1] + dy)


@wp.kernel
def project_velocities_to_reduced_kernel(
    velocities: wp.array(dtype=wp.vec2),
    modes: wp.array2d(dtype=float),
    v_reduced: wp.array(dtype=float),
    n_particles: int,
    n_modes: int,
):
    """Project velocities to reduced: v_r = V^T @ v"""
    mode_idx = wp.tid()
    if mode_idx >= n_modes:
        return
    
    accum = float(0.0)
    for i in range(n_particles):
        vel = velocities[i]
        accum += modes[i * 2, mode_idx] * vel[0]
        accum += modes[i * 2 + 1, mode_idx] * vel[1]
    
    v_reduced[mode_idx] = accum


@wp.kernel
def project_reduced_to_velocities_kernel(
    v_reduced: wp.array(dtype=float),
    modes: wp.array2d(dtype=float),
    velocities: wp.array(dtype=wp.vec2),
    n_particles: int,
    n_modes: int,
):
    """Reconstruct velocities from reduced: v = V @ v_r"""
    particle_idx = wp.tid()
    if particle_idx >= n_particles:
        return
    
    vx = float(0.0)
    vy = float(0.0)
    
    for m in range(n_modes):
        coeff = v_reduced[m]
        vx += modes[particle_idx * 2, m] * coeff
        vy += modes[particle_idx * 2 + 1, m] * coeff
    
    velocities[particle_idx] = wp.vec2(vx, vy)


@wp.kernel
def project_forces_to_reduced_kernel(
    forces: wp.array(dtype=wp.vec2),
    modes: wp.array2d(dtype=float),
    f_reduced: wp.array(dtype=float),
    n_particles: int,
    n_modes: int,
):
    """Project forces to reduced: f_r = V^T @ f"""
    mode_idx = wp.tid()
    if mode_idx >= n_modes:
        return
    
    accum = float(0.0)
    for i in range(n_particles):
        f = forces[i]
        accum += modes[i * 2, mode_idx] * f[0]
        accum += modes[i * 2 + 1, mode_idx] * f[1]
    
    f_reduced[mode_idx] = accum


# =============================================================================
# REDUCED-ORDER SOLVER
# =============================================================================

class SolverReduced:
    """
    Reduced-order implicit solver for 2D soft body simulation.
    
    Performs simulation in reduced space (r dimensions) instead of
    full space (N dimensions), providing significant speedup:
    - Linear solve: O(r³) instead of O(N^1.5)
    - With small r (~20-50), this can be 100x faster
    
    Mathematical Formulation:
    ------------------------
    Full system: (M - h²K) Δv = h*f
    Reduced system: (M_r - h²K_r) Δv_r = h*f_r
    
    where:
        M_r = V^T M V (reduced mass)
        K_r = V^T K V (reduced stiffness)
        f_r = V^T f   (reduced forces)
    
    Reconstruction: x = V @ q_r + x0
    
    Example:
        >>> basis = PODReducer(tolerance=1e-3).fit(snapshots, rest_pos)[0]
        >>> solver = SolverReduced(model, basis)
        >>> for step in range(n_steps):
        >>>     solver.step(state_in, state_out, dt)
    """
    
    def __init__(
        self,
        model: Any,
        basis: ReducedBasis,
        mass: float = 1.0,
    ):
        """
        Initialize reduced solver.
        
        Args:
            model: Warp Model object (full-order model)
            basis: ReducedBasis from POD
            mass: Particle mass (uniform)
        """
        self.model = model
        self.basis = basis
        self.mass = mass
        
        self.n_modes = basis.n_modes
        self.n_particles = basis.n_particles
        self.n_dof = 2 * self.n_particles
        self.device = model.device
        
        # Convert basis to Warp arrays
        self.modes = wp.array2d(basis.modes.astype(np.float32), dtype=float, device=self.device)
        
        # Rest positions as vec2 array
        rest_pos_2d = basis.rest_position.reshape(-1, 2)
        self.rest_positions = wp.array(rest_pos_2d.astype(np.float32), dtype=wp.vec2, device=self.device)
        
        # Pre-compute reduced mass matrix: M_r = V^T M V
        self.M_r = self._compute_reduced_mass()
        
        # Pre-compute reduced stiffness: K_r = V^T K V
        self.K_r = self._compute_reduced_stiffness()
        
        # Pre-compute reduced damping: D_r = V^T D V
        self.D_r = self._compute_reduced_damping()
        
        # Allocate reduced space arrays
        self.q_r = wp.zeros(self.n_modes, dtype=float, device=self.device)
        self.v_r = wp.zeros(self.n_modes, dtype=float, device=self.device)
        self.f_r = wp.zeros(self.n_modes, dtype=float, device=self.device)
        
        # Full-space force buffer
        self.f_full = wp.zeros(self.n_particles, dtype=wp.vec2, device=self.device)
        
        # System matrix (computed on first step)
        self._A_r = None
        self._lu_piv = None
        self._current_dt = None
        
        print(f"SolverReduced initialized:")
        print(f"  Full DOF: {self.n_dof}")
        print(f"  Reduced DOF: {self.n_modes}")
        print(f"  Compression: {basis.compression_ratio:.1f}x")
        print(f"  Estimated speedup: ~{basis.speedup_estimate:.0f}x")
    
    def _compute_reduced_mass(self) -> np.ndarray:
        """Compute M_r = V^T M V (uniform mass per particle)."""
        modes = self.basis.modes
        
        # Uniform mass matrix: M_diag = mass * I
        # M_r = V^T @ diag(mass) @ V = mass * V^T @ V
        M_r = self.mass * (modes.T @ modes)
        
        # Ensure symmetry and add regularization
        M_r = 0.5 * (M_r + M_r.T)
        M_r += np.eye(self.n_modes) * 1e-6
        
        return M_r.astype(np.float32)
    
    def _compute_reduced_stiffness(self) -> np.ndarray:
        """Compute K_r = V^T K V from spring stiffness."""
        model = self.model
        modes = self.basis.modes
        
        # Build full stiffness matrix
        K = np.zeros((self.n_dof, self.n_dof), dtype=np.float32)
        
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            spring_indices = model.spring_indices.numpy()
            spring_stiffness = model.spring_stiffness.numpy()
            
            for s in range(model.spring_count):
                i = spring_indices[s * 2]
                j = spring_indices[s * 2 + 1]
                k = spring_stiffness[s]
                
                # Isotropic spring: K_block = k * I
                for a in range(2):
                    K[i*2+a, i*2+a] += k
                    K[j*2+a, j*2+a] += k
                    K[i*2+a, j*2+a] -= k
                    K[j*2+a, i*2+a] -= k
        
        # Project: K_r = V^T @ K @ V
        K_r = modes.T @ K @ modes
        K_r = 0.5 * (K_r + K_r.T)
        K_r += np.eye(self.n_modes) * 1e-6
        
        return K_r.astype(np.float32)
    
    def _compute_reduced_damping(self) -> np.ndarray:
        """Compute D_r = V^T D V from spring damping."""
        model = self.model
        modes = self.basis.modes
        
        D = np.zeros((self.n_dof, self.n_dof), dtype=np.float32)
        
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            spring_indices = model.spring_indices.numpy()
            spring_damping = model.spring_damping.numpy()
            
            for s in range(model.spring_count):
                i = spring_indices[s * 2]
                j = spring_indices[s * 2 + 1]
                d = spring_damping[s]
                
                for a in range(2):
                    D[i*2+a, i*2+a] += d
                    D[j*2+a, j*2+a] += d
                    D[i*2+a, j*2+a] -= d
                    D[j*2+a, i*2+a] -= d
        
        D_r = modes.T @ D @ modes
        D_r = 0.5 * (D_r + D_r.T)
        D_r += np.eye(self.n_modes) * 1e-6
        
        return D_r.astype(np.float32)
    
    def _build_system_matrix(self, dt: float) -> np.ndarray:
        """Build A_r = M_r + dt*D_r + dt²*K_r (implicit Euler)"""
        dt2 = dt * dt
        return self.M_r + dt * self.D_r + dt2 * self.K_r
    
    def step(
        self,
        state_in: Any,
        state_out: Any,
        dt: float,
    ):
        """
        Advance simulation by one timestep using reduced-order integration.
        
        Algorithm:
        1. Project state to reduced space
        2. Evaluate forces in full space
        3. Project forces to reduced space
        4. Solve reduced linear system (r x r instead of N x N)
        5. Reconstruct full state
        """
        model = self.model
        
        # Build/update system matrix if needed
        if self._A_r is None or self._current_dt != dt:
            self._A_r = self._build_system_matrix(dt)
            self._lu_piv = scipy_linalg.lu_factor(self._A_r)
            self._current_dt = dt
        
        # 1. Project current state to reduced space
        wp.launch(
            project_positions_to_reduced_kernel,
            dim=self.n_modes,
            inputs=[state_in.particle_q, self.rest_positions, self.modes,
                    self.q_r, self.n_particles, self.n_modes],
            device=self.device
        )
        
        wp.launch(
            project_velocities_to_reduced_kernel,
            dim=self.n_modes,
            inputs=[state_in.particle_qd, self.modes, self.v_r,
                    self.n_particles, self.n_modes],
            device=self.device
        )
        
        # 2. Evaluate forces in full space
        self.f_full.zero_()
        
        # Import force kernels from utils
        from utils import eval_spring_2d, eval_fem_2d, eval_gravity_2d
        
        # Spring forces
        if hasattr(model, 'spring_count') and model.spring_count > 0:
            wp.launch(eval_spring_2d, dim=model.spring_count, inputs=[
                state_in.particle_q, state_in.particle_qd, model.spring_indices,
                model.spring_rest_length, model.spring_stiffness, model.spring_damping,
                self.f_full, model.spring_strains], device=self.device)
        
        # FEM forces (if model has triangles)
        if hasattr(model, 'tri_count') and model.tri_count > 0:
            wp.launch(eval_fem_2d, dim=model.tri_count, inputs=[
                state_in.particle_q, state_in.particle_qd, model.tri_indices,
                model.tri_poses, model.tri_materials, self.f_full, model.tri_strains],
                device=self.device)
        
        # Gravity
        if model.gravity is not None:
            g = model.gravity.numpy()[0]
            grav = wp.zeros(model.particle_count, dtype=wp.vec2, device=self.device)
            wp.launch(eval_gravity_2d, dim=model.particle_count,
                      inputs=[wp.vec2(g[0] * self.mass, g[1] * self.mass)],
                      outputs=[grav], device=self.device)
            # Add gravity to forces
            f_np = self.f_full.numpy()
            f_np += grav.numpy()
            self.f_full.assign(f_np)
        
        # 3. Project forces to reduced space
        wp.launch(
            project_forces_to_reduced_kernel,
            dim=self.n_modes,
            inputs=[self.f_full, self.modes, self.f_r,
                    self.n_particles, self.n_modes],
            device=self.device
        )
        
        # 4. Solve reduced linear system: A_r @ dv_r = dt * f_r
        wp.synchronize_device(self.device)
        
        f_r_np = self.f_r.numpy()
        v_r_np = self.v_r.numpy()
        q_r_np = self.q_r.numpy()
        
        # Solve using LU factorization (O(r²) per solve)
        rhs = dt * f_r_np
        dv_r_np = scipy_linalg.lu_solve(self._lu_piv, rhs)
        
        # Update reduced coordinates
        v_r_np += dv_r_np
        q_r_np += dt * v_r_np
        
        self.v_r.assign(v_r_np)
        self.q_r.assign(q_r_np)
        
        # 5. Reconstruct full state
        wp.launch(
            project_reduced_to_full_kernel,
            dim=self.n_particles,
            inputs=[self.q_r, self.rest_positions, self.modes,
                    state_out.particle_q, self.n_particles, self.n_modes],
            device=self.device
        )
        
        wp.launch(
            project_reduced_to_velocities_kernel,
            dim=self.n_particles,
            inputs=[self.v_r, self.modes, state_out.particle_qd,
                    self.n_particles, self.n_modes],
            device=self.device
        )
        
        # Apply boundary conditions
        from utils import apply_boundary_2d
        wp.launch(apply_boundary_2d, dim=model.particle_count, inputs=[
            state_out.particle_q, state_out.particle_qd, model.boxsize],
            device=self.device)
        
        return state_out
    
    def get_stats(self) -> dict:
        """Get solver statistics."""
        return {
            'n_full': self.n_dof,
            'n_reduced': self.n_modes,
            'compression_ratio': self.basis.compression_ratio,
            'speedup_estimate': self.basis.speedup_estimate,
        }
