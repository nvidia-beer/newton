"""
Shared utilities for Newton Warp tutorials.

This module contains common code used across the tutorial notebooks:
- Warp kernels for physics simulation (springs, FEM, collisions)
- State and Model classes for simulation setup
- Solver classes (Explicit, Implicit, ImplicitFEM)
- Visualization helpers

Usage:
    from utils import (
        State, Model, SolverExplicit, SolverImplicit, SolverImplicitFEM,
        run_simulation, get_colors, eval_spring_2d, apply_boundary_2d, ...
    )
"""

import numpy as np
import warp as wp
from warp.optim.linear import bicgstab, preconditioner
from warp.sparse import bsr_zeros, bsr_set_from_triplets
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
import time


# =============================================================================
# WARP KERNELS - Spring Physics
# =============================================================================

@wp.kernel
def eval_spring_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec2),
    spring_strains: wp.array(dtype=float),
):
    """Evaluate spring forces between connected particles.
    
    Computes spring forces using Hooke's law with damping:
    F = k * (length - rest_length) + d * relative_velocity
    """
    tid = wp.tid()
    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]
    if i == -1 or j == -1:
        return
    
    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]
    
    xij = x[i] - x[j]
    vij = v[i] - v[j]
    L = wp.length(xij)
    
    if L < 1e-6:
        spring_strains[tid] = 0.0
        return
    
    d_hat = xij / L
    extension = L - rest
    L_dot = wp.dot(d_hat, vij)
    
    spring_strains[tid] = extension / rest
    force = d_hat * (ke * extension + kd * L_dot)
    
    wp.atomic_sub(f, i, force)
    wp.atomic_add(f, j, force)


# =============================================================================
# WARP KERNELS - Integration (Explicit Euler)
# =============================================================================

@wp.kernel
def integrate_particles_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    f: wp.array(dtype=wp.vec2),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec2,
    dt: float,
    x_new: wp.array(dtype=wp.vec2),
    v_new: wp.array(dtype=wp.vec2),
):
    """First half of velocity Verlet integration."""
    tid = wp.tid()
    acc = f[tid] * inv_mass[tid] + gravity
    v_half = v[tid] + acc * (dt / 2.0)
    x_new[tid] = x[tid] + v_half * dt
    v_new[tid] = v_half


@wp.kernel
def finalize_velocity_2d(
    v: wp.array(dtype=wp.vec2),
    f: wp.array(dtype=wp.vec2),
    inv_mass: wp.array(dtype=float),
    gravity: wp.vec2,
    dt: float,
    v_new: wp.array(dtype=wp.vec2),
):
    """Second half of velocity Verlet integration."""
    tid = wp.tid()
    acc = f[tid] * inv_mass[tid] + gravity
    v_new[tid] = v[tid] + acc * (dt / 2.0)


@wp.kernel
def apply_boundary_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    boxsize: float
):
    """Apply box boundary collision with reflection."""
    tid = wp.tid()
    pos = x[tid]
    vel = v[tid]
    if pos[0] < 0.0:
        pos = wp.vec2(-pos[0], pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    elif pos[0] > boxsize:
        pos = wp.vec2(2.0 * boxsize - pos[0], pos[1])
        vel = wp.vec2(-vel[0], vel[1])
    if pos[1] < 0.0:
        pos = wp.vec2(pos[0], -pos[1])
        vel = wp.vec2(vel[0], -vel[1])
    elif pos[1] > boxsize:
        pos = wp.vec2(pos[0], 2.0 * boxsize - pos[1])
        vel = wp.vec2(vel[0], -vel[1])
    x[tid] = pos
    v[tid] = vel


# =============================================================================
# WARP KERNELS - Implicit Solver Matrix Building
# =============================================================================

@wp.kernel
def build_spring_matrix_2d(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat22f),
    indices: wp.array(dtype=int),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    dt: float,
    mass: float,
    Minv: float
):
    """Build stiffness matrix contributions from springs.
    
    For system (M + dt²K)Δv = dt*f, each spring contributes:
    - Diagonal blocks: mass + dt²*k (self-stiffness, positive)
    - Off-diagonal blocks: -dt²*k (coupling between particles, negative)
    """
    tid = wp.tid()
    i = indices[tid * 2 + 0]
    j = indices[tid * 2 + 1]
    k = spring_stiffness[tid]
    d = spring_damping[tid]
    dt2 = dt * dt
    
    # Diagonal: M + dt²K (positive-definite)
    diag = mass + dt * d + dt2 * k
    block_diag = wp.mat22f(diag, 0.0, 0.0, diag)
    
    # Off-diagonal: -dt²K coupling (negative)
    block_off = wp.mat22f(-dt2 * k, 0.0, 0.0, -dt2 * k)
    
    # Store 4 blocks per spring
    base = tid * 4
    rows[base] = i
    cols[base] = i
    values[base] = block_diag
    rows[base + 1] = j
    cols[base + 1] = j
    values[base + 1] = block_diag
    rows[base + 2] = i
    cols[base + 2] = j
    values[base + 2] = block_off
    rows[base + 3] = j
    cols[base + 3] = i
    values[base + 3] = block_off


@wp.kernel
def update_state_2d(
    dv: wp.array(dtype=wp.vec2),
    dt: float,
    pos_in: wp.array(dtype=wp.vec2),
    vel_in: wp.array(dtype=wp.vec2),
    pos_out: wp.array(dtype=wp.vec2),
    vel_out: wp.array(dtype=wp.vec2)
):
    """Update positions and velocities from velocity change."""
    tid = wp.tid()
    vel = vel_in[tid] + dv[tid]
    pos_out[tid] = pos_in[tid] + vel * dt
    vel_out[tid] = vel


@wp.kernel
def eval_gravity_2d(
    gravity: wp.vec2,
    forces: wp.array(dtype=wp.vec2)
):
    """Apply gravity to all particles."""
    tid = wp.tid()
    forces[tid] = gravity


# =============================================================================
# WARP KERNELS - FEM (Finite Element Method)
# =============================================================================

@wp.kernel
def eval_fem_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    indices: wp.array(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    materials: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec2),
    tri_strains: wp.array(dtype=float),
):
    """Evaluate FEM forces for triangular elements.
    
    Uses a Neo-Hookean-like material model with:
    - mu: shear modulus
    - lambda: bulk modulus
    - damping: velocity-dependent damping
    """
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]
    
    mat = materials[tid]
    k_mu = mat[0]
    k_lambda = mat[1]
    k_damp = mat[2]
    
    x0, x1, x2 = x[i], x[j], x[k]
    v0, v1, v2 = v[i], v[j], v[k]
    x10, x20 = x1 - x0, x2 - x0
    v10, v20 = v1 - v0, v2 - v0
    
    Ds = wp.mat22(x10[0], x20[0], x10[1], x20[1])
    Dm = pose[tid]
    det_Dm = wp.determinant(Dm)
    rest_area = 0.5 / wp.abs(det_Dm)
    
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area
    
    F = Ds * Dm
    dFdt = wp.mat22(v10[0], v20[0], v10[1], v20[1]) * Dm
    
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.max(wp.dot(col1, col1) + wp.dot(col2, col2), 0.01)
    
    P = F * k_mu * (Ic - 2.0) / Ic + dFdt * k_damp
    H = P * wp.transpose(Dm)
    f1 = wp.vec2(H[0, 0], H[1, 0])
    f2 = wp.vec2(H[0, 1], H[1, 1])
    
    J = wp.determinant(F)
    dJdx1 = wp.vec2(x20[1], -x20[0]) * det_Dm
    dJdx2 = wp.vec2(-x10[1], x10[0]) * det_Dm
    
    f_vol = (J - 1.0) * k_lambda
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2)) * k_damp
    f_total = f_vol + f_damp
    
    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f0 = -(f1 + f2)
    
    tri_strains[tid] = J - 1.0
    
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


@wp.kernel
def build_fem_matrix_2d(
    x: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=int),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=wp.vec3),
    fem_rows: wp.array(dtype=wp.int32),
    fem_cols: wp.array(dtype=wp.int32),
    fem_values: wp.array(dtype=wp.mat22f),
    dt: float
):
    """Build stiffness matrix contributions from FEM triangles."""
    tid = wp.tid()
    i = tri_indices[tid * 3 + 0]
    j = tri_indices[tid * 3 + 1]
    k = tri_indices[tid * 3 + 2]
    mat = tri_materials[tid]
    k_mu = mat[0]
    k_lambda = mat[1]
    
    x10 = x[j] - x[i]
    x20 = x[k] - x[i]
    Ds = wp.mat22(x10[0], x20[0], x10[1], x20[1])
    Dm = tri_poses[tid]
    det_Dm = wp.determinant(Dm)
    area = 0.5 / wp.abs(det_Dm)
    k_mu = k_mu * area
    k_lambda = k_lambda * area
    
    F = Ds * Dm
    col1 = wp.vec2(F[0, 0], F[1, 0])
    col2 = wp.vec2(F[0, 1], F[1, 1])
    Ic = wp.max(wp.dot(col1, col1) + wp.dot(col2, col2), 0.1)
    J = wp.determinant(F)
    K_eff = k_mu * (1.0 + 4.0 / Ic) + k_lambda * (1.0 + 2.0 * wp.abs(J - 1.0))
    K_sc = dt * dt * K_eff
    
    # Simplified: use scalar stiffness for all blocks
    block = wp.mat22f(K_sc, 0.0, 0.0, K_sc)
    block_neg = wp.mat22f(-K_sc * 0.5, 0.0, 0.0, -K_sc * 0.5)
    
    # 9 blocks per triangle: (i,i), (j,j), (k,k), (i,j), (j,i), (i,k), (k,i), (j,k), (k,j)
    base = tid * 9
    fem_rows[base + 0] = i
    fem_cols[base + 0] = i
    fem_values[base + 0] = block
    fem_rows[base + 1] = j
    fem_cols[base + 1] = j
    fem_values[base + 1] = block
    fem_rows[base + 2] = k
    fem_cols[base + 2] = k
    fem_values[base + 2] = block
    fem_rows[base + 3] = i
    fem_cols[base + 3] = j
    fem_values[base + 3] = block_neg
    fem_rows[base + 4] = j
    fem_cols[base + 4] = i
    fem_values[base + 4] = block_neg
    fem_rows[base + 5] = i
    fem_cols[base + 5] = k
    fem_values[base + 5] = block_neg
    fem_rows[base + 6] = k
    fem_cols[base + 6] = i
    fem_values[base + 6] = block_neg
    fem_rows[base + 7] = j
    fem_cols[base + 7] = k
    fem_values[base + 7] = block_neg
    fem_rows[base + 8] = k
    fem_cols[base + 8] = j
    fem_values[base + 8] = block_neg


# =============================================================================
# WARP KERNELS - SDF Collision
# =============================================================================

@wp.func
def bilinear_sample(
    data: wp.array2d(dtype=float),
    px: float,
    py: float,
    width: int,
    height: int,
) -> float:
    """Sample 2D array with bilinear interpolation."""
    px = wp.clamp(px, 0.0, float(width) - 1.001)
    py = wp.clamp(py, 0.0, float(height) - 1.001)
    
    x0 = int(px)
    y0 = int(py)
    x1 = wp.min(x0 + 1, width - 1)
    y1 = wp.min(y0 + 1, height - 1)
    
    fx = px - float(x0)
    fy = py - float(y0)
    
    v00 = data[y0, x0]
    v01 = data[y0, x1]
    v10 = data[y1, x0]
    v11 = data[y1, x1]
    
    v0 = v00 * (1.0 - fx) + v01 * fx
    v1 = v10 * (1.0 - fx) + v11 * fx
    
    return v0 * (1.0 - fy) + v1 * fy


@wp.kernel
def apply_sdf_collision_2d(
    x: wp.array(dtype=wp.vec2),
    v: wp.array(dtype=wp.vec2),
    sdf: wp.array2d(dtype=float),
    sdf_grad_x: wp.array2d(dtype=float),
    sdf_grad_y: wp.array2d(dtype=float),
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
    restitution: float,
    colliding: wp.array(dtype=int),
):
    """Apply SDF-based collision detection and response."""
    tid = wp.tid()
    colliding[tid] = 0
    
    pos = x[tid]
    vel = v[tid]
    
    # World to pixel coordinates
    px = (pos[0] - origin_x) * resolution
    py = (pos[1] - origin_y) * resolution
    
    # Bounds check
    if px < 0.0 or px >= float(width) or py < 0.0 or py >= float(height):
        world_w = float(width) / resolution
        world_h = float(height) / resolution
        
        if pos[0] < origin_x:
            pos = wp.vec2(origin_x + 0.01, pos[1])
            vel = wp.vec2(wp.abs(vel[0]) * restitution, vel[1])
        elif pos[0] > origin_x + world_w:
            pos = wp.vec2(origin_x + world_w - 0.01, pos[1])
            vel = wp.vec2(-wp.abs(vel[0]) * restitution, vel[1])
        
        if pos[1] < origin_y:
            pos = wp.vec2(pos[0], origin_y + 0.01)
            vel = wp.vec2(vel[0], wp.abs(vel[1]) * restitution)
        elif pos[1] > origin_y + world_h:
            pos = wp.vec2(pos[0], origin_y + world_h - 0.01)
            vel = wp.vec2(vel[0], -wp.abs(vel[1]) * restitution)
        
        x[tid] = pos
        v[tid] = vel
        return
    
    # Sample SDF
    sdf_value = bilinear_sample(sdf, px, py, width, height) / resolution
    
    # Mark as colliding when close to terrain
    if sdf_value < 0.1:
        colliding[tid] = 1
    
    # Collision response
    if sdf_value < 0.0:
        grad_x = bilinear_sample(sdf_grad_x, px, py, width, height)
        grad_y = bilinear_sample(sdf_grad_y, px, py, width, height)
        grad_len = wp.sqrt(grad_x * grad_x + grad_y * grad_y)
        
        if grad_len > 1e-6:
            nx = grad_x / grad_len
            ny = grad_y / grad_len
            normal = wp.vec2(nx, ny)
            
            # Push out of collision
            penetration = -sdf_value
            pos = pos + normal * penetration * 1.1
            
            # Reflect velocity
            vn = wp.dot(vel, normal)
            if vn < 0.0:
                vel = vel - normal * vn * (1.0 + restitution)
            
            x[tid] = pos
            v[tid] = vel


# =============================================================================
# STATE AND MODEL CLASSES
# =============================================================================

class State:
    """Simulation state containing particle positions, velocities, and forces."""
    
    def __init__(self):
        self.particle_q = None   # Positions
        self.particle_qd = None  # Velocities
        self.particle_f = None   # Forces


class Model:
    """Physics model containing particles, springs, and optionally FEM triangles."""
    
    def __init__(self, device='cuda'):
        self.device = wp.get_device(device)
        
        # Particle data
        self.particle_q = None
        self.particle_qd = None
        self.particle_mass = None
        self.particle_inv_mass = None
        self.particle_count = 0
        
        # Spring data
        self.spring_indices = None
        self.spring_rest_length = None
        self.spring_stiffness = None
        self.spring_damping = None
        self.spring_count = 0
        self.spring_strains = None
        
        # FEM triangle data (optional)
        self.tri_indices = None
        self.tri_count = 0
        self.tri_materials = None
        self.tri_poses = None
        self.tri_strains = None
        
        # Global properties
        self.gravity = None
        self.boxsize = 3.0
    
    def state(self):
        """Create a new state initialized from model."""
        s = State()
        if self.particle_count > 0:
            s.particle_q = wp.clone(self.particle_q)
            s.particle_qd = wp.clone(self.particle_qd)
            s.particle_f = wp.zeros(self.particle_count, dtype=wp.vec2, device=self.device)
        return s
    
    def set_gravity(self, g):
        """Set gravity vector."""
        if self.gravity is None:
            self.gravity = wp.zeros(1, dtype=wp.vec2, device=self.device)
        self.gravity.assign([wp.vec2(g[0], g[1])])
    
    @classmethod
    def from_circle(cls, radius=0.5, num_boundary=20, num_rings=3, device='cuda', 
                    boxsize=3.0, center=None, spring_stiffness=1000.0, spring_damping=5.0,
                    fem_mu=500.0, fem_lambda=500.0, fem_damping=10.0, use_fem=False):
        """Create a soft body from a circular mesh.
        
        Args:
            radius: Circle radius
            num_boundary: Number of points on boundary
            num_rings: Number of concentric rings inside
            device: Warp device
            boxsize: Simulation box size
            center: Center position (defaults to box center)
            spring_stiffness: Spring stiffness coefficient
            spring_damping: Spring damping coefficient
            fem_mu: FEM shear modulus (if use_fem=True)
            fem_lambda: FEM bulk modulus (if use_fem=True)
            fem_damping: FEM damping (if use_fem=True)
            use_fem: Whether to create FEM triangles
        """
        model = cls(device=device)
        model.boxsize = boxsize
        if center is None:
            center = (boxsize / 2.0, boxsize / 2.0)
        cx, cy = center
        
        # Generate points in concentric rings
        all_pts = []
        angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
        all_pts.append(np.c_[radius * np.cos(angles), radius * np.sin(angles)])
        for ring in range(1, num_rings + 1):
            r = radius * (num_rings - ring + 1) / (num_rings + 1)
            n = max(8, int(num_boundary * r / radius))
            a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / num_boundary * ring
            all_pts.append(np.c_[r * np.cos(a), r * np.sin(a)])
        all_pts.append([[0.0, 0.0]])  # Center point
        
        pts = np.vstack(all_pts) + np.array([cx, cy])
        model.particle_count = len(pts)
        
        # Triangulate for springs (and optionally FEM)
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                e = tuple(sorted([simplex[k], simplex[(k + 1) % 3]]))
                edges.add(e)
        edges = list(edges)
        model.spring_count = len(edges)
        
        # Compute rest lengths
        rest_lengths = [np.linalg.norm(pts[e[0]] - pts[e[1]]) for e in edges]
        
        # Create Warp arrays
        model.particle_q = wp.array(pts.astype(np.float32), dtype=wp.vec2, device=device)
        model.particle_qd = wp.zeros(model.particle_count, dtype=wp.vec2, device=device)
        model.particle_mass = wp.ones(model.particle_count, dtype=float, device=device)
        model.particle_inv_mass = wp.ones(model.particle_count, dtype=float, device=device)
        
        model.spring_indices = wp.array(np.array(edges).flatten(), dtype=int, device=device)
        model.spring_rest_length = wp.array(np.array(rest_lengths, dtype=np.float32), dtype=float, device=device)
        model.spring_stiffness = wp.full(model.spring_count, spring_stiffness, dtype=float, device=device)
        model.spring_damping = wp.full(model.spring_count, spring_damping, dtype=float, device=device)
        model.spring_strains = wp.zeros(model.spring_count, dtype=float, device=device)
        
        # Optional FEM triangles
        if use_fem:
            # Ensure consistent CCW winding for all triangles (critical for FEM!)
            valid_tris = []
            for simplex in tri.simplices:
                p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                # Cross product determines winding: positive = CCW, negative = CW
                cross = (p1 - p0)[0] * (p2 - p0)[1] - (p1 - p0)[1] * (p2 - p0)[0]
                if abs(cross) >= 1e-8:  # Skip degenerate triangles
                    if cross < 0:
                        simplex = [simplex[0], simplex[2], simplex[1]]  # Flip to CCW
                    valid_tris.append(simplex)
            
            model.tri_indices = wp.array(np.array(valid_tris).flatten(), dtype=int, device=device)
            model.tri_count = len(valid_tris)
            
            # Compute rest poses (inverse of initial deformation gradient)
            poses = []
            for simplex in valid_tris:
                p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                x10, x20 = p1 - p0, p2 - p0
                Dm = np.array([[x10[0], x20[0]], [x10[1], x20[1]]])
                poses.append(np.linalg.inv(Dm).astype(np.float32))
            model.tri_poses = wp.array(np.array(poses), dtype=wp.mat22, device=device)
            
            # Material properties: (mu, lambda, damping)
            materials = np.array([[fem_mu, fem_lambda, fem_damping]] * model.tri_count, dtype=np.float32)
            model.tri_materials = wp.array(materials, dtype=wp.vec3, device=device)
            model.tri_strains = wp.zeros(model.tri_count, dtype=float, device=device)
        
        # Initialize gravity
        model.gravity = wp.zeros(1, dtype=wp.vec2, device=device)
        model.gravity.assign([wp.vec2(0.0, -9.8)])
        
        print(f"Model: {model.particle_count} particles, {model.spring_count} springs" + 
              (f", {model.tri_count} triangles" if use_fem else ""))
        
        return model


# =============================================================================
# SOLVER CLASSES
# =============================================================================

class SolverExplicit:
    """Explicit Euler solver - springs only, conditionally stable.
    
    Stability requires: dt < 2 * sqrt(m / k)
    """
    
    def __init__(self, model):
        self.model = model
    
    def step(self, state_in, state_out, dt):
        """Perform one explicit time step."""
        m = self.model
        state_in.particle_f.zero_()
        
        # Evaluate spring forces
        wp.launch(eval_spring_2d, dim=m.spring_count, inputs=[
            state_in.particle_q, state_in.particle_qd, m.spring_indices,
            m.spring_rest_length, m.spring_stiffness, m.spring_damping,
            state_in.particle_f, m.spring_strains], device=m.device)
        
        g = m.gravity.numpy()[0]
        grav = wp.vec2(g[0], g[1])
        
        # First half-step
        wp.launch(integrate_particles_2d, dim=m.particle_count, inputs=[
            state_in.particle_q, state_in.particle_qd, state_in.particle_f,
            m.particle_inv_mass, grav, dt],
            outputs=[state_out.particle_q, state_out.particle_qd], device=m.device)
        
        # Apply boundaries
        wp.launch(apply_boundary_2d, dim=m.particle_count, inputs=[
            state_out.particle_q, state_out.particle_qd, m.boxsize], device=m.device)
        
        # Re-evaluate forces at new position
        state_in.particle_f.zero_()
        wp.launch(eval_spring_2d, dim=m.spring_count, inputs=[
            state_out.particle_q, state_out.particle_qd, m.spring_indices,
            m.spring_rest_length, m.spring_stiffness, m.spring_damping,
            state_in.particle_f, m.spring_strains], device=m.device)
        
        # Finalize velocity
        wp.launch(finalize_velocity_2d, dim=m.particle_count, inputs=[
            state_out.particle_qd, state_in.particle_f, m.particle_inv_mass, grav, dt],
            outputs=[state_out.particle_qd], device=m.device)
        
        return state_out


class SolverImplicit:
    """Implicit solver using BiCGSTAB - unconditionally stable."""
    
    def __init__(self, model, mass=1.0):
        self.model = model
        self.mass = mass
        self.Minv = 1.0 / mass
        
        # Allocate sparse matrix storage
        spr_blk = model.spring_count * 4
        
        self.spr_rows = wp.zeros(spr_blk, dtype=wp.int32, device=model.device)
        self.spr_cols = wp.zeros(spr_blk, dtype=wp.int32, device=model.device)
        self.spr_vals = wp.zeros(spr_blk, dtype=wp.mat22f, device=model.device)
        
        self.A = bsr_zeros(model.particle_count, model.particle_count, wp.mat22f, device=model.device)
        self.M = None
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
    
    def build_matrix(self, state, dt):
        """Assemble the system matrix."""
        m = self.model
        
        wp.launch(build_spring_matrix_2d, dim=m.spring_count, inputs=[
            self.spr_rows, self.spr_cols, self.spr_vals, m.spring_indices,
            m.spring_stiffness, m.spring_damping, wp.float32(dt), self.mass, self.Minv], 
            device=m.device)
        
        bsr_set_from_triplets(self.A, self.spr_rows, self.spr_cols, self.spr_vals, 
                              prune_numerical_zeros=True)
        
        self.M = preconditioner(self.A, ptype="diag")
    
    def step(self, state_in, state_out, dt):
        """Perform one implicit time step."""
        m = self.model
        
        self.build_matrix(state_in, dt)
        
        # Evaluate forces
        spr_f = wp.zeros(m.particle_count, dtype=wp.vec2, device=m.device)
        wp.launch(eval_spring_2d, dim=m.spring_count, inputs=[
            state_in.particle_q, state_in.particle_qd, m.spring_indices,
            m.spring_rest_length, m.spring_stiffness, m.spring_damping,
            spr_f, m.spring_strains], device=m.device)
        
        # Add gravity
        g = m.gravity.numpy()[0]
        rhs = wp.zeros(m.particle_count, dtype=wp.vec2, device=m.device)
        wp.launch(eval_gravity_2d, dim=m.particle_count, inputs=[wp.vec2(g[0], g[1])], 
                  outputs=[rhs], device=m.device)
        
        # RHS = dt * (gravity + spring_forces)
        rhs_np = rhs.numpy() + spr_f.numpy()
        rhs_np *= dt
        rhs.assign(rhs_np)
        
        # Solve linear system
        self.dv.zero_()
        bicgstab(self.A, rhs, self.dv, tol=1e-4, maxiter=50, M=self.M)
        
        # Update state
        wp.launch(update_state_2d, dim=m.particle_count, inputs=[
            self.dv, wp.float32(dt), state_in.particle_q, state_in.particle_qd],
            outputs=[state_out.particle_q, state_out.particle_qd], device=m.device)
        
        wp.launch(apply_boundary_2d, dim=m.particle_count, inputs=[
            state_out.particle_q, state_out.particle_qd, m.boxsize], device=m.device)
        
        return state_out


class SolverImplicitFEM:
    """Implicit FEM solver with springs."""
    
    def __init__(self, model, mass=1.0):
        self.model = model
        self.mass = mass
        self.Minv = 1.0 / mass
        
        spr_blk = model.spring_count * 4
        fem_blk = model.tri_count * 9
        total = spr_blk + fem_blk
        
        self.spr_rows = wp.zeros(spr_blk, dtype=wp.int32, device=model.device)
        self.spr_cols = wp.zeros(spr_blk, dtype=wp.int32, device=model.device)
        self.spr_vals = wp.zeros(spr_blk, dtype=wp.mat22f, device=model.device)
        self.fem_rows = wp.zeros(fem_blk, dtype=wp.int32, device=model.device)
        self.fem_cols = wp.zeros(fem_blk, dtype=wp.int32, device=model.device)
        self.fem_vals = wp.zeros(fem_blk, dtype=wp.mat22f, device=model.device)
        self.bsr_rows = wp.zeros(total, dtype=wp.int32, device=model.device)
        self.bsr_cols = wp.zeros(total, dtype=wp.int32, device=model.device)
        self.bsr_vals = wp.zeros(total, dtype=wp.mat22f, device=model.device)
        
        self.A = bsr_zeros(model.particle_count, model.particle_count, wp.mat22f, device=model.device)
        self.M = None
        self.dv = wp.zeros(model.particle_count, dtype=wp.vec2, device=model.device)
    
    def build_matrix(self, state, dt):
        """Assemble the system matrix with springs and FEM."""
        m = self.model
        
        wp.launch(build_spring_matrix_2d, dim=m.spring_count, inputs=[
            self.spr_rows, self.spr_cols, self.spr_vals, m.spring_indices,
            m.spring_stiffness, m.spring_damping, wp.float32(dt), self.mass, self.Minv], 
            device=m.device)
        
        wp.launch(build_fem_matrix_2d, dim=m.tri_count, inputs=[
            state.particle_q, m.tri_indices, m.tri_poses, m.tri_materials,
            self.fem_rows, self.fem_cols, self.fem_vals, wp.float32(dt)], device=m.device)
        
        # Concatenate triplets
        sn, fn = m.spring_count * 4, m.tri_count * 9
        wp.copy(self.bsr_rows, self.spr_rows, count=sn)
        wp.copy(self.bsr_cols, self.spr_cols, count=sn)
        wp.copy(self.bsr_vals, self.spr_vals, count=sn)
        wp.copy(self.bsr_rows, self.fem_rows, dest_offset=sn, count=fn)
        wp.copy(self.bsr_cols, self.fem_cols, dest_offset=sn, count=fn)
        wp.copy(self.bsr_vals, self.fem_vals, dest_offset=sn, count=fn)
        
        bsr_set_from_triplets(self.A, self.bsr_rows, self.bsr_cols, self.bsr_vals,
                              prune_numerical_zeros=True)
        self.M = preconditioner(self.A, ptype="diag")
    
    def step(self, state_in, state_out, dt):
        """Perform one implicit FEM time step."""
        m = self.model
        
        self.build_matrix(state_in, dt)
        
        # Evaluate all forces
        forces = wp.zeros(m.particle_count, dtype=wp.vec2, device=m.device)
        
        wp.launch(eval_spring_2d, dim=m.spring_count, inputs=[
            state_in.particle_q, state_in.particle_qd, m.spring_indices,
            m.spring_rest_length, m.spring_stiffness, m.spring_damping,
            forces, m.spring_strains], device=m.device)
        
        wp.launch(eval_fem_2d, dim=m.tri_count, inputs=[
            state_in.particle_q, state_in.particle_qd, m.tri_indices,
            m.tri_poses, m.tri_materials, forces, m.tri_strains], device=m.device)
        
        # Add gravity
        g = m.gravity.numpy()[0]
        rhs = wp.zeros(m.particle_count, dtype=wp.vec2, device=m.device)
        wp.launch(eval_gravity_2d, dim=m.particle_count, inputs=[wp.vec2(g[0], g[1])],
                  outputs=[rhs], device=m.device)
        
        rhs_np = rhs.numpy() + forces.numpy()
        rhs_np *= dt
        rhs.assign(rhs_np)
        
        # Solve
        self.dv.zero_()
        bicgstab(self.A, rhs, self.dv, tol=1e-4, maxiter=50, M=self.M)
        
        wp.launch(update_state_2d, dim=m.particle_count, inputs=[
            self.dv, wp.float32(dt), state_in.particle_q, state_in.particle_qd],
            outputs=[state_out.particle_q, state_out.particle_qd], device=m.device)
        
        wp.launch(apply_boundary_2d, dim=m.particle_count, inputs=[
            state_out.particle_q, state_out.particle_qd, m.boxsize], device=m.device)
        
        return state_out


class SolverImplicitFEM_SDF(SolverImplicitFEM):
    """Implicit FEM solver with SDF terrain collision."""
    
    def __init__(self, model, sdf_data, dt=0.01, mass=1.0, restitution=0.3):
        super().__init__(model, mass)
        self.sdf_data = sdf_data
        self.restitution = restitution
        self.colliding = wp.zeros(model.particle_count, dtype=int, device=model.device)
    
    def step(self, state_in, state_out, dt):
        """Perform one step with SDF collision."""
        # Do normal FEM step
        super().step(state_in, state_out, dt)
        
        # Apply SDF collision
        m = self.model
        sd = self.sdf_data
        
        wp.launch(apply_sdf_collision_2d, dim=m.particle_count, inputs=[
            state_out.particle_q, state_out.particle_qd,
            sd['sdf'], sd['sdf_grad_x'], sd['sdf_grad_y'],
            sd['resolution'], sd['origin_x'], sd['origin_y'],
            sd['width'], sd['height'], self.restitution, self.colliding], device=m.device)
        
        return state_out


# =============================================================================
# SIMULATION HELPERS
# =============================================================================

def run_simulation(model, solver, dt, sim_time, frame_interval, center, name="", 
                   explosion_threshold=10.0):
    """Run simulation and capture frames.
    
    Args:
        model: Physics model
        solver: Solver instance
        dt: Time step
        sim_time: Total simulation time
        frame_interval: Steps between captured frames
        center: Initial center position (for explosion detection)
        name: Name for logging
        explosion_threshold: Max displacement before marking as exploded
    
    Returns:
        tuple: (frames, elapsed_time, exploded)
    """
    steps = int(sim_time / dt)
    s_in, s_out = model.state(), model.state()
    print(f"Running {steps:,} {name} steps (dt={dt*1000:.2f}ms)...")
    start = time.time()
    frames = []
    exploded = False
    
    for step in range(steps):
        solver.step(s_in, s_out, dt)
        s_in, s_out = s_out, s_in
        
        pos = s_in.particle_q.numpy()
        
        # Detect explosion
        if not exploded:
            max_displacement = np.max(np.abs(pos - np.array([center[0], center[1]])))
            if max_displacement > explosion_threshold or np.any(np.isnan(pos)):
                print(f"  ⚠️ EXPLOSION DETECTED at step {step} (t={step*dt:.3f}s)!")
                exploded = True
        
        # Stop if simulation crashes
        if np.any(np.isnan(pos)) or np.any(np.abs(pos) > 50):
            print(f"  💥 SIMULATION CRASHED at step {step}")
            break
        
        if step % frame_interval == 0:
            frame_data = [
                pos.copy(),
                model.spring_strains.numpy().copy(),
                step * dt,
                exploded
            ]
            # Include triangle strains if available
            if hasattr(model, 'tri_strains') and model.tri_strains is not None:
                frame_data.insert(2, model.tri_strains.numpy().copy())
            frames.append(tuple(frame_data))
    
    elapsed = time.time() - start
    status = "EXPLODED" if exploded else "STABLE"
    print(f"  {name}: {len(frames)} frames captured, status: {status}, time: {elapsed:.2f}s")
    return frames, elapsed, exploded


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def get_colors(pos, boxsize, collision_thresh=0.15):
    """Get particle colors based on boundary proximity.
    
    Returns pink for particles near boundary, light blue otherwise.
    """
    return ['#FF69B4' if (p[0] < collision_thresh or p[0] > boxsize - collision_thresh or 
                          p[1] < collision_thresh or p[1] > boxsize - collision_thresh) 
            else '#87CEEB' for p in pos]


def create_spring_colormap():
    """Create colormap for spring strain visualization."""
    return LinearSegmentedColormap.from_list('spring', ['#FFE066', '#FF8800', '#CC0000'])


def create_fem_colormap():
    """Create colormap for FEM strain visualization."""
    return LinearSegmentedColormap.from_list('fem', ['#88DDAA', '#00AACC', '#0044AA'])


def draw_soft_body(ax, pos, spring_idx, strain, model, cmap_spring=None, title="", 
                   tri_idx=None, tri_strain=None, cmap_fem=None):
    """Draw soft body with springs and optional FEM triangles.
    
    Args:
        ax: Matplotlib axis
        pos: Particle positions
        spring_idx: Spring indices (N, 2)
        strain: Spring strains
        model: Model with boxsize
        cmap_spring: Spring colormap (default: yellow-orange-red)
        title: Plot title
        tri_idx: Triangle indices for FEM (optional)
        tri_strain: Triangle strains for FEM (optional)
        cmap_fem: FEM colormap (optional)
    """
    ax.clear()
    ax.set_facecolor('white')
    
    if cmap_spring is None:
        cmap_spring = create_spring_colormap()
    
    # Draw FEM triangles if provided
    if tri_idx is not None and tri_strain is not None:
        if cmap_fem is None:
            cmap_fem = create_fem_colormap()
        triang = Triangulation(pos[:, 0], pos[:, 1], tri_idx)
        ax.tripcolor(triang, np.clip(np.abs(tri_strain) / max(0.1, np.abs(tri_strain).max()), 0, 1),
                     cmap=cmap_fem, alpha=0.85)
    
    # Draw springs
    segs = [[pos[i], pos[j]] for i, j in spring_idx]
    lc = LineCollection(segs, cmap=cmap_spring, linewidths=2.5)
    lc.set_array(np.clip(np.abs(strain) / max(0.02, np.abs(strain).max()), 0, 1))
    ax.add_collection(lc)
    
    # Draw particles
    colors = get_colors(pos, model.boxsize)
    ax.scatter(pos[:, 0], pos[:, 1], s=35, c=colors, edgecolors='white', linewidths=0.8, zorder=5)
    
    ax.set_xlim(0, model.boxsize)
    ax.set_ylim(0, model.boxsize)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold', color='black')


# =============================================================================
# SDF / WORLD MAP
# =============================================================================

class WorldMap:
    """Load terrain from bitmap and compute SDF for collision."""
    
    def __init__(self, image_path, resolution=50.0, origin=(0.0, 0.0)):
        from scipy import ndimage
        from PIL import Image as PILImage
        
        self.resolution = resolution
        self.origin = np.array(origin)
        
        # Load image (white = passable, black = wall)
        img = PILImage.open(image_path).convert('L')
        img_array = np.array(img)
        img_array = np.flipud(img_array)  # y=0 at bottom
        self.bitmap = (img_array >= 128).astype(np.float32)
        
        # Compute SDF
        dist_passable = ndimage.distance_transform_edt(self.bitmap)
        dist_wall = ndimage.distance_transform_edt(1 - self.bitmap)
        self.sdf = (dist_passable - dist_wall).astype(np.float32)
        
        # Precompute gradient
        self.sdf_grad_x = ndimage.sobel(self.sdf, axis=1).astype(np.float32)
        self.sdf_grad_y = ndimage.sobel(self.sdf, axis=0).astype(np.float32)
        
        self.width = self.bitmap.shape[1]
        self.height = self.bitmap.shape[0]
        self.world_size = (self.width / resolution, self.height / resolution)
        
        print(f"WorldMap: {self.width}x{self.height} px, world size: {self.world_size[0]:.2f}x{self.world_size[1]:.2f}")
    
    def to_warp(self, device='cuda'):
        """Convert to Warp arrays for GPU."""
        return {
            'sdf': wp.array2d(self.sdf, dtype=float, device=device),
            'sdf_grad_x': wp.array2d(self.sdf_grad_x, dtype=float, device=device),
            'sdf_grad_y': wp.array2d(self.sdf_grad_y, dtype=float, device=device),
            'resolution': self.resolution,
            'origin_x': float(self.origin[0]),
            'origin_y': float(self.origin[1]),
            'width': self.width,
            'height': self.height,
        }
