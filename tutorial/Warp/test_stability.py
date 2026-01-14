#!/usr/bin/env python3
"""
Test script to find parameters where:
- Explicit integration EXPLODES
- Implicit integration stays STABLE
"""

import numpy as np
import warp as wp
from scipy.spatial import Delaunay
import time

wp.init()
print(f"Warp {wp.__version__} on {wp.get_device()}")

# ============================================================
# KERNELS (simplified from notebooks)
# ============================================================

@wp.kernel
def eval_spring_2d(
    x: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
    spring_indices: wp.array(dtype=int), spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float), spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec2),
):
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
        return
    
    d_hat = xij / L
    extension = L - rest
    L_dot = wp.dot(d_hat, vij)
    
    force = d_hat * (ke * extension + kd * L_dot)
    
    wp.atomic_sub(f, i, force)
    wp.atomic_add(f, j, force)


@wp.kernel
def integrate_explicit(
    x: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
    f: wp.array(dtype=wp.vec2), inv_mass: float,
    gravity: wp.vec2, dt: float,
    x_new: wp.array(dtype=wp.vec2), v_new: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    acc = f[tid] * inv_mass + gravity
    v_new[tid] = v[tid] + acc * dt
    x_new[tid] = x[tid] + v_new[tid] * dt


@wp.kernel
def integrate_implicit_simple(
    x: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.vec2),
    f: wp.array(dtype=wp.vec2), mass: float,
    spring_stiffness: float, dt: float,
    gravity: wp.vec2,
    x_new: wp.array(dtype=wp.vec2), v_new: wp.array(dtype=wp.vec2),
):
    """Simple implicit integration with diagonal approximation."""
    tid = wp.tid()
    
    # Diagonal implicit: (M + dt^2 * K_diag) * dv = dt * f
    # Approximate K_diag ~ k for spring-connected particles
    # This gives: dv = dt * f / (M + dt^2 * k)
    
    effective_mass = mass + dt * dt * spring_stiffness
    acc = (f[tid] + gravity * mass) / effective_mass
    
    dv = acc * dt
    v_new[tid] = v[tid] + dv
    x_new[tid] = x[tid] + v_new[tid] * dt


# ============================================================
# MODEL SETUP
# ============================================================

def create_circle_model(radius=0.5, num_boundary=16, num_rings=2, 
                        spring_stiffness=100.0, spring_damping=1.0,
                        boxsize=3.0, center=(1.5, 1.5), device='cuda'):
    """Create a simple circle mesh with springs."""
    
    # Generate points
    all_pts = []
    angles = np.linspace(0, 2*np.pi, num_boundary, endpoint=False)
    all_pts.append(np.c_[radius*np.cos(angles), radius*np.sin(angles)])
    for ring in range(1, num_rings+1):
        r = radius * (num_rings-ring+1) / (num_rings+1)
        n = max(6, int(num_boundary * r / radius))
        a = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/num_boundary*ring
        all_pts.append(np.c_[r*np.cos(a), r*np.sin(a)])
    all_pts.append([[0.0, 0.0]])
    pts_norm = np.vstack(all_pts)
    
    # Triangulate
    tri = Delaunay(pts_norm)
    valid_tris = []
    for simplex in tri.simplices:
        cent = np.mean(pts_norm[simplex], axis=0)
        if np.linalg.norm(cent) <= radius * 1.01:
            p0, p1, p2 = pts_norm[simplex]
            cross = (p1-p0)[0]*(p2-p0)[1] - (p1-p0)[1]*(p2-p0)[0]
            if abs(cross) >= 1e-8:
                valid_tris.append(simplex)
    
    pts = pts_norm * radius * boxsize / 2.0
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    
    # Extract edges
    edges = set()
    for t in valid_tris:
        for e in [(t[0],t[1]), (t[1],t[2]), (t[2],t[0])]:
            edges.add(tuple(sorted(e)))
    
    spring_idx, spring_len = [], []
    for v0, v1 in edges:
        spring_idx.extend([v0, v1])
        spring_len.append(np.linalg.norm(pts[v1] - pts[v0]))
    
    n_particles = len(pts)
    n_springs = len(edges)
    
    return {
        'particle_q': wp.array(pts.astype(np.float32), dtype=wp.vec2, device=device),
        'particle_qd': wp.zeros(n_particles, dtype=wp.vec2, device=device),
        'spring_indices': wp.array(np.array(spring_idx, dtype=np.int32), dtype=int, device=device),
        'spring_rest_length': wp.array(np.array(spring_len, dtype=np.float32), dtype=float, device=device),
        'spring_stiffness': wp.full(n_springs, spring_stiffness, dtype=float, device=device),
        'spring_damping': wp.full(n_springs, spring_damping, dtype=float, device=device),
        'n_particles': n_particles,
        'n_springs': n_springs,
        'center': center,
        'stiffness': spring_stiffness,
    }


def test_explicit(stiffness, dt, n_steps=100, device='cuda'):
    """Test explicit integration, return True if stable."""
    model = create_circle_model(spring_stiffness=stiffness, spring_damping=1.0, device=device)
    
    x = wp.clone(model['particle_q'])
    v = wp.clone(model['particle_qd'])
    x_new = wp.zeros_like(x)
    v_new = wp.zeros_like(v)
    f = wp.zeros(model['n_particles'], dtype=wp.vec2, device=device)
    
    gravity = wp.vec2(0.0, -1.0)
    
    for step in range(n_steps):
        f.zero_()
        wp.launch(eval_spring_2d, dim=model['n_springs'], inputs=[
            x, v, model['spring_indices'], model['spring_rest_length'],
            model['spring_stiffness'], model['spring_damping'], f], device=device)
        
        wp.launch(integrate_explicit, dim=model['n_particles'], inputs=[
            x, v, f, 1.0, gravity, dt, x_new, v_new], device=device)
        
        x, x_new = x_new, x
        v, v_new = v_new, v
        
        # Check for explosion
        pos = x.numpy()
        if np.any(np.isnan(pos)) or np.any(np.abs(pos) > 100):
            return False, step
    
    return True, n_steps


def test_implicit(stiffness, dt, n_steps=100, device='cuda'):
    """Test implicit integration (diagonal approximation), return True if stable."""
    model = create_circle_model(spring_stiffness=stiffness, spring_damping=1.0, device=device)
    
    x = wp.clone(model['particle_q'])
    v = wp.clone(model['particle_qd'])
    x_new = wp.zeros_like(x)
    v_new = wp.zeros_like(v)
    f = wp.zeros(model['n_particles'], dtype=wp.vec2, device=device)
    
    gravity = wp.vec2(0.0, -1.0)
    mass = 1.0
    
    for step in range(n_steps):
        f.zero_()
        wp.launch(eval_spring_2d, dim=model['n_springs'], inputs=[
            x, v, model['spring_indices'], model['spring_rest_length'],
            model['spring_stiffness'], model['spring_damping'], f], device=device)
        
        wp.launch(integrate_implicit_simple, dim=model['n_particles'], inputs=[
            x, v, f, mass, stiffness, dt, gravity, x_new, v_new], device=device)
        
        x, x_new = x_new, x
        v, v_new = v_new, v
        
        # Check for explosion
        pos = x.numpy()
        if np.any(np.isnan(pos)) or np.any(np.abs(pos) > 100):
            return False, step
    
    return True, n_steps


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SEARCHING FOR PARAMETERS: Explicit EXPLODES, Implicit STABLE")
    print("="*70)
    
    # Test different stiffness and timestep combinations
    stiffness_values = [100, 500, 1000, 2000, 5000]
    dt_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    
    print(f"\n{'Stiffness':<12} {'dt (ms)':<10} {'Stability Limit':<18} {'Explicit':<12} {'Implicit':<12}")
    print("-"*70)
    
    good_params = []
    
    for k in stiffness_values:
        stability_limit = 2 * np.sqrt(1.0/k)  # theoretical limit
        
        for dt in dt_values:
            exp_stable, exp_step = test_explicit(k, dt, n_steps=200)
            imp_stable, imp_step = test_implicit(k, dt, n_steps=200)
            
            exp_status = "✓ STABLE" if exp_stable else f"💥 step {exp_step}"
            imp_status = "✓ STABLE" if imp_stable else f"💥 step {imp_step}"
            
            # We want: explicit explodes AND implicit stable
            marker = ""
            if not exp_stable and imp_stable:
                marker = " ← GOOD!"
                good_params.append((k, dt, stability_limit))
            
            print(f"{k:<12} {dt*1000:<10.1f} {stability_limit*1000:.2f} ms{'':<10} {exp_status:<12} {imp_status:<12}{marker}")
    
    print("\n" + "="*70)
    if good_params:
        print("FOUND GOOD PARAMETERS (Explicit explodes, Implicit stable):")
        print("="*70)
        for k, dt, limit in good_params:
            print(f"  k={k}, dt={dt*1000:.1f}ms (stability limit: {limit*1000:.2f}ms)")
        
        # Pick the best one (largest dt where this works)
        best = max(good_params, key=lambda x: x[1])
        print(f"\nRECOMMENDED: k={best[0]}, dt={best[1]*1000:.1f}ms")
    else:
        print("NO GOOD PARAMETERS FOUND - need to adjust search range")
        print("Try lower stiffness or different dt values")
    print("="*70)
