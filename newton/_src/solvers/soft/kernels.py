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

"""Warp kernels for soft body simulation."""

import warp as wp

from newton import ParticleFlags
from newton._src.geometry.kernels import triangle_closest_point_barycentric

# Use ParticleFlags.ACTIVE value directly
PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


@wp.kernel
def eval_barycentric_constraints(
    positions: wp.array(dtype=wp.vec3), 
    velocities: wp.array(dtype=wp.vec3), 
    constraint_positions: wp.array(dtype=wp.vec3), 
    constraint_is_active: wp.array(dtype=wp.bool),
    constraint_types: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_weights: wp.array2d(dtype=wp.float32),
    constraint_vertex_count: wp.array(dtype=wp.int32),
    stiffness: wp.float32, 
    damping: wp.float32,
    f: wp.array(dtype=wp.vec3)
):
    """
    Evaluate barycentric constraints for vertex, edge, face, and tetra types.
    
    Constraint types:
    0 = vertex constraint (single point)
    1 = edge constraint (2 vertices with barycentric weights)
    2 = face constraint (3 vertices with barycentric weights)  
    3 = tetra constraint (4 vertices with barycentric weights)
    """
    tid = wp.tid()
    
    if not constraint_is_active[tid]:
        return
    
    constraint_type = constraint_types[tid]
    vertex_count = constraint_vertex_count[tid]
    
    # Compute current interpolated position based on constraint type
    current_pos = wp.vec3(0.0, 0.0, 0.0)
    current_vel = wp.vec3(0.0, 0.0, 0.0)
    
    if constraint_type == 0:
        # Vertex constraint - direct position
        current_pos = positions[tid]
        current_vel = velocities[tid]
    else:
        # Barycentric interpolation for edge, face, or tetra
        for i in range(vertex_count):
            vertex_idx = constraint_indices[tid, i]
            weight = constraint_weights[tid, i]
            
            if vertex_idx >= 0:
                current_pos += positions[vertex_idx] * weight
                current_vel += velocities[vertex_idx] * weight
    
    target_pos = constraint_positions[tid]
    
    # Spring force: F = k * (x_target - x)
    spring_force = (target_pos - current_pos) * stiffness
    
    # Damping force: F = -c * v
    damping_force = current_vel * damping
    
    # Total constraint force
    constraint_force = spring_force - damping_force
    
    # Distribute force to participating vertices based on barycentric weights
    if constraint_type == 0:
        # Vertex constraint - apply force directly
        wp.atomic_add(f, tid, constraint_force)
    else:
        # Distribute force to participating vertices
        for i in range(vertex_count):
            vertex_idx = constraint_indices[tid, i]
            weight = constraint_weights[tid, i]
            
            if vertex_idx >= 0:
                # Distribute force proportionally to barycentric weights
                distributed_force = constraint_force * weight
                wp.atomic_add(f, vertex_idx, distributed_force)


@wp.kernel
def clear_barycentric_constraints(
    constraint_is_active: wp.array(dtype=wp.bool),
    constraint_types: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_weights: wp.array2d(dtype=wp.float32),
    constraint_vertex_count: wp.array(dtype=wp.int32)
):
    """Clear all barycentric constraints."""
    tid = wp.tid()
    
    constraint_is_active[tid] = False
    constraint_types[tid] = 0
    
    # Clear indices and weights
    for i in range(4):
        constraint_indices[tid, i] = -1
        constraint_weights[tid, i] = 0.0
    
    constraint_vertex_count[tid] = 0


@wp.kernel
def set_barycentric_constraint(
    constraint_idx: wp.int32,
    constraint_type: wp.int32,
    target_position: wp.vec3,
    vertex_indices: wp.array(dtype=wp.int32),
    barycentric_weights: wp.array(dtype=wp.float32),
    vertex_count: wp.int32,
    constraint_is_active: wp.array(dtype=wp.bool),
    constraint_positions: wp.array(dtype=wp.vec3),
    constraint_types: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_weights: wp.array2d(dtype=wp.float32),
    constraint_vertex_count: wp.array(dtype=wp.int32)
):
    """Set a barycentric constraint with the given parameters."""
    # Set constraint data
    constraint_is_active[constraint_idx] = True
    constraint_positions[constraint_idx] = target_position
    constraint_types[constraint_idx] = constraint_type
    constraint_vertex_count[constraint_idx] = vertex_count
    
    # Set vertex indices and weights
    for i in range(vertex_count):
        constraint_indices[constraint_idx, i] = vertex_indices[i]
        constraint_weights[constraint_idx, i] = barycentric_weights[i]
    
    # Clear unused indices and weights
    for i in range(vertex_count, 4):
        constraint_indices[constraint_idx, i] = -1
        constraint_weights[constraint_idx, i] = 0.0


@wp.kernel
def init_barycentric_constraints_batch(
    constraint_indices: wp.array(dtype=wp.int32),
    constraint_types: wp.array(dtype=wp.int32),
    target_positions: wp.array(dtype=wp.vec3),
    vertex_indices_array: wp.array2d(dtype=wp.int32),
    barycentric_weights_array: wp.array2d(dtype=wp.float32),
    vertex_counts: wp.array(dtype=wp.int32),
    constraint_is_active: wp.array(dtype=wp.bool),
    constraint_positions: wp.array(dtype=wp.vec3),
    constraint_types_out: wp.array(dtype=wp.int32),
    constraint_indices_out: wp.array2d(dtype=wp.int32),
    constraint_weights_out: wp.array2d(dtype=wp.float32),
    constraint_vertex_count_out: wp.array(dtype=wp.int32)
):
    """Efficiently set multiple barycentric constraints in a single kernel launch."""
    tid = wp.tid()
    
    # Get constraint data for this thread
    constraint_idx = constraint_indices[tid]
    constraint_type = constraint_types[tid]
    target_position = target_positions[tid]
    vertex_count = vertex_counts[tid]
    
    # Set constraint data
    constraint_is_active[constraint_idx] = True
    constraint_positions[constraint_idx] = target_position
    constraint_types_out[constraint_idx] = constraint_type
    constraint_vertex_count_out[constraint_idx] = vertex_count
    
    # Set vertex indices and weights
    for i in range(vertex_count):
        constraint_indices_out[constraint_idx, i] = vertex_indices_array[tid, i]
        constraint_weights_out[constraint_idx, i] = barycentric_weights_array[tid, i]
    
    # Clear unused indices and weights
    for i in range(vertex_count, 4):
        constraint_indices_out[constraint_idx, i] = -1
        constraint_weights_out[constraint_idx, i] = 0.0


@wp.kernel
def build_system_matrix_sparse_kernel(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
    indices: wp.array(dtype=int),
    spring_stiffness: wp.array(dtype=wp.float32),
    spring_damping: wp.array(dtype=wp.float32),
    dt: wp.float32,
    mass: wp.float32,
    Minv: wp.float32
):
    """Build system matrix for implicit integration."""
    tid = wp.tid()
    i = indices[tid * 2 + 0]
    j = indices[tid * 2 + 1]
    
    # Get spring stiffness
    k = spring_stiffness[tid]
    
    # Pre-compute common terms
    dt2 = dt * dt
    dt2_k = dt2 * k
    dt_damping_Minv = dt * spring_damping[tid] * Minv
    
    # Create 3x3 blocks for system matrix A = M - h*D - h²*K
    block_ii = wp.mat33f(
        mass + dt_damping_Minv - dt2_k, 0.0, 0.0,
        0.0, mass + dt_damping_Minv - dt2_k, 0.0,
        0.0, 0.0, mass + dt_damping_Minv - dt2_k
    )
    
    block_jj = wp.mat33f(
        mass + dt_damping_Minv - dt2_k, 0.0, 0.0,
        0.0, mass + dt_damping_Minv - dt2_k, 0.0,
        0.0, 0.0, mass + dt_damping_Minv - dt2_k
    )
    
    # Off-diagonal blocks: positive stiffness coupling
    block_ij = wp.mat33f(
        dt2_k, 0.0, 0.0,
        0.0, dt2_k, 0.0,
        0.0, 0.0, dt2_k
    )
    
    # Store blocks
    block_idx = tid * 4
    
    # (i,i) block
    rows[block_idx] = i
    cols[block_idx] = i
    values[block_idx] = block_ii
    
    # (j,j) block
    rows[block_idx + 1] = j
    cols[block_idx + 1] = j
    values[block_idx + 1] = block_jj
    
    # (i,j) block
    rows[block_idx + 2] = i
    cols[block_idx + 2] = j
    values[block_idx + 2] = block_ij
    
    # (j,i) block
    rows[block_idx + 3] = j
    cols[block_idx + 3] = i
    values[block_idx + 3] = block_ij


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    if i == -1 or j == -1:
        return

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    
    # Protect against division by zero
    if l < 1.0e-6:
        return
    
    l_inv = 1.0 / l

    # Normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = wp.dot(dir, vij)

    # Damping based on relative velocity
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def eval_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate tetrahedral FEM forces using Stable Neo-Hookean material model."""
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.matrix_from_cols(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # Rest stability correction factor (Smith et al. 2018)
    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # Scale stiffness coefficients to account for volume
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = wp.matrix_from_cols(v10, v20, v30) * Dm

    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # Stable Neo-Hookean (Smith et al. 2018)
    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)

    # Deviatoric part with stable formulation
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * wp.transpose(Dm)

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    # Hydrostatic (volumetric) part
    J = wp.determinant(F)

    s = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s
    dJdx2 = wp.cross(x30, x10) * s
    dJdx3 = wp.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = -(f1 + f2 + f3)

    # Apply forces
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


@wp.kernel
def eval_triangles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat22),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate triangle membrane forces."""
    tid = wp.tid()

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]
    k_drag = materials[tid, 3]
    k_lift = materials[tid, 4]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]

    x10 = x1 - x0
    x20 = x2 - x0

    v10 = v1 - v0
    v20 = v2 - v0

    Dm = pose[tid]

    inv_rest_area = wp.determinant(Dm) * 2.0
    rest_area = 1.0 / inv_rest_area

    # Scale stiffness coefficients
    k_mu = k_mu * rest_area
    k_lambda = k_lambda * rest_area
    k_damp = k_damp * rest_area

    # F = Xs*Xm^-1
    F1 = x10 * Dm[0, 0] + x20 * Dm[1, 0]
    F2 = x10 * Dm[0, 1] + x20 * Dm[1, 1]

    dFdt1 = v10 * Dm[0, 0] + v20 * Dm[1, 0]
    dFdt2 = v10 * Dm[0, 1] + v20 * Dm[1, 1]

    Ic = wp.dot(F1, F1) + wp.dot(F2, F2)

    deviatoric_scale = (Ic - 2.0) / Ic
    P1 = F1 * k_mu * deviatoric_scale + dFdt1 * k_damp
    P2 = F2 * k_mu * deviatoric_scale + dFdt2 * k_damp

    f1 = P1 * Dm[0, 0] + P2 * Dm[0, 1]
    f2 = P1 * Dm[1, 0] + P2 * Dm[1, 1]

    # Area preservation
    n = wp.cross(x10, x20)
    area = wp.length(n) * 0.5

    act = activation[tid]

    c = area * inv_rest_area - 1.0 + act

    n = wp.normalize(n)
    dcdq = wp.cross(x20, n) * inv_rest_area * 0.5
    dcdr = wp.cross(n, x10) * inv_rest_area * 0.5

    f_area = k_lambda * c

    dcdt = wp.dot(dcdq, v1) + wp.dot(dcdr, v2) - wp.dot(dcdq + dcdr, v0)
    f_damp = k_damp * dcdt

    f1 = f1 + dcdq * (f_area + f_damp)
    f2 = f2 + dcdr * (f_area + f_damp)
    f0 = f1 + f2

    # Lift + Drag
    vmid = (v0 + v1 + v2) * 0.3333
    vdir = wp.normalize(vmid)

    f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))
    f_lift = n * (k_lift * area * (wp.HALF_PI - wp.acos(wp.dot(n, vdir)))) * wp.dot(vmid, vmid)

    f0 = f0 - f_drag - f_lift
    f1 = f1 + f_drag + f_lift
    f2 = f2 + f_drag + f_lift

    # Apply forces
    wp.atomic_add(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)


@wp.kernel
def eval_bending(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    rest: wp.array(dtype=float),
    bending_properties: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate bending forces for edges."""
    tid = wp.tid()
    eps = 1.0e-6

    ke = bending_properties[tid, 0]
    kd = bending_properties[tid, 1]

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    if i == -1 or j == -1 or k == -1 or l == -1:
        return

    rest_angle = rest[tid]

    x1 = x[i]
    x2 = x[j]
    x3 = x[k]
    x4 = x[l]

    v1 = v[i]
    v2 = v[j]
    v3 = v[k]
    v4 = v[l]

    n1 = wp.cross(x3 - x1, x4 - x1)
    n2 = wp.cross(x4 - x2, x3 - x2)
    e = x4 - x3

    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)

    if n1_length < eps or n2_length < eps or e_length < eps:
        return

    n1 = n1 / n1_length
    n2 = n2 / n2_length
    e_hat = e / e_length

    cos_theta = wp.dot(n1, n2)
    sin_theta = wp.dot(wp.cross(n1, n2), e_hat)
    theta = wp.atan2(sin_theta, cos_theta)

    d1 = n1 * e_length
    d2 = n2 * e_length
    d3 = n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)
    d4 = n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)

    f_elastic = ke * (theta - rest_angle)
    f_damp = kd * (wp.dot(d1, v1) + wp.dot(d2, v2) + wp.dot(d3, v3) + wp.dot(d4, v4))

    f_total = -e_length * (f_elastic + f_damp)

    wp.atomic_add(f, i, d1 * f_total)
    wp.atomic_add(f, j, d2 * f_total)
    wp.atomic_add(f, k, d3 * f_total)
    wp.atomic_add(f, l, d4 * f_total)


@wp.kernel
def update_state(
    dv: wp.array(dtype=wp.vec3), 
    dt: wp.float32, 
    positions_in: wp.array(dtype=wp.vec3), 
    velocities_in: wp.array(dtype=wp.vec3), 
    positions_out: wp.array(dtype=wp.vec3), 
    velocities_out: wp.array(dtype=wp.vec3)
):
    """Update particle positions and velocities."""
    tid = wp.tid()
    vel = velocities_in[tid] + dv[tid]
    positions_out[tid] = positions_in[tid] + vel * dt
    velocities_out[tid] = vel


@wp.kernel
def eval_triangles_contact(
    num_particles: int,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    materials: wp.array2d(dtype=float),
    particle_radius: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate triangle-particle contact forces."""
    tid = wp.tid()
    face_no = tid // num_particles
    particle_no = tid % num_particles

    pos = x[particle_no]

    i = indices[face_no, 0]
    j = indices[face_no, 1]
    k = indices[face_no, 2]

    if i == particle_no or j == particle_no or k == particle_no:
        return

    p = x[i]
    q = x[j]
    r = x[k]

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = wp.dot(diff, diff)
    n = wp.normalize(diff)
    c = wp.min(dist - particle_radius[particle_no], 0.0)
    fn = n * c * 1e5

    wp.atomic_sub(f, particle_no, fn)

    wp.atomic_add(f, i, fn * bary[0])
    wp.atomic_add(f, j, fn * bary[1])
    wp.atomic_add(f, k, fn * bary[2])


@wp.kernel
def eval_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    ground: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate particle-ground contact forces."""
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[tid]
    v = particle_v[tid]
    radius = particle_radius[tid]

    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - radius, 0.0)

    vn = wp.dot(n, v)
    jn = c * ke

    if c >= 0.0:
        return

    jd = min(vn, 0.0) * kd

    fn = jn + jd

    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    ft = wp.min(vs * kf, mu * wp.abs(fn))

    f[tid] = f[tid] - n * fn - vt * ft


@wp.kernel
def eval_gravity(
    gravity_force: wp.vec3,
    particle_flags: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
):
    """Apply gravity force to active particles."""
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) != 0:
        forces[tid] = gravity_force


@wp.kernel
def update_barycentric_constraints_batch(
    constraint_indices: wp.array(dtype=wp.int32),
    target_positions: wp.array(dtype=wp.vec3),
    constraint_is_active: wp.array(dtype=wp.bool),
    constraint_positions: wp.array(dtype=wp.vec3)
):
    """Update target positions for multiple barycentric constraints."""
    tid = wp.tid()
    
    constraint_idx = constraint_indices[tid]
    target_position = target_positions[tid]
    
    constraint_is_active[constraint_idx] = True
    constraint_positions[constraint_idx] = target_position


@wp.kernel
def eval_soft_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    particle_radius: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate soft contact forces from collision detection results."""
    tid = wp.tid()
    
    count = soft_contact_count[0]
    if tid >= count:
        return
    
    particle_idx = soft_contact_particle[tid]
    if particle_idx < 0:
        return
    
    # Get contact geometry
    body_pos = soft_contact_body_pos[tid]
    body_vel = soft_contact_body_vel[tid]
    n = soft_contact_normal[tid]
    
    # Get particle state
    x = particle_x[particle_idx]
    v = particle_v[particle_idx]
    radius = particle_radius[particle_idx]
    
    # Compute penetration (distance from particle to body surface)
    # The contact normal points from body to particle
    d = wp.dot(x - body_pos, n)
    penetration = radius - d
    
    if penetration > 0.0:
        # Relative velocity
        rel_v = v - body_vel
        vn = wp.dot(rel_v, n)
        vt = rel_v - vn * n

        # Normal force (spring + damping)
        fn = ke * penetration - kd * wp.min(vn, 0.0)

        # Tangential force (friction): use max(0, fn) so damping never yields negative friction
        vt_mag = wp.length(vt)
        if vt_mag > 1.0e-6:
            fn_safe = wp.max(0.0, fn)
            ft_max = mu * fn_safe
            ft = wp.min(kf * vt_mag, ft_max)
            force = fn * n - ft * (vt / vt_mag)
        else:
            force = fn * n

        wp.atomic_add(f, particle_idx, force)


@wp.kernel
def solve_soft_contacts_constraint(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_invmass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_m_inv: wp.array(dtype=float),
    body_I_inv: wp.array(dtype=wp.mat33),
    shape_body: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    particle_friction: wp.array(dtype=float),
    particle_ka: float,
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    contact_max: int,
    dt: float,
    relaxation: float,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    body_delta: wp.array(dtype=wp.spatial_vector),
):
    """Constraint-based soft-rigid contact solver (prevents penetration!).
    
    Similar to XPBD's solve_particle_shape_contacts, but designed for SolverSoft.
    This applies position corrections to prevent penetration, making it preventive
    rather than reactive like eval_soft_contacts.
    
    Key differences from eval_soft_contacts:
    - Preventive: applies corrections when c < particle_ka (before deep penetration)
    - Constraint-based: applies position deltas, not forces
    - Handles body transforms and velocities properly
    """
    tid = wp.tid()
    
    count = min(contact_max, soft_contact_count[0])
    if tid >= count:
        return
    
    shape_index = soft_contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = soft_contact_particle[tid]
    
    if (particle_flags[particle_index] & 1) == 0:  # PARTICLE_FLAG_ACTIVE = 1
        return
    
    px = particle_x[particle_index]
    pv = particle_v[particle_index]
    
    # Get body transform (identity if static/ground)
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
    
    # Body position in world space
    bx = wp.transform_point(X_wb, soft_contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)
    
    n = soft_contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]
    
    # Preventive: apply constraint when c < particle_ka (before deep penetration)
    if c > particle_ka:
        return
    
    # Per-particle friction: use particle value so per-vertex friction is effective (not diluted by shape mu)
    mu = particle_friction[particle_index]
    
    # Body velocity
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]
    
    body_w = wp.spatial_bottom(body_v_s)
    body_v = wp.spatial_top(body_v_s)
    
    # Compute body velocity at particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, soft_contact_body_vel[tid])
    
    # Relative velocity
    v = pv - bv
    
    # Normal constraint (cap correction to avoid explosion; 3*radius allows stronger friction cap)
    max_n = 3.0 * particle_radius[particle_index]
    lambda_n = wp.max(c, -max_n)
    delta_n = n * lambda_n
    
    # Friction constraint
    vn = wp.dot(n, v)
    vt = v - n * vn
    
    # Compute inverse masses
    w1 = particle_invmass[particle_index]
    w2 = 0.0
    if body_index >= 0:
        angular = wp.cross(r, n)
        q = wp.transform_get_rotation(X_wb)
        rot_angular = wp.quat_rotate_inv(q, angular)
        I_inv = body_I_inv[body_index]
        w2 = body_m_inv[body_index] + wp.dot(rot_angular, I_inv * rot_angular)
    denom = w1 + w2
    if denom == 0.0:
        return
    
    # Friction constraint
    lambda_f = wp.max(mu * lambda_n, -wp.length(vt) * dt)
    if wp.length(vt) > 1e-6:
        delta_f = wp.normalize(vt) * lambda_f
    else:
        delta_f = wp.vec3(0.0)
    
    # Total constraint correction (friction opposes motion, normal pushes away)
    delta_total = (delta_f - delta_n) / denom * relaxation
    
    # Apply correction weighted by inverse mass
    wp.atomic_add(delta, particle_index, w1 * delta_total)
    
    # Apply reaction to body (if not static)
    if body_index >= 0:
        delta_t = wp.cross(r, delta_total)
        wp.atomic_sub(body_delta, body_index, wp.spatial_vector(delta_total, delta_t))


@wp.kernel
def apply_particle_corrections(
    x_current: wp.array(dtype=wp.vec3),
    v_current: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    v_max: float,
    max_correction: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    """Apply constraint corrections to particle positions and update velocities.
    
    Uses additive velocity update (v_new = v_current + delta/dt) and clamps both
    position correction and resulting velocity for stability.
    """
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    
    xp = x_current[tid]
    vp = v_current[tid]
    d = delta[tid]
    
    # Clamp position correction magnitude to avoid explosion
    d_mag = wp.length(d)
    if d_mag > max_correction and d_mag > 1.0e-9:
        d = d * (max_correction / d_mag)
    
    # Apply position correction
    x_new = xp + d
    # Additive velocity update (correction impulse), then clamp
    v_new = vp + d / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new = v_new * (v_max / v_new_mag)
    
    x_out[tid] = x_new
    v_out[tid] = v_new
