# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Self-contact force kernels: vertex–triangle and edge–edge repulsion from
# TriMeshCollisionInfo (BVH-based). C² force law: tau = 0.5*radius.
# Friction forces use the same IPC-style model as the VBD solver.

import warp as wp

from ...geometry.kernels import triangle_closest_point

CONTACT_MIN_DISTANCE_FRACTION = 0.05
CONTACT_FORCE_LINEAR_DISTANCE_EPS = 1.0e-5
EDGE_EDGE_PARALLEL_EPSILON = 1.0e-5
NUM_EDGE_ENDPOINTS = 4
EDGE_EDGE_VERTEX_SHARE = 1.0 / float(NUM_EDGE_ENDPOINTS)


class _mat32(wp.types.matrix(shape=(3, 2), dtype=wp.float32)):
    pass


@wp.func
def _build_orthonormal_basis(n: wp.vec3):
    """Two orthonormal axes in the plane perpendicular to n."""
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]
        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]
        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]
    return b1, b2


@wp.func
def _compute_friction_force(
    mu: float,
    normal_contact_force: float,
    T: _mat32,
    u: wp.vec2,
    eps_u: float,
) -> wp.vec3:
    """IPC-style friction force (same model as VBD). Returns 3D friction force."""
    u_norm = wp.length(u)
    if u_norm > 0.0:
        if u_norm > eps_u:
            f1_SF_over_x = 1.0 / u_norm
        else:
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u
        return -mu * normal_contact_force * T * (f1_SF_over_x * u)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _self_contact_force_magnitude(dis: float, contact_radius: float, stiffness: float) -> float:
    """C² repulsion: k2/dis for tau > d > eps, else linear in penetration."""
    if dis >= contact_radius:
        return 0.0
    penetration = contact_radius - dis
    tau = contact_radius * 0.5
    if tau > dis and dis > CONTACT_FORCE_LINEAR_DISTANCE_EPS:
        k2 = 0.5 * tau * tau * stiffness
        return k2 / dis
    return stiffness * penetration


@wp.kernel
def eval_self_contact_vertex_triangle_from_collision_info(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    surface_tri_indices: wp.array(dtype=wp.int32),
    num_surface_tris: int,
    contact_radius: float,
    contact_stiffness: float,
    contact_force_cap: float,
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
):
    """Vertex–triangle repulsion and friction for pairs in collision_info (sparse)."""
    vid = wp.tid()
    if vid >= particle_q.shape[0]:
        return
    n_coll = wp.min(
        vertex_colliding_triangles_count[vid],
        vertex_colliding_triangles_buffer_sizes[vid],
    )
    if n_coll <= 0:
        return
    p = particle_q[vid]
    offset = vertex_colliding_triangles_offsets[vid]
    for i in range(n_coll):
        tri_idx = vertex_colliding_triangles[2 * (offset + i) + 1]
        if tri_idx < 0 or tri_idx >= num_surface_tris:
            continue
        ia = surface_tri_indices[tri_idx * 3 + 0]
        ib = surface_tri_indices[tri_idx * 3 + 1]
        ic = surface_tri_indices[tri_idx * 3 + 2]
        if vid == ia or vid == ib or vid == ic:
            continue
        a = particle_q[ia]
        b = particle_q[ib]
        c = particle_q[ic]
        closest_p, bary, _ = triangle_closest_point(a, b, c, p)
        diff = p - closest_p
        d = wp.length(diff)
        d_min = contact_radius * CONTACT_MIN_DISTANCE_FRACTION
        if d >= contact_radius or d <= d_min or d != d:
            continue
        n = diff / d
        mag = _self_contact_force_magnitude(d, contact_radius, contact_stiffness)
        if contact_force_cap > 0.0:
            mag = wp.min(mag, contact_force_cap)
        f = n * mag
        # Friction (VBD-style): relative tangential displacement u from velocities
        v_vertex = particle_qd[vid]
        v_tri = bary[0] * particle_qd[ia] + bary[1] * particle_qd[ib] + bary[2] * particle_qd[ic]
        dx = (v_vertex - v_tri) * dt
        u_3d = dx - n * wp.dot(n, dx)
        e0, e1 = _build_orthonormal_basis(n)
        T = _mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])
        u = wp.vec2(wp.dot(e0, u_3d), wp.dot(e1, u_3d))
        friction = _compute_friction_force(
            friction_mu, mag, T, u, friction_epsilon * dt
        )
        f_total = f + friction
        wp.atomic_add(particle_f, vid, f_total)
        wp.atomic_sub(particle_f, ia, f_total * bary[0])
        wp.atomic_sub(particle_f, ib, f_total * bary[1])
        wp.atomic_sub(particle_f, ic, f_total * bary[2])


@wp.kernel
def eval_self_contact_edge_edge_from_collision_info(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    edge_colliding_edges: wp.array(dtype=wp.int32),
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32),
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32),
    edge_colliding_edges_count: wp.array(dtype=wp.int32),
    surface_edge_indices: wp.array(dtype=wp.int32),
    num_surface_edges: int,
    contact_radius: float,
    contact_stiffness: float,
    contact_force_cap: float,
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
):
    """Edge–edge repulsion and friction for pairs in collision_info (sparse)."""
    e0_idx = wp.tid()
    if e0_idx >= num_surface_edges:
        return
    n_coll = wp.min(
        edge_colliding_edges_count[e0_idx],
        edge_colliding_edges_buffer_sizes[e0_idx],
    )
    if n_coll <= 0:
        return
    i0 = surface_edge_indices[e0_idx * 2 + 0]
    j0 = surface_edge_indices[e0_idx * 2 + 1]
    p0 = particle_q[i0]
    p1 = particle_q[j0]
    offset = edge_colliding_edges_offsets[e0_idx]
    for i in range(n_coll):
        e1 = edge_colliding_edges[2 * (offset + i) + 1]
        if e1 < 0 or e1 >= num_surface_edges:
            continue
        i1 = surface_edge_indices[e1 * 2 + 0]
        j1 = surface_edge_indices[e1 * 2 + 1]
        if i0 == i1 or i0 == j1 or j0 == i1 or j0 == j1:
            continue
        q0 = particle_q[i1]
        q1 = particle_q[j1]
        st = wp.closest_point_edge_edge(p0, p1, q0, q1, EDGE_EDGE_PARALLEL_EPSILON)
        s = st[0]
        t = st[1]
        d = st[2]
        d_min = contact_radius * CONTACT_MIN_DISTANCE_FRACTION
        if d >= contact_radius or d <= d_min or d != d:
            continue
        c1 = p0 + (p1 - p0) * s
        c2 = q0 + (q1 - q0) * t
        n = (c1 - c2) / d
        mag = _self_contact_force_magnitude(d, contact_radius, contact_stiffness)
        if contact_force_cap > 0.0:
            mag = wp.min(mag, contact_force_cap)
        f = n * mag
        # Friction (VBD-style): relative tangential displacement at closest points
        v_c1 = (1.0 - s) * particle_qd[i0] + s * particle_qd[j0]
        v_c2 = (1.0 - t) * particle_qd[i1] + t * particle_qd[j1]
        dx = (v_c1 - v_c2) * dt
        u_3d = dx - n * wp.dot(n, dx)
        e0_axis, e1_axis = _build_orthonormal_basis(n)
        T = _mat32(e0_axis[0], e1_axis[0], e0_axis[1], e1_axis[1], e0_axis[2], e1_axis[2])
        u = wp.vec2(wp.dot(e0_axis, u_3d), wp.dot(e1_axis, u_3d))
        friction = _compute_friction_force(
            friction_mu, mag, T, u, friction_epsilon * dt
        )
        f_total = f + friction
        wp.atomic_add(particle_f, i0, f_total * EDGE_EDGE_VERTEX_SHARE)
        wp.atomic_add(particle_f, j0, f_total * EDGE_EDGE_VERTEX_SHARE)
        wp.atomic_sub(particle_f, i1, f_total * EDGE_EDGE_VERTEX_SHARE)
        wp.atomic_sub(particle_f, j1, f_total * EDGE_EDGE_VERTEX_SHARE)


@wp.kernel
def build_system_matrix_self_contact_diagonal_kernel(
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33f),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32),
    contact_stiffness: wp.float32,
    dt: wp.float32,
    block_offset: wp.int32,
):
    """Lumped self-contact tangent: add dt² * k_eff * I to diagonal for each vertex in contact.
    k_eff = (VT count + EE contribution) * contact_stiffness. Vertex-triangle only here (EE would need per-vertex sum).
    """
    vid = wp.tid()
    n_vt = wp.min(
        vertex_colliding_triangles_count[vid],
        vertex_colliding_triangles_buffer_sizes[vid],
    )
    k_sum = float(n_vt) * contact_stiffness
    dt2_k = dt * dt * k_sum
    blk = wp.mat33f(dt2_k, 0.0, 0.0, 0.0, dt2_k, 0.0, 0.0, 0.0, dt2_k)
    idx = block_offset + vid
    rows[idx] = vid
    cols[idx] = vid
    values[idx] = blk
