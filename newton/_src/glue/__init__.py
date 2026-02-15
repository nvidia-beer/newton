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

"""Proximity-based glue: soft particle to rigid body anchor springs."""

import numpy as np
import warp as wp


def build_proximity_glue_pairs(
    particle_q_np: np.ndarray,
    rigid_vertices: np.ndarray,
    rigid_p: np.ndarray,
    rigid_q: wp.quat,
    start_particle: int,
    end_particle: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Find (particle_idx, rigid_anchor_local) pairs where particle is within epsilon of a rigid vertex.

    Returns:
        particle_indices: int32 array of particle indices.
        anchor_locals: (N, 3) float32 array of anchor positions in rigid body local frame.
        min_dist: minimum distance found in the checked range, or None.
    """
    rigid_verts_world = rigid_p + np.array(
        [wp.quat_rotate(rigid_q, wp.vec3(float(v[0]), float(v[1]), float(v[2]))) for v in rigid_vertices]
    )
    particle_indices = []
    anchor_locals = []
    min_d2_overall = float("inf")
    for pidx in range(start_particle, end_particle):
        p_world = np.asarray(particle_q_np[pidx], dtype=np.float64).reshape(3)
        best_d2 = float("inf")
        best_anchor_local = None
        for vi, v_local in enumerate(rigid_vertices):
            v_world = rigid_verts_world[vi]
            d2 = float(np.sum((p_world - v_world) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_anchor_local = v_local
        min_d2_overall = min(min_d2_overall, best_d2)
        if best_d2 < epsilon * epsilon and best_anchor_local is not None:
            particle_indices.append(pidx)
            anchor_locals.append(best_anchor_local)
    min_dist = np.sqrt(min_d2_overall) if min_d2_overall < float("inf") else None
    return (
        np.array(particle_indices, dtype=np.int32),
        np.array(anchor_locals, dtype=np.float32).reshape(-1, 3) if anchor_locals else np.empty((0, 3), dtype=np.float32),
        min_dist,
    )


@wp.kernel
def apply_glue_proximity_impulse_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    body_com: wp.array(dtype=wp.vec3),
    glue_particles: wp.array(dtype=int),
    glue_body: int,
    body_anchors_local: wp.array(dtype=wp.vec3),
    rest_lengths: wp.array(dtype=float),
    glue_ke: float,
    glue_kd: float,
    dt: float,
    max_vel_per_substep: float,
    inv_m_p_eff: float,
):
    """Apply spring impulse for each proximity-based (particle, rigid anchor) pair."""
    tid = wp.tid()
    glue_particle = glue_particles[tid]
    body_anchor_local = body_anchors_local[tid]
    rest_length = rest_lengths[tid]

    p = particle_q[glue_particle]
    vp = particle_qd[glue_particle]

    X_b = body_q[glue_body]
    q_b = wp.transform_get_rotation(X_b)
    p_b = wp.transform_get_translation(X_b)
    anchor_world = p_b + wp.quat_rotate(q_b, body_anchor_local)

    body_v_s = body_qd[glue_body]
    vel_linear = wp.spatial_top(body_v_s)
    vel_angular = wp.spatial_bottom(body_v_s)
    r = anchor_world - (p_b + wp.quat_rotate(q_b, body_com[glue_body]))
    vel_anchor = vel_linear + wp.cross(vel_angular, r)

    d = p - anchor_world
    length = wp.length(d)
    if length < 1.0e-6:
        return
    direction = d / length
    elongation = length - rest_length
    rel_vel = vp - vel_anchor
    vn = wp.dot(direction, rel_vel)
    F_mag = glue_ke * elongation + glue_kd * vn
    F_on_particle = -direction * F_mag
    F_on_body = direction * F_mag

    inv_m_p = wp.min(particle_inv_mass[glue_particle], inv_m_p_eff)
    inv_m_b = body_inv_m[glue_body]
    I_inv = body_inv_I[glue_body]

    dv_p = F_on_particle * inv_m_p * dt
    dv_b = F_on_body * inv_m_b * dt
    torque_world = wp.cross(r, F_on_body)
    torque_body = wp.quat_rotate_inv(q_b, torque_world)
    d_omega_body = I_inv * torque_body * dt
    d_omega_world = wp.quat_rotate(q_b, d_omega_body)
    dv_p_mag = wp.length(dv_p)
    if dv_p_mag > max_vel_per_substep:
        scale = max_vel_per_substep / dv_p_mag
        dv_p = dv_p * scale

    wp.atomic_add(particle_qd, glue_particle, dv_p)
    wp.atomic_add(body_qd, glue_body, wp.spatial_vector(dv_b, d_omega_world))
