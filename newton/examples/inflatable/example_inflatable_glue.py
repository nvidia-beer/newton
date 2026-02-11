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
Inflatable Glue Example - Soft + Rigid connected by proximity-based springs.

Rigid box sits on ground (z bottom at 0), inflatable sits on top of rigid's z+ face.
Glue pairs are found at init: each inflatable particle within epsilon of a rigid
vertex gets a spring to that closest rigid vertex.

Usage:
    python -m newton.examples inflatable_glue [--glue_epsilon 0.05]
    ./run-examples.sh inflatable_glue

Keys: [I/K/O] inflate/deflate/reset | [G/F] glue stronger/weaker
"""

import argparse
import warp as wp
import numpy as np

import newton
from newton.solvers import SolverInflatable, SolverXPBD, TetraBox
from newton._src.sim import SurfaceBox


def _build_proximity_glue_pairs(
    particle_q_np: np.ndarray,
    rigid_vertices: np.ndarray,
    rigid_p: np.ndarray,
    rigid_q: wp.quat,
    start_particle: int,
    end_particle: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Find (particle_idx, rigid_anchor_local) pairs where inflatable particle is within epsilon of some rigid vertex.
    Returns (glue_particle_indices, body_anchors_local, min_dist_for_debug).
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


class Example:
    """
    Rigid box + inflatable box connected by proximity-based glue springs.
    At init, each inflatable particle within epsilon of a rigid vertex gets a spring.
    Press I/K/O to inflate/deflate, G/F to adjust glue strength.
    """

    def __init__(
        self,
        viewer,
        size=(0.4, 0.4, 0.4),
        subdivisions=(5, 5, 5),
        mass: float = 1.0,
        rigid_mass: float = 0.05,
        rigid_pos=None,
        inflatable_pos=None,
        rigid_z_base: float = 0.0,
        stack_gap: float = 0.0,
        glue_epsilon: float = 0.05,
        glue_ke: float = 5.0e4,
        glue_kd: float = 200.0,
        glue_max_vel: float = 1.5,
        k_mu: float = 1.0e5,
        k_lambda: float = 1.0e5,
        k_damp: float = 1.0,
        spring_ke: float = 5.0e4,
        spring_kd: float = 1.0,
        gravity: float = 9.81,
        max_pressure: float = 5.0,
        substeps: int = 5,
        xpbd_iterations: int = 10,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0

        def to_size(s):
            if s is None:
                return None
            if isinstance(s, (int, float)):
                return (float(s), float(s), float(s))
            return tuple(float(x) for x in s)

        def to_sub(s):
            if s is None:
                return None
            if isinstance(s, int):
                return (s, s, s)
            return tuple(int(x) for x in s)

        self.size = to_size(size) or (0.4, 0.4, 0.4)
        self.subdivisions = to_sub(subdivisions) or (5, 5, 5)
        self.mass = mass
        self.rigid_mass = rigid_mass
        self.glue_epsilon = glue_epsilon
        self.glue_ke = glue_ke
        self.glue_kd = glue_kd
        self.glue_ke_step = 1.5  # multiply/divide by this when pressing G/F
        self.glue_max_vel = glue_max_vel
        self.inflatable_mass = mass  # total soft body mass for effective-particle cap
        self.viewer = viewer
        self.debug_track = False

        w, h, d = self.size
        # Stacked: rigid bottom at rigid_z_base, inflatable on top with optional stack_gap
        if rigid_pos is not None:
            rp = rigid_pos
        else:
            rp = (0.0, 0.0, rigid_z_base + d / 2.0)  # center so bottom z=rigid_z_base
        if inflatable_pos is not None:
            ip = inflatable_pos
        else:
            ip = (0.0, 0.0, rigid_z_base + d + stack_gap + d / 2.0)  # bottom at rigid_top + stack_gap
        self.rigid_pos = wp.vec3(float(rp[0]), float(rp[1]), float(rp[2]))
        self.inflatable_pos = wp.vec3(float(ip[0]), float(ip[1]), float(ip[2]))

        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=5e5, kd=1e3, kf=1e4, mu=0.5)
        )

        # --- Both in SAME world so they share coordinates and display stacked ---
        # Rigid has has_particle_collision=False so inflatable particles don't collide with it
        print("\n📦 [1/3] Creating rigid body (bottom)...", flush=True)
        builder.begin_world()
        surface_box = SurfaceBox(
            size=self.size,
            subdivisions=self.subdivisions,
            verbose=True,
        )
        surface_data = surface_box.get_mesh_data()
        rigid_vertices = np.array(surface_data["vertices"])
        rigid_mesh = newton.Mesh(
            surface_data["vertices"],
            surface_data["indices"],
            compute_inertia=True,
            is_solid=False,
        )
        scale = rigid_mass / rigid_mesh.mass if rigid_mesh.mass > 0 else 1.0
        I_np = np.array(rigid_mesh.I) * scale
        self.rigid_body_id = builder.add_body(
            xform=wp.transform(self.rigid_pos, wp.quat_identity()),
            mass=rigid_mass,
            com=rigid_mesh.com,
            I_m=wp.mat33(I_np),
        )
        joint_id = builder.add_joint_free(self.rigid_body_id)
        builder.add_articulation([joint_id], key="rigid_box")
        rigid_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=5e5, kd=100.0, kf=1e4, mu=0.5)
        rigid_shape_cfg.has_particle_collision = False  # no particle collision so inflatable can sit on top
        builder.add_shape_mesh(
            body=self.rigid_body_id,
            mesh=rigid_mesh,
            cfg=rigid_shape_cfg,
        )
        self._rigid_vertices = rigid_vertices

        # --- Inflatable (same world) on top of rigid ---
        print("\n📦 [2/3] Creating inflatable tetrahedral box (on top)...", flush=True)
        tetra_box = TetraBox(
            size=self.size,
            subdivisions=self.subdivisions,
            verbose=True,
        )
        tetra_data = tetra_box.get_mesh_data()
        tetra_vertices = tetra_data["vertices"]
        tetra_indices = tetra_data["indices"]
        tetrahedra = tetra_data["tetrahedra"]

        start_particle = builder.particle_count
        builder.add_soft_mesh(
            pos=self.inflatable_pos,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=tetra_vertices,
            indices=tetra_indices,
            scale=1.0,
            density=mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )

        added_springs = set()
        for t in range(len(tetrahedra)):
            tet_idx = [tetra_indices[t * 4 + k] for k in range(4)]
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for ei, ej in edges:
                i_local, j_local = tet_idx[ei], tet_idx[ej]
                if i_local > j_local:
                    i_local, j_local = j_local, i_local
                spring_key = (i_local, j_local)
                if spring_key not in added_springs:
                    added_springs.add(spring_key)
                    p0, p1 = tetra_vertices[i_local], tetra_vertices[j_local]
                    rest_length = float(np.linalg.norm(np.array(p1) - np.array(p0)))
                    builder.add_spring(
                        start_particle + i_local,
                        start_particle + j_local,
                        spring_ke,
                        spring_kd,
                        rest_length,
                    )

        self._start_particle = start_particle
        self._end_particle = builder.particle_count  # after add_soft_mesh
        builder.end_world()
        print("\n📦 [3/3] Glue: proximity-based (pairs found after init)", flush=True)

        self.model = builder.finalize()
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        # Strong ground contact to prevent inflatable from penetrating when swinging (bottom can dip below z=0)
        self.model.soft_contact_ke = 5.0e5
        self.model.soft_contact_kd = 2000.0
        self.model.soft_contact_kf = 5.0e5
        self.model.soft_contact_mu = 0.9
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.008),
            dtype=wp.float32,
            device=self.model.device,
        )

        self.inflatable_solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab",
        )
        self.xpbd_solver = SolverXPBD(model=self.model, iterations=xpbd_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_inflatable = self.model.state()
        self.state_xpbd = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Proximity glue: pair inflatable particles with closest rigid vertex if within epsilon
        particle_q_np = np.array(self.state_0.particle_q.numpy(), dtype=np.float64)
        if particle_q_np.ndim == 1:
            particle_q_np = particle_q_np.reshape(-1, 3)
        body_q_np = self.state_0.body_q.numpy()
        row = body_q_np[self.rigid_body_id]
        p_b = np.array([row[0], row[1], row[2]], dtype=np.float64)
        q_b = wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        self.glue_particle_indices, self._body_anchors_local_np, min_dist = _build_proximity_glue_pairs(
            particle_q_np,
            self._rigid_vertices,
            p_b,
            q_b,
            self._start_particle,
            self._end_particle,
            self.glue_epsilon,
        )
        # Rest lengths = distance at initial config
        rest_lengths = []
        for i, (pidx, anchor_local) in enumerate(zip(self.glue_particle_indices, self._body_anchors_local_np)):
            p_soft = particle_q_np[pidx]
            anchor_world = p_b + np.array(
                wp.quat_rotate(q_b, wp.vec3(float(anchor_local[0]), float(anchor_local[1]), float(anchor_local[2])))
            )
            rest_lengths.append(float(np.linalg.norm(p_soft - anchor_world)))
        self.glue_rest_lengths = wp.array(rest_lengths, dtype=wp.float32, device=self.model.device)
        self.body_anchors_local = wp.array(self._body_anchors_local_np, dtype=wp.vec3, device=self.model.device)
        self.glue_particle_indices_wp = wp.array(self.glue_particle_indices, dtype=wp.int32, device=self.model.device)
        n_glue = len(self.glue_particle_indices)
        if n_glue > 0:
            print(
                f"   proximity glue: {n_glue} springs (epsilon={self.glue_epsilon}m), "
                f"rest_len: min={min(rest_lengths):.3f}m max={max(rest_lengths):.3f}m",
                flush=True,
            )
        else:
            min_str = f" (closest particle–rigid vert: {min_dist:.4f}m)" if min_dist is not None else ""
            print(
                f"   WARNING: no glue pairs found (epsilon={self.glue_epsilon}m).{min_str} Try increasing --glue_epsilon.",
                flush=True,
            )

        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
            # Zero world offsets so both display at actual physics positions (stacked, not side-by-side)
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        self.current_pressure = 1.0
        self.pressure_step = 0.1
        if self.viewer:
            if hasattr(self.viewer, "renderer") and hasattr(self.viewer.renderer, "register_key_press"):
                self.viewer.renderer.register_key_press(self._on_key_press)
            elif hasattr(self.viewer, "register_key_press"):
                self.viewer.register_key_press(self._on_key_press)

        print(
            "\n📦 Ready! I/K/O inflate | G/F glue stronger/weaker",
            flush=True,
        )

    def _on_key_press(self, symbol, modifiers):
        KEY_I, KEY_K, KEY_O = 105, 107, 111
        KEY_G, KEY_F = 103, 102
        if symbol == KEY_I:
            self.current_pressure = min(5.0, self.current_pressure + self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
        elif symbol == KEY_K:
            self.current_pressure = max(0.5, self.current_pressure - self.pressure_step)
            print(f"   [Pressure: {self.current_pressure:.2f}x]", flush=True)
        elif symbol == KEY_O:
            self.current_pressure = 1.0
            print("   [Reset pressure]", flush=True)
        elif symbol == KEY_G:
            self.glue_ke = min(1.0e6, self.glue_ke * self.glue_ke_step)
            self.glue_kd = min(2000.0, self.glue_kd * (self.glue_ke_step ** 0.5))
            print(f"   [Glue stronger: ke={self.glue_ke:.0f} kd={self.glue_kd:.0f}]", flush=True)
        elif symbol == KEY_F:
            self.glue_ke = max(1.0e3, self.glue_ke / self.glue_ke_step)
            self.glue_kd = max(10.0, self.glue_kd / (self.glue_ke_step ** 0.5))
            print(f"   [Glue weaker: ke={self.glue_ke:.0f} kd={self.glue_kd:.0f}]", flush=True)

    def _apply_glue(self, state):
        """Apply glue spring impulses for all face vertices."""
        n = len(self.glue_particle_indices)
        if n == 0:
            return
        wp.launch(
            apply_glue_proximity_impulse_kernel,
            dim=n,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.model.particle_inv_mass,
                state.body_q,
                state.body_qd,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.model.body_com,
                self.glue_particle_indices_wp,
                self.rigid_body_id,
                self.body_anchors_local,
                self.glue_rest_lengths,
                self.glue_ke,
                self.glue_kd,
                self.sim_dt,
                self.glue_max_vel,
                1.0 / max(0.1 * self.inflatable_mass, 0.01),
            ],
            device=self.model.device,
        )

    def _check_keys(self):
        if not self.viewer or not hasattr(self.viewer, "renderer"):
            return
        renderer = self.viewer.renderer
        if not hasattr(renderer, "is_key_down"):
            return
        if not hasattr(self, "_key_cooldown"):
            self._key_cooldown = 0
        if self._key_cooldown > 0:
            self._key_cooldown -= 1
            return
        KEY_I, KEY_K, KEY_O = 105, 107, 111
        KEY_G, KEY_F = 103, 102
        if renderer.is_key_down(KEY_I):
            self.current_pressure = min(5.0, self.current_pressure + self.pressure_step)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_K):
            self.current_pressure = max(0.5, self.current_pressure - self.pressure_step)
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_O):
            self.current_pressure = 1.0
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_G):
            self.glue_ke = min(1.0e6, self.glue_ke * self.glue_ke_step)
            self.glue_kd = min(2000.0, self.glue_kd * (self.glue_ke_step ** 0.5))
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_F):
            self.glue_ke = max(1.0e3, self.glue_ke / self.glue_ke_step)
            self.glue_kd = max(10.0, self.glue_kd / (self.glue_ke_step ** 0.5))
            self._key_cooldown = 10

    def step(self):
        self._check_keys()
        self.inflatable_solver.set_pressure(self.current_pressure)

        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(state=self.state_0)

            self.inflatable_solver.step(
                state_in=self.state_0,
                state_out=self.state_inflatable,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )
            self.xpbd_solver.step(
                state_in=self.state_0,
                state_out=self.state_xpbd,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )

            wp.copy(self.state_1.body_q, self.state_xpbd.body_q)
            wp.copy(self.state_1.body_qd, self.state_xpbd.body_qd)
            wp.copy(self.state_1.joint_q, self.state_xpbd.joint_q)
            wp.copy(self.state_1.joint_qd, self.state_xpbd.joint_qd)
            wp.copy(self.state_1.particle_q, self.state_inflatable.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_inflatable.particle_qd)

            self._apply_glue(self.state_1)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)

        # Draw glue springs: inflatable vertex -> rigid anchor for each proximity pair
        n = len(self.glue_particle_indices)
        if n == 0:
            self.viewer.end_frame()
            return
        particle_q_np = self.state_0.particle_q.numpy()
        body_tf = self.state_0.body_q.numpy()[self.rigid_body_id]
        X_wb = wp.transform(*body_tf)
        anchors_local_np = self._body_anchors_local_np
        wo = None
        w_rigid = 0
        if (
            hasattr(self.viewer, "world_offsets")
            and self.viewer.world_offsets is not None
            and self.viewer.world_offsets.shape[0] > 0
            and self.model.body_world is not None
        ):
            wo = self.viewer.world_offsets.numpy()
            bw = self.model.body_world.numpy() if hasattr(self.model.body_world, "numpy") else self.model.body_world
            w_rigid = int(bw[self.rigid_body_id]) if self.rigid_body_id < len(bw) else 0
        starts_list = []
        ends_list = []
        for i in range(n):
            p_soft_raw = np.array(particle_q_np[self.glue_particle_indices[i]])
            anchor_local = wp.vec3(float(anchors_local_np[i, 0]), float(anchors_local_np[i, 1]), float(anchors_local_np[i, 2]))
            anchor_world = np.array(wp.transform_point(X_wb, anchor_local))
            anchor_display = anchor_world + wo[w_rigid] if wo is not None and w_rigid >= 0 and w_rigid < len(wo) else anchor_world
            starts_list.append(p_soft_raw)
            ends_list.append(anchor_display)
        starts = wp.array(starts_list, dtype=wp.vec3, device=self.model.device)
        ends = wp.array(ends_list, dtype=wp.vec3, device=self.model.device)
        colors = wp.array([wp.vec3(1.0, 0.6, 0.0)] * n, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/glue_springs", starts, ends, colors, width=0.02)

        # Draw glue pairs as balls (soft=red, rigid=blue)
        self.viewer.log_points(
            "/glue_vertex_soft",
            wp.array(np.array(particle_q_np[self.glue_particle_indices]).astype(np.float32), dtype=wp.vec3, device=self.model.device),
            wp.array([0.015] * n, dtype=wp.float32, device=self.model.device),
            wp.array([wp.vec3(1.0, 0.2, 0.0)] * n, dtype=wp.vec3, device=self.model.device),
        )
        rigid_ball_positions = np.array([np.array(wp.transform_point(X_wb, wp.vec3(a[0], a[1], a[2]))) for a in anchors_local_np])
        if wo is not None and w_rigid >= 0 and w_rigid < len(wo):
            rigid_ball_positions = rigid_ball_positions + wo[w_rigid]
        self.viewer.log_points("/glue_vertex_rigid", wp.array(rigid_ball_positions.astype(np.float32), dtype=wp.vec3, device=self.model.device), wp.array([0.015] * n, dtype=wp.float32, device=self.model.device), wp.array([wp.vec3(0.2, 0.6, 1.0)] * n, dtype=wp.vec3, device=self.model.device))

        self.viewer.end_frame()

    def _debug_track_positions(self, frame: int):
        """Print min/max spring lengths for the glued proximity pairs."""
        n = len(self.glue_particle_indices)
        if n == 0:
            return
        particle_q_np = self.state_0.particle_q.numpy()
        body_q_np = self.state_0.body_q.numpy()
        X_wb = wp.transform(*body_q_np[self.rigid_body_id])
        rest_lengths = self.glue_rest_lengths.numpy()
        lengths = []
        for i, pidx in enumerate(self.glue_particle_indices):
            p_soft = particle_q_np[pidx]
            anchor = np.array(wp.transform_point(X_wb, wp.vec3(self._body_anchors_local_np[i, 0], self._body_anchors_local_np[i, 1], self._body_anchors_local_np[i, 2])))
            lengths.append(float(np.linalg.norm(p_soft - anchor)))
        lengths = np.array(lengths)
        elongations = lengths - rest_lengths
        print(
            f"[DEBUG] Frame {frame} -- glue proximity ({len(lengths)} springs): "
            f"len=[{lengths.min():.3f},{lengths.max():.3f}] elong=[{elongations.min():.3f},{elongations.max():.3f}]",
            flush=True,
        )

    def run(self, num_frames: int = 600, debug_track: bool = True):
        self.debug_track = debug_track
        for frame in range(num_frames):
            self.step()
            self.render()
            if self.debug_track and (frame <= 5 or frame % 30 == 0):
                self._debug_track_positions(frame)
                vol = self.inflatable_solver.get_volume_ratio(self.state_0)
                print(f"  volume_ratio={vol}", flush=True)
            if frame % 60 == 0 and frame > 0 and not self.debug_track:
                vol = self.inflatable_solver.get_volume_ratio(self.state_0)
                print(
                    f"   Frame {frame}: pressure={self.current_pressure:.2f}x, "
                    f"volume_ratio={vol:.2f}x",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Glue: Rigid + Inflatable connected by proximity-based springs"
    )
    parser.add_argument("--size", type=float, nargs=3, default=[0.4, 0.4, 0.4], help="Box size (w h d) m - same for both")
    parser.add_argument("--subdivisions", type=int, nargs=3, default=[5, 5, 5], help="Subdivisions - same for both")
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--rigid_mass", type=float, default=0.05, help="Lighter = rigid responds more to soft-body pull")
    parser.add_argument("--rigid_pos", type=float, nargs=3, default=None, metavar=("X","Y","Z"), help="Rigid box center (x y z); overrides rigid_z_base")
    parser.add_argument("--inflatable_pos", type=float, nargs=3, default=None, metavar=("X","Y","Z"), help="Inflatable box center (x y z); overrides stacking")
    parser.add_argument("--rigid_z_base", type=float, default=0.0, help="Rigid box bottom z (when not using --rigid_pos)")
    parser.add_argument("--stack_gap", type=float, default=0.0, help="Gap (m) between rigid top and inflatable bottom when stacked")
    parser.add_argument("--glue_epsilon", type=float, default=0.05, help="Max distance (m) for proximity glue pairs at init")
    parser.add_argument("--glue_ke", type=float, default=5.0e4, help="Glue spring stiffness")
    parser.add_argument("--glue_kd", type=float, default=200.0, help="Glue spring damping")
    parser.add_argument("--glue_max_vel", type=float, default=1.5, help="Max velocity change per substep (prevents explosion)")
    parser.add_argument("--k_mu", type=float, default=1.0e5)
    parser.add_argument("--k_lambda", type=float, default=1.0e5)
    parser.add_argument("--k_damp", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max_pressure", type=float, default=5.0)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--xpbd_iterations", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=1800)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-debug-track", action="store_true", help="Disable position tracking (on by default)")

    args = parser.parse_args()

    wp.init()
    with wp.ScopedDevice(args.device):
        if args.headless:
            viewer = None
        else:
            try:
                viewer = newton.viewer.ViewerGL(width=1920, height=1080)
            except Exception as e:
                print(f"Could not create OpenGL viewer: {e}")
                try:
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                except Exception as e2:
                    print(f"Could not create Rerun viewer: {e2}")
                    viewer = None

        example = Example(
            viewer=viewer,
            size=args.size,
            subdivisions=args.subdivisions,
            mass=args.mass,
            rigid_mass=args.rigid_mass,
            rigid_pos=args.rigid_pos,
            inflatable_pos=args.inflatable_pos,
            rigid_z_base=args.rigid_z_base,
            stack_gap=args.stack_gap,
            glue_epsilon=args.glue_epsilon,
            glue_ke=args.glue_ke,
            glue_kd=args.glue_kd,
            glue_max_vel=args.glue_max_vel,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            gravity=args.gravity,
            max_pressure=args.max_pressure,
            substeps=args.substeps,
            xpbd_iterations=args.xpbd_iterations,
        )
        example.run(num_frames=args.num_frames, debug_track=not args.no_debug_track)


if __name__ == "__main__":
    main()
