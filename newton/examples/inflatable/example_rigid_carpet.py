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
Rigid Carpet - Rigid plates only: same rigid SurfaceBox size/subdivisions and index-based glue as used for worm + rigid base.

No inflatable. Use this to debug rigid-rigid glue and plate behavior without the worm:
  - Same rigid_size = (length, width/rigid_subdivision, rigid_height)
  - Same rigid_subdivisions = (subdivisions_x, subdivisions_y, 1) with subdivisions_y for Y (worm: rigid_subdivision_y)
  - Same index-based glue: glue_utils.build_rigid_rigid_glue_by_index (plate i +Y face → plate i+1 -Y face)

Collision: Ground uses collision_group=-1 (collides with all). Each plate has a unique positive
collision_group so rigids collide only with the ground, not with each other.

Run: python -m newton.examples rigid_carpet [--headless]
Override params: ... rigid_carpet --length 2 --width 1.5 --rigid_height 0.2 --subdivisions_x 8 --rigid_subdivision 20
"""

import argparse
import warp as wp
import numpy as np

import newton
from newton.solvers import SolverXPBD
from newton._src.sim import SurfaceBox, box_topology, glue_utils


@wp.kernel
def apply_glue_rigid_rigid_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_inv_m: wp.array(dtype=float),
    body_inv_I: wp.array(dtype=wp.mat33),
    body_com: wp.array(dtype=wp.vec3),
    glue_body_a: wp.array(dtype=int),
    glue_body_b: wp.array(dtype=int),
    anchor_a_local: wp.array(dtype=wp.vec3),
    anchor_b_local: wp.array(dtype=wp.vec3),
    rest_lengths: wp.array(dtype=float),
    glue_ke: float,
    glue_kd: float,
    dt: float,
    max_vel_body_per_substep: float,
):
    """Apply spring impulse between two rigid body anchor points."""
    tid = wp.tid()
    body_a = glue_body_a[tid]
    body_b = glue_body_b[tid]
    anchor_a = anchor_a_local[tid]
    anchor_b = anchor_b_local[tid]
    rest_len = rest_lengths[tid]

    X_a = body_q[body_a]
    q_a = wp.transform_get_rotation(X_a)
    p_a = wp.transform_get_translation(X_a)
    pos_a = p_a + wp.quat_rotate(q_a, anchor_a)

    X_b = body_q[body_b]
    q_b = wp.transform_get_rotation(X_b)
    p_b = wp.transform_get_translation(X_b)
    pos_b = p_b + wp.quat_rotate(q_b, anchor_b)

    d = pos_a - pos_b
    length = wp.length(d)
    if length < 1.0e-6:
        return
    direction = d / length
    elongation = length - rest_len

    v_s_a = body_qd[body_a]
    v_s_b = body_qd[body_b]
    vel_a = wp.spatial_top(v_s_a) + wp.cross(
        wp.spatial_bottom(v_s_a),
        pos_a - (p_a + wp.quat_rotate(q_a, body_com[body_a])),
    )
    vel_b = wp.spatial_top(v_s_b) + wp.cross(
        wp.spatial_bottom(v_s_b),
        pos_b - (p_b + wp.quat_rotate(q_b, body_com[body_b])),
    )
    rel_vel = vel_a - vel_b
    vn = wp.dot(direction, rel_vel)
    F_mag = glue_ke * elongation + glue_kd * vn
    F_on_a = -direction * F_mag
    F_on_b = direction * F_mag

    inv_m_a = body_inv_m[body_a]
    inv_m_b = body_inv_m[body_b]
    I_inv_a = body_inv_I[body_a]
    I_inv_b = body_inv_I[body_b]

    r_a = pos_a - (p_a + wp.quat_rotate(q_a, body_com[body_a]))
    r_b = pos_b - (p_b + wp.quat_rotate(q_b, body_com[body_b]))

    dv_a = F_on_a * inv_m_a * dt
    dv_b = F_on_b * inv_m_b * dt
    torque_a = wp.cross(r_a, F_on_a)
    torque_b = wp.cross(r_b, F_on_b)
    d_omega_a = wp.quat_rotate_inv(q_a, torque_a) * I_inv_a * dt
    d_omega_b = wp.quat_rotate_inv(q_b, torque_b) * I_inv_b * dt
    d_omega_a_w = wp.quat_rotate(q_a, d_omega_a)
    d_omega_b_w = wp.quat_rotate(q_b, d_omega_b)

    dv_a_mag = wp.length(dv_a)
    if dv_a_mag > max_vel_body_per_substep:
        dv_a = dv_a * (max_vel_body_per_substep / dv_a_mag)
    dv_b_mag = wp.length(dv_b)
    if dv_b_mag > max_vel_body_per_substep:
        dv_b = dv_b * (max_vel_body_per_substep / dv_b_mag)
    d_omega_a_mag = wp.length(d_omega_a_w)
    if d_omega_a_mag > max_vel_body_per_substep:
        d_omega_a_w = d_omega_a_w * (max_vel_body_per_substep / d_omega_a_mag)
    d_omega_b_mag = wp.length(d_omega_b_w)
    if d_omega_b_mag > max_vel_body_per_substep:
        d_omega_b_w = d_omega_b_w * (max_vel_body_per_substep / d_omega_b_mag)

    wp.atomic_add(body_qd, body_a, wp.spatial_vector(dv_a, d_omega_a_w))
    wp.atomic_add(body_qd, body_b, wp.spatial_vector(dv_b, d_omega_b_w))


class Example:
    """Rigid carpet: N plates on the ground, connected by glue springs. Debug example."""

    def __init__(
        self,
        viewer,
        length: float = 1.0,
        width: float = 1.0,
        rigid_subdivision: int = 15,
        subdivisions_x: int = 12,
        subdivisions_y: int = 1,
        subdivisions_z: int = 1,
        rigid_height: float = 0.1,
        rigid_mass: float = 0.005,
        glue_axis: str = "y",
        glue_epsilon: float = 0.02,
        glue_ke_rr: float = 6.0e3,
        glue_kd_rr: float = 80.0,
        glue_max_vel_body: float = 0.15,
        substeps: int = 5,
        xpbd_iterations: int = 10,
        num_frames: int = 7200,
        debug: bool = True,
        drop_height: float = 0.0,
        plate_plate_collision: bool = True,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.debug = debug
        self.num_frames = num_frames
        self.viewer = viewer
        self.plate_plate_collision = plate_plate_collision
        if glue_axis not in ("x", "y", "z"):
            raise ValueError(f"glue_axis must be 'x', 'y', or 'z', got {glue_axis!r}")
        self._glue_axis = glue_axis
        n = rigid_subdivision

        # Plate size and subdivisions depend on glue_axis (stacking direction).
        # rigid_subdivision = number of rigid bodies; subdivisions_y = subdivision within each rigid on Y axis.
        # In-plane: axis y → (x,z); axis x → (y,z); axis z → (x,y). Y slot uses subdivisions_y.
        if glue_axis == "y":
            rigid_size = (length, width / n, rigid_height)
            rigid_subdivisions = (subdivisions_x, subdivisions_y, subdivisions_z)
            rigid_sx, rigid_sy, rigid_sz = subdivisions_x, subdivisions_y, subdivisions_z
        elif glue_axis == "x":
            rigid_size = (length / n, width, rigid_height)
            rigid_subdivisions = (1, subdivisions_y, subdivisions_z)
            rigid_sx, rigid_sy, rigid_sz = 1, subdivisions_y, subdivisions_z
        else:  # z
            rigid_size = (length, width, rigid_height / n)
            rigid_subdivisions = (subdivisions_x, subdivisions_y, 1)
            rigid_sx, rigid_sy, rigid_sz = subdivisions_x, subdivisions_y, 1

        # Base z: so bottom of lowest plate is at z=0 when drop_height=0.
        rigid_size_z = rigid_size[2]
        base_z = rigid_size_z / 2.0
        rigid_center_z = base_z + drop_height

        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=5e5, kd=6e3, kf=1e4, mu=0.5, collision_group=-1, restitution=0.0)
        builder.add_ground_plane(cfg=ground_cfg)
        builder.begin_world()

        if drop_height > 0:
            print(f"\n🏖 Rigid Carpet: {rigid_subdivision} plates (glue {glue_axis}+↔{glue_axis}-) starting {drop_height}m above ground, dropping", flush=True)
        else:
            print(f"\n🏖 Rigid Carpet: {rigid_subdivision} plates (glue {glue_axis}+↔{glue_axis}-) ON the ground (z=0), connected by glue", flush=True)
        surface_box = SurfaceBox(size=rigid_size, subdivisions=rigid_subdivisions, verbose=False)
        surface_data = surface_box.get_mesh_data()
        # Use the same vertex array as the mesh so index-based glue matches box_topology (same mesh, same order).
        rigid_vertices = np.asarray(surface_box.vertices, dtype=np.float64)
        if glue_axis == "y" and rigid_subdivisions[0] > 0 and rigid_subdivisions[2] > 0:
            # Sanity check: first pair (i=0,k=0) should have same (x,z) in local frame.
            idx_p, idx_m = box_topology.vertex_index(rigid_sx, rigid_sy, rigid_sz, 0, rigid_sy, 0), box_topology.vertex_index(rigid_sx, rigid_sy, rigid_sz, 0, 0, 0)
            vp, vm = rigid_vertices[idx_p], rigid_vertices[idx_m]
            in_plane = (0, 2)  # x, z for Y
            if np.linalg.norm(np.array([float(vp[0]) - float(vm[0]), float(vp[2]) - float(vm[2])])) > 1e-5:
                print(f"   [WARN] Mesh vertex order may not match box_topology: first pair idx ({idx_p},{idx_m}) has (x,z) v_plus=({vp[0]:.4f},{vp[2]:.4f}) v_minus=({vm[0]:.4f},{vm[2]:.4f})", flush=True)
        rigid_mesh = newton.Mesh(
            surface_data["vertices"],
            surface_data["indices"],
            compute_inertia=True,
            is_solid=False,
        )
        scale = rigid_mass / rigid_mesh.mass if rigid_mesh.mass > 0 else 1.0
        I_np = np.array(rigid_mesh.I) * scale
        plate_cfg = newton.ModelBuilder.ShapeConfig(ke=5e5, kd=2e3, kf=1e4, mu=0.5, restitution=0.0)
        plate_cfg.has_particle_collision = False
        plate_cfg.has_shape_collision = True
        # Unique collision group per plate so rigids only collide with ground (-1), not with each other
        plate_shape_ids = []
        self.rigid_body_ids = []
        for i in range(rigid_subdivision):
            plate_cfg.collision_group = 1 + i
            if glue_axis == "y":
                rigid_center_y = -width / 2.0 + (i + 0.5) * (width / n)
                rigid_pos = (0.0, rigid_center_y, rigid_center_z)
            elif glue_axis == "x":
                rigid_center_x = -length / 2.0 + (i + 0.5) * (length / n)
                rigid_pos = (rigid_center_x, 0.0, rigid_center_z)
            else:  # z: stack vertically
                plate_z = base_z + drop_height + i * (rigid_height / n)
                rigid_pos = (0.0, 0.0, plate_z)
            body_id = builder.add_body(
                xform=wp.transform(
                    wp.vec3(float(rigid_pos[0]), float(rigid_pos[1]), float(rigid_pos[2])),
                    wp.quat_identity(),
                ),
                mass=rigid_mass,
                com=rigid_mesh.com,
                I_m=wp.mat33(I_np),
            )
            joint_id = builder.add_joint_free(body_id)
            builder.add_articulation([joint_id], key=f"plate_{i}")
            shape_id = builder.add_shape_mesh(body=body_id, mesh=rigid_mesh, cfg=plate_cfg)
            plate_shape_ids.append(shape_id)
            self.rigid_body_ids.append(body_id)

        self.model = builder.finalize()
        self._rigid_vertices = rigid_vertices

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Index-based glue: plate i +axis face → plate i+1 -axis face (axis from glue_axis)
        body_q_np = np.array(self.state_0.body_q.numpy(), dtype=np.float64)
        if body_q_np.ndim == 1:
            body_q_np = body_q_np.reshape(-1, 7)
        rr_a, rr_b, rr_anc_a, rr_anc_b, rr_rest, rr_idx_a, rr_idx_b = glue_utils.build_rigid_rigid_glue_by_index(
            body_q_np,
            self.rigid_body_ids,
            self._rigid_vertices,
            rigid_sx,
            rigid_sy,
            rigid_sz,
            glue_axis=glue_axis,
            debug_mismatch=debug,
        )
        self.rr_body_a = rr_a
        self.rr_body_b = rr_b
        self._rr_anchor_a_np = rr_anc_a if rr_anc_a.size > 0 else np.empty((0, 3), dtype=np.float32)
        self._rr_anchor_b_np = rr_anc_b if rr_anc_b.size > 0 else np.empty((0, 3), dtype=np.float32)
        self.rr_rest_lengths = rr_rest
        self.rr_body_a_wp = wp.array(self.rr_body_a, dtype=wp.int32, device=self.model.device)
        self.rr_body_b_wp = wp.array(self.rr_body_b, dtype=wp.int32, device=self.model.device)
        self.rr_anchor_a_wp = wp.array(self._rr_anchor_a_np, dtype=wp.vec3, device=self.model.device)
        self.rr_anchor_b_wp = wp.array(self._rr_anchor_b_np, dtype=wp.vec3, device=self.model.device)
        self.rr_rest_lengths_wp = wp.array(self.rr_rest_lengths, dtype=wp.float32, device=self.model.device)

        self.glue_ke_rr = glue_ke_rr
        self.glue_kd_rr = glue_kd_rr
        self.glue_max_vel_body = glue_max_vel_body

        # For visualization: color springs by Z-layer so multiple layers are visible
        self._rr_n_pairs_per_interface = box_topology.num_side_vertices(rigid_sx, rigid_sy, rigid_sz, glue_axis)
        self._rr_sz = rigid_sz

        self.xpbd_solver = SolverXPBD(self.model, iterations=xpbd_iterations)

        n_rr = len(self.rr_body_a)
        n_interface = max(0, len(self.rigid_body_ids) - 1)
        n_pairs_per = self._rr_n_pairs_per_interface
        expected_rr = n_interface * n_pairs_per
        n_z_layers = rigid_sz + 1
        print(f"   rigid-rigid glue: {n_rr} springs (index-based, topology), {n_z_layers} Z-layer(s) per face", flush=True)
        if expected_rr != n_rr:
            print(f"   [WARN] expected {expected_rr} springs (interfaces={n_interface} × pairs/face={n_pairs_per})", flush=True)
        if n_rr > 0:
            rmin, rmax = float(np.min(self.rr_rest_lengths)), float(np.max(self.rr_rest_lengths))
            print(f"   rest_len: min={rmin:.6g} max={rmax:.6g} m", flush=True)

        if self.debug and n_rr > 0 and glue_axis == "y":
            glue_utils.validate_rigid_rigid_glue_indices(
                body_q_np,
                self.rigid_body_ids,
                self._rigid_vertices,
                rigid_sx,
                rigid_sy,
                rigid_sz,
                glue_axis=glue_axis,
                distance_threshold=0.02,
                verbose=True,
            )
            self._debug_print_rigid_springs()

        if viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = False

        print(f"   Plates at z_center={rigid_center_z:.4f}. [G]/[F] glue stronger/weaker", flush=True)

    def _debug_print_rigid_springs(self):
        """Print rigid spring positions for debugging."""
        body_q_np = np.array(self.state_0.body_q.numpy())
        if body_q_np.ndim == 1:
            body_q_np = body_q_np.reshape(-1, 7)
        n_total = len(self.rr_body_a)
        print("\n[DEBUG] Rigid spring positions (init):", flush=True)
        elongations = []
        for k in range(min(n_total, 15)):
            ba, bb = int(self.rr_body_a[k]), int(self.rr_body_b[k])
            anchor_a, anchor_b = self._rr_anchor_a_np[k], self._rr_anchor_b_np[k]
            rest = float(self.rr_rest_lengths[k])
            row_a, row_b = body_q_np[ba], body_q_np[bb]
            p_a = np.array([row_a[0], row_a[1], row_a[2]], dtype=np.float64)
            q_a = wp.quat(float(row_a[3]), float(row_a[4]), float(row_a[5]), float(row_a[6]))
            p_b = np.array([row_b[0], row_b[1], row_b[2]], dtype=np.float64)
            q_b = wp.quat(float(row_b[3]), float(row_b[4]), float(row_b[5]), float(row_b[6]))
            pos_a = p_a + np.array(wp.quat_rotate(q_a, wp.vec3(float(anchor_a[0]), float(anchor_a[1]), float(anchor_a[2]))))
            pos_b = p_b + np.array(wp.quat_rotate(q_b, wp.vec3(float(anchor_b[0]), float(anchor_b[1]), float(anchor_b[2]))))
            actual = float(np.linalg.norm(pos_a - pos_b))
            elon = actual - rest
            elongations.append(elon)
            print(f"  rr[{k}] body_a={ba} body_b={bb} pos_a=({pos_a[0]:.3f},{pos_a[1]:.3f},{pos_a[2]:.3f}) pos_b=({pos_b[0]:.3f},{pos_b[1]:.3f},{pos_b[2]:.3f}) rest={rest:.4f} actual={actual:.4f} elon={elon:.4f}", flush=True)
        for k in range(15, n_total):
            ba, bb = int(self.rr_body_a[k]), int(self.rr_body_b[k])
            anchor_a, anchor_b = self._rr_anchor_a_np[k], self._rr_anchor_b_np[k]
            rest = float(self.rr_rest_lengths[k])
            row_a, row_b = body_q_np[ba], body_q_np[bb]
            p_a = np.array([row_a[0], row_a[1], row_a[2]], dtype=np.float64)
            q_a = wp.quat(float(row_a[3]), float(row_a[4]), float(row_a[5]), float(row_a[6]))
            p_b = np.array([row_b[0], row_b[1], row_b[2]], dtype=np.float64)
            q_b = wp.quat(float(row_b[3]), float(row_b[4]), float(row_b[5]), float(row_b[6]))
            pos_a = p_a + np.array(wp.quat_rotate(q_a, wp.vec3(float(anchor_a[0]), float(anchor_a[1]), float(anchor_a[2]))))
            pos_b = p_b + np.array(wp.quat_rotate(q_b, wp.vec3(float(anchor_b[0]), float(anchor_b[1]), float(anchor_b[2]))))
            actual = float(np.linalg.norm(pos_a - pos_b))
            elongations.append(actual - rest)
        if n_total > 15:
            print(f"  ... +{n_total - 15} more", flush=True)
        if elongations:
            print(f"  elongations: min={min(elongations):.4f} max={max(elongations):.4f} mean={np.mean(elongations):.4f}", flush=True)
        print("", flush=True)

    def _debug_print_runtime(self, state):
        if len(self.rr_body_a) == 0:
            return
        body_q_np = np.array(state.body_q.numpy()).reshape(-1, 7)
        elongations = []
        for k in range(len(self.rr_body_a)):
            ba, bb = int(self.rr_body_a[k]), int(self.rr_body_b[k])
            anchor_a, anchor_b = self._rr_anchor_a_np[k], self._rr_anchor_b_np[k]
            rest = float(self.rr_rest_lengths[k])
            row_a, row_b = body_q_np[ba], body_q_np[bb]
            p_a = np.array([row_a[0], row_a[1], row_a[2]], dtype=np.float64)
            q_a = wp.quat(float(row_a[3]), float(row_a[4]), float(row_a[5]), float(row_a[6]))
            p_b = np.array([row_b[0], row_b[1], row_b[2]], dtype=np.float64)
            q_b = wp.quat(float(row_b[3]), float(row_b[4]), float(row_b[5]), float(row_b[6]))
            pos_a = p_a + np.array(wp.quat_rotate(q_a, wp.vec3(float(anchor_a[0]), float(anchor_a[1]), float(anchor_a[2]))))
            pos_b = p_b + np.array(wp.quat_rotate(q_b, wp.vec3(float(anchor_b[0]), float(anchor_b[1]), float(anchor_b[2]))))
            actual = float(np.linalg.norm(pos_a - pos_b))
            elongations.append((k, actual - rest, actual, rest))
        worst = max(elongations, key=lambda x: abs(x[1]))
        print(f"   [DEBUG] t={self.sim_time:.2f}s elong min={min(e[1] for e in elongations):.4f} max={max(e[1] for e in elongations):.4f} worst=rr[{worst[0]}] elon={worst[1]:.4f} (actual={worst[2]:.4f} rest={worst[3]:.4f})", flush=True)

    def _apply_glue(self, state):
        n_rr = len(self.rr_body_a)
        if n_rr > 0:
            wp.launch(
                apply_glue_rigid_rigid_kernel,
                dim=n_rr,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                    self.model.body_com,
                    self.rr_body_a_wp,
                    self.rr_body_b_wp,
                    self.rr_anchor_a_wp,
                    self.rr_anchor_b_wp,
                    self.rr_rest_lengths_wp,
                    self.glue_ke_rr,
                    self.glue_kd_rr,
                    self.sim_dt,
                    self.glue_max_vel_body,
                ],
                device=self.model.device,
            )

    def step(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(state=self.state_0)
            self.xpbd_solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )
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
        # Draw rigid-rigid glue springs (anchor A → anchor B in world). Color by Z-layer so multiple layers are visible.
        n_rr = len(self.rr_body_a)
        if n_rr > 0:
            body_q_np = self.state_0.body_q.numpy()
            if body_q_np.ndim == 1:
                body_q_np = body_q_np.reshape(-1, 7)
            starts_list = []
            ends_list = []
            colors_list = []
            n_per = self._rr_n_pairs_per_interface
            n_z = self._rr_sz + 1  # number of Z layers (vertices along Z on the glue face)
            # Layer colors: alternate so each Z layer is distinct (e.g. green / cyan)
            layer_colors = [
                (0.2, 0.8, 0.4),  # green
                (0.2, 0.6, 0.9),  # cyan
                (0.9, 0.6, 0.2),  # orange
            ]
            for k in range(n_rr):
                ba = int(self.rr_body_a[k])
                bb = int(self.rr_body_b[k])
                anchor_a = self._rr_anchor_a_np[k]
                anchor_b = self._rr_anchor_b_np[k]
                row_a = body_q_np[ba]
                row_b = body_q_np[bb]
                p_a = np.array([row_a[0], row_a[1], row_a[2]], dtype=np.float64)
                q_a = wp.quat(float(row_a[3]), float(row_a[4]), float(row_a[5]), float(row_a[6]))
                p_b = np.array([row_b[0], row_b[1], row_b[2]], dtype=np.float64)
                q_b = wp.quat(float(row_b[3]), float(row_b[4]), float(row_b[5]), float(row_b[6]))
                pos_a = p_a + np.array(wp.quat_rotate(q_a, wp.vec3(float(anchor_a[0]), float(anchor_a[1]), float(anchor_a[2]))))
                pos_b = p_b + np.array(wp.quat_rotate(q_b, wp.vec3(float(anchor_b[0]), float(anchor_b[1]), float(anchor_b[2]))))
                starts_list.append(pos_a)
                ends_list.append(pos_b)
                # Pair index within interface: k % n_per. Topology order is i then k; k=0 is lower z, k=sz is upper z.
                z_layer = (k % n_per) % n_z
                rgb = layer_colors[z_layer % len(layer_colors)]
                colors_list.append(wp.vec3(rgb[0], rgb[1], rgb[2]))
            starts = wp.array(starts_list, dtype=wp.vec3, device=self.model.device)
            ends = wp.array(ends_list, dtype=wp.vec3, device=self.model.device)
            colors = wp.array(colors_list, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_lines("/glue_springs_rr", starts, ends, colors, width=0.015)
        self.viewer.end_frame()

    def run(self, num_frames: int | None = None):
        n = num_frames if num_frames is not None else self.num_frames
        print(f"\n🏖 Rigid Carpet running...", flush=True)
        for frame in range(n):
            self.step()
            self.render()
            if frame % 120 == 0:
                body_q = self.state_0.body_q.numpy()
                z_vals = body_q[: len(self.rigid_body_ids), 2]
                print(f"   Frame {frame}: body z min={float(np.min(z_vals)):.4f} max={float(np.max(z_vals)):.4f}", flush=True)
            if self.debug and len(self.rr_body_a) > 0 and frame % 60 == 0 and frame > 0:
                self._debug_print_runtime(self.state_0)
        print(f"\n🏖 Done.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Rigid Carpet: debug rigid-rigid glue springs")
    parser.add_argument("--length", type=float, default=1.0, help="Plate length (X) in m; larger helps proximity validation (default: 1.0)")
    parser.add_argument("--width", type=float, default=1.0, help="Total carpet width (Y) in m (default: 1.0)")
    parser.add_argument("--rigid_subdivision", type=int, default=15, help="Number of rigid bodies (default: 15)")
    parser.add_argument("--subdivisions_x", type=int, default=12, help="Rigid mesh subdivisions along first in-plane axis (default: 12)")
    parser.add_argument("--subdivisions_y", type=int, default=1, help="Subdivision within each rigid on Y axis (default: 1)")
    parser.add_argument("--subdivisions_z", type=int, default=1, help="Rigid mesh subdivisions along second in-plane axis (default: 1)")
    parser.add_argument("--rigid_height", type=float, default=0.1, help="Rigid Z size (height) in m (default: 0.1)")
    parser.add_argument(
        "--glue_axis",
        type=str,
        default="y",
        choices=("x", "y", "z"),
        help="Attachment axis: y = Y+↔Y- (plates along Y), x = X+↔X-, z = Z+↔Z- (default: y)",
    )
    parser.add_argument("--rigid_mass", type=float, default=0.005, help="Mass per rigid body (kg); default 0.005 for light rigids")
    parser.add_argument("--glue_epsilon", type=float, default=0.02)
    parser.add_argument("--glue_ke_rr", type=float, default=6.0e3, help="Rigid-rigid glue stiffness (N/m); default 6e3 (softer for stability)")
    parser.add_argument("--glue_kd_rr", type=float, default=80.0, help="Rigid-rigid glue damping; default 80")
    parser.add_argument("--glue_max_vel_body", type=float, default=0.15)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--xpbd_iterations", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=7200)
    parser.add_argument("--drop_height", type=float, default=0.3, help="Start carpet above ground and drop (default 0.3m)")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug output")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", type=str, default="rtx", choices=["gl", "rtx", "rerun", "null"], help="Viewer type (default: rtx)")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    wp.init()
    with wp.ScopedDevice(args.device):
        if args.headless or args.viewer == "null":
            viewer = None
        elif args.viewer == "rtx":
            try:
                viewer = newton.viewer.ViewerRTX(headless=False, width=1920, height=1080)
            except Exception as e:
                print(f"RTX viewer failed: {e}, falling back to GL")
                try:
                    viewer = newton.viewer.ViewerGL(width=1920, height=1080)
                except Exception:
                    viewer = None
        elif args.viewer == "gl":
            try:
                viewer = newton.viewer.ViewerGL(width=1920, height=1080)
            except Exception as e:
                print(f"Viewer failed: {e}")
                viewer = None
        elif args.viewer == "rerun":
            try:
                viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
            except Exception as e:
                print(f"Rerun viewer failed: {e}")
                viewer = None
        else:
            viewer = None
        example = Example(
            viewer=viewer,
            length=args.length,
            width=args.width,
            rigid_subdivision=args.rigid_subdivision,
            subdivisions_x=args.subdivisions_x,
            subdivisions_y=args.subdivisions_y,
            subdivisions_z=args.subdivisions_z,
            rigid_height=args.rigid_height,
            glue_axis=args.glue_axis,
            rigid_mass=args.rigid_mass,
            glue_epsilon=args.glue_epsilon,
            glue_ke_rr=args.glue_ke_rr,
            glue_kd_rr=args.glue_kd_rr,
            glue_max_vel_body=args.glue_max_vel_body,
            substeps=args.substeps,
            xpbd_iterations=args.xpbd_iterations,
            num_frames=args.num_frames,
            debug=not args.no_debug,
            drop_height=args.drop_height,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
