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
Inflatable Table with 4 Soft Legs Glued to Rigid Legs and Plate.

Each soft (inflatable) box has glue on BOTTOM and TOP:
  - BOTTOM: glued to its rigid leg (heavy box on ground).
  - TOP: glued to the rigid plate.
Rigid legs and plate use SurfaceBox meshes; proximity springs connect within epsilon.

Usage:
    python -m newton.examples inflatable_table_glue [--glue_epsilon 0.05]
    ./run-examples.sh inflatable_table_glue

Keys: [1-4] select leg | [I/K] inflate/deflate selected | [A/Z] all | [O] reset | [G/F] glue
"""

import argparse
import warp as wp
import numpy as np

import newton
from newton.solvers import SolverInflatable, SolverXPBD, TetraBox
from newton._src.sim import SurfaceBox
from newton._src.glue import build_proximity_glue_pairs, apply_glue_proximity_impulse_kernel


@wp.kernel
def scale_spring_range_kernel(
    original_rest_lengths: wp.array(dtype=wp.float32),
    scale: wp.float32,
    start_idx: wp.int32,
    count: wp.int32,
    scaled_rest_lengths: wp.array(dtype=wp.float32),
):
    """Scale spring rest lengths for a specific range."""
    tid = wp.tid()
    if tid < count:
        idx = start_idx + tid
        scaled_rest_lengths[idx] = original_rest_lengths[idx] * scale


@wp.kernel
def scale_tet_range_kernel(
    original_poses: wp.array(dtype=wp.mat33),
    scale: wp.float32,
    start_idx: wp.int32,
    count: wp.int32,
    scaled_poses: wp.array(dtype=wp.mat33),
):
    """Scale tetrahedra rest poses for a specific range."""
    tid = wp.tid()
    if tid < count:
        idx = start_idx + tid
        inv_scale = 1.0 / scale
        orig = original_poses[idx]
        scaled_poses[idx] = wp.mat33(
            orig[0, 0] * inv_scale, orig[0, 1] * inv_scale, orig[0, 2] * inv_scale,
            orig[1, 0] * inv_scale, orig[1, 1] * inv_scale, orig[1, 2] * inv_scale,
            orig[2, 0] * inv_scale, orig[2, 1] * inv_scale, orig[2, 2] * inv_scale
        )


class Example:
    """
    Table: one rigid plate (SurfaceBox) + 4 inflatable soft legs at corners.
    Legs are glued to the plate bottom by proximity-based springs.
    """

    def __init__(
        self,
        viewer,
        leg_size=(0.4, 0.4, 0.4),
        subdivisions=(5, 5, 5),
        plate_width: float = 2.0,
        plate_height: float = 0.05,
        mass: float = 1.0,
        plate_mass: float = 0.05,
        rigid_leg_mass: float = 10000000.0,
        particle_radius: float = 0.008,
        glue_epsilon: float = 0.05,
        glue_ke: float = 3.0e4,
        glue_kd: float = 300.0,
        glue_max_vel: float = 1.0,
        k_mu: float = 1.0e5,
        k_lambda: float = 1.0e5,
        k_damp: float = 1.0,
        spring_ke: float = 5.0e4,
        spring_kd: float = 1.0,
        gravity: float = 9.81,
        max_pressure: float = 5.0,
        substeps: int = 8,
        xpbd_iterations: int = 12,
        marble_radius: float = 0.03,
        marble_mass: float = 0.05,
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

        self.leg_size = to_size(leg_size) or (0.4, 0.4, 0.4)
        self.subdivisions = to_sub(subdivisions) or (5, 5, 5)
        self.plate_width = plate_width
        self.plate_height = plate_height
        self.mass = mass
        self.plate_mass = plate_mass
        self.rigid_leg_mass = rigid_leg_mass
        self.particle_radius = particle_radius
        self.glue_epsilon = glue_epsilon
        self.glue_ke = glue_ke
        self.glue_kd = glue_kd
        self.glue_ke_step = 1.5
        self.glue_max_vel = glue_max_vel
        self.inflatable_mass = mass
        self.max_pressure = max_pressure
        self.viewer = viewer
        self.selected_leg = 0
        self.leg_pressures = [1.0, 1.0, 1.0, 1.0]
        self.pressure_step = 0.1
        self.marble_radius = marble_radius
        self.marble_mass = marble_mass

        w, h, d = self.leg_size
        sx, sy, sz = self.subdivisions

        # Plate: resolution matched to legs (same vertex spacing as leg top face)
        # leg spacing = w/sx; plate needs plate_width / (w/sx) = plate_width*sx/w subdivisions
        plate_sub_x = max(sx, int(round(self.plate_width * sx / w)))
        plate_sub_y = max(sy, int(round(self.plate_width * sy / h)))
        plate_size = (self.plate_width, self.plate_width, self.plate_height)
        plate_subdivisions = (plate_sub_x, plate_sub_y, 1)

        # Stack: ground -> rigid legs (z=d/2) -> soft legs (z=d+d/2) -> plate (z=2d+plate_h/2)
        rigid_leg_center_z = d / 2.0
        soft_leg_center_z = d + d / 2.0
        leg_top_z = soft_leg_center_z + d / 2.0
        plate_center_z = leg_top_z + self.plate_height / 2.0

        ox = self.plate_width / 2.0 - w / 2.0
        corner_positions = [
            (ox, ox),
            (ox, -ox),
            (-ox, ox),
            (-ox, -ox),
        ]

        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=5e5, kd=1e3, kf=1e4, mu=0.5)
        )

        builder.begin_world()

        # --- Four rigid legs (SurfaceBox same size as soft legs, for proximity glue) ---
        print("\n📦 [1/4] Creating 4 rigid legs (SurfaceBox, heavy for stability)...", flush=True)
        rigid_leg_surface = SurfaceBox(
            size=self.leg_size,
            subdivisions=self.subdivisions,
            verbose=False,
        )
        rigid_leg_surface_data = rigid_leg_surface.get_mesh_data()
        self._rigid_leg_vertices = np.array(rigid_leg_surface_data["vertices"])
        rigid_leg_mesh = newton.Mesh(
            rigid_leg_surface_data["vertices"],
            rigid_leg_surface_data["indices"],
            compute_inertia=True,
            is_solid=False,
        )
        scale = rigid_leg_mass / rigid_leg_mesh.mass if rigid_leg_mesh.mass > 0 else 1.0
        rigid_leg_I_np = np.array(rigid_leg_mesh.I) * scale
        rigid_leg_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=5e5, kd=100.0, kf=1e4, mu=0.5)
        rigid_leg_shape_cfg.has_particle_collision = False
        self.rigid_leg_body_ids = []
        self.rigid_leg_shape_ids = []
        for i, (cx, cy) in enumerate(corner_positions):
            body_id = builder.add_body(
                xform=wp.transform(wp.vec3(float(cx), float(cy), rigid_leg_center_z), wp.quat_identity()),
                mass=rigid_leg_mass,
                com=rigid_leg_mesh.com,
                I_m=wp.mat33(rigid_leg_I_np),
                key=f"rigid_leg_{i}",
            )
            shape_id = builder.add_shape_mesh(
                body=body_id,
                mesh=rigid_leg_mesh,
                cfg=rigid_leg_shape_cfg,
            )
            self.rigid_leg_body_ids.append(body_id)
            self.rigid_leg_shape_ids.append(shape_id)

        # --- Rigid plate (SurfaceBox) ---
        print(
            f"\n📦 [2/4] Creating rigid plate (SurfaceBox) {plate_sub_x}x{plate_sub_y}x1 "
            f"(matches leg vertex spacing ~{w/sx:.3f}m)...",
            flush=True,
        )
        surface_box = SurfaceBox(
            size=plate_size,
            subdivisions=plate_subdivisions,
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
        scale = plate_mass / rigid_mesh.mass if rigid_mesh.mass > 0 else 1.0
        I_np = np.array(rigid_mesh.I) * scale
        self.plate_body_id = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, plate_center_z), wp.quat_identity()),
            mass=plate_mass,
            com=rigid_mesh.com,
            I_m=wp.mat33(I_np),
        )
        joint_id = builder.add_joint_free(self.plate_body_id)
        builder.add_articulation([joint_id], key="plate")
        rigid_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=5e5, kd=100.0, kf=1e4, mu=0.5)
        rigid_shape_cfg.has_particle_collision = False
        builder.add_shape_mesh(
            body=self.plate_body_id,
            mesh=rigid_mesh,
            cfg=rigid_shape_cfg,
        )
        self._plate_vertices = np.array(rigid_vertices)

        # --- Four soft legs (TetraBox) at corners, stacked on rigid legs ---
        print("\n📦 [3/4] Creating 4 inflatable legs (TetraBox)...", flush=True)
        tetra_box = TetraBox(
            size=self.leg_size,
            subdivisions=self.subdivisions,
            verbose=False,
        )
        tetra_data = tetra_box.get_mesh_data()
        tetra_vertices = tetra_data["vertices"]
        tetra_indices = tetra_data["indices"]
        tetrahedra = tetra_data["tetrahedra"]

        self.soft_start_particles = []
        self.soft_particle_counts = []
        self.soft_start_tets = []
        self.soft_tet_counts = []
        self.soft_start_springs = []
        self.soft_spring_counts = []

        for i, (cx, cy) in enumerate(corner_positions):
            leg_center = wp.vec3(float(cx), float(cy), soft_leg_center_z)
            start_particle = builder.particle_count
            start_tet = builder.tet_count
            start_spring = builder.spring_count
            self.soft_start_particles.append(start_particle)
            self.soft_start_tets.append(start_tet)
            self.soft_start_springs.append(start_spring)
            builder.add_soft_mesh(
                pos=leg_center,
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
            self.soft_particle_counts.append(builder.particle_count - start_particle)
            self.soft_tet_counts.append(builder.tet_count - start_tet)
            self.soft_spring_counts.append(builder.spring_count - start_spring)

        # --- Rigid marble (sphere) on the table: rolls with friction ---
        plate_top_z = plate_center_z + self.plate_height / 2.0
        marble_z = plate_top_z + marble_radius
        # Solid sphere inertia: I = (2/5)*m*r^2 along each axis
        I_sph = (2.0 / 5.0) * marble_mass * (marble_radius ** 2)
        marble_I = wp.mat33(
            (I_sph, 0.0, 0.0),
            (0.0, I_sph, 0.0),
            (0.0, 0.0, I_sph),
        )
        marble_cfg = newton.ModelBuilder.ShapeConfig(ke=5e5, kd=100.0, kf=1e4, mu=0.6)
        self.marble_body_id = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, marble_z), wp.quat_identity()),
            mass=marble_mass,
            com=wp.vec3(0.0, 0.0, 0.0),
            I_m=marble_I,
            key="marble",
        )
        builder.add_shape_sphere(
            body=self.marble_body_id,
            radius=marble_radius,
            cfg=marble_cfg,
        )
        print(f"\n📦 Marble: radius={marble_radius}m, mass={marble_mass}kg on table (friction mu=0.6)", flush=True)

        builder.end_world()
        print("\n📦 [4/4] Glue: each soft box has glue on BOTTOM (→rigid leg) and TOP (→plate)", flush=True)

        self.model = builder.finalize()
        self.original_tet_poses = wp.clone(self.model.tet_poses)
        self.original_spring_rest_length = wp.clone(self.model.spring_rest_length)
        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 5.0e5
        self.model.soft_contact_kd = 2000.0
        self.model.soft_contact_kf = 5.0e5
        self.model.soft_contact_mu = 0.9
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, particle_radius),
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

        # Proximity glue per leg: soft leg i bottom particles → rigid leg i top vertices
        particle_q_np = np.array(self.state_0.particle_q.numpy(), dtype=np.float64)
        if particle_q_np.ndim == 1:
            particle_q_np = particle_q_np.reshape(-1, 3)
        body_q_np = self.state_0.body_q.numpy()
        self.glue_by_leg = []
        total_glue = 0
        for leg_i in range(4):
            body_id = self.rigid_leg_body_ids[leg_i]
            start_p = self.soft_start_particles[leg_i]
            end_p = start_p + self.soft_particle_counts[leg_i]
            row = body_q_np[body_id]
            p_b = np.array([row[0], row[1], row[2]], dtype=np.float64)
            q_b = wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
            particle_indices, anchors_local_np, min_dist = build_proximity_glue_pairs(
                particle_q_np,
                self._rigid_leg_vertices,
                p_b,
                q_b,
                start_p,
                end_p,
                self.glue_epsilon,
            )
            rest_lengths = []
            for j, (pidx, anchor_local) in enumerate(zip(particle_indices, anchors_local_np)):
                p_soft = particle_q_np[pidx]
                anchor_world = p_b + np.array(
                    wp.quat_rotate(q_b, wp.vec3(float(anchor_local[0]), float(anchor_local[1]), float(anchor_local[2])))
                )
                rest_lengths.append(float(np.linalg.norm(p_soft - anchor_world)))
            n_leg_glue = len(particle_indices)
            total_glue += n_leg_glue
            self.glue_by_leg.append({
                "body_id": body_id,
                "particle_indices": particle_indices,
                "anchors_local_np": anchors_local_np,
                "rest_lengths": rest_lengths,
                "particle_indices_wp": wp.array(particle_indices, dtype=wp.int32, device=self.model.device),
                "anchors_local_wp": wp.array(anchors_local_np, dtype=wp.vec3, device=self.model.device),
                "rest_lengths_wp": wp.array(rest_lengths, dtype=wp.float32, device=self.model.device),
            })
            if n_leg_glue > 0:
                print(
                    f"   Leg {leg_i + 1}: {n_leg_glue} glue springs (epsilon={self.glue_epsilon}m)",
                    flush=True,
                )
            else:
                min_str = f" (closest: {min_dist:.4f}m)" if min_dist is not None else ""
                print(
                    f"   WARNING Leg {leg_i + 1}: no glue pairs.{min_str}",
                    flush=True,
                )
        print(f"   Total: {total_glue} glue springs (bottom of each soft leg → its rigid leg)", flush=True)

        # Plate glue: soft leg TOP particles → plate bottom (each soft box glued on top to plate)
        row = body_q_np[self.plate_body_id]
        plate_p_b = np.array([row[0], row[1], row[2]], dtype=np.float64)
        plate_q_b = wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
        plate_particle_indices, plate_anchors_local_np, plate_min_dist = build_proximity_glue_pairs(
            particle_q_np,
            self._plate_vertices,
            plate_p_b,
            plate_q_b,
            0,
            self.model.particle_count,
            self.glue_epsilon,
        )
        plate_rest_lengths = []
        for j, (pidx, anchor_local) in enumerate(zip(plate_particle_indices, plate_anchors_local_np)):
            p_soft = particle_q_np[pidx]
            anchor_world = plate_p_b + np.array(
                wp.quat_rotate(plate_q_b, wp.vec3(float(anchor_local[0]), float(anchor_local[1]), float(anchor_local[2])))
            )
            plate_rest_lengths.append(float(np.linalg.norm(p_soft - anchor_world)))
        n_plate_glue = len(plate_particle_indices)
        self.plate_glue = {
            "body_id": self.plate_body_id,
            "particle_indices": plate_particle_indices,
            "anchors_local_np": plate_anchors_local_np,
            "rest_lengths": plate_rest_lengths,
            "particle_indices_wp": wp.array(plate_particle_indices, dtype=wp.int32, device=self.model.device),
            "anchors_local_wp": wp.array(plate_anchors_local_np, dtype=wp.vec3, device=self.model.device),
            "rest_lengths_wp": wp.array(plate_rest_lengths, dtype=wp.float32, device=self.model.device),
        }
        if n_plate_glue > 0:
            print(f"   Plate: {n_plate_glue} glue springs (top of each soft leg → plate)", flush=True)
        else:
            plate_min_str = f" (closest: {plate_min_dist:.4f}m)" if plate_min_dist is not None else ""
            print(f"   WARNING: no plate glue pairs.{plate_min_str}", flush=True)

        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
            self.viewer.set_world_offsets((0.0, 0.0, 0.0))
            self._update_particle_colors()
            # Rigid legs: light blue
            LIGHT_BLUE = (0.3, 0.6, 1.0)
            self.viewer.update_shape_colors({sid: LIGHT_BLUE for sid in self.rigid_leg_shape_ids})

        if self.viewer:
            if hasattr(self.viewer, "renderer") and hasattr(self.viewer.renderer, "register_key_press"):
                self.viewer.renderer.register_key_press(self._on_key_press)
            elif hasattr(self.viewer, "register_key_press"):
                self.viewer.register_key_press(self._on_key_press)

        print("\n📦 Ready! [1-4] select leg | I/K selected | A/Z all | O reset | G/F glue", flush=True)

    def _update_particle_colors(self):
        """Show particles only for the selected leg (light blue)."""
        colors_np = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        start = self.soft_start_particles[self.selected_leg]
        count = self.soft_particle_counts[self.selected_leg]
        colors_np[start : start + count] = (0.3, 0.6, 1.0)
        self.particle_colors = wp.array(colors_np, dtype=wp.vec3, device=self.model.device)
        self.particle_radii = np.zeros(self.model.particle_count, dtype=np.float32)
        self.particle_radii[start : start + count] = self.particle_radius
        self.particle_radii_array = wp.array(self.particle_radii, dtype=wp.float32, device=self.model.device)

    def _apply_leg_pressure(self, leg_idx: int, pressure: float):
        """Apply pressure to one leg by scaling its rest configuration."""
        linear_scale = float(np.cbrt(pressure))
        start_tet = self.soft_start_tets[leg_idx]
        tet_count = self.soft_tet_counts[leg_idx]
        if tet_count > 0:
            wp.launch(
                scale_tet_range_kernel,
                dim=tet_count,
                inputs=[self.original_tet_poses, linear_scale, start_tet, tet_count],
                outputs=[self.model.tet_poses],
                device=self.model.device,
            )
        start_spring = self.soft_start_springs[leg_idx]
        spring_count = self.soft_spring_counts[leg_idx]
        if spring_count > 0:
            wp.launch(
                scale_spring_range_kernel,
                dim=spring_count,
                inputs=[self.original_spring_rest_length, linear_scale, start_spring, spring_count],
                outputs=[self.model.spring_rest_length],
                device=self.model.device,
            )

    def _apply_all_pressures(self):
        """Apply current pressure to all legs."""
        for leg_idx in range(4):
            self._apply_leg_pressure(leg_idx, self.leg_pressures[leg_idx])

    def _on_key_press(self, symbol, modifiers):
        KEY_1, KEY_2, KEY_3, KEY_4 = 49, 50, 51, 52
        KEY_A, KEY_Z = 97, 122
        KEY_I, KEY_K, KEY_O = 105, 107, 111
        KEY_G, KEY_F = 103, 102
        KEY_EQUAL, KEY_MINUS = 61, 45
        # Leg selection
        if symbol == KEY_1:
            self.selected_leg = 0
            self._update_particle_colors()
            print(f"   [Selected: Leg 1 - pressure: {self.leg_pressures[0]:.2f}x]", flush=True)
        elif symbol == KEY_2:
            self.selected_leg = 1
            self._update_particle_colors()
            print(f"   [Selected: Leg 2 - pressure: {self.leg_pressures[1]:.2f}x]", flush=True)
        elif symbol == KEY_3:
            self.selected_leg = 2
            self._update_particle_colors()
            print(f"   [Selected: Leg 3 - pressure: {self.leg_pressures[2]:.2f}x]", flush=True)
        elif symbol == KEY_4:
            self.selected_leg = 3
            self._update_particle_colors()
            print(f"   [Selected: Leg 4 - pressure: {self.leg_pressures[3]:.2f}x]", flush=True)
        # Inflate/deflate selected leg
        elif symbol in (KEY_I, KEY_EQUAL):
            leg = self.selected_leg
            self.leg_pressures[leg] = min(self.max_pressure, self.leg_pressures[leg] + self.pressure_step)
            self._apply_leg_pressure(leg, self.leg_pressures[leg])
            print(f"   [Leg {leg + 1}: {self.leg_pressures[leg]:.2f}x]", flush=True)
        elif symbol in (KEY_K, KEY_MINUS):
            leg = self.selected_leg
            self.leg_pressures[leg] = max(0.5, self.leg_pressures[leg] - self.pressure_step)
            self._apply_leg_pressure(leg, self.leg_pressures[leg])
            print(f"   [Leg {leg + 1}: {self.leg_pressures[leg]:.2f}x]", flush=True)
        # All legs
        elif symbol == KEY_A:
            for i in range(4):
                self.leg_pressures[i] = min(self.max_pressure, self.leg_pressures[i] + self.pressure_step)
            self._apply_all_pressures()
            print(f"   [ALL legs: {[f'{p:.1f}' for p in self.leg_pressures]}]", flush=True)
        elif symbol == KEY_Z:
            for i in range(4):
                self.leg_pressures[i] = max(0.5, self.leg_pressures[i] - self.pressure_step)
            self._apply_all_pressures()
            print(f"   [ALL legs: {[f'{p:.1f}' for p in self.leg_pressures]}]", flush=True)
        elif symbol == KEY_O:
            self.leg_pressures = [1.0, 1.0, 1.0, 1.0]
            self._apply_all_pressures()
            print("   [Reset ALL to 1.0x]", flush=True)
        elif symbol == KEY_G:
            self.glue_ke = min(1.0e6, self.glue_ke * self.glue_ke_step)
            self.glue_kd = min(2000.0, self.glue_kd * (self.glue_ke_step ** 0.5))
            print(f"   [Glue stronger: ke={self.glue_ke:.0f} kd={self.glue_kd:.0f}]", flush=True)
        elif symbol == KEY_F:
            self.glue_ke = max(1.0e3, self.glue_ke / self.glue_ke_step)
            self.glue_kd = max(10.0, self.glue_kd / (self.glue_ke_step ** 0.5))
            print(f"   [Glue weaker: ke={self.glue_ke:.0f} kd={self.glue_kd:.0f}]", flush=True)

    def _apply_glue(self, state):
        for data in self.glue_by_leg:
            n = len(data["particle_indices"])
            if n == 0:
                continue
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
                    data["particle_indices_wp"],
                    data["body_id"],
                    data["anchors_local_wp"],
                    data["rest_lengths_wp"],
                    self.glue_ke,
                    self.glue_kd,
                    self.sim_dt,
                    self.glue_max_vel,
                    1.0 / max(0.1 * self.inflatable_mass, 0.01),
                ],
                device=self.model.device,
            )
        # Plate glued to top of soft legs
        n_plate = len(self.plate_glue["particle_indices"])
        if n_plate > 0:
            wp.launch(
                apply_glue_proximity_impulse_kernel,
                dim=n_plate,
                inputs=[
                    state.particle_q,
                    state.particle_qd,
                    self.model.particle_inv_mass,
                    state.body_q,
                    state.body_qd,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                    self.model.body_com,
                    self.plate_glue["particle_indices_wp"],
                    self.plate_glue["body_id"],
                    self.plate_glue["anchors_local_wp"],
                    self.plate_glue["rest_lengths_wp"],
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
        KEY_1, KEY_2, KEY_3, KEY_4 = 49, 50, 51, 52
        KEY_A, KEY_Z = 97, 122
        KEY_I, KEY_K, KEY_O = 105, 107, 111
        KEY_G, KEY_F = 103, 102
        KEY_EQUAL, KEY_MINUS = 61, 45
        if renderer.is_key_down(KEY_1):
            self.selected_leg = 0
            self._update_particle_colors()
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_2):
            self.selected_leg = 1
            self._update_particle_colors()
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_3):
            self.selected_leg = 2
            self._update_particle_colors()
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_4):
            self.selected_leg = 3
            self._update_particle_colors()
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_I) or renderer.is_key_down(KEY_EQUAL):
            leg = self.selected_leg
            self.leg_pressures[leg] = min(self.max_pressure, self.leg_pressures[leg] + self.pressure_step)
            self._apply_leg_pressure(leg, self.leg_pressures[leg])
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_K) or renderer.is_key_down(KEY_MINUS):
            leg = self.selected_leg
            self.leg_pressures[leg] = max(0.5, self.leg_pressures[leg] - self.pressure_step)
            self._apply_leg_pressure(leg, self.leg_pressures[leg])
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_O):
            self.leg_pressures = [1.0, 1.0, 1.0, 1.0]
            self._apply_all_pressures()
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

        particle_q_np = self.state_0.particle_q.numpy()
        body_q_np = self.state_0.body_q.numpy()
        wo = None
        bw = None
        if (
            hasattr(self.viewer, "world_offsets")
            and self.viewer.world_offsets is not None
            and self.viewer.world_offsets.shape[0] > 0
            and self.model.body_world is not None
        ):
            wo = self.viewer.world_offsets.numpy()
            bw = self.model.body_world.numpy() if hasattr(self.model.body_world, "numpy") else self.model.body_world
        for leg_i, data in enumerate(self.glue_by_leg):
            n = len(data["particle_indices"])
            if n == 0:
                continue
            body_id = data["body_id"]
            X_wb = wp.transform(*body_q_np[body_id])
            w_rigid = int(bw[body_id]) if (bw is not None and body_id < len(bw)) else 0
            starts_list = []
            ends_list = []
            for j in range(n):
                pidx = data["particle_indices"][j]
                anchor_local = data["anchors_local_np"][j]
                p_soft_raw = np.array(particle_q_np[pidx])
                anchor_world = np.array(wp.transform_point(X_wb, wp.vec3(float(anchor_local[0]), float(anchor_local[1]), float(anchor_local[2]))))
                anchor_display = anchor_world + wo[w_rigid] if wo is not None and w_rigid >= 0 and w_rigid < len(wo) else anchor_world
                starts_list.append(p_soft_raw)
                ends_list.append(anchor_display)
            starts = wp.array(starts_list, dtype=wp.vec3, device=self.model.device)
            ends = wp.array(ends_list, dtype=wp.vec3, device=self.model.device)
            colors = wp.array([wp.vec3(1.0, 0.6, 0.0)] * n, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_lines(f"/glue_springs_leg_{leg_i}", starts, ends, colors, width=0.02)

        # Plate glue: soft top → plate bottom
        n_plate = len(self.plate_glue["particle_indices"])
        if n_plate > 0:
            body_id = self.plate_glue["body_id"]
            X_wb = wp.transform(*body_q_np[body_id])
            w_plate = int(bw[body_id]) if (bw is not None and body_id < len(bw)) else 0
            starts_list = []
            ends_list = []
            for j in range(n_plate):
                pidx = self.plate_glue["particle_indices"][j]
                anchor_local = self.plate_glue["anchors_local_np"][j]
                p_soft_raw = np.array(particle_q_np[pidx])
                anchor_world = np.array(wp.transform_point(X_wb, wp.vec3(float(anchor_local[0]), float(anchor_local[1]), float(anchor_local[2]))))
                anchor_display = anchor_world + wo[w_plate] if wo is not None and w_plate >= 0 and w_plate < len(wo) else anchor_world
                starts_list.append(p_soft_raw)
                ends_list.append(anchor_display)
            starts = wp.array(starts_list, dtype=wp.vec3, device=self.model.device)
            ends = wp.array(ends_list, dtype=wp.vec3, device=self.model.device)
            colors = wp.array([wp.vec3(0.3, 0.6, 1.0)] * n_plate, dtype=wp.vec3, device=self.model.device)
            self.viewer.log_lines("/glue_springs_plate", starts, ends, colors, width=0.02)

        # Show particles only for selected leg (highlight)
        if hasattr(self, "particle_radii_array") and self.model.particle_count > 0:
            self.viewer.log_points(
                "/model/particles",
                self.state_0.particle_q,
                self.particle_radii_array,
                self.particle_colors,
                hidden=False,
            )

        self.viewer.end_frame()

    def run(self, num_frames: int = 1800):
        for frame in range(num_frames):
            self.step()
            self.render()
            if frame % 60 == 0 and frame > 0:
                print(
                    f"   Frame {frame}: selected=Leg {self.selected_leg + 1}, pressures={[f'{p:.1f}' for p in self.leg_pressures]}",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Table with 4 soft legs glued to rigid plate (SurfaceBox)"
    )
    parser.add_argument("--leg_size", type=float, nargs=3, default=[0.4, 0.4, 0.4], help="Leg box size (same as inflatable_glue)")
    parser.add_argument("--subdivisions", type=int, nargs=3, default=[5, 5, 5], help="Subdivisions (same as inflatable_glue)")
    parser.add_argument("--plate_width", type=float, default=2.0, help="Plate width (m); larger = less pressure per leg")
    parser.add_argument("--plate_height", type=float, default=0.05, help="Plate thickness (m)")
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--plate_mass", type=float, default=0.05, help="Too light (e.g. 0.01) can cause explosion from glue forces")
    parser.add_argument("--rigid_leg_mass", type=float, default=10000000.0, help="Rigid leg mass in kg (default 1e7 when not fixed)")
    parser.add_argument("--particle_radius", type=float, default=0.008, help="Same as inflatable_glue")
    parser.add_argument("--glue_epsilon", type=float, default=0.05, help="Max distance (m) for proximity glue at init")
    parser.add_argument("--glue_ke", type=float, default=3.0e4, help="Softer than 5e4 reduces explosion risk")
    parser.add_argument("--glue_kd", type=float, default=300.0)
    parser.add_argument("--glue_max_vel", type=float, default=1.0, help="Cap velocity change per substep")
    parser.add_argument("--k_mu", type=float, default=1.0e5)
    parser.add_argument("--k_lambda", type=float, default=1.0e5)
    parser.add_argument("--k_damp", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max_pressure", type=float, default=5.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--xpbd_iterations", type=int, default=12)
    parser.add_argument("--marble_radius", type=float, default=0.03, help="Marble (sphere) radius in m")
    parser.add_argument("--marble_mass", type=float, default=0.05, help="Marble mass in kg")
    parser.add_argument("--num_frames", type=int, default=1800)
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
                print(f"Could not create OpenGL viewer: {e}")
                viewer = None
        elif args.viewer == "rerun":
            try:
                viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
            except Exception as e:
                print(f"Could not create Rerun viewer: {e}")
                viewer = None
        else:
            viewer = None

        example = Example(
            viewer=viewer,
            leg_size=args.leg_size,
            subdivisions=args.subdivisions,
            plate_width=args.plate_width,
            plate_height=args.plate_height,
            mass=args.mass,
            plate_mass=args.plate_mass,
            rigid_leg_mass=args.rigid_leg_mass,
            particle_radius=args.particle_radius,
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
            marble_radius=args.marble_radius,
            marble_mass=args.marble_mass,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
