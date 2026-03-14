# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
Inflatable worm on MPM sand — built on example_worm.

Uses the same worm as example_worm (SolverInflatable, torque, chambers), then adds
optional MPM sand and two-way coupling: sand impulses scattered to worm particles.

Usage:
    python -m newton.examples mpm_worm_sand
    python -m newton.examples mpm_worm_sand --no-sand   # worm only (same as example_worm)
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM
from newton._src.utils.mesh import extract_surface_from_tets

from newton.examples.inflatable.example_worm import Example as WormExample
from newton.examples.inflatable import mpm_soft_sand as sand_common


# Collider index for the worm in the MPM collider list (0=ground, 1=worm)
WORM_COLLIDER_ID = 1
# Ground plane: no soft-body particle may go below this (inflation object pushes back)
GROUND_Z = 0.0

# Sand bed extent (must match emit_sand). Kept lower so the worm sits above the sand for visualization (like MPM examples).
SAND_BED_TOP = 0.12
WORM_HEIGHT_ABOVE_SAND = 0.2

# Worm is much stiffer than sand (sand is light): minimal reaction so sand barely affects the worm.
SAND_IMPULSE_SCALE = 0.03
PARTICLE_VELOCITY_MAX = 1.0
SAND_CONTACT_Z_THRESHOLD = SAND_BED_TOP + 0.05
# If True, only the vertical (upward) component of sand impulse is applied — support without deforming.
SAND_SUPPORT_VERTICAL_ONLY = True

# Worm → sand: strong collider so sand cannot enter the deformable (pushed out); sand→worm reaction kept gentle via SAND_IMPULSE_SCALE
WORM_COLLIDER_PROJECTION_THRESHOLD = 0.055
WORM_COLLIDER_THICKNESS = 0.035
# Multiple project_outside passes so no sand particle remains inside the inflatable
PROJECT_OUTSIDE_ITERATIONS = 3

# Set True to print debug info (frame, collider count, impulses, velocities)
DEBUG_SAND = True
DEBUG_PRINT_EVERY_N_FRAMES = 10


class Example(WormExample):
    """Worm from example_worm, with optional MPM sand and two-way coupling."""

    def __init__(self, viewer, sand: bool = True, **kwargs):
        if sand:
            kwargs.setdefault("initial_height", SAND_BED_TOP + WORM_HEIGHT_ABOVE_SAND)
            kwargs.setdefault("substeps", 8)  # more substeps for stable first drop (like inflatable_box_sand)
        super().__init__(viewer, **kwargs)
        self.sand_enabled = sand
        if not self.sand_enabled:
            if self.viewer:
                print("Inflatable worm (no sand). [I]/[K] inflate/deflate, [C] cycle chamber.", flush=True)
            return

        # Surface mesh for worm collider (from base model)
        tet_indices = self.model.tet_indices.numpy()
        tet_flat = tet_indices.ravel() if tet_indices.ndim == 2 else tet_indices
        surface_tris = extract_surface_from_tets(tet_flat)
        surface_tris_flat = surface_tris.ravel().astype(np.int32)

        # Sand model (MPM)
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        voxel_size = 0.045
        self._emit_sand(sand_builder, voxel_size)
        self.sand_model = sand_builder.finalize()
        self.sand_model.particle_mu = 0.48
        self.sand_model.particle_ke = 1.0e15
        self.sand_state_0 = self.sand_model.state()

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = 1.0e-6
        mpm_options.grid_type = "fixed"
        mpm_options.grid_padding = 50
        mpm_options.max_active_cell_count = 1 << 15
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0

        self.mpm_solver = SolverImplicitMPM(self.sand_model, mpm_options)
        # Collider meshes must live on the sand (MPM) device so collision kernels can access them.
        sand_device = self.sand_model.device
        ground_verts, ground_indices = newton.utils.create_plane_mesh(2.0, 2.0)
        ground_verts_xyz = np.asarray(ground_verts[:, :3], dtype=np.float32)
        self._ground_mesh = wp.Mesh(
            wp.array(ground_verts_xyz, dtype=wp.vec3, device=sand_device),
            wp.array(ground_indices, dtype=wp.int32, device=sand_device),
        )
        self._worm_mesh_points = wp.zeros(
            self.model.particle_count, dtype=wp.vec3, device=sand_device
        )
        self._worm_mesh_indices = wp.array(
            surface_tris_flat, dtype=wp.int32, device=sand_device
        )
        self._update_worm_collider_mesh()
        worm_mesh = wp.Mesh(self._worm_mesh_points, self._worm_mesh_indices)
        self.mpm_solver.setup_collider(
            collider_meshes=[self._ground_mesh, worm_mesh],
            collider_body_ids=[None, None],
            collider_friction=[0.5, 0.5],
            collider_thicknesses=[None, WORM_COLLIDER_THICKNESS],
            collider_projection_threshold=[None, WORM_COLLIDER_PROJECTION_THRESHOLD],
            model=self.sand_model,
        )

        max_collider_nodes = 1 << 18
        self._collider_impulses = wp.zeros(max_collider_nodes, dtype=wp.vec3, device=self.model.device)
        self._collider_impulse_pos = wp.zeros(max_collider_nodes, dtype=wp.vec3, device=self.model.device)
        self._collider_ids = wp.full(max_collider_nodes, -1, dtype=int, device=self.model.device)
        self._particle_impulse = wp.zeros(
            self.model.particle_count, dtype=wp.vec3, device=self.model.device
        )
        # Number of valid collider entries (0 until first MPM step; then set in _collect_collider_impulses)
        self._collider_count = 0
        self._step_count = 0

        self.particle_render_colors = wp.full(
            self.sand_model.particle_count,
            value=wp.vec3(0.76, 0.70, 0.50),
            dtype=wp.vec3,
            device=self.sand_model.device,
        )
        self.show_impulses = False

        if self.viewer:
            self.viewer.show_particles = True  # show sand like MPM examples
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self.viewer.register_ui_callback(self._render_ui, position="side")
        print("Inflatable worm on MPM sand. [I]/[K] inflate/deflate, [C] cycle chamber.", flush=True)

    def _update_worm_collider_mesh(self):
        if self.sand_enabled:
            # Copy worm particle positions into collider mesh (handles cross-device if model != sand)
            wp.copy(self._worm_mesh_points, self.state_0.particle_q)

    def _collect_collider_impulses(self):
        if not self.sand_enabled:
            return
        # Zero/fill all buffers so no slot is ever read uninitialized
        self._collider_impulses.zero_()
        self._collider_impulse_pos.zero_()
        self._collider_ids.fill_(-1)
        imp, pos, cid = self.mpm_solver._collect_collider_impulses(self.sand_state_0)
        # Only use entries that hit the worm (cid == WORM_COLLIDER_ID), like inflatable_box_sand
        imp_np = np.asarray(imp.numpy())
        pos_np = np.asarray(pos.numpy())
        cid_np = np.asarray(cid.numpy())
        worm_mask = cid_np.ravel() == WORM_COLLIDER_ID
        n_worm = int(np.sum(worm_mask))
        if imp_np.ndim == 2:
            imp_flat = imp_np.reshape(-1, 3)
            pos_flat = pos_np.reshape(-1, 3)
        else:
            imp_flat = imp_np
            pos_flat = pos_np
        n_copy = min(n_worm, int(self._collider_impulses.shape[0]))
        if n_copy > 0:
            imp_worm = imp_flat[worm_mask][:n_copy]
            pos_worm = pos_flat[worm_mask][:n_copy]
            self._collider_impulses[:n_copy].assign(wp.array(imp_worm, dtype=wp.vec3, device=self.model.device))
            self._collider_impulse_pos[:n_copy].assign(wp.array(pos_worm, dtype=wp.vec3, device=self.model.device))
            self._collider_ids[:n_copy].fill_(WORM_COLLIDER_ID)
            if DEBUG_SAND:
                imp_mag = np.linalg.norm(imp_worm, axis=1)
                print(
                    f"[sand debug] collect: n_total={cid_np.size} n_worm={n_copy} "
                    f"imp_mag min={imp_mag.min():.2e} max={imp_mag.max():.2e} mean={imp_mag.mean():.2e}",
                    flush=True,
                )
        self._collider_count = n_copy

    def _apply_sand_impulses_to_worm(self):
        if not self.sand_enabled:
            return
        # Only apply when we have valid collider data from at least one MPM step
        if self._collider_count <= 0:
            if DEBUG_SAND and self._step_count % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                particle_z = self.state_0.particle_q.numpy()[:, 2]
                print(
                    f"[sand debug] step {self._step_count} skip: _collider_count=0 "
                    f"worm_z=[{particle_z.min():.3f}, {particle_z.max():.3f}]",
                    flush=True,
                )
            return
        # Only apply when worm is actually in contact with sand (avoid pre-contact garbage)
        particle_z = self.state_0.particle_q.numpy()[:, 2]
        worm_min_z = float(np.min(particle_z))
        worm_center_z = float(np.mean(particle_z))
        if np.any(np.isnan(particle_z)) or worm_min_z < -1.0 or worm_min_z > 100.0:
            if DEBUG_SAND and self._step_count % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[sand debug] step {self._step_count} skip: state corrupted "
                    f"worm_min_z={worm_min_z} nan={np.any(np.isnan(particle_z))}",
                    flush=True,
                )
            return
        if worm_min_z >= SAND_CONTACT_Z_THRESHOLD:
            if DEBUG_SAND and self._step_count % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[sand debug] step {self._step_count} skip: no contact "
                    f"worm_min_z={worm_min_z:.3f} >= thresh={SAND_CONTACT_Z_THRESHOLD:.3f}",
                    flush=True,
                )
            return
        qd_before = self.state_0.particle_qd.numpy()
        v_before = np.linalg.norm(qd_before, axis=1)
        self._particle_impulse.zero_()
        wp.launch(
            sand_common.scatter_impulses_to_particles,
            dim=self._collider_count,
            inputs=[
                self._collider_impulse_pos,
                self._collider_impulses,
                self.state_0.particle_q,
                self._particle_impulse,
            ],
            device=self.model.device,
        )
        part_imp = self._particle_impulse.numpy()
        part_imp_mag = np.linalg.norm(part_imp, axis=1)
        n_nonzero = int(np.sum(part_imp_mag > 1e-12))
        wp.launch(
            sand_common.apply_impulse_velocity_kick,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_qd,
                self._particle_impulse,
                self.model.particle_inv_mass,
                SAND_IMPULSE_SCALE,
                PARTICLE_VELOCITY_MAX,
                1 if SAND_SUPPORT_VERTICAL_ONLY else 0,
            ],
            device=self.model.device,
        )
        qd_after = self.state_0.particle_qd.numpy()
        v_after = np.linalg.norm(qd_after, axis=1)
        if DEBUG_SAND:
            print(
                f"[sand debug] step {self._step_count} APPLIED: worm_min_z={worm_min_z:.3f} center_z={worm_center_z:.3f} "
                f"_collider_count={self._collider_count} | "
                f"particle_impulse: n_nonzero={n_nonzero} mag max={part_imp_mag.max():.2e} | "
                f"vel before: max={v_before.max():.3f} | after: max={v_after.max():.3f} mean={v_after.mean():.3f}",
                flush=True,
            )

    def _emit_sand(self, sand_builder: newton.ModelBuilder, voxel_size: float):
        sand_common.emit_sand(sand_builder, voxel_size, SAND_BED_TOP)

    def step(self):
        self._check_keys()
        if self.sand_enabled:
            self._apply_sand_impulses_to_worm()
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(state=self.state_0)
            self.solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
            # No particle may go below ground
            wp.launch(
                sand_common.clamp_soft_particles_above_ground,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_q, self.state_0.particle_qd, GROUND_Z],
                device=self.model.device,
            )
            if self.sand_enabled:
                # Update collider from current deformed shape (tetras change each frame) and step sand
                self._update_worm_collider_mesh()
                worm_mesh = wp.Mesh(self._worm_mesh_points, self._worm_mesh_indices)
                self.mpm_solver.setup_collider(
                    collider_meshes=[self._ground_mesh, worm_mesh],
                    collider_body_ids=[None, None],
                    collider_friction=[0.5, 0.5],
                    collider_thicknesses=[None, WORM_COLLIDER_THICKNESS],
                    collider_projection_threshold=[None, WORM_COLLIDER_PROJECTION_THRESHOLD],
                    model=self.sand_model,
                )
                self.mpm_solver.step(
                    self.sand_state_0,
                    self.sand_state_0,
                    contacts=None,
                    control=None,
                    dt=self.sim_dt,
                )
                # Push sand particles outside the deformable so none remain inside
                for _ in range(PROJECT_OUTSIDE_ITERATIONS):
                    self.mpm_solver._project_outside(
                        self.sand_state_0, self.sand_state_0, self.sim_dt
                    )
                self._collect_collider_impulses()
        if self.sand_enabled:
            self._step_count += 1
        if self.sand_enabled and DEBUG_SAND and self._step_count % DEBUG_PRINT_EVERY_N_FRAMES == 0 and self._step_count > 0:
            particle_z = self.state_0.particle_q.numpy()[:, 2]
            v = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1)
            print(
                f"[sand debug] after step {self._step_count}: worm_z=[{particle_z.min():.3f}, {particle_z.max():.3f}] "
                f"vel max={v.max():.3f} _collider_count={getattr(self, '_collider_count', -1)}",
                flush=True,
            )

    def render(self):
        super().render()
        if not self.viewer or not self.sand_enabled:
            return
        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )
        if self.show_impulses:
            imp, pos, _ = self.mpm_solver._collect_collider_impulses(self.sand_state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + imp,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

    def _render_ui(self, imgui):
        if self.sand_enabled:
            _c, self.show_impulses = imgui.checkbox("Show sand impulses", self.show_impulses)


if __name__ == "__main__":
    import sys

    no_sand = "--no-sand" in sys.argv
    if no_sand:
        sys.argv.remove("--no-sand")
    viewer, args = newton.examples.init()
    example = Example(viewer, sand=not no_sand)
    newton.examples.run(example, args)
