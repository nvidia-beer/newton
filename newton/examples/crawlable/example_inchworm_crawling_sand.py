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
Inchworm crawling on MPM sand — built on example_inchworm_crawling.

Same inchworm (SolverCrawlable, gait, stick-slip) with optional MPM sand and two-way
coupling: sand impulses scattered to worm particles.

Friction:
- ground_friction (params JSON): stick-slip with the ground plane (z=0). Unchanged with sand.
- collider_friction (worm mesh in MPM): sand–worm friction; higher values give more
  tangential resistance from sand. So effective grip = plane stick-slip + sand reaction.

Usage:
    python -m newton.examples inchworm_crawling_sand
    python -m newton.examples inchworm_crawling_sand --no-sand   # same as inchworm_crawling
    python -m newton.examples inchworm_crawling_sand --stick-slip --params /path/to/params.json
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton._src.utils.mesh import extract_surface_from_tets

from newton.examples.crawlable import example_inchworm_crawling
from newton.examples.inflatable import mpm_soft_sand as sand_common

InchwormCrawlingExample = example_inchworm_crawling.Example
load_params = example_inchworm_crawling.load_params
save_params = example_inchworm_crawling.save_params
INCHWORM_PARAM_KEYS = example_inchworm_crawling.INCHWORM_PARAM_KEYS
DEFAULT_PARAMS_PATH = example_inchworm_crawling.DEFAULT_PARAMS_PATH
InchwormValidation = example_inchworm_crawling.InchwormValidation

# Collider index for the worm in the MPM collider list (0=ground, 1=worm)
WORM_COLLIDER_ID = 1
GROUND_Z = 0.0
SAND_BED_TOP = 0.12
# Sand only to the right of the inchworm (Y+): worm width/2 = 0.75, so sand starts at 0.85 with 0.1 gap
SAND_Y_MIN = 0.85  # right edge of sand bed (world Y); inchworm sits at y in [-width/2, width/2]
SAND_Y_MAX = 2.0
SAND_X_EXTENT = 1.0  # sand in x in [-SAND_X_EXTENT, SAND_X_EXTENT]

SAND_IMPULSE_SCALE = 0.03
PARTICLE_VELOCITY_MAX = 1.0
SAND_CONTACT_Z_THRESHOLD = SAND_BED_TOP + 0.05
SAND_SUPPORT_VERTICAL_ONLY = True

WORM_COLLIDER_PROJECTION_THRESHOLD = 0.055
WORM_COLLIDER_THICKNESS = 0.035
PROJECT_OUTSIDE_ITERATIONS = 3

# Worm–sand friction (MPM collider). Higher = more grip from sand.
WORM_SAND_FRICTION = 0.5
GROUND_SAND_FRICTION = 0.5

DEBUG_SAND = False
DEBUG_PRINT_EVERY_N_FRAMES = 10


class Example(InchwormCrawlingExample):
    """Inchworm crawling with optional MPM sand and two-way coupling."""

    def __init__(self, viewer, sand: bool = True, **kwargs):
        if sand:
            kwargs.setdefault("substeps", 8)
        super().__init__(viewer, **kwargs)
        self.sand_enabled = sand
        if self.sand_enabled:
            # Sand step includes numpy + MPM; do not use base class graph.
            self.graph = None
        if not self.sand_enabled:
            if self.viewer:
                print("Inchworm crawling (no sand). [I]/[K] pressure, --normal/--stick-slip.", flush=True)
            return

        tet_indices = self.model.tet_indices.numpy()
        tet_flat = tet_indices.ravel() if tet_indices.ndim == 2 else tet_indices
        surface_tris = extract_surface_from_tets(tet_flat)
        surface_tris_flat = surface_tris.ravel().astype(np.int32)

        sand_builder = newton.ModelBuilder()
        voxel_size = 0.045
        # Sand only to the right (Y+) of the inchworm so there is no initial collision
        bed_lo = np.array([-SAND_X_EXTENT, SAND_Y_MIN, 0.0])
        bed_hi = np.array([SAND_X_EXTENT, SAND_Y_MAX, SAND_BED_TOP])
        sand_common.emit_sand(sand_builder, voxel_size, SAND_BED_TOP, bed_lo=bed_lo, bed_hi=bed_hi)
        self.sand_model = sand_builder.finalize()
        self.sand_model.particle_mu = 0.48
        self.sand_model.particle_ke = 1.0e15
        self.sand_state_0 = self.sand_model.state()

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = 1.0e-6
        mpm_options.grid_type = "fixed"
        mpm_options.grid_padding = 50
        mpm_options.max_active_cell_count = 1 << 15
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0

        self.mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_options)
        sand_device = self.sand_model.device
        # Ground mesh large enough to cover sand region (y up to SAND_Y_MAX = 2)
        ground_verts, ground_indices = newton.utils.create_plane_mesh(4.0, 4.0)
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
        self.mpm_model.setup_collider(
            collider_meshes=[self._ground_mesh, worm_mesh],
            collider_body_ids=[None, None],
            collider_friction=[GROUND_SAND_FRICTION, WORM_SAND_FRICTION],
            collider_thicknesses=[None, WORM_COLLIDER_THICKNESS],
            collider_projection_threshold=[None, WORM_COLLIDER_PROJECTION_THRESHOLD],
            model=self.sand_model,
        )
        self.mpm_solver = SolverImplicitMPM(self.mpm_model, mpm_options)
        self.mpm_solver.enrich_state(self.sand_state_0)

        max_collider_nodes = 1 << 18
        self._collider_impulses = wp.zeros(max_collider_nodes, dtype=wp.vec3, device=self.model.device)
        self._collider_impulse_pos = wp.zeros(max_collider_nodes, dtype=wp.vec3, device=self.model.device)
        self._collider_ids = wp.full(max_collider_nodes, -1, dtype=int, device=self.model.device)
        self._particle_impulse = wp.zeros(
            self.model.particle_count, dtype=wp.vec3, device=self.model.device
        )
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
            self.viewer.show_particles = True
            if isinstance(self.viewer, newton.viewer.ViewerGL):
                self.viewer.register_ui_callback(self._render_ui, position="side")
        print("Inchworm crawling on MPM sand. [I]/[K] pressure, --normal/--stick-slip.", flush=True)

    def _update_worm_collider_mesh(self):
        if self.sand_enabled:
            wp.copy(self._worm_mesh_points, self.state_0.particle_q)

    def _collect_collider_impulses(self):
        if not self.sand_enabled:
            return
        self._collider_impulses.zero_()
        self._collider_impulse_pos.zero_()
        self._collider_ids.fill_(-1)
        imp, pos, cid = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
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
                    f"imp_mag min={imp_mag.min():.2e} max={imp_mag.max():.2e}",
                    flush=True,
                )
        self._collider_count = n_copy

    def _apply_sand_impulses_to_worm(self):
        if not self.sand_enabled:
            return
        if self._collider_count <= 0:
            return
        particle_z = self.state_0.particle_q.numpy()[:, 2]
        worm_min_z = float(np.min(particle_z))
        worm_center_z = float(np.mean(particle_z))
        if np.any(np.isnan(particle_z)) or worm_min_z < -1.0 or worm_min_z > 100.0:
            return
        if worm_min_z >= SAND_CONTACT_Z_THRESHOLD:
            return
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

    def step(self):
        self._check_keys()
        if self.gait_enabled:
            self._update_gait_pressure()
        if self.sand_enabled:
            self._apply_sand_impulses_to_worm()
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(state=self.state_0)
            if self.use_crawlable_stick_slip:
                self.solver.set_crawl_time(self.sim_time)
            self.solver.step(
                state_in=self.state_0,
                state_out=self.state_1,
                control=self.control,
                contacts=self.contacts,
                dt=self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
            wp.launch(
                sand_common.clamp_soft_particles_above_ground,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_q, self.state_0.particle_qd, GROUND_Z],
                device=self.model.device,
            )
            if self.sand_enabled:
                self._update_worm_collider_mesh()
                worm_mesh = wp.Mesh(self._worm_mesh_points, self._worm_mesh_indices)
                self.mpm_model.setup_collider(
                    collider_meshes=[self._ground_mesh, worm_mesh],
                    collider_body_ids=[None, None],
                    collider_friction=[GROUND_SAND_FRICTION, WORM_SAND_FRICTION],
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
                for _ in range(PROJECT_OUTSIDE_ITERATIONS):
                    self.mpm_solver.project_outside(
                        self.sand_state_0, self.sand_state_0, self.sim_dt
                    )
                self._collect_collider_impulses()
        if self.sim_time <= self.settle_seconds and self.settle_seconds > 0:
            n = self.model.particle_count
            self.state_0.particle_qd.assign(
                wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3, device=self.model.device)
            )
        if self.sand_enabled:
            self._step_count += 1

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
            imp, pos, _ = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inchworm crawling on MPM sand. Same params as inchworm_crawling; --no-sand to disable sand."
    )
    parser.add_argument("--params", type=str, default=DEFAULT_PARAMS_PATH, metavar="PATH")
    parser.add_argument("--save_params", type=str, default=None, metavar="PATH")
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--stick-slip", action="store_true", dest="stick_slip")
    parser.add_argument("--csv_log_dir", type=str, default=None, metavar="DIR")
    parser.add_argument("--csv_log_interval", type=int, default=None, metavar="N")
    parser.add_argument("--no-sand", action="store_true", dest="no_sand", help="Disable sand (same as inchworm_crawling).")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.params):
        print(f"Params file not found: {args.params}", file=sys.stderr)
        sys.exit(1)
    loaded = load_params(args.params)
    print(f"Loaded params from {args.params}", flush=True)

    if "gait_pressure_min" in loaded and "gait_pressure_max" in loaded:
        mn = float(loaded["gait_pressure_min"])
        mx = float(loaded["gait_pressure_max"])
        loaded["gait_baseline"] = (mn + mx) * 0.5
        loaded["gait_amplitude"] = (mx - mn) * 0.5

    if args.stick_slip:
        loaded["use_crawlable_stick_slip"] = True
        print("Override: use_crawlable_stick_slip = True (--stick-slip)", flush=True)
    elif args.normal:
        loaded["use_crawlable_stick_slip"] = False
        print("Override: use_crawlable_stick_slip = False (--normal)", flush=True)
    if args.csv_log_dir is not None:
        loaded["csv_log_dir"] = args.csv_log_dir
    if args.csv_log_interval is not None:
        loaded["csv_log_interval"] = args.csv_log_interval

    chamber_stiffness_scale = loaded.get("chamber_stiffness_scale")
    if chamber_stiffness_scale is not None and isinstance(chamber_stiffness_scale, list):
        chamber_stiffness_scale = [float(x) for x in chamber_stiffness_scale]
    active = loaded.get("chamber_active_inflation")
    if isinstance(active, list):
        chamber_inflation_disabled = [i for i, b in enumerate(active) if b != 1]
    else:
        chamber_inflation_disabled = [0, 2]

    wp.init()
    with wp.ScopedDevice(args.device):
        if args.headless:
            viewer = None
        else:
            try:
                viewer = newton.viewer.ViewerGL(width=1920, height=1080)
            except Exception as e:
                print(f"OpenGL viewer failed: {e}")
                try:
                    viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
                except Exception as e2:
                    print(f"Rerun viewer failed: {e2}")
                    viewer = None
        nch = loaded["num_chambers_x"] * loaded["num_chambers_y"] * loaded["num_chambers_z"]
        # Paper (Gamus et al.) three-link model: joints at 1/(2+β) and (1+β)/(2+β) along crawl axis.
        paper_beta = loaded.get("paper_beta", 2.0)
        example = Example(
            viewer=viewer,
            sand=not args.no_sand,
            length=loaded["length"],
            width=loaded["width"],
            height=loaded["height"],
            subdivisions_x=loaded["subdivisions_x"],
            subdivisions_y=loaded["subdivisions_y"],
            subdivisions_z=loaded["subdivisions_z"],
            num_chambers_x=loaded["num_chambers_x"],
            num_chambers_y=loaded["num_chambers_y"],
            num_chambers_z=loaded["num_chambers_z"],
            initial_height=loaded["initial_height"],
            total_mass=loaded["total_mass"],
            k_mu=loaded["k_mu"],
            k_lambda=loaded["k_lambda"],
            k_damp=loaded["k_damp"],
            spring_ke=loaded["spring_ke"],
            spring_kd=loaded["spring_kd"],
            gravity=loaded["gravity"],
            max_pressure=loaded["max_pressure"],
            substeps=loaded["substeps"],
            anisotropy_x=loaded["anisotropy_x"],
            anisotropy_y=loaded["anisotropy_y"],
            anisotropy_z=loaded["anisotropy_z"],
            torque_stiffness=loaded["torque_stiffness"],
            torque_damping=loaded["torque_damping"],
            chamber_stiffness_scale=chamber_stiffness_scale,
            chamber_inflation_disabled=chamber_inflation_disabled,
            ground_friction=loaded["ground_friction"],
            contact_offset=loaded["contact_offset"],
            contact_iterations=loaded["contact_iterations"],
            ground_ke=loaded["ground_ke"],
            particle_radius=loaded.get("particle_radius"),
            gait_enabled=loaded["gait_enabled"],
            gait_freq=loaded["gait_freq"],
            gait_amplitude=loaded["gait_amplitude"],
            gait_phase=loaded["gait_phase"],
            gait_baseline=loaded["gait_baseline"],
            settle_seconds=loaded["settle_seconds"],
            start_at_ground_level=loaded["start_at_ground_level"],
            use_crawlable_stick_slip=loaded["use_crawlable_stick_slip"],
            stick_slip_scale=loaded["stick_slip_scale"],
            stick_slip_amplitude=loaded["stick_slip_amplitude"],
            crawl_direction=loaded["crawl_direction"],
            paper_beta=paper_beta,
        )
        csv_path = InchwormValidation.csv_log_path(loaded.get("csv_log"), loaded.get("csv_log_dir"))
        example.run(
            num_frames=loaded["num_frames"],
            validate_contact=loaded["validate_contact"],
            stop_on_lost_contact=loaded["stop_on_lost_contact"],
            csv_log_path=csv_path,
            csv_log_interval=loaded["csv_log_interval"],
        )

        if args.save_params:
            effective = {k: loaded[k] for k in INCHWORM_PARAM_KEYS if k in loaded}
            effective["chamber_stiffness_scale"] = chamber_stiffness_scale
            effective["chamber_active_inflation"] = [1 if i not in chamber_inflation_disabled else 0 for i in range(nch)]
            save_params(args.save_params, effective, comment="Inchworm crawling sand")
            print(f"Saved params to {args.save_params}", flush=True)


if __name__ == "__main__":
    main()
