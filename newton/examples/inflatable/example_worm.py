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
Worm example – one axis stays straight (X, Y, or Z).

Uses SolverInflatable with torque (spring_rest_direction).
- Choose a single stiff axis (e.g. x): all springs aligned with that axis get torque
  on the whole object (all chambers), so that axis resists bending.
- Other axes can bend; inflation drives the motion.

[I]/[K] inflate/deflate, [C] cycle chamber.
"""

import argparse
import warp as wp
import numpy as np

import newton
from newton.solvers import SolverInflatable, TetraBox


def _chamber_index(ix: int, iy: int, iz: int, nx: int, ny: int, nz: int, disabled: set) -> int:
    """Row-major chamber index. Returns -1 if chamber is in disabled (non-inflatable), else ch."""
    ch = ix * (ny * nz) + iy * nz + iz
    return -1 if ch in disabled else ch


class Example:
    """Worm with torque: one axis (X, Y, or Z) stays straight; torque on whole object."""

    def __init__(
        self,
        viewer,
        length: float = 1.0,
        width: float = 2.0,
        height: float = 0.1,
        subdivisions_x: int = 10,
        subdivisions_y: int = 30,
        subdivisions_z: int = 4,
        num_chambers_x: int = 1,
        num_chambers_y: int = 2,
        num_chambers_z: int = 2,
        initial_height: float = 0.3,
        mass: float = 1.0,
        k_mu: float = 1.0e5,
        k_lambda: float = 1.0e5,
        k_damp: float = 1.0,
        spring_ke: float = 5.0e4,
        spring_kd: float = 1.0,
        gravity: float = 9.81,
        max_pressure: float = 5.0,
        substeps: int = 5,
        anisotropy_x: float = 1.2,
        anisotropy_y: float = 1.2,
        anisotropy_z: float = 1.2,
        torque_stiffness: float = 100.0,
        torque_damping: float = 2.0,
        stiff_axes: tuple[str, ...] = ("x",),  # single axis that stays straight: "x", "y", or "z"
        torque_display_axis: str = "all",  # which torque springs to draw: "all" | "x" | "y" | "z"
        chamber_stiffness_scale: list[float] | None = None,
        chamber_inflation_disabled: list[int] | None = None,
        ground_friction: float = 0.8,  # from example 12 (disabled_chambers)
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.subdivisions = (subdivisions_x, subdivisions_y, subdivisions_z)
        self.num_chambers_x = max(1, int(num_chambers_x))
        self.num_chambers_y = max(1, int(num_chambers_y))
        self.num_chambers_z = max(1, int(num_chambers_z))
        self.total_chambers = self.num_chambers_x * self.num_chambers_y * self.num_chambers_z
        self.chamber_stiffness_scale = chamber_stiffness_scale
        _disabled = chamber_inflation_disabled if chamber_inflation_disabled is not None else [0, 2]
        self.chamber_inflation_disabled = set(int(x) for x in _disabled)
        self.inflatable_chambers = sorted(
            c for c in range(self.total_chambers) if c not in self.chamber_inflation_disabled
        )
        self.initial_height = initial_height
        self.mass = mass
        self.max_pressure = max_pressure
        self.anisotropy_x = float(anisotropy_x)
        self.anisotropy_y = float(anisotropy_y)
        self.anisotropy_z = float(anisotropy_z)
        self.viewer = viewer
        # Single stiff axis only: X, Y, or Z. Springs along that axis get torque (stay straight) on the whole object.
        ax_in = stiff_axes if isinstance(stiff_axes, (list, tuple)) else [stiff_axes]
        ax_in = [str(a).strip().lower() for a in ax_in if str(a).strip()]
        if len(ax_in) != 1 or ax_in[0] not in ("x", "y", "z"):
            raise ValueError("stiff_axes must be exactly one of 'x', 'y', 'z' (e.g. stiff_axes='x' = X stays straight)")
        self.stiff_axes = (ax_in[0],)
        disp = (torque_display_axis or "all").strip().lower()
        self.torque_display_axis = disp if disp in ("all", "x", "y", "z") else "all"

        print(f"\n🪱 Torque worm: {self.stiff_axes[0].upper()} stays straight. Ground friction μ = {ground_friction}", flush=True)
        box = TetraBox(
            size=(self.length, self.width, self.height),
            subdivisions=self.subdivisions,
            verbose=False,
        )
        mesh_data = box.get_mesh_data()
        vertices = mesh_data["vertices"]
        indices = mesh_data["indices"]
        tetrahedra = mesh_data["tetrahedra"]

        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=5e5, kd=1e3, kf=1e4, mu=ground_friction),
        )
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, initial_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            scale=1.0,
            density=mass,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )

        added_springs = set()
        spring_chamber_list = []
        spring_pairs_local = []  # (i_local, j_local) in same order as builder springs
        for t in range(len(tetrahedra)):
            tet_indices = [indices[t * 4 + k] for k in range(4)]
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for ei, ej in edges:
                i_local, j_local = tet_indices[ei], tet_indices[ej]
                if i_local > j_local:
                    i_local, j_local = j_local, i_local
                spring_key = (i_local, j_local)
                if spring_key not in added_springs:
                    added_springs.add(spring_key)
                    spring_pairs_local.append((i_local, j_local))
                    p0, p1 = vertices[i_local], vertices[j_local]
                    mid_x = (float(p0[0]) + float(p1[0])) * 0.5
                    mid_y = (float(p0[1]) + float(p1[1])) * 0.5
                    mid_z = (float(p0[2]) + float(p1[2])) * 0.5
                    norm_x = max(0.0, min(1.0, (mid_x + self.length / 2.0) / self.length))
                    norm_y = max(0.0, min(1.0, (mid_y + self.width / 2.0) / self.width))
                    norm_z = max(0.0, min(1.0, (mid_z + self.height / 2.0) / self.height))
                    ix = min(int(norm_x * self.num_chambers_x), self.num_chambers_x - 1)
                    iy = min(int(norm_y * self.num_chambers_y), self.num_chambers_y - 1)
                    iz = min(int(norm_z * self.num_chambers_z), self.num_chambers_z - 1)
                    spring_chamber_list.append(
                        _chamber_index(ix, iy, iz, self.num_chambers_x, self.num_chambers_y, self.num_chambers_z, self.chamber_inflation_disabled)
                    )

        # Torque on all springs aligned with the single stiff axis (whole object, all chambers).
        axis_sets = {
            "x": set((min(i, j), max(i, j)) for i, j in box.get_axis_aligned_springs("x")),
            "y": set((min(i, j), max(i, j)) for i, j in box.get_axis_aligned_springs("y")),
            "z": set((min(i, j), max(i, j)) for i, j in box.get_axis_aligned_springs("z")),
        }
        spring_rest_direction = np.zeros((len(spring_pairs_local), 3), dtype=np.float32)
        stiff_ax = self.stiff_axes[0]
        for k, (a, b) in enumerate(spring_pairs_local):
            key = (min(a, b), max(a, b))
            if key in axis_sets[stiff_ax]:
                rest_vec = vertices[b] - vertices[a]
                L = float(np.linalg.norm(rest_vec))
                if L > 1e-9:
                    spring_rest_direction[k] = rest_vec / L
        torque_spring_indices = [
            k for k in range(len(spring_pairs_local))
            if np.linalg.norm(spring_rest_direction[k]) > 0.5
        ]
        # Which axis each torque spring is aligned with (for display filter: only X / only Y / only Z)
        self._torque_spring_axis_dict = {k: stiff_ax for k in torque_spring_indices}
        n_torque = len(torque_spring_indices)
        self._torque_spring_indices = torque_spring_indices

        # Tet chamber assignment
        tet_chamber_id_np = np.zeros(len(tetrahedra), dtype=np.int32)
        tet_chamber_mask_np = np.zeros(len(tetrahedra), dtype=np.int32)
        for t in range(len(tetrahedra)):
            vidx = [indices[t * 4 + k] for k in range(4)]
            centroid = np.mean(vertices[vidx], axis=0)
            norm_x = max(0.0, min(1.0, (float(centroid[0]) + self.length / 2.0) / self.length))
            norm_y = max(0.0, min(1.0, (float(centroid[1]) + self.width / 2.0) / self.width))
            norm_z = max(0.0, min(1.0, (float(centroid[2]) + self.height / 2.0) / self.height))
            ix = min(int(norm_x * self.num_chambers_x), self.num_chambers_x - 1)
            iy = min(int(norm_y * self.num_chambers_y), self.num_chambers_y - 1)
            iz = min(int(norm_z * self.num_chambers_z), self.num_chambers_z - 1)
            ch = ix * (self.num_chambers_y * self.num_chambers_z) + iy * self.num_chambers_z + iz
            tet_chamber_id_np[t] = ch
            tet_chamber_mask_np[t] = -1 if ch in self.chamber_inflation_disabled else ch

        self.model = builder.finalize()

        # Per-chamber stiffness (stiffer backbone)
        if self.chamber_stiffness_scale is not None:
            scales = list(self.chamber_stiffness_scale)
        elif self.chamber_inflation_disabled:
            stiff_scale = 4.0
            scales = [stiff_scale if c in self.chamber_inflation_disabled else 1.0 for c in range(self.total_chambers)]
        else:
            scales = None
        if scales is not None:
            while len(scales) < self.total_chambers:
                scales.append(1.0)
            scales = np.array(scales[: self.total_chambers], dtype=np.float32)
            materials_np = np.zeros((self.model.tet_count, 3), dtype=np.float32)
            for t in range(self.model.tet_count):
                c = tet_chamber_id_np[t]
                s = scales[c] if c < len(scales) else 1.0
                materials_np[t, 0] = k_mu * s
                materials_np[t, 1] = k_lambda * s
                materials_np[t, 2] = k_damp * s
            self.model.tet_materials.assign(wp.array(materials_np, dtype=wp.float32, device=self.model.device))

        self.model.gravity = wp.array([wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 2.0e5
        self.model.soft_contact_kd = 1.0e3
        self.model.soft_contact_kf = 5.0e5
        self.model.soft_contact_mu = ground_friction
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.008), dtype=wp.float32, device=self.model.device
        )

        ground_plane = (0.0, 0.0, 1.0, 0.0)
        self.solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab",
            torque_stiffness=torque_stiffness,
            torque_damping=torque_damping,
            spring_rest_direction=spring_rest_direction,
            use_constraint_contacts=True,
            contact_relaxation=0.7,
            contact_max_velocity=15.0,
            contact_max_correction=0.03,
            contact_iterations=5,
            handle_self_contact=True,  # prevent worm body from passing through itself when bending
            self_contact_radius=0.025,  # slightly larger to catch thin body folds
            self_contact_stiffness=2.0e5,  # stiffer to resist sharp bends
            ground_plane=ground_plane,
            ground_mu=ground_friction,
        )
        tet_chamber_mask = wp.array(tet_chamber_mask_np, dtype=wp.int32, device=self.model.device)
        spring_chamber_mask = wp.array(np.array(spring_chamber_list, dtype=np.int32), dtype=wp.int32, device=self.model.device)
        self.solver.set_chamber_mask(tet_chamber_mask, spring_chamber_mask=spring_chamber_mask, num_chambers=self.total_chambers)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True
            self.viewer.show_springs = True

        self.chamber_pressures = [1.0] * self.total_chambers
        self.pressure_step = 0.15
        self.active_chamber = self.inflatable_chambers[0] if self.inflatable_chambers else 0
        self._key_cooldown = 0

        if self.viewer:
            if hasattr(self.viewer, "renderer") and hasattr(self.viewer.renderer, "register_key_press"):
                self.viewer.renderer.register_key_press(self._on_key_press)
            elif hasattr(self.viewer, "register_key_press"):
                self.viewer.register_key_press(self._on_key_press)

        self._apply_pressure()
        self._print_help()

    def _apply_pressure(self):
        self.solver.anisotropy_x = self.anisotropy_x
        self.solver.anisotropy_y = self.anisotropy_y
        self.solver.anisotropy_z = self.anisotropy_z
        pressures = list(self.chamber_pressures)
        for c in self.chamber_inflation_disabled:
            if 0 <= c < len(pressures):
                pressures[c] = 1.0
        self.solver.set_chamber_pressures(pressures)

    def _set_pressure_delta(self, delta: float):
        if self.active_chamber in self.chamber_inflation_disabled:
            return
        p = self.chamber_pressures[self.active_chamber]
        self.chamber_pressures[self.active_chamber] = np.clip(p + delta, 0.5, self.max_pressure)
        self._apply_pressure()
        print(f"   [Chamber {self.active_chamber} pressure: {self.chamber_pressures[self.active_chamber]:.2f}x]", flush=True)

    def _cycle_chamber(self):
        if not self.inflatable_chambers:
            return
        try:
            idx = self.inflatable_chambers.index(self.active_chamber)
        except ValueError:
            idx = -1
        next_idx = (idx + 1) % len(self.inflatable_chambers)
        self.active_chamber = self.inflatable_chambers[next_idx]
        print(f"   [Active chamber: {self.active_chamber} / {self.total_chambers}]", flush=True)

    def _print_help(self):
        ax = self.stiff_axes[0].upper()
        print(f"\n🪱 Ready: {ax} stays straight. [I]/[K] inflate/deflate, [C] cycle chamber.", flush=True)

    def _on_key_press(self, symbol, modifiers):
        if symbol in (105, 61):
            self._set_pressure_delta(self.pressure_step)
        elif symbol in (107, 45):
            self._set_pressure_delta(-self.pressure_step)
        elif symbol == 99:
            self._cycle_chamber()

    def _check_keys(self):
        if not self.viewer or not hasattr(self.viewer, "renderer"):
            return
        r = self.viewer.renderer
        if not hasattr(r, "is_key_down"):
            return
        if getattr(self, "_key_cooldown", 0) > 0:
            self._key_cooldown -= 1
            return
        if r.is_key_down(105) or r.is_key_down(61):
            self._set_pressure_delta(self.pressure_step)
            self._key_cooldown = 10
        elif r.is_key_down(107) or r.is_key_down(45):
            self._set_pressure_delta(-self.pressure_step)
            self._key_cooldown = 10

    def step(self):
        self._check_keys()
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

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def run(self, num_frames: int = 14400):
        for frame in range(num_frames):
            self.step()
            self.render()
            if frame % 120 == 0:
                vol_ratio = self.solver.get_volume_ratio(self.state_0)
                p_str = ",".join(f"{p:.1f}" for p in self.chamber_pressures)
                print(f"   Frame {frame}: pressures=[{p_str}] vol_ratio={vol_ratio:.2f}x", flush=True)
        info = self.solver.get_inflation_info(self.state_0)
        print(f"\n🪱 Done. Final volume ratio: {info['current_ratio']:.2f}x", flush=True)


def _parse_csv(s: str | None, cast):
    return [cast(x.strip()) for x in s.split(",")] if s else None


def main():
    parser = argparse.ArgumentParser(
        description="Worm: one axis (x, y, or z) stays straight; torque on whole object."
    )
    parser.add_argument("--stiff_axes", type=str, default="x", help="Single axis that stays straight: x, y, or z (torque on whole object)")
    worm_geom = parser.add_argument_group("Worm geometry (size and subdivisions)")
    worm_geom.add_argument("--length", type=float, default=1.0, help="Worm length (X size, meters)")
    worm_geom.add_argument("--width", type=float, default=2.0, help="Worm width (Y size, meters)")
    worm_geom.add_argument("--height", type=float, default=0.1, help="Worm height (Z size, meters)")
    worm_geom.add_argument("--subdivisions_x", type=int, default=10, help="Mesh subdivisions along length (X)")
    worm_geom.add_argument("--subdivisions_y", type=int, default=30, help="Mesh subdivisions along width (Y)")
    worm_geom.add_argument("--subdivisions_z", type=int, default=4, help="Mesh subdivisions along height (Z)")
    parser.add_argument("--num_chambers_x", type=int, default=1)
    parser.add_argument("--num_chambers_y", type=int, default=2)
    parser.add_argument("--num_chambers_z", type=int, default=2)
    parser.add_argument("--initial_height", type=float, default=0.3)
    parser.add_argument("--mass", type=float, default=1.0, help="From example 12 (disabled_chambers)")
    parser.add_argument("--k_mu", type=float, default=1.0e5)
    parser.add_argument("--k_lambda", type=float, default=1.0e5)
    parser.add_argument("--k_damp", type=float, default=1.0)
    parser.add_argument("--spring_ke", type=float, default=5.0e4)
    parser.add_argument("--spring_kd", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max_pressure", type=float, default=5.0)
    parser.add_argument("--anisotropy_x", type=float, default=1.2)
    parser.add_argument("--anisotropy_y", type=float, default=1.2)
    parser.add_argument("--anisotropy_z", type=float, default=1.2)
    parser.add_argument("--torque_stiffness", type=float, default=100.0)
    parser.add_argument("--torque_damping", type=float, default=2.0)
    parser.add_argument(
        "--torque_display_axis",
        type=str,
        default="all",
        choices=("all", "x", "y", "z"),
        help="Which torque springs to draw: all, or only X-, Y-, or Z-aligned",
    )
    parser.add_argument(
        "--ground_friction",
        type=float,
        default=0.8,
        help="Ground friction (from example 12). Use e.g. 0.1 for slippery.",
    )
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=14400)
    parser.add_argument("--chamber_stiffness_scale", type=str, default=None)
    parser.add_argument("--chamber_inflation_disabled", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    chamber_stiffness_scale = _parse_csv(args.chamber_stiffness_scale, float)
    if args.chamber_inflation_disabled is not None and args.chamber_inflation_disabled.strip().lower() in ("none", "all", ""):
        chamber_inflation_disabled = []
    else:
        chamber_inflation_disabled = _parse_csv(args.chamber_inflation_disabled, int)
    stiff_axes_in = [a.strip().lower() for a in args.stiff_axes.replace(",", " ").split() if a.strip()]
    if len(stiff_axes_in) != 1 or stiff_axes_in[0] not in ("x", "y", "z"):
        parser.error("--stiff_axes must be exactly one of x, y, z (e.g. --stiff_axes x)")
    stiff_axes = (stiff_axes_in[0],)

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
        example = Example(
            viewer=viewer,
            length=args.length,
            width=args.width,
            height=args.height,
            subdivisions_x=args.subdivisions_x,
            subdivisions_y=args.subdivisions_y,
            subdivisions_z=args.subdivisions_z,
            num_chambers_x=args.num_chambers_x,
            num_chambers_y=args.num_chambers_y,
            num_chambers_z=args.num_chambers_z,
            initial_height=args.initial_height,
            mass=args.mass,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            gravity=args.gravity,
            max_pressure=args.max_pressure,
            substeps=args.substeps,
            anisotropy_x=args.anisotropy_x,
            anisotropy_y=args.anisotropy_y,
            anisotropy_z=args.anisotropy_z,
            torque_stiffness=args.torque_stiffness,
            torque_damping=args.torque_damping,
            stiff_axes=stiff_axes,
            torque_display_axis=args.torque_display_axis,
            chamber_stiffness_scale=chamber_stiffness_scale,
            chamber_inflation_disabled=chamber_inflation_disabled,
            ground_friction=args.ground_friction,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
