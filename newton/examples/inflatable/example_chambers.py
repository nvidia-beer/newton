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
Chambers Example - 3D N-chamber inflatable with anisotropic inflation.

This is the 3D counterpart of the 2D ChambersModel (soft_robotics/warp/models/chambers.py).
Chambers can be stacked along height (Z) or side-by-side along width (Y). For a flat
slab that curves under pressure (like a soft bending actuator), use:
- Flat rest shape: small height (Z), e.g. --length 0.4 --width 0.25 --height 0.06
- Two chambers side-by-side: --num_chambers 2 --chamber_axis y
- Inflate one chamber more than the other → the slab bends (curvature from pressure).

3D setup:
- Box size: length (X), width (Y), height (Z); Z is up
- chamber_axis "z": chambers = slices along height (stacked)
- chamber_axis "y": chambers = slices along width (left/right); differential pressure bends the slab
- Anisotropy is usually orthogonal to the chamber axis (e.g. for chamber_axis y use anisotropy_x, anisotropy_z; keep anisotropy_y=1).

Keys:
- [I] / [=]  - Inflate active chamber
- [K] / [-]  - Deflate active chamber
- [C]        - Cycle active chamber (when num_chambers > 1)

Usage:
    python -m newton.examples chambers --num_chambers 2 --chamber_axis y --height 0.06
"""

import argparse
import warp as wp
import numpy as np

import newton
from newton.solvers import SolverInflatable, TetraBox


class Example:
    """
    3D N-chamber inflatable box with anisotropic inflation.
    
    Uses a 3D tetrahedral mesh (TetraBox). Chambers are defined along the box
    height (Z). Configurable length (X), width (Y), height (Z), and number of chambers.
    Single-chamber mode focuses on anisotropic expansion (X, Y, Z scaling).
    """

    def __init__(
        self,
        viewer,
        length: float = 0.35,
        width: float = 0.35,
        height: float = 0.5,
        subdivisions_x: int = 4,
        subdivisions_y: int = 4,
        subdivisions_z: int = 5,
        num_chambers: int = 1,
        chamber_axis: str = "y",
        initial_height: float = 0.5,
        mass: float = 1.0,
        k_mu: float = 1.0e5,
        k_lambda: float = 1.0e5,
        k_damp: float = 1.0,
        spring_ke: float = 5.0e4,
        spring_kd: float = 1.0,
        gravity: float = 9.81,
        max_pressure: float = 5.0,
        substeps: int = 5,
        anisotropy_x: float = 1.0,
        anisotropy_y: float = 1.0,
        anisotropy_z: float = 1.0,
    ):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.substeps = substeps
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.subdivisions = (subdivisions_x, subdivisions_y, subdivisions_z)
        self.num_chambers = max(1, int(num_chambers))
        self.chamber_axis = (chamber_axis or "y").lower()
        if self.chamber_axis not in ("y", "z"):
            self.chamber_axis = "y"
        self.initial_height = initial_height
        self.mass = mass
        self.max_pressure = max_pressure
        self.anisotropy_x = float(anisotropy_x)
        self.anisotropy_y = float(anisotropy_y)
        self.anisotropy_z = float(anisotropy_z)
        self.viewer = viewer

        # 3D tetrahedral box: length (X), width (Y), height (Z); Z = chamber axis
        print(f"\n📦 Generating 3D tetrahedral box (chambers)...", flush=True)
        box = TetraBox(
            size=(self.length, self.width, self.height),
            subdivisions=self.subdivisions,
            verbose=True,
        )
        mesh_data = box.get_mesh_data()
        vertices = mesh_data["vertices"]
        indices = mesh_data["indices"]
        tetrahedra = mesh_data["tetrahedra"]
        print(
            f"   3D mesh: {len(vertices)} vertices, {len(tetrahedra)} tetrahedra",
            flush=True,
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=5e5,
                kd=1e3,
                kf=1e4,
                mu=0.5,
            )
        )
        start_particle = builder.particle_count
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
        spring_chamber_list = []  # chamber index per spring, same order as add_spring
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
                    p0, p1 = vertices[i_local], vertices[j_local]
                    rest_length = float(np.linalg.norm(p1 - p0))
                    builder.add_spring(
                        start_particle + i_local,
                        start_particle + j_local,
                        spring_ke,
                        spring_kd,
                        rest_length,
                    )
                    # Chamber from midpoint along chamber_axis (y or z)
                    if self.chamber_axis == "z":
                        mid = (float(p0[2]) + float(p1[2])) * 0.5
                        extent = self.height
                        mid_norm = (mid + extent / 2.0) / extent
                    else:
                        mid = (float(p0[1]) + float(p1[1])) * 0.5
                        extent = self.width
                        mid_norm = (mid + extent / 2.0) / extent
                    ch = int(mid_norm * self.num_chambers)
                    ch = max(0, min(ch, self.num_chambers - 1))
                    spring_chamber_list.append(ch)
        print(f"   Added {len(added_springs)} springs", flush=True)

        # Assign each tet to a chamber by centroid along chamber_axis (y = side-by-side, z = stacked)
        tet_chamber_mask_np = np.zeros(len(tetrahedra), dtype=np.int32)
        for t in range(len(tetrahedra)):
            vidx = [indices[t * 4 + k] for k in range(4)]
            centroid = np.mean(vertices[vidx], axis=0)
            if self.chamber_axis == "z":
                coord = float(centroid[2])
                extent = self.height
            else:
                coord = float(centroid[1])
                extent = self.width
            mid_norm = (coord + extent / 2.0) / extent
            ch = int(mid_norm * self.num_chambers)
            ch = max(0, min(ch, self.num_chambers - 1))
            tet_chamber_mask_np[t] = ch
        for c in range(self.num_chambers):
            n_tets = np.sum(tet_chamber_mask_np == c)
            print(f"   Chamber {c}: {n_tets} tetrahedra", flush=True)
        print(f"   Chamber axis: {self.chamber_axis} (y=side-by-side → bend, z=stacked)", flush=True)

        self.model = builder.finalize()
        self.model.gravity = wp.array(
            [wp.vec3(0.0, 0.0, -gravity)], dtype=wp.vec3, device=self.model.device
        )
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 500.0
        self.model.soft_contact_kf = 5.0e4
        self.model.soft_contact_mu = 0.9
        self.model.particle_ke = 1.0e5
        self.model.particle_kd = 1.0
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, 0.008),
            dtype=wp.float32,
            device=self.model.device,
        )

        self.solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab",
        )
        # Per-chamber masks: chamber_axis y = side-by-side (bend), z = stacked
        tet_chamber_mask = wp.array(
            tet_chamber_mask_np,
            dtype=wp.int32,
            device=self.model.device,
        )
        spring_chamber_mask = wp.array(
            np.array(spring_chamber_list, dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.solver.set_chamber_mask(
            tet_chamber_mask,
            spring_chamber_mask=spring_chamber_mask,
            num_chambers=self.num_chambers,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0,
        )

        if self.viewer:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True

        # Per-chamber pressures (one per chamber along Z)
        self.chamber_pressures = [1.0] * self.num_chambers
        self.current_pressure = 1.0  # for single-pressure fallback
        self.pressure_step = 0.15
        self.active_chamber = 0
        self._key_cooldown = 0

        if self.viewer:
            if hasattr(self.viewer, "renderer") and hasattr(
                self.viewer.renderer, "register_key_press"
            ):
                self.viewer.renderer.register_key_press(self._on_key_press)
            elif hasattr(self.viewer, "register_key_press"):
                self.viewer.register_key_press(self._on_key_press)

        self._apply_pressure()
        self._print_help()

    def _apply_pressure(self):
        self.solver.anisotropy_x = self.anisotropy_x
        self.solver.anisotropy_y = self.anisotropy_y
        self.solver.anisotropy_z = self.anisotropy_z
        self.solver.set_chamber_pressures(self.chamber_pressures)

    def _print_help(self):
        print(f"\n📦 3D Chambers (anisotropic inflation) ready!", flush=True)
        print(
            f"   3D box: length={self.length:.2f} width={self.width:.2f} height={self.height:.2f}m, "
            f"chambers={self.num_chambers} (axis={self.chamber_axis}), anisotropy=({self.anisotropy_x},{self.anisotropy_y},{self.anisotropy_z})",
            flush=True,
        )
        print(f"   [I] / [=]  - Inflate", flush=True)
        print(f"   [K] / [-]  - Deflate", flush=True)
        print(f"   [C]        - Cycle active chamber (when num_chambers > 1)", flush=True)

    def _on_key_press(self, symbol, modifiers):
        KEY_I = 105
        KEY_K = 107
        KEY_C = 99
        KEY_EQUAL = 61
        KEY_MINUS = 45

        if symbol in (KEY_I, KEY_EQUAL):
            p = self.chamber_pressures[self.active_chamber]
            self.chamber_pressures[self.active_chamber] = min(
                self.max_pressure, p + self.pressure_step
            )
            self._apply_pressure()
            print(
                f"   [Chamber {self.active_chamber} pressure: "
                f"{self.chamber_pressures[self.active_chamber]:.2f}x]",
                flush=True,
            )
        elif symbol in (KEY_K, KEY_MINUS):
            p = self.chamber_pressures[self.active_chamber]
            self.chamber_pressures[self.active_chamber] = max(0.5, p - self.pressure_step)
            self._apply_pressure()
            print(
                f"   [Chamber {self.active_chamber} pressure: "
                f"{self.chamber_pressures[self.active_chamber]:.2f}x]",
                flush=True,
            )
        elif symbol == KEY_C:
            self.active_chamber = (self.active_chamber + 1) % self.num_chambers
            print(f"   [Active chamber: {self.active_chamber} / {self.num_chambers}]", flush=True)

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
        KEY_I, KEY_K, KEY_C = 105, 107, 99
        KEY_EQUAL, KEY_MINUS = 61, 45
        if renderer.is_key_down(KEY_I) or renderer.is_key_down(KEY_EQUAL):
            p = self.chamber_pressures[self.active_chamber]
            self.chamber_pressures[self.active_chamber] = min(
                self.max_pressure, p + self.pressure_step
            )
            self._apply_pressure()
            print(
                f"   [Chamber {self.active_chamber} pressure: "
                f"{self.chamber_pressures[self.active_chamber]:.2f}x]",
                flush=True,
            )
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_K) or renderer.is_key_down(KEY_MINUS):
            p = self.chamber_pressures[self.active_chamber]
            self.chamber_pressures[self.active_chamber] = max(0.5, p - self.pressure_step)
            self._apply_pressure()
            print(
                f"   [Chamber {self.active_chamber} pressure: "
                f"{self.chamber_pressures[self.active_chamber]:.2f}x]",
                flush=True,
            )
            self._key_cooldown = 10
        elif renderer.is_key_down(KEY_C):
            self.active_chamber = (self.active_chamber + 1) % self.num_chambers
            print(f"   [Active chamber: {self.active_chamber}]", flush=True)
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

    def run(self, num_frames: int = 7200):
        print(f"\n📦 3D Chambers demo running...", flush=True)
        for frame in range(num_frames):
            self.step()
            self.render()
            if frame % 120 == 0:
                vol_ratio = self.solver.get_volume_ratio(self.state_0)
                p_str = ",".join(f"{p:.1f}" for p in self.chamber_pressures)
                print(
                    f"   Frame {frame}: chamber_pressures=[{p_str}] "
                    f"volume_ratio={vol_ratio:.2f}x",
                    flush=True,
                )
        info = self.solver.get_inflation_info(self.state_0)
        print(f"\n📦 Done. Final ratio: {info['current_ratio']:.2f}x", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Chambers: N-chamber anisotropic inflatable box"
    )
    parser.add_argument("--length", type=float, default=0.35, help="Box length (X axis, m)")
    parser.add_argument("--width", type=float, default=0.35, help="Box width (Y axis, m)")
    parser.add_argument("--height", type=float, default=0.5, help="Box height (Z axis, m)")
    parser.add_argument("--subdivisions_x", type=int, default=10, help="Subdivisions along X")
    parser.add_argument("--subdivisions_y", type=int, default=30, help="Subdivisions along Y")
    parser.add_argument("--subdivisions_z", type=int, default=2, help="Subdivisions along Z")
    parser.add_argument("--num_chambers", type=int, default=2, help="Number of chambers")
    parser.add_argument("--chamber_axis", type=str, default="y", choices=("y", "z"),
                        help="y=side-by-side (bend), z=stacked")
    parser.add_argument("--initial_height", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--k_mu", type=float, default=1.0e5)
    parser.add_argument("--k_lambda", type=float, default=1.0e5)
    parser.add_argument("--k_damp", type=float, default=1.0)
    parser.add_argument("--spring_ke", type=float, default=5.0e4)
    parser.add_argument("--spring_kd", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max_pressure", type=float, default=5.0)
    parser.add_argument(
        "--anisotropy_x", type=float, default=1.0,
        help="Anisotropy along X (1.0 = isotropic)",
    )
    parser.add_argument(
        "--anisotropy_y", type=float, default=1.0,
        help="Anisotropy along Y (1.0 = isotropic)",
    )
    parser.add_argument(
        "--anisotropy_z", type=float, default=1.0,
        help="Anisotropy along Z (e.g. 1.4 = elongate more vertically)",
    )
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=7200, help="Simulation frames (default 7200 = 2 min at 60 fps)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

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
            num_chambers=args.num_chambers,
            chamber_axis=args.chamber_axis,
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
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
