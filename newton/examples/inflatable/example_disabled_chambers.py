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
Disabled chambers example - 3D inflatable with chamber grid (same as chambers).

Chambers: num_chambers_x × num_chambers_y × num_chambers_z grid. Index ch = ix*(ny*nz)+iy*nz+iz.
[I]/[=] inflate, [K]/[-] deflate, [C] cycle chamber.

Options: --chamber_stiffness_scale 1.5,1.0,... (per-chamber FEM scale);
  --chamber_inflation_disabled 0,2 (chambers that never inflate, mask -1).
"""

import argparse
import warp as wp
import numpy as np

import newton
from newton.solvers import SolverInflatable, TetraBox


def _chamber_index(ix: int, iy: int, iz: int, nx: int, ny: int, nz: int, disabled: set) -> int:
    """Row-major chamber index; -1 if chamber is disabled."""
    ch = ix * (ny * nz) + iy * nz + iz
    return -1 if ch in disabled else ch


class Example:
    """3D N-chamber inflatable box (chamber grid + optional per-chamber stiffness / disabled)."""

    def __init__(
        self,
        viewer,
        length: float = 0.35,
        width: float = 0.35,
        height: float = 0.1,
        subdivisions_x: int = 4,
        subdivisions_y: int = 4,
        subdivisions_z: int = 4,
        num_chambers_x: int = 1,
        num_chambers_y: int = 2,
        num_chambers_z: int = 2,
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
        chamber_stiffness_scale: list[float] | None = None,
        chamber_inflation_disabled: list[int] | None = None,  # default [0, 2] = bottom two in 2x2 grid
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
        # 3D grid: nx, ny, nz
        self.num_chambers_x = max(1, int(num_chambers_x))
        self.num_chambers_y = max(1, int(num_chambers_y))
        self.num_chambers_z = max(1, int(num_chambers_z))
        self.total_chambers = self.num_chambers_x * self.num_chambers_y * self.num_chambers_z
        # Per-chamber FEM stiffness scale (optional): tet in chamber c uses k_mu*scale[c], etc.
        self.chamber_stiffness_scale = chamber_stiffness_scale
        # Chamber indices that never inflate (mask -1). Default [0, 2] = bottom two in 2x2 grid
        _disabled = chamber_inflation_disabled if chamber_inflation_disabled is not None else [0, 2]
        self.chamber_inflation_disabled = set(int(x) for x in _disabled)
        self.inflatable_chambers = sorted(
            c for c in range(self.num_chambers_x * self.num_chambers_y * self.num_chambers_z)
            if c not in self.chamber_inflation_disabled
        )
        self.initial_height = initial_height
        self.mass = mass
        self.max_pressure = max_pressure
        self.anisotropy_x = float(anisotropy_x)
        self.anisotropy_y = float(anisotropy_y)
        self.anisotropy_z = float(anisotropy_z)
        self.viewer = viewer

        # 3D tetrahedral box: length (X), width (Y), height (Z); Z = chamber axis
        print(f"\n🪱 Generating 3D tetrahedral box (worm = chambers)...", flush=True)
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
        spring_chamber_list = []  # chamber index per spring, same order as builder springs
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
                    # Chamber from spring edge midpoint (3D grid: X, Y, Z)
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

        # Assign each tet to a chamber by centroid (3D grid: X, Y, Z)
        # tet_chamber_id_np: raw chamber index (0..N-1) for stiffness lookup; tet_chamber_mask_np: -1 if disabled else chamber index for inflation
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
        for c in range(self.total_chambers):
            n_tets = np.sum(tet_chamber_mask_np == c)
            dis = " (inflation disabled)" if c in self.chamber_inflation_disabled else ""
            print(f"   Chamber {c}: {n_tets} tetrahedra{dis}", flush=True)
        n_no_inflate = int(np.sum(tet_chamber_mask_np == -1))
        if n_no_inflate > 0:
            print(f"   Non-inflatable (mask -1): {n_no_inflate} tetrahedra", flush=True)
        print(
            f"   Chamber grid: {self.num_chambers_x} (X) x {self.num_chambers_y} (Y) x {self.num_chambers_z} (Z) = {self.total_chambers} chambers",
            flush=True,
        )
        if self.chamber_inflation_disabled:
            print(f"   Inflation disabled for chamber(s): {sorted(self.chamber_inflation_disabled)} (mask -1)", flush=True)

        self.model = builder.finalize()

        # Per-chamber FEM material scaling at init only (not updated at runtime): stiffer backbone (disabled chambers) for worm movement
        if self.chamber_stiffness_scale is not None:
            scales = list(self.chamber_stiffness_scale)
        elif self.chamber_inflation_disabled:
            # Default: disabled chambers (e.g. 0,2) much stiffer to create worm backbone
            stiff_scale = 4.0
            scales = [stiff_scale if c in self.chamber_inflation_disabled else 1.0 for c in range(self.total_chambers)]
            print(f"   Default stiffness for non-inflatable chambers: {stiff_scale}x (worm backbone)", flush=True)
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
            self.model.tet_materials.assign(
                wp.array(materials_np, dtype=wp.float32, device=self.model.device)
            )
            # Verify: read back from model and print mean k_mu per chamber (confirms solver will see stiff backbone)
            readback = self.model.tet_materials.numpy()
            for ch in range(self.total_chambers):
                mask = tet_chamber_id_np == ch
                if np.any(mask):
                    mean_k_mu = float(np.mean(readback[mask, 0]))
                    expected = k_mu * (scales[ch] if ch < len(scales) else 1.0)
                    label = " (backbone)" if ch in self.chamber_inflation_disabled else ""
                    print(f"   Chamber {ch}: mean k_mu={mean_k_mu:.0f} (expected {expected:.0f}){label}", flush=True)
            print(f"   Per-chamber stiffness scale applied (FEM k_mu/k_lambda/k_damp)", flush=True)

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
            handle_self_contact=True,
            self_contact_radius=0.012,
            self_contact_stiffness=8.0e3,
            self_contact_force_cap=1.0,
            self_contact_edge_edge=False,
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
            num_chambers=self.total_chambers,
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

        # Per-chamber pressures (one per chamber; length = total_chambers for grid)
        self.chamber_pressures = [1.0] * self.total_chambers
        self.current_pressure = 1.0  # for single-pressure fallback
        self.pressure_step = 0.15
        self.active_chamber = self.inflatable_chambers[0] if self.inflatable_chambers else 0
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
        n_inflatable = len(self.inflatable_chambers)
        print(
            f"   [Active chamber: {self.active_chamber} / {self.total_chambers} (inflatable {next_idx + 1}/{n_inflatable})]",
            flush=True,
        )

    def _print_help(self):
        n_inf = len(self.inflatable_chambers)
        print(
            f"\n🪱 Worm ready! Box {self.length:.2f}x{self.width:.2f}x{self.height:.2f}m, "
            f"chambers {self.num_chambers_x}x{self.num_chambers_y}x{self.num_chambers_z}={self.total_chambers} "
            f"({n_inf} inflatable). [I]/[K] inflate/deflate, [C] cycle (inflatable only).",
            flush=True,
        )

    def _on_key_press(self, symbol, modifiers):
        if symbol in (105, 61):   # I, =
            self._set_pressure_delta(self.pressure_step)
        elif symbol in (107, 45):  # K, -
            self._set_pressure_delta(-self.pressure_step)
        elif symbol == 99:  # C
            self._cycle_chamber()

    def _check_keys(self):
        if not self.viewer or not hasattr(self.viewer, "renderer"):
            return
        r = self.viewer.renderer
        if not hasattr(r, "is_key_down"):
            return
        cooldown = getattr(self, "_key_cooldown", 0)
        if cooldown > 0:
            self._key_cooldown = cooldown - 1
            return
        if r.is_key_down(105) or r.is_key_down(61):
            self._set_pressure_delta(self.pressure_step)
            self._key_cooldown = 10
        elif r.is_key_down(107) or r.is_key_down(45):
            self._set_pressure_delta(-self.pressure_step)
            self._key_cooldown = 10
        # [C] cycle chamber handled only in _on_key_press to avoid double-cycle (press + key-held)

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
        print(f"\n🪱 Worm (chambers) demo running...", flush=True)
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
        print(f"\n🪱 Done. Final ratio: {info['current_ratio']:.2f}x", flush=True)


def _parse_csv(s: str | None, cast):
    return [cast(x.strip()) for x in s.split(",")] if s else None


def main():
    parser = argparse.ArgumentParser(description="Disabled chambers: 3D chamber-grid inflatable")
    parser.add_argument("--length", type=float, default=0.35)
    parser.add_argument("--width", type=float, default=0.35)
    parser.add_argument("--height", type=float, default=0.1)
    parser.add_argument("--subdivisions_x", type=int, default=10)
    parser.add_argument("--subdivisions_y", type=int, default=30)
    parser.add_argument("--subdivisions_z", type=int, default=4)
    parser.add_argument("--num_chambers_x", type=int, default=1)
    parser.add_argument("--num_chambers_y", type=int, default=2)
    parser.add_argument("--num_chambers_z", type=int, default=2)
    parser.add_argument("--initial_height", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--k_mu", type=float, default=1.0e5)
    parser.add_argument("--k_lambda", type=float, default=1.0e5)
    parser.add_argument("--k_damp", type=float, default=1.0)
    parser.add_argument("--spring_ke", type=float, default=5.0e4)
    parser.add_argument("--spring_kd", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max_pressure", type=float, default=5.0)
    parser.add_argument("--anisotropy_x", type=float, default=1.0)
    parser.add_argument("--anisotropy_y", type=float, default=1.0)
    parser.add_argument("--anisotropy_z", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=7200)
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
            chamber_stiffness_scale=chamber_stiffness_scale,
            chamber_inflation_disabled=chamber_inflation_disabled,
        )
        example.run(num_frames=args.num_frames)


if __name__ == "__main__":
    main()
