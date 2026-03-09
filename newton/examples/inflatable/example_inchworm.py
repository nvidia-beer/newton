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
Inchworm example – legged crawling with phase-shifted harmonic gait.

Implements the soft robot from arXiv:1911.05227 (Understanding Legged Crawling for
Soft-Robotics). Paper geometry: three-link model – beam length L=120mm, two segments
along the length (left and right), each with one bending actuator. Bending is in the
XZ plane (along the beam, belly down). Torque springs keep the spine stiff along X.

Chamber layout (2×1×2, paper-aligned):
  - Two segments along X (length): ix=0 = left segment, ix=1 = right segment.
  - One division in Y (width). Two in Z (height): iz=0 = bottom, iz=1 = top.
  - ch0=left bottom, ch1=left top, ch2=right bottom, ch3=right top.
  - Inflate only TOP of each segment (ch1, ch3) so beam bends DOWN (belly to ground).
  - ch0 and ch2 = backbone (stiff, no inflation).
"""

import argparse
import os
import sys

import warp as wp
import numpy as np

import newton
from newton.solvers import SolverInflatable, TetraBox

from newton.examples.crawlable.inchworm import (
    InchwormValidation,
    angles_and_contacts_from_metrics,
    get_paper_metrics,
)


def _chamber_index(ix: int, iy: int, iz: int, nx: int, ny: int, nz: int, disabled: set) -> int:
    """Row-major chamber index. Returns -1 if chamber is in disabled (non-inflatable), else ch."""
    ch = ix * (ny * nz) + iy * nz + iz
    return -1 if ch in disabled else ch


def _bottom_edge_vertex_indices(
    vertices: np.ndarray, height: float, width: float, tol_face: float = 0.001, tol_y: float = 0.001
) -> tuple[list[int], list[int]]:
    """Return (indices of bottom vertices at Y+, indices at Y-) in local frame.
    Bottom = z <= -height/2 + tol_face; Y+ = y >= width/2 - tol_y; Y- = y <= -width/2 + tol_y."""
    half_h = height * 0.5
    half_w = width * 0.5
    bottom_z_max = -half_h + tol_face
    y_plus_min = half_w - tol_y
    y_minus_max = -half_w + tol_y
    y_plus_indices = []
    y_minus_indices = []
    for i in range(len(vertices)):
        x, y, z = float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2])
        if z > bottom_z_max:
            continue
        if y >= y_plus_min:
            y_plus_indices.append(i)
        if y <= y_minus_max:
            y_minus_indices.append(i)
    return (y_plus_indices, y_minus_indices)


def _top_edge_vertex_indices(
    vertices: np.ndarray, height: float, width: float, tol_face: float = 0.001, tol_y: float = 0.001
) -> tuple[list[int], list[int]]:
    """Return (indices of top vertices at Y+, indices at Y-) in local frame.
    Top = z >= height/2 - tol_face; Y+ = y >= width/2 - tol_y; Y- = y <= -width/2 + tol_y.
    Same alignment as blue (bottom edges): two lines along X (length)."""
    half_h = height * 0.5
    half_w = width * 0.5
    top_z_min = half_h - tol_face
    y_plus_min = half_w - tol_y
    y_minus_max = -half_w + tol_y
    y_plus_indices = []
    y_minus_indices = []
    for i in range(len(vertices)):
        x, y, z = float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2])
        if z < top_z_min:
            continue
        if y >= y_plus_min:
            y_plus_indices.append(i)
        if y <= y_minus_max:
            y_minus_indices.append(i)
    return (y_plus_indices, y_minus_indices)


def _joint_cross_section_vertex_indices_y(
    vertices: np.ndarray,
    width: float,
    height: float,
    subdivisions_y: int = 6,
    tol_z: float = 0.001,
) -> tuple[list[int], list[int]]:
    """Return (φ1 joint verts, φ2 joint verts) as the top line of verts (Z+ surface) at 1/3 and 2/3 along Y.
    Same number of verts per joint: one row on the Z+ face at each y position."""
    half_w = width * 0.5
    half_h = height * 0.5
    top_z_min = half_h - tol_z
    y_phi1 = -half_w + width / 3.0   # 1/3 along Y (left joint)
    y_phi2 = -half_w + 2.0 * width / 3.0  # 2/3 along Y (right joint)
    grid_spacing_y = width / max(subdivisions_y, 1)
    tol_y = grid_spacing_y * 0.45
    left_indices = []
    right_indices = []
    for i in range(len(vertices)):
        x, y, z = float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2])
        if z < top_z_min:
            continue
        if abs(y - y_phi1) <= tol_y:
            left_indices.append(i)
        if abs(y - y_phi2) <= tol_y:
            right_indices.append(i)
    return (left_indices, right_indices)


def _inchworm_contact_and_joint_vertex_indices(
    vertices: np.ndarray,
    length: float,
    width: float,
    height: float,
    subdivisions: tuple[int, int, int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Compute the 4 vertex index arrays for the inchworm (initialization, once).
    Returns (ground_y_plus, ground_y_minus, joint_phi1, joint_phi2).
    - Ground (blue): bottom Y+ and Y- edges for contact validation and display.
    - Joints (green): top line (Z+ surface) at 1/3 and 2/3 along Y; same vert count per joint."""
    ground_y_plus, ground_y_minus = _bottom_edge_vertex_indices(vertices, height, width)
    joint_phi1, joint_phi2 = _joint_cross_section_vertex_indices_y(
        vertices, width, height, subdivisions_y=subdivisions[1]
    )
    return (ground_y_plus, ground_y_minus, joint_phi1, joint_phi2)


class Example:
    """Inchworm: two segments along length (paper); inflate top of each (ch1, ch3) → bend down; very stiff."""

    def __init__(
        self,
        viewer,
        length: float = 0.12,
        width: float = 0.03,
        height: float = 0.015,
        subdivisions_x: int = 8,
        subdivisions_y: int = 6,
        subdivisions_z: int = 3,
        num_chambers_x: int = 2,
        num_chambers_y: int = 1,
        num_chambers_z: int = 2,
        initial_height: float = 0.0,
        total_mass: float = 0.04,
        k_mu: float = 4.0e6,
        k_lambda: float = 4.0e6,
        k_damp: float = 3.0,
        spring_ke: float = 5.0e4,
        spring_kd: float = 1.0,
        gravity: float = 9.81,
        max_pressure: float = 5.0,
        substeps: int = 5,
        anisotropy_x: float = 1.5,
        anisotropy_y: float = 1.0,
        anisotropy_z: float = 1.15,
        torque_stiffness: float = 100.0,
        torque_damping: float = 20,
        chamber_stiffness_scale: list[float] | None = None,
        chamber_inflation_disabled: list[int] | None = None,
        ground_friction: float = 0.8,
        contact_offset: float = 0.0022,
        contact_iterations: int = 64,
        ground_ke: float = 1.0e7,
        particle_radius: float | None = None,
        gait_enabled: bool = True,
        gait_freq: float = 0.15,
        gait_amplitude: float = 0.80,
        gait_phase: float = 1.57,
        gait_baseline: float = 1.4,  # higher = more lift; keep moderate to avoid losing ground contact
        settle_seconds: float = 1.0,
        start_at_ground_level: bool = True,
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
        self.total_mass = total_mass
        self.ground_friction = ground_friction
        self.contact_offset = float(contact_offset)
        self.contact_iterations = int(contact_iterations)
        self.ground_ke = float(ground_ke)
        self.particle_radius_override = particle_radius
        self.start_at_ground_level = bool(start_at_ground_level)
        # Builder expects density (kg/m³). Use total_mass so that mesh mass matches paper (52 g).
        mesh_volume = float(length * width * height)
        density = total_mass / mesh_volume if mesh_volume > 0 else 1000.0
        self._density = density
        self.max_pressure = max_pressure
        self.anisotropy_x = float(anisotropy_x)
        self.anisotropy_y = float(anisotropy_y)
        self.anisotropy_z = float(anisotropy_z)
        self.viewer = viewer
        self.stiff_axes = ("x",)
        self.torque_display_axis = "all"

        print("\n🐛 Inchworm (arXiv:1911.05227): left/right chambers, phase-shifted gait.", flush=True)
        box = TetraBox(
            size=(self.length, self.width, self.height),
            subdivisions=self.subdivisions,
            verbose=False,
        )
        mesh_data = box.get_mesh_data()
        vertices = mesh_data["vertices"]
        indices = mesh_data["indices"]
        tetrahedra = mesh_data["tetrahedra"]

        # --- 4 vertex index arrays (ground = blue, joints = green), computed once at init ---
        (
            self._bottom_y_plus_indices,
            self._bottom_y_minus_indices,
            self._joint_left_indices,
            self._joint_right_indices,
        ) = _inchworm_contact_and_joint_vertex_indices(
            vertices,
            float(self.length),
            float(self.width),
            float(self.height),
            self.subdivisions,
        )
        print(
            f"   Ground (blue): Y+={len(self._bottom_y_plus_indices)}, Y-={len(self._bottom_y_minus_indices)}. "
            f"Joints (green) φ1={len(self._joint_left_indices)}, φ2={len(self._joint_right_indices)}.",
            flush=True,
        )

        # Place mesh so the bottom is on or just below the ground. start_at_ground_level=True
        # puts the mesh bottom exactly at z=0 (no penetration, no collision impulse).
        vertices_np = np.array(vertices, dtype=np.float64)
        mesh_min_z = float(np.min(vertices_np[:, 2]))
        if initial_height > 0:
            mesh_center_z = initial_height
        elif self.start_at_ground_level:
            mesh_center_z = -mesh_min_z  # bottom at world z=0, resting on ground
        else:
            mesh_center_z = -mesh_min_z - self.contact_offset  # slight penetration for grip

        builder = newton.ModelBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=6e5, kd=2e3, kf=4e4, mu=ground_friction),
        )
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, mesh_center_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            scale=1.0,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
        )

        added_springs = set()
        spring_chamber_list = []
        spring_pairs_local = []
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

        axis_sets = {
            "x": set((min(i, j), max(i, j)) for i, j in box.get_axis_aligned_springs("x")),
            "y": set((min(i, j), max(i, j)) for i, j in box.get_axis_aligned_springs("y")),
            "z": set((min(i, j), max(i, j)) for i, j in box.get_axis_aligned_springs("z")),
        }
        spring_rest_direction = np.zeros((len(spring_pairs_local), 3), dtype=np.float32)
        for k, (a, b) in enumerate(spring_pairs_local):
            key = (min(a, b), max(a, b))
            if key in axis_sets["x"]:
                rest_vec = vertices[b] - vertices[a]
                L = float(np.linalg.norm(rest_vec))
                if L > 1e-9:
                    spring_rest_direction[k] = rest_vec / L
        torque_spring_indices = [
            k for k in range(len(spring_pairs_local))
            if np.linalg.norm(spring_rest_direction[k]) > 0.5
        ]
        self._torque_spring_axis_dict = {k: "x" for k in torque_spring_indices}
        self._torque_spring_indices = torque_spring_indices

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

        if self.chamber_stiffness_scale is not None:
            scales = list(self.chamber_stiffness_scale)
        elif self.chamber_inflation_disabled:
            # Softer backbone (stiff_scale 15) for very compliant motion; 60 = stiffer.
            stiff_scale = 15.0
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
        # Stiff contact so Y+ and Y- (and whole bottom) stay on ground (tuned for paper: no leg lift-off)
        self.model.soft_contact_ke = 1.2e6
        self.model.soft_contact_kd = 1.2e5
        self.model.soft_contact_kf = 3.0e6
        self.model.soft_contact_mu = ground_friction
        self.model.particle_ke = 6.0e5
        self.model.particle_kd = 400.0
        # Particle radius: contact when z < radius; slightly larger = more particles in contact
        if self.particle_radius_override is not None:
            particle_radius = float(self.particle_radius_override)
        else:
            particle_radius = min(0.0018, float(height) * 0.14)
        self._particle_radius_base = float(particle_radius)
        self.model.particle_radius = wp.array(
            np.full(self.model.particle_count, particle_radius), dtype=wp.float32, device=self.model.device
        )
        # Threshold for "in contact with ground": bottom Y+/Y- verts must have z <= this (ground z=0).
        self._contact_z_threshold = max(particle_radius * 2.0, 0.002)

        # Per-particle display radius: blue (in-contact) bottom verts drawn larger.
        self._particle_display_radius = wp.array(
            np.full(self.model.particle_count, particle_radius, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.model.particle_display_radius = self._particle_display_radius

        # Per-particle colors for viewer: bottom Y+/Y- verts = blue (in contact) or hot pink (above threshold).
        self._particle_colors = wp.array(
            np.full((self.model.particle_count, 3), (0.7, 0.6, 0.4), dtype=np.float32),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.model.particle_colors = self._particle_colors

        # Force-based ground plane: very stiff so no particle can lift. Gentle gait keeps contact.
        ground_plane = (0.0, 0.0, 1.0, 0.0)
        self.solver = SolverInflatable(
            model=self.model,
            dt=self.sim_dt,
            mass=total_mass,
            max_volume_ratio=max_pressure,
            solver_type="bicgstab",
            torque_stiffness=torque_stiffness,
            torque_damping=torque_damping,
            spring_rest_direction=spring_rest_direction,
            use_constraint_contacts=True,
            contact_relaxation=0.15,
            contact_max_velocity=0.10,
            contact_max_correction=0.00035,
            contact_iterations=self.contact_iterations,
            handle_self_contact=True,
            self_contact_radius=max(0.008, float(height) * 0.6),
            self_contact_stiffness=1.2e6,
            self_contact_force_cap=8.0,
            ground_plane=ground_plane,
            ground_ke=self.ground_ke,
            ground_kd=4.0e5,
            ground_kf=3.0e6,
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
        self.gait_enabled = gait_enabled
        self.gait_freq = float(gait_freq)
        self.gait_amplitude = float(gait_amplitude)
        self.gait_phase = float(gait_phase)
        self.gait_baseline = float(gait_baseline)
        self.settle_seconds = float(settle_seconds)
        self.start_at_ground_level = bool(start_at_ground_level)

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
        msg = "\n🐛 Inchworm ready. Ch1=left segment (top), Ch3=right segment (top); bend down; phase-shifted gait."
        if not self.gait_enabled:
            msg += " [I]/[K] inflate/deflate, [C] cycle chamber."
        print(msg, flush=True)

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

    def _update_gait_pressure(self):
        """Phase-shifted harmonic gait: left (ch1) and right (ch3) leg pressures.
        During settle_seconds, pressures stay at 1.0 so the robot stays fixed to the ground."""
        if not self.gait_enabled or not self.inflatable_chambers:
            return
        if self.sim_time < self.settle_seconds:
            # No inflation during settle: robot rests on ground, no jump
            for i in self.inflatable_chambers:
                if 0 <= i < len(self.chamber_pressures):
                    self.chamber_pressures[i] = 1.0
            self._apply_pressure()
            return
        omega = 2.0 * np.pi * self.gait_freq
        t = self.sim_time - self.settle_seconds  # gait phase starts after settle
        b, A = self.gait_baseline, self.gait_amplitude
        phi = self.gait_phase
        p_left = np.clip(b + A * np.sin(omega * t), 0.5, self.max_pressure)
        p_right = np.clip(b + A * np.sin(omega * t + phi), 0.5, self.max_pressure)
        if len(self.chamber_pressures) > 1:
            self.chamber_pressures[1] = p_left
        if len(self.chamber_pressures) > 3:
            self.chamber_pressures[3] = p_right
        self._apply_pressure()

    def step(self):
        self._check_keys()
        if self.gait_enabled:
            self._update_gait_pressure()
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
        # During settle: zero velocities so the robot rests on the ground without bouncing
        if self.sim_time <= self.settle_seconds and self.settle_seconds > 0:
            n = self.model.particle_count
            self.state_0.particle_qd.assign(
                wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3, device=self.model.device)
            )

    def _update_particle_colors_for_contact(self, state):
        """Set particle_colors and particle_display_radius:
        - Bottom Y+/Y-: blue (in contact, larger radius) or hot pink (lost contact).
        - Joints φ1, φ2 at 1/3 and 2/3 along Y (X-aligned cross-sections): green.
        - Rest: default."""
        q = np.array(state.particle_q.numpy(), dtype=np.float64)
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        z = q[:, 2]
        threshold = self._contact_z_threshold
        base_r = self._particle_radius_base
        blue_radius = base_r * 5.0
        green_radius = base_r * 6.0  # joint lines (paper Fig. 3) – large so clearly visible
        colors_np = np.full((self.model.particle_count, 3), (0.7, 0.6, 0.4), dtype=np.float32)
        radii_np = np.full(self.model.particle_count, base_r, dtype=np.float32)
        blue = np.array([0.0, 0.4, 1.0], dtype=np.float32)
        hot_pink = np.array([1.0, 0.41, 0.71], dtype=np.float32)
        green = np.array([0.0, 1.0, 0.25], dtype=np.float32)  # bright green for φ1, φ2 joints
        # Joints at 1/3 and 2/3 along Y, X-aligned (paper Fig. 3 = Y–Z slice)
        for i in self._joint_left_indices:
            colors_np[i] = green
            radii_np[i] = green_radius
        for i in self._joint_right_indices:
            colors_np[i] = green
            radii_np[i] = green_radius
        # Bottom Y+/Y-: blue when in contact, hot pink when above ground
        for i in self._bottom_y_plus_indices:
            if z[i] > threshold:
                colors_np[i] = hot_pink
            else:
                colors_np[i] = blue
                radii_np[i] = blue_radius
        for i in self._bottom_y_minus_indices:
            if z[i] > threshold:
                colors_np[i] = hot_pink
            else:
                colors_np[i] = blue
                radii_np[i] = blue_radius
        self._particle_colors.assign(wp.array(colors_np, dtype=wp.vec3, device=self.model.device))
        self._particle_display_radius.assign(wp.array(radii_np, dtype=wp.float32, device=self.model.device))

    def render(self):
        if self.viewer is None:
            return
        self._update_particle_colors_for_contact(self.state_0)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def get_joint_vertex_indices(self) -> tuple[list[int], list[int]]:
        """Return (φ1 joint indices, φ2 joint indices) for the three-link model (paper Fig. 3).
        Paper is Y–Z slice; joints at 1/3 and 2/3 along Y, X-aligned cross-sections. Use for
        observations or segment angles; component to the movement like τ1, τ2 in the paper."""
        return (list(self._joint_left_indices), list(self._joint_right_indices))

    def get_joint_positions(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_joint_positions, right_joint_positions) as (N,3) arrays in world frame.
        Use np.mean(left, axis=0) / np.mean(right, axis=0) for joint-line centroids (e.g. segment angles)."""
        q = np.array(state.particle_q.numpy(), dtype=np.float64)
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        left = q[self._joint_left_indices] if self._joint_left_indices else np.zeros((0, 3))
        right = q[self._joint_right_indices] if self._joint_right_indices else np.zeros((0, 3))
        return (left, right)

    def validate_bottom_contact(self, state, *, verbose: bool = True) -> tuple[bool, dict]:
        """Check that bottom Y+ and Y- vertices never rise above ground (z <= threshold).
        Returns (all_in_contact, info_dict). info_dict has: ok, y_plus_above, y_minus_above,
        max_z_y_plus, max_z_y_minus, count_y_plus, count_y_minus."""
        q = np.array(state.particle_q.numpy(), dtype=np.float64)
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        z = q[:, 2]
        threshold = self._contact_z_threshold
        y_plus_indices = self._bottom_y_plus_indices
        y_minus_indices = self._bottom_y_minus_indices

        z_y_plus = z[y_plus_indices] if y_plus_indices else np.array([])
        z_y_minus = z[y_minus_indices] if y_minus_indices else np.array([])
        above_y_plus = np.where(z_y_plus > threshold)[0]
        above_y_minus = np.where(z_y_minus > threshold)[0]

        y_plus_above = [int(i) for i in above_y_plus]
        y_minus_above = [int(i) for i in above_y_minus]
        max_z_yp = float(np.max(z_y_plus)) if z_y_plus.size else 0.0
        max_z_ym = float(np.max(z_y_minus)) if z_y_minus.size else 0.0
        ok = len(y_plus_above) == 0 and len(y_minus_above) == 0
        info = {
            "ok": ok,
            "y_plus_above": y_plus_above,
            "y_minus_above": y_minus_above,
            "max_z_y_plus": max_z_yp,
            "max_z_y_minus": max_z_ym,
            "count_y_plus": len(y_plus_indices),
            "count_y_minus": len(y_minus_indices),
            "threshold": threshold,
        }
        if verbose and not ok:
            n_yp = len(y_plus_above)
            n_ym = len(y_minus_above)
            print(
                f"   [CONTACT VIOLATION] t={self.sim_time:.3f}s: bottom Y+ verts above ground: {n_yp} (max_z={max_z_yp:.4f}), "
                f"bottom Y- above: {n_ym} (max_z={max_z_ym:.4f}); threshold={threshold:.4f}",
                flush=True,
            )
        return (ok, info)

    def run(
        self,
        num_frames: int = 14400,
        validate_contact: bool = True,
        stop_on_lost_contact: bool = False,
        csv_log_path: str | None = None,
        csv_log_interval: int = 10,
    ):
        validation = InchwormValidation(csv_log_path, log_interval=csv_log_interval)
        had_csv = validation.is_logging
        try:
            for frame in range(num_frames):
                self.step()
                if validate_contact:
                    ok, info = self.validate_bottom_contact(self.state_0, verbose=True)
                    if stop_on_lost_contact and not ok:
                        print(f"\n🐛 STOPPED at frame {frame} (bottom Y+ or Y- lost contact with ground).", flush=True)
                        break
                self.render()
                if csv_log_interval > 0 and frame % csv_log_interval == 0:
                    m = get_paper_metrics(
                        self.state_0.particle_q,
                        self._bottom_y_plus_indices or [],
                        self._bottom_y_minus_indices or [],
                        self._joint_left_indices or [],
                        self._joint_right_indices or [],
                    )
                    if validation.is_logging:
                        phi1_deg, phi2_deg, x1_mm, x2_mm = angles_and_contacts_from_metrics(m)
                        validation.log_row(
                            frame, self.sim_time,
                            m["y_left_ground"], m["z_left_ground"],
                            m["y_right_ground"], m["z_right_ground"],
                            m["y_link_left"], m["z_link_left"],
                            m["y_link_right"], m["z_link_right"],
                            t_norm=None, phi1_deg=phi1_deg, phi2_deg=phi2_deg, x1_mm=x1_mm, x2_mm=x2_mm,
                            fn_left_raw=None, fn_right_raw=None, ft=None,
                        )
                    else:
                        vol_ratio = self.solver.get_volume_ratio(self.state_0)
                        print(f"   Frame {frame} t={self.sim_time:.1f}s vol={vol_ratio:.2f}x contact_ok={m['contact_ok']}", flush=True)
        except KeyboardInterrupt:
            print("\n🐛 Stopped by user (Ctrl+C).", flush=True)
        finally:
            validation.close()
        info = self.solver.get_inflation_info(self.state_0)
        print(f"\n🐛 Done. Final volume ratio: {info['current_ratio']:.2f}x", flush=True)
        if had_csv and csv_log_path:
            print(f"   CSV saved: {os.path.abspath(csv_log_path)} — render GIF from crawlable/inchworm (e.g. render-inchworm-gif.sh)", flush=True)


def _parse_csv(s: str | None, cast):
    return [cast(x.strip()) for x in s.split(",")] if s else None


def main():
    parser = argparse.ArgumentParser(
        description="Inchworm: two bending actuators (ch1/ch3), phase-shifted harmonic gait (arXiv:1911.05227)."
    )
    worm_geom = parser.add_argument_group("Geometry (paper: L=120mm, M=52g)")
    worm_geom.add_argument("--length", type=float, default=0.12, help="Length (X, meters)")
    worm_geom.add_argument("--width", type=float, default=0.03, help="Width (Y, meters)")
    worm_geom.add_argument("--height", type=float, default=0.015, help="Height (Z, meters)")
    worm_geom.add_argument("--subdivisions_x", type=int, default=8)
    worm_geom.add_argument("--subdivisions_y", type=int, default=6)
    worm_geom.add_argument("--subdivisions_z", type=int, default=3)
    parser.add_argument("--num_chambers_x", type=int, default=2, help="Paper: 2 segments along length")
    parser.add_argument("--num_chambers_y", type=int, default=1)
    parser.add_argument("--num_chambers_z", type=int, default=2, help="Top/bottom for bend direction")
    parser.add_argument("--initial_height", type=float, default=0.0, help="Mesh center z (m); 0 = auto (legs on floor, slight penetration)")
    parser.add_argument("--mass", type=float, default=0.052, help="Total mass (kg); paper M=52g")
    parser.add_argument("--k_mu", type=float, default=4.0e6, help="Shear modulus; lower = softer (paper ~6e7)")
    parser.add_argument("--k_lambda", type=float, default=4.0e6, help="Bulk modulus; lower = softer")
    parser.add_argument("--k_damp", type=float, default=3.0, help="Material damping; lower = less resistance to motion")
    parser.add_argument("--spring_ke", type=float, default=5.0e4)
    parser.add_argument("--spring_kd", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max_pressure", type=float, default=5.0)
    parser.add_argument("--anisotropy_x", type=float, default=1.5, help="Expand along beam for bend")
    parser.add_argument("--anisotropy_y", type=float, default=1.0)
    parser.add_argument("--anisotropy_z", type=float, default=1.15)
    parser.add_argument("--torque_stiffness", type=float, default=250000.0, help="X-axis spine (paper: stiff)")
    parser.add_argument("--torque_damping", type=float, default=18.0)
    parser.add_argument(
        "--ground_friction",
        type=float,
        default=0.39,
        help="Coulomb friction μ (higher = legs stay on ground; paper 0.389, default 0.8 for grip).",
    )
    parser.add_argument(
        "--contact_offset",
        type=float,
        default=0.0022,
        help="Vertical offset (m): mesh center = rest_height - contact_offset. Smaller = higher rest = more lift, risk of loss of contact (default 0.0022).",
    )
    parser.add_argument(
        "--contact_iterations",
        type=int,
        default=64,
        help="Contact solver iterations per step; more = better friction/contact (default 64).",
    )
    parser.add_argument(
        "--ground_ke",
        type=float,
        default=1.0e7,
        help="Ground normal stiffness; higher = firmer contact, less lift-off (default 1e7).",
    )
    parser.add_argument(
        "--particle_radius",
        type=float,
        default=None,
        metavar="R",
        help="Override particle contact radius (m); slightly larger = more particles in contact (default: min(0.0018, height*0.14)).",
    )
    parser.add_argument("--substeps", type=int, default=100, help="More substeps = better contact, no float")
    parser.add_argument("--num_frames", type=int, default=14400)
    parser.add_argument("--validate_contact", action="store_true", default=True, help="Check bottom Y+/Y- verts stay on ground (default: True)")
    parser.add_argument("--no_validate_contact", action="store_false", dest="validate_contact", help="Disable contact validation")
    parser.add_argument("--stop_on_lost_contact", action="store_true", help="Stop simulation when bottom loses contact")
    parser.add_argument("--csv_log", type=str, default=None, metavar="PATH", help="Write metrics and parameters to CSV for tuning (default: inchworm_YYYY-MM-DD_HH-MM-SS.csv). Use --csv_log '' to disable.")
    parser.add_argument("--csv_log_dir", type=str, default=None, metavar="DIR", help="Directory for default CSV (e.g. /workspace in Docker so file appears on host in mounted isaac-lab).")
    parser.add_argument("--csv_log_interval", type=int, default=10, metavar="N", help="Log CSV every N frames (default 10; smaller = more samples).")
    parser.add_argument("--chamber_stiffness_scale", type=str, default=None)
    parser.add_argument("--chamber_inflation_disabled", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    gait_group = parser.add_argument_group("Gait (phase-shifted harmonic)")
    gait_group.add_argument("--no_gait", action="store_true", help="Disable automatic gait; use [I]/[K] and [C] to drive")
    gait_group.add_argument("--gait_freq", type=float, default=0.15, help="Gait frequency (Hz); paper quasistatic 0.1–0.3")
    gait_group.add_argument("--gait_amplitude", type=float, default=0.80, help="Pressure amplitude; higher = more lift (default 0.80); keep moderate to avoid losing ground contact")
    gait_group.add_argument("--gait_phase", type=float, default=1.57, help="Phase ψ (rad); paper ψ≈π/2 for robustness")
    gait_group.add_argument("--gait_baseline", type=float, default=1.4, help="Baseline pressure; higher = stronger bend/lift (default 1.4); keep moderate to avoid losing ground contact")
    parser.add_argument("--settle_seconds", type=float, default=1.0, help="Seconds with no gait so robot rests on ground before moving (default 1.0); avoids initial jump.")
    parser.add_argument("--no_start_at_ground_level", action="store_true", help="Place mesh slightly below ground (contact_offset) for grip; default is bottom at z=0 to avoid jump.")
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
            total_mass=args.mass,
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
            chamber_stiffness_scale=chamber_stiffness_scale,
            chamber_inflation_disabled=chamber_inflation_disabled,
            ground_friction=args.ground_friction,
            contact_offset=args.contact_offset,
            contact_iterations=args.contact_iterations,
            ground_ke=args.ground_ke,
            particle_radius=args.particle_radius,
            gait_enabled=not args.no_gait,
            gait_freq=args.gait_freq,
            gait_amplitude=args.gait_amplitude,
            gait_phase=args.gait_phase,
            gait_baseline=args.gait_baseline,
            settle_seconds=args.settle_seconds,
            start_at_ground_level=not args.no_start_at_ground_level,
        )
        example.run(
            num_frames=args.num_frames,
            validate_contact=args.validate_contact,
            stop_on_lost_contact=args.stop_on_lost_contact,
            csv_log_path=InchwormValidation.csv_log_path(args.csv_log, args.csv_log_dir),
            csv_log_interval=args.csv_log_interval,
        )


if __name__ == "__main__":
    main()
