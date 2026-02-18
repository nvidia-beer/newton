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
Shared code for inflatable + MPM sand examples (e.g. inflatable_box_sand, mpm_worm_sand).

Uses a single scatter method: for each sand-contact impulse, find the nearest
deformable particle and add the impulse there (atomic). Caller pre-filters
impulses to the deformable collider and launches with dim=collider_count.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton


@wp.kernel
def scatter_impulses_to_particles(
    impulse_pos: wp.array(dtype=wp.vec3),
    impulses: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_impulse: wp.array(dtype=wp.vec3),
):
    """For each sand-contact impulse, find nearest deformable particle and add impulse (atomic)."""
    i = wp.tid()
    pos = impulse_pos[i]
    imp = impulses[i]
    n = particle_q.shape[0]
    best_j = int(0)
    best_d2 = float(wp.length_sq(particle_q[0] - pos))
    for j in range(1, n):
        d2 = wp.length_sq(particle_q[j] - pos)
        if d2 < best_d2:
            best_d2 = d2
            best_j = j
    wp.atomic_add(particle_impulse, best_j, imp)


@wp.kernel
def apply_impulse_velocity_kick(
    particle_qd: wp.array(dtype=wp.vec3),
    particle_impulse: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    scale: float,
    v_max: float,
    vertical_only: int,
):
    """Add impulse/mass to velocity. If vertical_only, only upward component (support, no deformation)."""
    i = wp.tid()
    inv_m = particle_inv_mass[i]
    imp = particle_impulse[i]
    if vertical_only != 0:
        imp = wp.vec3(0.0, 0.0, wp.max(imp[2], 0.0))
    v_new = particle_qd[i] + scale * imp * inv_m
    v_mag = wp.length(v_new)
    if v_mag > v_max and v_mag > 1.0e-9:
        v_new = v_new * (v_max / v_mag)
    particle_qd[i] = v_new


@wp.kernel
def clamp_soft_particles_above_ground(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    ground_z: float,
):
    """No particle may go below the ground; the inflation object pushes back."""
    i = wp.tid()
    q = particle_q[i]
    if q[2] < ground_z:
        particle_q[i] = wp.vec3(q[0], q[1], ground_z)
        v = particle_qd[i]
        if v[2] < 0.0:
            particle_qd[i] = wp.vec3(v[0], v[1], 0.0)


def emit_sand(
    sand_builder: newton.ModelBuilder,
    voxel_size: float,
    sand_bed_top: float,
) -> None:
    """Emit a bed of MPM sand particles (shared by box and worm sand examples)."""
    particles_per_cell = 3.0
    density = 2500.0
    bed_lo = np.array([-1.0, -1.0, 0.0])
    bed_hi = np.array([1.0, 1.0, sand_bed_top])
    bed_res = np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size).astype(int)
    cell_size = (bed_hi - bed_lo) / bed_res
    radius = float(np.max(cell_size) * 0.5)
    mass = float(np.prod(cell_size) * density)
    sand_builder.add_particle_grid(
        pos=wp.vec3(bed_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=bed_res[0] + 1,
        dim_y=bed_res[1] + 1,
        dim_z=bed_res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )
