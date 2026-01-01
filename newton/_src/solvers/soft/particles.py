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

"""Particle force evaluation kernels for soft body simulation."""

import warp as wp

from newton import ParticleFlags

# Use ParticleFlags.ACTIVE value directly
PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)


@wp.func
def particle_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    """Compute normal and tangential friction force for a single contact."""
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d

    # Contact force
    fn = jn + jd

    # Friction force
    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Coulomb condition
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))

    # Total force
    return -n * fn - vt * ft


@wp.kernel
def eval_particle_forces(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
):
    """Evaluate particle-particle interaction forces using hash grid acceleration."""
    tid = wp.tid()

    # Order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # Hash grid has not been built yet
        return
    if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]

    f = wp.vec3()

    # Particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:
            # Compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            if err <= k_cohesion:
                n = n / d
                vrel = v - particle_v[index]

                f = f + particle_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f

