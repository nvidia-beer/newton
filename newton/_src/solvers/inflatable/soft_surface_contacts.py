# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Soft-body surface-to-surface contact forces.

For each pair of soft bodies (A, B), surface particles of A are queried
against B's live BVH mesh and vice versa. The BVH is refitted from
current particle positions each substep so contacts are against the
deformed geometry, not the rest shape.

Design
------
Each soft body registers a live :class:`wp.Mesh` built from its surface
triangles. Mesh vertex index ``v`` maps directly to
``particle_q[start + v]``, where ``start`` is the body's particle offset
in the global array. Before each force evaluation, :meth:`SoftBodySurfaceContacts.update`
copies current particle positions into every mesh's ``points`` buffer and
calls ``mesh.refit()`` to rebuild the BVH in place.

The contact kernel queries every surface particle of body A against all
other bodies' BVH meshes via ``wp.mesh_query_point_sign_normal``.  A
contact fires when the particle's contact sphere (radius
``particle_radius[p]``) overlaps the mesh surface:

    penetration = particle_radius[p] − signed_distance   (> 0 when overlapping)

The penalty force pushes the particle along the mesh outward normal:

    F = ke · penetration · n  −  kd · min(v·n, 0) · n

Because each body's surface particles are queried against every other
body's mesh, forces are applied symmetrically without explicit
action-reaction bookkeeping: A's particles feel B's mesh, and B's
particles feel A's mesh.

The BVH query radius per body is set at init to the rest-pose bounding-box
diagonal, so even deeply interpenetrating particles find the nearest surface
triangle.  Forces are gated by ``penetration > 0`` so a large search radius
never produces spurious repulsion.
"""

from __future__ import annotations

import numpy as np
import warp as wp

_PARTICLE_FLAG_ACTIVE = wp.constant(1)


@wp.kernel
def _eval_soft_surface_contacts_kernel(
    particle_q: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    surface_particle_idx: wp.array[wp.int32],
    surface_body_idx: wp.array[wp.int32],
    mesh_ids: wp.array[wp.uint64],
    mesh_search_radii: wp.array[float],
    n_bodies: int,
    ke: float,
    kd: float,
    particle_f: wp.array[wp.vec3],
):
    """One thread per surface particle — accumulate penalty from all other bodies' meshes.

    Args:
        particle_q: All soft-body particle positions [m], shape ``(N,)``.
        particle_v: All soft-body particle velocities [m/s], shape ``(N,)``.
        particle_radius: Per-particle contact radius [m], shape ``(N,)``.
        particle_flags: Per-particle activity flags, shape ``(N,)``.
        surface_particle_idx: Global particle index for each surface vertex,
            shape ``(S,)`` where S = total surface vertices across all bodies.
        surface_body_idx: Body index (0 .. n_bodies-1) for each surface vertex,
            shape ``(S,)``.
        mesh_ids: Live :class:`wp.Mesh` id per body, shape ``(n_bodies,)``.
        mesh_search_radii: BVH query radius per body [m], shape ``(n_bodies,)``.
            Set to each body's bounding-box diagonal so that even deeply
            interpenetrating particles find the nearest surface triangle.
        n_bodies: Number of registered soft bodies.
        ke: Normal penalty stiffness [N/m].
        kd: Normal damping [N·s/m].
        particle_f: Output force accumulator [N], shape ``(N,)``.
    """
    tid = wp.tid()
    p = surface_particle_idx[tid]
    my_body = surface_body_idx[tid]

    if (particle_flags[p] & _PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_q[p]
    v = particle_v[p]
    r = particle_radius[p]

    f = wp.vec3(0.0, 0.0, 0.0)

    for b in range(n_bodies):
        if b == my_body:
            continue

        mesh_id = mesh_ids[b]
        max_dist = mesh_search_radii[b]
        sign = float(0.0)
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)

        if wp.mesh_query_point_sign_normal(mesh_id, x, max_dist, sign, face_index, face_u, face_v):
            closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
            diff = x - closest
            raw_dist = wp.length(diff)
            if raw_dist < 1.0e-8:
                continue

            n = (diff / raw_dist) * sign  # outward normal of body b at closest point
            signed_dist = raw_dist * sign

            pen = r - signed_dist
            if pen <= 0.0:
                continue

            vn = wp.dot(v, n)
            fn = ke * pen - kd * wp.min(vn, 0.0)
            f = f + n * fn

    wp.atomic_add(particle_f, p, f)


class SoftBodySurfaceContacts:
    """Manages live BVH meshes for N soft bodies and evaluates mutual surface contact forces.

    Args:
        particle_q: The model's particle position array (used to seed initial mesh points).
        particle_ranges: List of ``(start, count)`` tuples — the slice of
            ``particle_q`` that belongs to each soft body.  Vertex index ``v``
            in body ``i``'s surface mesh maps to ``particle_q[start_i + v]``.
        surface_triangles_list: Per-body list of surface triangle index arrays,
            shape ``(T_i, 3)``, with indices local to the body (0 .. count_i-1).
            Bodies without surface data (``None``) are silently skipped.
        ke: Normal penalty stiffness [N/m].
        kd: Normal damping [N·s/m].
        device: Warp device string.
    """

    def __init__(
        self,
        particle_q: wp.array,
        particle_ranges: list[tuple[int, int]],
        surface_triangles_list: list[np.ndarray | None],
        ke: float = 1.0e3,
        kd: float = 1.0e2,
        device: str = "cuda",
    ) -> None:
        self.ke = float(ke)
        self.kd = float(kd)
        self.device = device

        self._meshes: list[wp.Mesh] = []
        self._mesh_starts: list[int] = []
        self._mesh_counts: list[int] = []
        mesh_ids_list: list[int] = []
        search_radii: list[float] = []

        # Accumulate surface-particle → global-index / body-index mappings.
        surface_p_idx: list[int] = []
        surface_b_idx: list[int] = []

        pts_np = particle_q.numpy()

        for body_idx, ((start, count), tris) in enumerate(zip(particle_ranges, surface_triangles_list, strict=False)):
            if tris is None or len(tris) == 0:
                # No surface data — register a placeholder mesh so body
                # indices stay consistent; it will never be queried against
                # itself and emits no contacts.
                dummy_pts = wp.zeros(1, dtype=wp.vec3, device=device)
                dummy_idx = wp.zeros(3, dtype=wp.int32, device=device)
                mesh = wp.Mesh(points=dummy_pts, indices=dummy_idx)
                search_radii.append(0.0)
            else:
                tris_np = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
                pts_body = np.asarray(pts_np[start : start + count], dtype=np.float32)

                mesh_pts = wp.array(pts_body, dtype=wp.vec3, device=device)
                mesh_idx = wp.array(tris_np.flatten(), dtype=wp.int32, device=device)
                mesh = wp.Mesh(points=mesh_pts, indices=mesh_idx)

                # BVH search radius = bounding-box diagonal of this body's rest mesh.
                # This ensures even deeply interpenetrating particles find the nearest
                # surface triangle, regardless of penetration depth.
                bbox_diag = float(np.linalg.norm(pts_body.max(axis=0) - pts_body.min(axis=0)))
                search_radii.append(bbox_diag)

                # Unique surface vertices (triangle indices, local to this body).
                unique_verts = np.unique(tris_np)
                for v in unique_verts:
                    surface_p_idx.append(start + int(v))
                    surface_b_idx.append(body_idx)

            self._meshes.append(mesh)
            self._mesh_starts.append(start)
            self._mesh_counts.append(count)
            mesh_ids_list.append(mesh.id)

        self.n_bodies = len(self._meshes)
        self._mesh_ids_wp = wp.array(np.array(mesh_ids_list, dtype=np.uint64), dtype=wp.uint64, device=device)
        self._mesh_search_radii = wp.array(np.array(search_radii, dtype=np.float32), dtype=float, device=device)
        self._surface_p_idx = wp.array(np.array(surface_p_idx, dtype=np.int32), dtype=wp.int32, device=device)
        self._surface_b_idx = wp.array(np.array(surface_b_idx, dtype=np.int32), dtype=wp.int32, device=device)
        self._n_surface = len(surface_p_idx)

    # ------------------------------------------------------------------

    def update(self, particle_q: wp.array) -> None:
        """Copy current particle positions into each mesh's point buffer and refit BVH.

        Args:
            particle_q: Current particle positions [m], shape ``(N,)``.
        """
        for mesh, start, count in zip(self._meshes, self._mesh_starts, self._mesh_counts, strict=False):
            if count == 0 or mesh.points.shape[0] == 1:
                continue  # placeholder / no-surface body
            wp.copy(mesh.points, particle_q, dest_offset=0, src_offset=start, count=count)
            mesh.refit()

    def eval_forces(
        self,
        particle_q: wp.array,
        particle_v: wp.array,
        particle_radius: wp.array,
        particle_flags: wp.array,
        particle_count: int,
    ) -> wp.array:
        """Compute and return per-particle surface contact forces [N].

        Args:
            particle_q: Current positions [m], shape ``(N,)``.
            particle_v: Current velocities [m/s], shape ``(N,)``.
            particle_radius: Contact radii [m], shape ``(N,)``.
            particle_flags: Activity flags, shape ``(N,)``.
            particle_count: Total number of particles ``N``.

        Returns:
            Force array [N], shape ``(N,)``, dtype :class:`wp.vec3`.
        """
        forces = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        if self._n_surface == 0 or self.n_bodies < 2:
            return forces

        wp.launch(
            kernel=_eval_soft_surface_contacts_kernel,
            dim=self._n_surface,
            inputs=[
                particle_q,
                particle_v,
                particle_radius,
                particle_flags,
                self._surface_p_idx,
                self._surface_b_idx,
                self._mesh_ids_wp,
                self._mesh_search_radii,
                self.n_bodies,
                self.ke,
                self.kd,
            ],
            outputs=[forces],
            device=self.device,
        )
        return forces
