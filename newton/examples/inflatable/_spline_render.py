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

"""Per-finger spline overlay for the inflatable gripper example.

Reads the keypoint bindings authored by
``newton_mesh_tools.write_asset_usd`` into ``customData["finger_skeletons"]``
on the soft asset's TetMesh prim. Each keypoint is barycentrically
bound to a specific tet:

    p_k = sum_{i=0..3} bary[k, i] * particle_q[ tets[ tet_idx[k] ][i] ]

Computing this from the current ``particle_q`` each frame yields a
spline that follows whatever deformation the FEM produces, including
chamber inflation and stem-driven motion. The overlay renders the
per-finger curve as line segments through consecutive keypoints (cheap
piecewise-linear visualisation of the cubic-bspline curves baked into
the USD) plus a sphere at each keypoint.

The same evaluation kernel is the one a future inverse-kinematics or
capsule-collider proxy would use — see
``newton_mesh_tools.skeleton.evaluate_keypoints`` for the analogous
host-side helper.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _gather_keypoints_from_tets(
    particle_q: wp.array(dtype=wp.vec3),
    tet_vertex_indices: wp.array(dtype=wp.int32, ndim=2),  # (num_tets, 4)
    keypoint_tet_indices: wp.array(dtype=wp.int32),  # (F · K,)
    keypoint_bary: wp.array(dtype=wp.float32),  # (F · K · 4,) flat
    # output:
    keypoint_world: wp.array(dtype=wp.vec3),  # (F · K,)
):
    """``p_k = sum_i bary[i] * particle_q[ tets[tet_idx, i] ]`` per keypoint.

    Pure read-out from current particle positions: the binding is
    affine, so a keypoint stays consistent with whatever deformation
    the FEM solver produced for its host tet.
    """
    kid = wp.tid()
    tid = keypoint_tet_indices[kid]
    base_b = kid * 4
    p0 = particle_q[tet_vertex_indices[tid, 0]]
    p1 = particle_q[tet_vertex_indices[tid, 1]]
    p2 = particle_q[tet_vertex_indices[tid, 2]]
    p3 = particle_q[tet_vertex_indices[tid, 3]]
    b0 = keypoint_bary[base_b + 0]
    b1 = keypoint_bary[base_b + 1]
    b2 = keypoint_bary[base_b + 2]
    b3 = keypoint_bary[base_b + 3]
    keypoint_world[kid] = b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


@wp.kernel
def _build_segment_endpoints(
    keypoint_world: wp.array(dtype=wp.vec3),
    seg_start_idx: wp.array(dtype=wp.int32),  # (num_segments,)
    seg_end_idx: wp.array(dtype=wp.int32),  # (num_segments,)
    # output
    starts_out: wp.array(dtype=wp.vec3),
    ends_out: wp.array(dtype=wp.vec3),
):
    """Map per-finger keypoint pairs to start/end vec3 arrays for ``log_lines``."""
    sid = wp.tid()
    starts_out[sid] = keypoint_world[seg_start_idx[sid]]
    ends_out[sid] = keypoint_world[seg_end_idx[sid]]


def _decode_skeleton_dict(
    raw: dict,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Decode the flat ``customData["finger_skeletons"]`` dict.

    Returns ``(num_fingers, num_keypoints_per_finger, tet_indices,
    bary_weights, radii)`` where ``tet_indices`` is shape ``(F·K,)``
    int32, ``bary_weights`` is shape ``(F·K·4,)`` float32, and
    ``radii`` is either shape ``(F·K,)`` float32 (post-bake assets) or
    ``None`` (legacy assets that predate the per-keypoint radius bake).
    All arrays are in finger-major order, matching how mesh-tools wrote
    them. Returns ``None`` for missing / malformed payloads — the
    consumer can then disable the overlay.

    Lives here (not in mesh-tools) so the example doesn't pull
    ``newton_mesh_tools`` as a runtime dependency.
    """
    if not raw:
        return None
    try:
        F = int(raw["num_fingers"])
        K = int(raw["num_keypoints_per_finger"])
    except (KeyError, TypeError, ValueError):
        return None
    if F <= 0 or K <= 0:
        return None
    try:
        tet_idx = np.asarray(list(raw["tet_indices"]), dtype=np.int32)
        bary = np.asarray(list(raw["bary_weights"]), dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if tet_idx.size != F * K or bary.size != F * K * 4:
        return None
    radii: np.ndarray | None = None
    raw_radii = raw.get("radii") if hasattr(raw, "get") else None
    if raw_radii is not None:
        try:
            radii_np = np.asarray(list(raw_radii), dtype=np.float32)
        except (TypeError, ValueError):
            radii_np = None
        else:
            if radii_np.size == F * K:
                radii = radii_np
    return F, K, tet_idx, bary, radii


class FingerSplineLines:
    """Per-frame spline overlay for the gripper's per-finger keypoints.

    Constructed once at example init from the asset's raw
    ``finger_skeletons`` customData dict and the model's tet vertex
    array. Call :meth:`log` each render pass to refresh the keypoint
    world positions from the current ``particle_q`` and submit lines
    + points to the viewer.

    Attributes:
        num_fingers: Number of fingers represented (rows in the
            keypoint set).
        num_keypoints_per_finger: Constant K per finger.
        num_segments: ``num_fingers * (K - 1)`` line segments rendered.
        num_keypoints: ``num_fingers * K`` total keypoints.
    """

    def __init__(
        self,
        *,
        skeletons_dict: dict | None,
        model_tet_vertex_indices: wp.array,  # ``model.tet_indices`` flat int array
        device: wp.Device | str,
    ) -> None:
        decoded = _decode_skeleton_dict(skeletons_dict) if skeletons_dict else None
        if decoded is None:
            self.num_fingers = 0
            self.num_keypoints_per_finger = 0
            self.num_segments = 0
            self.num_keypoints = 0
            self.keypoint_radii_np: np.ndarray | None = None
            self._device = device
            return
        F, K, tet_idx_np, bary_np, radii_np = decoded
        self.num_fingers = F
        self.num_keypoints_per_finger = K
        self.num_keypoints = F * K
        self.num_segments = F * max(0, K - 1)
        # Per-keypoint collision-proxy radii baked into the asset; ``None``
        # for legacy assets without the bake. Exposed verbatim so callers
        # like the capsule colliders can derive per-capsule radii from
        # consecutive keypoint pairs without re-decoding the customData.
        self.keypoint_radii_np: np.ndarray | None = (
            radii_np.astype(np.float32, copy=True) if radii_np is not None else None
        )
        self._device = device
        # Per-segment endpoint lookups: consecutive keypoints within each finger.
        seg_start = np.empty(self.num_segments, dtype=np.int32)
        seg_end = np.empty(self.num_segments, dtype=np.int32)
        for fi in range(F):
            for s in range(K - 1):
                idx = fi * (K - 1) + s
                seg_start[idx] = fi * K + s
                seg_end[idx] = fi * K + s + 1
        self._tet_idx = wp.array(tet_idx_np, dtype=wp.int32, device=device)
        self._bary = wp.array(bary_np, dtype=wp.float32, device=device)
        self._seg_start_idx = wp.array(seg_start, dtype=wp.int32, device=device)
        self._seg_end_idx = wp.array(seg_end, dtype=wp.int32, device=device)
        self._tet_vertex_indices = model_tet_vertex_indices
        self._keypoint_world = wp.zeros(self.num_keypoints, dtype=wp.vec3, device=device)
        self._seg_starts = wp.zeros(self.num_segments, dtype=wp.vec3, device=device)
        self._seg_ends = wp.zeros(self.num_segments, dtype=wp.vec3, device=device)
        # ``viewer.log_points`` advertises ``radii: wp.array | float`` and
        # ``colors: wp.array | tuple`` but the GL backend's
        # ``update_from_points`` requires both as arrays. Allocate them
        # once; ``log`` refreshes the contents only when the caller
        # changes the value.
        self._radii: wp.array | None = None
        self._last_radius: float = -1.0
        self._point_colors: wp.array | None = None
        self._last_point_color: tuple[float, float, float] | None = None

    @property
    def keypoint_world(self) -> wp.array | None:
        """World-space keypoint positions [m], shape ``(num_fingers · K,)``.

        ``None`` when no skeleton was decoded (e.g. legacy asset). The
        buffer is filled by :meth:`update_world`; downstream consumers
        (capsule colliders, IK solvers, etc.) should call
        :meth:`update_world` first each frame, then read this array.
        """
        if self.num_keypoints == 0:
            return None
        return self._keypoint_world

    def update_world(self, particle_q: wp.array) -> None:
        """Refresh keypoint world positions from the current FEM state.

        Runs the per-keypoint barycentric gather (``p_k = sum_i bary[k,i]
        · particle_q[ tets[tet_idx[k], i] ]``) and rebuilds the
        per-segment endpoint arrays used by :meth:`log`. Cheap (one
        small kernel per pass) so it's safe to call every render frame
        regardless of whether the overlay is visible — capsule
        colliders that read :attr:`keypoint_world` rely on it.
        """
        if self.num_keypoints == 0:
            return
        wp.launch(
            _gather_keypoints_from_tets,
            dim=self.num_keypoints,
            inputs=[
                particle_q,
                self._tet_vertex_indices,
                self._tet_idx,
                self._bary,
            ],
            outputs=[self._keypoint_world],
            device=self._device,
        )
        if self.num_segments > 0:
            wp.launch(
                _build_segment_endpoints,
                dim=self.num_segments,
                inputs=[
                    self._keypoint_world,
                    self._seg_start_idx,
                    self._seg_end_idx,
                ],
                outputs=[self._seg_starts, self._seg_ends],
                device=self._device,
            )

    def log(
        self,
        viewer,
        *,
        line_name: str = "finger_splines",
        point_name: str = "finger_spline_keypoints",
        line_color: tuple[float, float, float] = (0.95, 0.55, 0.10),
        point_color: tuple[float, float, float] = (0.10, 0.20, 0.95),
        line_width: float = 0.005,
        point_radius: float = 0.012,
        hidden: bool = False,
    ) -> None:
        """Submit the current keypoint buffers to the viewer.

        Pure visualisation: assumes :meth:`update_world` was called
        earlier this frame. ``hidden`` suppresses rendering without
        affecting the keypoint buffers (capsule colliders sharing this
        renderer's :attr:`keypoint_world` keep working when the overlay
        is toggled off).
        """
        if self.num_keypoints == 0:
            return
        # Lines: per-finger consecutive-keypoint segments.
        if self.num_segments > 0:
            viewer.log_lines(
                line_name,
                self._seg_starts,
                self._seg_ends,
                line_color,
                width=line_width,
                hidden=hidden,
            )
        # Points: spheres at each keypoint. ``log_points`` takes a
        # per-point radius *array* and a per-point colors *array*; build
        # / refresh both lazily so caller-changed values take effect
        # without a per-frame allocation.
        if self._radii is None or float(point_radius) != self._last_radius:
            self._radii = wp.full(
                self.num_keypoints,
                float(point_radius),
                dtype=wp.float32,
                device=self._device,
            )
            self._last_radius = float(point_radius)
        pc = (float(point_color[0]), float(point_color[1]), float(point_color[2]))
        if self._point_colors is None or self._last_point_color != pc:
            self._point_colors = wp.full(
                self.num_keypoints,
                wp.vec3(*pc),
                dtype=wp.vec3,
                device=self._device,
            )
            self._last_point_color = pc
        viewer.log_points(
            point_name,
            self._keypoint_world,
            radii=self._radii,
            colors=self._point_colors,
            hidden=hidden,
        )
