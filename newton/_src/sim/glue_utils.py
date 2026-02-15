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
Glue spring builders using box topology (index-based rigid-rigid and soft-rigid).

Rigid-rigid glue may only attach same-axis opposing sides: X+ to X-, Y+ to Y-, or
Z+ to Z-. Pairing uses box_topology indices only (same (i,k) on +side and -side).
All rigid bodies must use the same mesh (same vertex layout) so indices refer to
the same local positions on every body.
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None

from . import box_topology


def proximity_threshold_for_side(
    size_0: float,
    size_1: float,
    subdiv_0: int,
    subdiv_1: int,
    *,
    factor: float = 2.5,
) -> float:
    """
    In-plane distance threshold for proximity pairing on any box side.

    For a side, the vertices lie in a plane; give the two in-plane extents and
    their subdivision counts. Returns factor * cell_diagonal.
    """
    if subdiv_0 <= 0 or subdiv_1 <= 0:
        return max(size_0, size_1)
    d0 = size_0 / subdiv_0
    d1 = size_1 / subdiv_1
    cell_diag = float(np.sqrt(d0 * d0 + d1 * d1))
    return factor * cell_diag


def compute_rigid_rigid_glue_pairs_by_proximity(
    body_q_np: np.ndarray,
    rigid_body_id_a: int,
    rigid_body_id_b: int,
    rigid_vertices: np.ndarray,
    rigid_sx: int,
    rigid_sy: int,
    rigid_sz: int,
    *,
    side_size_xz: tuple[float, float] | None = None,
    distance_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find rigid-rigid glue pairs by in-plane (x,z) proximity. Y+ on body A to Y- on body B.
    For validation: compare with topology pairs to verify mesh order matches.
    Returns (idx_a, idx_b, distances).
    """
    if wp is None:
        raise RuntimeError("warp (wp) is required for glue_utils")
    row_a = body_q_np[rigid_body_id_a]
    row_b = body_q_np[rigid_body_id_b]
    p_a = np.array([float(row_a[0]), float(row_a[1]), float(row_a[2])], dtype=np.float64)
    q_a = wp.quat(float(row_a[3]), float(row_a[4]), float(row_a[5]), float(row_a[6]))
    p_b = np.array([float(row_b[0]), float(row_b[1]), float(row_b[2])], dtype=np.float64)
    q_b = wp.quat(float(row_b[3]), float(row_b[4]), float(row_b[5]), float(row_b[6]))

    idx_plus = box_topology.get_side_vertex_indices(rigid_sx, rigid_sy, rigid_sz, box_topology.SIDE_Y_MAX)
    idx_minus = box_topology.get_side_vertex_indices(rigid_sx, rigid_sy, rigid_sz, box_topology.SIDE_Y_MIN)
    n_plus = len(idx_plus)
    n_minus = len(idx_minus)

    if side_size_xz is not None:
        size_x, size_z = side_size_xz
        distance_threshold = max(
            distance_threshold,
            proximity_threshold_for_side(size_x, size_z, rigid_sx, rigid_sz),
        )

    world_plus = np.empty((n_plus, 3), dtype=np.float64)
    for p in range(n_plus):
        v = rigid_vertices[int(idx_plus[p])]
        world_plus[p] = p_a + np.array(wp.quat_rotate(q_a, wp.vec3(float(v[0]), float(v[1]), float(v[2]))))
    world_minus = np.empty((n_minus, 3), dtype=np.float64)
    for m in range(n_minus):
        v = rigid_vertices[int(idx_minus[m])]
        world_minus[m] = p_b + np.array(wp.quat_rotate(q_b, wp.vec3(float(v[0]), float(v[1]), float(v[2]))))

    used_minus = np.zeros(n_minus, dtype=bool)
    idx_a_list = []
    idx_b_list = []
    dist_list = []
    for p in range(n_plus):
        best_m = -1
        best_d = float("inf")
        xz_p = world_plus[p][[0, 2]]
        for m in range(n_minus):
            if used_minus[m]:
                continue
            xz_m = world_minus[m][[0, 2]]
            d = float(np.linalg.norm(xz_p - xz_m))
            if d < best_d:
                best_d = d
                best_m = m
        if best_m >= 0 and best_d <= distance_threshold:
            used_minus[best_m] = True
            idx_a_list.append(int(idx_plus[p]))
            idx_b_list.append(int(idx_minus[best_m]))
            dist_list.append(float(np.linalg.norm(world_plus[p] - world_minus[best_m])))
    return (
        np.array(idx_a_list, dtype=np.int32),
        np.array(idx_b_list, dtype=np.int32),
        np.array(dist_list, dtype=np.float64),
    )


def validate_rigid_rigid_glue_indices(
    body_q_np: np.ndarray,
    rigid_body_ids: list[int],
    rigid_vertices: np.ndarray,
    rigid_sx: int,
    rigid_sy: int,
    rigid_sz: int,
    *,
    glue_axis: str = "y",
    distance_threshold: float = 0.5,
    verbose: bool = True,
) -> tuple[bool, dict]:
    """
    Validate that topology-based glue indices match proximity (mesh order correct).

    Compares pairs from box_topology.get_side_vertex_pair_indices with
    compute_rigid_rigid_glue_pairs_by_proximity for the first body pair.
    Proximity is Y+↔Y- only; use glue_axis='y' for meaningful comparison.
    Returns (ok, report).
    """
    idx_plus_arr, idx_minus_arr = box_topology.get_side_vertex_pair_indices(
        rigid_sx, rigid_sy, rigid_sz, glue_axis
    )
    n_pairs = len(idx_plus_arr)
    report = {
        "n_pairs": n_pairs,
        "mismatches": [],
        "order_ok": True,
        "set_ok": True,
        "first_index_pair": None,
        "first_prox_pair": None,
        "max_xz_dist": 0.0,
    }
    if n_pairs == 0 or len(rigid_body_ids) < 2:
        return True, report

    idx_a_prox, idx_b_prox, _ = compute_rigid_rigid_glue_pairs_by_proximity(
        body_q_np,
        int(rigid_body_ids[0]),
        int(rigid_body_ids[1]),
        rigid_vertices,
        rigid_sx,
        rigid_sy,
        rigid_sz,
        distance_threshold=distance_threshold,
    )
    if len(idx_a_prox) > 0:
        order = np.argsort(idx_a_prox)
        idx_a_prox = idx_a_prox[order]
        idx_b_prox = idx_b_prox[order]
    index_set = set(zip(idx_plus_arr.tolist(), idx_minus_arr.tolist()))
    prox_set = set(zip(idx_a_prox.tolist(), idx_b_prox.tolist()))
    if not (prox_set <= index_set):
        report["set_ok"] = False
        for pair in prox_set - index_set:
            report["mismatches"].append(("prox_not_in_index", pair[0], pair[1]))
    if index_set != prox_set:
        for pair in index_set - prox_set:
            report["mismatches"].append(("index_only", pair[0], pair[1]))
    n_prox = len(idx_a_prox)
    order_idx = np.argsort(idx_plus_arr)
    idx_plus_sorted = idx_plus_arr[order_idx]
    idx_minus_sorted = idx_minus_arr[order_idx]
    order_ok = n_prox <= n_pairs
    if order_ok and n_prox > 0:
        order_ok = (
            np.array_equal(idx_plus_sorted[:n_prox], idx_a_prox)
            and np.array_equal(idx_minus_sorted[:n_prox], idx_b_prox)
        )
    report["order_ok"] = order_ok
    report["first_index_pair"] = (int(idx_plus_arr[0]), int(idx_minus_arr[0]))
    report["first_prox_pair"] = (int(idx_a_prox[0]), int(idx_b_prox[0])) if n_prox else (None, None)

    id_a, id_b = int(rigid_body_ids[0]), int(rigid_body_ids[1])
    row_a = body_q_np[id_a]
    row_b = body_q_np[id_b]
    p_a = np.array([float(row_a[0]), float(row_a[1]), float(row_a[2])], dtype=np.float64)
    q_a = wp.quat(float(row_a[3]), float(row_a[4]), float(row_a[5]), float(row_a[6]))
    p_b = np.array([float(row_b[0]), float(row_b[1]), float(row_b[2])], dtype=np.float64)
    q_b = wp.quat(float(row_b[3]), float(row_b[4]), float(row_b[5]), float(row_b[6]))
    max_xz = 0.0
    for p in range(n_pairs):
        va = rigid_vertices[int(idx_plus_arr[p])]
        vb = rigid_vertices[int(idx_minus_arr[p])]
        wa = p_a + np.array(wp.quat_rotate(q_a, wp.vec3(float(va[0]), float(va[1]), float(va[2]))))
        wb = p_b + np.array(wp.quat_rotate(q_b, wp.vec3(float(vb[0]), float(vb[1]), float(vb[2]))))
        xz_dist = float(np.linalg.norm(wa[[0, 2]] - wb[[0, 2]]))
        max_xz = max(max_xz, xz_dist)
    report["max_xz_dist"] = max_xz

    ok = report["set_ok"] and report["order_ok"]
    if verbose and not ok:
        print("[glue_utils] validate_rigid_rigid_glue_indices: set_ok=%s order_ok=%s" % (report["set_ok"], report["order_ok"]), flush=True)
        print("  first index-based pair (idx_plus, idx_minus):", report["first_index_pair"], flush=True)
        if report["first_prox_pair"][0] is not None:
            print("  first proximity-based pair (idx_a, idx_b):", report["first_prox_pair"], flush=True)
        for m in report["mismatches"][:5]:
            print("  mismatch:", m, flush=True)
    if verbose and ok:
        print("[glue_utils] validate_rigid_rigid_glue_indices: OK (index matches proximity, max_xz=%.4f)" % max_xz, flush=True)
    return ok, report


def build_rigid_rigid_glue_by_index(
    body_q_np: np.ndarray,
    rigid_body_ids: list[int],
    rigid_vertices: np.ndarray,
    rigid_sx: int,
    rigid_sy: int,
    rigid_sz: int,
    *,
    glue_axis: str = "y",
    debug_mismatch: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build rigid-rigid glue pairs from box_topology indices only.

    All rigid bodies must use the same mesh. Pairing from
    box_topology.get_side_vertex_pair_indices(sx, sy, sz, glue_axis).
    Returns
    body_a, body_b, anchor_a, anchor_b, rest_lengths, idx_a, idx_b.
    """
    if wp is None:
        raise RuntimeError("warp (wp) is required for glue_utils")
    if glue_axis not in ("x", "y", "z"):
        raise ValueError(f"glue_axis must be 'x', 'y', or 'z', got {glue_axis!r}")
    min_rest = 1e-5
    body_a_list, body_b_list = [], []
    anchor_a_list, anchor_b_list = [], []
    rest_lengths_list = []
    idx_a_list, idx_b_list = [], []
    n_bodies = len(rigid_body_ids)
    n_verts = len(rigid_vertices)

    expected_verts = box_topology.num_vertices(rigid_sx, rigid_sy, rigid_sz)
    if n_verts != expected_verts:
        raise ValueError(
            f"glue_utils: rigid_vertices has {n_verts} vertices but topology (sx={rigid_sx}, sy={rigid_sy}, sz={rigid_sz}) "
            f"expects {expected_verts}. Use the same mesh for all rigid bodies."
        )

    idx_plus_arr, idx_minus_arr = box_topology.get_side_vertex_pair_indices(
        rigid_sx, rigid_sy, rigid_sz, glue_axis
    )
    n_pairs = len(idx_plus_arr)
    expected_springs = (n_bodies - 1) * n_pairs
    in_plane = (1, 2) if glue_axis == "x" else ((0, 2) if glue_axis == "y" else (0, 1))

    n_skipped = 0
    for ii in range(n_bodies - 1):
        bi = int(rigid_body_ids[ii])
        bj = int(rigid_body_ids[ii + 1])
        if body_q_np.shape[0] == n_bodies:
            row_i = body_q_np[ii]
            row_j = body_q_np[ii + 1]
        else:
            row_i = body_q_np[bi]
            row_j = body_q_np[bj]
        p_i = np.array([float(row_i[0]), float(row_i[1]), float(row_i[2])], dtype=np.float64)
        q_i = wp.quat(float(row_i[3]), float(row_i[4]), float(row_i[5]), float(row_i[6]))
        p_j = np.array([float(row_j[0]), float(row_j[1]), float(row_j[2])], dtype=np.float64)
        q_j = wp.quat(float(row_j[3]), float(row_j[4]), float(row_j[5]), float(row_j[6]))

        for p in range(n_pairs):
            idx_plus = int(idx_plus_arr[p])
            idx_minus = int(idx_minus_arr[p])
            if idx_plus >= n_verts or idx_minus >= n_verts:
                n_skipped += 1
                continue
            v_plus = rigid_vertices[idx_plus]
            v_minus = rigid_vertices[idx_minus]
            if debug_mismatch and ii == 0 and p == 0:
                d_in = np.array(
                    [float(v_plus[in_plane[0]]) - float(v_minus[in_plane[0]]),
                     float(v_plus[in_plane[1]]) - float(v_minus[in_plane[1]])],
                )
                if np.linalg.norm(d_in) > 1e-6:
                    print(
                        f"   [DEBUG] RR local in-plane ({glue_axis}) mismatch: v_plus=({float(v_plus[in_plane[0]]):.4f},{float(v_plus[in_plane[1]]):.4f}) "
                        f"v_minus=({float(v_minus[in_plane[0]]):.4f},{float(v_minus[in_plane[1]]):.4f}) | diff={np.linalg.norm(d_in):.6f}",
                        flush=True,
                    )
            pos_plus = p_i + np.array(wp.quat_rotate(q_i, wp.vec3(float(v_plus[0]), float(v_plus[1]), float(v_plus[2]))))
            pos_minus = p_j + np.array(wp.quat_rotate(q_j, wp.vec3(float(v_minus[0]), float(v_minus[1]), float(v_minus[2]))))
            rest_len = max(float(np.linalg.norm(pos_plus - pos_minus)), min_rest)
            body_a_list.append(bi)
            body_b_list.append(bj)
            anchor_a_list.append(v_plus)
            anchor_b_list.append(v_minus)
            rest_lengths_list.append(rest_len)
            idx_a_list.append(idx_plus)
            idx_b_list.append(idx_minus)

    n_actual = len(body_a_list)
    if n_skipped > 0:
        import warnings
        warnings.warn(
            f"glue_utils: skipped {n_skipped} pairs (vertex index >= n_verts={n_verts}). "
            f"Expected {expected_springs} springs, got {n_actual}.",
            UserWarning,
            stacklevel=2,
        )
    if debug_mismatch and expected_springs != n_actual and n_skipped == 0:
        import warnings
        warnings.warn(
            f"glue_utils: expected {expected_springs} springs (interfaces={n_bodies - 1}, pairs/interface={n_pairs}), got {n_actual}.",
            UserWarning,
            stacklevel=2,
        )

    return (
        np.array(body_a_list, dtype=np.int32),
        np.array(body_b_list, dtype=np.int32),
        np.array(anchor_a_list, dtype=np.float32).reshape(-1, 3) if anchor_a_list else np.empty((0, 3), dtype=np.float32),
        np.array(anchor_b_list, dtype=np.float32).reshape(-1, 3) if anchor_b_list else np.empty((0, 3), dtype=np.float32),
        np.array(rest_lengths_list, dtype=np.float32),
        np.array(idx_a_list, dtype=np.int32),
        np.array(idx_b_list, dtype=np.int32),
    )
