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
Box topology utilities for SurfaceBox and TetraBox.

Shared vertex grid convention (same in both meshes):
- Loop order: i, j, k. Linear index: idx = i + (sx+1)*j + (sx+1)*(sy+1)*k.
- Local position: x = -w/2 + w*i/sx, y = -h/2 + h*j/sy, z = -d/2 + d*k/sz.
- Sides (six planar sides of the box): x_min (i=0), x_max (i=sx), y_min (j=0), y_max (j=sy), z_min (k=0), z_max (k=sz).

Rigid-rigid glue pairing: get_side_vertex_pair_indices(axis) supports X+↔X-, Y+↔Y-, Z+↔Z-.
Same in-plane indices pair when axis-aligned (e.g. Y: same (i,k), X: same (j,k), Z: same (i,j)).
"""

from __future__ import annotations

import numpy as np
from typing import Iterator

# Side identifiers (six planar sides of the box; outward normal direction)
SIDE_X_MIN = "x_min"  # i=0
SIDE_X_MAX = "x_max"  # i=sx
SIDE_Y_MIN = "y_min"  # j=0
SIDE_Y_MAX = "y_max"  # j=sy
SIDE_Z_MIN = "z_min"  # k=0
SIDE_Z_MAX = "z_max"  # k=sz


def vertex_index(sx: int, sy: int, sz: int, i: int, j: int, k: int) -> int:
    """Linear vertex index for the box grid. Loop order: i, then j, then k."""
    return i + (sx + 1) * j + (sx + 1) * (sy + 1) * k


def num_vertices(sx: int, sy: int, sz: int) -> int:
    """Total number of vertices in the grid."""
    return (sx + 1) * (sy + 1) * (sz + 1)


def iter_side_vertices(
    sx: int, sy: int, sz: int, side: str
) -> Iterator[tuple[int, int, int, int]]:
    """
    Yield (i, j, k, linear_index) for every vertex on the given box side.

    side: one of SIDE_X_MIN, SIDE_X_MAX, SIDE_Y_MIN, SIDE_Y_MAX, SIDE_Z_MIN, SIDE_Z_MAX.
    """
    if side == SIDE_X_MIN:
        i = 0
        for j in range(sy + 1):
            for k in range(sz + 1):
                yield i, j, k, vertex_index(sx, sy, sz, i, j, k)
    elif side == SIDE_X_MAX:
        i = sx
        for j in range(sy + 1):
            for k in range(sz + 1):
                yield i, j, k, vertex_index(sx, sy, sz, i, j, k)
    elif side == SIDE_Y_MIN:
        j = 0
        for i in range(sx + 1):
            for k in range(sz + 1):
                yield i, j, k, vertex_index(sx, sy, sz, i, j, k)
    elif side == SIDE_Y_MAX:
        j = sy
        for i in range(sx + 1):
            for k in range(sz + 1):
                yield i, j, k, vertex_index(sx, sy, sz, i, j, k)
    elif side == SIDE_Z_MIN:
        k = 0
        for i in range(sx + 1):
            for j in range(sy + 1):
                yield i, j, k, vertex_index(sx, sy, sz, i, j, k)
    elif side == SIDE_Z_MAX:
        k = sz
        for i in range(sx + 1):
            for j in range(sy + 1):
                yield i, j, k, vertex_index(sx, sy, sz, i, j, k)
    else:
        raise ValueError(f"Unknown side: {side!r}. Use SIDE_* constants.")


def get_side_vertex_indices(sx: int, sy: int, sz: int, side: str) -> np.ndarray:
    """Return a 1D array of linear vertex indices for all vertices on the given box side."""
    indices = [idx for (_, _, _, idx) in iter_side_vertices(sx, sy, sz, side)]
    return np.array(indices, dtype=np.int32)


# Axis to (side_plus, side_minus) for rigid glue pairing (same-axis opposing sides)
_SIDE_PAIR: dict[str, tuple[str, str]] = {
    "x": (SIDE_X_MAX, SIDE_X_MIN),
    "y": (SIDE_Y_MAX, SIDE_Y_MIN),
    "z": (SIDE_Z_MAX, SIDE_Z_MIN),
}


def num_side_vertices(sx: int, sy: int, sz: int, axis: str) -> int:
    """Number of vertices on one side along the given axis. axis in ('x','y','z')."""
    if axis == "x":
        return (sy + 1) * (sz + 1)
    if axis == "y":
        return (sx + 1) * (sz + 1)
    if axis == "z":
        return (sx + 1) * (sy + 1)
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def get_side_vertex_pair_indices(
    sx: int, sy: int, sz: int, axis: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices_plus, indices_minus) for rigid-rigid glue on the given axis.

    Rigid glue only attaches same-axis opposing sides: X+↔X-, Y+↔Y-, Z+↔Z-.
    axis: 'x', 'y', or 'z'. When axis-aligned, same in-plane coords pair:
      - x: same (j,k); y: same (i,k); z: same (i,j).
    Order matches SurfaceBox vertex build (same loop order: i outer, then k for Y).
    """
    if axis not in _SIDE_PAIR:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    idx_plus_list: list[int] = []
    idx_minus_list: list[int] = []
    if axis == "y":
        # Explicit loop: i outer, k inner. Same (i,k) on +Y (j=sy) and -Y (j=0) -> same (x,z) in local.
        # Use same iter_side_vertices order for both sides so pairing is unambiguous.
        for (_, _, _, idx_p), (_, _, _, idx_m) in zip(
            iter_side_vertices(sx, sy, sz, SIDE_Y_MAX),
            iter_side_vertices(sx, sy, sz, SIDE_Y_MIN),
        ):
            idx_plus_list.append(idx_p)
            idx_minus_list.append(idx_m)
    else:
        side_plus, side_minus = _SIDE_PAIR[axis]
        for (_, _, _, idx_p), (_, _, _, idx_m) in zip(
            iter_side_vertices(sx, sy, sz, side_plus),
            iter_side_vertices(sx, sy, sz, side_minus),
        ):
            idx_plus_list.append(idx_p)
            idx_minus_list.append(idx_m)
    return np.array(idx_plus_list, dtype=np.int32), np.array(idx_minus_list, dtype=np.int32)


def num_y_side_vertices(sx: int, sy: int, sz: int) -> int:
    """Number of vertices on one Y side (same for +Y and -Y). Equals (sx+1)*(sz+1)."""
    return num_side_vertices(sx, sy, sz, "y")


def get_y_side_vertex_pair_indices(sx: int, sy: int, sz: int) -> tuple[np.ndarray, np.ndarray]:
    """(indices_y_max, indices_y_min) for rigid-rigid glue. Convenience for axis='y'."""
    return get_side_vertex_pair_indices(sx, sy, sz, "y")


def index_to_ijk(sx: int, sy: int, sz: int, idx: int) -> tuple[int, int, int]:
    """Convert linear index to grid (i, j, k)."""
    n_ij = (sx + 1) * (sy + 1)
    k = idx // n_ij
    r = idx % n_ij
    j = r // (sx + 1)
    i = r % (sx + 1)
    return i, j, k
