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
Surface Box Generator

Generates a complete surface-only (hollow) box mesh with all 6 faces.
Uses the same vertex grid layout as TetraBox for consistency (same size/subdivisions).

Face and vertex index convention
--------------------------------
- size = (w, h, d): extent along x, y, z. Vertices in local frame: x in [-w/2, w/2], etc.
- subdivisions = (sx, sy, sz). Vertex grid (i, j, k) with i in [0, sx], j in [0, sy], k in [0, sz].
- Linear index (loop order k, j, i — k outer, i inner):  idx = i + (sx+1)*j + (sx+1)*(sy+1)*k.
- Local position: x = -w/2 + w*i/sx, y = -h/2 + h*j/sy, z = -d/2 + d*k/sz.

Faces (outward normal):
  -X (i=0), +X (i=sx), -Y (j=0), +Y (j=sy), -Z (k=0), +Z (k=sz).
Same (i,k) on -Y and +Y gives the same (x,z); only y differs. So pairing (i,0,k) with (i,sy,k)
connects the same physical (x,z) on the two Y faces.
"""

import numpy as np

from . import box_topology as _topo


class SurfaceBox:
    """
    Generates a surface-only (hollow) box mesh with all 6 faces.

    Same dimensions and subdivision layout as TetraBox for visual consistency.
    Vertices and triangles are generated directly to ensure no faces are missing.

    Parameters
    ----------
    size : tuple[float, float, float] or float
        Box dimensions (width, height, depth)
    subdivisions : tuple[int, int, int] or int
        Number of subdivisions per axis

    Attributes
    ----------
    vertices : np.ndarray
        Vertex positions, shape (N, 3)
    surface_triangles : np.ndarray
        Surface triangle indices, shape (M, 3), normals outward
    """

    def __init__(self, size=(1.0, 1.0, 1.0), subdivisions=(4, 4, 4), verbose: bool = True):
        if isinstance(size, (int, float)):
            self.size = (float(size), float(size), float(size))
        else:
            self.size = tuple(float(s) for s in size)

        if isinstance(subdivisions, int):
            self.subdivisions = (subdivisions, subdivisions, subdivisions)
        else:
            self.subdivisions = tuple(int(s) for s in subdivisions)

        self.verbose = verbose
        self.vertices, self.surface_triangles = self._create_box_surface()

        if self.verbose:
            print(
                f"SurfaceBox: {len(self.vertices)} vertices, "
                f"{len(self.surface_triangles)} surface triangles (6 faces)",
                flush=True,
            )

    def _create_box_surface(self):
        """Create complete box surface with all 6 faces, outward normals."""
        w, h, d = self.size
        sx, sy, sz = self.subdivisions
        sx, sy, sz = max(1, sx), max(1, sy), max(1, sz)

        vertices = []
        vertex_map = {}
        # Loop order k, j, i so linear index = i + (sx+1)*j + (sx+1)*(sy+1)*k (matches box_topology)
        for k in range(sz + 1):
            for j in range(sy + 1):
                for i in range(sx + 1):
                    x = -w / 2 + w * i / sx if sx > 0 else 0.0
                    y = -h / 2 + h * j / sy if sy > 0 else 0.0
                    z = -d / 2 + d * k / sz if sz > 0 else 0.0
                    vertex_map[(i, j, k)] = len(vertices)
                    vertices.append([x, y, z])

        faces = []

        # -X (i=0): CCW when viewed from outside (-x)
        for j in range(sy):
            for k in range(sz):
                v0 = vertex_map[(0, j, k)]
                v1 = vertex_map[(0, j, k + 1)]
                v2 = vertex_map[(0, j + 1, k + 1)]
                v3 = vertex_map[(0, j + 1, k)]
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

        # +X (i=sx)
        for j in range(sy):
            for k in range(sz):
                v0 = vertex_map[(sx, j, k)]
                v1 = vertex_map[(sx, j + 1, k)]
                v2 = vertex_map[(sx, j + 1, k + 1)]
                v3 = vertex_map[(sx, j, k + 1)]
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

        # -Y (j=0)
        for i in range(sx):
            for k in range(sz):
                v0 = vertex_map[(i, 0, k)]
                v1 = vertex_map[(i + 1, 0, k)]
                v2 = vertex_map[(i + 1, 0, k + 1)]
                v3 = vertex_map[(i, 0, k + 1)]
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

        # +Y (j=sy)
        for i in range(sx):
            for k in range(sz):
                v0 = vertex_map[(i, sy, k)]
                v1 = vertex_map[(i, sy, k + 1)]
                v2 = vertex_map[(i + 1, sy, k + 1)]
                v3 = vertex_map[(i + 1, sy, k)]
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

        # -Z (k=0)
        for i in range(sx):
            for j in range(sy):
                v0 = vertex_map[(i, j, 0)]
                v1 = vertex_map[(i, j + 1, 0)]
                v2 = vertex_map[(i + 1, j + 1, 0)]
                v3 = vertex_map[(i + 1, j, 0)]
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

        # +Z (k=sz)
        for i in range(sx):
            for j in range(sy):
                v0 = vertex_map[(i, j, sz)]
                v1 = vertex_map[(i + 1, j, sz)]
                v2 = vertex_map[(i + 1, j + 1, sz)]
                v3 = vertex_map[(i, j + 1, sz)]
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

        return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)

    def get_mesh_data(self):
        """Get mesh data for rigid body collision and rendering."""
        return {
            "vertices": self.vertices.astype(np.float32),
            "indices": self.surface_triangles.flatten().astype(np.int32),
            "surface_triangles": self.surface_triangles.astype(np.int32),
        }

    # --- Topology / side index helpers (same convention as TetraBox; see box_topology.py) ---

    def vertex_index(self, i: int, j: int, k: int) -> int:
        """Linear vertex index for grid (i, j, k). Subdivisions taken from self.subdivisions."""
        sx, sy, sz = self.subdivisions
        return _topo.vertex_index(sx, sy, sz, i, j, k)

    def get_side_vertex_indices(self, side: str) -> np.ndarray:
        """Indices of all vertices on the given box side. side: SIDE_X_MIN, SIDE_Y_MAX, etc."""
        sx, sy, sz = self.subdivisions
        return _topo.get_side_vertex_indices(sx, sy, sz, side)

    def get_side_vertex_pair_indices(self, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """(indices on +side, indices on -side) for same in-plane coords. axis: 'x', 'y', or 'z'."""
        sx, sy, sz = self.subdivisions
        return _topo.get_side_vertex_pair_indices(sx, sy, sz, axis)

    def info(self):
        """Print mesh statistics."""
        print(f"SurfaceBox Mesh:")
        print(f"  Size: {self.size}")
        print(f"  Subdivisions: {self.subdivisions}")
        print(f"  Vertices: {len(self.vertices)}")
        print(f"  Surface triangles: {len(self.surface_triangles)} (6 faces)")


def create_surface_box(size=(1.0, 1.0, 1.0), subdivisions=(4, 4, 4), verbose: bool = True):
    """Convenience function to create a surface box mesh."""
    return SurfaceBox(size=size, subdivisions=subdivisions, verbose=verbose)
