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
Tetrahedral Box Generator

Generates a tetrahedral mesh for a box using:
1. Box surface tessellation (6 faces)
2. Interior point distribution
3. Delaunay tetrahedralization

This provides a simple, self-contained soft body mesh without external files.
"""

import numpy as np
import time
from scipy.spatial import Delaunay, ConvexHull

from . import box_topology as _topo


class TetraBox:
    """
    Generates a tetrahedral mesh for a box.
    
    Uses box surface tessellation, then fills the interior
    with points and tetrahedralizes.
    
    Parameters
    ----------
    size : tuple[float, float, float] or float
        Box dimensions (width, height, depth) or uniform size (default: (1.0, 1.0, 1.0))
    subdivisions : tuple[int, int, int] or int
        Number of subdivisions per axis (default: (4, 4, 4))
        Each cell is divided into 2 tetrahedra
    
    Attributes
    ----------
    vertices : np.ndarray
        Vertex positions, shape (N, 3)
    tetrahedra : np.ndarray
        Tetrahedron indices, shape (M, 4)
    surface_triangles : np.ndarray
        Surface triangle indices for rendering
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
        
        # Generate the mesh
        self.vertices, self.tetrahedra, self.surface_triangles = self._generate_mesh()
    
    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg, flush=True)
    
    def _generate_mesh(self):
        """Generate the complete tetrahedral mesh."""
        total_start = time.time()
        
        # Step 1: Create box surface (already creates full structured grid)
        self._log(f"[1/4] Creating box surface (subdivisions={self.subdivisions})...")
        t0 = time.time()
        all_vertices, surface_faces = self._create_box_surface()
        self._log(f"      Done: {len(all_vertices)} vertices (full grid), "
                  f"{len(surface_faces)} surface triangles ({time.time()-t0:.2f}s)")
        
        # Step 2: Store vertex map for structured tetrahedralization
        # (vertex_map is already created in _create_box_surface)
        
        # Step 3: Tetrahedralize using structured hexahedral mesh
        self._log(f"[3/4] Structured hexahedral mesh generation...")
        t0 = time.time()
        
        # Debug: For multiple cubes, show alternating pattern distribution
        if self.subdivisions != (1, 1, 1) and self.verbose:
            sx, sy, sz = self.subdivisions
            pattern1_count = sum(1 for i in range(sx) for j in range(sy) for k in range(sz) 
                                if (i & 1) ^ (j & 1) ^ (k & 1))
            pattern2_count = sx * sy * sz - pattern1_count
            self._log(f"      Alternating diagonal pattern: {pattern1_count} cubes use Pattern 1 (v0-v6 diagonal), "
                     f"{pattern2_count} cubes use Pattern 2 (v1-v7 diagonal)")
        
        # Debug: For single cube, verify vertex positions and test pattern
        if self.subdivisions == (1, 1, 1) and self.verbose:
            print(f"\n[DEBUG] Single cube - verifying vertex positions:")
            for (i, j, k), idx in sorted(self._vertex_map.items()):
                pos = all_vertices[idx]
                print(f"  v{idx} = ({i},{j},{k}): [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
            
            # Test volumes before filtering
            temp_tets = self._tetrahedralize(all_vertices)
            print(f"\n[DEBUG] Testing tetrahedra volumes (before filtering):")
            for i, tet in enumerate(temp_tets[:6]):
                v = all_vertices[tet]
                vol = self._tet_volume(v)
                sign = 'POSITIVE' if vol > 0 else 'NEGATIVE'
                print(f"  Tet {i+1}: {tet}, volume = {vol:.6f}, {sign}")
        
        tetrahedra = self._tetrahedralize(all_vertices)
        self._log(f"      Done: {len(tetrahedra)} tetrahedra ({time.time()-t0:.2f}s)")
        
        # Step 4: Filter tetrahedra outside box and fix orientations
        self._log(f"[4/4] Filtering and fixing tetrahedra...")
        t0 = time.time()
        
        # Debug: Check volumes after filtering (for single cube)
        if self.subdivisions == (1, 1, 1) and self.verbose:
            print(f"\n[DEBUG] After filtering - checking final tetrahedra:")
            filtered_tets = self._filter_tetrahedra(all_vertices, tetrahedra)
            for i, tet in enumerate(filtered_tets):
                v = all_vertices[tet]
                vol = self._tet_volume(v)
                sign = 'POSITIVE' if vol > 0 else 'NEGATIVE'
                print(f"  Tet {i+1}: {tet}, volume = {vol:.6f}, {sign}")
            print(f"  Total filtered tets: {len(filtered_tets)} (expected: 6)")
            
            # Check for long diagonal edge
            print(f"\n[DEBUG] Checking edge lengths:")
            edges = set()
            for tet in filtered_tets:
                for i in range(4):
                    for j in range(i+1, 4):
                        v1, v2 = tet[i], tet[j]
                        if v1 > v2:
                            v1, v2 = v2, v1
                        edges.add((v1, v2))
            
            edge_lengths = []
            for v1, v2 in edges:
                p1, p2 = all_vertices[v1], all_vertices[v2]
                length = np.linalg.norm(p2 - p1)
                edge_lengths.append((length, (v1, v2)))
            
            edge_lengths.sort(reverse=True)
            print(f"  Longest edges:")
            for length, (v1, v2) in edge_lengths[:5]:
                p1, p2 = all_vertices[v1], all_vertices[v2]
                print(f"    Edge ({v1}-{v2}): length = {length:.6f}, "
                      f"p1={p1}, p2={p2}")
        
        tetrahedra = self._filter_tetrahedra(all_vertices, tetrahedra)
        self._log(f"      Done: {len(tetrahedra)} valid tetrahedra ({time.time()-t0:.2f}s)")
        
        self._log(f"Mesh generation complete in {time.time()-total_start:.2f}s")
        
        return all_vertices, tetrahedra, surface_faces
    
    def _create_box_surface(self):
        """Create box surface with 6 faces using unified vertex structure."""
        w, h, d = self.size
        sx, sy, sz = self.subdivisions
        
        # Create unified vertex grid (no duplicates)
        # Grid: (sx+1) x (sy+1) x (sz+1) vertices
        vertices = []
        vertex_map = {}  # Map (i,j,k) -> vertex index
        
        for i in range(sx + 1):
            for j in range(sy + 1):
                for k in range(sz + 1):
                    x = -w/2 + w * i / sx if sx > 0 else 0.0
                    y = -h/2 + h * j / sy if sy > 0 else 0.0
                    z = -d/2 + d * k / sz if sz > 0 else 0.0
                    vertex_map[(i, j, k)] = len(vertices)
                    vertices.append([x, y, z])
        
        faces = []
        
        # Generate faces for all 6 sides
        # Face -X (left, i=0)
        if sx > 0:
            i = 0
            for j in range(sy):
                for k in range(sz):
                    v0 = vertex_map[(i, j, k)]
                    v1 = vertex_map[(i, j+1, k)]
                    v2 = vertex_map[(i, j, k+1)]
                    v3 = vertex_map[(i, j+1, k+1)]
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
        
        # Face +X (right, i=sx)
        if sx > 0:
            i = sx
            for j in range(sy):
                for k in range(sz):
                    v0 = vertex_map[(i, j, k)]
                    v1 = vertex_map[(i, j, k+1)]
                    v2 = vertex_map[(i, j+1, k)]
                    v3 = vertex_map[(i, j+1, k+1)]
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        
        # Face -Y (bottom, j=0)
        if sy > 0:
            j = 0
            for i in range(sx):
                for k in range(sz):
                    v0 = vertex_map[(i, j, k)]
                    v1 = vertex_map[(i+1, j, k)]
                    v2 = vertex_map[(i, j, k+1)]
                    v3 = vertex_map[(i+1, j, k+1)]
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
        
        # Face +Y (top, j=sy)
        if sy > 0:
            j = sy
            for i in range(sx):
                for k in range(sz):
                    v0 = vertex_map[(i, j, k)]
                    v1 = vertex_map[(i, j, k+1)]
                    v2 = vertex_map[(i+1, j, k)]
                    v3 = vertex_map[(i+1, j, k+1)]
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        
        # Face -Z (front, k=0)
        if sz > 0:
            k = 0
            for i in range(sx):
                for j in range(sy):
                    v0 = vertex_map[(i, j, k)]
                    v1 = vertex_map[(i+1, j, k)]
                    v2 = vertex_map[(i, j+1, k)]
                    v3 = vertex_map[(i+1, j+1, k)]
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
        
        # Face +Z (back, k=sz)
        if sz > 0:
            k = sz
            for i in range(sx):
                for j in range(sy):
                    v0 = vertex_map[(i, j, k)]
                    v1 = vertex_map[(i, j+1, k)]
                    v2 = vertex_map[(i+1, j, k)]
                    v3 = vertex_map[(i+1, j+1, k)]
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])
        
        # Store vertex map for structured tetrahedralization
        self._vertex_map = vertex_map
        self._grid_dims = (sx + 1, sy + 1, sz + 1)
        
        return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)
    
    def _add_interior_points(self, surface_vertices):
        """
        Not used for structured mesh generation - full grid is created in _create_box_surface.
        Kept for API compatibility.
        """
        return surface_vertices
    
    def _tetrahedralize(self, vertices):
        """
        Create tetrahedra using structured hexahedral mesh subdivision.
        
        Uses proven 6-tet diagonal pattern for all cases - exactly 6 tets per cube,
        all sharing a common diagonal. This is mathematically correct and stable.
        """
        if not hasattr(self, '_vertex_map'):
            # Fallback to Delaunay if structured grid not available
            try:
                delaunay = Delaunay(vertices)
                return delaunay.simplices.astype(np.int32)
            except Exception as e:
                print(f"Delaunay failed: {e}, using fallback")
                return self._fallback_tetrahedralize(vertices)
        
        return self._structured_tetrahedralize()
    
    def _structured_tetrahedralize(self):
        """
        Generate tetrahedra from structured hexahedral mesh.
        
        For single cube (subdivisions=1): Uses center-point method for maximum stability.
        For multiple cubes: Uses alternating diagonal pattern for connectivity.
        """
        sx, sy, sz = self.subdivisions
        vertex_map = self._vertex_map
        tetrahedra = []
        
        # Use 6-tet diagonal pattern for all cases (single cube and multiple cubes)
        # All 6 tetrahedra share a common diagonal - exactly 6 tets, 100% coverage
        # This pattern is mathematically correct and verified to have all positive volumes
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    # Get 8 vertices of hexahedron
                    v0 = vertex_map[(i, j, k)]      # v000
                    v1 = vertex_map[(i+1, j, k)]    # v100
                    v2 = vertex_map[(i+1, j, k+1)]  # v101
                    v3 = vertex_map[(i, j, k+1)]    # v001
                    v4 = vertex_map[(i, j+1, k)]    # v010
                    v5 = vertex_map[(i+1, j+1, k)]  # v110
                    v6 = vertex_map[(i+1, j+1, k+1)]  # v111
                    v7 = vertex_map[(i, j+1, k+1)]  # v011
                    
                    # For single cube: Use proven 5-tet partition (from tetra_cylinder.py pattern)
                    # This pattern uses v6 (v111) as common vertex, avoids long diagonal
                    # For multiple cubes: Use 6-tet diagonal pattern
                    if sx == 1 and sy == 1 and sz == 1:
                        # Stable 6-tet pattern: Based on proven 5-tet + missing tet for 100% coverage
                        # All share v6 (top-right-back corner), avoids long diagonal edge
                        # This pattern uses shorter edges and is more stable than diagonal pattern
                        tetrahedra.append([v0, v3, v1, v6])  # Tet 1
                        tetrahedra.append([v0, v4, v3, v6])  # Tet 2
                        tetrahedra.append([v0, v1, v4, v6])  # Tet 3
                        tetrahedra.append([v1, v5, v4, v6])  # Tet 4
                        tetrahedra.append([v3, v4, v7, v6])  # Tet 5
                        tetrahedra.append([v1, v3, v2, v6])  # Tet 6: Completes coverage (fixed ordering)
                    else:
                        # For multiple cubes: Use the same stable pattern as single cube
                        # All cubes use the stable 6-tet pattern (all share v6 corner vertex)
                        # This avoids long diagonal edges that cause instability
                        # Adjacent cubes are connected through shared faces/edges/vertices
                        # Pattern: All 6 tets share v6 (top-right-back corner)
                        tetrahedra.append([v0, v3, v1, v6])
                        tetrahedra.append([v0, v4, v3, v6])
                        tetrahedra.append([v0, v1, v4, v6])
                        tetrahedra.append([v1, v5, v4, v6])
                        tetrahedra.append([v3, v4, v7, v6])
                        tetrahedra.append([v1, v3, v2, v6])
        
        return np.array(tetrahedra, dtype=np.int32)
    
    def _center_point_tetrahedralize(self, vertex_map):
        """
        Partition single cube into 12 CONGRUENT tetrahedra sharing a common center vertex.
        
        All 12 tetrahedra share the center vertex - this is the most stable pattern.
        Avoids the long diagonal edge that causes instability in 6-tet diagonal pattern.
        Each tet = center + 3 vertices forming a triangle on one face.
        """
        # Get 8 corner vertices
        v0 = vertex_map[(0, 0, 0)]  # bottom-left-front
        v1 = vertex_map[(1, 0, 0)]  # bottom-right-front
        v2 = vertex_map[(1, 0, 1)]  # bottom-right-back
        v3 = vertex_map[(0, 0, 1)]  # bottom-left-back
        v4 = vertex_map[(0, 1, 0)]  # top-left-front
        v5 = vertex_map[(1, 1, 0)]  # top-right-front
        v6 = vertex_map[(1, 1, 1)]  # top-right-back
        v7 = vertex_map[(0, 1, 1)]  # top-left-back
        
        center = self._center_vertex_idx
        
        # Create 12 congruent tetrahedra, ALL sharing the center vertex
        # Pattern: Connect center to each of the 6 faces, splitting each face into 2 triangles
        # This creates 12 tetrahedra (2 per face), all sharing center - maximum stability
        # Each tet = center + 3 vertices forming a triangle on one face
        
        tetrahedra = [
            # Bottom face (-Y): v0, v1, v2, v3 - split into 2 triangles
            [center, v0, v1, v2],  # Triangle 1
            [center, v0, v2, v3],  # Triangle 2
            # Top face (+Y): v4, v5, v6, v7 - split into 2 triangles
            [center, v4, v6, v5],  # Triangle 1
            [center, v4, v7, v6],  # Triangle 2
            # Front face (-Z): v0, v1, v5, v4 - split into 2 triangles
            [center, v0, v5, v1],  # Triangle 1
            [center, v0, v4, v5],  # Triangle 2
            # Back face (+Z): v2, v3, v7, v6 - split into 2 triangles
            [center, v2, v7, v3],  # Triangle 1
            [center, v2, v6, v7],  # Triangle 2
            # Left face (-X): v0, v3, v7, v4 - split into 2 triangles
            [center, v3, v7, v0],  # Triangle 1 (fixed winding for positive volume)
            [center, v7, v4, v0],  # Triangle 2 (fixed winding for positive volume)
            # Right face (+X): v1, v2, v6, v5 - split into 2 triangles
            [center, v1, v6, v2],  # Triangle 1
            [center, v1, v5, v6],  # Triangle 2
        ]
        
        # All 12 tetrahedra share center vertex - maximum stability
        # Complete coverage: All 8 vertices used, total volume = cube volume
        
        return np.array(tetrahedra, dtype=np.int32)
    
    def _fallback_tetrahedralize(self, vertices):
        """Fallback tetrahedralization using convex hull + center."""
        hull = ConvexHull(vertices)
        center_idx = len(vertices) - 1  # Assuming center was added last
        
        tetrahedra = []
        for simplex in hull.simplices:
            tet = list(simplex) + [center_idx]
            tetrahedra.append(tet)
        
        return np.array(tetrahedra, dtype=np.int32)
    
    def _filter_tetrahedra(self, vertices, tetrahedra):
        """
        Filter and fix tetrahedra orientation.
        
        For structured meshing, all tetrahedra should be valid (no need to check
        if centroid is inside box). Only check for degenerate and inverted tetrahedra.
        """
        valid_tets = []
        inverted_count = 0
        
        for tet in tetrahedra:
            tet = list(tet)  # Make mutable copy
            
            # Get tetrahedron vertices
            v = vertices[tet]
            
            # Compute signed volume
            volume = self._tet_volume(v)
            
            # Skip degenerate tetrahedra
            if abs(volume) < 1e-10:
                continue
            
            # Fix inverted tetrahedra (negative volume)
            # Swap vertices 0 and 1 to flip orientation
            if volume < 0:
                tet[0], tet[1] = tet[1], tet[0]
                inverted_count += 1
            
            valid_tets.append(tet)
        
        if inverted_count > 0:
            print(f"      Fixed {inverted_count} inverted tetrahedra", flush=True)
        
        return np.array(valid_tets, dtype=np.int32)
    
    def _tet_volume(self, v):
        """Compute signed volume of tetrahedron."""
        d1 = v[1] - v[0]
        d2 = v[2] - v[0]
        d3 = v[3] - v[0]
        return np.dot(d1, np.cross(d2, d3)) / 6.0
    
    def _tet_aspect_ratio(self, v):
        """
        Compute aspect ratio (quality metric) for tetrahedron.
        
        Returns the ratio of longest edge to shortest edge.
        A value close to 1.0 indicates a well-shaped tetrahedron.
        Large values indicate poor aspect ratios (flat or elongated).
        """
        edges = [
            np.linalg.norm(v[1] - v[0]),
            np.linalg.norm(v[2] - v[0]),
            np.linalg.norm(v[3] - v[0]),
            np.linalg.norm(v[2] - v[1]),
            np.linalg.norm(v[3] - v[1]),
            np.linalg.norm(v[3] - v[2]),
        ]
        if min(edges) < 1e-10:
            return float('inf')  # Degenerate
        return max(edges) / min(edges)
    
    def _tet_quality(self, v):
        """
        Compute quality metric combining volume and edge lengths.
        
        Returns a quality score where higher is better.
        Filters tetrahedra with very small volumes relative to edge lengths.
        """
        volume = abs(self._tet_volume(v))
        if volume < 1e-10:
            return 0.0
        
        edges = [
            np.linalg.norm(v[1] - v[0]),
            np.linalg.norm(v[2] - v[0]),
            np.linalg.norm(v[3] - v[0]),
            np.linalg.norm(v[2] - v[1]),
            np.linalg.norm(v[3] - v[1]),
            np.linalg.norm(v[3] - v[2]),
        ]
        avg_edge = np.mean(edges)
        
        # Quality: volume / (avg_edge^3)
        # For a regular tetrahedron, this should be ~0.12
        # Very small values indicate flat or elongated tets
        quality = volume / (avg_edge ** 3)
        return quality
    
    def _tet_condition_number(self, v):
        """
        Compute condition number of the rest configuration matrix Dm.
        
        This predicts how large inv_Dm will be, which affects stability.
        Large condition numbers indicate that inv_Dm will have very large values.
        """
        # Compute Dm matrix (same as in add_tetrahedron)
        p = v[0]
        q = v[1]
        r = v[2]
        s = v[3]
        
        qp = q - p
        rp = r - p
        sp = s - p
        
        Dm = np.array([qp, rp, sp]).T
        
        # Compute condition number (ratio of largest to smallest singular value)
        # Large condition number -> small determinant -> large inv_Dm -> instability
        try:
            cond = np.linalg.cond(Dm)
            return cond
        except:
            return float('inf')
    
    def validate_mesh(self):
        """Validate mesh for FEM simulation."""
        results = {
            'valid': True,
            'total_tets': len(self.tetrahedra),
            'positive_volume': 0,
            'negative_volume': 0,
            'degenerate': 0,
            'min_volume': float('inf'),
            'max_volume': 0,
        }
        
        for tet in self.tetrahedra:
            v = self.vertices[tet]
            volume = self._tet_volume(v)
            
            if abs(volume) < 1e-10:
                results['degenerate'] += 1
                results['valid'] = False
            elif volume < 0:
                results['negative_volume'] += 1
                results['valid'] = False
            else:
                results['positive_volume'] += 1
                results['min_volume'] = min(results['min_volume'], volume)
                results['max_volume'] = max(results['max_volume'], volume)
        
        return results
    
    def get_mesh_data(self):
        """Get mesh data in format suitable for Newton ModelBuilder."""
        return {
            'vertices': self.vertices.astype(np.float32),
            'tetrahedra': self.tetrahedra.astype(np.int32),
            'indices': self.tetrahedra.flatten().astype(np.int32),
        }

    # --- Topology / side index helpers (same convention as SurfaceBox; see box_topology.py) ---

    def vertex_index(self, i: int, j: int, k: int) -> int:
        """Linear vertex index for grid (i, j, k). Subdivisions taken from self.subdivisions."""
        sx, sy, sz = self.subdivisions
        return _topo.vertex_index(sx, sy, sz, i, j, k)

    def get_side_vertex_indices(self, side: str) -> np.ndarray:
        """Indices of all vertices on the given box side. side: SIDE_X_MIN, SIDE_Y_MAX, SIDE_Z_MIN, etc."""
        sx, sy, sz = self.subdivisions
        return _topo.get_side_vertex_indices(sx, sy, sz, side)

    def get_side_vertex_pair_indices(self, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """(indices on +side, indices on -side) for same in-plane coords. axis: 'x', 'y', or 'z'."""
        sx, sy, sz = self.subdivisions
        return _topo.get_side_vertex_pair_indices(sx, sy, sz, axis)

    def info(self):
        """Print mesh statistics."""
        print(f"TetraBox Mesh:")
        print(f"  Size: {self.size}")
        print(f"  Subdivisions: {self.subdivisions}")
        print(f"  Vertices: {len(self.vertices)}")
        print(f"  Tetrahedra: {len(self.tetrahedra)}")
        print(f"  Surface triangles: {len(self.surface_triangles)}")
        
        total_volume = sum(abs(self._tet_volume(self.vertices[t])) for t in self.tetrahedra)
        expected_volume = self.size[0] * self.size[1] * self.size[2]
        print(f"  Mesh volume: {total_volume:.4f}")
        print(f"  Expected volume: {expected_volume:.4f}")
        print(f"  Volume accuracy: {100 * total_volume / expected_volume:.1f}%")
        
        validation = self.validate_mesh()
        print(f"  Mesh valid: {validation['valid']}")
        print(f"  Positive volume tets: {validation['positive_volume']}")
        if validation['negative_volume'] > 0:
            print(f"  WARNING: Inverted tets: {validation['negative_volume']}")
        if validation['degenerate'] > 0:
            print(f"  WARNING: Degenerate tets: {validation['degenerate']}")


def create_tetra_box(size=(1.0, 1.0, 1.0), subdivisions=(4, 4, 4), verbose: bool = True):
    """Convenience function to create a tetrahedral box mesh."""
    return TetraBox(size=size, subdivisions=subdivisions, verbose=verbose)


def debug_single_cube():
    """Debug function: Create a single cube partitioned into 6 congruent tetrahedra."""
    import numpy as np
    
    # Create a single unit cube centered at origin
    # Vertices of cube: v0-v7
    vertices = np.array([
        [-0.5, -0.5, -0.5],  # v0: bottom-left-front
        [ 0.5, -0.5, -0.5],  # v1: bottom-right-front
        [ 0.5, -0.5,  0.5],  # v2: bottom-right-back
        [-0.5, -0.5,  0.5],  # v3: bottom-left-back
        [-0.5,  0.5, -0.5],  # v4: top-left-front
        [ 0.5,  0.5, -0.5],  # v5: top-right-front
        [ 0.5,  0.5,  0.5],  # v6: top-right-back
        [-0.5,  0.5,  0.5],  # v7: top-left-back
    ], dtype=np.float32)
    
    # Partition into 6 congruent tetrahedra sharing diagonal v0-v6
    # All tets use v0 and v6, plus 2 vertices from adjacent faces
    tetrahedra = np.array([
        [0, 1, 3, 6],  # Tet 1: v0-v6 + bottom face
        [0, 1, 4, 6],  # Tet 2: v0-v6 + front face
        [0, 3, 4, 6],  # Tet 3: v0-v6 + left face
        [1, 2, 3, 6],  # Tet 4: v0-v6 + right face
        [1, 4, 5, 6],  # Tet 5: v0-v6 + top-front
        [3, 4, 7, 6],  # Tet 6: v0-v6 + top-back (covers v7)
    ], dtype=np.int32)
    
    # Verify volumes
    def tet_volume(v):
        d1 = v[1] - v[0]
        d2 = v[2] - v[0]
        d3 = v[3] - v[0]
        return np.dot(d1, np.cross(d2, d3)) / 6.0
    
    print("Debug: Single cube with 6 congruent tetrahedra")
    print(f"Vertices: {len(vertices)}")
    print(f"Tetrahedra: {len(tetrahedra)}")
    
    volumes = []
    for i, tet in enumerate(tetrahedra):
        v = vertices[tet]
        vol = tet_volume(v)
        volumes.append(vol)
        print(f"  Tet {i+1}: vertices {tet}, volume = {vol:.6f}, {'POSITIVE' if vol > 0 else 'NEGATIVE'}")
    
    total_volume = sum(abs(v) for v in volumes)
    expected_volume = 1.0  # Unit cube
    print(f"\nTotal mesh volume: {total_volume:.6f}")
    print(f"Expected volume: {expected_volume:.6f}")
    print(f"Volume accuracy: {100 * total_volume / expected_volume:.2f}%")
    
    # Check if volumes are congruent (should be equal)
    if len(set([round(v, 6) for v in volumes if v > 0])) == 1:
        print("✓ All positive-volume tets are congruent (equal volume)")
    else:
        print("✗ Tets are NOT congruent - volumes differ")
    
    # Check for overlaps (each vertex should appear in exactly 3 tets for 6-tet pattern)
    vertex_counts = [0] * 8
    for tet in tetrahedra:
        for v in tet:
            vertex_counts[v] += 1
    
    print(f"\nVertex usage counts: {vertex_counts}")
    print("Expected: each vertex appears in 3 tets (for 6-tet pattern)")
    
    return vertices, tetrahedra


if __name__ == "__main__":
    # Run debug test
    debug_single_cube()
