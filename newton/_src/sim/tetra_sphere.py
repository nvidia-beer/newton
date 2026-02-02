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
Tetrahedral Sphere Generator

Generates a tetrahedral mesh for a sphere using:
1. Icosphere surface tessellation (geodesic subdivision)
2. Radial interior point distribution
3. Delaunay tetrahedralization

This provides a simple, self-contained soft body mesh without external files.
"""

import numpy as np
import time
from scipy.spatial import Delaunay, ConvexHull


class TetraSphere:
    """
    Generates a tetrahedral mesh for a sphere.
    
    Uses icosphere subdivision for uniform surface tessellation,
    then fills the interior with radial layers and tetrahedralizes.
    
    Parameters
    ----------
    radius : float
        Radius of the sphere (default: 1.0)
    subdivisions : int
        Number of icosphere subdivisions (default: 2)
        - 0: 12 vertices (icosahedron)
        - 1: 42 vertices
        - 2: 162 vertices
        - 3: 642 vertices
        - 4: 2562 vertices
    interior_layers : int
        Number of interior radial layers (default: 2)
        More layers = more tetrahedra, better volume representation
    
    Attributes
    ----------
    vertices : np.ndarray
        Vertex positions, shape (N, 3)
    tetrahedra : np.ndarray
        Tetrahedron indices, shape (M, 4)
    surface_triangles : np.ndarray
        Surface triangle indices for rendering
    
    Example
    -------
    >>> sphere = TetraSphere(radius=0.5, subdivisions=2, interior_layers=2)
    >>> print(f"Vertices: {len(sphere.vertices)}, Tetrahedra: {len(sphere.tetrahedra)}")
    """
    
    def __init__(self, radius: float = 1.0, subdivisions: int = 2, interior_layers: int = 2, verbose: bool = True):
        self.radius = radius
        self.subdivisions = subdivisions
        self.interior_layers = interior_layers
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
        
        # Step 1: Create icosphere surface
        self._log(f"[1/4] Creating icosphere (subdivisions={self.subdivisions})...")
        t0 = time.time()
        surface_vertices, surface_faces = self._create_icosphere()
        self._log(f"      Done: {len(surface_vertices)} surface vertices, "
                  f"{len(surface_faces)} triangles ({time.time()-t0:.2f}s)")
        
        # Step 2: Add interior points (radial layers + center)
        self._log(f"[2/4] Adding interior points (layers={self.interior_layers})...")
        t0 = time.time()
        all_vertices = self._add_interior_points(surface_vertices)
        self._log(f"      Done: {len(all_vertices)} total vertices ({time.time()-t0:.2f}s)")
        
        # Step 3: Tetrahedralize
        self._log(f"[3/4] Delaunay tetrahedralization...")
        t0 = time.time()
        tetrahedra = self._tetrahedralize(all_vertices)
        self._log(f"      Done: {len(tetrahedra)} tetrahedra ({time.time()-t0:.2f}s)")
        
        # Step 4: Filter tetrahedra outside sphere and fix orientations
        self._log(f"[4/4] Filtering and fixing tetrahedra...")
        t0 = time.time()
        tetrahedra = self._filter_tetrahedra(all_vertices, tetrahedra)
        self._log(f"      Done: {len(tetrahedra)} valid tetrahedra ({time.time()-t0:.2f}s)")
        
        self._log(f"Mesh generation complete in {time.time()-total_start:.2f}s")
        
        return all_vertices, tetrahedra, surface_faces
    
    def _create_icosphere(self):
        """
        Create an icosphere by subdividing an icosahedron.
        
        Returns vertices and triangular faces.
        """
        # Golden ratio
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        
        # Icosahedron vertices (12 vertices)
        vertices = np.array([
            [-1,  phi, 0],
            [ 1,  phi, 0],
            [-1, -phi, 0],
            [ 1, -phi, 0],
            [0, -1,  phi],
            [0,  1,  phi],
            [0, -1, -phi],
            [0,  1, -phi],
            [ phi, 0, -1],
            [ phi, 0,  1],
            [-phi, 0, -1],
            [-phi, 0,  1],
        ], dtype=np.float64)
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Icosahedron faces (20 triangles)
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)
        
        # Subdivide
        for level in range(self.subdivisions):
            t0 = time.time()
            vertices, faces = self._subdivide(vertices, faces)
            if self.verbose:
                print(f"        Level {level+1}/{self.subdivisions}: "
                      f"{len(vertices)} verts, {len(faces)} faces ({time.time()-t0:.2f}s)", flush=True)
        
        # Scale to desired radius
        vertices = vertices * self.radius
        
        return vertices, faces
    
    def _subdivide(self, vertices, faces):
        """
        Subdivide each triangle into 4 triangles.
        
        Each edge midpoint is projected onto the unit sphere.
        """
        # Edge midpoint cache to avoid duplicates
        edge_cache = {}
        new_faces = []
        vertices = list(vertices)
        
        def get_midpoint(i1, i2):
            """Get or create midpoint vertex index."""
            key = (min(i1, i2), max(i1, i2))
            if key in edge_cache:
                return edge_cache[key]
            
            # Create midpoint
            p1, p2 = np.array(vertices[i1]), np.array(vertices[i2])
            mid = (p1 + p2) / 2.0
            
            # Project to unit sphere
            mid = mid / np.linalg.norm(mid)
            
            idx = len(vertices)
            vertices.append(mid)
            edge_cache[key] = idx
            return idx
        
        for tri in faces:
            v0, v1, v2 = tri
            
            # Get midpoints
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            
            # Create 4 new triangles
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])
        
        return np.array(vertices), np.array(new_faces, dtype=np.int32)
    
    def _add_interior_points(self, surface_vertices):
        """
        Add interior points for volumetric tetrahedralization.
        
        Creates radial layers from center to surface.
        """
        all_vertices = [surface_vertices]
        
        # Add center point
        center = np.array([[0.0, 0.0, 0.0]])
        all_vertices.append(center)
        
        # Add radial layers
        if self.interior_layers > 0:
            for layer in range(1, self.interior_layers + 1):
                # Fraction of radius for this layer
                r_fraction = layer / (self.interior_layers + 1)
                
                # Scale surface vertices to this radius
                # Use fewer points for inner layers
                layer_vertices = surface_vertices * r_fraction
                
                # Optionally subsample inner layers for efficiency
                if layer < self.interior_layers:
                    # Keep every Nth vertex for inner layers
                    step = max(1, 2 ** (self.interior_layers - layer))
                    layer_vertices = layer_vertices[::step]
                
                all_vertices.append(layer_vertices)
        
        return np.vstack(all_vertices)
    
    def _tetrahedralize(self, vertices):
        """
        Create tetrahedra using Delaunay triangulation.
        """
        try:
            delaunay = Delaunay(vertices)
            return delaunay.simplices.astype(np.int32)
        except Exception as e:
            print(f"Delaunay failed: {e}, using fallback")
            return self._fallback_tetrahedralize(vertices)
    
    def _fallback_tetrahedralize(self, vertices):
        """
        Fallback tetrahedralization using convex hull + center.
        """
        # Get convex hull triangles
        hull = ConvexHull(vertices)
        
        # Find center point index (should be first interior point)
        center_idx = len(vertices) - 1  # Assuming center was added last
        
        # Create tetrahedra by connecting hull faces to center
        tetrahedra = []
        for simplex in hull.simplices:
            tet = list(simplex) + [center_idx]
            tetrahedra.append(tet)
        
        return np.array(tetrahedra, dtype=np.int32)
    
    def _filter_tetrahedra(self, vertices, tetrahedra):
        """
        Filter and fix tetrahedra orientation.
        
        - Removes tetrahedra with centroids outside the sphere
        - Removes degenerate (zero volume) tetrahedra  
        - Fixes inverted tetrahedra by swapping vertex order
        """
        valid_tets = []
        inverted_count = 0
        
        for tet in tetrahedra:
            tet = list(tet)  # Make mutable copy
            
            # Get tetrahedron vertices
            v = vertices[tet]
            
            # Compute centroid
            centroid = v.mean(axis=0)
            
            # Check if centroid is inside sphere (with small margin)
            dist = np.linalg.norm(centroid)
            if dist <= self.radius * 1.01:
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
    
    def validate_mesh(self):
        """
        Validate mesh for FEM simulation.
        
        Checks:
        - All tetrahedra have positive volume (correct orientation)
        - No degenerate tetrahedra
        - Mesh is watertight (optional)
        
        Returns
        -------
        dict with validation results
        """
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
        """
        Get mesh data in format suitable for Newton ModelBuilder.
        
        Returns
        -------
        dict with:
            - vertices: np.ndarray, shape (N, 3), float32
            - tetrahedra: np.ndarray, shape (M, 4), int32
            - indices: np.ndarray, flattened tetrahedra indices
        """
        return {
            'vertices': self.vertices.astype(np.float32),
            'tetrahedra': self.tetrahedra.astype(np.int32),
            'indices': self.tetrahedra.flatten().astype(np.int32),
        }
    
    def info(self):
        """Print mesh statistics."""
        print(f"TetraSphere Mesh:")
        print(f"  Radius: {self.radius}")
        print(f"  Subdivisions: {self.subdivisions}")
        print(f"  Interior layers: {self.interior_layers}")
        print(f"  Vertices: {len(self.vertices)}")
        print(f"  Tetrahedra: {len(self.tetrahedra)}")
        print(f"  Surface triangles: {len(self.surface_triangles)}")
        
        # Compute volume
        total_volume = sum(abs(self._tet_volume(self.vertices[t])) for t in self.tetrahedra)
        expected_volume = (4/3) * np.pi * self.radius**3
        print(f"  Mesh volume: {total_volume:.4f}")
        print(f"  Expected volume: {expected_volume:.4f}")
        print(f"  Volume accuracy: {100 * total_volume / expected_volume:.1f}%")
        
        # Validate mesh
        validation = self.validate_mesh()
        print(f"  Mesh valid: {validation['valid']}")
        print(f"  Positive volume tets: {validation['positive_volume']}")
        if validation['negative_volume'] > 0:
            print(f"  WARNING: Inverted tets: {validation['negative_volume']}")
        if validation['degenerate'] > 0:
            print(f"  WARNING: Degenerate tets: {validation['degenerate']}")


def create_tetra_sphere(radius: float = 1.0, subdivisions: int = 2, interior_layers: int = 2, verbose: bool = True):
    """
    Convenience function to create a tetrahedral sphere mesh.
    
    Parameters
    ----------
    radius : float
        Sphere radius (default: 1.0)
    subdivisions : int
        Icosphere subdivision level (default: 2)
        Higher = more surface detail
    interior_layers : int
        Number of radial interior layers (default: 2)
        Higher = more volume tetrahedra
    verbose : bool
        Print progress info (default: True)
    
    Returns
    -------
    TetraSphere instance with vertices, tetrahedra, and surface_triangles
    """
    return TetraSphere(radius=radius, subdivisions=subdivisions, interior_layers=interior_layers, verbose=verbose)


if __name__ == "__main__":
    # Demo: create and display mesh info
    print("Creating TetraSphere meshes with different parameters:\n")
    
    for subdivs in [1, 2, 3]:
        for layers in [1, 2, 3]:
            sphere = TetraSphere(radius=1.0, subdivisions=subdivs, interior_layers=layers, verbose=False)
            print(f"subdivisions={subdivs}, layers={layers}: "
                  f"{len(sphere.vertices)} verts, {len(sphere.tetrahedra)} tets")
    
    print("\n" + "="*50)
    print("Detailed info for default sphere (subdivs=2, layers=2):")
    print("="*50)
    sphere = TetraSphere(radius=1.0, subdivisions=2, interior_layers=2, verbose=True)
    sphere.info()
