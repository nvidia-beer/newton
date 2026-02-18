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

import numpy as np

# Default number of segments for mesh generation
default_num_segments = 32


def create_sphere_mesh(
    radius=1.0,
    num_latitudes=default_num_segments,
    num_longitudes=default_num_segments,
    reverse_winding=False,
):
    """Create a sphere mesh with specified parameters.

    Generates vertices and triangle indices for a UV sphere using
    latitude/longitude parametrization. Each vertex contains position,
    normal, and UV coordinates.

    Args:
        radius (float): Sphere radius. Defaults to 1.0.
        num_latitudes (int): Number of horizontal divisions (latitude lines).
            Defaults to default_num_segments.
        num_longitudes (int): Number of vertical divisions (longitude lines).
            Defaults to default_num_segments.
        reverse_winding (bool): If True, reverses triangle winding order.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.
    """
    vertices = []
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta

            u = float(j) / num_longitudes
            v = float(i) / num_latitudes

            vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1

            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_ellipsoid_mesh(
    rx=1.0,
    ry=1.0,
    rz=1.0,
    num_latitudes=default_num_segments,
    num_longitudes=default_num_segments,
    reverse_winding=False,
):
    """Create an ellipsoid mesh with specified semi-axes.

    Generates vertices and triangle indices for a UV ellipsoid using
    latitude/longitude parametrization. Each vertex contains position,
    normal, and UV coordinates. The ellipsoid is centered at the origin
    with semi-axes along the X, Y, and Z axes.

    Args:
        rx (float): Semi-axis length along the X direction. Defaults to 1.0.
        ry (float): Semi-axis length along the Y direction. Defaults to 1.0.
        rz (float): Semi-axis length along the Z direction. Defaults to 1.0.
        num_latitudes (int): Number of horizontal divisions (latitude lines).
            Defaults to default_num_segments.
        num_longitudes (int): Number of vertical divisions (longitude lines).
            Defaults to default_num_segments.
        reverse_winding (bool): If True, reverses triangle winding order.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.
    """
    vertices = []
    indices = []

    for i in range(num_latitudes + 1):
        theta = i * np.pi / num_latitudes
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(num_longitudes + 1):
            phi = j * 2 * np.pi / num_longitudes
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            # Unit sphere coordinates
            ux = cos_phi * sin_theta
            uy = cos_theta
            uz = sin_phi * sin_theta

            # Scaled position for ellipsoid
            px = ux * rx
            py = uy * ry
            pz = uz * rz

            # Normal for ellipsoid: gradient of (x/rx)^2 + (y/ry)^2 + (z/rz)^2 = 1
            # n = (2x/rx^2, 2y/ry^2, 2z/rz^2), normalized
            nx = ux / rx
            ny = uy / ry
            nz = uz / rz
            n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
            if n_len > 1e-10:
                nx /= n_len
                ny /= n_len
                nz /= n_len

            u = float(j) / num_longitudes
            v = float(i) / num_latitudes

            vertices.append([px, py, pz, nx, ny, nz, u, v])

    for i in range(num_latitudes):
        for j in range(num_longitudes):
            first = i * (num_longitudes + 1) + j
            second = first + num_longitudes + 1

            if reverse_winding:
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
            else:
                indices.extend([first, first + 1, second, second, first + 1, second + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_capsule_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
    """Create a capsule (pill-shaped) mesh with hemispherical ends.

    Generates vertices and triangle indices for a capsule shape consisting
    of a cylinder with hemispherical caps at both ends.

    Args:
        radius (float): Radius of the capsule.
        half_height (float): Half the height of the cylindrical portion
            (distance from center to hemisphere start).
        up_axis (int): Axis along which the capsule extends (0=X, 1=Y, 2=Z).
            Defaults to 1 (Y-axis).
        segments (int): Number of segments for tessellation.
            Defaults to default_num_segments.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.
    """
    vertices = []
    indices = []

    x_dir, y_dir, z_dir = ((1, 2, 0), (2, 0, 1), (1, 2, 0))[up_axis]
    up_vector = np.zeros(3)
    up_vector[up_axis] = half_height

    for i in range(segments + 1):
        theta = i * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(segments + 1):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            z = cos_phi * sin_theta
            y = cos_theta
            x = sin_phi * sin_theta

            u = cos_theta * 0.5 + 0.5
            v = cos_phi * sin_theta * 0.5 + 0.5

            xyz = x, y, z
            x, y, z = xyz[x_dir], xyz[y_dir], xyz[z_dir]
            xyz = np.array((x, y, z), dtype=np.float32) * radius
            if j < segments // 2:
                xyz += up_vector
            else:
                xyz -= up_vector

            vertices.append([*xyz, x, y, z, u, v])

    nv = len(vertices)
    for i in range(segments + 1):
        for j in range(segments + 1):
            first = (i * (segments + 1) + j) % nv
            second = (first + segments + 1) % nv
            indices.extend([first, second, (first + 1) % nv, second, (second + 1) % nv, (first + 1) % nv])

    vertex_data = np.array(vertices, dtype=np.float32)
    index_data = np.array(indices, dtype=np.uint32)

    return vertex_data, index_data


def create_cone_mesh(radius, half_height, up_axis=1, segments=default_num_segments):
    """Create a cone mesh with circular base and pointed top.

    Generates vertices and triangle indices for a cone shape. Implemented as
    a cylinder with zero top radius to ensure correct normal calculations.

    Args:
        radius (float): Radius of the cone's circular base.
        half_height (float): Half the total height of the cone
            (distance from center to tip/base).
        up_axis (int): Axis along which the cone extends (0=X, 1=Y, 2=Z).
            Defaults to 1 (Y-axis).
        segments (int): Number of segments around the circumference.
            Defaults to default_num_segments.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.
    """
    # render it as a cylinder with zero top radius so we get correct normals on the sides
    return create_cylinder_mesh(radius, half_height, up_axis, segments, 0.0)


def create_cylinder_mesh(radius, half_height, up_axis=1, segments=default_num_segments, top_radius=None):
    """Create a cylinder or truncated cone mesh.

    Generates vertices and triangle indices for a cylindrical shape with
    optional different top and bottom radii (creating a truncated cone when
    different). Includes circular caps at both ends.

    Args:
        radius (float): Radius of the bottom circular face.
        half_height (float): Half the total height of the cylinder
            (distance from center to top/bottom face).
        up_axis (int): Axis along which the cylinder extends (0=X, 1=Y, 2=Z).
            Defaults to 1 (Y-axis).
        segments (int): Number of segments around the circumference.
            Defaults to default_num_segments.
        top_radius (float, optional): Radius of the top circular face.
            If None, uses same radius as bottom (true cylinder).
            If different, creates a truncated cone.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.

    Raises:
        ValueError: If up_axis is not 0, 1, or 2.
    """
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")

    x_dir, y_dir, z_dir = (
        (1, 2, 0),
        (0, 1, 2),
        (2, 0, 1),
    )[up_axis]

    indices = []

    cap_vertices = []
    side_vertices = []

    # create center cap vertices
    position = np.array([0, -half_height, 0])[[x_dir, y_dir, z_dir]]
    normal = np.array([0, -1, 0])[[x_dir, y_dir, z_dir]]
    cap_vertices.append([*position, *normal, 0.5, 0.5])
    cap_vertices.append([*-position, *-normal, 0.5, 0.5])

    if top_radius is None:
        top_radius = radius
    side_slope = -np.arctan2(top_radius - radius, 2 * half_height)

    # create the cylinder base and top vertices
    for j in (-1, 1):
        center_index = max(j, 0)
        if j == 1:
            radius = top_radius
        for i in range(segments):
            theta = 2 * np.pi * i / segments

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = cos_theta
            y = j * half_height
            z = sin_theta

            position = np.array([radius * x, y, radius * z])

            normal = np.array([x, side_slope, z])
            normal = normal / np.linalg.norm(normal)
            uv = (i / (segments - 1), (j + 1) / 2)
            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            side_vertices.append(vertex)

            normal = np.array([0, j, 0])
            uv = (cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5)
            vertex = np.hstack([position[[x_dir, y_dir, z_dir]], normal[[x_dir, y_dir, z_dir]], uv])
            cap_vertices.append(vertex)

            cs = center_index * segments
            indices.extend([center_index, i + cs + 2, (i + 1) % segments + cs + 2][::-j])

    # create the cylinder side indices
    for i in range(segments):
        index1 = len(cap_vertices) + i + segments
        index2 = len(cap_vertices) + ((i + 1) % segments) + segments
        index3 = len(cap_vertices) + i
        index4 = len(cap_vertices) + ((i + 1) % segments)

        indices.extend([index1, index2, index3, index2, index4, index3])

    vertex_data = np.array(np.vstack((cap_vertices, side_vertices)), dtype=np.float32)
    index_data = np.array(indices, dtype=np.uint32)

    return vertex_data, index_data


def create_arrow_mesh(
    base_radius, base_height, cap_radius=None, cap_height=None, up_axis=1, segments=default_num_segments
):
    """Create an arrow mesh with cylindrical shaft and conical head.

    Generates vertices and triangle indices for an arrow shape consisting of
    a cylindrical base (shaft) with a conical cap (arrowhead) at the top.

    Args:
        base_radius (float): Radius of the cylindrical shaft.
        base_height (float): Height of the cylindrical shaft portion.
        cap_radius (float, optional): Radius of the conical arrowhead base.
            If None, defaults to base_radius * 1.8.
        cap_height (float, optional): Height of the conical arrowhead.
            If None, defaults to base_height * 0.18.
        up_axis (int): Axis along which the arrow extends (0=X, 1=Y, 2=Z).
            Defaults to 1 (Y-axis).
        segments (int): Number of segments for tessellation.
            Defaults to default_num_segments.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.

    Raises:
        ValueError: If up_axis is not 0, 1, or 2.
    """
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be between 0 and 2")
    if cap_radius is None:
        cap_radius = base_radius * 1.8
    if cap_height is None:
        cap_height = base_height * 0.18

    up_vector = np.array([0, 0, 0])
    up_vector[up_axis] = 1

    base_vertices, base_indices = create_cylinder_mesh(base_radius, base_height / 2, up_axis, segments)
    cap_vertices, cap_indices = create_cone_mesh(cap_radius, cap_height / 2, up_axis, segments)

    base_vertices[:, :3] += base_height / 2 * up_vector
    # move cap slightly lower to avoid z-fighting
    cap_vertices[:, :3] += (base_height + cap_height / 2 - 1e-3 * base_height) * up_vector

    vertex_data = np.vstack((base_vertices, cap_vertices))
    index_data = np.hstack((base_indices, cap_indices + len(base_vertices)))

    return vertex_data, index_data


def create_box_mesh(extents):
    """Create a rectangular box (cuboid) mesh.

    Generates vertices and triangle indices for a box shape with 6 faces.
    Each face consists of 2 triangles with proper normals pointing outward.

    Args:
        extents (tuple[float, float, float]): Half-extents of the box in each
            dimension (half_width, half_length, half_height). The full box
            dimensions will be twice these values.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords).
            - indices (np.ndarray): Uint32 array of triangle indices for
              rendering.
    """
    x_extent, y_extent, z_extent = extents

    vertices = [
        # Position                        Normal    UV
        [-x_extent, -y_extent, -z_extent, -1, 0, 0, 0, 0],
        [-x_extent, -y_extent, z_extent, -1, 0, 0, 1, 0],
        [-x_extent, y_extent, z_extent, -1, 0, 0, 1, 1],
        [-x_extent, y_extent, -z_extent, -1, 0, 0, 0, 1],
        [x_extent, -y_extent, -z_extent, 1, 0, 0, 0, 0],
        [x_extent, -y_extent, z_extent, 1, 0, 0, 1, 0],
        [x_extent, y_extent, z_extent, 1, 0, 0, 1, 1],
        [x_extent, y_extent, -z_extent, 1, 0, 0, 0, 1],
        [-x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 0],
        [-x_extent, -y_extent, z_extent, 0, -1, 0, 1, 0],
        [x_extent, -y_extent, z_extent, 0, -1, 0, 1, 1],
        [x_extent, -y_extent, -z_extent, 0, -1, 0, 0, 1],
        [-x_extent, y_extent, -z_extent, 0, 1, 0, 0, 0],
        [-x_extent, y_extent, z_extent, 0, 1, 0, 1, 0],
        [x_extent, y_extent, z_extent, 0, 1, 0, 1, 1],
        [x_extent, y_extent, -z_extent, 0, 1, 0, 0, 1],
        [-x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 0],
        [-x_extent, y_extent, -z_extent, 0, 0, -1, 1, 0],
        [x_extent, y_extent, -z_extent, 0, 0, -1, 1, 1],
        [x_extent, -y_extent, -z_extent, 0, 0, -1, 0, 1],
        [-x_extent, -y_extent, z_extent, 0, 0, 1, 0, 0],
        [-x_extent, y_extent, z_extent, 0, 0, 1, 1, 0],
        [x_extent, y_extent, z_extent, 0, 0, 1, 1, 1],
        [x_extent, -y_extent, z_extent, 0, 0, 1, 0, 1],
    ]

    # fmt: off
    indices = [
        0, 1, 2,
        0, 2, 3,
        4, 6, 5,
        4, 7, 6,
        8, 10, 9,
        8, 11, 10,
        12, 13, 14,
        12, 14, 15,
        16, 17, 18,
        16, 18, 19,
        20, 22, 21,
        20, 23, 22,
    ]
    # fmt: on
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_plane_mesh(width, length):
    """Create a rectangular plane mesh in the XY plane.

    Generates vertices and triangle indices for a flat rectangular plane
    lying in the XY plane (Z=0) with upward-pointing normals.

    Args:
        width (float): Width of the plane in the X direction.
        length (float): Length of the plane in the Y direction.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (4, 8) where each
              vertex contains [x, y, z, nx, ny, nz, u, v] (position, normal,
              UV coords). All vertices have Z=0 and normals pointing up
              (0, 0, 1).
            - indices (np.ndarray): Uint32 array of 6 triangle indices forming
              2 triangles with counterclockwise winding.
    """
    half_width = width / 2
    half_length = length / 2

    # Create 4 vertices for a rectangle in XY plane (Z=0)
    vertices = [
        # Position                           Normal      UV
        [-half_width, -half_length, 0.0, 0, 0, 1, 0, 0],  # bottom-left
        [half_width, -half_length, 0.0, 0, 0, 1, 1, 0],  # bottom-right
        [half_width, half_length, 0.0, 0, 0, 1, 1, 1],  # top-right
        [-half_width, half_length, 0.0, 0, 0, 1, 0, 1],  # top-left
    ]

    # Create 2 triangles (6 indices) - counterclockwise winding
    indices = [
        0,
        1,
        2,  # first triangle
        0,
        2,
        3,  # second triangle
    ]

    return (np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32))


def load_tetrahedral_mesh(mesh_file: str):
    """
    Load a tetrahedral mesh from a .mesh file (Medit format).

    This function loads tetrahedral meshes used for soft body / FEM simulation.
    The mesh format contains both surface vertices and interior volume elements (tetrahedra),
    enabling realistic deformation physics.

    The .mesh file format (Medit format) contains:
        MeshVersionFormatted 1
        Dimension 3

        Vertices
        <num_vertices>
        x1 y1 z1 ref1
        x2 y2 z2 ref2
        ...

        Tetrahedra
        <num_tetrahedra>
        v1 v2 v3 v4 ref1
        v1 v2 v3 v4 ref2
        ...

        End

    Note: Vertex indices in .mesh files are 1-based and are converted to 0-based for Newton.

    Args:
        mesh_file (str): Path to the .mesh file to load.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices (np.ndarray): Float32 array of shape (N, 3) - vertex positions
            - tetrahedra (np.ndarray): Int32 array of shape (M, 4) - tetrahedron vertex indices (0-based)

    Raises:
        FileNotFoundError: If the mesh file does not exist.
        ValueError: If the mesh file format is invalid or no tetrahedra are found.
    """
    import os

    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    vertices = []
    tetrahedra = []

    with open(mesh_file, "r") as f:
        lines = f.readlines()

    # Find section markers
    vertex_start = None
    tetra_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "Vertices":
            vertex_start = i + 2  # Skip count line
        elif stripped == "Tetrahedra":
            tetra_start = i + 2  # Skip count line

    # Parse vertices
    if vertex_start is not None:
        for i in range(vertex_start, len(lines)):
            line = lines[i].strip()
            if not line or line == "Tetrahedra":
                break
            parts = line.split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append([x, y, z])

    # Parse tetrahedra
    if tetra_start is not None:
        for i in range(tetra_start, len(lines)):
            line = lines[i].strip()
            if not line or line == "End":
                break
            parts = line.split()
            if len(parts) >= 4:
                # Convert from 1-based to 0-based indexing
                v1, v2, v3, v4 = (
                    int(parts[0]) - 1,
                    int(parts[1]) - 1,
                    int(parts[2]) - 1,
                    int(parts[3]) - 1,
                )
                tetrahedra.append([v1, v2, v3, v4])

    if len(tetrahedra) == 0:
        raise ValueError("No tetrahedra found in mesh file!")

    return np.array(vertices, dtype=np.float32), np.array(tetrahedra, dtype=np.int32)


def save_tetrahedral_mesh(mesh_file: str, vertices: np.ndarray, tetrahedra: np.ndarray):
    """
    Save a tetrahedral mesh to a .mesh file (Medit format).

    Args:
        mesh_file (str): Path where to save the .mesh file.
        vertices (np.ndarray): Array of shape (N, 3) - vertex positions.
        tetrahedra (np.ndarray): Array of shape (M, 4) - tetrahedron vertex indices (0-based).
    
    Note:
        Vertex indices are converted from 0-based to 1-based for the file format.
    """
    import os
    
    vertices = np.asarray(vertices, dtype=np.float32)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    
    if vertices.shape[1] != 3:
        raise ValueError(f"Vertices must have shape (N, 3), got {vertices.shape}")
    if tetrahedra.shape[1] != 4:
        raise ValueError(f"Tetrahedra must have shape (M, 4), got {tetrahedra.shape}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(mesh_file) if os.path.dirname(mesh_file) else '.', exist_ok=True)
    
    with open(mesh_file, 'w') as f:
        # Write header
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n")
        f.write("\n")
        
        # Write vertices
        f.write("Vertices\n")
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.17g} {v[1]:.17g} {v[2]:.17g} 1\n")
        f.write("\n")
        
        # Write tetrahedra (convert to 1-based indexing)
        f.write("Tetrahedra\n")
        f.write(f"{len(tetrahedra)}\n")
        for tet in tetrahedra:
            f.write(f"{tet[0]+1} {tet[1]+1} {tet[2]+1} {tet[3]+1} 1\n")
        f.write("\n")
        
        # Write end marker
        f.write("End\n")


# -----------------------------------------------------------------------------
# Surface mesh extraction (tet boundary / tri mesh → surface tris + edges)
# -----------------------------------------------------------------------------


def _face_key(i: int, j: int, k: int):
    """Canonical key for a triangle face (sorted indices)."""
    return tuple(sorted((int(i), int(j), int(k))))


def extract_surface_from_tets(tet_indices: np.ndarray) -> np.ndarray:
    """
    Extract boundary triangle faces from a tetrahedral mesh.
    Each tet has 4 faces; a face is on the boundary iff it appears exactly once.

    Parameters
    ----------
    tet_indices : ndarray
        Shape (tet_count * 4,) or (tet_count, 4). Each tet is 4 vertex indices.

    Returns
    -------
    ndarray
        Shape (num_surface_tris, 3). Each row is (v0, v1, v2) for a boundary triangle.
    """
    tet_indices = np.asarray(tet_indices, dtype=np.int32)
    if tet_indices.ndim == 2:
        tet_indices = tet_indices.ravel()
    tet_count = tet_indices.shape[0] // 4
    face_count: dict = {}
    face_to_verts: dict = {}

    for t in range(tet_count):
        i0 = int(tet_indices[t * 4 + 0])
        i1 = int(tet_indices[t * 4 + 1])
        i2 = int(tet_indices[t * 4 + 2])
        i3 = int(tet_indices[t * 4 + 3])
        faces = [
            (i0, i1, i2),
            (i0, i1, i3),
            (i0, i2, i3),
            (i1, i2, i3),
        ]
        for (a, b, c) in faces:
            key = _face_key(a, b, c)
            face_count[key] = face_count.get(key, 0) + 1
            face_to_verts[key] = (a, b, c)

    boundary = [face_to_verts[k] for k, count in face_count.items() if count == 1]
    if not boundary:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(boundary, dtype=np.int32)


def triangles_to_edges(tri_indices: np.ndarray) -> np.ndarray:
    """
    Build unique edges from triangle indices. Each triangle contributes 3 edges;
    deduplicate by canonical (min, max) edge key.

    Parameters
    ----------
    tri_indices : ndarray
        Shape (num_tris, 3) or (num_tris * 3,). Vertex indices per triangle.

    Returns
    -------
    ndarray
        Shape (num_edges, 2). Each row is (v0, v1) with v0 < v1.
    """
    if tri_indices.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if tri_indices.ndim == 1:
        n = tri_indices.shape[0] // 3
        tris = tri_indices.reshape(n, 3)
    else:
        tris = np.asarray(tri_indices, dtype=np.int32)
    seen: set = set()
    edges: list = []
    for (a, b, c) in tris:
        for u, v in [(a, b), (b, c), (c, a)]:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(edges, dtype=np.int32)


def get_surface_triangles_and_edges(
    tri_indices: np.ndarray | None,
    tet_indices: np.ndarray | None,
    tri_count: int,
    tet_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get surface triangle indices (N, 3) and surface edge indices (E, 2).
    Uses tri_indices as surface if tri_count > 0; otherwise extracts boundary from tets.

    Returns
    -------
    surface_tris : ndarray (N, 3)
    surface_edges : ndarray (E, 2)
    """
    if tri_count > 0 and tri_indices is not None and tri_indices.size >= tri_count * 3:
        if tri_indices.ndim == 1:
            surface_tris = np.asarray(
                tri_indices[: tri_count * 3].reshape(tri_count, 3), dtype=np.int32
            )
        else:
            surface_tris = np.asarray(tri_indices[:tri_count], dtype=np.int32)
    elif tet_count > 0 and tet_indices is not None and tet_indices.size >= tet_count * 4:
        tet_flat = np.asarray(tet_indices[: tet_count * 4], dtype=np.int32)
        if tet_flat.ndim == 2:
            tet_flat = tet_flat.ravel()
        surface_tris = extract_surface_from_tets(tet_flat)
    else:
        surface_tris = np.zeros((0, 3), dtype=np.int32)

    surface_edges = triangles_to_edges(surface_tris)
    return surface_tris, surface_edges
