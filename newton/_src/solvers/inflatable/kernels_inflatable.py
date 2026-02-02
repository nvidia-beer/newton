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
Warp kernels for inflatable soft body simulation.

These kernels implement the rest configuration scaling approach for inflation:
- Scale spring rest lengths for inflation
- Scale tetrahedra rest poses (Dm_inv) for inflation
- Compute volume from tetrahedra
"""

import warp as wp


@wp.kernel
def scale_spring_rest_lengths_kernel(
    original_rest_lengths: wp.array(dtype=wp.float32),
    scale: wp.float32,
    scaled_rest_lengths: wp.array(dtype=wp.float32),
):
    """
    Scale spring rest lengths by a factor.
    
    For inflation, springs "want" to be longer when scaled up.
    
    Parameters
    ----------
    original_rest_lengths : array
        Original (unscaled) spring rest lengths
    scale : float
        Linear scale factor (cbrt(volume_ratio) for 3D)
    scaled_rest_lengths : array (output)
        Scaled spring rest lengths
    """
    sid = wp.tid()
    scaled_rest_lengths[sid] = original_rest_lengths[sid] * scale


@wp.kernel
def scale_tet_poses_kernel(
    original_poses: wp.array(dtype=wp.mat33),
    scale: wp.float32,
    scaled_poses: wp.array(dtype=wp.mat33),
):
    """
    Scale tetrahedra rest poses (Dm_inv).
    
    The rest pose is the inverse of the rest shape matrix Dm.
    To scale the rest shape by 's', we scale Dm by 's', so Dm_inv scales by '1/s'.
    
    This makes the tetrahedra "want" to be larger when scaled up.
    
    Parameters
    ----------
    original_poses : array
        Original (unscaled) rest poses (Dm_inv matrices)
    scale : float
        Linear scale factor (cbrt(volume_ratio) for 3D)
    scaled_poses : array (output)
        Scaled rest poses
    """
    tid = wp.tid()
    inv_scale = 1.0 / scale
    orig = original_poses[tid]
    
    # Scale each element of the 3x3 matrix
    scaled_poses[tid] = wp.mat33(
        orig[0, 0] * inv_scale, orig[0, 1] * inv_scale, orig[0, 2] * inv_scale,
        orig[1, 0] * inv_scale, orig[1, 1] * inv_scale, orig[1, 2] * inv_scale,
        orig[2, 0] * inv_scale, orig[2, 1] * inv_scale, orig[2, 2] * inv_scale
    )


@wp.kernel
def compute_volume_kernel(
    positions: wp.array(dtype=wp.vec3),
    tet_indices: wp.array2d(dtype=wp.int32),
    tet_volumes: wp.array(dtype=wp.float32),
):
    """
    Compute volume of each tetrahedron.
    
    Volume = |det([e1, e2, e3])| / 6
    where e1, e2, e3 are edge vectors from vertex 0.
    
    Parameters
    ----------
    positions : array
        Current particle positions
    tet_indices : array2d
        Tetrahedra indices (Nx4)
    tet_volumes : array (output)
        Volume of each tetrahedron
    """
    tid = wp.tid()
    
    # Get tetrahedron vertex indices
    i0 = tet_indices[tid, 0]
    i1 = tet_indices[tid, 1]
    i2 = tet_indices[tid, 2]
    i3 = tet_indices[tid, 3]
    
    # Get vertex positions
    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]
    
    # Edge vectors from vertex 0
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    
    # Volume = |det([e1, e2, e3])| / 6
    # det = e1 · (e2 × e3)
    cross = wp.cross(e2, e3)
    det = wp.dot(e1, cross)
    
    tet_volumes[tid] = wp.abs(det) / 6.0


@wp.kernel
def scale_tet_materials_kernel(
    original_materials: wp.array2d(dtype=wp.float32),
    scale: wp.float32,
    scaled_materials: wp.array2d(dtype=wp.float32),
):
    """
    Scale tetrahedra material properties for inflation.
    
    This is optional - material properties can be kept constant
    or scaled with volume for different inflation behaviors.
    
    Parameters
    ----------
    original_materials : array2d
        Original material properties (k_mu, k_lambda, k_damp)
    scale : float
        Volume scale factor
    scaled_materials : array2d (output)
        Scaled material properties
    """
    tid = wp.tid()
    
    # Keep material properties constant during inflation
    # (material stiffness doesn't change with size)
    scaled_materials[tid, 0] = original_materials[tid, 0]
    scaled_materials[tid, 1] = original_materials[tid, 1]
    scaled_materials[tid, 2] = original_materials[tid, 2]


@wp.kernel
def compute_surface_area_kernel(
    positions: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_areas: wp.array(dtype=wp.float32),
):
    """
    Compute area of each surface triangle.
    
    Area = |e1 × e2| / 2
    where e1, e2 are edge vectors.
    
    Parameters
    ----------
    positions : array
        Current particle positions
    tri_indices : array2d
        Triangle indices (Nx3)
    tri_areas : array (output)
        Area of each triangle
    """
    tid = wp.tid()
    
    # Get triangle vertex indices
    i0 = tri_indices[tid, 0]
    i1 = tri_indices[tid, 1]
    i2 = tri_indices[tid, 2]
    
    # Get vertex positions
    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    
    # Edge vectors
    e1 = p1 - p0
    e2 = p2 - p0
    
    # Area = |e1 × e2| / 2
    cross = wp.cross(e1, e2)
    tri_areas[tid] = wp.length(cross) * 0.5
