#!/usr/bin/env python3
"""
Create spot_fixed.mesh from spot.mesh with proper centering and filtering.

This script loads spot.mesh, centers it so bottom is at Z=0, fixes inverted tetrahedra
by flipping them (swapping vertices), removes degenerate tetrahedra (zero/near-zero volume),
and saves it as spot_fixed.mesh for direct loading without calculations.

Usage:
    python -m newton.examples.soft.create_spot_fixed_mesh
"""

import numpy as np
import os

import newton
import newton.examples
from newton.utils import load_tetrahedral_mesh, save_tetrahedral_mesh


def compute_tet_volume(v):
    """Compute signed volume of tetrahedron."""
    d1 = v[1] - v[0]
    d2 = v[2] - v[0]
    d3 = v[3] - v[0]
    return np.dot(d1, np.cross(d2, d3)) / 6.0

def compute_tet_determinant(v):
    """Compute determinant of Dm matrix (used for inv_Dm calculation).
    
    This is the SAME as volume * 6.0, but we compute it directly.
    In the kernel: inv_rest_volume = det(inv_Dm) * 6.0
    But inv_Dm = inv(Dm), so det(inv_Dm) = 1/det(Dm)
    So inv_rest_volume = (1/det(Dm)) * 6.0 = 6.0 / det(Dm)
    If det(Dm) is small, inv_rest_volume becomes huge -> instability!
    """
    d1 = v[1] - v[0]
    d2 = v[2] - v[0]
    d3 = v[3] - v[0]
    Dm = np.array([d1, d2, d3]).T
    return np.linalg.det(Dm)

def compute_inv_determinant(v):
    """Compute determinant of inv_Dm (what the kernel actually uses).
    
    In the kernel, Dm = pose[tid] which is inv_Dm from builder.
    So kernel computes: inv_rest_volume = det(inv_Dm) * 6.0
    Since det(inv_Dm) = 1/det(Dm), we can compute this directly.
    """
    det_Dm = compute_tet_determinant(v)
    if abs(det_Dm) < 1e-20:
        return float('inf')  # Singular matrix
    return 1.0 / det_Dm

def compute_tet_condition_number(v):
    """Compute condition number of Dm matrix.
    
    Large condition number -> small determinant -> large inv_Dm -> instability.
    Returns inf if matrix is singular.
    """
    d1 = v[1] - v[0]
    d2 = v[2] - v[0]
    d3 = v[3] - v[0]
    Dm = np.array([d1, d2, d3]).T
    try:
        cond = np.linalg.cond(Dm)
        return cond
    except:
        return float('inf')


def main():
    # Load original spot.mesh
    spot_mesh_file = newton.examples.get_asset("spot.mesh")
    
    if not os.path.exists(spot_mesh_file):
        print(f"Error: {spot_mesh_file} not found!")
        return
    
    print(f"Loading {spot_mesh_file}...")
    vertices_raw, tetras = load_tetrahedral_mesh(spot_mesh_file)
    original_tetra_count = len(tetras)
    print(f"Loaded mesh: {len(vertices_raw)} vertices, {original_tetra_count} tetrahedra")
    
    # Check original mesh size
    extents_raw = vertices_raw.max(axis=0) - vertices_raw.min(axis=0)
    max_extent_raw = np.max(extents_raw)
    print(f"Original mesh extents: {extents_raw}, max: {max_extent_raw:.3f}m")
    
    # CRITICAL: Newton uses Z-up coordinate system (ground is at Z=0)
    # Check which axis appears to be vertical (largest extent)
    vertical_axis_idx = np.argmax(extents_raw)
    axis_names = ['X', 'Y', 'Z']
    print(f"Detected vertical axis: {axis_names[vertical_axis_idx]} (extent: {extents_raw[vertical_axis_idx]:.3f}m)")
    
    # Convert Y-up to Z-up if needed (Y-up: X=right, Y=up, Z=forward -> Z-up: X=right, Y=forward, Z=up)
    if vertical_axis_idx == 1:  # Y is vertical (Y-up mesh)
        print(f"  Converting Y-up mesh to Z-up for Newton...")
        # Y-up to Z-up: swap Y and Z axes
        # [x, y, z] in Y-up -> [x, z, y] in Z-up
        # This makes Y (up) become Z (up), and Z (forward) become Y (forward)
        vertices_converted = vertices_raw.copy()
        vertices_converted[:, [1, 2]] = vertices_raw[:, [2, 1]]  # Swap Y and Z
        vertices_raw = vertices_converted
        print(f"  ✓ Converted: Y-up -> Z-up (swapped Y and Z axes)")
        # Recalculate extents after conversion
        extents_raw = vertices_raw.max(axis=0) - vertices_raw.min(axis=0)
        print(f"  New extents: {extents_raw}, max: {np.max(extents_raw):.3f}m")
    elif vertical_axis_idx != 2:  # Not Y and not Z
        print(f"  ⚠ WARNING: Mesh appears to use {axis_names[vertical_axis_idx]}-up, but Newton uses Z-up!")
        print(f"  No automatic conversion available - mesh may appear incorrectly oriented.")
    
    # Trust the original mesh - just center and normalize it
    # Center mesh around origin
    center = vertices_raw.mean(axis=0)
    vertices_centered = vertices_raw - center
    
    # Normalize to 10m to preserve stable rest volumes while avoiding numerical precision issues
    # Normalizing to 1m makes rest volumes too small (~1e-17), but 10m gives reasonable volumes
    # This is a compromise: large enough for stable FEM, small enough to avoid precision issues
    target_size = 10.0  # Target max extent in meters
    
    extents_centered = vertices_centered.max(axis=0) - vertices_centered.min(axis=0)
    max_extent_centered = np.max(extents_centered)
    
    if max_extent_centered > 0:
        # Normalize by bounding box max to target_size
        scale_factor = target_size / max_extent_centered
        vertices_normalized = vertices_centered * scale_factor
        
        # Verify normalization
        extents_normalized = vertices_normalized.max(axis=0) - vertices_normalized.min(axis=0)
        max_extent_normalized = np.max(extents_normalized)
        
        print(f"Normalized mesh to {target_size}m:")
        print(f"  Original max extent: {max_extent_centered:.3f}m")
        print(f"  Scale factor: {scale_factor:.6f}")
        print(f"  Normalized max extent: {max_extent_normalized:.3f}m (target: {target_size}m)")
        print(f"  Rest volumes will scale by {scale_factor**3:.2e} (scale³)")
        print(f"  Use --scale 0.1 in example_bouncing_mesh.py to get ~1m mesh in simulation")
        
        # IMPORTANT: Thresholds need to scale with volume scaling (scale_factor^3)
        # If we filtered with threshold T in original scale, volumes become T * scale_factor^3
        # So threshold in normalized scale should be T * scale_factor^3
        volume_scale_factor = scale_factor ** 3
        print(f"  Volume scale factor: {volume_scale_factor:.2e}")
    else:
        vertices_normalized = vertices_centered
        scale_factor = 1.0
        volume_scale_factor = 1.0
        print(f"⚠ Warning: Mesh has zero extent, skipping normalization")
    
    # Shift so bottom is at Z=0
    min_z = vertices_normalized[:, 2].min()
    vertices_final = vertices_normalized.copy()
    vertices_final[:, 2] -= min_z  # Shift so minimum Z is 0
    
    extents_final = vertices_final.max(axis=0) - vertices_final.min(axis=0)
    max_extent_final = np.max(extents_final)
    print(f"Final mesh: Z range [{vertices_final[:, 2].min():.3f}, {vertices_final[:, 2].max():.3f}], max extent: {max_extent_final:.3f}m")
    
    if max_extent_final <= 20.0:
        print(f"  ✓ Mesh size is acceptable ({max_extent_final:.1f}m)")
    else:
        print(f"  ⚠ Warning: Mesh is still large ({max_extent_final:.1f}m)")
    
    vertices_centered = vertices_final  # Use final processed vertices
    
    # Fix inverted tetrahedra AFTER normalization (use normalized vertices for consistency)
    # CRITICAL: We need to fix using the same vertices we'll use for volume checking later
    # Otherwise, fixing based on original scale but checking volumes in normalized scale can cause mismatches
    print("Fixing tetrahedra winding (using normalized vertices for consistency)...")
    tetras_fixed_indices = []
    inverted_count = 0
    
    # Fix using normalized vertices (same scale as volume checking)
    for t in range(len(tetras)):
        tet = tetras[t].copy()
        v_normalized = vertices_centered[tet]  # Use normalized vertices
        volume_normalized = compute_tet_volume(v_normalized)
        
        # Fix inverted tetrahedra by flipping (swapping vertices)
        if volume_normalized < 0:
            # Inverted tetrahedron - fix by swapping first two vertices
            tet[0], tet[1] = tet[1], tet[0]
            inverted_count += 1
            
            # Verify fix worked
            v_normalized_fixed = vertices_centered[tet]
            volume_normalized_fixed = compute_tet_volume(v_normalized_fixed)
            if volume_normalized_fixed < 0:
                # Still inverted - might be degenerate, will be filtered later
                pass
        
        tetras_fixed_indices.append(tet)
    
    tetras_fixed = np.array(tetras_fixed_indices, dtype=np.int32)
    
    if inverted_count > 0:
        print(f"✓ Fixed {inverted_count} inverted tetrahedra (flipped by swapping vertices to correct winding)")
    
    # Minimal validation: Only remove truly degenerate tetrahedra (zero volume, NaN, Inf)
    # Trust the original mesh - it worked before, so we just fix flipped ones and remove only the worst
    print("\n🔍 Minimal validation (trusting original mesh)...")
    print("  Only removing truly degenerate tetrahedra (zero volume, NaN, Inf)")
    print("  No filtering based on determinants/condition numbers - original mesh should work!")
    
    valid_tetras = []
    degenerate_count = 0
    
    for i, tet in enumerate(tetras_fixed):
        v = vertices_centered[tet]
        volume = compute_tet_volume(v)
        
        # Only filter truly degenerate: zero volume, NaN, or Inf
        if not np.isfinite(volume) or abs(volume) < 1e-20:
            degenerate_count += 1
            if degenerate_count <= 5:  # Show first few
                print(f"  Removing tetra {i}: volume={volume:.2e} (truly degenerate)")
        else:
            # Keep all other tetrahedra - trust the original mesh
            valid_tetras.append(tet)
    
    # Update tetras_fixed to only include valid tetrahedra
    if degenerate_count > 0:
        print(f"⚠ Removed {degenerate_count} truly degenerate tetrahedra (zero volume, NaN, or Inf)")
        tetras_fixed = np.array(valid_tetras, dtype=np.int32)
    else:
        tetras_fixed = np.array(valid_tetras, dtype=np.int32)
        print(f"✓ All tetrahedra are valid")
    
    # Print validation results
    total_before_filtering = len(tetras_fixed) + degenerate_count
    print(f"\n{'='*70}")
    print(f"Validation results:")
    print(f"{'='*70}")
    print(f"  Original tetrahedra: {original_tetra_count}")
    print(f"  After pre-filtering: {len(tetras)}")
    print(f"  Fixed inverted tetrahedra: {inverted_count} (flipped, not removed)")
    print(f"  Removed degenerate: {degenerate_count} (zero volume, NaN, or Inf)")
    print(f"  Valid tetrahedra: {len(tetras_fixed)}")
    if total_before_filtering > 0:
        kept_pct = 100.0 * len(tetras_fixed) / total_before_filtering
        print(f"  ✓ Kept {len(tetras_fixed)} tetrahedra ({kept_pct:.1f}%)")
    print(f"{'='*70}")
    
    if len(tetras_fixed) == 0:
        print(f"\n❌ ERROR: No valid tetrahedra remaining after filtering!")
        print(f"   All tetrahedra were degenerate or inverted")
        print(f"   Please check the source mesh.")
        return
    
    if degenerate_count > 0:
        print(f"\n✓ Filtered mesh: {len(tetras_fixed)} valid tetrahedra (removed {degenerate_count} truly degenerate ones)")
    else:
        print(f"\n✓ All tetrahedra are valid - mesh is ready to use!")
    
    # Verify mesh Z range before saving
    min_z_final = vertices_centered[:, 2].min()
    max_z_final = vertices_centered[:, 2].max()
    print(f"\n🔍 Verifying mesh before saving...")
    print(f"  Z range: [{min_z_final:.6f}, {max_z_final:.6f}]")
    if abs(min_z_final) > 1e-6:
        print(f"  ⚠ WARNING: Mesh bottom is not at Z=0! min_z={min_z_final:.6f}")
        print(f"  Fixing by shifting Z coordinates...")
        vertices_centered[:, 2] -= min_z_final
        min_z_final = vertices_centered[:, 2].min()
        max_z_final = vertices_centered[:, 2].max()
        print(f"  ✓ Fixed: Z range is now [{min_z_final:.6f}, {max_z_final:.6f}]")
    else:
        print(f"  ✓ Mesh bottom is at Z=0 (min_z={min_z_final:.6f})")
    
    if len(valid_tetras) == 0:
        print(f"\n❌ ERROR: No valid tetrahedra remaining!")
        print(f"   Cannot create mesh file.")
        return
    
    # Save to spot_fixed.mesh in examples/assets directory
    fixed_mesh_file = newton.examples.get_asset("spot_fixed.mesh")
    mesh_dir = os.path.dirname(fixed_mesh_file)
    if mesh_dir:
        os.makedirs(mesh_dir, exist_ok=True)
    
    print(f"\n💾 Saving fixed mesh to: {fixed_mesh_file}")
    print(f"  Vertices: {len(vertices_centered)}")
    print(f"  Tetrahedra: {len(valid_tetras)}")
    min_z_save = vertices_centered[:, 2].min()
    max_z_save = vertices_centered[:, 2].max()
    print(f"  Z range BEFORE save: [{min_z_save:.6f}, {max_z_save:.6f}]")
    if abs(min_z_save) > 1e-6:
        print(f"  ⚠ CRITICAL: Mesh bottom is NOT at Z=0! This will cause mesh to appear under ground!")
        print(f"  Fixing now...")
        vertices_centered[:, 2] -= min_z_save
        min_z_save = vertices_centered[:, 2].min()
        max_z_save = vertices_centered[:, 2].max()
        print(f"  ✓ Fixed: Z range is now [{min_z_save:.6f}, {max_z_save:.6f}]")
    
    # Save the FINAL filtered tetrahedra (valid_tetras)
    tetras_final = np.array(valid_tetras, dtype=np.int32)
    save_tetrahedral_mesh(fixed_mesh_file, vertices_centered, tetras_final)
    print(f"  ✓ Mesh saved successfully!")
    
    # Verify the saved file by reloading it
    print(f"\n🔍 Verifying saved mesh file...")
    try:
        vertices_loaded, tetras_loaded = load_tetrahedral_mesh(fixed_mesh_file)
        min_z_loaded = vertices_loaded[:, 2].min()
        max_z_loaded = vertices_loaded[:, 2].max()
        print(f"  Loaded mesh Z range: [{min_z_loaded:.6f}, {max_z_loaded:.6f}]")
        if abs(min_z_loaded) > 1e-6:
            print(f"  ⚠ WARNING: Loaded mesh bottom is NOT at Z=0! min_z={min_z_loaded:.6f}")
            print(f"  The mesh file may not have been saved correctly!")
        else:
            print(f"  ✓ Verified: Mesh bottom is at Z=0 (min_z={min_z_loaded:.6f})")
    except Exception as e:
        print(f"  ⚠ Could not verify saved mesh: {e}")
    
    # Verify final mesh size and count
    extents_check = vertices_centered.max(axis=0) - vertices_centered.min(axis=0)
    max_extent_check = np.max(extents_check)
    print(f"\n{'='*70}")
    print(f"✓ Successfully created spot_fixed.mesh")
    print(f"{'='*70}")
    print(f"  Final mesh statistics:")
    print(f"    Vertices: {len(vertices_centered)}")
    print(f"    Tetrahedra: {len(tetras_final)} (filtered from original {original_tetra_count})")
    print(f"    Removed: {original_tetra_count - len(tetras_final)} degenerate tetrahedra ({100.0 * (original_tetra_count - len(tetras_final)) / original_tetra_count:.1f}%)")
    print(f"    Mesh size: {extents_check}, max: {max_extent_check:.3f}m")
    print(f"    Z range: [{vertices_centered[:, 2].min():.6f}, {vertices_centered[:, 2].max():.6f}]")
    print(f"  ✓ Mesh is ready for simulation!")
    print(f"  Use --scale 0.1 in example_bouncing_mesh.py to get ~{max_extent_check*0.1:.2f}m mesh in simulation")
    print(f"{'='*70}")
    
    print(f"\nYou can now use this mesh directly:")
    print(f"  python -m newton.examples.soft.example_bouncing_mesh --mesh_file examples/assets/spot_fixed.mesh")
    print(f"  Or test stability:")
    print(f"  ./newton/.devcontainer/test-tetra-stability.sh examples/assets/spot_fixed.mesh")


if __name__ == "__main__":
    main()
