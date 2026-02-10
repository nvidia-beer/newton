#!/usr/bin/env python3
"""
Create ball.mesh from TetraSphere for debugging bouncing_mesh example.

This script generates a tetrahedral sphere mesh (same as bouncing_ball.py uses)
and saves it in .mesh format for use with the bouncing_mesh example.

Usage:
    python -m newton.examples.soft.create_ball_mesh
"""

import numpy as np
import os

import newton
from newton.solvers import TetraSphere
from newton.utils import save_tetrahedral_mesh
import newton.examples

def main():
    # Parameters matching bouncing_ball.py defaults
    radius = 0.3
    subdivisions = 2
    interior_layers = 2
    
    print("Creating tetrahedral sphere mesh...")
    print(f"  Radius: {radius}")
    print(f"  Subdivisions: {subdivisions}")
    print(f"  Interior layers: {interior_layers}")
    
    # Generate sphere mesh
    sphere = TetraSphere(
        radius=radius,
        subdivisions=subdivisions,
        interior_layers=interior_layers,
        verbose=True
    )
    
    # Get mesh data
    mesh_data = sphere.get_mesh_data()
    vertices = mesh_data['vertices']
    tetrahedra = mesh_data['tetrahedra']
    
    print(f"\nGenerated mesh:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Tetrahedra: {len(tetrahedra)}")
    
    # Validate mesh
    validation = sphere.validate_mesh()
    print(f"\nMesh validation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Positive volume: {validation['positive_volume']}")
    if validation['negative_volume'] > 0:
        print(f"  WARNING: Inverted: {validation['negative_volume']}")
    if validation['degenerate'] > 0:
        print(f"  WARNING: Degenerate: {validation['degenerate']}")
    
    # Save to ball.mesh in examples/assets directory using get_asset for proper path resolution
    mesh_file = newton.examples.get_asset("ball.mesh")
    mesh_dir = os.path.dirname(mesh_file)
    if mesh_dir:
        os.makedirs(mesh_dir, exist_ok=True)
    
    print(f"\nSaving to: {mesh_file}")
    save_tetrahedral_mesh(mesh_file, vertices, tetrahedra)
    
    print(f"✓ Successfully saved ball.mesh")
    print(f"\nYou can now use this mesh for debugging:")
    print(f"  python -m newton.examples.soft.example_bouncing_mesh")
    print(f"  (ball.mesh will be used automatically if available)")

if __name__ == "__main__":
    main()
