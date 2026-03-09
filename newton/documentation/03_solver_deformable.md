# SolverDeformable: Self-Collision

**Location:** `newton/_src/solvers/deformable/solver_deformable.py`, `kernels_self_contact.py`, `tri_mesh_collision.py`

## Role

`SolverDeformable` extends `SolverSoft` by adding **self-collision**: the mesh can collide with itself. This avoids inter-penetration when the body folds (e.g. inchworm bending). Detection is done on the **surface** (triangles and edges); repulsion is applied between vertex–triangle and edge–edge pairs.

## Detection: Surface and BVH

- **Surface mesh:** from the volumetric mesh, surface triangles and edges are extracted (`get_surface_triangles_and_edges`). Stored as `_surface_tri_indices`, `_surface_edge_indices`, with counts `_num_surface_tris`, `_num_surface_edges`.
- **Collision detector:** `TriMeshCollisionDetector` (in `tri_mesh_collision.py`) builds BVHs on the surface and runs:
  - **Vertex–triangle:** each vertex is tested against surface triangles within a **margin** (query radius).
  - **Edge–edge:** pairs of surface edges within the margin.
- **Margin vs radius:** `self_contact_margin` is the BVH query radius; it must be \(\ge\) `self_contact_radius`. Typically margin \(\approx 1.5\times\) radius so no contacts are missed. Force law uses `self_contact_radius`, `self_contact_stiffness`, `self_contact_force_cap`.

## Repulsion Law

- **Distance:** \(d\) = distance between vertex and triangle (or between two edges).
- **Repulsion:** C²-style law in `_self_contact_force_magnitude`:
  - If \(d \ge\) `self_contact_radius`: no force.
  - Else penetration = `self_contact_radius` \(- d\). For \(d\) in \((\varepsilon, \tau)\) with \(\tau = 0.5\times\) radius: force magnitude \(\propto k_2/d\). Otherwise linear in penetration: \(k \cdot \text{penetration}\).
- **Force cap:** total repulsion magnitude per pair can be capped by `self_contact_force_cap` (0 = no cap).

## Vertex–Triangle and Edge–Edge

- **Vertex–triangle:** kernel `eval_self_contact_vertex_triangle_from_collision_info` uses `TriMeshCollisionDetector` output. For each vertex–triangle pair in the collision list, it computes repulsion and optional **friction** (IPC-style, using tangent plane and slip velocity).
- **Edge–edge:** kernel `eval_self_contact_edge_edge_from_collision_info` does the same for edge–edge pairs. Near-parallel edges are skipped (epsilon `edge_edge_parallel_epsilon`).

## Implicit Tangent (Matrix)

So that self-contact does not destabilize the implicit step, a **lumped tangent** is added to the system matrix for vertices that are in vertex–triangle contact:

- Diagonal block for vertex \(i\): \(h^2\,k_{\mathrm{eff}}\,\mathbf{I}\), where \(k_{\mathrm{eff}}\) is derived from stiffness and number of contacts. Implemented in `_add_extra_matrix_blocks` → `build_system_matrix_self_contact_diagonal_kernel`.
- Edge–edge is **not** linearized into the matrix; it contributes only to the RHS (forces). So vertex–triangle is “implicit”, edge–edge remains explicit in the matrix sense.

## Step Sequence

In `step`:

1. If self-contact is enabled, refit BVH from current positions and run vertex–triangle and (optionally) edge–edge detection.
2. Rebuild system matrix (including the extra self-contact diagonal blocks) and preconditioner.
3. Evaluate all forces (springs, FEM, ground, soft contacts, **self-contact**, etc.).
4. RHS = \(h\times\) total force; solve \(A\,\Delta\mathbf{v} = \mathbf{f}\); update state.
5. Apply constraint contact corrections if used.

## Parameters (Same Names as in Code)

- `handle_self_contact`: enable self-collision.
- `self_contact_radius`: distance threshold for repulsion.
- `self_contact_margin`: BVH query radius (\(\ge\) radius).
- `self_contact_stiffness`, `self_contact_force_cap`.
- `self_contact_edge_edge`: include edge–edge repulsion.
- `self_contact_friction_mu`, `self_contact_friction_epsilon`: friction for self-contact.
- `vertex_collision_buffer_pre_alloc`, `edge_collision_buffer_pre_alloc`: per-vertex / per-edge collision list sizes.

## Naming in Code

- `eval_self_contact_forces`, `_add_extra_matrix_blocks`
- `_surface_tri_indices`, `_surface_edge_indices`, `_trimesh_collision_detector`
- `TriMeshCollisionDetector`, `vertex_triangle_collision_detection`, `edge_edge_collision_detection`
- `eval_self_contact_vertex_triangle_from_collision_info`, `eval_self_contact_edge_edge_from_collision_info`
- `build_system_matrix_self_contact_diagonal_kernel`
