# SolverSoft: Implicit Integration and FEM

**Location:** `newton/_src/solvers/soft/solver_soft.py`, `kernels.py`

## Role

`SolverSoft` is the base soft-body solver. It uses **implicit integration** and sparse linear algebra so that stiff materials (e.g. high \(k_\mu\), \(k_\lambda\)) remain stable at practical timesteps.

## Implicit Integration

Each step solves for the velocity change \(\Delta\mathbf{v}\):

\[
A\,\Delta\mathbf{v} = \mathbf{f}
\]

with

\[
A = M - h D - h^2 K
\]

- \(M\): mass matrix (lumped, diagonal).
- \(D\): damping matrix (from springs and materials).
- \(K\): stiffness matrix (springs, tetrahedra, triangles).
- \(h\): timestep `dt`.
- \(\mathbf{f}\): \(h\) times the total force (all force contributions summed, then multiplied by \(h\)).

After solving, state is updated with `update_state`: \(\mathbf{q}^{n+1} = \mathbf{q}^n + h\,\mathbf{v}^{n+1}\), \(\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta\mathbf{v}\).

## System Matrix Assembly

The matrix \(A\) is built in `_build_system_matrix`:

1. **Diagonal (per particle):** mass and spring contributions (stiffness and damping from each spring incident to the particle). Kernels: `build_system_matrix_diagonal_kernel`, `build_system_matrix_diagonal_mass_kernel`.
2. **Spring off-diagonals:** blocks for each spring \((i,j)\) and \((j,i)\). Kernel: `build_system_matrix_sparse_kernel`.
3. **Tetrahedral FEM:** tangent stiffness from current deformation (when `state` is provided). Kernel: `build_system_matrix_tet_kernel`.
4. **Triangle FEM:** lumped tangent for triangles (when `state` is provided). Kernel: `build_system_matrix_tri_kernel`.

Subclasses (e.g. `SolverDeformable`) can add more blocks via `_add_extra_matrix_blocks`.

## FEM and Springs

- **Tetrahedra:** Neo-Hookean (or similar) model; rest pose stored in `tet_poses` (e.g. \(\mathbf{D}_m^{-1}\)), materials in `tet_materials` (\(k_\mu\), \(k_\lambda\), \(k_{\mathrm{damp}}\)). Forces: `eval_tetrahedral_forces` → `eval_tetrahedra`.
- **Triangles:** membrane model with rest poses and materials. Forces: `eval_triangle_forces` → `eval_triangles`.
- **Springs:** linear springs with rest length, stiffness `spring_stiffness`, damping `spring_damping`. Forces: `eval_spring_forces` → `eval_springs`. Spring potential: \(V = \frac{1}{2}k_e(\ell - \ell_0)^2\); damping in velocity along the spring direction.
- **Bending:** edge-based bending (e.g. rest angle). Forces: `eval_bending_forces` → `eval_bending`.

## Ground and Other Forces

- **Ground plane:** optional force-based ground contact with normal stiffness/damping and Coulomb friction; plane \(\mathbf{n}\cdot\mathbf{x} + d = 0\). Implemented in `eval_particle_ground_contact_forces` using `_ground_plane`, `_ground_ke`, `_ground_kd`, `_ground_kf`, `_ground_mu`.
- **Soft contacts:** collision with rigid shapes; forces from `eval_soft_contact_forces`.
- **Constraint contacts:** when `use_constraint_contacts` is True, constraint-based contact corrections are applied after the implicit step (XPBD-style position corrections).

## Linear Solver and Preconditioner

- Solver type: `solver_type` in `{"bicgstab", "cg", "gmres", "cr"}` (default `bicgstab`).
- Preconditioner: `preconditioner_type` in `{"id", "diag", "diag_abs"}`.
- Arrays: `A_bsr` (BSR format), `M_bsr` (preconditioner), `dv` (solution). When FEM tangent is in the matrix (`use_fem_tangent_in_matrix`), the matrix and preconditioner are rebuilt each step from current `state`.

## Naming in Code

- `model.spring_indices`, `model.spring_rest_length`, `model.spring_stiffness`, `model.spring_damping`
- `model.tet_indices`, `model.tet_poses`, `model.tet_materials`
- `model.tri_indices`, `model.tri_poses`, `model.tri_materials`
- `_build_system_matrix`, `implicit_integration`, `eval_spring_forces`, `eval_tetrahedral_forces`, `eval_triangle_forces`, `eval_bending_forces`
