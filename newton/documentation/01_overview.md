# Overview: Paper to Implementation

## Paper Reference

**Gamus et al., "Understanding Legged Crawling for Soft Robots," arXiv:1911.05227**

The inchworm crawling example implements the same physical scenario as in the paper: a soft robot with two bending segments (left/right along the beam), inflated chambers to produce bending, and a phase-shifted harmonic gait. When the **crawlable** stick-slip model is enabled, ground friction follows the paper’s hybrid stick-slip formulation so the robot can crawl.

## Solver Stack (Inheritance)

The implementation is layered; each solver extends the previous one:

```
SolverSoft          → implicit integration, FEM (tetrahedra, triangles), springs, ground plane
    ↓
SolverDeformable   → + self-collision (vertex–triangle, edge–edge, BVH)
    ↓
SolverInflatable   → + inflation (rest-config scaling), chambers, anisotropy, torque (folding)
    ↓
SolverCrawlable    → + paper stick-slip ground contact, kinematic displacement ±Δd
```

- **`example_inchworm_crawling.py`** uses `SolverCrawlable` and `TetraBox` to build the mesh.
- With `--crawlable` (or `use_crawlable_stick_slip=True`), the paper’s stick-slip friction and kinematic displacement are active; otherwise behaviour matches the non-crawlable inchworm (visible bending, Coulomb ground contact only).

## Main Code Names (Same as in Code)

| Concept | Code / Doc name |
|--------|------------------|
| Soft body solver (base) | `SolverSoft` |
| With self-collision | `SolverDeformable` |
| With inflation and chambers | `SolverInflatable` |
| With paper stick-slip | `SolverCrawlable` |
| Inchworm example class | `Example` (in `example_inchworm_crawling.py`) |
| Box mesh builder | `TetraBox` |
| Chamber layout | `num_chambers_x`, `num_chambers_y`, `num_chambers_z`; `tet_chamber_mask`, `spring_chamber_mask` |
| Left/right contact and joints | `_bottom_y_plus_indices`, `_bottom_y_minus_indices`; `_joint_left_indices` (φ₁), `_joint_right_indices` (φ₂) |
| Gait | `gait_freq`, `gait_amplitude`, `gait_phase`, `gait_baseline`; chamber pressures for ch1 (left), ch3 (right) |
| Stick-slip | `use_crawlable_stick_slip`; `set_crawl_contact_groups`, `set_gait_params`; `paper_model` (Δ, d, φ₁, φ₂) |

## What This Documentation Covers

- **Implicit integration** and system matrix (in `02_solver_soft.md`).
- **FEM and springs**: tetrahedra, triangles, spring stiffness/damping (in `02_solver_soft.md`).
- **Self-collision**: vertex–triangle, edge–edge, BVH, repulsion, friction (in `03_solver_deformable.md`).
- **Inflation and chambers**: rest-config scaling, anisotropy, per-chamber pressures (in `04_solver_inflatable.md`).
- **Torque (folding)**: spine stiffness via springs with rest direction (in `04_solver_inflatable.md`).
- **Slip and grip**: paper stick-slip state machine, tangential force, kinematic displacement ±Δd (in `05_solver_crawlable.md`).
- **Example structure**: geometry, chamber layout, gait, contact groups, validation (in `06_example_inchworm.md`).
- **Mathematical summary**: equations in one place (in `07_mathematical_summary.md`).

The glue layer (`_src/glue`) is not described here, as it is not required to understand the inchworm crawling example or the paper-aligned physics.
