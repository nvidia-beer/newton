# Overview: Paper to Implementation

## Paper Reference

**Gamus et al., "Understanding Legged Crawling for Soft Robots," arXiv:1911.05227**

The inchworm crawling example implements the same physical scenario as in the paper: a soft robot with two bending segments (left/right along the beam), inflated chambers to produce bending, and a phase-shifted harmonic gait. **Ground contact** uses **Coulomb friction** on the analytic ground plane (`SolverSoft` / `SolverDeformable` / `SolverInflatable`), with friction magnitude capped relative to effective normal and weight where applicable (see `SolverSoft` / `eval_particle_ground_contacts`). The paper’s kinematic notation and Method 1 algebra are summarized in **`02_paper_model.md`**, **`method1_prescribed_angles.md`**, and **§7** of **`07_mathematical_summary.md`**.

## Solver Stack (Inheritance)

The implementation is layered; each solver extends the previous one:

```
SolverSoft          → implicit integration, FEM (tetrahedra, triangles), springs, ground plane
    ↓
SolverDeformable   → + self-collision (vertex–triangle, edge–edge, BVH)
    ↓
SolverInflatable   → + inflation (rest-config scaling), per-chamber pressures, torque springs (folding)
```

The inchworm example **`example_inchworm_crawling.py`** uses **`SolverInflatable`**, builds the body with **`TetraBox`**, and drives pressures with **`gait_traveling_wave.py`** (traveling-wave / phase-shifted chamber pressures).

## Main Code Names (Same as in Code)

| Concept | Code / Doc name |
|--------|------------------|
| Soft body solver (base) | `SolverSoft` |
| With self-collision | `SolverDeformable` |
| With inflation and chambers | `SolverInflatable` |
| Inchworm example class | `Example` (in `example_inchworm_crawling.py`) |
| Box mesh builder | `TetraBox` |
| Chamber layout | `num_chambers_x`, `num_chambers_y`, `num_chambers_z`; `tet_chamber_mask`, `spring_chamber_mask` |
| Left/right contact and joints | `_bottom_y_plus_indices`, `_bottom_y_minus_indices`; `_joint_left_indices` (φ₁), `_joint_right_indices` (φ₂) |
| Gait | `TravelingWaveGait`, `CrawlPressureOrchestratorBase`; params `gait_freq`, `gait_amplitude`, `gait_phase`, `gait_baseline`; chamber pressures for actuated chambers |

## What This Documentation Covers

- **Implicit integration** and system matrix (in `02_solver_soft.md`).
- **FEM and springs**: tetrahedra, triangles, spring stiffness/damping (in `02_solver_soft.md`).
- **Self-collision**: vertex–triangle, edge–edge, BVH, repulsion, friction (in `03_solver_deformable.md`).
- **Inflation and chambers**: rest-config scaling, per-chamber isotropic pressures (in `04_solver_inflatable.md`).
- **Torque (folding)**: spine stiffness via springs with rest direction (in `04_solver_inflatable.md`).
- **Example structure**: geometry, chamber layout, gait, contact groups, validation (in `06_example_inchworm.md`).
- **Mathematical summary**: equations in one place (in `07_mathematical_summary.md`).

The glue layer (`_src/glue`) is not described here, as it is not required to understand the inchworm crawling example or the paper-aligned physics.
