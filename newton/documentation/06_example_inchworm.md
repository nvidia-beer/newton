# Inchworm Crawling

This section describes the inchworm crawling example: a soft body with multiple inflatable chambers driven by a **traveling-wave gait**, simulated with **`SolverInflatable`** and **Coulomb ground contact**. Code and parameters live under `newton/examples/crawlable/` (metrics and JSON in the `inchworm/` subfolder).

## What the example does

The robot is a **soft box** (tetrahedral mesh from `TetraBox`) with a chamber grid along the body. Selected chambers are actuated by time-varying **volume-ratio pressures** (`set_chamber_pressures`); inactive chambers can stay near rest stiffness (`chamber_inflation_disabled`, `chamber_stiffness_scale`). **`gait_traveling_wave.py`** orchestrates a phase-shifted pattern so the body arches and relaxes in sequence. Directed motion comes from **friction with the ground plane** and the gait-driven deformation.

## Simulation and key parameters

The figures below show the inchworm in the 3D simulation and how **joints** (φ₁, φ₂) and **legs** (ground-contact points) are defined in the particle model.

![Soft inchworm segment in 3D simulation: arch-shaped body on the ground plane, with mesh and contact/joint markers.](figures/inchworm.png)

![Particle view: blue lines = leg (ground-contact) particles; green line = joint (φ₁, φ₂) particles; faint cloud = rest of the soft body.](figures/inchworm_joints_legs_3d.png)

![Y–Z slice (paper Fig. 3): ground contacts (blue), joints φ₁ and φ₂ (green). Forward motion along Y (crawl axis) not shown.](figures/inchworm_joints_legs_2d.png)

**Key parameters** (in `inchworm_params.json`), by effect:

- **Geometry — length, subdivisions**  
  The body is built from a box; its size is set by length, width, and height. Subdivisions set how many cells there are along each axis. Finer subdivisions give a smoother bend and more stable contact at higher compute cost.

- **Physics — torque axis stiffness**  
  Springs along the long axis resist bending. Their stiffness (and damping) controls how strongly the body returns to straight; higher values make the gait stiffer and more responsive.

- **Inflation — active chambers and max pressure**  
  Per-chamber flags and stiffness scales select which regions are driven by the gait. A global `max_pressure` caps inflation.

- **Gait — amplitude and baseline**  
  Amplitude is the pressure swing in actuated chambers (larger ⇒ more bend). Baseline is the mean pressure (higher ⇒ more lift). Together they set arching during the gait cycle.

## Geometry and chambers

The mesh is a subdivided box. Chambers are laid out along the box (e.g. multiple slices along Y and Z). Each tetrahedron and spring is assigned to a chamber by position. Inflating opposing sides bends the body.

## Gait and contact

After a short settle phase, the traveling-wave gait updates chamber pressures each frame or substep. **`SolverInflatable`** integrates the soft body with implicit steps; ground interaction uses the **soft solver’s particle–plane contact** (normal penalty + Coulomb friction, with friction magnitude capped relative to effective normal / weight—see `SolverSoft`). Bottom vertices are classified for **validation** and **paper-style metrics** (`get_paper_metrics`, joint groups) when \(z\) is below a threshold. Optional CSV logging records Y–Z metrics for analysis.

## Running the example

From the repo, run via the devcontainer script (e.g. `run-examples.sh inchworm_crawling`) or `python -m newton.examples inchworm_crawling`. Parameters are read from JSON (e.g. `inchworm/inchworm_params.json`).

For code and parameter tables, see **Appendix B**.
