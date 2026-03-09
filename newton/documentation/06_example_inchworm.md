# Inchworm Crawling

This section describes the inchworm crawling example: a soft body with two bending segments and two inflatable chambers that crawls using the paper’s stick–slip rule. The implementation lives in `newton/examples/crawlable/` (script, params, and paper metrics in the `inchworm/` subfolder).

## What the example does

The robot is a **soft box** (tetrahedral mesh) with **four chambers** along the body. Two chambers (left and right on top) are used for actuation; inflating them bends the body. A **phase-shifted harmonic gait** drives the two chamber pressures so the body arches and relaxes in sequence. With **stick–slip** enabled, the simulation applies the paper’s friction and kinematic step: at each step one “foot” (one set of bottom vertices) slips while the other sticks, and the body advances by a computed displacement. The result is directed crawling along the crawl axis (Y in the example).

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
  A per-chamber flag selects which chambers are driven by the gait (typically the two top ones). A global max pressure caps inflation so the mesh does not over-inflate.

- **Slip and grip — gait amplitude and baseline**  
  Amplitude is the pressure swing in each active chamber (larger ⇒ more bend and step size). Baseline is the mean pressure (higher ⇒ body lifts more and grip can improve). Together they set arching and foot pressure during the stick–slip cycle.

## Geometry and chambers

The mesh is a subdivided box. Chambers are laid out along the box (e.g. 2 along the width × 1 × 2 in height). Each tetrahedron and each spring is assigned to a chamber by position. The two “top” chambers (left and right) are the actuated ones; the others form the backbone and are not driven. Inflating a top chamber bends the body downward on that side.

## Gait and contact

After a short settle phase (fixed pressure, no motion), the gait runs: left and right chamber pressures follow a phase-shifted sine so the body alternates which side is arched. The solver uses the paper’s contact rule: normal forces from equilibrium, slippage criterion to choose which foot slips, and a kinematic step that advances the body. Bottom vertices are treated as “in contact” when their height is at or below a small threshold; that set is used for validation and for the stick–slip groups. Optional CSV logging records paper-aligned metrics (contact positions, joint positions, etc.) for analysis.

## Running the example

From the repo, run the example via the devcontainer script (e.g. `run-examples.sh inchworm_crawling [--crawlable]`) or from inside the container with the project’s Python module. Parameters are read from a JSON file (e.g. `inchworm/inchworm_params.json`); paths and options are documented in the script.

For the full code and parameter tables and the implementation map (paper equations and crawlable solver → code), see **Appendix B**.
