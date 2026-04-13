# Code and implementation map

This appendix maps the inchworm example (Section 5) to code names and parameters. Ground physics use **`SolverInflatable`** with **Coulomb friction** on `add_ground_plane`; chamber actuation uses **`gait_traveling_wave.py`**.

## Code and parameter reference

| Concept | Code / parameter |
|--------|-------------------|
| **Mesh** | `TetraBox(size=(length, width, height), subdivisions=(subdivisions_x, subdivisions_y, subdivisions_z))` |
| **Chamber layout** | `num_chambers_x`, `num_chambers_y`, `num_chambers_z`; assignment by centroid; `chamber_inflation_disabled` |
| **Active chambers** | `chamber_active_inflation` (list: 1 = gait-driven, 0 = not) |
| **Chamber masks / pressures** | `tet_chamber_mask`, `spring_chamber_mask`; `set_chamber_mask(...)`; `set_chamber_pressures(...)` |

| Concept | Code / parameter |
|--------|-------------------|
| **Bottom contact (feet)** | `_bottom_y_plus_indices`, `_bottom_y_minus_indices` |
| **Joints φ₁, φ₂** | `_joint_left_indices`, `_joint_right_indices` (bottom-edge vertices at paper fractions along Y for `paper_beta`) |
| **Helper index sets** | `_bottom_edge_vertex_indices`, `_joint_cross_section_vertex_indices_y`, `_inchworm_contact_and_joint_vertex_indices` |

| Concept | Code / parameter |
|--------|-------------------|
| **Ground** | `add_ground_plane` (stiffness, friction `mu`) |
| **Soft body** | `add_soft_mesh` (vertices/indices/tets from TetraBox, density, \(k_\mu\), \(k_\lambda\), \(k_{\mathrm{damp}}\)) |
| **Springs** | All tet edges; chamber by centroid; torque springs use `torque_stiffness`, `torque_damping`, `spring_rest_direction`; `_torque_spring_indices` |
| **Solver** | `SolverInflatable` (`newton.solvers` / `newton._src.solvers.inflatable`) |

| Concept | Code / parameter |
|--------|-------------------|
| **Gait** | `TravelingWaveGait`, `CrawlPressureOrchestratorBase`; pressures from `gait_freq`, `gait_amplitude`, `gait_phase`, `gait_baseline` (and optional min/max from JSON) |
| **Contact threshold** | Bottom vertex “in contact” if \(z \le\) threshold (from `particle_radius` / `_contact_z_threshold`) |
| **Validation** | `validate_bottom_contact(state)`; optional `stop_on_lost_contact` in `run` |
| **Display colors** | Blue = in contact, hot pink = above threshold, green = joint vertices |

| Concept | Code / parameter |
|--------|-------------------|
| **Paper metrics** | `get_paper_metrics(...)` in `inchworm/paper.py` |
| **CSV logging** | `InchwormValidation`, `csv_log_interval` |
| **Entry** | `Example` in `example_inchworm_crawling.py`; `main()` → `Example(...).run(...)` |
| **CLI** | `run-examples.sh inchworm_crawling` or `python -m newton.examples inchworm_crawling` |

| Parameter group | Names (same as in JSON/code) |
|-----------------|------------------------------|
| Geometry | `length`, `width`, `height`, `subdivisions_*`, `num_chambers_*` |
| Material | `k_mu`, `k_lambda`, `k_damp`, `spring_ke`, `spring_kd`, `total_mass`, `gravity` |
| Inflation | `max_pressure`, `chamber_stiffness_scale`, `chamber_inflation_disabled` |
| Torque | `torque_stiffness`, `torque_damping` |
| Contact | `ground_friction`, `contact_offset`, `contact_iterations`, `ground_ke`, `particle_radius` |
| Gait | `gait_enabled`, `gait_freq`, `gait_amplitude`, `gait_phase`, `gait_baseline`, `settle_seconds`, optional `gait_pressure_min` / `max` |
| Run | `num_frames`, `validate_contact`, `stop_on_lost_contact`, `csv_log_interval` |

## Paper model (reference)

Geometry and notation from Gamus et al. (arXiv:1911.05227) appear in **`02_paper_model.md`**, **`method1_prescribed_angles.md`**, and **§7** of **`07_mathematical_summary.md`**. CSV metrics and validation use bottom/joint vertex groups from **`inchworm/paper.py`**.

**Parameter names (paper vs code):** Paper: \(L\), \(M\), \(\beta\), \(\mu\), \(k\); gait: \(\gamma\), \(A\), \(\omega\), \(\psi\). Inflation: chamber pressures \(p_c\); torque: \(k_\tau\), \(k_d\), rest direction. Indices: left/right bottom lines and joint lines for φ₁, φ₂.
