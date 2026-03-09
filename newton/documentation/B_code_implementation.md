# Code and implementation map

This appendix collects the code and parameter reference for the inchworm example (Section 5) and the implementation map linking the paper’s mathematics and the crawlable solver to the codebase.

## Code and parameter reference

Tables below map the concepts in Section 5 to the actual code names and parameters for implementation and tuning.

| Concept | Code / parameter |
|--------|-------------------|
| **Mesh** | `TetraBox(size=(length, width, height), subdivisions=(subdivisions_x, subdivisions_y, subdivisions_z))` |
| **Chamber layout** | `num_chambers_x`, `num_chambers_y`, `num_chambers_z`; assignment by centroid, mask for disabled chambers |
| **Active chambers** | `chamber_active_inflation` (list: 1 = driven by gait, 0 = not); chambers 1 and 3 = top left/right |
| **Chamber masks / pressures** | `tet_chamber_mask`, `spring_chamber_mask`; `set_chamber_mask(...)`; `chamber_pressures`, `set_chamber_pressures(...)` |

| Concept | Code / parameter |
|--------|-------------------|
| **Bottom contact (feet)** | `_bottom_y_plus_indices`, `_bottom_y_minus_indices` (bottom vertices Y+ and Y− edges) |
| **Joints φ₁, φ₂** | `_joint_left_indices`, `_joint_right_indices` (bottom-edge vertices at 1/(2+β) and (1+β)/(2+β) along crawl axis Y; e.g. 1/4 and 3/4 for β=2) |
| **Helper index sets** | `_bottom_edge_vertex_indices`, `_top_edge_vertex_indices`, `_joint_cross_section_vertex_indices_y`, `_inchworm_contact_and_joint_vertex_indices` |

| Concept | Code / parameter |
|--------|-------------------|
| **Ground** | `add_ground_plane` (stiffness; mu=0 when stick–slip so crawl kernel applies friction) |
| **Soft body** | `add_soft_mesh` (vertices/indices/tets from TetraBox, density from `total_mass`/volume, \(k_\mu\), \(k_\lambda\), \(k_{\mathrm{damp}}\)) |
| **Springs** | All tet edges; chamber by centroid; torque springs (long-axis) use `torque_stiffness`, `torque_damping`, `spring_rest_direction`; listed in `_torque_spring_indices` |
| **Solver** | `SolverCrawlable(model, dt, mass, ...)` with contact and stick–slip options |
| **Stick–slip setup** | `set_crawl_contact_groups(model, left_contact, right_contact, joint_phi1, joint_phi2)`, `set_gait_params(..., crawl_axis=1, L=width, M=total_mass, ...)` |

| Concept | Code / parameter |
|--------|-------------------|
| **Gait pressures** | Left = baseline + amplitude·sin(ωt), right = baseline + amplitude·sin(ωt + phase); written to active chamber pressures each step |
| **Gait params** | `gait_freq`, `gait_amplitude`, `gait_phase`, `gait_baseline`, `settle_seconds` |
| **Contact threshold** | Bottom vertex in contact if \(z \le\) contact threshold (e.g. 2× particle_radius or 0.002) |
| **Validation** | `validate_bottom_contact(state)`; optional `stop_on_lost_contact` in run |
| **Display colors** | Blue = in contact, hot pink = lost contact, green = joint vertices |

| Concept | Code / parameter |
|--------|-------------------|
| **Paper metrics** | `get_paper_metrics(particle_q, bottom_y_plus, bottom_y_minus, joint_left, joint_right)` in `inchworm/paper.py` (e.g. `y_left_ground`, `z_left_ground`, …) |
| **CSV logging** | `InchwormValidation`, `csv_log_interval`; output under `inchworm/` |
| **Entry** | Class `Example` in `example_inchworm_crawling.py`; `main()` → `Example(...).run(...)` |
| **CLI** | `run-examples.sh inchworm_crawling [--crawlable] [options]` or `python -m newton.examples inchworm_crawling ...` |

| Parameter group | Names (same as in JSON/code) |
|-----------------|------------------------------|
| Geometry | `length`, `width`, `height`, `subdivisions_x`, `subdivisions_y`, `subdivisions_z`, `num_chambers_x`, `num_chambers_y`, `num_chambers_z` |
| Material | `k_mu`, `k_lambda`, `k_damp`, `spring_ke`, `spring_kd`, `total_mass`, `gravity` |
| Inflation | `max_pressure`, `chamber_stiffness_scale`, `chamber_inflation_disabled`, `anisotropy_x`, `anisotropy_y`, `anisotropy_z` |
| Torque | `torque_stiffness`, `torque_damping` |
| Contact | `ground_friction`, `contact_offset`, `contact_iterations`, `ground_ke`, `particle_radius` |
| Gait | `gait_enabled`, `gait_freq`, `gait_amplitude`, `gait_phase`, `gait_baseline`, `settle_seconds` |
| Crawl | `use_crawlable_stick_slip`, `stick_slip_scale`, `stick_slip_amplitude`, `crawl_direction`, `contact_constraint_stiffness` |
| Run | `num_frames`, `validate_contact`, `stop_on_lost_contact`, `csv_log_interval` |

## Implementation map (paper and crawlable solver)

Where to find the paper’s mathematics and the crawl logic used by the inchworm example. The solver stack is **Soft → Deformable → Inflatable → Crawlable** (each extends the previous), under `newton/_src/solvers/` in `soft/`, `deformable/`, `inflatable/`, `crawlable/`.

**Paper model (Section 2)** — `crawlable/paper_model.py`:

| Equation / concept | Code |
|-------------------|------|
| \(l\), \(\theta\), \(d\), \(x_c\) | `compute_l()`, `compute_theta()`, `compute_d()`, `compute_xc()` |
| Normal forces \(f_{n1}, f_{n2}\); tangential balance \(f_{t,1}=-f_{t,2}\) | `compute_fn()`; single \(f_t\) in state machine |
| \(\Delta\), \(\dot{d}\) (paper Eq. (8)) | `compute_delta_direct()`, `compute_d_dot()`; solver uses \((d-d_{\mathrm{prev}})/\Delta t\) |
| Signed slip force \(f_t = \mu f_{n,s}\operatorname{sign}(\dot{d})\); hybrid states | `crawl_state_step_simple()`; slip direction passed to kernel |
| Joint angles from mesh | `joint_angles_from_positions()` |

**Crawlable solver** — `solver_crawlable.py`, `kernels_crawlable.py`: slip force direction \(\operatorname{sign}(\dot{d}) \times\) crawl direction in `eval_particle_ground_contact_forces()`; kinematic displacement \(\pm\Delta d\) in `apply_crawl_kinematic_displacement`.

**Inchworm example:** geometry, chambers, gait, contact groups → `example_inchworm_crawling.py` in `newton/examples/crawlable/`.

**Parameter names (paper vs code):** Paper: \(L\), \(M\), \(\beta\), \(\mu\), \(k\); gait: \(\gamma\), \(A\), \(\omega\), \(\psi\). Inflation: chamber pressures \(p_c\), anisotropy; torque: \(k_\tau\), \(k_d\), rest direction. Crawl: left/right contact and joint indices, crawl axis, crawl direction.
