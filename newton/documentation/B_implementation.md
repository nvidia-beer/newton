# Implementation Map

Minimal reference for locating the mathematics in the codebase. For the paper model see Section 2; for the dynamic extension Section 3; for the solver layers Section 4.

**Solver layers:** Soft → Deformable → Inflatable → Crawlable (each extends the previous). Implemented under `newton/_src/solvers/` in `soft/`, `deformable/`, `inflatable/`, `crawlable/`.

## Paper model (Section 2) — `crawlable/paper_model.py`

| Equation / concept | Code |
|-------------------|------|
| \(l\), \(\theta\), \(d\), \(x_c\) | `compute_l()`, `compute_theta()`, `compute_d()`, `compute_xc()` |
| Normal forces \(f_{n1}, f_{n2}\); tangential balance \(f_{t,1}=-f_{t,2}\) | `compute_fn()`; single \(f_t\) in state machine |
| \(\Delta\), \(\dot{d}\) (paper Eq. (8)) | `compute_delta_direct()`, `compute_d_dot()`; solver uses \((d-d_{\mathrm{prev}})/\Delta t\) |
| Signed slip force \(f_t = \mu f_{n,s}\operatorname{sign}(\dot{d})\); hybrid states | `crawl_state_step_simple()`; slip direction passed to kernel |
| Joint angles from mesh | `joint_angles_from_positions()` |

## Crawlable solver — `solver_crawlable.py`, `kernels_crawlable.py`

Slip force direction: \(\operatorname{sign}(\dot{d}) \times\) crawl direction in `eval_particle_ground_contacts_forces()`; kinematic displacement \(\pm\Delta d\) in `apply_crawl_kinematic_displacement`.

**Inchworm example** (geometry, chambers, gait, contact groups): `example_inchworm_crawling.py` in `newton/examples/crawlable/`.

**Main parameter names:** Paper: \(L\), \(M\), \(\beta\), \(\mu\), \(k\); gait: \(\gamma\), \(A\), \(\omega\), \(\psi\). Inflation: chamber pressures \(p_c\), anisotropy; torque: \(k_\tau\), \(k_d\), rest direction. Crawl: left/right contact and joint indices, crawl axis, crawl direction.
