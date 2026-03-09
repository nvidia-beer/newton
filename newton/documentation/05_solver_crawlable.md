# SolverCrawlable: Stick-Slip and Paper Kinematics

**Location:** `newton/_src/solvers/crawlable/solver_crawlable.py`, `kernels_crawlable.py`, `paper_model.py`

## Role

`SolverCrawlable` extends `SolverInflatable` with the **paper’s stick-slip ground contact** (Gamus et al., arXiv:1911.05227). The robot is modelled as a three-link system in the plane (e.g. Y–Z): two “feet” (left and right contact groups) and two “joints” (φ₁, φ₂). Depending on the sign of Δ (slippage criterion), one foot sticks and the other slips; when slipping, a tangential force \(f_t = \mu f_n\) is applied and the **body is displaced kinematically** by ±Δd along the crawl axis. This produces net crawling motion.

## Paper Model (paper_model.py)

The file `paper_model.py` implements the kinematics and hybrid contact model from the paper.

### Geometry and Conventions

- **Link length:** \(l = L/(2+\beta)\). Parameters \(M, L, \beta, \mu, g, k\) come from the solver object via `set_gait_params(M=..., L=..., ...)` (defaults match paper Table I).
- **Central link orientation:** \(\theta\) with \(\tan\theta = (\sin\phi_1 - \sin\phi_2)/(\cos\phi_1 + \cos\phi_2 - \beta)\). Function: `compute_theta(phi1, phi2, beta)`.
- **Horizontal distance between contacts:** \(d = l\bigl(\beta\cos\theta - \cos(\phi_1-\theta) - \cos(\phi_2+\theta)\bigr)\). Function: `compute_d`.
- **CoM horizontal offset:** \(x_c\) from left contact; function `compute_xc`.
- **Slippage criterion:** \(\Delta = x_c - d/2\). \(\Delta > 0\) ⇒ left foot slips; \(\Delta < 0\) ⇒ right foot slips. Also implemented as `compute_delta_direct`: \(\Delta \propto (1+\beta)/(2(2+\beta))\,l\,\bigl[\cos(\phi_2+\theta) - \cos(\phi_1-\theta)\bigr]\).

### Normal Forces

- **Contact normal forces:** \(f_{n1}\), \(f_{n2}\) from equilibrium (function `compute_fn(xc, d, M, g)`): \(f_{n1} = (1 - x_c/d)Mg\), \(f_{n2} = (x_c/d)Mg\) when \(d>0\).

### Gait (Reference Angles)

- **Harmonic gait (paper):** \(\phi_{1}^{\mathrm{ref}} = \gamma + A\sin(\omega t + \psi/2)\), \(\phi_{2}^{\mathrm{ref}} = \gamma + A\sin(\omega t - \psi/2)\). Simulation uses mesh angles from `joint_angles_from_positions`, not prescribed torques.

### State Machine

- **States:** `SLIP_STICK` (left slips), `STICK_SLIP` (right slips).
- **State machine:** `crawl_state_step_simple(...)` uses Δ to decide which foot slips; returns slip leg, \(f_t\), slip direction, \(d\), and \(f_{n1}, f_{n2}\).

### Joint Angles from Mesh

- **From positions:** `joint_angles_from_positions(left_contact, left_joint, right_joint, right_contact, crawl_axis, vertical_axis)` returns \((\phi_1,\phi_2)\) from the 4 node positions in the 2D plane (crawl_axis × vertical_axis). Used each step to feed the state machine.

## Method 1 (Prescribed Joint Angles) — Implementation Without Prescribed Angles

A full explanation of **Method 1: Prescribed Joint Angles Only** (paper definition and implementation with inferred angles and hysteresis) is in **method1_prescribed_angles.md**. Summary below.

The paper’s **Method 1** assumes \(\varphi_i(t) = \phi_i(t)\): the joint angles are the *inputs*, and only the kinematic relation for \(\dot{d}\) (paper Eq. (8)), the slippage criterion \(\Delta\), and \(f_t = \mu f_{n,s}\,\mathrm{sign}(\dot{d})\) are needed to decide which foot slips and with what force. In the **simulation angles are not prescribed**; bending is driven by **chamber pressure** (inflation), and the resulting deformable mesh determines the shape. The same Method 1 logic is implemented as follows.

1. **Infer angles from the mesh each step.**  
   From the current particle positions the mean positions of the left contact, right contact, left joint, and right joint are taken (the groups set by `set_crawl_contact_groups`). \((\phi_1,\phi_2)\) is then computed via `joint_angles_from_positions(...)` in the 2D plane (crawl_axis × vertical_axis). The “prescribed” angles in the paper are thus replaced by **angles read off the current state** of the soft body.

2. **Use the same Method 1 equations.**  
   With \((\phi_1,\phi_2)\) the code computes \(\theta\), \(d\), \(x_c\), and \(\Delta = x_c - d/2\) (paper geometry and slippage criterion). Which foot slips is decided from the sign of \(\Delta\); the slip force magnitude is set to \(\mu f_{n,s}\) and the direction from the sign of \(\dot{d}\).

3. **Time derivative of contact distance \(\dot{d}\) (paper Eq. (8)).**  
   Method 1 uses \(\dot{d} = l\bigl[\sin(\varphi_2+\theta)(\dot{\varphi}_2+\dot{\theta}) + \sin(\varphi_1-\theta)(\dot{\varphi}_1-\dot{\theta}) - \beta\sin\theta\,\dot{\theta}\bigr]\). Prescribed \(\dot{\varphi}_i\) are not available; they are approximated from the **previous** step: \(\dot{\varphi}_i \approx (\varphi_i - \varphi_{i,\mathrm{prev}})/\Delta t\). When both \(\phi_1\) and \(\phi_2\) from the previous step are available, \(\dot{\theta}\) is computed from \(\theta(\phi_1,\phi_2)\) and the full Eq. (8) is used in `compute_d_dot_from_angular_velocities`; otherwise the code falls back to \(\dot{d} \approx (d - d_{\mathrm{prev}})/\Delta t\). The result is “Method 1 with angles from the mesh and \(\dot{d}\) from Eq. (8) when possible.”

4. **Kinematic displacement.**  
   After the step the same rule as the paper is applied: body displacement by \(\pm\Delta d\) along the crawl axis according to which foot slipped, via `apply_crawl_kinematic_displacement`.

The implementation is **Method 1 in the sense of the paper** (no torque balance solve; only geometry, \(\Delta\), \(\dot{d}\), and slip force), with “prescribed” angles replaced by **angles inferred from the deformable mesh** at each step. The gait (phase-shifted sines) is applied in **pressure** space in the example; the resulting motion yields the \(\phi_1,\phi_2\) that drive the state machine.

**Hysteresis for rectangular \(f_t\) (Fig. 6):** Because inferred angles can jitter, \(\dot{d}\) may cross zero often. In `crawl_state_step_simple` a dead band `d_dot_eps = 1e-5` is used: when \(|\dot{d}| < d\_{\mathrm{dot\_eps}}\) the *previous* slip direction is kept so \(f_t\) stays flat at \(\pm\mu f_n\) and the plot of \(f_t\) vs \(t/T\) is rect-shaped. See **method1_prescribed_angles.md** (§4).

## Crawlable Solver Setup

- **Contact groups:** `set_crawl_contact_groups(model, left_contact_indices, right_contact_indices, left_joint_indices, right_joint_indices)`. These are the particle indices for the two “feet” and the two “joint” lines (φ₁, φ₂). Stored as `_crawl_left_indices`, `_crawl_right_indices`, `_crawl_left_joint_indices`, `_crawl_right_joint_indices`; binary arrays `_crawl_particle_in_left`, `_crawl_particle_in_right` for the kernel.
- **Gait and physics:** `set_gait_params(gamma, A, freq_hz, psi, k, M, L, beta, mu, g, crawl_axis, slip_force_scale, crawl_direction, use_paper_ratios, contact_constraint_stiffness)`. `crawl_direction` = +1 or -1 for direction of crawl along the crawl axis. `contact_constraint_stiffness` (optional): if > 0, both feet are constrained to same height (paper Eq. 2).

## Ground Contact Force (Stick-Slip)

When crawl is enabled, `eval_particle_ground_contact_forces` does **not** use the default Coulomb kernel for the crawl axis; it uses the paper stick-slip model:

1. Compute centroids of left/right contact and left/right joint from current `state.particle_q`.
2. Compute \(\phi_1,\phi_2\) via `joint_angles_from_positions`.
3. Run one step of the state machine (`crawl_state_step_simple`) to get `(slip_leg_state, ft_signed, slip_dir, d, fn1, fn2)`; update `_crawl_d_prev = d`, `_crawl_last_slip_dir`, `_crawl_last_slip_leg`, and (for CSV only) raw normal force sums per foot.
4. If bend is small (e.g. \(\max(|\phi_1-\pi|,|\phi_2-\pi|) < \mathrm{min\_bend}\)), force stick-stick (no slip force).
5. Launch `eval_particle_ground_contacts_crawl` with `slip_leg` (0=left slip, 1=right slip, 2=both stick), `ft_magnitude`, `slip_direction_sign`, `crawl_axis`, `slip_force_scale`.

**Kernel behaviour:**

- Normal force: same as standard ground contact (penetration + damping).
- **Tangential:**  
  - **Slipping group:** apply force in slip direction, magnitude per particle = `ft_magnitude * slip_force_scale / n_left` (or n_right), capped by Coulomb \(\mu f_{n,\mathrm{eff}}\) so the paper relation \(|f_t| = \mu f_n\) is respected and no moment lifts the leg.  
  - **Sticking group:** Coulomb friction in the tangent plane; optionally zero out the component along the crawl axis so only the slip leg drives motion.

## Kinematic Displacement (Paper Rule)

After the implicit step, the paper says: “Update contact positions: sticking contact unchanged, slipping contact position changes by Δd in slip direction.” So the **body** (all particles) is displaced by ±Δd along the crawl axis:

- \(\Delta d = d_{\mathrm{new}} - d_{\mathrm{prev}}\) from the current geometry after the step.
- If left slipped: body displacement = \(-\Delta d \times \mathrm{crawl\_direction}\) (so the left contact “moves back” relative to body).
- If right slipped: body displacement = \(+\Delta d \times \mathrm{crawl\_direction}\).
- Kernel `apply_crawl_kinematic_displacement(particle_q, crawl_axis, displacement)` adds this displacement to every particle’s position. No velocity kick; the motion is kinematic as in the paper.

Implemented in `SolverCrawlable.step`: after `super().step(...)`, if crawl is enabled and `_crawl_apply_kinematic_disp` is True, compute \(d_{\mathrm{new}}\) from the new state, then apply the displacement.

## Naming in Code

- `set_crawl_contact_groups`, `set_gait_params`, `set_crawl_time`
- `_crawl_left_indices`, `_crawl_right_indices`, `_crawl_left_joint_indices`, `_crawl_right_joint_indices`
- `_crawl_d_prev`, `_crawl_last_slip_dir`, `_crawl_last_slip_leg`, `_crawl_apply_kinematic_disp`, `_crawl_last_fn_left_raw`, `_crawl_last_fn_right_raw` (raw F·n sums for CSV logging only)
- `eval_particle_ground_contacts_crawl`, `apply_crawl_kinematic_displacement`
- `compute_d`, `compute_xc`, `compute_delta_direct`, `joint_angles_from_positions`, `crawl_state_step_simple`
- `slip_leg`, `ft_magnitude`, `slip_direction_sign`, `crawl_axis`, `crawl_direction`, `slip_force_scale`
