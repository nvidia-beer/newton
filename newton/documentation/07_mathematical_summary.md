# Mathematical Summary: Equations and Implementation

This document collects the main equations used in the implementation, with names that match the code and the previous sections. It supports a rigorous link between the paper (arXiv:1911.05227) and the Newton inchworm crawling simulation.

---

## 1. Implicit Integration (SolverSoft)

- **Unknown:** \(\Delta\mathbf{v}\) (velocity change).
- **System:** \(A\,\Delta\mathbf{v} = \mathbf{f}\), with
  \[
  A = M - h D - h^2 K.
  \]
- **RHS:** \(\mathbf{f} = h\,\sum \mathbf{F}\) (sum of all force contributions, then multiplied by timestep \(h\)).
- **Update:** \(\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta\mathbf{v}\), \(\mathbf{q}^{n+1} = \mathbf{q}^n + h\,\mathbf{v}^{n+1}\).

Mass \(M\) is lumped (diagonal); \(D\) and \(K\) come from springs, tetrahedra, and triangles (and optionally self-contact diagonal in Deformable).

---

## 2. Springs

- **Linear spring (edge \((i,j)\)):** rest length \(\ell_0\), stiffness \(k_e\), damping \(k_d\).
  \[
  \mathbf{F}_{\mathrm{spring}} = k_e(\ell - \ell_0)\,\mathbf{d} + k_d\,\dot{\ell}\,\mathbf{d}, \quad \ell = |\mathbf{x}_j - \mathbf{x}_i|,\quad \mathbf{d} = (\mathbf{x}_j - \mathbf{x}_i)/\ell.
  \]
- **Tangent in matrix:** contributes \(h^2 k_e \mathbf{d}\mathbf{d}^T\) and \(h k_d \mathbf{d}\mathbf{d}^T\) to diagonal and off-diagonal blocks for \(i,j\).

---

## 3. Tetrahedral FEM

- **Deformation gradient:** \(\mathbf{F} = \mathbf{D}_s \mathbf{D}_m^{-1}\) (current edges × rest inverse).
- **Rest pose:** stored as `tet_poses` (e.g. \(\mathbf{D}_m^{-1}\)); materials: \(k_\mu\), \(k_\lambda\), \(k_{\mathrm{damp}}\).
- **Neo-Hookean (or similar):** stress from \(\mathbf{F}\); forces from divergence of stress; tangent stiffness from linearization. Exact form in `eval_tetrahedra` and `build_system_matrix_tet_kernel`.

---

## 4. Self-Collision (SolverDeformable)

- **Repulsion (vertex–triangle or edge–edge):** distance \(d\), radius \(r\), stiffness \(k\).
  - If \(d \ge r\): \(F = 0\).
  - If \(\tau < d < r\) with \(\tau = 0.5 r\): \(F = k_2/d\) (C²-style).
  - Else: \(F = k\,(r - d)\) (linear penetration).
- **Force cap:** \(|F| \le F_{\mathrm{cap}}\) per pair.
- **Implicit tangent:** for vertices in vertex–triangle contact, add \(h^2 k_{\mathrm{eff}} \mathbf{I}\) to diagonal of \(A\).

---

## 5. Inflation (SolverInflatable)

- **Volume ratio:** \(p = V/V_0 \in [1, p_{\max}]\).
- **Linear scale:** \(s = p^{1/3}\).
- **Spring rest lengths:** \(\ell_0^{\mathrm{new}} = \ell_0^{\mathrm{orig}} \cdot s\).
- **Tet rest pose:** \(\mathbf{D}_m^{-1}\big|_{\mathrm{new}} = \mathbf{D}_m^{-1}\big|_{\mathrm{orig}} / s\) (isotropic). Anisotropic: scale columns by \(1/(s\,a_x)\), \(1/(s\,a_y)\), \(1/(s\,a_z)\).
- **Per-chamber:** tet/spring in chamber \(c\) uses \(s_c = p_c^{1/3}\) (and anisotropy if set).

---

## 6. Torque (Folding / Spine)

- **Rest direction:** \(\mathbf{d}_0\) (unit); current direction: \(\mathbf{d} = (\mathbf{x}_j - \mathbf{x}_i)/\ell\).
- **Angle:** \(\alpha = \arccos(\mathbf{d}\cdot\mathbf{d}_0)\) (spine angle \(\alpha\), not the paper’s \(\theta\)).
- **Rotation axis:** \(\mathbf{a} = (\mathbf{d}_0 \times \mathbf{d})/|\mathbf{d}_0 \times \mathbf{d}|\).
- **Angular velocity (along axis):** \(\omega\) from \(\mathbf{d}\) and \(\dot{\mathbf{d}}\).
- **Torque magnitude:** \(\tau = -(k_\tau \alpha + k_d \omega)\,\ell\).
- **Forces at endpoints:** \(\mathbf{F} = (\tau\,\mathbf{a} \times \mathbf{d})/\ell\) at \(i\), \(-\mathbf{F}\) at \(j\).

---

## 7. Paper Stick-Slip (SolverCrawlable)

### Geometry (paper_model)

- **Link length:** \(l = L/(2+\beta)\).
- **Central angle:** \(\tan\theta = \dfrac{\sin\phi_1 - \sin\phi_2}{\cos\phi_1 + \cos\phi_2 - \beta}\).
- **Contact distance:** \(d = l\bigl(\beta\cos\theta - \cos(\phi_1-\theta) - \cos(\phi_2+\theta)\bigr)\).
- **CoM from left contact:** \(x_c\) (see `compute_xc`).
- **Slippage criterion:** \(\Delta = x_c - d/2\). \(\Delta > 0 \Rightarrow\) left slips; \(\Delta < 0 \Rightarrow\) right slips.

### Normal forces

- \(f_{n1} = (1 - x_c/d) M g\), \(f_{n2} = (x_c/d) M g\) (for \(d > 0\)).

### Slip force

- When foot slips: \(|f_t| = \mu f_n\) on that foot; direction = slip direction (sign of \(\dot{d}\) or from state machine). In the kernel, tangential force is distributed over the slipping group and capped per particle by \(\mu f_{n,\mathrm{eff}}\) so the leg does not lift.

### Kinematic displacement

- After implicit step: \(\Delta d = d_{\mathrm{new}} - d_{\mathrm{prev}}\).
- If left slipped: displace all particles by \(-\Delta d \cdot \mathrm{sign}\) along crawl axis.
- If right slipped: displace by \(+\Delta d \cdot \mathrm{sign}\).
- This is the paper’s “update contact positions: slipping contact position changes by Δd”.

### Gait (reference angles)

- \(\phi_1^{\mathrm{ref}} = \gamma + A\sin(\omega t + \psi/2)\), \(\phi_2^{\mathrm{ref}} = \gamma + A\sin(\omega t - \psi/2)\).
- Actuation torques in the paper: \(\tau_i = k(\phi_i^{\mathrm{ref}} - \pi)\). In the simulation, bending is driven by chamber pressure; the state machine uses current \(\phi_1,\phi_2\) from mesh positions, not the reference angles directly.

### Method 1 without prescribed angles

The paper’s **Method 1: Prescribed Joint Angles Only** assumes prescribed joint angles \(\varphi_i(t)\). In the simulation \(\phi_1,\phi_2\) are **inferred** from the current mesh (mean positions of contact and joint groups → `joint_angles_from_positions`), then the same Method 1 equations are applied: \(\Delta\) for which foot slips, paper Eq. (8) for \(\dot{d}\) when \(\dot{\phi}_i\) are available from the previous step (else \((d - d_{\mathrm{prev}})/\Delta t\)), and \(f_t = \mu f_{n,s}\,\mathrm{sign}(\dot{d})\). A **hysteresis band** \(|\dot{d}| < d\_{\mathrm{dot\_eps}}\) (e.g. \(10^{-5}\)) keeps the previous slip direction so \(f_t(t)\) stays flat and Fig. 6 is rect-shaped. See **method1_prescribed_angles.md** for the full explanation and **05_solver_crawlable.md** (§ “Method 1 (Prescribed Joint Angles)”) for solver details.

---

## 8. Ground Contact (Force-Based)

- **Plane:** \(\mathbf{n}\cdot\mathbf{x} + d = 0\); penetration \(c = \mathbf{n}\cdot\mathbf{x} + d - r \le 0\) (r = particle radius).
- **Normal:** \(f_n = k_e c + k_d \min(\dot{c},0)\); \(\mathbf{F}_n = -f_n \mathbf{n}\).
- **Coulomb (sticking group or default):** \(|\mathbf{F}_t| \le \mu f_{n,\mathrm{eff}}\); tangent opposes slip direction.
- **Crawl kernel:** slip group gets \(f_t\) in slip direction (paper); stick group gets Coulomb with optional zero component along crawl axis.

---

## 9. Naming Index

| Concept | Symbol / name in doc and code |
|--------|---------------------------------|
| Timestep | \(h\), `dt` |
| System matrix | \(A\), `A_bsr` |
| Velocity change | \(\Delta\mathbf{v}\), `dv` |
| Spring stiffness / damping | \(k_e\), \(k_d\), `spring_stiffness`, `spring_damping` |
| Rest length | \(\ell_0\), `spring_rest_length` |
| Tet rest pose | \(\mathbf{D}_m^{-1}\), `tet_poses` |
| Pressure (volume ratio) | \(p\), `current_pressure`, `chamber_pressures` |
| Chamber index | \(c\), `tet_chamber_mask`, `spring_chamber_mask` |
| Joint angles | \(\phi_1,\phi_2\), `joint_angles_from_positions` |
| Contact distance | \(d\), `compute_d`, `_crawl_d_prev` |
| Slippage criterion | \(\Delta\), `compute_delta_direct` |
| Slip tangential force | \(f_t\), `ft_magnitude` |
| Kinematic displacement | \(\Delta d\), `apply_crawl_kinematic_displacement` |
| Torque stiffness / damping | \(k_\tau\), \(k_d\), `torque_stiffness`, `torque_damping` |

This summary is intended to be a single reference for the mathematics behind the implementation and its correspondence to the paper.
