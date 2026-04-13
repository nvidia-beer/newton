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
- **Tet rest pose:** \(\mathbf{D}_m^{-1}\big|_{\mathrm{new}} = \mathbf{D}_m^{-1}\big|_{\mathrm{orig}} / s\) (isotropic).
- **Per-chamber:** tet/spring in chamber \(c\) uses \(s_c = p_c^{1/3}\) (isotropic per region).

---

## 6. Torque (Folding / Spine)

- **Rest direction:** \(\mathbf{d}_0\) (unit); current direction: \(\mathbf{d} = (\mathbf{x}_j - \mathbf{x}_i)/\ell\).
- **Angle:** \(\alpha = \arccos(\mathbf{d}\cdot\mathbf{d}_0)\) (spine angle \(\alpha\), not the paper’s \(\theta\)).
- **Rotation axis:** \(\mathbf{a} = (\mathbf{d}_0 \times \mathbf{d})/|\mathbf{d}_0 \times \mathbf{d}|\).
- **Angular velocity (along axis):** \(\omega\) from \(\mathbf{d}\) and \(\dot{\mathbf{d}}\).
- **Torque magnitude:** \(\tau = -(k_\tau \alpha + k_d \omega)\,\ell\).
- **Forces at endpoints:** \(\mathbf{F} = (\tau\,\mathbf{a} \times \mathbf{d})/\ell\) at \(i\), \(-\mathbf{F}\) at \(j\).

---

## 7. Paper model: geometry and stick-slip (Gamus et al.)

The paper defines link geometry (\(l\), \(\theta\), contact distance \(d\), \(x_c\)), slippage \(\Delta = x_c - d/2\), normal forces \(f_{n1},f_{n2}\), slip force \(f_t = \mu f_{n,s}\operatorname{sign}(\dot{d})\), and (in the quasistatic paper) kinematic displacement \(\pm\Delta d\) after each step. **Method 1** and rectangular \(f_t(t)\) are discussed in **`method1_prescribed_angles.md`**.

The **inchworm example** couples this geometry to a **dynamic** simulation: **`SolverInflatable`**, **Coulomb friction** for ground contact (**§8**), and **chamber pressures** from **`gait_traveling_wave.py`**. Metrics use grouped vertex positions (`inchworm/paper.py`).

**Geometry (paper):**

- **Link length:** \(l = L/(2+\beta)\).
- **Central angle:** \(\tan\theta = (\sin\phi_1 - \sin\phi_2)/(\cos\phi_1 + \cos\phi_2 - \beta)\).
- **Contact distance:** \(d = l(\beta\cos\theta - \cos(\phi_1-\theta) - \cos(\phi_2+\theta))\).
- **Slippage criterion:** \(\Delta = x_c - d/2\).

**Gait (reference angles in paper):** \(\phi_i^{\mathrm{ref}} = \gamma + A\sin(\omega t \pm \psi/2)\). In Newton, bending is driven by **chamber pressures**, not explicit joint torques.

---

## 8. Ground Contact (Force-Based)

- **Plane:** \(\mathbf{n}\cdot\mathbf{x} + d = 0\); penetration \(c = \mathbf{n}\cdot\mathbf{x} + d - r \le 0\) (r = particle radius).
- **Normal:** \(f_n = k_e c + k_d \min(\dot{c},0)\); \(\mathbf{F}_n = -f_n \mathbf{n}\).
- **Coulomb:** \(|\mathbf{F}_t| \le \mu f_{n,\mathrm{eff}}\); tangent opposes slip velocity. Effective normal for the friction cap can be limited using particle mass and \(\|g\|\) so stiff penalties do not produce unphysical tangential forces (`SolverSoft` / `eval_particle_ground_contacts`).

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
| Joint angles (metrics) | \(\phi_1,\phi_2\) from grouped vertex positions in `inchworm/paper.py` |
| Contact distance (paper) | \(d\) — paper notation (`02_paper_model.md`) |
| Torque stiffness / damping | \(k_\tau\), \(k_d\), `torque_stiffness`, `torque_damping` |

This summary is intended to be a single reference for the mathematics behind the implementation and its correspondence to the paper.
