# Method 1: Prescribed Joint Angles Only

This document explains the paper’s **Method 1** (Gamus et al., arXiv:1911.05227) and how it is implemented in the Newton inchworm crawling simulation when angles are **not** prescribed but **inferred** from the deformable mesh.

---

## 1. What the paper means by “Method 1”

In the paper, **Method 1: Prescribed Joint Angles Only** is the case where:

- Joint angles \(\varphi_1(t)\), \(\varphi_2(t)\) are **given as inputs** (prescribed).
- You do **not** solve torque balance for the angles.
- You only need:
  - **Paper Eq. (8)** — time derivative of contact distance:
    \[
    \dot{d} = l\,\frac{\sin\varphi_2\,\dot{\varphi}_2 - \sin\varphi_1\,\dot{\varphi}_1}{\sin\theta}
    \]
    or the equivalent form in terms of \(\theta\), \(\dot{\theta}\) (see `paper_model.py`).
  - **Slippage criterion** \(\Delta = x_c - d/2\): \(\Delta > 0 \Rightarrow\) left foot slips; \(\Delta < 0 \Rightarrow\) right foot slips.
  - **Tangential force:** \(f_t = \mu f_{n,s}\,\mathrm{sign}(\dot{d})\), where \(f_{n,s}\) is the normal force at the slipping foot.

So in Method 1 the angles are **inputs**; the state machine uses \(\Delta\) to decide which foot slips and \(\dot{d}\) to set the sign of \(f_t\). The result is a **rectangular** \(f_t(t)\): flat at \(+\mu f_n\) for part of the cycle and at \(-\mu f_n\) for the rest, with sharp transitions (paper Fig. 6).

---

## 2. Why the simulation has no “prescribed” angles

In the simulation:

- Bending is driven by **chamber pressure** (inflation), not by commanded joint angles.
- The shape is determined by the **deformable mesh** (FEM, contacts, etc.).
- The simulation does **not** have \(\varphi_1(t)\), \(\varphi_2(t)\) as explicit inputs.

The implementation uses **the same Method 1 equations**, but with angles **inferred from the current mesh** each step instead of prescribed.

---

## 3. How Method 1 is implemented (angles inferred from mesh)

Implementation is in `paper_model.py` and `solver_crawlable.py`.

### 3.1 Infer angles each step

- From current particle positions the **mean positions** of the four groups are taken: left contact, right contact, left joint, right joint (set by `set_crawl_contact_groups`).
- \((\phi_1, \phi_2)\) is computed in the 2D plane (crawl_axis × vertical_axis) via **`joint_angles_from_positions`**.
- The paper’s “prescribed” \(\varphi_i(t)\) are thus replaced by **angles read from the current state** of the soft body.

### 3.2 Same Method 1 geometry and forces

- With \((\phi_1, \phi_2)\) the code computes \(\theta\), \(d\), \(x_c\), and \(\Delta = x_c - d/2\) (paper geometry and slippage criterion).
- **Which foot slips:** from sign of \(\Delta\) (e.g. `crawl_state_step_simple` returns `SLIP_STICK` or `STICK_SLIP`).
- **Slip force magnitude:** \(\mu f_{n,s}\) at the slipping foot.
- **Slip force direction:** from the sign of \(\dot{d}\).

### 3.3 Time derivative of contact distance \(\dot{d}\) (paper Eq. (8))

- The paper uses the analytic \(\dot{d}\) in terms of \(\varphi_i\), \(\dot{\varphi}_i\), \(\theta\), \(\dot{\theta}\).
- Prescribed \(\dot{\varphi}_i\) are not available; the implementation approximates:
  - \(\dot{\varphi}_i \approx (\phi_i - \phi_{i,\mathrm{prev}})/\Delta t\) from the **previous** step.
  - When \(\phi_{1,\mathrm{prev}}\) and \(\phi_{2,\mathrm{prev}}\) exist, \(\dot{\theta}\) is computed from \(\theta(\phi_1,\phi_2)\) and the full Eq. (8) is used in **`compute_d_dot_from_angular_velocities`**.
  - Otherwise the code falls back to \(\dot{d} \approx (d - d_{\mathrm{prev}})/\Delta t\).

The result is “Method 1 with angles from the mesh and \(\dot{d}\) from Eq. (8) when possible.”

### 3.4 Kinematic displacement

- After the step the same rule as the paper is applied: **body displacement** by \(\pm\Delta d\) along the crawl axis according to which foot slipped (**`apply_crawl_kinematic_displacement`**).

---

## 4. Hysteresis band for rectangular \(f_t\) (Fig. 6)

In the paper, with smooth prescribed angles, \(\dot{d}\) has a clear sign for long intervals, so \(f_t = \pm\mu f_n\) is **flat** and the plot of \(f_t\) vs \(t/T\) looks like a **rectangle** (flat at \(+1\), then \(-1\), then \(+1\) after normalization).

In the simulation, \(\phi_1\), \(\phi_2\) come from the deformable mesh and can **jitter**; \(\dot{d}\) can cross zero every few steps. If \(f_t = \mu f_n\,\mathrm{sign}(\dot{d})\) were used with no dead band, the **sign of \(\dot{d}\)** would flip often and the plot would show many **spikes** instead of a clean rect.

To recover a **rect-like** \(f_t(t)\), a **hysteresis band** is used in **`crawl_state_step_simple`** (in `paper_model.py`):

- **Parameter:** When the gait **period** \(T\) is provided (from the solver’s \(\omega\): \(T = 2\pi/\omega\)), \(d\_{\mathrm{dot\_eps}} = \alpha \cdot l \cdot (2\pi/T)\) with \(\alpha = 0.1\), so the dead band scales with cycle time (characteristic rate \(l/T\)). Otherwise a fixed fallback (e.g. \(10^{-2}\)) is used.
- **If \(|\dot{d}| < d\_{\mathrm{dot\_eps}}\):** keep the **previous** slip direction (and thus the same \(f_t = \pm\mu f_n\)). Do **not** flip sign and do **not** set \(f_t = 0\).
- **If \(\dot{d} \ge d\_{\mathrm{dot\_eps}}\):** set slip direction to \(+1\) (\(f_t = +\mu f_n\)).
- **If \(\dot{d} \le -d\_{\mathrm{dot\_eps}}\):** set slip direction to \(-1\) (\(f_t = -\mu f_n\)).

The sign of \(f_t\) is **changed** only when \(\dot{d}\) clearly crosses the band. That keeps \(f_t\) flat over most of the cycle and yields a rectangular shape in Fig. 6, consistent with the paper’s Method 1 behaviour.

---

## 5. Summary

| Paper Method 1 | This implementation |
|----------------|--------------------|
| Prescribed \(\varphi_1(t), \varphi_2(t)\) | Angles **inferred** from mesh each step via `joint_angles_from_positions` |
| \(\dot{d}\) from Eq. (8) with prescribed \(\dot{\varphi}_i\) | \(\dot{d}\) from Eq. (8) using \(\dot{\varphi}_i \approx (\phi_i - \phi_{i,\mathrm{prev}})/\Delta t\), or \((d - d_{\mathrm{prev}})/\Delta t\) fallback |
| \(f_t = \mu f_{n,s}\,\mathrm{sign}(\dot{d})\) | Same, with **hysteresis**: keep previous sign when \(|\dot{d}| < d\_{\mathrm{dot\_eps}}\) so \(f_t(t)\) is rect-shaped |
| Kinematic displacement \(\pm\Delta d\) | Same, via `apply_crawl_kinematic_displacement` |

See **05_solver_crawlable.md** for solver setup and **07_mathematical_summary.md** for equation references.
