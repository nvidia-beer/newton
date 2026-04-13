# Method 1: Prescribed Joint Angles Only (paper)

This note summarizes the paper’s **Method 1** (Gamus et al., arXiv:1911.05227): prescribed joint angles, Eq. (8) for \(\dot{d}\), slippage \(\Delta\), slip force \(f_t = \mu f_{n,s}\operatorname{sign}(\dot{d})\), and kinematic displacement \(\pm\Delta d\) in the **quasistatic** paper model.

The **inchworm example** uses **`SolverInflatable`**, **Coulomb ground contact**, and **`gait_traveling_wave.py`** for chamber pressures; see **`07_mathematical_summary.md`** §7–8 and **`06_example_inchworm.md`**.

---

## 1. What the paper means by “Method 1”

In the paper, **Method 1: Prescribed Joint Angles Only** is the case where:

- Joint angles \(\varphi_1(t)\), \(\varphi_2(t)\) are **given as inputs** (prescribed).
- **Paper Eq. (8)** gives \(\dot{d}\) from \(\varphi_i\), \(\dot{\varphi}_i\), \(\theta\), \(\dot{\theta}\).
- **Slippage criterion** \(\Delta = x_c - d/2\): \(\Delta > 0 \Rightarrow\) left foot slips; \(\Delta < 0 \Rightarrow\) right foot slips.
- **Tangential force:** \(f_t = \mu f_{n,s}\,\mathrm{sign}(\dot{d})\).

After each quasistatic solve, the paper advances slipping contact by **\(\Delta d\)** along the crawl axis.

---

## 2. Continuum simulation (Newton inchworm)

- Bending is driven by **chamber pressure** (inflation), not by commanded joint torques.
- **Ground friction** is the soft solver’s **Coulomb** model (`eval_particle_ground_contacts`).

---

## 3. Hysteresis and rectangular \(f_t\) (Fig. 6)

When prescribed angles are smooth, \(\dot{d}\) keeps a definite sign and \(f_t\) alternates between \(\pm\mu f_n\), giving a **rectangular** \(f_t(t)\). With angles inferred from a deformable mesh, \(\dot{d}\) can fluctuate; the paper discusses using a **dead band** on \(|\dot{d}|\) so the slip direction (and thus the sign of \(f_t\)) does not chatter. Details are in the paper and in **§7** of **`07_mathematical_summary.md`**.

---

## 4. Summary

| Paper Method 1 | Newton inchworm |
|----------------|-----------------|
| Prescribed \(\varphi_1(t), \varphi_2(t)\) | Chamber pressures from traveling-wave gait; shape from FEM |
| Paper quasistatic slip + \(\pm\Delta d\) | Implicit dynamics; Coulomb contact (**§8**); pressures from gait |

See **`07_mathematical_summary.md`** for symbols and **`02_paper_model.md`** for full paper summary.
