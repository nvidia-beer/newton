# The Paper’s Model

This section summarizes the model of Gamus et al. (“Understanding Legged Crawling for Soft Robots,” arXiv:1911.05227): a planar, quasistatic, three-link system with two point contacts and a hybrid stick–slip rule. Notation and equations follow the paper; equation numbers (e.g. Eq. (3), (4), (8), (9)) refer to that reference.

## 2.1 Geometry and Notation

The robot is a **three-link chain** in the plane (e.g. the vertical plane containing the crawl direction and gravity). The two end links are the “legs”; the middle link is the “body.” There are **two point contacts** (feet) and **two joint angles** \(\phi_1\), \(\phi_2\) (interior angles at the two joints; \(\pi\) when flat).

![Three-link robot model. Geometry: link lengths \(l\), \(\beta l\); joint angles \(\phi_1\), \(\phi_2\); central link angle \(\theta\); contact distance \(d\); horizontal positions \(x_1\), \(x_2\) of the feet (\(x_2 = x_1 + d\)); CoM horizontal offset \(x_c\). Actuation: torques \(\tau_1\), \(\tau_2\); joint stiffness \(k\). Masses: \(m\) (outer links), \(\beta m\) (centre).](figures/paper_3.png)

**Parameters (paper Table I), see figure above:**

| Symbol | Meaning |
|--------|---------|
| \(L\) | Total beam length (m) |
| \(\beta\) | Dimensionless shape parameter (paper uses \(\beta = 2\)) |
| \(M\) | Total mass (kg) |
| \(g\) | Gravity magnitude |
| \(\mu\) | Coefficient of friction at contact |
| \(k\) | Joint stiffness (paper); simulation uses spine torque stiffness instead |

**Derived:**

- **Link length**
  \[
  l = \frac{L}{2 + \beta}.
  \]

- **Kinematic ground-contact constraint** (both feet on the ground):
  \[
  \sin(\phi_1 - \theta) - \sin(\phi_2 + \theta) + \beta\sin\theta = 0.
  \]
  This enforces that both contacts lie on the ground plane; the central link angle \(\theta\) follows from it.

- **Central link orientation** \(\theta\) (angle of the middle link w.r.t. horizontal), equivalent to the constraint above:
  \[
  \tan\theta = \frac{\sin\phi_1 - \sin\phi_2}{\cos\phi_1 + \cos\phi_2 - \beta}.
  \]

- **Horizontal distance between the two contact points** (contact distance):
  \[
  d = l\,\bigl(\beta\cos\theta - \cos(\phi_1 - \theta) - \cos(\phi_2 + \theta)\bigr).
  \]

- **Horizontal distance of the centre of mass from the left contact** \(x_c\):
  \[
  x_c = \frac{l}{2(2+\beta)}\Bigl((2+\beta)\beta\cos\theta - (3+2\beta)\cos(\phi_1-\theta) - \cos(\phi_2+\theta)\Bigr).
  \]

These four quantities (\(l\), \(\theta\), \(d\), \(x_c\)) are purely geometric given \(\phi_1\), \(\phi_2\), and \(\beta\).

## 2.2 Equilibrium: Normal Forces at the Feet

Under quasistatic equilibrium, the normal forces at the two contacts (left 1, right 2) are determined by moment balance. For \(d > 0\):

\[
f_{n1} = \left(1 - \frac{x_c}{d}\right) M g, \qquad
f_{n2} = \frac{x_c}{d}\,M g.
\]

So the left foot carries more load when \(x_c < d/2\), the right when \(x_c > d/2\).

From **horizontal equilibrium**, the tangential contact forces satisfy
\[
f_{t,1} = -f_{t,2} \equiv f_t(t).
\]
So there is a single unknown \(f_t\); the system is statically indeterminate until the stick–slip state fixes which foot slips.

## 2.3 Slippage Criterion

The paper’s **slippage criterion** is the sign of

\[
\Delta = x_c - \frac{d}{2}.
\]

- \(\Delta > 0\) ⇒ the left foot slips (right sticks).

- \(\Delta < 0\) ⇒ the right foot slips (left sticks).

An equivalent form used in the paper is
\[
\Delta \propto \frac{1+\beta}{2(2+\beta)}\,l\,\bigl(\cos(\phi_2 + \theta) - \cos(\phi_1 - \theta)\bigr).
\]

So the **difference in link orientation** (through the cosines) decides which foot slips.

## 2.4 Slip Force and Coulomb Limit

When a foot is slipping, the tangential force at that foot is at the Coulomb limit. The **signed** force (paper) is
\[
f_t = \mu\, f_{n,s}\, \operatorname{sign}(\dot{d}),
\]
where \(s = 1\) when \(\Delta > 0\) (left foot slips) and \(s = 2\) when \(\Delta < 0\) (right foot slips). The \(\operatorname{sign}(\dot{d})\) term sets the **direction** of the friction force (opposing the slip velocity). So \(|f_t| = \mu f_n\) on the slipping foot, with direction given by the slip direction.

The **time derivative of the contact distance** (paper Eq. (8)) is
\[
\dot{d} = l\,\bigl[\sin(\phi_2+\theta)(\dot{\phi}_2+\dot{\theta}) + \sin(\phi_1-\theta)(\dot{\phi}_1-\dot{\theta}) - \beta\sin\theta\,\dot{\theta}\bigr].
\]
In discrete time, \(\dot{d}\) is approximated by \((d_{\mathrm{new}} - d_{\mathrm{prev}})/\Delta t\). The sticking foot has tangential force below the limit (no slip). This is the **hybrid** rule: one foot sticks, one slips.

## 2.5 Kinematic Update (How the Body Moves)

The paper does not integrate slip forces over time. Instead, it **updates contact positions** after each step:

- **Sticking contact** — position unchanged.

- **Slipping contact** — its position changes by \(\Delta d\) in the slip direction.

So the **body** (the whole three-link system) is displaced by \(\pm\Delta d\) along the crawl axis: if the left foot slipped, the body moves so that the left contact has effectively moved by \(\Delta d\); equivalently, the body is displaced by \(-\Delta d\) in the crawl direction (and similarly for the right). Net displacement per cycle is what produces crawling.

Formally: let \(d_{\mathrm{new}}\) be the contact distance after the step (from the new \(\phi_1,\phi_2\)). Then

\[
\Delta d = d_{\mathrm{new}} - d_{\mathrm{prev}}.
\]

The body is displaced by \(\pm\Delta d\) along the horizontal crawl axis according to which foot slipped.

## 2.6 Gait: Reference Angles and Actuation

The paper uses a **harmonic gait** for the reference joint angles. The general form is \(\phi_i^{\mathrm{ref}} = \gamma_i + A_i\sin(\omega t \pm \psi/2)\) per joint; for ideal (symmetric) gaits the paper has \(\gamma_1 = \gamma_2\) and \(A_1 = A_2\), giving the symmetric form used here:

\[
\phi_1^{\mathrm{ref}}(t) = \gamma + A\,\sin(\omega t + \psi/2), \qquad
\phi_2^{\mathrm{ref}}(t) = \gamma + A\,\sin(\omega t - \psi/2).
\]

Typical choice: \(\gamma = \pi/2\), phase \(\psi \approx \pi/2\) for robustness.

In Newton, bending is achieved by **chamber pressures** (inflation), not by the paper’s joint torques. Ground contact uses **Coulomb friction** (`SolverSoft` / `SolverInflatable`; see `07_mathematical_summary.md` §8). Angles \(\phi_1,\phi_2\) for **metrics** come from grouped vertex positions (`inchworm/paper.py`).

## 2.7 Simulation Solutions

Simulation solutions for the three-link robot’s configuration compare prescribed joint angles (solid curves) with prescribed torques under realistic and low stiffness. The joint angles \(\phi_1\), \(\phi_2\) over one normalized cycle are shown below (blue and purple respectively).

![Joint angles φ₁ (blue) and φ₂ (purple) over normalized time t/T.](figures/fig4_simulation_a.png)

The positions of the left and right contacts \(x_1\), \(x_2\) over the cycle, together with snapshots of the robot at selected times, are shown next (blue: left contact, purple: right contact; gray: robot outline).

![Position of left x₁ (blue) and right x₂ (purple) contacts and snapshots of the robot (gray).](figures/fig4_simulation_b.png)

## 2.8 Summary for Extension

For a more complicated model (more links, more contacts, 3D), the same ideas apply:

| Step | Description |
|------|-------------|
| **1. Geometry** | Define link lengths and joint angles; compute contact distance(s) and CoM offset(s). |
| **2. Equilibrium** | Normal forces at each contact from balance. |
| **3. Slippage criterion** | A scalar (like \(\Delta\)) that decides which contact(s) slip. |
| **4. Slip law** | \(|f_t| = \mu f_n\) on slipping contacts; stick elsewhere. |
| **5. Kinematic update** | Update contact positions by \(\Delta d\) (or analogues) for slipping contacts; body displacement follows. |

Section 3 describes how this model is embedded in a dynamic, deformable simulation.