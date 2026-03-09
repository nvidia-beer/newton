# From Paper to Simulation

The paper’s model (Section 2) is quasistatic and kinematic: at each step it solves equilibrium and applies a kinematic displacement \(\pm\Delta d\). To simulate a deformable body (a soft tetrahedral mesh), the model adds time evolution, elasticity, and a coupling of the paper’s stick–slip rule to the continuum. This section describes those extensions in mathematical form.

## 3.1 Time Stepping and Implicit Integration

Time is discretised with step \(h\). Each timestep the linear system \(A\,\Delta\mathbf{v} = \mathbf{f}\) is solved for \(\Delta\mathbf{v}\), with \(A = M - h\,D - h^2 K\) (mass \(M\), tangent damping \(D\), tangent stiffness \(K\) from linearising the forces). Then \(\mathbf{v}\) and \(\mathbf{q}\) are updated. The only difference between explicit and implicit is the velocity step: explicit uses \(M^{-1}\), implicit uses \(A^{-1}\); implicit is used for stability at practical \(h\). (Solver options for \(A\,\Delta\mathbf{v}=\mathbf{f}\) are in Section 4.)

| Step | Explicit | Implicit |
|------|----------|----------|
| **New velocity** | \(\mathbf{v}^{n+1} = \mathbf{v}^n + h\,M^{-1}\mathbf{F}^n\) | \(\mathbf{v}^{n+1} = \mathbf{v}^n + h\,A^{-1}\mathbf{F}^n\) |
| **New position** | \(\mathbf{q}^{n+1} = \mathbf{q}^n + h\,\mathbf{v}^{n+1}\) | \(\mathbf{q}^{n+1} = \mathbf{q}^n + h\,\mathbf{v}^{n+1}\) |

## 3.2 Deformable Body: FEM and Springs

The body is a **tetrahedral mesh**. Each tetrahedron has a rest shape (matrix \(\mathbf{D}_m\) or its inverse \(\mathbf{D}_m^{-1}\)) and a deformation gradient from current vertex positions:

\[
\mathbf{F} = \mathbf{D}_s\,\mathbf{D}_m^{-1}.
\] A hyperelastic model (e.g. Neo-Hookean) gives stress from \(\mathbf{F}\); forces at vertices come from the divergence of stress. This contributes to the stiffness \(K\) and to the force \(\mathbf{f}\).

**Springs** are added along edges (e.g. for stability or to model fibre reinforcement). A linear spring between vertices \(i,j\) with rest length \(\ell_0\), stiffness \(k_e\), and damping \(k_d\) gives

\[
\mathbf{F}_{\mathrm{spring}} = k_e(\ell - \ell_0)\,\hat{\mathbf{d}} + k_d\,\dot{\ell}\,\hat{\mathbf{d}}, \quad \ell = |\mathbf{x}_j - \mathbf{x}_i|, \quad \hat{\mathbf{d}} = (\mathbf{x}_j - \mathbf{x}_i)/\ell.
\]
(Here \(\hat{\mathbf{d}}\) is the unit direction along the spring, distinct from the paper’s contact distance \(d\).)

These forces and their linearisations are included in the same system matrix \(A\) and RHS \(\mathbf{f}\).

## 3.3 Inflation and Chambers (Actuation)

The paper actuates via torques \(\tau_i = k(\phi_i^{\mathrm{ref}} - \pi)\). In a continuum, "inflation" is mimicked by **scaling the rest configuration**: no explicit pressure force is applied; the rest lengths and rest shapes are changed so that the elastic energy drives the body toward a larger (or differently shaped) configuration.

- **Volume ratio** \(p = V/V_0\): \(p\) is restricted to \([1, p_{\max}]\). Linear scale for 3D: \(s = p^{1/3}\).

- **Springs:** new rest length \(\ell_0^{\mathrm{new}} = \ell_0^{\mathrm{orig}} \cdot s\).

- **Tetrahedra:** rest pose (e.g. \(\mathbf{D}_m^{-1}\)) is scaled by \(1/s\) so the rest shape is larger when \(p > 1\).

**Chambers:** the body can be divided into regions (chambers) with different “pressures” \(p_c\). Each tetrahedron and spring is assigned to a chamber; its rest state is scaled by \(s_c = p_c^{1/3}\) (and optionally by anisotropy factors \(a_x, a_y, a_z\) per axis). So the left and right segments of the inchworm can have different \(p\), producing bending. This replaces the paper’s torque actuation with a continuum equivalent.

**Anisotropy:** scaling can differ along axes (e.g. \(s\,a_z\) along the vertical) so that inflation elongates the body more in one direction; useful for bending or “lift.”

## 3.4 Spine Torque (Bend Stiffness)

The paper’s joints have stiffness \(k\) (torque per radian). In the continuum **torque resistance** is added along selected edges (e.g. the long axis): a spring has a rest direction \(\mathbf{d}_0\); the current direction is \(\mathbf{d}\). The angle \(\alpha\) between them (notation \(\alpha\) avoids collision with the paper’s central link angle \(\theta\)) and the rotation axis are
\[
\alpha = \arccos(\mathbf{d}\cdot\mathbf{d}_0), \qquad \mathbf{a} = \frac{\mathbf{d}_0 \times \mathbf{d}}{|\mathbf{d}_0 \times \mathbf{d}|},
\]
and the torque magnitude (with \(\omega\) the angular rate about \(\mathbf{a}\)) is
\[
\tau = -(k_\tau\,\alpha + k_d\,\omega)\,L.
\]
This torque is converted to forces at the two endpoints. This gives a **spine stiffness** that resists bending and keeps the deformable body behaviour close to the paper’s link model.

## 3.5 Self-Contact

When the body bends, different parts can come into contact. **Self-collision** is added: repulsion between surface vertex–triangle and edge–edge pairs when distance is below a radius \(r\). The force magnitude is a \(C^2\)-style function of distance (e.g. linear in penetration for very close, \(\propto 1/d\) in an intermediate range, zero beyond \(r\)). A force cap avoids huge impulses. This is purely a continuum addition; the paper’s model has no self-contact.

## 3.6 Ground Contact and the Paper’s Stick–Slip

**Normal contact** with the ground is standard: penetration \(c \le 0\) (with particle radius), with normal force and reaction

\[
f_n = k_e c + k_d \min(\dot{c},0), \qquad \mathbf{F}_n = -f_n\mathbf{n}.
\]

**Tangential (stick–slip):** the paper’s rule is used. From the current deformed shape "left" and "right" contact groups and the two "joint" lines are identified; the effective \(\phi_1\), \(\phi_2\) are computed from the positions of these four nodes in the plane. Then \(d\), \(x_c\), \(\Delta\) are computed and which foot slips is decided. On the slipping foot a tangential force of magnitude \(\mu f_n\) is applied in the slip direction (and capped per particle so the leg does not lift). On the sticking foot Coulomb friction is used (no slip). After the implicit step the **kinematic displacement** \(\pm\Delta d\) is applied along the crawl axis, as in the paper, so that the net motion is consistent with "slipping contact moves by \(\Delta d\)."

So: the **dynamics** (FEM, springs, time stepping, inflation, self-contact) are the extension; the **contact law** (normal forces, \(\Delta\), slip force, kinematic \(\Delta d\)) are the paper’s model applied to the continuum.

## 3.7 Coupling Summary

1. **At each timestep:** update inflation (chamber pressures) from the gait; scale rest configurations.

2. **Build** \(A\) and \(\mathbf{f}\) from mass, damping, elasticity (FEM + springs), spine torque, gravity, ground normal force, and paper stick–slip tangential force.

3. **Solve** \(A\,\Delta\mathbf{v} = \mathbf{f}\); then update \(\mathbf{q}\), \(\mathbf{v}\).

4. **Apply** the kinematic displacement \(\pm\Delta d\) along the crawl axis from the paper rule.

5. Apply self-contact corrections.

This gives a single, consistent formulation: the paper’s simple model is embedded in a dynamic, deformable simulation and can be generalised to more links, more chambers, or 3D by extending the same geometry and stick–slip logic. Section 4 details how these ingredients are organised as solver layers.

## 3.8 Paper vs Implementation: Elasticity and Actuation

The paper (arXiv:1911.05227) states: *“In order to account for the elasticity of the continuous structure we introduce equivalent torsion springs at the joints with linear stiffness \(k\), which are at rest when the robot is ‘flat’. To account for the bending actuation of the two beam segments, we apply additional internal input torques \(\tau_i(t)\) at the joints.”* Total internal torque at joint \(i\) (paper Eq. 1):

\[
T_i(t) = \tau_i(t) - k(\phi_i(t) - \pi).
\]

Elastic restoring \(-k(\phi_i - \pi)\) resists bending away from flat (\(\phi_i = \pi\)); actuation \(\tau_i(t)\) drives the joint. With the choice \(\tau_i = k(\phi_i^{\mathrm{ref}} - \pi)\) (paper Eq. 15), the total torque becomes \(T_i = -k(\phi_i - \phi_i^{\mathrm{ref}})\), i.e. a spring that pulls the actual angle toward the reference angle from the gait.

**Implementation.**

- **Elasticity.** The simulation does not use discrete joints. “Joints” are vertex groups used to measure \(\phi_1,\phi_2\) and for the stick–slip state machine. Bending resistance is implemented as **torque on lengthwise springs** (Section 3.4). Each such spring has a rest direction \(\mathbf{d}_0\) (flat). The angle is \(\theta = \arccos(\mathbf{d}\cdot\mathbf{d}_0)\) and the torque magnitude is \(\tau = -(k_\tau\theta + k_d\omega)L\). So elasticity is “resist deviation from flat,” analogous to the paper’s \(-k(\phi_i - \pi)\), but applied per lengthwise spring via \(k_\tau\) (`torque_stiffness`), not as a single \(k\) at a lumped joint.

- **Actuation.** The paper’s \(\tau_i = k(\phi_i^{\mathrm{ref}} - \pi)\) is **not** applied as explicit joint torques. Actuation is **chamber pressure**: left and right chambers get different pressures from the gait; inflation scales rest configurations so the mesh deforms. The resulting \(\phi_1,\phi_2\) are **computed from the deformed mesh** and used in the stick–slip state machine and paper geometry (\(d\), \(\Delta\)). Thus \(\phi_i^{\mathrm{ref}}\) appears in the gait and state machine, but the simulator does not apply a torque \(T_i = -k(\phi_i - \phi_i^{\mathrm{ref}})\); bending is driven by pressure, not by reference-angle torque.

| Aspect | Paper | Implementation |
|--------|--------|----------------|
| **Elasticity** | Torsion springs at joints: \(-k(\phi_i - \pi)\) | Torque on lengthwise springs: \(-(k_\tau\theta + k_d\omega)L\), rest direction \(\mathbf{d}_0\) (flat) |
| **Actuation** | Joint torques \(\tau_i = k(\phi_i^{\mathrm{ref}} - \pi)\) → net \(T_i = -k(\phi_i - \phi_i^{\mathrm{ref}})\) | Chamber pressure (gait) → deformation → \(\phi_1,\phi_2\) from mesh |
| **\(\phi_i^{\mathrm{ref}}\)** | In the dynamics (torque law) | In crawl state machine and paper geometry; not in joint torque law |

So: elasticity is represented in spirit (torsion-like resistance to bending); actuation is a different mechanism (pressure-driven deformation) that produces similar crawling behaviour. A more detailed comparison is in `documentation/paper_vs_implementation_elasticity.md`.
