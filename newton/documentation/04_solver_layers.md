# Solver Layers: Mathematical View

The simulation is built as **four layers** (Section 3 gives the continuum model they extend). Each layer adds a well-defined set of terms to the dynamical system. This section states, in mathematical terms only, what each layer contributes. That is the “solver implementation”: which equations go in at each stage.

**Convention.** The body has particle positions \(\mathbf{q}\) and velocities \(\mathbf{v}\). Each timestep the linear system \(A\,\Delta\mathbf{v} = \mathbf{f}\) is solved for \(\Delta\mathbf{v}\), then \(\mathbf{v}\) and \(\mathbf{q}\) are updated.

- **\(\mathbf{f}\) (right-hand side):** \(h\) times the sum of all forces (elastic, gravity, contact, etc.) evaluated at the current or predicted state. Any layer that adds a force adds that force (times \(h\)) to \(\mathbf{f}\).

- **\(A\) (system matrix):** \(A = M - hD - h^2 K\). \(M\) is mass; \(D\) and \(K\) come from **linearising** those forces with respect to velocity and position. So whenever a layer adds a state-dependent force to \(\mathbf{f}\), the same layer (or the same physics) contributes the corresponding **tangent damping** \(D\) and **tangent stiffness** \(K\) into \(A\).

In short: forces go into \(\mathbf{f}\); their derivatives (w.r.t. state) go into \(A\). That is what “implicit” means here.

**Resolving the system.** The system \(A\,\Delta\mathbf{v} = \mathbf{f}\) is solved iteratively with a Krylov method; \(A\) is not inverted explicitly. Four options are available:

| Solver    | Best for              | Advantages                                      | Notes |
|-----------|------------------------|--------------------------------------------------|-------|
| **CG**    | Symmetric positive definite \(A\) | Low memory, simple; fast when \(A\) is well-conditioned | Fails or is undefined if \(A\) is nonsymmetric or indefinite |
| **BiCGSTAB** | General (nonsymmetric) \(A\) | No symmetry required; often robust; default choice | Convergence can be irregular; may need more iterations |
| **GMRES** | General \(A\)         | Monotonic residual decrease; no breakdown       | Memory grows with iterations (or use restarts, which can stagnate) |
| **CR**    | Symmetric \(A\) (possibly indefinite) | Minimises residual norm; can handle some indefinite systems | Like CG but for residual; still requires symmetry |

For the soft-body system, \(A = M - hD - h^2 K\) is usually **symmetric** (mass, damping, and elasticity stiffness are symmetric). Then **CG** or **CR** are appropriate and typically efficient. With contact, friction, or extra terms from upper layers, \(A\) can become **nonsymmetric** or less well-conditioned; **BiCGSTAB** (the default) or **GMRES** then remain valid and are preferred. A diagonal (or block-diagonal) preconditioner is used to improve conditioning and reduce iteration count. The maximum number of iterations is capped (e.g. 50); large or stiff meshes may require a higher limit for convergence.

---

## Layer 1: Soft (base dynamics)

**Role:** Time evolution of a soft body in contact with the ground. (Physics in Section 3.1–3.2.)

**Adds to the system:** \(M\), \(D\), \(K\) (mass, damping from springs/material, stiffness from FEM and springs); \(\mathbf{f}\) from elasticity, springs \(k_e(\ell-\ell_0)\hat{\mathbf{d}} + k_d\dot{\ell}\hat{\mathbf{d}}\), gravity, and ground contact (normal + Coulomb). Update \(\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta\mathbf{v}\), \(\mathbf{q}^{n+1} = \mathbf{q}^n + h\,\mathbf{v}^{n+1}\). This layer does not use the paper’s stick–slip rule.

---

## Layer 2: Deformable (self-contact)

**Role:** Prevent different parts of the body from passing through each other when it bends. (Physics in Section 3.5.)

**Adds to the system:** Repulsion forces (vertex–triangle, edge–edge) to \(\mathbf{f}\); optionally diagonal stiffness \(h^2 k_{\mathrm{eff}}\mathbf{I}\) in \(A\) for vertices in contact.

---

## Layer 3: Inflatable (actuation and spine stiffness)

**Role:** Drive shape change by “inflation” (rest-configuration scaling) and resist bending with a spine torque. (Physics in Section 3.3–3.4.)

**Adds to the system:** Rest scaling \(s = p^{1/3}\) (and per-chamber \(s_c\)) and anisotropy change the effective rest state of Layer 1; spine torque \(\tau = -(k_\tau\alpha + k_d\omega)L\) (angle \(\alpha = \arccos(\mathbf{d}\cdot\mathbf{d}_0)\)) adds terms to \(\mathbf{f}\) and, when linearised, to \(K\) and \(D\).

---

## Layer 4: Crawlable (paper stick–slip and kinematic update)

**Role:** Replace Layer 1 ground friction with the paper’s stick–slip rule and apply kinematic displacement \(\pm\Delta d\) after each step. (Geometry and rule in Section 2; coupling in Section 3.6.)

**Changes to the system:** From mesh, effective \(\phi_1,\phi_2\) and then \(d\), \(x_c\), \(\Delta\) are computed; slip leg is chosen by sign of \(\Delta\). Tangential force: \(\mu f_n\) in slip direction on slipping group (capped per particle), Coulomb on sticking group. After the step, all particles are displaced by \(\pm\Delta d\) along the crawl axis. Normal forces unchanged; mass, elasticity, inflation, self-contact unchanged.

---

## Summary Table

What each layer adds to the system matrix \(A\) (mass, damping, stiffness) and the force vector \(\mathbf{f}\) (RHS of \(A\,\Delta\mathbf{v}=\mathbf{f}\)). In implicit integration, forces go into \(\mathbf{f}\) and their linearisations into \(A\).

| Layer   | Adds to \(A\) (matrix) | Adds to \(\mathbf{f}\) (forces) |
|--------|------------------------|----------------------------------|
| **Soft**       | \(M\), \(D\), \(K\) from mass, damping, FEM, springs | Elasticity, springs, gravity, ground contact (normal + Coulomb) |
| **Deformable** | Optional diagonal stiffness \(h^2 k_{\mathrm{eff}}\mathbf{I}\) for vertices in contact | Repulsion forces (vertex–triangle, edge–edge) |
| **Inflatable** | Spine torque linearisation in \(K\), \(D\); rest scaling changes existing FEM/spring \(K\) | Spine torque force; rest scaling changes existing forces |
| **Crawlable**   | — | Ground tangential force from paper rule (slip/stick by \(\Delta\)); post-step kinematic \(\pm\Delta d\) |

The inchworm example uses all four layers. To extend the model (e.g. more chambers, more contact groups, 3D), the same mathematical structure is kept; additional geometry and indexing (e.g. more contact groups, chamber assignments) are added, and the solver layers stay the same.
