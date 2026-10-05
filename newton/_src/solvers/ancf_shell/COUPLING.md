# ANCF tire ↔ rigid vehicle interface coupling

How the ANCF shell tires and the MuJoCo rigid vehicle are advanced together,
which interface iterations exist in `newton/_src/solvers/ancf_shell/coupling.py`,
`schur.py` and `coupled_newton.py`, the mathematics of each, and why the
default is the coupled shell/interface Newton method.

## Methods at a glance

| Method | Where | Status | Original papers |
|---|---|---|---|
| Partitioned Gauss–Seidel (§2) | `coupling.py`, `acceleration=False` | uncaptured fallback only | [Felippa, Park & Farhat 2001](https://doi.org/10.1016/S0045-7825(00)00391-1); instability: [Causin, Gerbeau & Nobile 2005](https://doi.org/10.1016/j.cma.2004.12.005), [Förster, Wall & Ramm 2007](https://doi.org/10.1016/j.cma.2006.09.002) |
| Aitken dynamic relaxation (§3) | `_aitken_coefficient`, `_relax_coordinates` | uncaptured fallback for every method | [Irons & Tuck 1969](https://doi.org/10.1002/nme.1620010302); [Küttler & Wall 2008](https://doi.org/10.1007/s00466-008-0255-5); [preCICE](https://precice.org/configuration-acceleration.html) |
| Adaptive Aitken (§4) | `_adaptive_interface_step`, `coupling-method=adaptive` | kept; `auto` fallback (odd substeps, > 64 dofs, Superjeep) | same as §3 plus this implementation's device-side stopping test and predictor |
| Interface quasi-Newton, finite-difference seeded (§5) | — | **removed** | [Broyden 1965](https://doi.org/10.1090/S0025-5718-1965-0198670-6); IQN-ILS: [Degroote, Bathe & Vierendeels 2009](https://doi.org/10.1016/j.compstruc.2008.11.013) |
| Schur-condensed interface tangent (§6) | `schur.py` (`RigidShellSchurResponse.build_tangent`, `ShellSchurResponse.solve_displacement`) + `kernels_coupled.make_inverse` | kept as the tangent used by coupled Newton; no longer standalone | [Guyan 1965](https://doi.org/10.2514/3.2874); [Fernández & Moubachir 2005](https://doi.org/10.1016/j.compstruc.2004.04.021); [Michler, van Brummelen & de Borst 2005](https://doi.org/10.1002/fld.850) |
| Coupled shell/interface Newton (§7) | `coupled_newton.py`, `coupling-method=coupled-newton` | **default** via `auto` | Anderson mixing: [Anderson 1965](https://doi.org/10.1145/321296.321305), [Walker & Ni 2011](https://doi.org/10.1137/10078356X); implicit differentiation: [Blondel et al. 2022](https://arxiv.org/abs/2105.15183) |
| Shell model and integrator (all) | `solver_ancf_shell.py` | — | ANCF: [Shabana 1998](https://doi.org/10.1023/A:1008072517368), ANCF3423: Recuero & Negrut 2016 (TR-2016-11); HHT-α: [Hilber, Hughes & Taylor 1977](https://doi.org/10.1002/eqe.4290050306) |

## 1. The partitioned problem

Each substep of length *h* advances two solvers that know nothing about each
other's internals:

- **Rigid side** (`SolverMuJoCo`). Given the tire wrench **w**ᵢ ∈ ℝ⁶ on each
  spindle body *i*, integrate the articulated vehicle and return the
  generalised velocity **v** ∈ ℝⁿᵛ (the chassis free joint, suspension and wheel
  dofs; *n*ᵥ ≤ 64 for the vehicles shipped here).
- **Shell side** (`SolverANCFShell`, ANCF3423 elements, HHT-α integration).
  Given the spindle motion, prescribe the bead ring (Dirichlet nodes) at the
  end of the substep (ANCF3423 after [Shabana 1998](https://doi.org/10.1023/A:1008072517368) and
  [Recuero & Negrut 2016](#references); HHT-α after [Hilber, Hughes & Taylor 1977](https://doi.org/10.1002/eqe.4290050306)), solve the tire with Newton + PCG, and return the reaction
  wrench **w**ᵢ(**v**) from the bead forces and the tire's inertia.

Write the rigid update driven by the shell wrench as the **response map**

$$
\mathbf{R}(\mathbf{v}) = \text{rigid velocity at } t_{n+1}
\text{ after one rigid step under } \mathbf{w}(\mathbf{v}),
$$

where the bead prescription uses **v** as the end-of-step spindle velocity.
The two solvers agree when the velocity used to drive the beads equals the
velocity the rigid body actually attains:

$$
\mathbf{r}(\mathbf{v}) \equiv \mathbf{R}(\mathbf{v}) - \mathbf{v} = \mathbf{0}.
\tag{1}
$$

Every method below is a different root finder for (1). They are *nested*:
each adds information about the interface Jacobian ∂**r**/∂**v**.

Two mechanical facts shape all of them:

- **Added mass.** The tire is an elastic body of comparable mass to the wheel
  and much stiffer than the suspension. ∂**R**/∂**v** therefore has eigenvalues
  of order *m*ₜᵢᵣₑ/*m*ᵥᵥₕₑₑₗ, and the plain staggered iteration diverges or needs
  many passes — the classic added-mass instability of partitioned schemes
  ([Causin, Gerbeau & Nobile 2005](https://doi.org/10.1016/j.cma.2004.12.005); [Förster, Wall & Ramm 2007](https://doi.org/10.1016/j.cma.2006.09.002)).
- **Cost.** One evaluation of **R** is a complete implicit tire solve for all
  four tires. With four tires, 6–10 substeps per frame and *k* interface
  trials, the tire solve is *k*·substeps·4 times the whole frame budget.

All trials inside a substep restart from the same *t*ₙ state (packed ANCF
state and rigid snapshot, see `InterfaceCouplerGS._presave`), so a trial never
contaminates the accepted trajectory. The last rigid response is always
accepted unscaled: convergence changes the amount of work, never the physics.

## 2. Partitioned Gauss–Seidel (base iteration)

$$
\mathbf{v}_{k+1} = \mathbf{R}(\mathbf{v}_k), \qquad k = 0,\dots,n-1 .
$$

A block Gauss–Seidel (Dirichlet–Neumann) fixed-point iteration
([Felippa, Park & Farhat 2001](https://doi.org/10.1016/S0045-7825(00)00391-1)). It converges only if the spectral radius ρ(∂**R**/∂**v**) < 1,
which the added-mass argument above violates for a loaded tire. Kept only as
the uncaptured fallback path (`acceleration=False`); not selectable from the
examples.

## 3. Aitken dynamic relaxation

With **r**ₖ = **R**(**v**ₖ) − **v**ₖ,

$$
\omega_k = -\,\omega_{k-1}\,
\frac{\mathbf{r}_{k-1}\cdot(\mathbf{r}_k-\mathbf{r}_{k-1})}
     {\lVert \mathbf{r}_k-\mathbf{r}_{k-1}\rVert^2},
\qquad
\mathbf{v}_{k+1} = \mathbf{v}_k + \omega_k\,\mathbf{r}_k .
\tag{2}
$$

This is the vector form of Aitken's Δ² acceleration ([Irons & Tuck 1969](https://doi.org/10.1002/nme.1620010302))
as used for FSI by [Küttler & Wall 2008](https://doi.org/10.1007/s00466-008-0255-5) and [preCICE](https://precice.org/configuration-acceleration.html). ω is the secant estimate of
the scalar contraction factor; the first trial of each substep uses
`initial_relaxation` (0.1). Kernel: `_aitken_coefficient` / `_relax_coordinates`
(`_aitken_weight` holds the ω update in float64).

Properties: cannot diverge for ω ∈ (0, 1]; one scalar for *n*ᵥ coordinates;
converges linearly. Used with a fixed count of `gs_iters` evaluations it costs
six tire solves per substep regardless of what the step is doing. It is the
uncaptured fallback for every method below.

## 4. Adaptive Aitken (`coupling-method=adaptive`)

The same update (2) plus device-side control flow, so that it runs inside a
single CUDA graph (`wp.capture_if` / `wp.capture_while`):

- **Per-coordinate stopping test** (kernel `_adaptive_interface_step`,
  function `_scaled_error`):

  $$
  \max_j \frac{|r_{k,j}|}{\text{atol} + \text{rtol}\,\max(|v_{k,j}|,|R_j(\mathbf{v}_k)|)} \le 1
  $$

  with atol = 10⁻³ m/s (rad/s) and rtol = 10⁻³, after at least three trials.
  The accepted response is always the raw rigid state of the last trial.
- **Predictor.** After a converged substep, the next substep starts from
  **v**ₙ + Δ**v**ₙ₋₁ (`_predict_interface_velocity`), so steady driving needs
  ~3 trials and the full budget is spent only on transients.

This is the method `auto` selects for odd substep counts, more than 64
velocity coordinates, and for rigid models coupled Newton rejects (today: the
Superjeep, whose double-wishbone suspension exceeds 32 rigid dofs). On the
Superjeep at 10 substeps it reaches a tighter interface residual than the
quasi-Newton path it replaced (0.003 vs 0.006) with the same trajectory, at
~1.5× the tire evaluations of coupled Newton.

## 5. Interface quasi-Newton (removed)

The scalar ω of (2) was replaced by an approximate inverse Jacobian
**H** ≈ (∂**r**/∂**v**)⁻¹ ∈ ℝⁿᵛˣⁿᵛ:

$$
\mathbf{v}_{k+1} = \mathbf{v}_k - \mathbf{H}\,\mathbf{r}_k ,
$$

seeded by finite-difference probing (one full tire solve per velocity
coordinate, perturbation 0.05 m/s) and updated within the substep by the
"good" Broyden rank-one secant formula ([Broyden 1965](https://doi.org/10.1090/S0025-5718-1965-0198670-6))

$$
\mathbf{H} \leftarrow \mathbf{H} +
\frac{(\mathbf{s}-\mathbf{H}\mathbf{y})\,\mathbf{s}^{\mathsf T}\mathbf{H}}
     {\mathbf{s}^{\mathsf T}\mathbf{H}\mathbf{y}},
\qquad \mathbf{s}=\Delta\mathbf{v},\ \mathbf{y}=\Delta\mathbf{r},
$$

which is the interface quasi-Newton family (IQN-ILS) of
[Degroote, Bathe & Vierendeels 2009](https://doi.org/10.1016/j.compstruc.2008.11.013). It was removed because the seeding costs *n*ᵥ tire solves
per refresh, the Superjeep's condensed tangent (section 6) kept rejecting its
estimate and falling back to this probing (2 500 probe solves in a 25 s
drive), and it caused a CUDA illegal-memory-access failure at 6 substeps. No
shipped configuration selected it as a primary method. Adaptive Aitken
(section 4) is more robust and only ~1.5× more expensive on the one asset
that used it.

## 6. Condensed Schur interface tangent (`schur.py`, used by coupled Newton)

Instead of probing, obtain the interface Jacobian analytically from the shell's
own tangent. Partition the HHT-linearised shell system into free dofs *f* and
bead dofs *b*:

$$
\begin{bmatrix}
\mathbf{K}_{ff} & \mathbf{K}_{fb}\\
\mathbf{K}_{bf} & \mathbf{K}_{bb}
\end{bmatrix}
\begin{bmatrix}\delta\mathbf{x}_f\\ \delta\mathbf{x}_b\end{bmatrix}
=
\begin{bmatrix}\mathbf{0}\\ \delta\mathbf{f}_b\end{bmatrix},
\qquad
\mathbf{K}_{\text{eff}} = \frac{\mathbf{M}}{\beta h^2} + (1+\alpha)\,\mathbf{K}_t .
$$

Eliminating the interior (static condensation, [Guyan 1965](https://doi.org/10.2514/3.2874)) gives the
bead-ring impedance

$$
\mathbf{Z}_b = \mathbf{K}_{bb} - \mathbf{K}_{bf}\,\mathbf{K}_{ff}^{-1}\,\mathbf{K}_{fb} ,
$$

the Schur complement of **K**_ff. Only its action on the six rigid motions of
each spindle is needed, so `RigidShellSchurResponse.build_tangent` assembles the
tangent and `ShellSchurResponse.solve_displacement` solves
**K**_ff **δx**_f = −**K**_fb **δx**_b for six unit spindle motions per tire
with the shell's SPD PCG (`nrhs = 6·tires`), and projects the bead reaction onto
the spindle to obtain the 6×6 tire impedance **Z**ᵢ. The rigid side contributes
its joint-space mass matrix **M** (+ armature) through the articulation
Jacobian **J**, so the Newton step on (1) is

$$
\mathbf{H} = \mathbf{S}^{-1}(\mathbf{M}+\mathbf{M}_{\text{arm}}),
\qquad
\mathbf{S} = \mathbf{M} + \mathbf{M}_{\text{arm}} + \sum_i \mathbf{J}_i^{\mathsf T}\mathbf{Z}_i\mathbf{J}_i
$$

(`kernels_coupled.make_inverse`, tiled Gauss–Jordan in float64, then
`response_inverse` for the product with the rigid mass). This is the exact-Jacobian
interface Newton method of [Fernández & Moubachir 2005](https://doi.org/10.1016/j.compstruc.2004.04.021) and
[Michler, van Brummelen & de Borst 2005](https://doi.org/10.1002/fld.850), specialised to a shell whose interior solve is
already available. The tangent is refreshed at most every 50 substeps
(`interface_linearization_count`); it changes only the iteration matrix, never
the mass or the transferred wrench. Requires one articulation, rigid terrain
and an even substep count. It no longer runs on its own; coupled Newton
consumes it.

## 7. Coupled shell/interface Newton (`coupling-method=coupled-newton`, default via `auto`)

Sections 2–6 pay one *complete* tire solve per interface trial and throw the
shell iterate away between trials. Coupled Newton keeps it. Each interface
correction solves the shell interior once with seven right-hand sides per
tire — the six spindle motion columns of section 6 plus the current free-shell
residual (`force_rhs`) — and advances **both** unknowns from that one linear
solve:

$$
\begin{aligned}
\delta\mathbf{x}_f &= -\mathbf{K}_{ff}^{-1}\bigl(\mathbf{R}_f + \mathbf{K}_{fb}\,\delta\mathbf{x}_b(\delta\mathbf{v})\bigr),\\
\delta\mathbf{v} &= -\mathbf{H}\,\mathbf{r}(\mathbf{v}) .
\end{aligned}
$$

The shell and interface increments are combined by a mass-weighted Anderson
mixing step ([Anderson 1965](https://doi.org/10.1145/321296.321305); [Walker & Ni 2011](https://doi.org/10.1137/10078356X);
kernels `aa_products`, `aa_update`), and the convergence test of section 4 is applied to the
interface residual (`check`). The normal `gs-iters` budget gains three reserve
evaluations (`maximum = 2·n − 1`) that are used only while the residual stays
above 0.025. Small tires whose symmetric Gauss–Seidel sweep fits one CUDA block
use at most four PCG iterations per correction; larger tires keep `pcg-iters`.
Everything else is unchanged: the same HHT integration, contact law, cavity
pressure and the full reaction torque.

**Why this is the default.** The per-substep cost approaches a single implicit
tire solve instead of *k* of them, while the condensed tangent gives Newton-type
convergence of the interface (2–3 corrections on straight driving). Measured
on the Warthog and Sherp profiles it reduced tire work by roughly the factor of
the interface trial count at the same 6 / 2 / 10 solver budget and the same
acceptance residuals; on the Superjeep, which it does not support, `auto`
falls back to adaptive Aitken. It is also the only method that provides a
usable interface Jacobian: a future adjoint through the vehicle dynamics would
differentiate the converged fixed point implicitly,
d**v**/dθ = −(∂**r**/∂**v**)⁻¹ ∂**r**/∂θ ([Blondel et al. 2022](https://arxiv.org/abs/2105.15183)), reusing **H**
rather than back-propagating through an iteration.

### Preconditions and fallbacks

| Condition | Result |
|---|---|
| `gs-iters ≥ 3`, even substeps, `n_v ≤ 64`, one articulation, rigid terrain, finite SPD rigid mass, `joint_dof_count ≤ 32` | coupled Newton |
| any of the above fails | adaptive Aitken (section 4) |
| graph capture unavailable (uncaptured execution) | fixed Aitken (section 3) |

## References

- Irons, B. M. & Tuck, R. C. (1969). A version of the Aitken accelerator for computer iteration. *International Journal for Numerical Methods in Engineering* 1(3), 275–277. <https://doi.org/10.1002/nme.1620010302>
- Küttler, U. & Wall, W. A. (2008). Fixed-point fluid–structure interaction solvers with dynamic relaxation. *Computational Mechanics* 43, 61–72. <https://doi.org/10.1007/s00466-008-0255-5>
- preCICE acceleration configuration (Aitken, IQN-ILS): <https://precice.org/configuration-acceleration.html>
- Degroote, J., Bathe, K.-J. & Vierendeels, J. (2009). Performance of a new partitioned procedure versus a monolithic procedure in fluid–structure interaction. *Computers & Structures* 87(11–12), 793–801. <https://doi.org/10.1016/j.compstruc.2008.11.013>
- Broyden, C. G. (1965). A class of methods for solving nonlinear simultaneous equations. *Math. Comp.* 19, 577–593. <https://doi.org/10.1090/S0025-5718-1965-0198670-6>
- Causin, P., Gerbeau, J.-F. & Nobile, F. (2005). Added-mass effect in the design of partitioned algorithms for fluid–structure problems. *Computer Methods in Applied Mechanics and Engineering* 194, 4506–4527. <https://doi.org/10.1016/j.cma.2004.12.005>
- Förster, C., Wall, W. A. & Ramm, E. (2007). Artificial added mass instabilities in sequential staggered coupling of nonlinear structures and incompressible viscous flows. *Computer Methods in Applied Mechanics and Engineering* 196, 1278–1293. <https://doi.org/10.1016/j.cma.2006.09.002>
- Felippa, C. A., Park, K. C. & Farhat, C. (2001). Partitioned analysis of coupled mechanical systems. *Computer Methods in Applied Mechanics and Engineering* 190, 3247–3270. <https://doi.org/10.1016/S0045-7825(00)00391-1>
- Guyan, R. J. (1965). Reduction of stiffness and mass matrices. *AIAA J.* 3(2), 380. <https://doi.org/10.2514/3.2874>
- Fernández, M. Á. & Moubachir, M. (2005). A Newton method using exact Jacobians for solving fluid–structure coupling. *Computers & Structures* 83, 127–142. <https://doi.org/10.1016/j.compstruc.2004.04.021>
- Michler, C., van Brummelen, E. H. & de Borst, R. (2005). An interface Newton–Krylov solver for fluid–structure interaction. *International Journal for Numerical Methods in Fluids* 47, 1189–1195. <https://doi.org/10.1002/fld.850>
- Anderson, D. G. (1965). Iterative procedures for nonlinear integral equations. *J. ACM* 12(4), 547–560. <https://doi.org/10.1145/321296.321305>
- Walker, H. F. & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis* 49(4), 1715–1735. <https://doi.org/10.1137/10078356X>
- Hilber, H. M., Hughes, T. J. R. & Taylor, R. L. (1977). Improved numerical dissipation for time integration algorithms in structural dynamics. *Earthq. Eng. Struct. Dyn.* 5, 283–292. <https://doi.org/10.1002/eqe.4290050306>
- Shabana, A. A. (1998). Computer implementation of the absolute nodal coordinate formulation for flexible multibody dynamics. *Nonlinear Dyn.* 16, 293–306. <https://doi.org/10.1023/A:1008072517368>
- Recuero, A. & Negrut, D. (2016). Chrono support for ANCF finite elements: formulation and validation aspects. UW–Madison SBEL Technical Report TR-2016-11 (the ANCF3423 shell element used here).
- Blondel, M. et al. (2022). Efficient and modular implicit differentiation. *NeurIPS*. <https://arxiv.org/abs/2105.15183>
