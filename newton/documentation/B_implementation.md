# Implementation Map

Minimal reference for locating the mathematics and the inchworm code. For the paper model see Section 2; for the dynamic extension Section 3; for solver layers Section 4.

**Solver layers:** **Soft → Deformable → Inflatable** under `newton/_src/solvers/`. The inchworm example uses **`SolverInflatable`**.

## Paper model (Section 2)

Summarized in **`02_paper_model.md`** and **§7** of **`07_mathematical_summary.md`** (geometry \(l\), \(\theta\), \(d\), \(x_c\), slippage \(\Delta\), slip force, kinematic \(\Delta d\) in the paper).

## Inchworm example

**`newton/examples/crawlable/example_inchworm_crawling.py`:** `TetraBox` mesh, chamber masks, `SolverInflatable`, traveling-wave gait (`gait_traveling_wave.py`), metrics in `inchworm/paper.py`.

**Main parameter names:** Paper: \(L\), \(M\), \(\beta\), \(\mu\), \(k\); gait: \(\gamma\), \(A\), \(\omega\), \(\psi\). Inflation: chamber pressures \(p_c\); torque: \(k_\tau\), \(k_d\), rest direction. Contact/joint vertex index sets for metrics and validation.
