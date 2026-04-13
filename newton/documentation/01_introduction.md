# Introduction

## The Paper

**Gamus et al., "Understanding Legged Crawling for Soft Robots," arXiv:1911.05227**

The paper models a **soft inchworm**: a flexible body that crawls by bending and by alternating which foot slips on the ground. The formulation is a **three-link chain** in a plane with two point contacts (left and right feet). At each step, geometry determines load sharing between the feet; a **slippage criterion** decides which foot slips; and the body advances by a **kinematic displacement** \(\Delta d\). That minimal setup keeps the physics explicit and extensible.

## Relation to the Paper and Attribution

- **Section 2** summarizes the model in Gamus et al. (arXiv:1911.05227): geometry, equilibrium, slippage, slip force, kinematic update, gait.
- **Sections 3–4 and the appendices** describe the **Newton simulation**: implicit integration, FEM, springs, inflation, spine torque, self-contact, and **Coulomb ground contact** in `SolverSoft` / `SolverInflatable`. **§7** of `07_mathematical_summary.md` lists the paper’s geometry and slip-force relations next to the implementation.

## Purpose and Structure

This documentation states the paper’s model, explains how a deformable-body simulation is built in Newton, and describes the **three** core solver layers (Soft → Deformable → Inflatable). Appendix B maps the inchworm example to code and parameters.

| Section        | Content |
|----------------|---------|
| **Section 2**  | The paper's model (`02_paper_model.md`). |
| **Section 3**  | From paper to simulation: implicit integration, FEM, inflation, chambers, spine torque, self-contact (`03_dynamic_extension.md`). |
| **Section 4**  | Solver layers: what each layer adds to \(A\) and \(\mathbf{f}\) (`04_solver_layers.md`). |
| **Section 5**  | Inchworm example (`06_example_inchworm.md`). Appendix B: code map. |
| **Appendix A**  | Equation reference (`A_equations_reference.md`). |
| **Appendix B**  | Code map (`B_code_implementation.md`, `B_implementation.md`). |
