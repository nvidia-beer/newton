# Introduction

## The Paper

**Gamus et al., "Understanding Legged Crawling for Soft Robots," arXiv:1911.05227**

The paper models a **soft inchworm**: a flexible body that crawls by bending and by alternating which foot slips on the ground. The formulation is a **three-link chain** in a plane with two point contacts (left and right feet). At each step, geometry determines load sharing between the feet; a **slippage criterion** decides which foot slips; and the body advances by a **kinematic displacement** \(\Delta d\). There is no time integration of slip forces—only geometry, equilibrium, and a single displacement per step. That minimal setup keeps the physics explicit and extensible.

## Relation to the Paper and Attribution

- **Section 2** is a summary of the model in Gamus et al. (arXiv:1911.05227). Equations are cited with the paper’s equation numbers where applicable (e.g. Eq. (3)–(5) for geometry, Eq. (6b) for normal forces, Eq. (8) for \(\dot{d}\), Eq. (9) for \(\Delta\), Eq. (10) for the slip force). The kinematic ground-contact constraint, tangential force balance \(f_{t,1} = -f_{t,2}\), signed slip force \(f_t = \mu f_{n,s}\operatorname{sign}(\dot{d})\), and stick–slip rule follow the paper (including Section II.B).
- **Sections 3–4 and the appendices** describe the **Newton simulation**: how that model is embedded in a dynamic, deformable-body simulation (implicit integration, FEM, springs, inflation, spine torque, self-contact) and how the solver is structured in four layers. The contact law and kinematic update \(\pm\Delta d\) are from the paper; the rest is the implementation and extension.

## Purpose and Structure

This document states the paper’s model in full mathematical form, explains how it is embedded in a dynamic deformable-body simulation, and describes the solver as a stack of four layers. The presentation is in equations and concepts rather than code; Appendix B gives the code and implementation map for the inchworm example.

| Section        | Content |
|----------------|---------|
| **Section 2**  | The paper's model: geometry (\(l\), \(\theta\), \(d\), \(x_c\)), equilibrium (normal forces), slippage criterion, slip law, kinematic update, gait. |
| **Section 3**  | From paper to simulation: implicit integration, deformable body (FEM, springs), inflation and chambers, spine torque, self-contact, coupling to the paper's stick–slip. |
| **Section 4**  | Solver layers: what each layer adds to \(A\) and \(\mathbf{f}\) (mass, damping, stiffness, forces). |
| **Section 5**  | Inchworm example: what it does, figures, key parameters, how to run. See Appendix B for code and parameter tables. |
| **Appendix A**  | Equation reference. |
| **Appendix B**  | Code and implementation map (inchworm code/parameter reference; paper and crawlable solver → code). |
