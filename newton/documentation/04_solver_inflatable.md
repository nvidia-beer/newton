# SolverInflatable: Inflation and Chambers

**Location:** `newton/_src/solvers/inflatable/solver_inflatable.py`, `kernels_inflatable.py`, `kernels_bend.py`

## Role

`SolverInflatable` extends `SolverDeformable` with **inflation**: the rest configuration of the FEM (tetrahedra and springs) is scaled so that the body “inflates” toward a target volume ratio. No explicit pressure force is applied; the material’s stiffness drives deformation toward the new rest state. **Chambers** allow different regions (e.g. left/right segments) to have different pressures for bending. **Torque** (bend resistance) is implemented via springs that have a rest direction and resist rotation out of that direction (spine stiffness).

## Inflation Mechanism: Rest-Configuration Scaling

- **Volume ratio:** pressure is represented as a **volume ratio** \(p = V/V_0\) (e.g. \(p=1\) no change, \(p=2\) double volume). Stored as `current_pressure`, clamped to `[1, max_volume_ratio]`.
- **Linear scale:** for 3D, linear dimensions scale as \(s = p^{1/3}\). So rest lengths and rest poses are scaled by \(s\) (springs) and \(\mathbf{D}_m^{-1}\) by \(1/s\) (tetrahedra).
- **Storage:** original rest state is stored once: `original_tet_poses`, `original_spring_rest_length`. At each pressure update, scaled values are written into `model.tet_poses` and `model.spring_rest_length`.

Kernels:

- **Springs:** `scale_spring_rest_lengths_kernel`: `rest_length_out = original_rest_length * scale`.
- **Tet poses:** `scale_tet_poses_kernel`: each 3×3 rest pose (e.g. \(\mathbf{D}_m^{-1}\)) is scaled by `inv_scale = 1/scale` so the rest shape is larger when \(p>1\).

## Per-Chamber Isotropic Expansion

- **Per-chamber tet poses:** `scale_tet_poses_per_chamber_kernel`: each tet has a chamber index from `tet_chamber_mask`; its rest pose is scaled isotropically by that chamber’s pressure \(s_c = p_c^{1/3}\). Mask \(-1\) means “no inflation” (stiff region).

## Chambers

- **Chamber layout:** the mesh is divided into logical chambers (e.g. 2×1×2 along X, Y, Z). Each tetrahedron and each spring is assigned to a chamber via `tet_chamber_mask` and `spring_chamber_mask` (integer indices; \(-1\) = non-inflatable).
- **Per-chamber pressure:** `set_chamber_mask(tet_chamber_mask, spring_chamber_mask, num_chambers)` then `set_chamber_pressures([p0, p1, ...])`. Each chamber \(c\) has pressure \(p_c\); springs/tets in chamber \(c\) use \(s_c = p_c^{1/3}\) (isotropic).
- **Inchworm:** chambers 1 and 3 (left and right top segments) are inflated; chambers 0 and 2 can be disabled (`chamber_inflation_disabled`) so the “backbone” stays stiffer. Names in example: `inflatable_chambers`, `chamber_pressures`, `tet_chamber_mask`, `spring_chamber_mask`.

## Volume and Ratio

- **Current volume:** sum of tetrahedron volumes from current positions; kernel `compute_volume_kernel` (volume = \(|det(\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3)|/6\)).
- **Initial volume:** cached at first step (or from initial state) in `_initial_volume`.
- **Volume ratio:** `get_volume_ratio(state)` = current volume / initial volume; used for logging and tuning.

## Torque (Folding / Spine Stiffness)

- **Idea:** some springs are “spine” springs (e.g. along the length axis). They have a **rest direction** stored in `spring_rest_direction`. In addition to the linear spring force, a **torque** resists rotation of the spring axis away from this rest direction.
- **Model:** let \(\mathbf{d}\) be the current unit direction of the spring and \(\mathbf{d}_0\) the rest direction. Angle \(\alpha = \arccos(\mathbf{d}\cdot\mathbf{d}_0)\) (notation \(\alpha\) to avoid collision with the paper’s central link angle \(\theta\)). Torque magnitude: \(\tau = -(k_\tau\,\alpha + k_d\,\omega)\,L\), where \(\omega\) is angular velocity along the rotation axis. This torque is converted to forces at the two endpoints: \(\mathbf{F} = (\tau\,\mathbf{a}\times\mathbf{d})/L\) (and opposite at the other vertex), with \(\mathbf{a}\) the rotation axis.
- **Kernel:** `eval_springs_linear_and_torque` in `kernels_bend.py`. When `torque_stiffness > 0` and `spring_rest_direction` is set, `SolverInflatable.eval_spring_forces` uses this kernel instead of the standard spring kernel.
- **Parameters:** `torque_stiffness`, `torque_damping`, `spring_rest_direction` (per-spring vec3). In the inchworm, only springs aligned with the long axis (X) get a non-zero rest direction so the spine resists bending.

## Naming in Code

- `set_pressure`, `set_chamber_pressures`, `set_chamber_mask`
- `original_tet_poses`, `original_spring_rest_length`, `tet_chamber_mask`, `spring_chamber_mask`, `_chamber_pressures_array`
- `scale_tet_poses_kernel`, `scale_tet_poses_per_chamber_kernel`, `scale_spring_rest_lengths_per_chamber_kernel`
- `eval_springs_linear_and_torque`, `torque_stiffness`, `torque_damping`, `spring_rest_direction`
- `compute_volume`, `get_volume_ratio`, `get_initial_volume`
