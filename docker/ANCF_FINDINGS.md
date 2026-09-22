# ANCF / super jeep — consolidated findings

Distilled from `AUDIT_ANCF_JEEP.md`, `TASK_ANCF_JEEP.md`, `PERF_ANCF_SAND_RL.md`,
`warp-mpm-optimize.md` (2026-09-09 to 2026-09-14 sessions, deleted after consolidation).
Paths relative to `third_party/newton/`. `S` = `newton/_src/solvers/ancf_shell/solver_ancf_shell.py`,
`K` = `.../kernels_stiffness.py`, `E` = `.../kernels_element.py`, `A` = `.../kernels_assembly.py`,
`C` = `.../kernels_contact.py`, `R` = `.../solver_ancf_shell_rigid.py`.
Chrono reference = `/home/beer/Dev/Physics/chrono/src/chrono/fea/ChElementShellANCF_3423.cpp`.
Known-good locked budget: `E=50MPa, kn=20k, alpha-damp=0.15, substeps=10, nr=2, pcg=25`.

## Suspension L/R mirror bug (fixed 2026-09-14)

`double_wishbone_rigid.xml`/`superjeep_rims_only.xml` negated every Y-coordinate ported from
the Chrono donor vehicle's `*_DoubleWishbone.cpp`/`*_PitmanArm.cpp` (chrono_models/vehicle/feda/) on the (wrong) premise that
Chrono uses "Y=left" while the MJCF needed "Y=right". For any right-handed X-forward/
Z-up frame — which both Chrono's `ChDoubleWishbone` and MuJoCo use — Y=left is forced
by the right-hand rule (`Y = Z×X`); there is no valid right-handed frame with X-forward/
Z-up where +Y is right. The negation was a real mirror transform, applied uniformly to
every hardpoint (wishbone, upright, spindle, tierod, Pitman arm, idler arm), confirmed
against the Chrono source to 5 significant figures. Net effect: `_fl` bodies sat at -Y
(physically the vehicle's right side) — a left-right-mirrored vehicle. Topology (loop
closures, joint DOFs) was otherwise correct; this was purely a sign bug.

Fixed by removing the negation (Chrono's raw values now used as-is). Downstream code that
had compensated for the bug also needed reverting: `example_double_wishbone_ancf_tires.py`'s
`_WHEEL_ORDER`/`_SPINDLE_ZU` display-index workaround (was `(("FL",1),("FR",0),...)`, now
the natural `(("FL",0),("FR",1),...)`), and `env_cfg.py`'s `spindle_positions` tuple.
A 20-geom purple "chassis wireframe" visual block (`contype="0"`, not part of Chrono's
model, redundant with the per-arm capsules) was briefly removed as excessive, then
restored on request — kept in both MJCF files with corrected coordinates.

Also found and fixed while reviewing the render: the upright knuckle's TIEROD_U marker
offset was copy-pasted identically across all 4 corners, but is only geometrically
correct for the front (tierod routes through the pitman-arm/steering-link chain); the
rear tierods mount directly to chassis at a different fore-aft point, so the shared
marker was off by 0.414 m in X for RL/RR — fixed with a rear-specific offset
(`+0.22464, ±0.15178, -0.30589` vs front's `-0.18941, ±0.15178, -0.30589`).

**TSDA per-axle spring tuning (2026-09-15):** all 4 corners previously shared one
`springlength` (0.72763), but the chassis COM (`x=-0.111`, from the Chrono donor `*_Chassis.cpp`) sits
0.111 m aft of the wheelbase midpoint, giving a 46.64/53.36 front/rear static load split
(12977 N vs 14848 N per corner, from chassis mass 5672.87 kg alone) — with a uniform
spring the rear visibly sagged lower than the front. Fixed with per-axle `springlength`:
front 0.77275 m, rear 0.79737 m (same `k=76000`/`c=23000` for both), so each axle's
preload `k*(L0-0.602)` matches its actual static share. Unsprung mass/payload excluded,
so expect a small residual settle — not yet re-verified in Docker.

## Open solver bugs (ranked by impact), vs Chrono `ChElementShellANCF_3423`

1. **EAS never applied in the batched path** — `K:875-886` zeroes `elem_eas_alpha`
   every NR iteration right before it's read; net effect is EAS ≡ 0 (thickness/membrane
   locking back). Fix: stop zeroing alpha per-iteration, only zero `elem_HE`/`elem_KA`;
   drop the `±0.1` clamp. **Caution:** enabling this alone (without §7, natural-frame
   ANS/EAS) caused `ancf_rigid_mujoco_tires` to go NaN in bisection — §7 must land first.
2. **Read-before-write race on `rz_old` inside PCG** — `S:260-289`, unordered blocks
   break conjugacy nondeterministically. Fix: ping-pong `rz_old`/`rz_new` in Python
   instead of writing back in-kernel.
3. **Dirichlet (bead) rows solved unconstrained, then discarded** — `S:1116-1132`;
   K_eff/residual keep bead rows, PCG spends iterations on them, `_zero_dirichlet_acc`
   discards the result afterward. Fix: static per-DOF mask zeroing rows/cols + diagonal=1
   in `_update_K_eff_inplace_batched`, mask residual before `_negate` and before
   `_accum_nr_residual_sq`.
4. **HHT history force one iterate stale** — `S:1611` copies `global_f_int→global_f_int0`
   before the final Δa/`_hht_kinematic`. Fix: re-evaluate force once after final kinematic
   update.
5. **Lumped translational mass is 1/3 of physical** — `E:731-742`: `m_elem/12` per DOF ×3
   axes = `m_elem/3` total vs Chrono's `m_elem/4`. Cascades: gravity 1/3 physical (+96N
   spurious lift per tire vs analytic `tare_fz`), `contact_kn_ceiling` 3× too low, wave
   speed √3 too high, `kd·dt/m` 3× larger than derived. Fix: `m_per_pos = m_elem/4.0`
   (changes every example's tuning).
6. **Director DOFs given full position mass** — `E:732-737`; should be `~5e-6·m_pos`
   for h=10mm, not `m_per_pos`. Comment claims it was for PCG conditioning but the
   Jacobi preconditioner already removes diagonal scaling from that. Fix in preconditioner
   (block-Jacobi 6×6) if conditioning regresses, not in M.
7. **ANS/EAS applied in global frame, not natural (ξ,η,ζ) frame** — `E:217-233` vs Chrono
   `:440-446,1249-1268,1080-1087`. For the tire mesh this is only correct at θ=90° around
   the circumference; elsewhere NR converges to the wrong equilibrium. Fix: precompute
   per-element rest rotation `R0` from `J0c`, rotate E/g into that frame before ANS/EAS.
   Verify on `ancf_shell_drop` (flat plate) first. Highest-impact, largest patch.
8. **Contact friction/damping explicit, not in tangent** — `C:49-67,105-117`; regularized
   Coulomb slope ~5.8e5 N·s/m untangented → 600N/node chatter, no true static friction.
   `kd·dt/m` ≈ 3.5 (>2) with the 1/3 mass bug. Fix: add velocity Jacobians to tangent
   diagonal with HHT factor `γ/(βdt)`.
9. Tangent is Gauss-Newton only (material part); missing initial-stress term
   `GdᵀΣGd` and EAS condensation `−GᵀK_α⁻¹G` — linear not quadratic NR convergence.
   K_eff stays SPD by construction (no cuDSS/Gershgorin issue). Add after 1-7 land.
10. cuDSS path solves a *different* system (Gershgorin shift ≈ heavily under-relaxed
    modified Newton, fp32). **Decision made: dropped cuDSS entirely** (3.3× slower,
    461µs/tire sequential factorization, not batched) — removed from the codebase.
11. `nr_max_du` trust-region clamp — **removed** (never fired once 1-4 are fixed; was
    masking divergence).
12. Dead work removed: unassembled pressure tangent `elem_Kp`, debug dump blocks,
    `_eager` hook.

**Recommended order:** (0) sanity checks with `--set gs-iters=1` /
`update_data_interval=1` → (1) §2+§3+§1 (rz_old, Dirichlet mask, EAS) verified against
`ancf_rim_shell`/`ancf_rigid_mujoco_tires` as regression baselines → (2) example-8-specific
coupling bugs below → (3) §4 (HHT) + §8 (contact tangent) → (4) §5+§6 (mass) → (5) locked
budget re-verification → (6) §7 (ANS/EAS frame) + §9 (geometric stiffness) on flat plate
first → (7) perf items below.

**One physics change per Docker run** — stacking multiple fixes at once made
`ancf_rigid_mujoco_tires` go NaN with no way to attribute which change caused it.

## Example-8 (`double_wishbone_ancf_tires`) coupling bugs — all CONFIRMED, proved by bisection

Root cause of instability, ranked: (1) GS coupler never resyncs MuJoCo — with
`update_data_interval=0` the vehicle is integrated `gs_iters` times per substep, not once
(fix: `update_data_interval=1`, or resync after `_restore_rigid`); (2) hub velocity read
from `cvel` without COM-offset correction — wrong by `ω × 1.65m` once wheels spin, fixed
via `v_hub = cvel_lin + ω × (xpos − subtree_com[root])`; (3) explicit friction/damping
chatter under load; (4) 1/3 mass bug tightens `kd·dt/m` margins 3×; (5) velocity-proportional
bugs (bead-predictor double-count from `vel_predict_dt=dt` stacking with the HHT
predictor, GS k>0 extrapolating from an already-advanced pose, `α_m` mass-proportional
damping leaking into the hub wrench as drag) are all invisible parked and only bite when
driving. The `§3.2.2 cvel COM-offset fix` was verified to make the vehicle settle cleanly
(drift 0.00mm, corner loads balance to vehicle weight, no divergence in 180 frames).

## Performance — GPU-latency bound, not compute bound

Single vehicle at the locked budget: ~200 tiny CUDA graph nodes per NR iteration (150 from
the PCG loop), ~20% idle gaps, GPU busy only ~53% under nsys. In order of payoff:

1. Restore locked budget once correctness bugs are fixed (64× fewer NR iterations/frame
   than the 80/16 that was compensating for the bugs).
2. Fuse the PCG loop into one kernel per tire (block-reduced dot/xrz, K streamed from L2)
   — replaces 150 graph nodes with 1, ~295→60-80µs.
3. Delete unassembled `elem_Kp` pressure-tangent computation.
4. MuJoCo `iterations` 50→2-6 (already done in example 2's config).
5. Fold ~10 memsets + ~20 one-liner kernels into two ("assemble f_ext + residual",
   "zero scratch").
6. `compute_element_K_from_B` via `wp.tile_matmul` per element instead of one thread per
   entry re-reading B.

Element kernel `compute_element_forces_stiffness_batched_gp` is register-bound (255
regs/thread → 1 block/SM, ¼ occupancy) — fine at 4 tires, becomes the dominant cost at
E≥10 parallel envs (~25ms/frame). Fix for RL scale: thread-per-Gauss-point decomposition
targeting ≤64 regs.

## MPM (implicit sand solver) — RL-scale bottlenecks, ranked

Measured on `double_wishbone_ancf_sand` (GB10, 1.84M particles, 4 tires): 120ms→38ms/frame
(26fps) through `collider_basis="Q1"`, `gs-iters=1` (whole DW frame as one graph), PCG
memset folding, active-particle compaction, cell-budget tightening (2×→1.25×), fused PCG.
Physics unchanged throughout (drift 0.00mm, Fz 14-16.8kN/corner).

- **B1** (biggest at scale): ANCF element kernel register-bound, same as above — ~25ms/frame
  at E=10 as-is, ~6ms after fix.
- **B2**: ANCF PCG is a constant ~2000-launch floor per frame (6.7ms) — batching envs into
  one launch keeps it flat but it never gets cheaper without algorithmic change (block-Jacobi
  6×6 preconditioner, or fuse dot into xrz).
- **B3**: MPM node sorts (`compress_node_indices`) rebuild the full space partition every
  step even though the active cell set barely changes — linear in cell budget × envs
  (~27ms/frame at E=10 as-is). Space-reuse-with-dilation was prototyped and **removed**
  (net worse at E=1 — dilated empty cells get iterated 25× by `gs_solve`); worth re-adding
  at E≥10 without dilation (k=4, accepting particles crossing cell boundaries are frozen
  ≤k-1 frames).
- **B4**: MPM Gauss-Seidel fixed at 25 iterations × 8 colours = 200 launches/frame,
  sub-linear in E only if all envs share one MPM solver (`environment_count`) — must not
  run E separate solvers.
- **B5**: budget-sized nodal kernels scale linearly with cell budget; shrinks automatically
  if B3 tightens the budget to the real active count.
- **B6**: per-tire Python loops in coupling (`accumulate_wheel_wrenches`) issue 2
  launches/tire/substep — fix with one kernel `dim=n_tires` instead of a Python for-loop.
- **B7**: O(N) particle passes (self-copy, memsets) scale with total particle count across
  envs; skip self-copy when `state_in is state_out`.
- **B8** (stability, not throughput): sand→tire force path is explicit with a one-frame
  lag (MPM at 60Hz, ANCF at 600Hz). Stable at E=50MPa/10mm; thinner/softer tires need a
  substep-rate MPM or spindle-only path.

Not bottlenecks (measured, don't chase): MuJoCo (2ms/frame, batches natively), DtoD sync
copies, host graph-launch overhead, rendering (irrelevant headless).

## MPM optimization ideas not yet tried (catalog, unranked beyond §ordering below)

Because only the 4 tires ever touch MPM particles (chassis/suspension never contact
sand/snow/water), several generic MPM optimizations specialize into cheaper tire-driven
versions. Investigation order: (1) confirm collision broadphase is already tire-only, not
testing the whole vehicle against the grid; (2) tire-driven active-region tracking (bound
grid/particle activation to 4 moving AABBs around tire footprints); (3) particle
down-sampling + fixed budget per env; (4) sparse/block-based active grid (check if already
present); (5) MLS-MPM/APIC transfer if not already used; (6) simplified contact Jacobian
restricted to the 4 known tire bodies; (7) regional/async time-stepping (highest effort,
only if still needed); (8) convex MPM-rigid coupling / neural hybrid fallback (research-
frontier, last resort).

## Required init sequence for `SolverANCFShell` + `InterfaceCouplerGS`

(Also in memory as [[feedback_code_must_run]].) Every step required before an example is
runnable:

1. `build_ancf_tire_mesh` + `SolverANCFShell(...)`
2. `set_dirichlet_nodes` + `set_cavity`
3. `ancf_solver.capture_graph(sim_dt)` — **required before `graph_step`**
4. Restore `node_x`/`node_D`/zero velocities after capture warmup
5. Build rendering model (`add_world` + particles + triangles) + finalize
6. `SolverMuJoCo(model)`
7. `InterfaceCouplerGS` + `coupler.allocate(state_0)`
8. Seed `f_int0`: `step_kinematics` → `_prescribe_beads` → `recompute_f_int` → copy `f_int0`
9. `coupler._save_rigid` + `coupler._pack_ancf` — re-prime double buffers

## Example 6 (`double_wishbone_ancf_terrain`) — heightfield terrain + Chrono SCM soil (2026-09-15, unrun)

> **Retired 2026-09-17.** Replaced by `double_wishbone_track` (config `06_double_wishbone_track.json`):
> terrains come from `third_party/newton-terrain-tool` (TrackGen corridor + lunar highland) instead of
> the generated rock field, and the vehicle is driven through the `TrackDriver` policy interface.
> `TerrainSCM` (rigid + Bekker-Wong soil) stays in the solver; only the rigid mode is wired into the
> new example. The notes below are kept as the design record of the terrain contact hook.

Goal: RL on "vertically challenging" terrain (arXiv:2409.02383 / 2409.17469: Verti-4-Wheeler on
Chrono, rigid triangle mesh from a grayscale BMP, obs = heading error + speed + SWAE(64x64 local
heightmap), action = speed + steer, reward 50·Δd − 10·stall − 20·Σmax(0,|roll,pitch|−30°) −
timeout(10·d+100), T = 15–20 s; neither paper models tire compliance or inflation — CTIS is new).

- **Heightfield, not mesh.** ANCF tires are point clouds, so the surface query is one bilinear
  lookup + gradient normal (O(1), no BVH, graph-capturable) and the soil's plastic state lives on
  the same 2D grid. Grid convention = `newton.Heightfield` (row = Y, col = X, Z-up, centred).
  Same grid goes to MuJoCo as an `hfield` (hidden shape) for the rim stops / chassis.
- **Solver hook** `SolverANCFShell.terrain` (`terrain_scm.TerrainSCM`): `update_plastic()` once
  per substep before `_hht_predict` (Chrono `ComputeInternalForces` order), `apply_contact()` in
  every NR residual eval right after `apply_ground_contact*`. Forces land in `global_f_ext`, so
  `accumulate_wheel_wrenches` picks them up unchanged. The flat plane is parked at z = −1e3.
- **Locked budget preserved.** Elastic branch is the flat kernel's `kn/kd/mu` law along the
  local normal (Chrono `elastic_K ≡ kn / A_node`; `damping_R ≡ kd / A_node`), with the exact
  diagonal tangent. `--soil rigid` is therefore the flat-plane physics on a bumpy surface. The
  soil adds only: Bekker-Wong yield `σ = (Kc/b + Kphi)·z^n` → cell `sigma_yield`/`z_p`
  (monotone, `atomic_max`, so ruts persist and unloading is elastic), Janosi-Hanamoto
  `(1 − exp(−j/K))` on the shear, and cohesion. Presets from Chrono demos (hard/soft/LETE
  sand/regolith). Expected on `hard`: tread node ≈ 625 N / 0.01 m² ≈ 62 kPa → z ≈ 4 cm rut.
- Terrain PNG (16-bit, white = `--terrain-max-h`) is generated once into
  `examples/ancf/assets/terrain_hills_<res>_<seed>.png` (rock field: fractal noise + rounded
  boulders) — edit in any image editor. `--terrain-difficulty w` scales the map (papers' Eq. 1
  blend flat↔rugged), `--terrain-flat-radius` levels the spawn disc.
- gs-iters > 1: `z_p`, `sigma_yield`, `kshear` are mutated by the ANCF step, so the example
  re-registers them with `InterfaceCouplerGS.allocate(extra_arrays=...)`.
- Known simplification vs Chrono: no bulldozing/erosion stencil, fixed `1/b = 1/tire_width`
  instead of the per-patch convex-hull perimeter/area, plastic write goes to the nearest grid
  node only (grid cell 15.6 cm at 256 px / 40 m vs ~20 cm circumferential node spacing).
