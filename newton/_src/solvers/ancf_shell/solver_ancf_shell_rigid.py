# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""SolverANCFShellRigid — ANCF shell solver with per-wheel bilateral-ready coupling.

Extends :class:`SolverANCFShell` with:

1. Per-spindle xfrc staging arrays (N_tires separate arrays instead of one flat
   batched array) so each wheel's reaction wrench is accumulated independently.
2. Gauss-Seidel coupling iterations for tighter FEM-to-rigid coupling: callers
   can drive multiple ANCF-step / wrench-accumulate / MuJoCo-dynamics cycles
   inside a single substep to reduce the explicit-coupling lag.
3. A :meth:`set_spindle_indices` convenience API for multi-spindle configurations
   that registers all wheel spindle body indices at once.
4. Preparation API for future bilateral rim-body constraint upgrade: staging arrays
   and accumulation kernels are structured to be plugged into a constraint solver
   without restructuring the per-tire data layout.

Coordinate conventions (unchanged from the rest of the ANCF stack)::

    MuJoCo Z-up : x_fwd, y_lat, z_up
    ANCF Y-up   : x_lat, y_up,  z_fwd   (axle along X, tread at Y=0)
    Z-up -> Y-up : (x, y, z) -> (y, z, x)
    Y-up -> Z-up : (x, y, z) -> (z, x, y)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.solvers.ancf_shell.solver_ancf_shell import SolverANCFShell

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _accum_bead_wrench_wheel(
    global_f_int: wp.array[float],
    node_x: wp.array[wp.vec3],
    bead_idx: wp.array[wp.int32],  # GLOBAL flat indices into N*n_nodes
    xpos: wp.array2d[wp.vec3],
    world_idx: int,  # MuJoCo world index for this tire (env in parallel-world)
    spindle_mj: int,  # MuJoCo body index of spindle in world_idx
    lateral_offset: float,  # ANCF X offset to add to hub_x (e * lateral_spacing)
    staging: wp.array[wp.spatial_vector],  # shape (1,)
    n_bead: int,
):
    """Accumulate bead constraint reactions for one tire into ``staging[0]``.

    Dim = n_bead.  bead_idx holds GLOBAL flat indices (already offset by e*n_nodes).

    world_idx / lateral_offset fix the two bugs in the original implementation:
      1. world_idx  — reads xpos[world_idx, spindle_mj] not xpos[0, ...] so that
                      each parallel MuJoCo world gets its own hub position.
      2. lateral_offset — adds e * lateral_spacing to hub_x in ANCF Y-up so the
                           moment arm is computed from the correct hub centre
                           (parallel worlds share MuJoCo Y=0 but ANCF tires are
                           laterally spaced).
    """
    i = wp.tid()
    global_idx = bead_idx[i]  # already global
    base = global_idx * 6

    fx = -global_f_int[base + 0]
    fy = -global_f_int[base + 1]
    fz = -global_f_int[base + 2]
    f = wp.vec3(fx, fy, fz)

    # Hub centre: MuJoCo Z-up (x_fwd, y_lat, z_up) -> ANCF Y-up (x_lat, y_up, z_fwd)
    # + lateral_offset corrects for parallel-world layout where all MuJoCo spindles
    # sit at Y=0 but ANCF tires are spread in X by lateral_spacing.
    pos_zu = xpos[world_idx, spindle_mj]
    hub_ancf = wp.vec3(pos_zu[1] + lateral_offset, pos_zu[2], pos_zu[0])

    r = node_x[global_idx] - hub_ancf
    tau = wp.cross(r, f)

    wp.atomic_add(staging, 0, wp.spatial_vector(tau[0], tau[1], tau[2], fx, fy, fz))


@wp.kernel
def _staging_to_xfrc_wheel(
    staging: wp.array[wp.spatial_vector],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    world_idx: int,  # MuJoCo world index — fixes parallel-world write target
    spindle_mj: int,
    alpha: float,
    tare_fz: float,
):
    """Map ANCF Y-up staging wrench to MuJoCo Z-up ``xfrc_applied[world_idx, spindle_mj]``.

    Single-threaded (dim = 1).

    world_idx selects the correct MuJoCo world:
      parallel-world layout (ancf_rigid_mujoco_tires): world_idx = tire_idx
      single-world layout   (double_wishbone):          world_idx = 0

    Coupling architecture (wrench decomposition):
      Forces  (f_zu)  — TWO-WAY: ANCF bead reactions → xfrc_applied, and
                        MuJoCo xpos/cvel → bead prescription each substep.
      Torques         — ONE-WAY / ZEROED: spin is kinematically prescribed via
                        rim_omega_wp (not a MuJoCo DOF); the other two torque
                        axes also have no corresponding MuJoCo rotational DOF on
                        the spindle, so all three torque components are zeroed
                        rather than letting MuJoCo absorb them silently as
                        constraint forces.
    """
    w = staging[0]
    f_zu = wp.vec3(w[5], w[3], w[4] + tare_fz)

    cur = xfrc_applied[world_idx, spindle_mj]
    xfrc_applied[world_idx, spindle_mj] = wp.spatial_vector(
        cur[0] + alpha * f_zu[0],
        cur[1] + alpha * f_zu[1],
        cur[2] + alpha * f_zu[2],
        cur[3],
        cur[4],
        cur[5],
    )


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class SolverANCFShellRigid(SolverANCFShell):
    """ANCF shell solver with per-wheel bilateral-ready coupling to a rigid body.

    Wraps :class:`SolverANCFShell` (``n_envs = n_tires``) and adds per-wheel
    staging arrays plus the coupling API used by multi-tire rigid-vehicle examples.

    Typical usage::

        ancf = SolverANCFShellRigid(model, ancf_model, n_tires=4, ...)
        for w in range(4):
            ancf.setup_wheel(w, spindle_mj=spindle_ids[w],
                             bead_idx_np=bead_np, tare_fz=tire_weight)
        ancf.capture_graph(dt)

        # Inside substep loop:
        ancf.graph_step()
        ancf.accumulate_wheel_wrenches(xfrc_applied, xpos)
        mj_solver.step_dynamics(xfrc_applied=xfrc_applied)
    """

    def __init__(
        self,
        model,
        ancf_model,
        n_tires: int | None = None,
        **kwargs,
    ):
        """Initialise the per-wheel coupling data.

        Args:
            model: Newton :class:`~newton.Model` (provides device and gravity).
            ancf_model: :class:`~newton._src.solvers.ancf_shell.ANCFShellModel`
                holding the shared tire mesh and material data.
            n_tires: number of tires (= number of ANCF environments).
                If None, falls back to ``n_envs`` in kwargs (so callers that use
                the base-class ``n_envs=N`` API work without change).
            **kwargs: forwarded verbatim to :class:`SolverANCFShell`.
        """
        if n_tires is None:
            # Caller used the SolverANCFShell API: n_envs=N. Use that value.
            n_tires = int(kwargs.pop("n_envs", 4))
        else:
            kwargs.pop("n_envs", None)
        super().__init__(model, ancf_model, n_envs=n_tires, **kwargs)
        self.n_tires = n_tires

        # Per-tire coupling data — populated by setup_wheel().
        self._spindle_mj_arr: list[int] = []
        self._world_idx_per_tire: list[int] = []  # MuJoCo world for each tire
        self._lateral_offset_per_tire: list[float] = []  # ANCF X hub offset per tire
        self._bead_idx_per_tire: list[wp.array] = []
        self._xfrc_stg_per_tire: list[wp.array] = []
        self._tare_fz_per_tire: list[float] = []

        self._coupling_alpha: float = 1.0
        # Set True for one-shot per-wheel wrench diagnostics (adds D→H sync per wheel).
        self._debug_wrenches: bool = False

    # ------------------------------------------------------------------
    # Setup API
    # ------------------------------------------------------------------

    def setup_wheel(
        self,
        tire_idx: int,
        spindle_mj: int,
        bead_idx_np: np.ndarray,
        tare_fz: float,
        world_idx: int | None = None,
        lateral_offset: float = 0.0,
        device: str = "cuda:0",
    ) -> None:
        """Register one wheel's coupling data.

        Must be called once per tire before :meth:`capture_graph`.

        Args:
            tire_idx: 0-based wheel index (= ANCF env index).
            spindle_mj: MuJoCo body index of this wheel's spindle.
            bead_idx_np: GLOBAL flat bead node indices into the N*n_nodes array.
            tare_fz: tire weight tare [N].
            world_idx: MuJoCo world containing this spindle.
                       Defaults to ``tire_idx`` (parallel-world layout).
                       Pass 0 for single-world layout (double_wishbone style).
            lateral_offset: ANCF X offset added to hub position [m].
                            For parallel-world layout: ``tire_idx * lateral_spacing``.
                            For single-world layout: 0 (spindle at correct Y already).
            device: CUDA device string.
        """
        w_idx = tire_idx if world_idx is None else world_idx

        while len(self._spindle_mj_arr) <= tire_idx:
            self._spindle_mj_arr.append(-1)
            self._world_idx_per_tire.append(0)
            self._lateral_offset_per_tire.append(0.0)
            self._bead_idx_per_tire.append(None)
            self._xfrc_stg_per_tire.append(None)
            self._tare_fz_per_tire.append(0.0)

        self._spindle_mj_arr[tire_idx] = int(spindle_mj)
        self._world_idx_per_tire[tire_idx] = int(w_idx)
        self._lateral_offset_per_tire[tire_idx] = float(lateral_offset)
        self._bead_idx_per_tire[tire_idx] = wp.array(bead_idx_np.astype(np.int32), dtype=wp.int32, device=device)
        self._xfrc_stg_per_tire[tire_idx] = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        self._tare_fz_per_tire[tire_idx] = float(tare_fz)

    def set_spindle_indices(self, spindle_indices: list[int]) -> None:
        """Set MuJoCo body indices for all spindles at once.

        Convenience wrapper for multi-spindle configurations where bead data has
        already been registered via :meth:`setup_wheel` and only the spindle body
        indices need to be updated (e.g. after re-loading a MuJoCo model).

        Args:
            spindle_indices: MuJoCo body indices, one per tire in tire order.
        """
        if len(spindle_indices) != self.n_tires:
            raise ValueError(f"set_spindle_indices: expected {self.n_tires} indices, got {len(spindle_indices)}.")
        self._spindle_mj_arr = [int(idx) for idx in spindle_indices]

    # ------------------------------------------------------------------
    # Coupling
    # ------------------------------------------------------------------

    def accumulate_wheel_wrenches(
        self,
        xfrc_applied: wp.array2d,
        xpos: wp.array2d,
        device: str = "cuda:0",
    ) -> None:
        """Accumulate bead reactions for all wheels and write to ``xfrc_applied``.

        Call after the ANCF step (or after each GS ANCF sub-iteration) and before
        the MuJoCo ``step_dynamics`` call.  The method does **not** zero
        ``xfrc_applied`` — the caller is responsible for clearing it each substep
        before invoking this method, so that multiple GS coupling passes add
        incrementally.

        Args:
            xfrc_applied: MuJoCo external force array, shape (n_worlds, n_bodies),
                dtype ``wp.spatial_vector``.  Written in-place.
            xpos: MuJoCo body world-frame positions from the most recent
                ``step_kinematics`` call, shape (n_worlds, n_bodies),
                dtype ``wp.vec3``.
            device: CUDA device string for kernel launches.
        """

        for w in range(self.n_tires):
            stg = self._xfrc_stg_per_tire[w]
            if stg is None:
                raise RuntimeError(f"Wheel {w} has no coupling data — call setup_wheel() first.")

            stg.zero_()
            n_bead = len(self._bead_idx_per_tire[w])
            world_idx = self._world_idx_per_tire[w]
            lat_off = self._lateral_offset_per_tire[w]

            wp.launch(
                _accum_bead_wrench_wheel,
                dim=n_bead,
                inputs=[
                    self.global_f_int,
                    self.node_x,
                    self._bead_idx_per_tire[w],
                    xpos,
                    world_idx,
                    self._spindle_mj_arr[w],
                    lat_off,
                    stg,
                    n_bead,
                ],
                device=device,
            )

            wp.launch(
                _staging_to_xfrc_wheel,
                dim=1,
                inputs=[
                    stg,
                    xfrc_applied,
                    world_idx,
                    self._spindle_mj_arr[w],
                    self._coupling_alpha,
                    self._tare_fz_per_tire[w],
                ],
                device=device,
            )

            if self._debug_wrenches:
                wp.synchronize_device(device)
                sv = stg.numpy()[0]
                tare = self._tare_fz_per_tire[w]
                # ANCF Y-up → Z-up: fz→fx_zu, fx→fy_zu, fy+tare→fz_zu
                print(
                    f"  [rigid-dbg] wheel={w} world={world_idx} smj={self._spindle_mj_arr[w]}"
                    f"  lat_off={lat_off:.3f}m"
                    f"  staging_raw=(fx={sv[3]:.1f} fy={sv[4]:.1f} fz={sv[4] + tare:.1f} N)"
                    f"  torque=({sv[0]:.1f},{sv[1]:.1f},{sv[2]:.1f} N·m)"
                )
