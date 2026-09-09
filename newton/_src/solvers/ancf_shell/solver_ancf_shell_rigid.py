# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""SolverANCFShellRigid — ANCF shell solver with per-wheel coupling to rigid spindles.

Extends :class:`SolverANCFShell` (``n_envs = n_tires``) with per-wheel staging
of the tyre's net external load and its transfer to MuJoCo's ``xfrc_applied``.

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


@wp.kernel
def _accum_external_wrench_wheel(
    global_f_ext: wp.array[float],
    node_x: wp.array[wp.vec3],
    xpos: wp.array2d[wp.vec3],
    world_idx: int,
    spindle_mj: int,
    lateral_offset: float,
    node_base: int,  # e * n_nodes — first global node index of this tire
    staging: wp.array[wp.spatial_vector],  # shape (1,)
):
    """Accumulate the tire's net EXTERNAL load into ``staging[0]``.  Dim = n_nodes.

    For a quasi-static tire the hub reaction equals the sum of the external
    forces acting on the shell (contact + gravity + pressure), which is what
    ``global_f_ext`` holds.  The gravity term is cancelled downstream by
    ``tare_fz``.  Bead-node ``global_f_int`` must NOT be used for this: at
    Dirichlet nodes it is the constraint reaction, O(E*V), not the transmitted
    load.  Staging layout: (tau_x, tau_y, tau_z, f_x, f_y, f_z) in ANCF Y-up.
    """
    i = wp.tid()
    global_idx = node_base + i
    base = global_idx * 6

    f = wp.vec3(global_f_ext[base + 0], global_f_ext[base + 1], global_f_ext[base + 2])

    pos_zu = xpos[world_idx, spindle_mj]
    hub_ancf = wp.vec3(pos_zu[1] + lateral_offset, pos_zu[2], pos_zu[0])
    r = node_x[global_idx] - hub_ancf
    tau = wp.cross(r, f)

    wp.atomic_add(staging, 0, wp.spatial_vector(tau[0], tau[1], tau[2], f[0], f[1], f[2]))


@wp.kernel
def _staging_to_xfrc_wheel(
    staging: wp.array[wp.spatial_vector],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    world_idx: int,
    spindle_mj: int,
    tare_fz: float,
    torque_alpha: float,  # scale on the reaction torque; 0 = force only
):
    """Map the ANCF Y-up staging wrench to MuJoCo Z-up ``xfrc_applied[world_idx, spindle_mj]``.

    Single-threaded (dim = 1).  mujoco_warp's xfrc layout is force first:
    [0:3] = force, [3:6] = torque.  ``torque_alpha`` defaults to 0 because the
    explicit, one-substep-lagged torque path into the axle/kingpin hinges is
    unstable at vehicle loads (measured 2026-09-09); wheel spin then follows the
    axle actuator kinematically.
    """
    w = staging[0]
    f_zu = wp.vec3(w[5], w[3], w[4] + tare_fz)
    tau_zu = wp.vec3(w[2], w[0], w[1])

    cur = xfrc_applied[world_idx, spindle_mj]
    xfrc_applied[world_idx, spindle_mj] = wp.spatial_vector(
        cur[0] + f_zu[0],
        cur[1] + f_zu[1],
        cur[2] + f_zu[2],
        cur[3] + torque_alpha * tau_zu[0],
        cur[4] + torque_alpha * tau_zu[1],
        cur[5] + torque_alpha * tau_zu[2],
    )


class SolverANCFShellRigid(SolverANCFShell):
    """ANCF shell solver with per-wheel coupling to rigid spindle bodies.

    Typical usage::

        ancf = SolverANCFShellRigid(model, ancf_model, n_tires=4, ...)
        for w in range(4):
            ancf.setup_wheel(w, spindle_mj=spindle_ids[w],
                             bead_idx_np=bead_np, tare_fz=tire_weight)
        ancf.capture_graph(dt)

        # Inside substep loop:
        ancf.graph_step()
        ancf.accumulate_wheel_wrenches(xfrc_applied, xpos)
        mj_solver.step_dynamics(state)
    """

    def __init__(self, model, ancf_model, n_tires: int | None = None, torque_alpha: float = 0.0, **kwargs):
        """Initialise the per-wheel coupling data.

        Args:
            model: Newton :class:`~newton.Model` (provides device and gravity).
            ancf_model: Mesh and material data shared by all tyres.
            n_tires: Number of tyres (= number of ANCF environments).  If None,
                falls back to ``n_envs`` in kwargs.
            torque_alpha: Scale on the tyre reaction torque fed back to the
                spindle.  0 (default) feeds back the force only.
            **kwargs: Forwarded to :class:`SolverANCFShell`.
        """
        if n_tires is None:
            n_tires = int(kwargs.pop("n_envs", 4))
        else:
            kwargs.pop("n_envs", None)
        super().__init__(model, ancf_model, n_envs=n_tires, **kwargs)
        self.n_tires = n_tires
        self.torque_alpha = float(torque_alpha)

        # Per-tire coupling data — populated by setup_wheel().
        self._spindle_mj_arr: list[int] = []
        self._world_idx_per_tire: list[int] = []
        self._lateral_offset_per_tire: list[float] = []
        self._bead_idx_per_tire: list[wp.array] = []
        self._xfrc_stg_per_tire: list[wp.array] = []
        self._tare_fz_per_tire: list[float] = []

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
        """Register one wheel's coupling data.  Call once per tyre before :meth:`capture_graph`.

        Args:
            tire_idx: 0-based wheel index (= ANCF env index).
            spindle_mj: MuJoCo body index of this wheel's spindle.
            bead_idx_np: GLOBAL flat bead node indices into the N*n_nodes array.
            tare_fz: Tyre weight tare [N].
            world_idx: MuJoCo world containing this spindle.  Defaults to
                ``tire_idx`` (parallel-world layout); pass 0 for a single world.
            lateral_offset: ANCF X offset added to the hub position [m]
                (parallel-world layout: ``tire_idx * lateral_spacing``).
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

    def accumulate_wheel_wrenches(self, xfrc_applied: wp.array2d, xpos: wp.array2d, device: str = "cuda:0") -> None:
        """Sum each tyre's external load and add it to ``xfrc_applied`` on its spindle.

        Call after the ANCF step and before MuJoCo ``step_dynamics``.  Does not
        zero ``xfrc_applied``; the caller clears it once per substep.

        Args:
            xfrc_applied: MuJoCo external force array, shape (n_worlds, n_bodies).
            xpos: MuJoCo body positions from the most recent ``step_kinematics``.
            device: CUDA device string.
        """
        n_nodes = self.ancf.n_nodes
        for w in range(self.n_tires):
            stg = self._xfrc_stg_per_tire[w]
            if stg is None:
                raise RuntimeError(f"Wheel {w} has no coupling data — call setup_wheel() first.")
            stg.zero_()
            wp.launch(
                _accum_external_wrench_wheel,
                dim=n_nodes,
                inputs=[
                    self.global_f_ext,
                    self.node_x,
                    xpos,
                    self._world_idx_per_tire[w],
                    self._spindle_mj_arr[w],
                    self._lateral_offset_per_tire[w],
                    w * n_nodes,
                    stg,
                ],
                device=device,
            )
            wp.launch(
                _staging_to_xfrc_wheel,
                dim=1,
                inputs=[
                    stg,
                    xfrc_applied,
                    self._world_idx_per_tire[w],
                    self._spindle_mj_arr[w],
                    self._tare_fz_per_tire[w],
                    self.torque_alpha,
                ],
                device=device,
            )
