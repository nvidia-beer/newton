# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""SolverANCFShellRigid — ANCF shell solver with per-wheel coupling to rigid spindles.

Extends :class:`SolverANCFShell` (``n_envs = n_tires``) with per-wheel staging
of the tyre's momentum-balanced reaction and its transfer to MuJoCo's ``xfrc_applied``.

Coordinate conventions (unchanged from the rest of the ANCF stack)::

    MuJoCo Z-up : x_fwd, y_lat, z_up
    ANCF Y-up   : x_lat, y_up,  z_fwd   (axle along X, tread at Y=0)
    Z-up -> Y-up : (x, y, z) -> (y, z, x)
    Y-up -> Z-up : (x, y, z) -> (z, x, y)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .solver_ancf_shell import _HHT_ALPHA, _HHT_GAMMA, SolverANCFShell

# 16 spatial vectors = 384 bytes: each wheel starts on a 128-byte boundary.
_WRENCH_STRIDE = wp.constant(16)


@wp.kernel
def _accum_external_wrench_wheel(
    global_f_ext: wp.array[float],
    node_x: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_xdd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
    lumped_mass: wp.array[float],
    xpos: wp.array2d[wp.vec3],
    world_idx: int,
    spindle_mj: int,
    lateral_offset: float,
    node_base: int,  # e * n_nodes — first global node index of this tire
    staging: wp.array[wp.spatial_vector],  # shape (1,)
):
    """Net tire-to-hub wrench from linear and angular momentum balance.

    Translational reaction is Σ(f_ext - M a). Director generalized forces
    contribute D × (f_D - M_D Ddd) to the angular reaction, in addition to
    r × (f_x - M_x xdd). Internal objective shell forces cancel in this sum.
    Staging order is (torque, force), in the ANCF Y-up frame.
    """
    i = wp.tid()
    global_idx = node_base + i
    base = global_idx * 6

    f = wp.vec3(global_f_ext[base], global_f_ext[base + 1], global_f_ext[base + 2])
    f -= wp.cw_mul(wp.vec3(lumped_mass[base], lumped_mass[base + 1], lumped_mass[base + 2]), node_xdd[global_idx])
    fd = wp.vec3(global_f_ext[base + 3], global_f_ext[base + 4], global_f_ext[base + 5])
    fd -= wp.cw_mul(wp.vec3(lumped_mass[base + 3], lumped_mass[base + 4], lumped_mass[base + 5]), node_Ddd[global_idx])

    pos_zu = xpos[world_idx, spindle_mj]
    hub_ancf = wp.vec3(pos_zu[1] + lateral_offset, pos_zu[2], pos_zu[0])
    r = node_x[global_idx] - hub_ancf
    tau = wp.cross(r, f) + wp.cross(node_D[global_idx], fd)

    wp.atomic_add(staging, 0, wp.spatial_vector(tau[0], tau[1], tau[2], f[0], f[1], f[2]))


@wp.kernel
def _coupling_force_average(
    current: wp.array[float],
    previous: wp.array[float],
    older: wp.array[float],
    weights: wp.vec3,
    bdf: bool,
    valid: wp.array[int],
    average: wp.array[float],
):
    i = wp.tid()
    w = weights
    if bdf:
        w = wp.vec3(1.0, 0.0, 0.0)
        if valid[0] != 0:
            w = wp.vec3(2.0 / 3.0, 1.0 / 3.0, 0.0)
    average[i] = w[0] * current[i] + w[1] * previous[i] + w[2] * older[i]


@wp.func
def _node_impulse_wrench(
    forces: wp.array[float],
    x: wp.array[wp.vec3],
    d: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    dv: wp.array[wp.vec3],
    old_x: wp.array[wp.vec3],
    old_d: wp.array[wp.vec3],
    old_v: wp.array[wp.vec3],
    old_dv: wp.array[wp.vec3],
    mass: wp.array[float],
    hub: wp.vec3,
    i: int,
    dt: float,
) -> wp.spatial_vector:
    b = 6 * i
    mx = wp.vec3(mass[b], mass[b + 1], mass[b + 2])
    md = wp.vec3(mass[b + 3], mass[b + 4], mass[b + 5])
    external = wp.vec3(forces[b], forces[b + 1], forces[b + 2])
    external_d = wp.vec3(forces[b + 3], forces[b + 4], forces[b + 5])

    momentum = wp.cw_mul(mx, v[i])
    delta_momentum = wp.cw_mul(mx, v[i] - old_v[i])
    # Expand the momentum difference before dividing by dt. Subtracting two
    # nearly equal angular momenta erases small accelerations during rolling.
    delta_angular = (
        wp.cross(x[i] - old_x[i], momentum)
        + wp.cross(old_x[i] - hub, delta_momentum)
        + wp.cross(d[i] - old_d[i], wp.cw_mul(md, dv[i]))
        + wp.cross(old_d[i], wp.cw_mul(md, dv[i] - old_dv[i]))
    )
    force = external - delta_momentum / dt
    torque = wp.cross(x[i] - hub, external) + wp.cross(d[i], external_d) - delta_angular / dt
    return wp.spatial_vector(torque[0], torque[1], torque[2], force[0], force[1], force[2])


@wp.kernel
def _accum_impulse_wheel(
    forces: wp.array[float],
    x: wp.array[wp.vec3],
    d: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    dv: wp.array[wp.vec3],
    old_x: wp.array[wp.vec3],
    old_d: wp.array[wp.vec3],
    old_v: wp.array[wp.vec3],
    old_dv: wp.array[wp.vec3],
    mass: wp.array[float],
    xpos: wp.array2d[wp.vec3],
    world: int,
    spindle: int,
    lateral_offset: float,
    node_base: int,
    dt: float,
    staging: wp.array[wp.spatial_vector],
):
    i = node_base + wp.tid()
    pos = xpos[world, spindle]
    hub = wp.vec3(pos[1] + lateral_offset, pos[2], pos[0])
    wrench = _node_impulse_wrench(forces, x, d, v, dv, old_x, old_d, old_v, old_dv, mass, hub, i, dt)
    wp.atomic_add(staging, 0, wrench)


@wp.kernel(enable_backward=False)
def _accum_wheel_impulses(
    forces: wp.array[float],
    x: wp.array[wp.vec3],
    d: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    dv: wp.array[wp.vec3],
    old_x: wp.array[wp.vec3],
    old_d: wp.array[wp.vec3],
    old_v: wp.array[wp.vec3],
    old_dv: wp.array[wp.vec3],
    mass: wp.array[float],
    xpos: wp.array2d[wp.vec3],
    maps: wp.array[wp.vec2i],
    offsets: wp.array[wp.vec2],
    n: int,
    dt: float,
    staging: wp.array[wp.spatial_vector],
):
    # A padded accumulator keeps separate wheels on separate cache lines.
    wheel, lane = wp.tid()
    if lane < n:
        world, spindle = maps[wheel][0], maps[wheel][1]
        p = xpos[world, spindle]
        hub = wp.vec3(p[1] + offsets[wheel][0], p[2], p[0])
        w = _node_impulse_wrench(forces, x, d, v, dv, old_x, old_d, old_v, old_dv, mass, hub, wheel * n + lane, dt)
        wp.atomic_add(staging, _WRENCH_STRIDE * wheel, w)


@wp.kernel(enable_backward=False)
def _transfer_wheel_wrenches(
    staging: wp.array[wp.spatial_vector],
    maps: wp.array[wp.vec2i],
    offsets: wp.array[wp.vec2],
    torque_alpha: float,
    xfrc: wp.array2d[wp.spatial_vector],
):
    wheel = wp.tid()
    w = staging[_WRENCH_STRIDE * wheel]
    wp.atomic_add(
        xfrc,
        maps[wheel][0],
        maps[wheel][1],
        wp.spatial_vector(
            w[5], w[3], w[4] + offsets[wheel][1], torque_alpha * w[2], torque_alpha * w[0], torque_alpha * w[1]
        ),
    )


def _allocate_coupling_history(solver):
    solver._coupling_dt = 0.0
    solver._coupling_impulse_rate = wp.zeros_like(solver.global_f_ext)
    solver._coupling_bdf = not hasattr(solver, "global_f_ext0")
    solver._coupling_bdf_valid = wp.zeros(1, dtype=int, device=solver.device)
    solver._coupling_external_previous = [wp.clone(solver.global_f_ext), wp.zeros_like(solver.global_f_ext)]
    if hasattr(solver, "global_f_ext0"):
        # Integrating the HHT equilibrium with Newmark's velocity formula
        # weights external forces at n+1, n, n-1 by .56, .38, .06.
        a, g = _HHT_ALPHA, _HHT_GAMMA
        solver._coupling_force_weights = wp.vec3(g * (1 + a), (1 - g) * (1 + a) - g * a, -(1 - g) * a)
    else:
        solver._coupling_force_weights = wp.vec3(1.0, 0.0, 0.0)

    solver._coupling_previous = [wp.clone(a) for a in (solver.node_x, solver.node_D, solver.node_xd, solver.node_Dd)]


def _begin_coupling_step(solver, dt: float):
    """Save time-n tire momentum and force history before prescribing bead motion.

    Args:
        solver: Coupled shell solver.
        dt: Substep duration [s]. Call again after every interface rewind.
    """
    solver._coupling_dt = dt
    if solver._coupling_bdf:
        wp.copy(solver._coupling_external_previous[0], solver._coupling_impulse_rate)
        wp.copy(solver._coupling_bdf_valid, solver.hist_valid)
    else:
        wp.copy(solver._coupling_external_previous[0], solver.global_f_ext)
    if hasattr(solver, "global_f_ext0"):
        wp.copy(solver._coupling_external_previous[1], solver.global_f_ext0)
    for old, current in zip(
        solver._coupling_previous, (solver.node_x, solver.node_D, solver.node_xd, solver.node_Dd), strict=True
    ):
        wp.copy(old, current)


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
    [0:3] = force, [3:6] = torque. Full torque transfer requires a converged
    two-way interface iteration when tire inertia is large relative to the rim.
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
                             bead_idx_np=bead_np, tare_fz=0.0)
        ancf.capture_graph(dt)

        # Inside each interface trial (see InterfaceCouplerGS):
        ancf.begin_coupling_step(dt)
        prescribe_bead_motion()
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
        _allocate_coupling_history(self)

        # Per-tire coupling data — populated by setup_wheel().
        self._spindle_mj_arr: list[int] = []
        self._world_idx_per_tire: list[int] = []
        self._lateral_offset_per_tire: list[float] = []
        self._bead_idx_per_tire: list[wp.array[wp.int32] | None] = []
        self._xfrc_stg_per_tire: list[wp.array[wp.spatial_vector] | None] = []
        self._tare_fz_per_tire: list[float] = []
        self._wrench_staging = wp.zeros(self.n_tires * _WRENCH_STRIDE, dtype=wp.spatial_vector, device=self.device)
        self._wrench_bindings = None

    begin_coupling_step = _begin_coupling_step

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
            tare_fz: Optional upward bias [N]. Use zero when the rigid asset
                excludes FEM tire mass; adding its weight would cancel shell gravity.
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
        self._xfrc_stg_per_tire[tire_idx] = self._wrench_staging[
            tire_idx * _WRENCH_STRIDE : tire_idx * _WRENCH_STRIDE + 1
        ]
        self._tare_fz_per_tire[tire_idx] = float(tare_fz)
        if len(self._spindle_mj_arr) == self.n_tires and all(i >= 0 for i in self._spindle_mj_arr):
            self._wrench_bindings = (
                wp.array(
                    list(zip(self._world_idx_per_tire, self._spindle_mj_arr, strict=True)),
                    dtype=wp.vec2i,
                    device=device,
                ),
                wp.array(
                    list(zip(self._lateral_offset_per_tire, self._tare_fz_per_tire, strict=True)),
                    dtype=wp.vec2,
                    device=device,
                ),
            )

    def accumulate_wheel_wrenches(
        self, xfrc_applied: wp.array2d[wp.spatial_vector], xpos: wp.array2d[wp.vec3], device: str = "cuda:0"
    ) -> None:
        """Sum each tyre's momentum-balanced reaction and add it to ``xfrc_applied`` on its spindle.

        Call after the ANCF step and before MuJoCo ``step_dynamics``.  Does not
        zero ``xfrc_applied``; the caller clears it once per substep.

        Args:
            xfrc_applied: MuJoCo external force array, shape (n_worlds, n_bodies).
            xpos: MuJoCo body positions from the most recent ``step_kinematics``.
            device: CUDA device string.
        """
        n_nodes = self.ancf.n_nodes
        if getattr(self, "_coupling_dt", 0.0) > 0.0:
            wp.launch(
                _coupling_force_average,
                dim=self.global_f_ext.shape[0],
                inputs=[
                    self.global_f_ext,
                    *self._coupling_external_previous,
                    self._coupling_force_weights,
                    self._coupling_bdf,
                    self._coupling_bdf_valid,
                    self._coupling_impulse_rate,
                ],
                device=device,
            )
        if (
            getattr(self, "_wrench_bindings", None) is not None
            and self._coupling_dt > 0.0
            and wp.get_device(device).is_cuda
        ):
            self._wrench_staging.zero_()
            wp.launch(
                _accum_wheel_impulses,
                dim=(self.n_tires, ((n_nodes + 255) // 256) * 256),
                block_dim=256,
                inputs=[
                    self._coupling_impulse_rate,
                    self.node_x,
                    self.node_D,
                    self.node_xd,
                    self.node_Dd,
                    *self._coupling_previous,
                    self.lumped_mass_tiled,
                    xpos,
                    *self._wrench_bindings,
                    n_nodes,
                    self._coupling_dt,
                    self._wrench_staging,
                ],
                device=device,
            )
            wp.launch(
                _transfer_wheel_wrenches,
                dim=self.n_tires,
                inputs=[self._wrench_staging, *self._wrench_bindings, self.torque_alpha, xfrc_applied],
                device=device,
            )
            return
        for w in range(self.n_tires):
            stg = self._xfrc_stg_per_tire[w]
            if stg is None:
                raise RuntimeError(f"Wheel {w} has no coupling data — call setup_wheel() first.")
            stg.zero_()
            if getattr(self, "_coupling_dt", 0.0) > 0.0:
                wp.launch(
                    _accum_impulse_wheel,
                    dim=n_nodes,
                    inputs=[
                        self._coupling_impulse_rate,
                        self.node_x,
                        self.node_D,
                        self.node_xd,
                        self.node_Dd,
                        *self._coupling_previous,
                        self.lumped_mass_tiled,
                        xpos,
                        self._world_idx_per_tire[w],
                        self._spindle_mj_arr[w],
                        self._lateral_offset_per_tire[w],
                        w * n_nodes,
                        self._coupling_dt,
                        stg,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    _accum_external_wrench_wheel,
                    dim=n_nodes,
                    inputs=[
                        self.global_f_ext,
                        self.node_x,
                        self.node_D,
                        self.node_xdd,
                        self.node_Ddd,
                        self.lumped_mass_tiled,
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
