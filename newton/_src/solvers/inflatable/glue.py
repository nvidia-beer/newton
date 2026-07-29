# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kinematic glue for the implicit soft-body solver.

:class:`SolverGlue`
    Solver-side kinematic glue.  Owns the glue arrays and BSR cache,
    runs the two per-substep hooks called from
    :meth:`~.SolverImplicitSoft.implicit_integration`:

    1. :meth:`~SolverGlue.apply_override` — capture the constraint reaction
       and overwrite ``particle_f[p] := target_dv[p]``.
    2. :meth:`~SolverGlue.apply_filter` — row/column-eliminate the glued
       DOFs in the BSR matrix and Schur-condense the prescribed motion into
       the unpinned RHS.

    (Sifakis SIGGRAPH 2012 §3 / Baraff–Witkin SIGGRAPH '98 §5)

:class:`Glue`
    Newton / Featherstone rigid-body coupling.  Creates a :class:`SolverGlue`,
    drives pinned particles to track a Newton rigid body each substep, and
    forwards the elastic reaction into ``State.body_f``.  Also serves as the
    base class for backend-specific subclasses.

:class:`GlueMuJoCo`
    MuJoCo split-step coupling (defined in :mod:`._glue_mujoco`).  Same shared
    infrastructure as :class:`Glue` but reads body pose from ``mjw_data`` arrays
    and writes the reaction into ``xfrc_applied``.
"""

import math

import numpy as np
import warp as wp

from newton._src.sim import Model, State

from .kernels import apply_dirichlet_pin_kernel, filter_dirichlet_pin_in_bsr_kernel

# ===========================================================================
# Kernels
# ===========================================================================


@wp.kernel
def _gather_positions_kernel(
    particle_q: wp.array[wp.vec3],
    glue_indices: wp.array[wp.int32],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = particle_q[glue_indices[i]]


@wp.kernel
def _gather_target_positions_kernel(
    body_q: wp.array[wp.transform],
    rigid_body: int,
    local_offsets: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    """Body-frame anchor positions in world space: body_p + R(body_rot) · local_offset."""
    i = wp.tid()
    body_p = wp.transform_get_translation(body_q[rigid_body])
    body_rot = wp.transform_get_rotation(body_q[rigid_body])
    out[i] = body_p + wp.quat_rotate(body_rot, local_offsets[i])


@wp.kernel
def _set_mask_kernel(
    glue_indices: wp.array[wp.int32],
    mask: wp.array[wp.int32],
):
    mask[glue_indices[wp.tid()]] = 1


@wp.kernel
def _update_target_dv_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    rigid_body: int,
    local_offsets: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    glue_indices: wp.array[wp.int32],
    inv_dt: float,
    max_corr_vel: float,
    glue_damping: float,
    target_dv: wp.array[wp.vec3],
):
    """Position-level bilateral constraint: target_dv drives each glue particle to x_target in one step.

    Equivalent to the KKT position-level equality constraint for backward-Euler integration::

        target_dv = glue_damping·(v_target − v_particle) + clamp(x_target − x_particle, max_corr_vel·dt) / dt

    Newton body_q is a ``wp.transform`` (COM position + Warp quaternion ``(x,y,z,w)``).
    Newton body_qd layout: ``[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]``.

    Kinematic prediction: the target is set at the body's *predicted* next-step position
    ``body_p + v_com*dt`` rather than the current position, making the elastic reaction
    velocity-proportional and providing implicit coupling damping.
    """
    tid = wp.tid()
    pidx = glue_indices[tid]

    body_p = wp.transform_get_translation(body_q[rigid_body])
    body_rot = wp.transform_get_rotation(body_q[rigid_body])

    # Newton body_qd: linear velocity first, angular second.
    sv = body_qd[rigid_body]
    v_com = wp.vec3(sv[0], sv[1], sv[2])
    omega = wp.vec3(sv[3], sv[4], sv[5])

    dt = 1.0 / inv_dt
    body_p_pred = body_p + v_com * dt

    x_target = body_p_pred + wp.quat_rotate(body_rot, local_offsets[tid])
    v_target = v_com + wp.cross(omega, x_target - body_p_pred)

    pos_corr = (x_target - particle_q[pidx]) * inv_dt
    mag = wp.length(pos_corr)
    if mag > max_corr_vel:
        pos_corr = pos_corr * (max_corr_vel / mag)

    target_dv[pidx] = glue_damping * (v_target - particle_qd[pidx]) + pos_corr


@wp.kernel
def _accumulate_reaction_kernel(
    reaction: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    glue_indices: wp.array[wp.int32],
    body_id: int,
    body_q: wp.array[wp.transform],
    coupling_gain: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Accumulate glue reaction force + torque into Newton body_f [N, N·m].

    Torque τ = r × F where r = particle_pos − body_com.
    """
    tid = wp.tid()
    pidx = glue_indices[tid]
    f = reaction[pidx] * coupling_gain
    body_com = wp.transform_get_translation(body_q[body_id])
    r = particle_q[pidx] - body_com
    tau = wp.cross(r, f)
    wp.atomic_add(body_f, body_id, wp.spatial_vector(f[0], f[1], f[2], tau[0], tau[1], tau[2]))


@wp.kernel
def _zero_body_f_kernel(
    body_f: wp.array[wp.spatial_vector],
    body_id: int,
):
    body_f[body_id] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _add_body_f_fz_kernel(
    body_f: wp.array[wp.spatial_vector],
    body_id: int,
    fz: float,
):
    """Add a pure Z-force to body_f (single-thread kernel, dim=1)."""
    sv = body_f[body_id]
    body_f[body_id] = wp.spatial_vector(sv[0], sv[1], sv[2] + fz, sv[3], sv[4], sv[5])


@wp.kernel
def _rigid_velocity_damp_kernel(
    body_qd: wp.array[wp.spatial_vector],
    body_id: int,
    kd: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Apply dashpot-to-ground: F = −kd · v_rigid, dim=1.

    Newton body_qd layout: ``[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]``.
    """
    sv = body_qd[body_id]
    xsv = body_f[body_id]
    body_f[body_id] = wp.spatial_vector(
        xsv[0] - kd * sv[0],
        xsv[1] - kd * sv[1],
        xsv[2] - kd * sv[2],
        xsv[3],
        xsv[4],
        xsv[5],
    )


# ===========================================================================
# SolverGlue — solver-side pin state
# ===========================================================================


class SolverGlue:
    """Solver-side kinematic pin state.

    Owns the pin mask, target velocity, reaction buffer, and the BSR
    pre-filter snapshot.  Call :meth:`apply_override` then
    :meth:`apply_filter` each substep from inside
    :meth:`~.SolverImplicitSoft.implicit_integration` (after force
    accumulation, before the linear solve).

    Attach to a solver by assigning::

        solver._solver_glue = SolverGlue(solver)

    :class:`Glue` does this automatically when constructed.

    Args:
        solver: :class:`~newton.solvers.SolverImplicitSoft` (or subclass).
        filter_A: Enable BSR row/col elimination (Schur condensation).
            Keep ``True`` unless debugging.
    """

    def __init__(self, solver, filter_A: bool = True) -> None:
        self._solver = solver
        self._mask: wp.array | None = None
        self._target_dv: wp.array | None = None
        self._reaction: wp.array | None = None
        self._gravity = wp.vec3(0.0, 0.0, 0.0)
        self._A_bsr_clean_values: wp.array | None = None
        self.filter_A = filter_A
        solver._solver_glue = self

    def set_pin(
        self,
        mask: "wp.array | None",
        target_dv: "wp.array | None",
        reaction: "wp.array | None",
    ) -> None:
        """Enable or disable the kinematic pin.

        Args:
            mask: Per-particle ``int32`` flag, ``1`` for pinned,
                shape ``[particle_count]``. Pass all three as ``None`` to disable.
            target_dv: Per-particle target velocity change [m/s],
                shape ``[particle_count]``, ``vec3``.
            reaction: Output buffer for the per-particle reaction force [N],
                shape ``[particle_count]``, ``vec3``.
        """
        if mask is None and target_dv is None and reaction is None:
            self._mask = None
            self._target_dv = None
            self._reaction = None
            # BSR matrix is built once and never rebuilt.  If apply_filter ever
            # modified it, restore the clean snapshot so the CG solves correctly
            # without the pin active.
            if self._A_bsr_clean_values is not None:
                wp.copy(dest=self._solver.A_bsr.values, src=self._A_bsr_clean_values)
            return
        if mask is None or target_dv is None or reaction is None:
            raise ValueError("SolverGlue.set_pin: pass all three as None to disable, or all three non-None to enable.")
        n = self._solver.model.particle_count
        for name, arr in (("mask", mask), ("target_dv", target_dv), ("reaction", reaction)):
            if arr.shape[0] != n:
                raise ValueError(f"SolverGlue.set_pin: {name} length {arr.shape[0]} != particle_count {n}")
        self._mask = mask
        self._target_dv = target_dv
        self._reaction = reaction
        self.refresh_gravity()

    def refresh_gravity(self) -> None:
        """Re-snapshot ``model.gravity`` for the override kernel.

        Triggers a device→host copy when ``model.gravity`` is a ``wp.array``,
        so call outside any captured CUDA-graph region.  Re-call after
        changing gravity at runtime.
        """
        g = self._solver.model.gravity
        if isinstance(g, wp.array):
            arr = g.numpy()[0]
            self._gravity = wp.vec3(float(arr[0]), float(arr[1]), float(arr[2]))
        elif g is None:
            self._gravity = wp.vec3(0.0, 0.0, 0.0)
        else:
            self._gravity = wp.vec3(float(g[0]), float(g[1]), float(g[2]))

    def apply_override(self, model: Model, state_in: State, dt: float) -> None:
        """Capture the pin reaction and override ``particle_f`` at pinned DOFs.

        For each pinned ``p``: writes the constraint reaction into ``reaction``,
        then overwrites ``particle_f[p] := target_dv[p]`` so that after
        :meth:`apply_filter` sets ``A[p,p] = I`` the linear solve yields
        ``Δv[p] = target_dv[p]`` exactly.  No-op when the pin is not set.
        """
        if self._mask is None:
            return
        target_scale = 1.0 if self.filter_A else self._solver.mass
        wp.launch(
            kernel=apply_dirichlet_pin_kernel,
            dim=model.particle_count,
            inputs=[
                self._mask,
                self._target_dv,
                self._gravity,
                wp.float32(self._solver.mass),
                wp.float32(target_scale),
                float(dt),
                state_in.particle_f,
                self._reaction,
            ],
            device=model.device,
        )

    def apply_filter(self, model: Model, state_in: State) -> None:
        """Row/col-eliminate pinned DOFs in ``A_bsr`` and Schur-condense the RHS.

        The pre-filter BSR snapshot is taken on the first call (warmup,
        outside any CUDA-graph capture); subsequent substeps restore it
        via ``wp.copy``, which is CUDA-graph capturable.  No-op when the
        pin is disabled or :attr:`filter_A` is ``False``.
        """
        if self._mask is None or not self.filter_A:
            return
        if self._A_bsr_clean_values is None:
            self._A_bsr_clean_values = wp.clone(self._solver.A_bsr.values)
        else:
            wp.copy(dest=self._solver.A_bsr.values, src=self._A_bsr_clean_values)
        wp.launch(
            kernel=filter_dirichlet_pin_in_bsr_kernel,
            dim=model.particle_count,
            inputs=[
                self._mask,
                self._target_dv,
                self._solver.A_bsr.offsets,
                self._solver.A_bsr.columns,
                self._solver.A_bsr.values,
                state_in.particle_f,
            ],
            device=model.device,
        )


# ===========================================================================
# Helpers
# ===========================================================================


def _compute_cfl_vel(model, dt: float) -> float:
    """Return the CFL-derived maximum position-correction speed [m/s].

    ``v_CFL = h_min / dt`` where ``h_min`` is the shortest tet edge.
    Computed once at construction — no free parameter.
    """
    if model.tet_indices is None or model.particle_q is None:
        return math.inf

    idx = model.tet_indices.numpy().reshape(-1, 4)
    pos = model.particle_q.numpy()
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    h_min = math.inf
    for i, j in edge_pairs:
        diff = pos[idx[:, i]] - pos[idx[:, j]]
        h_min = min(h_min, float(np.linalg.norm(diff, axis=1).min()))

    return h_min / dt


# ===========================================================================
# Glue — Newton / Featherstone coupling (also the shared base)
# ===========================================================================


class Glue:
    """Bilateral coupling between FEM soft particles and a Newton rigid body.

    Reads body pose from Newton ``State.body_q`` / ``body_qd`` and writes the
    elastic reaction into ``State.body_f``.  Also serves as the base class for
    :class:`GlueMuJoCo`.

    Each substep the target velocity for pinned particle ``p`` is set to:

    .. math::

        \\delta v_p = (v_{\\text{rigid}} - v_p)
                    + \\frac{x_{\\text{rigid,pred}} - x_p}{\\Delta t}

    This is the discrete KKT condition for a bilateral equality constraint
    :math:`x_p = x_{\\text{rigid}}` under Backward-Euler integration.

    Args:
        solver: :class:`~newton.solvers.SolverImplicitSoft` (or subclass).
        glue_indices: Indices of glued particles, shape ``[n_glue]``, ``int32``.
        local_offsets: Body-frame offsets from rigid COM to each glued particle,
            shape ``[n_glue]``, ``vec3`` [m].
        coupling_gain: Total reaction scale (dimensionless). Normalised internally
            by ``n_glue`` so the same value is stable regardless of mesh resolution
            or how many soft bodies share the hub.
        device: Warp device string (e.g. ``"cuda:0"``).
        substep_dt: FEM substep size [s].
        gravity_comp_fz: Constant upward force [N] added to ``body_f``.
        kd_rigid: Velocity damping coefficient [N·s/m].
        glue_damping: D-gain on the velocity-matching term (dimensionless, default 1.0).
            Values > 1 overdamp the constraint; 0 disables velocity tracking.
    """

    def __init__(
        self,
        solver,
        glue_indices: "wp.array",
        local_offsets: "wp.array",
        coupling_gain: float,
        device: str,
        substep_dt: float,
        gravity_comp_fz: float = 0.0,
        kd_rigid: float = 0.0,
        glue_damping: float = 1.0,
    ) -> None:
        n = solver.model.particle_count
        self._solver_glue = SolverGlue(solver)
        self._glue_indices = glue_indices
        self._local_offsets = local_offsets
        self._n_glue = int(glue_indices.shape[0])
        # Normalise by glue count so coupling_gain is "reaction per unit mean
        # violation" — independent of mesh resolution and number of fingers.
        self._coupling_gain = float(coupling_gain) / max(self._n_glue, 1)
        self._gravity_comp_fz = float(gravity_comp_fz)
        self._kd_rigid = float(kd_rigid)
        self._glue_damping = float(glue_damping)
        self._max_corr_vel = _compute_cfl_vel(solver.model, float(substep_dt))
        self._device = device

        self._mask_on = wp.zeros(n, dtype=wp.int32, device=device)
        wp.launch(
            kernel=_set_mask_kernel,
            dim=self._n_glue,
            inputs=[glue_indices, self._mask_on],
            device=device,
        )
        self._target_dv = wp.zeros(n, dtype=wp.vec3, device=device)
        self._reaction = wp.zeros(n, dtype=wp.vec3, device=device)
        self._solver_glue.set_pin(self._mask_on, self._target_dv, self._reaction)

    # ------------------------------------------------------------------

    def toggle(self, enabled: bool) -> None:
        """Enable or disable the pin (call before ``capture()``).

        Args:
            enabled: ``True`` to activate, ``False`` to release.
        """
        if enabled:
            self._solver_glue.set_pin(self._mask_on, self._target_dv, self._reaction)
        else:
            self._solver_glue.set_pin(None, None, None)
            self._target_dv.zero_()

    def get_pin_positions(self, particle_q: "wp.array") -> "wp.array":
        """Return a ``[n_glue]`` ``vec3`` array of current glued-particle positions.

        Args:
            particle_q: Current particle positions [m], shape ``[particle_count]``, ``vec3``.
        """
        out = wp.empty(self._n_glue, dtype=wp.vec3, device=self._device)
        wp.launch(
            _gather_positions_kernel,
            dim=self._n_glue,
            inputs=[particle_q, self._glue_indices, out],
            device=self._device,
        )
        return out

    def get_target_positions(self, body_q, body_id: int) -> "wp.array":
        """Return a ``[n_glue]`` ``vec3`` array of body-frame anchor positions in world space.

        Args:
            body_q: Newton body transforms, shape ``[body_count]``, ``wp.transform`` [m].
            body_id: Newton body index of the rigid body.
        """
        out = wp.empty(self._n_glue, dtype=wp.vec3, device=self._device)
        wp.launch(
            _gather_target_positions_kernel,
            dim=self._n_glue,
            inputs=[body_q, body_id, self._local_offsets, out],
            device=self._device,
        )
        return out

    @property
    def n_glue(self) -> int:
        """Number of glued particles."""
        return self._n_glue

    @property
    def n_pins(self) -> int:
        """Number of glued particles.

        .. deprecated::
            Use :attr:`n_glue` instead.
        """
        return self._n_glue

    # ------------------------------------------------------------------
    # Per-substep: before the soft solver step
    # ------------------------------------------------------------------

    def update_target_dv(
        self,
        body_q,
        body_qd,
        body_id: int,
        particle_q,
        particle_qd,
        inv_dt: float,
    ) -> None:
        """Compute position-level constraint target velocity for pinned particles.

        Call each substep **before** the soft solver step.

        Args:
            body_q: Newton body transforms, shape ``[body_count]``, ``wp.transform`` [m].
            body_qd: Newton body velocities ``[lin, ang]``,
                shape ``[body_count]``, ``spatial_vector`` [m/s, rad/s].
            body_id: Newton body index of the rigid body.
            particle_q: Current particle positions [m], ``vec3``.
            particle_qd: Current particle velocities [m/s], ``vec3``.
            inv_dt: Reciprocal substep size [1/s].
        """
        wp.launch(
            _update_target_dv_kernel,
            dim=self._n_glue,
            inputs=[
                body_q,
                body_qd,
                body_id,
                self._local_offsets,
                particle_q,
                particle_qd,
                self._glue_indices,
                float(inv_dt),
                self._max_corr_vel,
                self._glue_damping,
                self._target_dv,
            ],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Per-substep: after the soft solver step
    # ------------------------------------------------------------------

    def apply_reaction(
        self,
        body_f,
        body_id: int,
        body_qd=None,
        body_q=None,
        particle_q=None,
    ) -> None:
        """Forward elastic reaction force + torque into Newton ``body_f``.

        Zeroes ``body_f[body_id]`` first, then accumulates the pin reaction.
        Call each substep **after** the soft solver step.

        Args:
            body_f: Newton external force array, shape ``[body_count]``,
                ``spatial_vector`` [N, N·m].
            body_id: Newton body index of the rigid body.
            body_qd: Required when :attr:`kd_rigid` is non-zero.
            body_q: Required for torque coupling (τ = r × F).
            particle_q: Required for torque coupling.
        """
        wp.launch(_zero_body_f_kernel, dim=1, inputs=[body_f, body_id], device=self._device)
        wp.launch(
            _accumulate_reaction_kernel,
            dim=self._n_glue,
            inputs=[
                self._reaction,
                particle_q,
                self._glue_indices,
                body_id,
                body_q,
                self._coupling_gain,
                body_f,
            ],
            device=self._device,
        )
        if self._gravity_comp_fz != 0.0:
            wp.launch(
                _add_body_f_fz_kernel,
                dim=1,
                inputs=[body_f, body_id, wp.float32(self._gravity_comp_fz)],
                device=self._device,
            )
        if self._kd_rigid != 0.0 and body_qd is not None:
            wp.launch(
                _rigid_velocity_damp_kernel,
                dim=1,
                inputs=[body_qd, body_id, wp.float32(self._kd_rigid), body_f],
                device=self._device,
            )


# ---------------------------------------------------------------------------
# MuJoCo kernel re-exports
#
# Placed after the Glue class definition so that when _glue_mujoco imports
# Glue from this module (circular but safe at this point), the class is
# already bound in the partial module object.
# ---------------------------------------------------------------------------
from ._glue_mujoco import GlueMuJoCo, _zero_xfrc_kernel  # noqa: E402, F401
