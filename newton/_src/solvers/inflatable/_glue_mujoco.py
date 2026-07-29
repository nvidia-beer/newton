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

"""MuJoCo split-step coupling for the inflatable glue system.

Defines :class:`GlueMuJoCo`, a subclass of :class:`~.glue.Glue` that reads
body pose from ``mjw_data`` arrays (``xpos`` / ``xquat`` / ``cvel``) and
writes the elastic reaction into ``xfrc_applied``.

MuJoCo-specific array conventions handled here:

* Body state is 2D ``[nworld, nbody]`` instead of 1D ``[nbody]``.
* ``cvel`` is angular-first ``[ang, lin]`` (opposite to Newton).
* ``xquat`` is WXYZ ``[w, x, y, z]`` (opposite to Warp's XYZW).
* ``xfrc_applied`` is a 2D ``spatial_vector`` array; writes use per-component
  indexing to avoid the CUDA error 700 that whole-vector 2D writes trigger.
"""

import warp as wp

from .glue import Glue

# ===========================================================================
# MuJoCo kernels
# ===========================================================================


@wp.kernel
def _zero_xfrc_kernel(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    body_id: int,
):
    """Zero xfrc_applied[0, body_id] using per-component writes (world_idx=0, dim=1).

    Per-component writes avoid the CUDA error 700 that whole-vector 2D
    ``spatial_vector`` writes trigger in this Warp build.
    """
    xfrc_applied[0, body_id][0] = float(0.0)
    xfrc_applied[0, body_id][1] = float(0.0)
    xfrc_applied[0, body_id][2] = float(0.0)
    xfrc_applied[0, body_id][3] = float(0.0)
    xfrc_applied[0, body_id][4] = float(0.0)
    xfrc_applied[0, body_id][5] = float(0.0)


@wp.kernel
def _update_target_dv_mujoco_kernel(
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    cvel: wp.array2d[wp.spatial_vector],
    world_idx: int,
    mj_body_idx: int,
    local_offsets: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    glue_indices: wp.array[wp.int32],
    inv_dt: float,
    max_corr_vel: float,
    glue_damping: float,
    target_dv: wp.array[wp.vec3],
):
    """Glue target velocity using MuJoCo's post-kinematics predicted body pose.

    ``step_kinematics`` pre-advances the body to its predicted next-step position,
    so no additional kinematic prediction is needed here — implicit damping comes
    for free (equivalent to the Featherstone ``body_p_pred = body_p + v_com*dt`` path).

    MuJoCo ``cvel`` layout: ``[ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]`` —
    angular first, opposite to Newton/Featherstone.

    ``xquat`` is WXYZ ``[w,x,y,z]``; Warp's ``wp.quat_rotate`` expects XYZW ``[x,y,z,w]``.
    """
    tid = wp.tid()
    pidx = glue_indices[tid]

    body_p = xpos[world_idx, mj_body_idx]
    # WXYZ → XYZW conversion.
    q_mj = xquat[world_idx, mj_body_idx]
    body_rot = wp.quat(q_mj[1], q_mj[2], q_mj[3], q_mj[0])
    # Angular first in MuJoCo cvel.
    sv = cvel[world_idx, mj_body_idx]
    omega = wp.vec3(sv[0], sv[1], sv[2])
    v_com = wp.vec3(sv[3], sv[4], sv[5])

    x_target = body_p + wp.quat_rotate(body_rot, local_offsets[tid])
    v_target = v_com + wp.cross(omega, x_target - body_p)

    pos_corr = (x_target - particle_q[pidx]) * inv_dt
    mag = wp.length(pos_corr)
    if mag > max_corr_vel:
        pos_corr = pos_corr * (max_corr_vel / mag)

    target_dv[pidx] = glue_damping * (v_target - particle_qd[pidx]) + pos_corr


@wp.kernel
def _accumulate_reaction_mujoco_kernel(
    reaction: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    glue_indices: wp.array[wp.int32],
    world_idx: int,
    mj_body_idx: int,
    xpos: wp.array2d[wp.vec3],
    coupling_gain: float,
    staging: wp.array[wp.spatial_vector],
):
    """Accumulate glue reaction force + torque into a 1-element staging buffer.

    τ = r × F where r = particle_pos − body_origin (``xpos``).
    Uses a 1D staging buffer because ``wp.atomic_add`` on 2D ``spatial_vector``
    arrays triggers CUDA error 700.  Caller must zero the buffer before launch
    and commit with :func:`_write_staging_to_xfrc_kernel` afterwards.
    """
    tid = wp.tid()
    pidx = glue_indices[tid]
    f = reaction[pidx] * coupling_gain
    r = particle_q[pidx] - xpos[world_idx, mj_body_idx]
    tau = wp.cross(r, f)
    wp.atomic_add(staging, 0, wp.spatial_vector(f[0], f[1], f[2], tau[0], tau[1], tau[2]))


@wp.kernel
def _write_staging_to_xfrc_kernel(
    staging: wp.array[wp.spatial_vector],
    xfrc_applied: wp.array2d[wp.spatial_vector],
    world_idx: int,
    mj_body_idx: int,
):
    """Copy staging[0] into xfrc_applied[world, body], one component per thread (dim=6).

    Per-component writes match the mujoco_warp ``reset_xfrc_applied`` pattern
    and avoid the CUDA error 700 that whole-vector 2D writes produce.
    """
    tid = wp.tid()  # 0..5
    xfrc_applied[world_idx, mj_body_idx][tid] = staging[0][tid]


@wp.kernel
def _add_xfrc_fz_kernel(
    xfrc_applied: wp.array2d[wp.spatial_vector],
    world_idx: int,
    mj_body_idx: int,
    fz: float,
):
    """Add a pure Z-force to xfrc_applied[world, body] (dim=1)."""
    xfrc_applied[world_idx, mj_body_idx][2] = xfrc_applied[world_idx, mj_body_idx][2] + fz


@wp.kernel
def _rigid_velocity_damp_mujoco_kernel(
    cvel: wp.array2d[wp.spatial_vector],
    world_idx: int,
    mj_body_idx: int,
    kd: float,
    xfrc_applied: wp.array2d[wp.spatial_vector],
):
    """Dashpot-to-ground via xfrc_applied: F = −kd · v_rigid (dim=1).

    MuJoCo cvel: ``[ang, lin]`` — linear velocity at indices 3–5.
    Per-component writes to avoid the whole-vector 2D write CUDA error.
    """
    vx = cvel[world_idx, mj_body_idx][3]
    vy = cvel[world_idx, mj_body_idx][4]
    vz = cvel[world_idx, mj_body_idx][5]
    xfrc_applied[world_idx, mj_body_idx][0] = xfrc_applied[world_idx, mj_body_idx][0] - kd * vx
    xfrc_applied[world_idx, mj_body_idx][1] = xfrc_applied[world_idx, mj_body_idx][1] - kd * vy
    xfrc_applied[world_idx, mj_body_idx][2] = xfrc_applied[world_idx, mj_body_idx][2] - kd * vz


# ===========================================================================
# GlueMuJoCo
# ===========================================================================


class GlueMuJoCo(Glue):
    """Bilateral coupling between FEM soft particles and a MuJoCo rigid body.

    Subclasses :class:`~.glue.Glue` — shares all construction, ``toggle``,
    ``get_pin_positions``, ``get_target_positions``, and ``n_glue``.  Overrides
    :meth:`update_target_dv` and :meth:`apply_reaction` with MuJoCo-specific
    implementations.

    Args:
        solver: :class:`~newton.solvers.SolverImplicitSoft` (or subclass).
        glue_indices: Indices of glued particles, shape ``[n_glue]``, ``int32``.
        local_offsets: Body-frame offsets from rigid COM to each glued particle,
            shape ``[n_glue]``, ``vec3`` [m].
        coupling_gain: Scale factor applied to the elastic reaction (dimensionless).
        device: Warp device string (e.g. ``"cuda:0"``).
        substep_dt: FEM substep size [s].
        gravity_comp_fz: Constant upward force [N] added to ``xfrc_applied``.
        kd_rigid: Velocity damping coefficient [N·s/m].
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Staging buffer: wp.atomic_add on 2D spatial_vector arrays triggers
        # CUDA error 700; accumulate here then commit with _write_staging_to_xfrc_kernel.
        self._xfrc_staging = wp.zeros(1, dtype=wp.spatial_vector, device=self._device)

    # ------------------------------------------------------------------
    # Per-substep: before the soft solver step
    # ------------------------------------------------------------------

    def update_target_dv(
        self,
        xpos,
        xquat,
        cvel,
        world_idx: int,
        mj_body_idx: int,
        particle_q,
        particle_qd,
        inv_dt: float,
    ) -> None:
        """Compute pin target velocities from MuJoCo post-kinematics predicted pose.

        Call each substep **after** :meth:`~.SolverMuJoCo.step_kinematics` and
        **before** the soft solver step.

        Args:
            xpos: ``[nworld, nbody]`` body frame origin positions [m], ``vec3``.
            xquat: ``[nworld, nbody]`` body frame orientations, ``quat`` (WXYZ).
            cvel: ``[nworld, nbody]`` body velocities (angular first), ``spatial_vector``.
            world_idx: World index (``0`` for single-world simulations).
            mj_body_idx: MuJoCo body index of the rigid body.
            particle_q: Current particle positions [m], shape ``[particle_count]``.
            particle_qd: Current particle velocities [m/s], shape ``[particle_count]``.
            inv_dt: Reciprocal substep size [1/s].
        """
        wp.launch(
            _update_target_dv_mujoco_kernel,
            dim=self._n_glue,
            inputs=[
                xpos,
                xquat,
                cvel,
                world_idx,
                mj_body_idx,
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
        xfrc_applied,
        world_idx: int,
        mj_body_idx: int,
        cvel=None,
        xpos=None,
        particle_q=None,
    ) -> None:
        """Forward elastic reaction force + torque into MuJoCo ``xfrc_applied``.

        Caller must zero ``xfrc_applied`` before this call.  Call each substep
        **after** the soft solver step and **before**
        :meth:`~.SolverMuJoCo.step_dynamics`.

        Args:
            xfrc_applied: ``[nworld, nbody]`` external wrench array [N, N·m].
            world_idx: World index.
            mj_body_idx: MuJoCo body index of the rigid body.
            cvel: Required when :attr:`kd_rigid` is non-zero.
            xpos: Required for torque coupling (τ = r × F).
            particle_q: Required for torque coupling.
        """
        if xpos is not None and particle_q is not None:
            self._xfrc_staging.zero_()
            wp.launch(
                _accumulate_reaction_mujoco_kernel,
                dim=self._n_glue,
                inputs=[
                    self._reaction,
                    particle_q,
                    self._glue_indices,
                    world_idx,
                    mj_body_idx,
                    xpos,
                    self._coupling_gain,
                    self._xfrc_staging,
                ],
                device=self._device,
            )
            wp.launch(
                _write_staging_to_xfrc_kernel,
                dim=6,
                inputs=[self._xfrc_staging, xfrc_applied, world_idx, mj_body_idx],
                device=self._device,
            )
        if self._gravity_comp_fz != 0.0:
            wp.launch(
                _add_xfrc_fz_kernel,
                dim=1,
                inputs=[xfrc_applied, world_idx, mj_body_idx, wp.float32(self._gravity_comp_fz)],
                device=self._device,
            )
        if self._kd_rigid != 0.0 and cvel is not None:
            wp.launch(
                _rigid_velocity_damp_mujoco_kernel,
                dim=1,
                inputs=[cvel, world_idx, mj_body_idx, wp.float32(self._kd_rigid), xfrc_applied],
                device=self._device,
            )
