# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint shell/interfacial Newton corrections with a reused condensed tangent.

The six spindle columns and the free-shell residual share the linearized
interior solve. Each correction advances both unknowns; a coupling trial does
not restart the HHT solve from time n. Physical tire momentum and full reaction
wrenches still determine the rigid response.
"""

import numpy as np
import warp as wp

from ...sim.articulation import compute_body_spatial_inertia, eval_jacobian, eval_mass_matrix
from . import kernels_coupling as cp
from . import schur as sh
from . import solver_ancf_shell as fem
from ._kinematic_cache import KinematicStateCache
from .kernels_coupled import (
    aa_gather,
    aa_products,
    aa_update,
    copy_rigid,
    make_inverse,
    project,
    rigid_mass,
)


@wp.kernel(enable_backward=False)
def force_rhs(rhs: wp.array[float], R: wp.array[float], free: wp.array[float], n: int):
    tire, row = wp.tid()
    rhs[(tire * 7 + 6) * n + row] = -R[tire * n + row] * free[tire * n + row]


@wp.kernel(enable_backward=False)
def check(
    guess: wp.array[float],
    raw: wp.array[float],
    inverse: wp.array2d[wp.float64],
    corrected: wp.array[float],
    res: wp.array[float],
    history: wp.array[float],
    active: wp.array[int],
    conv: wp.array[int],
    iters: wp.array[int],
    progress: wp.array[int],
    i: int,
    maximum: int,
    normal: int,
    atol: float,
    rtol: float,
):
    _, lane = wp.tid()
    norm = wp.float64(0.0)
    scaled = float(0.0)
    if lane < guess.shape[0]:
        r = raw[lane] - guess[lane]
        history[lane] = r
        norm = wp.float64(r * r)
        scaled = cp._scaled_error(r, guess[lane], raw[lane], atol, rtol)
        corr = wp.float64(0.0)
        for k in range(guess.shape[0]):
            corr += inverse[lane, k] * wp.float64(raw[k] - guess[k])
        corrected[lane] = guess[lane] + float(corr)
    total = wp.tile_sum(wp.tile(norm))
    largest = wp.tile_max(wp.tile(scaled))
    if lane == 0:
        res[i] = float(wp.sqrt(total[0]))
        iters[0] = i + 1
        conv[0] = int(i >= 2 and largest[0] <= 1.0)
        active[0] = int(i < maximum and conv[0] == 0 and (i < normal or total[0] > wp.float64(0.000625)))
        progress[0] = int(conv[0] != 0 or (i >= 2 and res[i] < 0.5 * res[0]))


@wp.kernel(enable_backward=False)
def choose_basis(
    totals: wp.array[int],
    ready: wp.array[int],
    converged: wp.array[int],
    i: int,
    refresh: wp.array[int],
):
    refresh[0] = int(i == 0 and (totals[0] % 10 == 0 or ready[0] == 0 or converged[0] == 0))


@wp.kernel(enable_backward=False)
def insert_free(dx: wp.array[float], X: wp.array[float], n: int):
    tire, row = wp.tid()
    X[(tire * 7 + 6) * n + row] = dx[tire * n + row]


@wp.kernel(enable_backward=False)
def reduced_matrix(
    M: wp.array3d[float],
    arm: wp.array[float],
    Z: wp.array3d[wp.float64],
    S: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    v = wp.float64(M[0, i, j])
    if i == j:
        v += wp.float64(arm[i])
    for t in range(Z.shape[0]):
        v += Z[t, i, j]
    S[i, j] = v


@wp.kernel(enable_backward=False)
def response_inverse(
    M: wp.array3d[float],
    arm: wp.array[float],
    Sinv: wp.array2d[wp.float64],
    H: wp.array2d[wp.float64],
):
    i, j = wp.tid()
    val = wp.float64(0.0)
    for k in range(Sinv.shape[0]):
        m = wp.float64(M[0, k, j])
        if k == j:
            m += wp.float64(arm[k])
        val += Sinv[i, k] * m
    H[i, j] = val


@wp.kernel(enable_backward=False)
def reduced_rhs(
    M: wp.array3d[float],
    arm: wp.array[float],
    local: wp.array3d[wp.float64],
    J: wp.array3d[float],
    rows: wp.array[int],
    raw: wp.array[float],
    guess: wp.array[float],
    rhs: wp.array[wp.float64],
):
    i = wp.tid()
    value = wp.float64(0.0)
    for j in range(rhs.shape[0]):
        m = wp.float64(M[0, i, j])
        if i == j:
            m += wp.float64(arm[i])
        value += m * wp.float64(raw[j] - guess[j])
    for t in range(local.shape[0]):
        for mode in range(6):
            value -= wp.float64(J[0, rows[t] * 6 + mode, i]) * local[t, mode, 6]
    rhs[i] = value


@wp.kernel(enable_backward=False)
def reduced_update(Sinv: wp.array2d[wp.float64], rhs: wp.array[wp.float64], delta: wp.array[float]):
    i = wp.tid()
    v = wp.float64(0.0)
    for j in range(rhs.shape[0]):
        v += Sinv[i, j] * rhs[j]
    delta[i] = float(v)


@wp.kernel(enable_backward=False)
def update_acc(
    X: wp.array[float],
    J: wp.array3d[float],
    rows: wp.array[int],
    delta: wp.array[float],
    free: wp.array[float],
    acc: wp.array[wp.vec3],
    dacc: wp.array[wp.vec3],
    n: int,
    scale: float,
):
    tire, node = wp.tid()
    i = tire * n + node
    if free[i * 6] != 0.0:
        dx = wp.vec3(0.0)
        dd = wp.vec3(0.0)
        for j in range(3):
            dx[j] = X[(tire * 7 + 6) * n * 6 + node * 6 + j]
            dd[j] = X[(tire * 7 + 6) * n * 6 + node * 6 + 3 + j]
        for mode in range(6):
            dv = float(0.0)
            for col in range(delta.shape[0]):
                dv += J[0, rows[tire] * 6 + mode, col] * delta[col]
            for j in range(3):
                dx[j] += X[(tire * 7 + mode) * n * 6 + node * 6 + j] * dv
                dd[j] += X[(tire * 7 + mode) * n * 6 + node * 6 + 3 + j] * dv
        acc[i] += scale * dx
        dacc[i] += scale * dd


@wp.kernel(enable_backward=False)
def update_guess(guess: wp.array[float], delta: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    guess[i] += delta[i]
    out[i] = guess[i]


class CoupledShellNewton:
    """Retain shell Newton iterates while correcting the rigid interface.

    Six cached columns describe spindle translation/rotation; the seventh
    solves the current free-shell residual. A mass-weighted Anderson step
    mixes both corrections together. The second half of the evaluation budget is reserved
    for steps whose interface residual remains large after the normal budget.
    """

    def __init__(self, coupler, state_0, kinematic_arrays, linear_iterations=4):
        self.coupler = coupler
        self.supported = True
        # The block-local SGS path tolerates a smaller inner budget. Larger
        # distributed systems retain the configured budget.
        if coupler._ancf.pcg._sweep_kernel is None:
            linear_iterations = coupler._ancf.pcg.max_iters
        self.linear_iterations = min(linear_iterations, coupler._ancf.pcg.max_iters)
        self.rigid_response = self.coupler._shell_response
        assert self.rigid_response is not None
        self.shell = self.coupler._ancf
        self.device = self.shell.device
        self.model = self.coupler._rigid_model
        self.reference = self.model.state()
        wp.copy(self.reference.body_q, state_0.body_q)
        wp.copy(self.reference.joint_q, state_0.joint_q)
        eval_jacobian(
            self.model,
            self.reference,
            J=self.rigid_response.jacobian,
            joint_S_s=self.rigid_response._motion_subspace,
        )
        eval_mass_matrix(
            self.model,
            self.reference,
            H=self.rigid_response.mass,
            J=self.rigid_response.jacobian,
            body_I_s=self.rigid_response._spatial_inertia,
        )
        mass = self.rigid_response.mass.numpy()[0] + np.diag(self.model.joint_armature.numpy())
        if not np.isfinite(mass).all() or np.linalg.eigvalsh(mass).min() <= 1e-09 or self.model.joint_dof_count > 32:
            self.supported = False
            return
        self.shell.set_coupled_launch_config()
        self.response = sh.ShellSchurResponse(
            self.shell.blk_offsets.numpy(),
            self.shell.blk_columns.numpy(),
            self.shell.n_envs,
            7,
            self.device,
            max_iters=self.linear_iterations,
        )
        self.boundary = wp.zeros(self.shell.node_x.size * 42, dtype=float, device=self.device)
        self.delta = wp.zeros_like(self.coupler._iterate_velocity)
        self.inverse = wp.zeros_like(self.coupler._response_inverse)
        self.matrix = wp.zeros_like(self.inverse)
        self.rhs = wp.zeros(self.model.joint_dof_count, dtype=wp.float64, device=self.device)
        self.do_basis = wp.ones(1, dtype=int, device=self.device)
        self.prior_converged = wp.zeros_like(self.do_basis)
        self.progress = wp.zeros_like(self.do_basis)
        self.inverse_kernel = make_inverse(self.model.joint_dof_count)
        self.scratch_count = wp.zeros_like(self.do_basis)
        self.views = [
            a.reshape((self.shell.n_envs * 7, self.response.n))
            for a in (self.boundary, self.response.displacement, self.response.velocity)
        ]
        self.views += [
            a.reshape((self.shell.n_envs, self.response.n))
            for a in (self.shell.lumped_mass_tiled, self.shell.K_contact_diag)
        ]
        self.projected_flat = self.response.impedance.reshape((self.shell.n_envs * 49,))
        self.aa_n = self.shell.lumped_mass_tiled.size + self.delta.size
        self.aa_current = wp.zeros(self.aa_n, dtype=float, device=self.device)
        self.aa_output = wp.zeros_like(self.aa_current)
        self.aa_prev_x = wp.zeros_like(self.aa_current)
        self.aa_prev_r = wp.zeros_like(self.aa_current)
        self.aa_weights = wp.zeros_like(self.aa_current)
        self.aa_nums = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.pcg = fem.PcgSolverBatched(
            self.shell.n_envs,
            self.shell.pcg.n_dof,
            self.shell.pcg.nnz,
            self.device,
            max_iters=self.linear_iterations,
        )
        self.pcg.set_graph(self.shell.blk_offsets.numpy(), self.shell.blk_columns.numpy())
        self.pcg.coarse_x = self.shell.node_x
        self.pcg.coarse_D = self.shell.node_D
        self.pcg.coarse_free = self.shell._dirichlet_dof_mask
        self.kinematic_cache = KinematicStateCache(kinematic_arrays)
        self.rigid_copy_size = max(self.model.body_count, state_0.joint_q.size, state_0.joint_qd.size)
        self.maximum = 2 * self.coupler._n_iters - 1
        self.coupler.interface_residual = wp.zeros(self.maximum + 1, dtype=float, device=self.device)

    def _restore_rigid(self, state):
        snapshot = self.coupler._rigid_snap[self.coupler._rigid_cur]
        wp.launch(
            copy_rigid,
            dim=self.rigid_copy_size,
            inputs=[
                snapshot["bq"],
                snapshot["bqd"],
                snapshot["jq"],
                snapshot["jqd"],
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
            ],
            device=self.device,
        )

    def reset(self):
        """Invalidate the cached tangent and iteration history after a reset."""
        if self.supported:
            self.progress.zero_()
            self.prior_converged.zero_()
            self.aa_prev_x.zero_()
            self.aa_prev_r.zero_()

    def _gather(self, dest, dt):
        wp.launch(
            aa_gather,
            dim=self.aa_n,
            inputs=[
                self.shell.node_xdd,
                self.shell.node_Ddd,
                self.coupler._iterate_velocity,
                self.shell._dirichlet_dof_mask,
                self.shell.lumped_mass_tiled,
                self.rigid_response.mass,
                self.model.joint_armature,
                fem.HHT_GAMMA * dt,
                dest,
                self.aa_weights,
            ],
            device=self.device,
        )

    def _refresh(self, dt, i):
        self.shell._assemble_element_stiffness_batched()
        wp.launch(
            sh._rigid_motion_map,
            dim=(self.shell.n_envs, self.shell.ancf.n_nodes, 6),
            inputs=[
                self.rigid_response.bodies,
                self.reference.body_q,
                self.model.body_com,
                self.shell.node_x,
                self.shell.node_D,
                self.shell.ancf.n_nodes,
                7,
                self.boundary,
            ],
            device=self.device,
        )
        cv, boundary_scale = self.rigid_response.build_tangent(dt)

        def basis():
            self.response.update_geometry(self.shell.node_x, self.shell.node_D, self.shell._dirichlet_dof_mask)

            def add_force():
                wp.launch(
                    force_rhs,
                    dim=(self.shell.n_envs, self.response.n),
                    inputs=[
                        self.response.rhs,
                        self.shell.residual,
                        self.shell._dirichlet_dof_mask,
                        self.response.n,
                    ],
                    device=self.device,
                )

            self.response.solve_displacement(
                self.shell.bsr_values_batched,
                self.shell._dirichlet_dof_mask,
                self.boundary,
                boundary_scale,
                extra_rhs=add_force,
            )

        def free():
            wp.launch(
                fem.apply_dirichlet_to_bsr,
                dim=self.shell.bsr_values_batched.size,
                inputs=[self.shell._dirichlet_nnz_mask, self.shell.bsr_values_batched],
                device=self.device,
            )
            wp.launch(
                fem.mask_dof,
                dim=self.shell.residual.size,
                inputs=[self.shell._dirichlet_dof_mask, self.shell.residual],
                device=self.device,
            )
            wp.launch(
                fem.negate,
                dim=self.shell.residual.size,
                inputs=[self.shell.residual, self.shell.neg_R],
                device=self.device,
            )
            self.pcg.solve(
                self.shell.blk_offsets,
                self.shell.blk_columns,
                self.shell.bsr_values_batched,
                self.shell.neg_R,
                self.shell.da,
                compute_residual_report=False,
            )
            wp.launch(
                insert_free,
                dim=(self.shell.n_envs, self.response.n),
                inputs=[self.shell.da, self.response.displacement, self.response.n],
                device=self.device,
            )

        wp.launch(
            choose_basis,
            dim=1,
            inputs=[
                self.coupler.interface_totals,
                self.coupler._response_ready,
                self.prior_converged,
                i,
                self.do_basis,
            ],
            device=self.device,
        )
        wp.capture_if(self.do_basis, basis, free)
        wp.launch(
            sh._response_velocities,
            dim=self.response.velocity.size,
            inputs=[
                self.response.displacement,
                self.boundary,
                self.shell._dirichlet_dof_mask,
                self.response.n,
                7,
                dt,
                cv,
                self.response.velocity,
            ],
            device=self.device,
        )
        wp.launch_tiled(
            project,
            dim=self.shell.n_envs * 49,
            inputs=[
                *self.views,
                7,
                (self.response.n + 127) // 128,
                dt * fem.HHT_GAMMA * (1 + fem.HHT_ALPHA),
                self.do_basis,
                self.projected_flat,
            ],
            block_dim=128,
            device=self.device,
        )

        def factor():
            wp.launch(
                sh._project_rigid_impedance,
                dim=self.rigid_response.impedance.shape,
                inputs=[
                    self.rigid_response.jacobian,
                    self.rigid_response.rows,
                    self.response.impedance,
                    self.rigid_response.impedance,
                ],
                device=self.device,
            )
            wp.launch(
                reduced_matrix,
                dim=self.matrix.shape,
                inputs=[
                    self.rigid_response.mass,
                    self.model.joint_armature,
                    self.rigid_response.impedance,
                    self.matrix,
                ],
                device=self.device,
            )
            wp.launch(
                self.inverse_kernel,
                dim=(1, max(32, 1 << (self.model.joint_dof_count**2 - 1).bit_length())),
                block_dim=max(32, 1 << (self.model.joint_dof_count**2 - 1).bit_length()),
                inputs=[
                    self.matrix,
                    self.inverse,
                    self.coupler._response_ready,
                    self.coupler._response_age,
                    self.coupler._response_refreshed,
                    self.scratch_count,
                ],
                device=self.device,
            )
            wp.launch(
                response_inverse,
                dim=self.matrix.shape,
                inputs=[
                    self.rigid_response.mass,
                    self.model.joint_armature,
                    self.inverse,
                    self.coupler._response_inverse,
                ],
                device=self.device,
            )
            wp.launch(
                cp._finish_shell_linearization,
                dim=1,
                inputs=[
                    self.coupler._response_ready,
                    self.coupler._response_age,
                    self.coupler._response_refreshed,
                    self.coupler.interface_linearization_count,
                    self.coupler._linearization_failed,
                ],
                device=self.device,
            )

        wp.capture_if(self.do_basis, factor)

    def substep(
        self,
        state_0,
        state_rigid,
        control,
        dt,
        kinematics_fn,
        prescribe_fn,
        accumulate_fn,
        dynamics_fn,
        **kwargs,
    ):
        trial_kinematics = kwargs.get("trial_kinematics_fn") or kinematics_fn

        def prescribe():
            trial_kinematics(state_0, state_rigid, control, None, dt)
            # Complete the pre/post pair so stateful callbacks keep their phase.
            prescribe_fn()
            prescribe_fn()
            self.shell._launch_dirichlet_pred_override(dt)

        def evaluate():
            wp.launch(
                fem.hht_kinematic,
                dim=self.shell.node_x.size,
                inputs=[
                    self.shell.x_pred,
                    self.shell.xd_pred,
                    self.shell.node_xdd,
                    dt,
                    fem.HHT_BETA,
                    fem.HHT_GAMMA,
                ],
                outputs=[self.shell.node_x, self.shell.node_xd],
                device=self.device,
            )
            wp.launch(
                fem.hht_kinematic,
                dim=self.shell.node_D.size,
                inputs=[
                    self.shell.D_pred,
                    self.shell.Dd_pred,
                    self.shell.node_Ddd,
                    dt,
                    fem.HHT_BETA,
                    fem.HHT_GAMMA,
                ],
                outputs=[self.shell.node_D, self.shell.node_Dd],
                device=self.device,
            )
            self.shell._evaluate_forces_batched(dt, assemble_tangent=False)
            wp.launch(
                fem.flatten_vec3_pair,
                dim=self.shell.node_x.size,
                inputs=[self.shell.node_xdd, self.shell.node_Ddd, self.shell.a_flat],
                device=self.device,
            )
            wp.launch(
                fem.pointwise_mul,
                dim=self.shell.a_flat.size,
                inputs=[
                    self.shell.lumped_mass_tiled,
                    self.shell.a_flat,
                    self.shell.M_a,
                ],
                device=self.device,
            )
            wp.launch(
                fem.build_residual,
                dim=self.shell.a_flat.size,
                inputs=[
                    self.shell.M_a,
                    self.shell.global_f_int,
                    self.shell.global_f_int0,
                    self.shell.global_f_ext,
                    self.shell.global_f_ext0,
                    fem.HHT_ALPHA,
                    self.shell.residual,
                ],
                device=self.device,
            )
            self._restore_rigid(state_0)
            if self.kinematic_cache:
                self.kinematic_cache.copy(restore=True)
            else:
                kinematics_fn(state_0, state_rigid, control, None, dt)
            accumulate_fn()
            dynamics_fn(state_rigid)

        self._restore_rigid(state_0)
        kinematics_fn(state_0, state_rigid, control, None, dt)
        self.kinematic_cache.copy(restore=False)
        wp.copy(self.prior_converged, self.progress)
        self.coupler._unpack_ancf(self.coupler._ancf_cur)
        self.coupler._restore_extra()
        wp.copy(self.reference.body_q, state_0.body_q)
        wp.copy(self.reference.joint_q, state_0.joint_q)
        eval_jacobian(
            self.model,
            self.reference,
            J=self.rigid_response.jacobian,
            joint_S_s=self.rigid_response._motion_subspace,
        )
        wp.launch(
            compute_body_spatial_inertia,
            dim=self.model.body_count,
            inputs=[
                self.model.body_inertia,
                self.model.body_mass,
                self.reference.body_q,
                self.rigid_response._spatial_inertia,
            ],
            device=self.device,
        )
        wp.launch(
            rigid_mass,
            dim=self.rigid_response.mass.shape,
            inputs=[
                self.model.articulation_start,
                self.model.articulation_end,
                self.model.joint_child,
                self.rigid_response._spatial_inertia,
                self.rigid_response.jacobian,
                self.rigid_response.mass,
            ],
            device=self.device,
        )
        self.shell.begin_coupling_step(dt)
        wp.copy(self.shell.global_f_int0, self.shell.global_f_int)
        wp.copy(self.shell.global_f_ext0, self.shell.global_f_ext)
        wp.launch(
            fem.hht_predict,
            dim=self.shell.node_x.size,
            inputs=[
                self.shell.node_x,
                self.shell.node_xd,
                self.shell.node_xdd,
                dt,
                fem.HHT_BETA,
                fem.HHT_GAMMA,
            ],
            outputs=[self.shell.x_pred, self.shell.xd_pred],
            device=self.device,
        )
        wp.launch(
            fem.hht_predict,
            dim=self.shell.node_D.size,
            inputs=[
                self.shell.node_D,
                self.shell.node_Dd,
                self.shell.node_Ddd,
                dt,
                fem.HHT_BETA,
                fem.HHT_GAMMA,
            ],
            outputs=[self.shell.D_pred, self.shell.Dd_pred],
            device=self.device,
        )
        wp.launch(
            cp._predict_interface_velocity,
            dim=state_0.joint_qd.size,
            inputs=[
                state_0.joint_qd,
                self.coupler._velocity_increment,
                self.coupler._interface_converged,
            ],
            device=self.device,
        )
        wp.copy(self.coupler._iterate_velocity, state_0.joint_qd)
        self.coupler._predict_pose(state_0, dt)
        prescribe()
        self.coupler._interface_active.fill_(1)
        self.coupler.interface_residual.zero_()
        maximum = self.maximum

        def iteration(i):
            evaluate()
            wp.launch(
                check,
                dim=(1, 32),
                block_dim=32,
                inputs=[
                    self.coupler._iterate_velocity,
                    state_rigid.joint_qd,
                    self.coupler._response_inverse,
                    self.coupler._corrected_velocity,
                    self.coupler.interface_residual,
                    self.coupler._aitken_previous,
                    self.coupler._interface_active,
                    self.coupler._interface_converged,
                    self.coupler.interface_iterations,
                    self.progress,
                    i,
                    maximum,
                    self.coupler._n_iters - 1,
                    self.coupler._velocity_atol,
                    self.coupler._velocity_rtol,
                ],
                device=self.device,
            )

            def correction():
                self._refresh(dt, i)
                wp.launch(
                    reduced_rhs,
                    dim=self.delta.size,
                    inputs=[
                        self.rigid_response.mass,
                        self.model.joint_armature,
                        self.response.impedance,
                        self.rigid_response.jacobian,
                        self.rigid_response.rows,
                        state_rigid.joint_qd,
                        self.coupler._iterate_velocity,
                        self.rhs,
                    ],
                    device=self.device,
                )
                wp.launch(
                    reduced_update,
                    dim=self.delta.size,
                    inputs=[self.inverse, self.rhs, self.delta],
                    device=self.device,
                )
                self._gather(self.aa_current, dt)
                wp.launch(
                    update_acc,
                    dim=(self.shell.n_envs, self.shell.ancf.n_nodes),
                    inputs=[
                        self.response.displacement,
                        self.rigid_response.jacobian,
                        self.rigid_response.rows,
                        self.delta,
                        self.shell._dirichlet_dof_mask,
                        self.shell.node_xdd,
                        self.shell.node_Ddd,
                        self.shell.ancf.n_nodes,
                        1 / (fem.HHT_BETA * dt * dt),
                    ],
                    device=self.device,
                )
                wp.launch(
                    update_guess,
                    dim=self.delta.size,
                    inputs=[
                        self.coupler._iterate_velocity,
                        self.delta,
                        state_0.joint_qd,
                    ],
                    device=self.device,
                )
                self._gather(self.aa_output, dt)
                self.aa_nums.zero_()
                wp.launch_tiled(
                    aa_products,
                    dim=(self.aa_n + 127) // 128,
                    block_dim=128,
                    inputs=[
                        self.aa_current,
                        self.aa_output,
                        self.aa_prev_r,
                        self.aa_weights,
                        self.aa_nums,
                    ],
                    device=self.device,
                )
                wp.launch(
                    aa_update,
                    dim=self.shell.node_x.size + self.delta.size,
                    inputs=[
                        self.aa_current,
                        self.aa_output,
                        self.aa_prev_x,
                        self.aa_prev_r,
                        self.aa_nums,
                        i,
                        self.shell.node_xdd,
                        self.shell.node_Ddd,
                        self.coupler._iterate_velocity,
                        self.shell._dirichlet_dof_mask,
                        fem.HHT_GAMMA * dt,
                    ],
                    device=self.device,
                )
                wp.copy(state_0.joint_qd, self.coupler._iterate_velocity)
                self.coupler._predict_pose(state_0, dt)
                prescribe()

            def accept():
                wp.launch(
                    copy_rigid,
                    dim=self.rigid_copy_size,
                    inputs=[
                        state_rigid.body_q,
                        state_rigid.body_qd,
                        state_rigid.joint_q,
                        state_rigid.joint_qd,
                        state_0.body_q,
                        state_0.body_qd,
                        state_0.joint_q,
                        state_0.joint_qd,
                    ],
                    device=self.device,
                )

            wp.capture_if(self.coupler._interface_active, correction, accept)

        for i in range(maximum + 1):
            wp.capture_if(self.coupler._interface_active, lambda i=i: iteration(i))
        wp.launch(
            cp._save_corrected_increment,
            dim=self.delta.size,
            inputs=[
                state_0.joint_qd,
                self.coupler._corrected_velocity,
                self.coupler._previous_corrected,
                self.coupler._velocity_increment,
                self.coupler._interface_converged,
                self.coupler.interface_iterations,
                self.coupler.interface_totals,
            ],
            device=self.device,
        )
        self.coupler._presave(state_0)
        return maximum + 1
