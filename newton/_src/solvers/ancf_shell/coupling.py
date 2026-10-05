# Copyright (c) 2022-2026, The Newton Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Partitioned and condensed coupling for ANCF shells and rigid spindles.

This specialized runtime is independent of the experimental proxy/ADMM framework.
Vehicle examples provide the rigid kinematics, bead prescription, and wrench callbacks.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim.articulation import eval_fk
from ._kinematic_cache import KinematicStateCache
from .coupled_newton import CoupledShellNewton
from .kernels_coupling import (
    _adaptive_interface_step,
    _aitken_coefficient,
    _integrate_interface_joints,
    _pack_ancf,
    _predict_interface_velocity,
    _relax_coordinates,
    _save_interface_increment,
    _unpack_ancf,
)
from .schur import RigidShellSchurResponse


class InterfaceCouplerGS:
    """Interface Gauss-Seidel (IGS) coupler for FEM + rigid-body split solvers.

    Iterates both solvers within each substep to reduce the interface lag.
    Stability depends on convergence of the shared interface DOFs.

    The ANCF solver's internal NR+PCG loop can run as a captured CUDA graph,
    or the caller can capture the complete coupling loop; neither needs a
    GPU-to-CPU synchronisation per substep.
    Adaptive mode can skip converged trials inside a CUDA conditional graph;
    otherwise the fixed iteration budget runs in full.

    **Double-buffer layout** — two pre-allocated ANCF pack buffers (A and B)
    alternate roles each substep via an integer flip.  The *pre-save* (packing
    the end-of-substep ANCF state into the inactive buffer) happens at the
    **end** of each substep, so the next substep's restore can start
    immediately. Saving still incurs device work once per substep.

    ANCF restore uses a single :func:`_unpack_ancf` kernel dispatch instead of
    eight separate ``wp.copy`` calls, reducing CUDA kernel launch overhead by
    8× per GS iteration.

    Extra arrays registered via *extra_arrays* are saved and restored only if
    they are modified during GS iterations.  Arrays that are read-only during
    GS (e.g. accumulated rim-angle) should **not** be registered — do not
    pass them as *extra_arrays*.

    With acceleration enabled, every rigid dynamics trial starts at *t_n*;
    only the prescribed interface pose is iterated. The last rigid response
    is accepted without wrench scaling. Inspect ``interface_residual`` when
    selecting the iteration count; a fixed count does not guarantee convergence.

    ``coupled_newton`` replaces repeated cold starts of the interface solve: the
    shell Newton iterate persists across interface corrections and a condensed
    interface tangent is reused between refreshes. Every accepted state still
    comes from a full physical response; exhausting the iteration budget does not
    guarantee convergence.

    Args:
        ancf_solver: A :class:`~newton.solvers.SolverANCFShell` instance.
        n_iters: Maximum number of GS coupling iterations per substep.
            ``1`` selects a single exchange; convergence depends on the
            tire impedance and rigid inertia, and must be checked for each model.
        acceleration: Apply vector Aitken relaxation to the joint velocity
            interface and construct end-of-step trial poses consistently.
        initial_relaxation: Initial Aitken weight per substep, in (0, 1].
            This controls the interface iteration, not the physical wrench.
        adaptive: Predict the next interface velocity using its previous
            increment only after a converged step, and stop captured CUDA trials
            when their velocity residual
            meets the tolerance. Without response reuse, at least three trials run
            (or the full budget when it is smaller). Requires ``acceleration=True``. The caller must
            warm up its solvers outside conditional capture to allocate workspace.
            Uncaptured execution retains the full iteration budget.
        velocity_atol: Absolute per-coordinate velocity tolerance [m/s or rad/s].
            Defaults to 1 mm/s for translation and 1 mrad/s for rotation,
            combined with the relative tolerance for adaptive stopping. Inspect ``interface_residual`` for trials that
            exhaust the iteration budget; this is not a convergence guarantee.
        velocity_rtol: Relative per-coordinate velocity tolerance. A trial
            converges when every error is at most ``atol + rtol * max(abs(guess),
            abs(response))``. ``interface_iterations`` and ``interface_totals``
            expose the adaptive trial count on the device, without readbacks.
        coupled_newton: Keep shell Newton iterates between interface corrections
            in CUDA captures. Requires adaptive coupling, condensed tangents
            (``interface_body_indices`` in :meth:`allocate`), and at least three
            evaluations. Small block-local shell systems use up to
            four inner PCG iterations; larger systems retain the shell budget.
            Three extra evaluations are reserved for large interface residuals.
            Rigid models it does not support (singular mass matrix, more than 32
            dofs) fall back to adaptive coupling. ``interface_linearization_count``
            counts the condensed tangent builds. Callbacks in this captured loop
            must not depend on the host-side ``gs_iter`` value; uncaptured
            execution uses adaptive Aitken relaxation.
    """

    def __init__(
        self,
        ancf_solver,
        n_iters: int = 2,
        acceleration: bool = False,
        initial_relaxation: float = 0.1,
        adaptive: bool = False,
        velocity_atol: float = 1.0e-3,
        velocity_rtol: float = 1.0e-3,
        coupled_newton: bool = False,
    ) -> None:
        self._ancf = ancf_solver
        self.acceleration = bool(acceleration)
        self.adaptive = bool(adaptive)
        self._coupled_newton = bool(coupled_newton)
        # Reported by the acceptance probes: True while coupled Newton (with its
        # reused condensed interface tangent) is active; allocate() clears it on fallback.
        self.reuse_response = self._coupled_newton
        self._coupled_solver = None
        if coupled_newton and (not self.adaptive or n_iters < 3):
            raise ValueError("Coupled Newton requires adaptive coupling and at least three interface evaluations.")
        self._shell_response = None
        if self.adaptive and not self.acceleration:
            raise ValueError("Adaptive stopping requires accelerated coupling.")
        if velocity_atol <= 0.0 or velocity_rtol < 0.0:
            raise ValueError("Velocity tolerances require atol > 0 and rtol >= 0.")
        self._velocity_atol = float(velocity_atol)
        self._velocity_rtol = float(velocity_rtol)
        if not 0.0 < initial_relaxation <= 1.0:
            raise ValueError("initial_relaxation must be in (0, 1].")
        self._initial_relaxation = float(initial_relaxation)
        self._n_iters = n_iters
        self._n_nodes = None  # set by allocate()
        self.gs_iter = 0  # current GS iteration inside substep(); see prescribe_fn

    # ── allocation ──────────────────────────────────────────────────────────

    def allocate(
        self,
        state_0,
        extra_arrays: list[wp.array] | None = None,
        model=None,
        interface_body_indices: list[int] | np.ndarray | None = None,
        kinematic_state_arrays: list[wp.array] | None = None,
    ) -> None:
        """Allocate GPU double-buffer storage.  Call once after solver init.

        Args:
            state_0: Newton ``State`` whose ``body_q``, ``body_qd``,
                ``joint_q``, and ``joint_qd`` are used as templates for rigid
                snapshots.
            extra_arrays: Optional list of ``wp.array`` objects that are
                **written to** during GS iterations and therefore need
                per-iteration save/restore.  Read-only arrays (e.g. rim-angle
                accumulator that is only updated outside the GS loop) should
                not be registered here.
            kinematic_state_arrays: Optional rigid kinematic arrays that the trial
                pose or rigid response overwrites. Accelerated coupling caches their
                time-n values when ``trial_kinematics_fn`` is supplied; coupled
                Newton also uses them between corrections. Other rigid setup
                data must remain valid at time n. Without this cache, each
                response reruns ``kinematics_fn``.
            model: Rigid model used for joint integration and forward kinematics
                when acceleration is enabled.
            interface_body_indices: Spindle body indices, one per batched implicit
                tire. Required for coupled Newton, which condenses the shell tangent
                onto these bodies.
        """
        a = self._ancf
        dev = a.node_x.device

        n = a.node_x.shape[0]  # total node count (all envs)
        self._n_nodes = n
        # 30 floats per node: 6 vec3 arrays (3 f32 each) + f_int (6 f32) + f_int0 (6 f32)
        pack_size = 30 * n

        # ── ANCF double buffer (A=0, B=1) ────────────────────────────────
        self._ancf_pack = [
            wp.zeros(pack_size, dtype=float, device=dev),
            wp.zeros(pack_size, dtype=float, device=dev),
        ]
        self._ancf_cur = 0
        # Prime buffer[0] with the initial ANCF state so the first substep
        # can restore without a prior save.
        self._pack_ancf(self._ancf_cur)

        # ── Rigid double buffer ───────────────────────────────────────────
        def _mk_rigid():
            return {
                "bq": wp.clone(state_0.body_q),
                "bqd": wp.clone(state_0.body_qd),
                "jq": wp.clone(state_0.joint_q),
                "jqd": wp.clone(state_0.joint_qd),
            }

        self._rigid_snap = [_mk_rigid(), _mk_rigid()]
        self._iterate_velocity = wp.clone(state_0.joint_qd)
        self.interface_linearization_count = wp.zeros(1, dtype=int, device=dev)
        self._linearization_failed = wp.zeros(1, dtype=int, device=dev)
        if self._coupled_newton:
            if interface_body_indices is None or model is None:
                raise ValueError("Coupled Newton requires the rigid model and one spindle body index per tire.")
            nv = state_0.joint_qd.shape[0]
            if nv > 64:
                raise ValueError("Coupled Newton supports at most 64 generalized velocities.")
            self._shell_response = RigidShellSchurResponse(a, model, interface_body_indices)
            self._response_inverse = wp.zeros((nv, nv), dtype=wp.float64, device=dev)
            self._response_ready = wp.zeros(1, dtype=int, device=dev)
            self._response_age = wp.zeros(1, dtype=int, device=dev)
            self._response_refreshed = wp.zeros(1, dtype=int, device=dev)
            self._corrected_velocity = wp.clone(state_0.joint_qd)
            self._previous_corrected = wp.clone(state_0.joint_qd)
        self._zero_qdd = wp.zeros_like(state_0.joint_qd)
        self._predicted_velocity = wp.zeros_like(state_0.joint_qd)
        self._aitken_previous = wp.zeros_like(state_0.joint_qd)
        self._aitken_weight = wp.ones(1, dtype=float, device=dev)
        self.interface_residual = wp.zeros(self._n_iters, dtype=float, device=dev)
        self._rigid_model = model
        self._velocity_increment = wp.zeros_like(state_0.joint_qd)
        self._interface_active = wp.ones(1, dtype=int, device=dev)
        self._interface_converged = wp.zeros(1, dtype=int, device=dev)
        self.interface_iterations = wp.zeros(1, dtype=int, device=dev)
        self.interface_totals = wp.zeros(2, dtype=int, device=dev)
        if self.acceleration and model is None:
            raise ValueError("Accelerated coupling requires the rigid model in allocate().")
        self._rigid_cur = 0
        # Prime rigid buffer[0] with initial state.
        self._save_rigid(state_0, self._rigid_cur)

        # ── Extra application arrays (truly read-write during GS) ─────────
        self._extra_live = list(extra_arrays or [])
        # HHT weights both internal and external forces at the previous step.
        # Restore external history too when retrying the same substep.
        for name in ("global_f_ext", "global_f_ext0"):
            array = getattr(a, name, None)
            if array is not None and all(array is not other for other in self._extra_live):
                self._extra_live.append(array)
        self._extra_snaps = [wp.clone(arr) for arr in self._extra_live]

        if self._coupled_newton:
            candidate = CoupledShellNewton(self, state_0, kinematic_state_arrays or ())
            if candidate.supported:
                self._coupled_solver = candidate
            else:
                # Singular rigid mass or too many dofs: run adaptive Aitken coupling instead.
                self._coupled_newton = False
                self.reuse_response = False
                self._shell_response = None
        self._kinematic_cache = KinematicStateCache(
            (kinematic_state_arrays or ()) if self.acceleration and not self._coupled_newton else ()
        )

    def reset(self, state_0) -> None:
        """Refresh both snapshots after the caller resets rigid and tire states.

        This preserves buffer identities used by captured graphs. Reset solver
        integration histories before calling this method.
        """
        if self._n_nodes is None:
            raise RuntimeError("Call allocate() before reset().")
        for index in range(2):
            self._pack_ancf(index)
            self._save_rigid(state_0, index)
        self._save_extra()
        if self._coupled_solver is not None:
            self._coupled_solver.reset()
        self._aitken_previous.zero_()
        self.interface_residual.zero_()
        self._velocity_increment.zero_()
        self._interface_converged.zero_()
        self.interface_iterations.zero_()
        self.interface_totals.zero_()
        self.interface_linearization_count.zero_()
        self._linearization_failed.zero_()
        if self._coupled_newton:
            self._response_ready.zero_()
            self._response_age.zero_()
            self._response_refreshed.zero_()
            wp.copy(self._corrected_velocity, state_0.joint_qd)
            wp.copy(self._previous_corrected, state_0.joint_qd)

    # ── pack / unpack helpers ────────────────────────────────────────────────

    def _pack_ancf(self, buf_idx: int) -> None:
        """Pack live ANCF state into pack buffer *buf_idx* (1 kernel dispatch)."""
        self._pack_ancf_buffer(self._ancf_pack[buf_idx])

    def _pack_ancf_buffer(self, buffer: wp.array) -> None:
        a = self._ancf
        wp.launch(
            _pack_ancf,
            dim=self._n_nodes,
            inputs=[
                a.node_x,
                a.node_xd,
                a.node_xdd,
                a.node_D,
                a.node_Dd,
                a.node_Ddd,
                a.global_f_int,
                a.global_f_int0,
                buffer,
                self._n_nodes,
            ],
            device=a.node_x.device,
        )

    def _unpack_ancf(self, buf_idx: int) -> None:
        """Unpack ANCF pack buffer *buf_idx* into live ANCF arrays (1 kernel)."""
        self._unpack_ancf_buffer(self._ancf_pack[buf_idx])

    def _unpack_ancf_buffer(self, buffer: wp.array) -> None:
        a = self._ancf
        wp.launch(
            _unpack_ancf,
            dim=self._n_nodes,
            inputs=[
                buffer,
                a.node_x,
                a.node_xd,
                a.node_xdd,
                a.node_D,
                a.node_Dd,
                a.node_Ddd,
                a.global_f_int,
                a.global_f_int0,
                self._n_nodes,
            ],
            device=a.node_x.device,
        )

    # ── rigid snapshot helpers ───────────────────────────────────────────────

    def _save_rigid(self, state_0, buf_idx: int) -> None:
        r = self._rigid_snap[buf_idx]
        wp.copy(r["bq"], state_0.body_q)
        wp.copy(r["bqd"], state_0.body_qd)
        wp.copy(r["jq"], state_0.joint_q)
        wp.copy(r["jqd"], state_0.joint_qd)

    def _restore_rigid(self, state_0) -> None:
        """Restore Newton State body/joint arrays from current rigid snapshot."""
        r = self._rigid_snap[self._rigid_cur]
        wp.copy(state_0.body_q, r["bq"])
        wp.copy(state_0.body_qd, r["bqd"])
        wp.copy(state_0.joint_q, r["jq"])
        wp.copy(state_0.joint_qd, r["jqd"])

    # ── extra arrays ─────────────────────────────────────────────────────────

    def _save_extra(self) -> None:
        for live, snap in zip(self._extra_live, self._extra_snaps, strict=False):
            wp.copy(snap, live)

    def _restore_extra(self) -> None:
        for live, snap in zip(self._extra_live, self._extra_snaps, strict=False):
            wp.copy(live, snap)

    # ── pre-save (called at END of substep, not start) ───────────────────────

    def _presave(self, state_0) -> None:
        """Pack t_{n+1} ANCF + rigid state into the inactive double buffer and flip.

        By running at the END of each substep (after the last GS iteration),
        the next substep's restore can start immediately without any save
        overhead in the hot path — the data is already there.
        """
        nxt = 1 - self._ancf_cur
        self._pack_ancf(nxt)  # 1 kernel: ANCF → pack[nxt]
        self._save_rigid(state_0, 1 - self._rigid_cur)  # 4 copies: rigid → snap[nxt]
        if self._extra_live:
            self._save_extra()
        self._ancf_cur = nxt
        self._rigid_cur = 1 - self._rigid_cur

    # ── substep ─────────────────────────────────────────────────────────────

    def _predict_pose(self, state_0, dt: float) -> None:
        """Integrate an interface guess from the saved start-of-step coordinates."""
        model = self._rigid_model
        wp.launch(
            _integrate_interface_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_X_c,
                model.body_com,
                self._rigid_snap[self._rigid_cur]["jq"],
                state_0.joint_qd,
                self._zero_qdd,
                dt,
                state_0.joint_q,
                self._predicted_velocity,
            ],
            device=model.device,
        )
        eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    def substep(
        self,
        state_0,
        state_rigid,
        control,
        dt: float,
        kinematics_fn,
        prescribe_fn,
        accumulate_fn,
        dynamics_fn,
        ancf_step_fn=None,
        trial_kinematics_fn=None,
    ) -> int:
        """Run one IGS-coupled substep.

        The double-buffer invariant on entry: ``_ancf_pack[_ancf_cur]`` and
        ``_rigid_snap[_rigid_cur]`` already hold the *t_n* snapshot (filled by
        :meth:`_presave` at the end of the previous substep, or by
        :meth:`allocate` for the very first substep).  No save copy occurs in
        this hot path.

        Args:
            state_0: Newton ``State`` at *t_n* (read/write — advanced to
                *t_{n+1}* on return).
            state_rigid: Newton ``State`` used as the kinematics/dynamics
                output buffer.
            control: Newton ``Control`` passed verbatim to *kinematics_fn*.
            dt: Substep time delta [s].
            kinematics_fn: Callable ``(state_0, state_rigid, control, None, dt)``
                that runs MuJoCo forward kinematics, filling ``solver.xpos``
                and ``solver.cvel``.
            prescribe_fn: Zero-argument callable.  Prescribes ANCF bead BCs
                from the current ``solver.xpos`` / ``solver.cvel``.  Called
                **twice** per GS iteration (pre- and post-ANCF step).
            accumulate_fn: Zero-argument callable.  Reads ANCF forces and
                writes ``solver.xfrc_applied``.
            dynamics_fn: Callable ``(state_rigid)`` that runs MuJoCo
                constraint solve and integration.
            ancf_step_fn: Optional direct shell step used when capturing the
                complete coupling loop; default replays the shell graph.
            trial_kinematics_fn: Optional cheaper replacement for the predicted
                pose kinematics, with the same arguments as ``kinematics_fn``.
                It must populate every pose/velocity read by ``prescribe_fn``.
                Requires acceleration. Full ``kinematics_fn`` still runs at t_n
                before every rigid dynamics evaluation, including response probes.

        Returns:
            Number of scheduled iterations. With adaptive CUDA capture, read
            ``interface_iterations`` outside the simulation loop for the actual
            device-side count. ``interface_totals`` holds substeps and trials.
        """
        if self._n_nodes is None:
            raise RuntimeError("InterfaceCouplerGS: call allocate() before substep().")

        if trial_kinematics_fn is not None and not self.acceleration:
            raise ValueError("Trial-only kinematics requires accelerated coupling.")
        if self._coupled_solver is not None and self._ancf.device.is_capturing:
            return self._coupled_solver.substep(
                state_0,
                state_rigid,
                control,
                dt,
                kinematics_fn,
                prescribe_fn,
                accumulate_fn,
                dynamics_fn,
                trial_kinematics_fn=trial_kinematics_fn,
            )

        def predict_pose():
            self._predict_pose(state_0, dt)

        def accept_response():
            wp.copy(state_0.joint_qd, state_rigid.joint_qd)
            wp.copy(state_0.joint_q, state_rigid.joint_q)
            wp.copy(state_0.body_q, state_rigid.body_q)
            wp.copy(state_0.body_qd, state_rigid.body_qd)

        cache_kinematics = bool(self._kinematic_cache) and trial_kinematics_fn is not None
        if cache_kinematics:
            self._restore_rigid(state_0)
            kinematics_fn(state_0, state_rigid, control, None, dt)
            self._kinematic_cache.copy(restore=False)

        if self.adaptive:
            wp.launch(
                _predict_interface_velocity,
                dim=state_0.joint_qd.shape[0],
                inputs=[
                    state_0.joint_qd,
                    self._velocity_increment,
                    self._interface_converged,
                ],
                device=self._ancf.node_x.device,
            )
        if self.acceleration:
            self._predict_pose(state_0, dt)

        if self.adaptive:
            self.interface_residual.zero_()
            self._interface_active.fill_(1)
        conditional = self.adaptive and self._ancf.node_x.device.is_capturing

        def evaluate_response():
            # ── restore ANCF to t_n (1 kernel, not 8 copies) ──────────────
            self._unpack_ancf(self._ancf_cur)

            # ── restore extra read-write arrays (if any) ──────────────────
            if self._extra_live:
                self._restore_extra()

            # ── kinematics from state_0 (= x_sp^k) ────────────────────────
            # Evaluate the trial pose for bead prescription; accelerated
            # trials are then rewound before rigid dynamics.
            (trial_kinematics_fn or kinematics_fn)(state_0, state_rigid, control, None, dt)

            if hasattr(self._ancf, "begin_coupling_step"):
                self._ancf.begin_coupling_step(dt)

            # ── pre-step bead prescription ────────────────────────────────
            prescribe_fn()

            # ── ANCF FEM step (internal CUDA graph) ───────────────────────
            if ancf_step_fn is None:
                self._ancf.graph_step()
            else:
                ancf_step_fn()

            # ── post-step bead snap ────────────────────────────────────────
            prescribe_fn()

            # Restore t_n before dynamics. Only an unaccelerated first trial
            # already has t_n kinematics and can skip this call.
            #
            # NOTE: accumulate_fn() is intentionally placed AFTER this block.
            # step_kinematics calls _apply_mjc_control which launches
            # apply_mjc_body_f_kernel over all bodies, writing state.body_f
            # (typically zeros) to xfrc_applied and clearing any forces that
            # accumulate_fn() may have written.  Accumulating after kinematics
            # ensures the forces survive into dynamics.
            if self.gs_iter > 0 or self.acceleration:
                self._restore_rigid(state_0)
                if cache_kinematics:
                    self._kinematic_cache.copy(restore=True)
                else:
                    kinematics_fn(state_0, state_rigid, control, None, dt)

            # ── accumulate ANCF wrenches → solver.xfrc_applied ───────────
            accumulate_fn()

            # ── MuJoCo dynamics ────────────────────────────────────────────
            dynamics_fn(state_rigid)

        def trial(k):
            # Accelerated trials already carry an end-of-step pose; unaccelerated
            # ones extrapolate their first bead prescription.
            self.gs_iter = k
            if self.acceleration:
                wp.copy(self._iterate_velocity, state_0.joint_qd)

            evaluate_response()

            if self.adaptive:
                wp.launch(
                    _adaptive_interface_step,
                    dim=1,
                    inputs=[
                        self._iterate_velocity,
                        state_rigid.joint_qd,
                        self._aitken_previous,
                        self._aitken_weight,
                        state_0.joint_qd,
                        self.interface_residual,
                        self._interface_active,
                        self._interface_converged,
                        self.interface_iterations,
                        k,
                        self._n_iters,
                        self._velocity_atol,
                        self._velocity_rtol,
                        self._initial_relaxation,
                        conditional,
                    ],
                    device=self._ancf.node_x.device,
                )
                # The final raw rigid response is always accepted. Convergence
                # changes the amount of work, never scales a physical wrench.
                if k + 1 == self._n_iters:
                    accept_response()
                elif conditional and k >= 2:
                    wp.capture_if(self._interface_active, predict_pose, accept_response)
                else:
                    self._predict_pose(state_0, dt)
                return
            if self.acceleration:
                wp.launch(
                    _aitken_coefficient,
                    dim=1,
                    inputs=[
                        self._iterate_velocity,
                        state_rigid.joint_qd,
                        self._aitken_previous,
                        self._aitken_weight,
                        self.interface_residual,
                        k,
                        self._initial_relaxation,
                    ],
                    device=self._ancf.node_x.device,
                )
                wp.launch(
                    _relax_coordinates,
                    dim=state_0.joint_qd.shape[0],
                    inputs=[
                        self._iterate_velocity,
                        state_rigid.joint_qd,
                        self._aitken_weight,
                        state_0.joint_qd,
                    ],
                    device=self._ancf.node_x.device,
                )
                if k + 1 == self._n_iters:
                    wp.copy(state_0.joint_qd, state_rigid.joint_qd)
                    wp.copy(state_0.joint_q, state_rigid.joint_q)
                    wp.copy(state_0.body_q, state_rigid.body_q)
                    wp.copy(state_0.body_qd, state_rigid.body_qd)
                else:
                    self._predict_pose(state_0, dt)
            else:
                wp.copy(state_0.body_q, state_rigid.body_q)
                wp.copy(state_0.body_qd, state_rigid.body_qd)
                wp.copy(state_0.joint_q, state_rigid.joint_q)
                wp.copy(state_0.joint_qd, state_rigid.joint_qd)

        for k in range(self._n_iters):
            if conditional and k >= 3:
                wp.capture_if(self._interface_active, lambda k=k: trial(k))
            else:
                trial(k)
        if self.adaptive:
            wp.launch(
                _save_interface_increment,
                dim=state_0.joint_qd.shape[0],
                inputs=[
                    state_0.joint_qd,
                    self._rigid_snap[self._rigid_cur]["jqd"],
                    self._velocity_increment,
                    self.interface_iterations,
                    self.interface_totals,
                ],
                device=self._ancf.node_x.device,
            )

        # ── pre-save t_{n+1} into inactive buffer (free for next substep) ─
        self._presave(state_0)
        return self._n_iters
