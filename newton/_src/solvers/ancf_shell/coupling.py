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
    _advance_interface_iteration,
    _advance_response_probe,
    _aitken_coefficient,
    _choose_response_refresh,
    _finish_shell_linearization,
    _integrate_interface_joints,
    _make_response_inverse,
    _newton_interface_step,
    _pack_ancf,
    _predict_interface_velocity,
    _probe_guess,
    _relax_coordinates,
    _response_column,
    _response_correction,
    _response_secant_apply,
    _response_secant_curvature,
    _response_secant_products,
    _save_corrected_increment,
    _save_interface_increment,
    _unpack_ancf,
    _update_response_secant,
)
from .schur import RigidShellSchurResponse


class InterfaceCouplerGS:
    """Interface Gauss-Seidel (IGS) coupler for FEM + rigid-body split solvers.

    Iterates both solvers within each substep to reduce the interface lag.
    Stability depends on convergence of the shared interface DOFs.

    The ANCF solver's internal NR+PCG loop can run as a captured CUDA graph,
    or the caller can capture the complete coupling loop. When *tol* is
    ``0.0`` (default), no GPU-to-CPU synchronisation is needed per substep.
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

    ``reuse_response`` replaces repeated cold starts of the interface solve with
    a small inverse Jacobian of the velocity residual. Finite-difference probes
    seed it after startup, and good Broyden updates follow the current substep.
    Refresh probes are fully rewound, including HHT force history. Every accepted
    state still comes from a full physical response. Early stopping requires a
    velocity residual test; exhausting the iteration budget does not guarantee convergence.

    Args:
        ancf_solver: A :class:`~newton.solvers.SolverANCFShell` instance.
        n_iters: Maximum number of GS coupling iterations per substep.
            ``1`` selects a single exchange; convergence depends on the
            tire impedance and rigid inertia, and must be checked for each model.
        tol: Convergence tolerance for spindle position change [m].  ``0.0``
            (default) disables early exit, avoiding any GPU→CPU sync per substep.
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
            in CUDA captures. Requires response reuse, condensed tangents, and at
            least three evaluations. Small block-local shell systems use up to
            four inner PCG iterations; larger systems retain the shell budget.
            Three extra evaluations are reserved for large interface residuals.
            Singular rigid mass matrices retain partitioned coupling.
        reuse_response: Reuse an interface inverse Jacobian in captured adaptive
            solves with at most 64 joint velocity coordinates. Intended for small
            vehicle interfaces. Uses Aitken during startup and when an estimate is
            invalid; a valid reused response can stop after one verified trial.
            ``interface_probe_count`` counts the additional full tire
            evaluations used to refresh the response; add this to the trial count
            when measuring solver work. With condensed shell tangents,
            ``interface_linearization_count`` counts the cheaper tangent builds.
            Finite-difference refreshes are separated by at least five
            times the number of velocity coordinates; invalid estimates use Aitken
            between refreshes. Callbacks in this captured loop must not depend
            on the host-side ``gs_iter`` value. Uncaptured execution uses legacy Aitken.
    """

    def __init__(
        self,
        ancf_solver,
        n_iters: int = 2,
        tol: float = 0.0,
        acceleration: bool = False,
        initial_relaxation: float = 0.1,
        adaptive: bool = False,
        velocity_atol: float = 1.0e-3,
        velocity_rtol: float = 1.0e-3,
        reuse_response: bool = False,
        coupled_newton: bool = False,
    ) -> None:
        self._ancf = ancf_solver
        self.acceleration = bool(acceleration)
        self.adaptive = bool(adaptive)
        self.reuse_response = bool(reuse_response)
        self._coupled_newton = bool(coupled_newton)
        self._coupled_solver = None
        if coupled_newton and (not reuse_response or n_iters < 3):
            raise ValueError("Coupled Newton requires response reuse and at least three interface evaluations.")
        if self.reuse_response and not self.adaptive:
            raise ValueError("Response reuse requires adaptive coupling.")
        self._shell_response = None
        self._response_warmup_steps = 200
        self._response_refresh_interval = 500
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
        self._tol = tol
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
            interface_body_indices: Optional spindle body indices, one per batched
                implicit tire. Enables condensed shell tangents for response reuse,
                with finite-difference fallback if the estimate is invalid.
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
        self.interface_probe_count = wp.zeros(1, dtype=int, device=dev)
        self.interface_linearization_count = wp.zeros(1, dtype=int, device=dev)
        self._linearization_failed = wp.zeros(1, dtype=int, device=dev)
        if interface_body_indices is not None:
            if not self.reuse_response or model is None:
                raise ValueError("Shell condensation requires response reuse and a rigid model.")
            self._shell_response = RigidShellSchurResponse(a, model, interface_body_indices)
            self._response_refresh_interval = 50
        if self.reuse_response:
            if state_0.joint_qd.shape[0] > 64:
                raise ValueError("Response reuse supports at most 64 generalized velocities.")
            nv = state_0.joint_qd.shape[0]
            self._response_matrix = wp.zeros((nv, nv), dtype=wp.float64, device=dev)
            self._response_inverse = wp.zeros_like(self._response_matrix)
            self._response_ready = wp.zeros(1, dtype=int, device=dev)
            self._response_age = wp.zeros(1, dtype=int, device=dev)
            self._response_refreshed = wp.zeros(1, dtype=int, device=dev)
            self._response_refresh = wp.zeros(1, dtype=int, device=dev)
            self._response_guess = wp.zeros_like(state_0.joint_qd)
            self._interface_iteration = wp.zeros(1, dtype=int, device=dev)
            self._probe_column = wp.zeros(1, dtype=int, device=dev)
            self._probe_active = wp.zeros(1, dtype=int, device=dev)
            self._secant_previous_guess = wp.zeros_like(state_0.joint_qd)
            self._secant_hy = wp.zeros(nv, dtype=wp.float64, device=dev)
            self._secant_sh = wp.zeros(nv, dtype=wp.float64, device=dev)
            # Small interfaces benefit from fewer launches; larger matrices
            # otherwise serialize most of the response work on one GPU thread.
            self._parallel_response = nv >= 8
            self._response_correction = wp.zeros(nv if self._parallel_response else 0, dtype=wp.float64, device=dev)
            self._secant_curvature = wp.zeros(1, dtype=wp.float64, device=dev)
            self._corrected_velocity = wp.clone(state_0.joint_qd)
            self._previous_corrected = wp.clone(state_0.joint_qd)
            self._response_rigid = _mk_rigid()
            self._response_pack = wp.zeros(pack_size, dtype=float, device=dev)
            self._inverse_kernel = _make_response_inverse(nv)
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
        if self.reuse_response:
            self._response_extra = [wp.clone(arr) for arr in self._extra_live]

        self._kinematic_cache = KinematicStateCache(
            (kinematic_state_arrays or ()) if self.acceleration and not self._coupled_newton else ()
        )

        if self._coupled_newton:
            if self._shell_response is None:
                raise ValueError("Coupled Newton requires condensed shell tangents.")
            candidate = CoupledShellNewton(self, state_0, kinematic_state_arrays or ())
            if candidate.supported:
                self._coupled_solver = candidate

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
        self.interface_probe_count.zero_()
        self.interface_linearization_count.zero_()
        self._linearization_failed.zero_()
        if self.reuse_response:
            self._response_ready.zero_()
            self._response_age.zero_()
            self._response_refreshed.zero_()
            self._response_refresh.zero_()
            self._secant_previous_guess.zero_()
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
        interface_body_indices: np.ndarray | None = None,
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
            interface_body_indices: Optional 1-D int32 NumPy array of Newton
                body indices for convergence checking (GPU→CPU sync per iter
                only when *tol* > 0).
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
        sp_prev: np.ndarray | None = None

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
        reusing = conditional and self.reuse_response
        if reusing:
            wp.load_module(module=self._inverse_kernel.module, device=self._ancf.node_x.device)

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

        def calibrate_finite_difference():
            dev = self._ancf.node_x.device
            wp.copy(self._response_guess, self._iterate_velocity)
            for key, name in (
                ("bq", "body_q"),
                ("bqd", "body_qd"),
                ("jq", "joint_q"),
                ("jqd", "joint_qd"),
            ):
                wp.copy(self._response_rigid[key], getattr(state_rigid, name))
            # Preserve this evaluated response while every probe rewinds to the same t_n.
            self._pack_ancf_buffer(self._response_pack)
            for live, snap in zip(self._extra_live, self._response_extra, strict=True):
                wp.copy(snap, live)
            self._probe_column.zero_()
            self._probe_active.fill_(1)
            # Capture one complete response, not one copy per velocity column.
            # A 0.05 velocity probe resolves float32 bead-pose differences at this dt.

            def probe():
                wp.launch(
                    _probe_guess,
                    dim=state_0.joint_qd.shape[0],
                    inputs=[self._response_guess, state_0.joint_qd, self._probe_column, 0.05],
                    device=dev,
                )
                self._predict_pose(state_0, dt)
                evaluate_response()
                wp.launch(
                    _response_column,
                    dim=state_0.joint_qd.shape[0],
                    inputs=[
                        self._response_rigid["jqd"],
                        state_rigid.joint_qd,
                        self._response_matrix,
                        self._probe_column,
                        0.05,
                    ],
                    device=dev,
                )
                wp.launch(
                    _advance_response_probe,
                    dim=1,
                    inputs=[self._probe_column, self._probe_active, state_0.joint_qd.shape[0]],
                    device=dev,
                )

            wp.capture_while(self._probe_active, probe)
            wp.launch(
                self._inverse_kernel,
                dim=1,
                inputs=[
                    self._response_matrix,
                    self._response_inverse,
                    self._response_ready,
                    self._response_age,
                    self._response_refreshed,
                    self.interface_probe_count,
                ],
                device=dev,
            )
            self._unpack_ancf_buffer(self._response_pack)
            for live, snap in zip(self._extra_live, self._response_extra, strict=True):
                wp.copy(live, snap)
            for key, name in (
                ("bq", "body_q"),
                ("bqd", "body_qd"),
                ("jq", "joint_q"),
                ("jqd", "joint_qd"),
            ):
                wp.copy(getattr(state_rigid, name), self._response_rigid[key])
            wp.copy(self._iterate_velocity, self._response_guess)
            accumulate_fn()

        def calibrate_response():
            if self._shell_response is None:
                calibrate_finite_difference()
                return
            self._shell_response.refresh(state_0, dt, self._response_inverse, self._response_ready)
            wp.launch(
                _finish_shell_linearization,
                dim=1,
                inputs=[
                    self._response_ready,
                    self._response_age,
                    self._response_refreshed,
                    self.interface_linearization_count,
                    self._linearization_failed,
                ],
                device=self._ancf.node_x.device,
            )

            def fallback():
                calibrate_finite_difference()
                # An unsupported/noisy tangent must not trigger a full finite-
                # difference rebuild at each subsequent substep.
                self._response_age.fill_(-5 * state_0.joint_qd.shape[0])

            wp.capture_if(self._linearization_failed, fallback)

        def trial(k):
            nonlocal sp_prev
            # Accelerated trials already carry an end-of-step pose. Legacy
            # unaccelerated callers extrapolate their first bead prescription.
            self.gs_iter = k
            if self.acceleration:
                wp.copy(self._iterate_velocity, state_0.joint_qd)

            evaluate_response()

            if reusing:
                wp.launch(
                    _choose_response_refresh,
                    dim=1,
                    inputs=[
                        self._iterate_velocity,
                        state_rigid.joint_qd,
                        self._aitken_previous,
                        self._response_ready,
                        self._response_age,
                        self.interface_totals,
                        self._response_refreshed,
                        self._response_refresh,
                        self._interface_iteration,
                        self._velocity_atol,
                        self._velocity_rtol,
                        self._response_warmup_steps,
                        self._response_refresh_interval,
                        1 if self._shell_response is not None else 5 * state_0.joint_qd.shape[0],
                        self.interface_linearization_count
                        if self._shell_response is not None
                        else self.interface_probe_count,
                    ],
                    device=self._ancf.node_x.device,
                )
                wp.capture_if(self._response_refresh, calibrate_response)

            if self.adaptive:
                if reusing:
                    if self._parallel_response:
                        wp.launch(
                            _response_secant_products,
                            dim=self._iterate_velocity.shape[0],
                            inputs=[
                                self._iterate_velocity,
                                state_rigid.joint_qd,
                                self._secant_previous_guess,
                                self._aitken_previous,
                                self._response_inverse,
                                self._response_ready,
                                self._response_refresh,
                                self._secant_hy,
                                self._secant_sh,
                                self._interface_iteration,
                            ],
                            device=self._ancf.node_x.device,
                        )
                        wp.launch(
                            _response_secant_curvature,
                            dim=1,
                            inputs=[
                                self._iterate_velocity,
                                state_rigid.joint_qd,
                                self._secant_previous_guess,
                                self._aitken_previous,
                                self._response_ready,
                                self._response_refresh,
                                self._secant_hy,
                                self._interface_iteration,
                                self._secant_curvature,
                            ],
                            device=self._ancf.node_x.device,
                        )
                        wp.launch(
                            _response_secant_apply,
                            dim=self._iterate_velocity.shape[0],
                            inputs=[
                                self._iterate_velocity,
                                self._secant_previous_guess,
                                self._response_inverse,
                                self._response_ready,
                                self._secant_hy,
                                self._secant_sh,
                                self._secant_curvature,
                            ],
                            device=self._ancf.node_x.device,
                        )
                        wp.launch(
                            _response_correction,
                            dim=self._iterate_velocity.shape[0],
                            inputs=[
                                self._iterate_velocity,
                                state_rigid.joint_qd,
                                self._response_inverse,
                                self._response_ready,
                                self._response_correction,
                            ],
                            device=self._ancf.node_x.device,
                        )
                    else:
                        wp.launch(
                            _update_response_secant,
                            dim=1,
                            inputs=[
                                self._iterate_velocity,
                                state_rigid.joint_qd,
                                self._secant_previous_guess,
                                self._aitken_previous,
                                self._response_inverse,
                                self._response_ready,
                                self._response_refresh,
                                self._secant_hy,
                                self._secant_sh,
                                self._interface_iteration,
                            ],
                            device=self._ancf.node_x.device,
                        )
                    wp.launch(
                        _newton_interface_step,
                        dim=1,
                        inputs=[
                            self._iterate_velocity,
                            state_rigid.joint_qd,
                            self._aitken_previous,
                            self._aitken_weight,
                            self._response_inverse,
                            self._response_correction,
                            self._response_ready,
                            state_0.joint_qd,
                            self.interface_residual,
                            self._interface_active,
                            self._interface_converged,
                            self.interface_iterations,
                            self._interface_iteration,
                            self._n_iters,
                            self._velocity_atol,
                            self._velocity_rtol,
                            self._initial_relaxation,
                            conditional,
                        ],
                        device=self._ancf.node_x.device,
                    )
                    wp.copy(self._corrected_velocity, state_0.joint_qd)
                else:
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
                if reusing:
                    wp.capture_if(self._interface_active, predict_pose, accept_response)
                elif k + 1 == self._n_iters:
                    accept_response()
                elif conditional and (reusing or k >= 2):
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

            # ── convergence check (GPU→CPU, skipped when tol == 0) ────────
            if self._tol > 0.0 and interface_body_indices is not None:
                sp_new = state_0.body_q.numpy()[interface_body_indices, :3]
                if sp_prev is not None:
                    if float(np.max(np.abs(sp_new - sp_prev))) < self._tol:
                        self._presave(state_0)
                        return k + 1
                sp_prev = sp_new

        # Reuse one trial graph, including its conditional calibration loop.
        # Unrolling trials duplicates every FEM solve even when branches are idle.
        if reusing:
            self._interface_iteration.zero_()

            def reused_trial():
                trial(0)
                wp.launch(
                    _advance_interface_iteration,
                    dim=1,
                    inputs=[self._interface_iteration],
                    device=self._ancf.node_x.device,
                )

            wp.capture_while(self._interface_active, reused_trial)
        else:
            for k in range(self._n_iters):
                if conditional and k >= (1 if reusing else 3):
                    wp.capture_if(self._interface_active, lambda k=k: trial(k))
                else:
                    completed = trial(k)
                    if completed is not None:
                        return completed
        if reusing:
            wp.launch(
                _save_corrected_increment,
                dim=state_0.joint_qd.shape[0],
                inputs=[
                    state_0.joint_qd,
                    self._corrected_velocity,
                    self._previous_corrected,
                    self._velocity_increment,
                    self._interface_converged,
                    self.interface_iterations,
                    self.interface_totals,
                ],
                device=self._ancf.node_x.device,
            )

        elif self.adaptive:
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
