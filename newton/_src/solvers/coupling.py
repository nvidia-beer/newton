# Copyright (c) 2022-2026, The Newton Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Interface coupling utilities for split FEM + rigid-body co-simulation."""

from __future__ import annotations

import numpy as np
import warp as wp

# ── ANCF state pack / unpack kernels ────────────────────────────────────────
# Pack all eight dynamic ANCF arrays (6 × vec3 node arrays + global_f_int +
# global_f_int0) into a single contiguous float buffer.  Using one kernel
# dispatch instead of eight separate wp.copy calls reduces CUDA launch
# overhead by 8×.
#
# global_f_int0 MUST be packed alongside global_f_int.  After each
# graph_step() the solver writes ``global_f_int0 = global_f_int`` (HHT
# history update).  Without restoring global_f_int0, GS iteration k=1
# would see the future (k=0) forces as its "previous-step" history, causing
# the HHT predictor to overshoot on every second iteration → cascade NaN.
#
# Buffer layout (n = total node count across all envs):
#   [0        … 3n)    node_x   (3 floats per node)
#   [3n       … 6n)    node_xd
#   [6n       … 9n)    node_xdd
#   [9n       … 12n)   node_D
#   [12n      … 15n)   node_Dd
#   [15n      … 18n)   node_Ddd
#   [18n      … 24n)   global_f_int   (6 floats per node)
#   [24n      … 30n)   global_f_int0  (6 floats per node)
# Total: 30n floats.


@wp.kernel
def _pack_ancf(
    nx: wp.array[wp.vec3],
    nxd: wp.array[wp.vec3],
    nxdd: wp.array[wp.vec3],
    nd: wp.array[wp.vec3],
    ndd: wp.array[wp.vec3],
    nddd: wp.array[wp.vec3],
    fint: wp.array[float],
    fint0: wp.array[float],
    buf: wp.array[float],
    n: int,
):
    i = wp.tid()
    b3 = i * 3
    v = nx[i]
    buf[b3] = v[0]
    buf[b3 + 1] = v[1]
    buf[b3 + 2] = v[2]
    v = nxd[i]
    buf[3 * n + b3] = v[0]
    buf[3 * n + b3 + 1] = v[1]
    buf[3 * n + b3 + 2] = v[2]
    v = nxdd[i]
    buf[6 * n + b3] = v[0]
    buf[6 * n + b3 + 1] = v[1]
    buf[6 * n + b3 + 2] = v[2]
    v = nd[i]
    buf[9 * n + b3] = v[0]
    buf[9 * n + b3 + 1] = v[1]
    buf[9 * n + b3 + 2] = v[2]
    v = ndd[i]
    buf[12 * n + b3] = v[0]
    buf[12 * n + b3 + 1] = v[1]
    buf[12 * n + b3 + 2] = v[2]
    v = nddd[i]
    buf[15 * n + b3] = v[0]
    buf[15 * n + b3 + 1] = v[1]
    buf[15 * n + b3 + 2] = v[2]
    base_f = 18 * n + i * 6
    base_f0 = 24 * n + i * 6
    base_s = i * 6
    buf[base_f] = fint[base_s]
    buf[base_f + 1] = fint[base_s + 1]
    buf[base_f + 2] = fint[base_s + 2]
    buf[base_f + 3] = fint[base_s + 3]
    buf[base_f + 4] = fint[base_s + 4]
    buf[base_f + 5] = fint[base_s + 5]
    buf[base_f0] = fint0[base_s]
    buf[base_f0 + 1] = fint0[base_s + 1]
    buf[base_f0 + 2] = fint0[base_s + 2]
    buf[base_f0 + 3] = fint0[base_s + 3]
    buf[base_f0 + 4] = fint0[base_s + 4]
    buf[base_f0 + 5] = fint0[base_s + 5]


@wp.kernel
def _unpack_ancf(
    buf: wp.array[float],
    nx: wp.array[wp.vec3],
    nxd: wp.array[wp.vec3],
    nxdd: wp.array[wp.vec3],
    nd: wp.array[wp.vec3],
    ndd: wp.array[wp.vec3],
    nddd: wp.array[wp.vec3],
    fint: wp.array[float],
    fint0: wp.array[float],
    n: int,
):
    i = wp.tid()
    b3 = i * 3
    nx[i] = wp.vec3(buf[b3], buf[b3 + 1], buf[b3 + 2])
    nxd[i] = wp.vec3(buf[3 * n + b3], buf[3 * n + b3 + 1], buf[3 * n + b3 + 2])
    nxdd[i] = wp.vec3(buf[6 * n + b3], buf[6 * n + b3 + 1], buf[6 * n + b3 + 2])
    nd[i] = wp.vec3(buf[9 * n + b3], buf[9 * n + b3 + 1], buf[9 * n + b3 + 2])
    ndd[i] = wp.vec3(buf[12 * n + b3], buf[12 * n + b3 + 1], buf[12 * n + b3 + 2])
    nddd[i] = wp.vec3(buf[15 * n + b3], buf[15 * n + b3 + 1], buf[15 * n + b3 + 2])
    base_f = 18 * n + i * 6
    base_f0 = 24 * n + i * 6
    base_s = i * 6
    fint[base_s] = buf[base_f]
    fint[base_s + 1] = buf[base_f + 1]
    fint[base_s + 2] = buf[base_f + 2]
    fint[base_s + 3] = buf[base_f + 3]
    fint[base_s + 4] = buf[base_f + 4]
    fint[base_s + 5] = buf[base_f + 5]
    fint0[base_s] = buf[base_f0]
    fint0[base_s + 1] = buf[base_f0 + 1]
    fint0[base_s + 2] = buf[base_f0 + 2]
    fint0[base_s + 3] = buf[base_f0 + 3]
    fint0[base_s + 4] = buf[base_f0 + 4]
    fint0[base_s + 5] = buf[base_f0 + 5]


class InterfaceCouplerGS:
    """Interface Gauss-Seidel (IGS) coupler for FEM + rigid-body split solvers.

    Eliminates the one-step explicit coupling lag that causes instability at
    vehicle loads by iterating both solvers within each substep until the
    shared interface DOFs converge.

    The ANCF solver's internal NR+PCG loop still runs as a captured CUDA graph.
    The outer GS loop runs in Python.  When *tol* is ``0.0`` (default), all
    *n_iters* iterations run unconditionally — no GPU→CPU synchronisation per
    substep.

    **Double-buffer layout** — two pre-allocated ANCF pack buffers (A and B)
    alternate roles each substep via an integer flip.  The *pre-save* (packing
    the end-of-substep ANCF state into the inactive buffer) happens at the
    **end** of each substep, so the next substep's restore can start
    immediately without a save copy in the hot path.  This hides the save
    latency behind the tail of the previous substep.

    ANCF restore uses a single :func:`_unpack_ancf` kernel dispatch instead of
    eight separate ``wp.copy`` calls, reducing CUDA kernel launch overhead by
    8× per GS iteration.

    Extra arrays registered via *extra_arrays* are saved and restored only if
    they are modified during GS iterations.  Arrays that are read-only during
    GS (e.g. accumulated rim-angle) should **not** be registered — do not
    pass them as *extra_arrays*.

    Key k=0 optimisation: on the first GS iteration the Newton State is
    already at *t_n*, so the first :func:`kinematics_fn` call simultaneously
    fills ``solver.xpos`` for bead prescription **and** prepares the MuJoCo
    internal state for :func:`dynamics_fn`.  The redundant second kinematics
    call is skipped on *k=0*, saving one full kinematics evaluation per
    substep.

    Args:
        ancf_solver: A :class:`~newton.solvers.SolverANCFShell` instance.
        n_iters: Number of GS coupling iterations per substep.
            ``1`` is equivalent to the explicit scheme.  ``2`` eliminates the
            leading-order lag and is sufficient for FEDA-class loads.
        tol: Convergence tolerance for spindle position change [m].  ``0.0``
            (default) disables early exit, avoiding any GPU→CPU sync per substep.
    """

    def __init__(self, ancf_solver, n_iters: int = 2, tol: float = 0.0) -> None:
        self._ancf = ancf_solver
        self._n_iters = n_iters
        self._tol = tol
        self._n_nodes = None  # set by allocate()

    # ── allocation ──────────────────────────────────────────────────────────

    def allocate(
        self,
        state_0,
        extra_arrays: list[wp.array] | None = None,
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
        self._rigid_cur = 0
        # Prime rigid buffer[0] with initial state.
        self._save_rigid(state_0, self._rigid_cur)

        # ── Extra application arrays (truly read-write during GS) ─────────
        self._extra_live = list(extra_arrays or [])
        self._extra_snaps = [wp.clone(arr) for arr in self._extra_live]

    # ── pack / unpack helpers ────────────────────────────────────────────────

    def _pack_ancf(self, buf_idx: int) -> None:
        """Pack live ANCF state into pack buffer *buf_idx* (1 kernel dispatch)."""
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
                self._ancf_pack[buf_idx],
                self._n_nodes,
            ],
        )

    def _unpack_ancf(self, buf_idx: int) -> None:
        """Unpack ANCF pack buffer *buf_idx* into live ANCF arrays (1 kernel)."""
        a = self._ancf
        wp.launch(
            _unpack_ancf,
            dim=self._n_nodes,
            inputs=[
                self._ancf_pack[buf_idx],
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

        Returns:
            Number of GS iterations actually executed (≤ *n_iters*).
        """
        if self._n_nodes is None:
            raise RuntimeError("InterfaceCouplerGS: call allocate() before substep().")

        sp_prev: np.ndarray | None = None

        for k in range(self._n_iters):
            # ── restore ANCF to t_n (1 kernel, not 7 copies) ──────────────
            self._unpack_ancf(self._ancf_cur)

            # ── restore extra read-write arrays (if any) ──────────────────
            if self._extra_live:
                self._restore_extra()

            # ── kinematics from state_0 (= x_sp^k) ────────────────────────
            # k=0: state_0 IS t_n → single call serves bead prescription AND
            #      mjData initialisation for step_dynamics (no second call needed).
            # k>0: second call from restored t_n runs after bead work (below).
            kinematics_fn(state_0, state_rigid, control, None, dt)

            # ── pre-step bead prescription ────────────────────────────────
            prescribe_fn()

            # ── ANCF FEM step (internal CUDA graph) ───────────────────────
            self._ancf.graph_step()

            # ── post-step bead snap ────────────────────────────────────────
            prescribe_fn()

            # ── restore rigid to t_n + re-run kinematics (k > 0 only) ─────
            # Skipped on k=0: state_0 still equals t_n, mjData still valid.
            #
            # NOTE: accumulate_fn() is intentionally placed AFTER this block.
            # step_kinematics calls _apply_mjc_control which launches
            # apply_mjc_body_f_kernel over all bodies, writing state.body_f
            # (typically zeros) to xfrc_applied and clearing any forces that
            # accumulate_fn() may have written.  Accumulating after kinematics
            # ensures the forces survive into dynamics.
            if k > 0:
                self._restore_rigid(state_0)
                kinematics_fn(state_0, state_rigid, control, None, dt)

            # ── accumulate ANCF wrenches → solver.xfrc_applied ───────────
            accumulate_fn()

            # ── MuJoCo dynamics ────────────────────────────────────────────
            dynamics_fn(state_rigid)

            # ── propagate rigid → state_0 (= x_sp^{k+1}) ─────────────────
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

        # ── pre-save t_{n+1} into inactive buffer (free for next substep) ─
        self._presave(state_0)
        return self._n_iters
