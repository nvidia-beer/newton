# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
SolverANCFShell — implicit HHT time integrator for ANCF3423 shell elements.

Integration scheme: Hilbert-Hughes-Taylor (HHT-alpha), parameters:
    alpha = -0.2  (spectral damping, matches Chrono demo)
    beta  = (1 - alpha)^2 / 4 = 0.36
    gamma = (1 - 2*alpha)  / 2 = 0.70

HHT residual per Newton-Raphson iteration:
    R = M·a_{n+1} + (1+α)·f_int(x_{n+1}) − α·f_int(x_n) − f_ext = 0

Kinematic update from current acceleration a = q̈_{n+1}:
    x_{n+1} = x_pred + β·dt²·a
    v_{n+1} = v_pred + γ·dt·a
    x_pred  = x_n + dt·v_n + dt²·(0.5-β)·a_n
    v_pred  = v_n + dt·(1-γ)·a_n

Effective stiffness:
    K_eff = (1/(β·dt²))·M + (1+α)·K_t

Notes:
    - ANCF state (node_x, node_D, velocities, accelerations) is managed
      internally by the solver, not stored in Newton's State object.
    - The Newton Model/State/Control objects are accepted by step() to comply
      with SolverBase, but only gravity is read from the model.
    - No rigid-body hub coupling in phase 1.
"""

import math
import sys
import threading
import time

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels_assembly import (
    accumulate_cavity_volume,
    accumulate_centroid,
    add_diag_to_bsr_values_batched,
    assemble_sparse_stiffness,
    build_scatter_map,
    cavity_gas_law,
    compute_pressure_force_stiffness,
    compute_pressure_force_stiffness_batched_gp,
    scatter_elem_to_bsr,
    scatter_elem_to_bsr_batched,
    scatter_forces,
    scatter_forces_batched,
    zero_bsr_values,
)
from .kernels_contact import (
    add_diag_to_bsr_values,
    apply_ground_contact,
    apply_ground_contact_batched,
    apply_rim_contact,
    apply_rim_contact_batched,
    zero_contact_diag,
)
from .kernels_element import compute_lumped_mass
from .kernels_stiffness import (
    _eas_solve_damping_batched,
    _zero_eas_and_accum_batched,
    compute_element_forces_stiffness,
    compute_element_forces_stiffness_batched_gp,
    compute_element_K_from_B,
    compute_rest_jacobians,
)
from .model_ancf_shell import ANCFShellModel

# ---------------------------------------------------------------------------
# PCG linear solver
#
# PcgSolverBatched solves N independent systems sharing one sparsity pattern.
# Per-iteration kernel sequence:
#   1. _bsr_mv_flat_batched                     — Ap=K·p  (N*n_dof threads)
#   2. pAp.zero_() + _dot_tile_batched          — pAp=p·Ap (N*n_chunks blocks, tile_atomic_add)
#   3. rz_new.zero_() + _pcg_update_xrz_tile   — x,r,z update + rz_new (N*n_chunks blocks)
#   4. _pcg_update_p_tile_batched               — p update + rz_old swap (N*n_chunks blocks)
#
# n_chunks = n_dof_pad/128: all tile launches use N*n_chunks blocks (fills all SMs)
# instead of the old dim=[N] that left N−1 of N SMs idle.
# ---------------------------------------------------------------------------

TILE_PCG = wp.constant(128)  # block size for tile-based PCG kernels


@wp.func
def _pcg_precond_safe(r: float, d: float) -> float:
    """Jacobi preconditioner: z = D^-1 r, safe for zero diagonal."""
    return r / d if wp.abs(d) > 1.0e-30 else r


@wp.kernel
def _apply_diag_precond(
    diag: wp.array[float],
    r: wp.array[float],
    z: wp.array[float],
):
    """z = D^-1 * r.  Used by PcgSolverBatched."""
    i = wp.tid()
    d = diag[i]
    z[i] = r[i] / d if wp.abs(d) > 1.0e-30 else r[i]


# ---------------------------------------------------------------------------
# Batched kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _extract_diag_batched(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[float],
    diag: wp.array[float],
    n_dof: int,
    nnz: int,
    n_dof_pad: int,
):
    """dim = N*n_dof. Extracts diagonal into padded diag array."""
    tid = wp.tid()
    env = tid // n_dof
    i = tid % n_dof
    base = env * nnz
    diag_idx = env * n_dof_pad + i
    for ptr in range(offsets[i], offsets[i + 1]):
        if columns[ptr] == i:
            diag[diag_idx] = values[base + ptr]
            return
    diag[diag_idx] = float(0.0)


# ---------------------------------------------------------------------------
# Tile-based + fused PCG kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _pcg_load_rhs(b: wp.array[float], r: wp.array[float], n_dof: int, n_dof_pad: int):
    """dim = N*n_dof. Copy b (unpadded) into r (padded) at correct positions."""
    tid = wp.tid()
    env = tid // n_dof
    i = tid % n_dof
    r[env * n_dof_pad + i] = b[tid]


@wp.kernel
def _pcg_store_x(x_pad: wp.array[float], x_ext: wp.array[float], n_dof: int, n_dof_pad: int):
    """dim = N*n_dof. Copy x (padded internal) back to x_ext (unpadded external)."""
    tid = wp.tid()
    env = tid // n_dof
    i = tid % n_dof
    x_ext[tid] = x_pad[env * n_dof_pad + i]


@wp.kernel
def _dot_tile_batched(
    a: wp.array[float],
    b: wp.array[float],
    out: wp.array[float],
    n_dof_pad: int,
    n_chunks: int,
):
    """One block per (env, chunk). wp.launch_tiled(dim=[N*n_chunks], block_dim=TILE_PCG).

    Replaces the single-block-per-env design (dim=[N]) that left N-1 of N SMs idle.
    Each block processes TILE_PCG=128 elements and atomically accumulates into out[env].
    Caller must zero out[] before launch.
    """
    tid = wp.tid()  # block index 0..N*n_chunks-1
    env = tid // n_chunks
    chunk = tid % n_chunks
    off = env * n_dof_pad + chunk * TILE_PCG

    ta = wp.tile_load(a, shape=TILE_PCG, offset=off)
    tb = wp.tile_load(b, shape=TILE_PCG, offset=off)
    s = wp.tile_sum(ta * tb)
    wp.tile_atomic_add(out, s, offset=env)


@wp.kernel
def _bsr_mv_flat_batched(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[float],
    x: wp.array[float],
    y: wp.array[float],
    n_dof: int,
    nnz: int,
    n_dof_pad: int,
):
    """dim = N*n_dof.  y[env, i] = A_env · x[env, i].  Padded stride n_dof_pad per env."""
    tid = wp.tid()
    env = tid // n_dof
    i = tid % n_dof
    base_dof = env * n_dof_pad
    base_nnz = env * nnz
    acc = float(0.0)
    for ptr in range(offsets[i], offsets[i + 1]):
        acc = acc + values[base_nnz + ptr] * x[base_dof + columns[ptr]]
    y[base_dof + i] = acc


@wp.kernel
def _pcg_update_xrz_tile_batched(
    rz_old: wp.array[float],
    pAp: wp.array[float],
    p: wp.array[float],
    Ap: wp.array[float],
    x: wp.array[float],
    r: wp.array[float],
    diag: wp.array[float],
    z: wp.array[float],
    rz_new: wp.array[float],  # accumulator — must be zeroed before launch
    n_dof_pad: int,
    n_chunks: int,
):
    """One block per (env, chunk). wp.launch_tiled(dim=[N*n_chunks], block_dim=TILE_PCG).

    Updates x, r, z for one tile and accumulates rz_new atomically.
    x/r/z tile_stores are conflict-free (different blocks own different offsets).
    rz_new uses tile_atomic_add; caller must zero rz_new[] before each launch.
    """
    tid = wp.tid()
    env = tid // n_chunks
    chunk = tid % n_chunks
    off = env * n_dof_pad + chunk * TILE_PCG

    rz_old_val = rz_old[env]
    pAp_val = pAp[env]
    alpha = rz_old_val / pAp_val if wp.abs(pAp_val) > 1.0e-30 else float(0.0)

    tx = wp.tile_load(x, shape=TILE_PCG, offset=off)
    tp = wp.tile_load(p, shape=TILE_PCG, offset=off)
    tap = wp.tile_load(Ap, shape=TILE_PCG, offset=off)
    tr = wp.tile_load(r, shape=TILE_PCG, offset=off)
    td = wp.tile_load(diag, shape=TILE_PCG, offset=off)

    tr_new = tr - alpha * tap
    tz_new = wp.tile_map(_pcg_precond_safe, tr_new, td)

    wp.tile_store(x, tx + alpha * tp, offset=off)
    wp.tile_store(r, tr_new, offset=off)
    wp.tile_store(z, tz_new, offset=off)

    s = wp.tile_sum(tr_new * tz_new)
    wp.tile_atomic_add(rz_new, s, offset=env)


@wp.kernel
def _pcg_update_p_tile_batched(
    rz_new: wp.array[float],  # written by previous kernel (via tile_atomic_add)
    rz_old: wp.array[float],  # in/out: read for beta, written with rz_new
    z: wp.array[float],
    p: wp.array[float],
    n_dof_pad: int,
    n_chunks: int,
):
    """One block per (env, chunk). wp.launch_tiled(dim=[N*n_chunks], block_dim=TILE_PCG).

    Reads rz_new (fully written by previous kernel — CUDA graph node boundary).
    Read-only on both scalars: the caller swaps the rz_old/rz_new buffers
    between iterations.  (An in-kernel ``rz_old[env] = rz_new`` write raced
    with the read in sibling blocks of the same env, giving beta = 1 for
    whichever chunks ran late and breaking conjugacy.)
    """
    tid = wp.tid()
    env = tid // n_chunks
    chunk = tid % n_chunks
    off = env * n_dof_pad + chunk * TILE_PCG

    rz_new_val = rz_new[env]
    rz_old_val = rz_old[env]
    beta = rz_new_val / rz_old_val if wp.abs(rz_old_val) > 1.0e-30 else float(0.0)

    tz = wp.tile_load(z, shape=TILE_PCG, offset=off)
    tp = wp.tile_load(p, shape=TILE_PCG, offset=off)
    wp.tile_store(p, tz + beta * tp, offset=off)


class PcgSolverBatched:
    """PCG for N independent ANCF systems sharing the same sparsity pattern.

    All vector arguments to solve() are flat [N*n_dof].
    K values are flat [N*nnz].  Offsets and columns are shared.
    """

    def __init__(self, n_envs: int, n_dof: int, nnz: int, device: str, max_iters: int = 200):
        self.n_envs = n_envs
        self.n_dof = n_dof
        self.nnz = nnz
        self.device = device
        self.max_iters = max_iters
        N = n_envs
        # Pad n_dof to next multiple of TILE_PCG (128) for tile kernels.
        n_dof_pad = int(math.ceil(n_dof / 128) * 128)
        self._n_dof_pad = n_dof_pad
        # n_chunks: blocks per env in multi-block tile launches (N*n_chunks total blocks).
        # Each block handles exactly TILE_PCG=128 DOFs → fills all SMs vs old dim=[N].
        self._n_chunks = n_dof_pad // int(TILE_PCG)
        self.r = wp.zeros(N * n_dof_pad, dtype=float, device=device)
        self.z = wp.zeros(N * n_dof_pad, dtype=float, device=device)
        self.p = wp.zeros(N * n_dof_pad, dtype=float, device=device)
        self.Ap = wp.zeros(N * n_dof_pad, dtype=float, device=device)
        self.diag = wp.zeros(N * n_dof_pad, dtype=float, device=device)
        self.rz_old = wp.zeros(N, dtype=float, device=device)
        self.rz_new = wp.zeros(N, dtype=float, device=device)
        # Initial ||r||^2_M, kept so convergence can be reported after a solve.
        self.rz_init = wp.zeros(N, dtype=float, device=device)
        self.pAp = wp.zeros(N, dtype=float, device=device)
        self.rz_last = self.rz_old
        self._x_pad = wp.zeros(N * n_dof_pad, dtype=float, device=device)

    def _dot(self, a: wp.array, b: wp.array, out: wp.array) -> None:
        """Multi-block dot product. out[] must be zeroed by caller (or use _dot_zeroed)."""
        n_ch = self._n_chunks
        wp.launch_tiled(
            _dot_tile_batched,
            dim=[self.n_envs * n_ch],
            inputs=[a, b, out, self._n_dof_pad, n_ch],
            block_dim=int(TILE_PCG),
            device=self.device,
        )

    def solve(self, K_offsets: wp.array, K_columns: wp.array, K_values: wp.array, b: wp.array, x: wp.array) -> None:
        dev, N, n, n_pad = self.device, self.n_envs, self.n_dof, self._n_dof_pad
        n_ch, nnz = self._n_chunks, self.nnz

        # Setup: extract Jacobi diagonal preconditioner
        self.diag.zero_()
        wp.launch(
            _extract_diag_batched,
            dim=N * n,
            inputs=[K_offsets, K_columns, K_values, self.diag, n, nnz, n_pad],
            device=dev,
        )

        # Load RHS, zero solution, z = D⁻¹r, p = z, rz_old = r·z
        self.r.zero_()
        self._x_pad.zero_()
        wp.launch(_pcg_load_rhs, dim=N * n, inputs=[b, self.r, n, n_pad], device=dev)
        wp.launch(_apply_diag_precond, dim=N * n_pad, inputs=[self.diag, self.r, self.z], device=dev)
        wp.copy(self.p, self.z)
        self.rz_old.zero_()
        self._dot(self.r, self.z, self.rz_old)
        wp.copy(self.rz_init, self.rz_old)

        # The two r·z scalars ping-pong between iterations (Python-side swap of
        # the array references, no device work).  Both kernels below are
        # read-only on rz_old, so no intra-kernel race is possible.
        rz_old, rz_new = self.rz_old, self.rz_new
        for _ in range(self.max_iters):
            # SpMV: Ap = K·p
            wp.launch(
                _bsr_mv_flat_batched,
                dim=N * n,
                inputs=[K_offsets, K_columns, K_values, self.p, self.Ap, n, nnz, n_pad],
                device=dev,
            )
            # pAp = p·Ap  (multi-block: N*n_chunks blocks vs old N blocks → all SMs busy)
            # _dot_tile_batched uses tile_atomic_add so rz_new must be zeroed first.
            self.pAp.zero_()
            self._dot(self.p, self.Ap, self.pAp)
            # x += α·p,  r -= α·Ap,  z = D⁻¹r,  rz_new = Σ r·z  (multi-block tile)
            rz_new.zero_()
            wp.launch_tiled(
                _pcg_update_xrz_tile_batched,
                dim=[N * n_ch],
                inputs=[
                    rz_old,
                    self.pAp,
                    self.p,
                    self.Ap,
                    self._x_pad,
                    self.r,
                    self.diag,
                    self.z,
                    rz_new,
                    n_pad,
                    n_ch,
                ],
                block_dim=int(TILE_PCG),
                device=dev,
            )
            # p = z + β·p   (β = rz_new / rz_old)
            wp.launch_tiled(
                _pcg_update_p_tile_batched,
                dim=[N * n_ch],
                inputs=[rz_new, rz_old, self.z, self.p, n_pad, n_ch],
                block_dim=int(TILE_PCG),
                device=dev,
            )
            rz_old, rz_new = rz_new, rz_old
        # Whichever buffer holds the last r·z after the final swap (for residual_report).
        self.rz_last = rz_old

        wp.launch(_pcg_store_x, dim=N * n, inputs=[self._x_pad, x, n, n_pad], device=dev)


# ---------------------------------------------------------------------------
# Progress spinner for long JIT compilations
# ---------------------------------------------------------------------------


class _CompileProgress:
    """Context manager that shows an ASCII spinner + elapsed time on one line.

    Usage::
        with _CompileProgress("Building mass matrix"):
            wp.launch(kernel, ...)
            wp.synchronize_device(dev)
    """

    _FRAMES = r"|/-\\"

    def __init__(self, label: str, interval: float = 0.15):
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        elapsed = time.perf_counter() - self._t0
        sys.stdout.write(f"\r  [done]  {self._label}  ({elapsed:.1f}s)\n")
        sys.stdout.flush()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._t0
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r  [{frame}]  {self._label}  {elapsed:.0f}s elapsed ...")
            sys.stdout.flush()
            time.sleep(self._interval)
            i += 1


# ---------------------------------------------------------------------------
# HHT parameters
# ---------------------------------------------------------------------------
_HHT_ALPHA = -0.2
_HHT_BETA = (1.0 - _HHT_ALPHA) ** 2 / 4.0  # 0.36
_HHT_GAMMA = (1.0 - 2.0 * _HHT_ALPHA) / 2.0  # 0.70


# ---------------------------------------------------------------------------
# State integration kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _hht_predict(
    x_n: wp.array[wp.vec3],
    xd_n: wp.array[wp.vec3],
    xdd_n: wp.array[wp.vec3],
    dt: float,
    beta: float,
    gamma: float,
    # outputs
    x_pred: wp.array[wp.vec3],
    xd_pred: wp.array[wp.vec3],
):
    """Newmark predictor step."""
    i = wp.tid()
    x_pred[i] = x_n[i] + dt * xd_n[i] + (dt * dt * (0.5 - beta)) * xdd_n[i]
    xd_pred[i] = xd_n[i] + (dt * (1.0 - gamma)) * xdd_n[i]


@wp.kernel
def _hht_kinematic(
    x_pred: wp.array[wp.vec3],
    xd_pred: wp.array[wp.vec3],
    xdd_new: wp.array[wp.vec3],  # current NR iterate a_{n+1}
    dt: float,
    beta: float,
    gamma: float,
    # outputs
    x_new: wp.array[wp.vec3],
    xd_new: wp.array[wp.vec3],
):
    """Compute x_{n+1} and v_{n+1} from a_{n+1}."""
    i = wp.tid()
    x_new[i] = x_pred[i] + (beta * dt * dt) * xdd_new[i]
    xd_new[i] = xd_pred[i] + (gamma * dt) * xdd_new[i]


@wp.kernel
def _apply_acceleration_update(
    da_flat: wp.array[float],  # (n_nodes*6,) displacement correction Δu from PCG
    inv_bdt2: float,  # 1/(β·dt²) — converts Δu [m] → Δa [m/s²]
    node_xdd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
):
    """Convert the displacement-formulation solve result to an acceleration increment.

    K_eff = M/(β·dt²) + K_t uses the displacement formulation, so the linear
    solve returns Δu [m]; the acceleration correction is Δa = Δu/(β·dt²).
    """
    i = wp.tid()
    base = i * 6
    du = wp.vec3(da_flat[base], da_flat[base + 1], da_flat[base + 2])
    dD = wp.vec3(da_flat[base + 3], da_flat[base + 4], da_flat[base + 5])
    node_xdd[i] = node_xdd[i] + inv_bdt2 * du
    node_Ddd[i] = node_Ddd[i] + inv_bdt2 * dD


@wp.kernel
def _zero_dirichlet_acc(
    dirichlet_idx: wp.array[wp.int32],
    node_xdd: wp.array[wp.vec3],
    node_Ddd: wp.array[wp.vec3],
):
    """Zero position and director accelerations for pinned (Dirichlet) nodes.

    Called once per NR iteration after _apply_acceleration_update so the HHT
    kinematic update never displaces pinned nodes away from their predictor value.
    """
    i = wp.tid()
    idx = dirichlet_idx[i]
    node_xdd[idx] = wp.vec3(0.0, 0.0, 0.0)
    node_Ddd[idx] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _dirichlet_pred_override(
    dirichlet_idx: wp.array[wp.int32],
    node_x: wp.array[wp.vec3],
    node_xd: wp.array[wp.vec3],
    node_D: wp.array[wp.vec3],
    node_Dd: wp.array[wp.vec3],
    x_pred: wp.array[wp.vec3],
    xd_pred: wp.array[wp.vec3],
    D_pred: wp.array[wp.vec3],
    Dd_pred: wp.array[wp.vec3],
):
    """Pin the Newmark predictor to the prescribed state at Dirichlet nodes.

    ``_hht_predict`` advances every node by ``dt·xd``; with the acceleration
    zeroed at pinned nodes that predictor value *is* their end-of-step state,
    one substep ahead of what the caller prescribed (double-counting any
    hub-velocity extrapolation the caller already applied).  Overriding the
    predictor makes the prescribed ``node_x/xd/D/Dd`` the exact state the
    pinned nodes hold throughout the solve.
    """
    i = wp.tid()
    idx = dirichlet_idx[i]
    x_pred[idx] = node_x[idx]
    xd_pred[idx] = node_xd[idx]
    D_pred[idx] = node_D[idx]
    Dd_pred[idx] = node_Dd[idx]


@wp.kernel
def _build_dirichlet_dof_mask(dirichlet_idx: wp.array[wp.int32], dof_mask: wp.array[float]):
    """dim = n_dirichlet.  Zero the 6 DOF-mask entries of each pinned node (mask pre-filled with 1)."""
    i = wp.tid()
    base = dirichlet_idx[i] * 6
    for k in range(6):
        dof_mask[base + k] = float(0.0)


@wp.kernel
def _build_dirichlet_nnz_mask(
    dof_mask: wp.array[float],  # (N*n_dof,)
    bsr_offsets: wp.array[wp.int32],  # (n_dof+1,) shared CSR pattern
    bsr_columns: wp.array[wp.int32],  # capacity-sized; only offsets[-1] entries valid
    n_dof: int,
    nnz_cap: int,  # per-env stride of bsr_values_batched (allocation capacity)
    nnz_mask: wp.array[float],  # (N*nnz_cap,) pre-filled with 1
):
    """dim = N*n_dof.  +1 keep, 0 pinned row or column, -1 pinned diagonal."""
    tid = wp.tid()
    env = tid // n_dof
    r = tid % n_dof
    dof_base = env * n_dof
    mr = dof_mask[dof_base + r]
    for ptr in range(bsr_offsets[r], bsr_offsets[r + 1]):
        c = bsr_columns[ptr]
        v = mr * dof_mask[dof_base + c]
        if r == c and mr == 0.0:
            v = float(-1.0)
        nnz_mask[env * nnz_cap + ptr] = v


@wp.kernel
def _mask_dof(mask: wp.array[float], arr: wp.array[float]):
    """arr[i] *= mask[i] — zero the Dirichlet rows of a DOF vector."""
    i = wp.tid()
    arr[i] = arr[i] * mask[i]


@wp.kernel
def _apply_dirichlet_to_bsr(mask_nnz: wp.array[float], values: wp.array[float]):
    """Eliminate Dirichlet rows/columns from K_eff in place.

    mask_nnz: +1 keep, 0 zero (row or column pinned), -1 pinned diagonal → 1.
    With the matching residual rows zeroed the pinned DOFs decouple exactly and
    the linear solve returns 0 for them; the free DOFs no longer receive the
    constraint reaction through the off-diagonals.
    """
    i = wp.tid()
    m = mask_nnz[i]
    v = values[i]
    if m < 0.0:
        v = float(1.0)
    elif m == 0.0:
        v = float(0.0)
    values[i] = v


@wp.kernel
def _assemble_rhs_gravity(
    mass_flat: wp.array[float],  # lumped mass per DOF (n_nodes*6,)
    gravity: wp.vec3,
    rhs: wp.array[float],  # output: add gravity to rhs
):
    """Add M_lump * g to the right-hand-side force vector."""
    i = wp.tid()
    # gravity acts only on position DOFs (sub 0=x, 1=y, 2=z)
    sub = i % 6
    if sub == 0:
        rhs[i] = rhs[i] + mass_flat[i] * gravity[0]
    elif sub == 1:
        rhs[i] = rhs[i] + mass_flat[i] * gravity[1]
    elif sub == 2:
        rhs[i] = rhs[i] + mass_flat[i] * gravity[2]


@wp.kernel
def _add_node_f_ext_persistent(
    f_pers: wp.array[wp.vec3],  # per-node persistent forces [N * n_nodes]
    global_f_ext: wp.array[float],  # (N * n_nodes * 6,) flat DOF vector
):
    """Add persistent per-node forces to global_f_ext after per-NR zero_().

    Allows external contact forces computed once per substep (from committed
    positions) to survive the per-NR-iteration global_f_ext.zero_() — identical
    to how FEM eval_particle_ground_contacts writes to the constant RHS before
    the CG solve.  One thread per node.
    """
    i = wp.tid()
    base = i * 6
    f = f_pers[i]
    wp.atomic_add(global_f_ext, base + 0, f[0])
    wp.atomic_add(global_f_ext, base + 1, f[1])
    wp.atomic_add(global_f_ext, base + 2, f[2])


@wp.kernel
def _zero_nr_slot(out: wp.array[float], nr_slot: int, nr_max: int):
    """Zero one NR history slot per env (diagnostics only)."""
    e = wp.tid()
    out[e * nr_max + nr_slot] = 0.0


@wp.kernel
def _accum_nr_residual_sq(
    residual: wp.array[float],
    out: wp.array[float],
    nr_slot: int,
    nr_max: int,
    n_dof: int,
):
    """Accumulate ||R||^2 per env into an NR history slot (diagnostics only).

    Writes only to ``out``; the solve is untouched.
    """
    i = wp.tid()
    e = i // n_dof
    wp.atomic_add(out, e * nr_max + nr_slot, residual[i] * residual[i])


@wp.kernel
def _build_residual(
    # M · a_{n+1}  (mass-weighted acceleration)
    M_a: wp.array[float],  # (n_dof,) M·a_new
    # (1+alpha)*f_int(x_{n+1})
    f_int_new: wp.array[float],  # (n_dof,)
    # alpha * f_int(x_n)
    f_int_old: wp.array[float],  # (n_dof,)
    # external forces (gravity + contact)
    f_ext: wp.array[float],  # (n_dof,)
    alpha: float,
    # output
    residual: wp.array[float],  # (n_dof,)
):
    """R = M·a + (1+α)·f_int_new − α·f_int_old − f_ext."""
    i = wp.tid()
    residual[i] = M_a[i] + (1.0 + alpha) * f_int_new[i] - alpha * f_int_old[i] - f_ext[i]


@wp.kernel
def _flatten_vec3_pair(
    xdd: wp.array[wp.vec3],
    Ddd: wp.array[wp.vec3],
    flat: wp.array[float],  # (n_nodes*6,)
):
    """Pack (xdd, Ddd) into a flat DOF vector."""
    i = wp.tid()
    base = i * 6
    flat[base] = xdd[i][0]
    flat[base + 1] = xdd[i][1]
    flat[base + 2] = xdd[i][2]
    flat[base + 3] = Ddd[i][0]
    flat[base + 4] = Ddd[i][1]
    flat[base + 5] = Ddd[i][2]


@wp.kernel
def _zero_eas_alpha(eas_alpha: wp.array2d[float]):
    """Reset all 5 EAS modes to zero for one element.

    Called once per timestep, before the NR loop.  Prevents cross-timestep
    accumulation of float-32 roundoff: for a rigid-body translation J_lifted ≠
    J_ref at the ~1e-6 level (1.3 m vs 0.3 m positions), so HE ≠ 0 exactly and
    α drifts by ~3e-6 per NR call.  Over 100 NR calls (2 frames × 10 substeps
    × 5 NR) α reaches ~3e-4, creating asymmetric forces that violate the
    discrete rotational symmetry and eventually cause NaN.

    Within-timestep warm-start (across the 5 NR iterations) is preserved.
    """
    e = wp.tid()
    eas_alpha[e, 0] = float(0.0)
    eas_alpha[e, 1] = float(0.0)
    eas_alpha[e, 2] = float(0.0)
    eas_alpha[e, 3] = float(0.0)
    eas_alpha[e, 4] = float(0.0)


@wp.kernel
def _negate(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    dst[i] = -src[i]


@wp.kernel
def _pointwise_scale(src: wp.array[float], s: float, dst: wp.array[float]):
    i = wp.tid()
    dst[i] = src[i] * s


# ---------------------------------------------------------------------------
# SolverANCFShell
# ---------------------------------------------------------------------------


class SolverANCFShell(SolverBase):
    """Implicit HHT-alpha integrator for ANCF3423 shell elements.

    Args:
        model:       Newton Model (used for device and gravity).
        ancf_model:  :class:`~newton._src.solvers.ancf_shell.ANCFShellModel`
                     holding the mesh and material data.
        ground_z:    Height of the rigid ground plane [m] (default 0.0).
        kn:          Normal contact stiffness [N/m] (default 2e6).
        kd:          Normal contact damping [N·s/m] (default 13.0).
        mu:          Coulomb friction coefficient (default 0.9).
        v_reg:       Friction regularisation velocity [m/s] (default 1e-3).
        nr_max_iter: Max Newton-Raphson iterations per step (default 5).
        nr_tol:      Newton-Raphson residual tolerance (default 1e-2).
        pcg_max_iter: Max PCG iterations per NR step (default 200).
        rim_radius:  Inner rim cylinder radius [m].  0.0 = no rim contact.
        rim_kn:      Rim contact normal stiffness [N/m] (default 2e6).
        rim_kd:      Rim contact normal damping [N·s/m] (default 13.0).
        thickness_gp: Through-thickness Gauss points, 3 or 5 (default 3).
    """

    def __init__(
        self,
        model: Model,
        ancf_model: ANCFShellModel,
        ground_z: float = 0.0,
        kn: float = 2.0e6,
        kd: float = 13.0,
        mu: float = 0.9,
        v_reg: float = 1.0e-3,
        nr_max_iter: int = 3,
        pcg_max_iter: int = 200,
        n_envs: int = 1,
        rim_radius: float = 0.0,
        rim_kn: float = 2.0e6,
        rim_kd: float = 13.0,
        thickness_gp: int = 3,
    ):
        super().__init__(model)
        self.ancf = ancf_model
        self.n_envs = n_envs

        self.ground_z = ground_z
        self.kn = kn
        # Diagnostics: when True (set BEFORE capture_graph) each NR iteration
        # records ||R||^2 into _nr_res_hist.  Purely additive - no solve change.
        self.debug_residuals = False
        self.kd = kd
        self.mu = mu
        self.v_reg = v_reg
        self.nr_max_iter = nr_max_iter
        self.thickness_gp = 3 if thickness_gp == 3 else 5

        # When True, _step_batched zeros the NR acceleration update at
        # Dirichlet (bead) nodes each iteration — same as _step_single.
        # Default False so existing callers (ancf_rim_shell_dw) are unaffected.
        # Must be set BEFORE capture_graph() so the flag is baked into the
        # captured CUDA graph.
        self._fix_dirichlet_in_batched: bool = False

        self.rim_radius = rim_radius
        self.rim_kn = rim_kn
        self.rim_kd = rim_kd
        self._rim_hub_y: wp.array | None = None
        self._rim_hub_z: wp.array | None = None

        n = ancf_model.n_nodes
        ne = ancf_model.n_elems
        n_dof = n * 6
        N = n_envs
        dev = self.device

        # ---- ANCF state arrays (flat [N*n_nodes] when N>1) ----
        x0_np = ancf_model.node_x0.numpy()  # (n_nodes, 3)
        D0_np = ancf_model.node_D0.numpy()
        if N > 1:
            x0_tiled = np.tile(x0_np, (N, 1))  # (N*n_nodes, 3)
            D0_tiled = np.tile(D0_np, (N, 1))
            self.node_x = wp.array(x0_tiled, dtype=wp.vec3, device=dev)
            self.node_D = wp.array(D0_tiled, dtype=wp.vec3, device=dev)
        else:
            self.node_x = wp.clone(ancf_model.node_x0)
            self.node_D = wp.clone(ancf_model.node_D0)
        self.node_xd = wp.zeros(N * n, dtype=wp.vec3, device=dev)
        self.node_Dd = wp.zeros(N * n, dtype=wp.vec3, device=dev)
        self.node_xdd = wp.zeros(N * n, dtype=wp.vec3, device=dev)
        self.node_Ddd = wp.zeros(N * n, dtype=wp.vec3, device=dev)

        # ---- Element-level scratch (flat [N*n_elems, ...]) ----
        self.elem_f = wp.zeros((N * ne, 24), dtype=float, device=dev)
        self.elem_K = wp.zeros((N * ne, 24, 24), dtype=float, device=dev)
        # Follower-pressure force scratch (residual only; no pressure tangent).
        self.elem_fp = wp.zeros((N * ne, 24), dtype=float, device=dev)
        # Per-element surface orientation sign (+1 if winding normal points along
        # the reference outward director, else −1) so positive pressure inflates.
        x0_ref = ancf_model.node_x0.numpy()  # (n_nodes, 3)
        D0_ref = ancf_model.node_D0.numpy()
        en_ref = ancf_model.elem_nodes.numpy()  # (n_elems, 4)
        Xc = x0_ref[en_ref]  # (n_elems, 4, 3)
        n_ref = np.cross(Xc[:, 2] - Xc[:, 0], Xc[:, 3] - Xc[:, 1])
        D_avg = D0_ref[en_ref].mean(axis=1)  # (n_elems, 3)
        sgn = np.where(np.einsum("ij,ij->i", n_ref, D_avg) >= 0.0, 1.0, -1.0)
        self.elem_psign = wp.array(sgn.astype(np.float32), dtype=float, device=dev)

        # ---- Sealed-gas cavity (CTIS) state ----
        # Reference enclosed volume V_ref (same formula as the device kernel,
        # computed once on the host from the reference geometry). Shared across
        # envs (reference topology is identical).
        c0 = x0_ref.mean(axis=0)  # reference centroid
        xbar0 = Xc.mean(axis=1)  # (n_elems, 3)
        nrm0 = 0.5 * sgn[:, None] * n_ref  # outward area-normals
        V_ref = float((1.0 / 3.0) * np.einsum("ij,ij->i", xbar0 - c0, nrm0).sum())
        self._V_ref = max(V_ref, 1.0e-6)
        self._inv_n_nodes = 1.0 / float(n)
        self.cav_centroid_sum = wp.zeros(N * 3, dtype=float, device=dev)
        self.cav_V = wp.zeros(N, dtype=float, device=dev)  # enclosed volume
        self.cav_p = wp.zeros(N, dtype=float, device=dev)  # absolute pressure
        self.cav_Kgas = wp.zeros(N, dtype=float, device=dev)  # m·R·T (CTIS control)
        self.cav_pbuild = wp.zeros(N, dtype=float, device=dev)  # build pressure
        # Per-GP B-row scratch: one slot per (element, Gauss point) so the
        # GP-parallel kernel can store and read without races.
        n_gp_total = 4 * self.thickness_gp
        self.n_gp_total = n_gp_total
        self.gp_bd = wp.zeros((N * ne * n_gp_total, 24, 3), dtype=float, device=dev)
        self.gp_bs0 = wp.zeros((N * ne * n_gp_total, 24), dtype=float, device=dev)
        self.gp_b13 = wp.zeros((N * ne * n_gp_total, 24), dtype=float, device=dev)
        self.gp_b23 = wp.zeros((N * ne * n_gp_total, 24), dtype=float, device=dev)
        self.gp_w = wp.zeros((N * ne * n_gp_total,), dtype=float, device=dev)
        # EAS accumulators for GP-parallel kernel (atomically summed across GP threads)
        self.elem_HE = wp.zeros((N * ne, 5), dtype=float, device=dev)
        self.elem_KA = wp.zeros((N * ne, 15), dtype=float, device=dev)

        # EAS alpha tiled [N*n_elems, 5]
        eas_np = ancf_model.elem_eas_alpha.numpy()  # (n_elems, 5)
        if N > 1:
            ancf_model.elem_eas_alpha = wp.array(np.tile(eas_np, (N, 1)), dtype=float, device=dev)

        # elem_mat tiled [N*n_elems, 11].
        # If the caller pre-built a per-env mat array (shape [N*n_elems, 11])
        # it is used as-is; otherwise the single-env mat is tiled uniformly.
        if N > 1:
            mat_np = ancf_model.elem_mat.numpy()  # (n_elems, 11) or (N*n_elems, 11)
            if mat_np.shape[0] == ne:
                ancf_model.elem_mat = wp.array(np.tile(mat_np, (N, 1)), dtype=float, device=dev)

        # ---- Global DOF vectors (flat [N*n_dof]) ----
        self.global_f_int = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.global_f_int0 = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.global_f_ext = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.residual = wp.zeros(N * n_dof, dtype=float, device=dev)
        self._nr_res_hist = wp.zeros(N * nr_max_iter, dtype=float, device=dev)
        self.M_a = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.da = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.a_flat = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.neg_R = wp.zeros(N * n_dof, dtype=float, device=dev)
        self.K_contact_diag = wp.zeros(N * n_dof, dtype=float, device=dev)

        # ---- Live gauge pressure (p − p_build), per-env [Pa] ----
        # Written each NR iteration by the cavity gas-law kernel from the current
        # enclosed volume, then read by the follower-pressure force kernel.
        # 0 = built shape held by elastics; >0 over-inflates; <0 deflates.
        self.pressure = wp.zeros(N, dtype=float, device=dev)
        self._n_per_env = n  # nodes per env
        self._ne_per_env = ne  # elements per env

        # ---- Gravity (constant, read once) ----
        self._gravity = model_gravity(model)

        # ---- Predictor scratch (flat [N*n_nodes]) ----
        self.x_pred = wp.zeros(N * n, dtype=wp.vec3, device=dev)
        self.D_pred = wp.zeros(N * n, dtype=wp.vec3, device=dev)
        self.xd_pred = wp.zeros(N * n, dtype=wp.vec3, device=dev)
        self.Dd_pred = wp.zeros(N * n, dtype=wp.vec3, device=dev)

        # ---- Persistent per-node external forces (survive per-NR global_f_ext.zero_()) ----
        # Caller zeros this buffer each substep, then writes contact/external forces once
        # from committed positions.  The NR loop adds it back each iteration via
        # _add_node_f_ext_persistent, keeping the load constant during the solve.
        self.node_f_ext_persistent = wp.zeros(N * n, dtype=wp.vec3, device=dev)

        # ---- Lumped mass ----
        # Single-env mass computed once; replicated N times for flat ops.
        self.lumped_mass = wp.zeros(n_dof, dtype=float, device=dev)
        self.lumped_mass_scaled = wp.zeros(n_dof, dtype=float, device=dev)
        self._precompute_mass()

        # ---- Rest-configuration Jacobian inverses + EAS T0 basis (per element,
        # constant for the solver's lifetime — see compute_rest_jacobians) ----
        ne = self.ancf.n_elems
        self.elem_det_J0c = wp.zeros(ne, dtype=float, device=dev)
        self.elem_T0c0_d = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c0_s = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c1_d = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c1_s = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c2_d = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c2_s = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c3_d = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_T0c3_s = wp.zeros(ne, dtype=wp.vec3, device=dev)
        self.elem_J0inv_a = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_b = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_cc = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_d = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_tA = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_tB = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_tC = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.elem_J0inv_tD = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self._precompute_rest_jacobians()
        # Tiled versions used by batched DOF-level kernels
        if N > 1:
            mass_np = self.lumped_mass.numpy()
            self.lumped_mass_tiled = wp.array(np.tile(mass_np, N), dtype=float, device=dev)
            self.lumped_mass_scaled_tiled = wp.zeros(N * n_dof, dtype=float, device=dev)
        else:
            self.lumped_mass_tiled = self.lumped_mass
            self.lumped_mass_scaled_tiled = self.lumped_mass_scaled

        # ---- Build K_eff sparsity + scatter map ----
        self._init_keff_structure()

        # ---- Batched K_eff values (N independent value arrays, shared sparsity) ----
        if N > 1:
            nnz = int(self.K_eff.values.shape[0])
            self._nnz = nnz
            # Seed with N tiled copies of the real reference-config K_eff.values
            # (populated above by _init_keff_structure), not zeros: PcgSolverBatched
            # never reads this before the first _update_K_eff_inplace_batched call
            # in the step loop, so it doesn't matter there -- but CudssSolverBatched
            # runs a real FACTORIZATION on whatever is here at construction time, and
            # a zero (singular, non-SPD) matrix sent cuDSS's factorization into a
            # near-hanging pathological path (~240s instead of ~10ms on real values).
            keff_vals_np = self.K_eff.values.numpy()
            self.bsr_values_batched = wp.array(np.tile(keff_vals_np, N), dtype=float, device=dev)
        else:
            self._nnz = int(self.K_eff.values.shape[0])
            self.bsr_values_batched = self.K_eff.values  # alias for N=1

        # ---- Linear solver (batched Jacobi-PCG; works for N=1 too) ----
        self.pcg = PcgSolverBatched(N, n_dof, self._nnz, dev, max_iters=pcg_max_iter)

        # ---- Dirichlet nodes (optional bead coupling) ----
        self._dirichlet_idx: wp.array | None = None
        self._dirichlet_dof_mask: wp.array | None = None
        self._dirichlet_nnz_mask: wp.array | None = None

        # ---- CUDA graph (populated by capture_graph()) ----
        self._graph: wp.Graph | None = None

    # ------------------------------------------------------------------
    def get_node_x(self, env_idx: int = 0) -> wp.array:
        """Return node positions for environment env_idx as a [n_nodes] vec3 array."""
        n = self.ancf.n_nodes
        if self.n_envs == 1:
            return self.node_x
        return wp.array(self.node_x.numpy()[env_idx * n : (env_idx + 1) * n], dtype=wp.vec3, device=self.device)

    # ------------------------------------------------------------------
    @property
    def cavity_volume_ref(self) -> float:
        """Reference enclosed cavity volume [m³] at the built shape (per env)."""
        return self._V_ref

    def _as_env_array(self, values) -> np.ndarray:
        if np.isscalar(values):
            return np.full(self.n_envs, float(values), dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape[0] != self.n_envs:
            raise ValueError(f"expected {self.n_envs} values, got {arr.shape[0]}")
        return arr

    def set_cavity(self, nominal_pressure, build_pressure) -> None:
        """Set the sealed-gas cavity state per environment [Pa].

        The CTIS control is the air amount; we parameterise it by the nominal
        pressure (the pressure the cavity would hold at the reference volume):
        ``K_gas = m·R·T = nominal_pressure · V_ref``.  The live pressure then
        self-regulates as ``p = K_gas / V(x)`` and the shell sees the gauge load
        ``p − build_pressure``.  Updates device arrays in place (graph-safe) —
        call between steps, not inside the substep loop.

        Args:
            nominal_pressure: target/nominal absolute pressure (scalar or per-env).
            build_pressure: pressure the rest shape was built at (scalar or per-env).
        """
        self.cav_Kgas.assign(self._as_env_array(nominal_pressure) * self._V_ref)
        self.cav_pbuild.assign(self._as_env_array(build_pressure))

    # ------------------------------------------------------------------
    def set_rim(
        self,
        hub_y: wp.array,
        hub_z: wp.array,
    ) -> None:
        """Bind the per-env hub-centre Y/Z arrays for cylindrical rim contact.

        Must be called before :meth:`capture_graph` so the kernel references
        are baked into the CUDA graph.  Pass the same pre-allocated wp.arrays
        that the example updates each frame before replaying the graph.

        Args:
            hub_y: float wp.array of shape (n_envs,) — hub centre Y [m].
            hub_z: float wp.array of shape (n_envs,) — hub centre Z [m].
        """
        self._rim_hub_y = hub_y
        self._rim_hub_z = hub_z

    def set_dirichlet_nodes(self, idx: np.ndarray | None) -> None:
        """Pin nodes (Dirichlet boundary condition) for the NR solve.

        Each NR iteration the pinned rows are removed from the linear system
        (residual rows zeroed, K_eff rows/columns zeroed with a unit diagonal),
        the predictor is overridden with the prescribed ``node_x/xd/D/Dd`` so
        the pinned nodes hold exactly that state during the solve, and their
        acceleration update is zeroed.  Populate the prescribed state *before*
        calling :meth:`step`.

        Args:
            idx: global flat node indices to pin (already offset by
                ``env * n_nodes`` for multi-env builds), or ``None`` to clear.
        """
        if idx is None:
            self._dirichlet_idx = None
            self._dirichlet_dof_mask = None
            self._dirichlet_nnz_mask = None
            return
        dev = self.device
        self._dirichlet_idx = wp.array(np.asarray(idx, dtype=np.int32), dtype=wp.int32, device=dev)
        n_dirichlet = self._dirichlet_idx.shape[0]
        n_dof = self.ancf.n_nodes * 6
        N = self.n_envs

        # DOF mask over the flat [N*n_dof] vectors: 1 free, 0 pinned.
        self._dirichlet_dof_mask = wp.full(N * n_dof, 1.0, dtype=float, device=dev)
        wp.launch(
            _build_dirichlet_dof_mask,
            dim=n_dirichlet,
            inputs=[self._dirichlet_idx, self._dirichlet_dof_mask],
            device=dev,
        )

        # Per-nnz mask over bsr_values_batched [N*nnz_cap].  K_eff is a warp
        # BsrMatrix whose columns/values are allocated at triplet capacity
        # (self._nnz = per-env stride); offsets bound the valid entries, the
        # tail is never read by the SpMV and stays at +1.
        self._dirichlet_nnz_mask = wp.full(N * self._nnz, 1.0, dtype=float, device=dev)
        wp.launch(
            _build_dirichlet_nnz_mask,
            dim=N * n_dof,
            inputs=[
                self._dirichlet_dof_mask,
                self.K_eff.offsets,
                self.K_eff.columns,
                n_dof,
                self._nnz,
                self._dirichlet_nnz_mask,
            ],
            device=dev,
        )

    def _launch_dirichlet_pred_override(self) -> None:
        wp.launch(
            _dirichlet_pred_override,
            dim=self._dirichlet_idx.shape[0],
            inputs=[
                self._dirichlet_idx,
                self.node_x,
                self.node_xd,
                self.node_D,
                self.node_Dd,
                self.x_pred,
                self.xd_pred,
                self.D_pred,
                self.Dd_pred,
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    def _init_keff_structure(self):
        """Compile stiffness kernel, assemble K_eff at reference config to fix
        sparsity, then build the scatter map for graph-capture-safe in-place updates."""
        dev = self.device
        ne = self.ancf.n_elems

        self.elem_f.zero_()
        self.elem_K.zero_()
        with _CompileProgress(f"stiffness kernel  ({ne} elem)"):
            wp.launch(
                compute_element_forces_stiffness,
                dim=ne,
                inputs=[
                    self.ancf.node_x0,
                    self.ancf.node_D0,
                    self.node_xd,
                    self.node_Dd,
                    self.ancf.node_x0,
                    self.ancf.node_D0,
                    self.ancf.elem_nodes,
                    self.ancf.elem_h,
                    self.ancf.elem_mat,
                    self.elem_f,
                    self.elem_K,
                    self.gp_bd,
                    self.gp_bs0,
                    self.gp_b13,
                    self.gp_b23,
                    self.ancf.elem_eas_alpha,
                    self.ancf.elem_fiber_cos,
                    self.ancf.elem_fiber_sin,
                    self.thickness_gp,
                    self.elem_det_J0c,
                    self.elem_T0c0_d,
                    self.elem_T0c0_s,
                    self.elem_T0c1_d,
                    self.elem_T0c1_s,
                    self.elem_T0c2_d,
                    self.elem_T0c2_s,
                    self.elem_T0c3_d,
                    self.elem_T0c3_s,
                    self.elem_J0inv_a,
                    self.elem_J0inv_b,
                    self.elem_J0inv_cc,
                    self.elem_J0inv_d,
                    self.elem_J0inv_tA,
                    self.elem_J0inv_tB,
                    self.elem_J0inv_tC,
                    self.elem_J0inv_tD,
                ],
                device=dev,
            )
            wp.synchronize_device(dev)

        # Assemble K_eff sparsity from K_t only (mass is diagonal — it fits
        # inside K_t's sparsity since the diagonal is always present).
        self.K_eff = assemble_sparse_stiffness(self.ancf.elem_nodes, self.elem_K, self.ancf.n_nodes, dev)

        with _CompileProgress("scatter map"):
            self.scatter_map = build_scatter_map(self.ancf.elem_nodes, self.K_eff, dev)

    # ------------------------------------------------------------------
    def _update_K_eff_inplace(self, scale_K: float) -> None:
        """Zero K_eff.values and refill from elem_K + lumped mass in-place.

        K_eff = (1/β/dt²)·M_lump + (1+α)·K_t
        M_lump is diagonal — added via add_diag_to_bsr_values using
        lumped_mass_scaled (= lumped_mass / β/dt²) precomputed in capture_graph.

        No memory allocation — safe to call inside a CUDA graph capture region.
        """
        dev = self.device
        n_nz = self.K_eff.values.shape[0]
        wp.launch(zero_bsr_values, dim=n_nz, inputs=[self.K_eff.values], device=dev)
        wp.launch(
            scatter_elem_to_bsr,
            dim=self.ancf.n_elems,
            inputs=[self.elem_K, self.scatter_map, scale_K, self.K_eff.values],
            device=dev,
        )
        wp.launch(
            add_diag_to_bsr_values,
            dim=self.ancf.n_nodes * 6,
            inputs=[self.lumped_mass_scaled, self.K_eff.offsets, self.K_eff.columns, self.K_eff.values],
            device=dev,
        )

    # ------------------------------------------------------------------
    def _update_K_eff_inplace_batched(self, scale_K: float) -> None:
        """Batched variant of _update_K_eff_inplace.

        Writes N independent value blocks into bsr_values_batched[N*nnz].
        Shared offsets/columns are in K_eff.offsets / K_eff.columns.
        """
        dev = self.device
        N = self.n_envs
        ne = self.ancf.n_elems
        n_dof = self.ancf.n_nodes * 6
        nnz = self._nnz
        wp.launch(zero_bsr_values, dim=N * nnz, inputs=[self.bsr_values_batched], device=dev)
        wp.launch(
            scatter_elem_to_bsr_batched,
            dim=N * ne * 24,
            inputs=[self.elem_K, self.scatter_map, scale_K, self.bsr_values_batched, ne, nnz],
            device=dev,
        )
        wp.launch(
            add_diag_to_bsr_values_batched,
            dim=N * n_dof,
            inputs=[
                self.lumped_mass_scaled_tiled,
                self.K_eff.offsets,
                self.K_eff.columns,
                self.bsr_values_batched,
                n_dof,
                nnz,
            ],
            device=dev,
        )

    # ------------------------------------------------------------------
    def _precompute_mass(self):
        """Compute lumped (diagonal) mass directly from element volumes.

        Replaces the old consistent-mass 24x24 kernel whose nested DOF-DOF
        loop caused multi-minute nvcc compile times.  The full consistent mass
        was row-summed into a diagonal anyway, so this is equivalent for the
        implicit HHT solver.
        """
        dev = self.device
        ne = self.ancf.n_elems
        with _CompileProgress(f"lumped mass  ({ne} elem)"):
            wp.launch(
                compute_lumped_mass,
                dim=ne,
                inputs=[
                    self.ancf.node_x0,
                    self.ancf.node_D0,
                    self.ancf.elem_nodes,
                    self.ancf.elem_h,
                    self.ancf.elem_mat,
                    self.lumped_mass,
                ],
                device=dev,
            )
            wp.synchronize_device(dev)

    def _precompute_rest_jacobians(self):
        """Compute per-element rest-configuration Jacobian inverses + EAS T0 basis once.

        compute_element_forces_stiffness[_batched_gp] used to recompute this
        block from scratch on every call (every NR iteration, and in the
        GP-parallel batched kernel, redundantly again per Gauss-point thread
        of the same element) even though it depends only on the rest
        configuration (node_x0/node_D0) and fiber angle, never the current
        deformed state.  See compute_rest_jacobians's docstring.
        """
        dev = self.device
        ne = self.ancf.n_elems
        wp.launch(
            compute_rest_jacobians,
            dim=ne,
            inputs=[
                self.ancf.node_x0,
                self.ancf.node_D0,
                self.ancf.elem_nodes,
                self.ancf.elem_h,
                self.ancf.elem_fiber_cos,
                self.ancf.elem_fiber_sin,
            ],
            outputs=[
                self.elem_det_J0c,
                self.elem_T0c0_d,
                self.elem_T0c0_s,
                self.elem_T0c1_d,
                self.elem_T0c1_s,
                self.elem_T0c2_d,
                self.elem_T0c2_s,
                self.elem_T0c3_d,
                self.elem_T0c3_s,
                self.elem_J0inv_a,
                self.elem_J0inv_b,
                self.elem_J0inv_cc,
                self.elem_J0inv_d,
                self.elem_J0inv_tA,
                self.elem_J0inv_tB,
                self.elem_J0inv_tC,
                self.elem_J0inv_tD,
            ],
            device=dev,
        )
        wp.synchronize_device(dev)

    # ------------------------------------------------------------------
    def _step_single(self, dt: float) -> None:
        """Single-env (N=1) HHT step — original implementation."""
        dev = self.device
        n = self.ancf.n_nodes
        n_dof = n * 6
        alpha = _HHT_ALPHA
        beta = _HHT_BETA
        gamma = _HHT_GAMMA

        wp.copy(self.global_f_int0, self.global_f_int)

        wp.launch(_zero_eas_alpha, dim=self.ancf.n_elems, inputs=[self.ancf.elem_eas_alpha], device=dev)

        wp.launch(
            _hht_predict,
            dim=n,
            inputs=[self.node_x, self.node_xd, self.node_xdd, dt, beta, gamma],
            outputs=[self.x_pred, self.xd_pred],
            device=dev,
        )
        wp.launch(
            _hht_predict,
            dim=n,
            inputs=[self.node_D, self.node_Dd, self.node_Ddd, dt, beta, gamma],
            outputs=[self.D_pred, self.Dd_pred],
            device=dev,
        )
        if self._dirichlet_idx is not None:
            self._launch_dirichlet_pred_override()

        scale_K = 1.0 + alpha
        # K_eff Jacobian correction for stiffness-proportional Rayleigh damping.
        # f_damp = α_d·K·v where v = xd_pred + γ·dt·a  →  ∂f_damp/∂a = α_d·K·γ·dt.
        # The (1+α) HHT scale and the 1/(β·dt²) division give:
        #   K_eff += (1+α)·α_d·K·(γ/(β·dt))
        # Without this, K_eff is ~175x too small at dt≈1.67ms, α_d=0.15,
        # so each NR step overshoots by ~175x and the iteration diverges.
        _alpha_d = getattr(self, "_alpha_damp", 0.0)
        scale_K_Keff = scale_K * (1.0 + _alpha_d * gamma / (beta * dt))

        for _nr in range(self.nr_max_iter):
            wp.launch(
                _hht_kinematic,
                dim=n,
                inputs=[self.x_pred, self.xd_pred, self.node_xdd, dt, beta, gamma],
                outputs=[self.node_x, self.node_xd],
                device=dev,
            )
            wp.launch(
                _hht_kinematic,
                dim=n,
                inputs=[self.D_pred, self.Dd_pred, self.node_Ddd, dt, beta, gamma],
                outputs=[self.node_D, self.node_Dd],
                device=dev,
            )

            ne = self.ancf.n_elems
            self.elem_f.zero_()
            self.elem_K.zero_()
            wp.launch(
                compute_element_forces_stiffness,
                dim=ne,
                inputs=[
                    self.node_x,
                    self.node_D,
                    self.node_xd,
                    self.node_Dd,
                    self.ancf.node_x0,
                    self.ancf.node_D0,
                    self.ancf.elem_nodes,
                    self.ancf.elem_h,
                    self.ancf.elem_mat,
                    self.elem_f,
                    self.elem_K,
                    self.gp_bd,
                    self.gp_bs0,
                    self.gp_b13,
                    self.gp_b23,
                    self.ancf.elem_eas_alpha,
                    self.ancf.elem_fiber_cos,
                    self.ancf.elem_fiber_sin,
                    self.thickness_gp,
                    self.elem_det_J0c,
                    self.elem_T0c0_d,
                    self.elem_T0c0_s,
                    self.elem_T0c1_d,
                    self.elem_T0c1_s,
                    self.elem_T0c2_d,
                    self.elem_T0c2_s,
                    self.elem_T0c3_d,
                    self.elem_T0c3_s,
                    self.elem_J0inv_a,
                    self.elem_J0inv_b,
                    self.elem_J0inv_cc,
                    self.elem_J0inv_d,
                    self.elem_J0inv_tA,
                    self.elem_J0inv_tB,
                    self.elem_J0inv_tC,
                    self.elem_J0inv_tD,
                ],
                device=dev,
            )
            # --- Sealed-gas cavity: V(x) → p = K_gas/V → gauge load (p − p_build).
            self.cav_centroid_sum.zero_()
            wp.launch(accumulate_centroid, dim=n, inputs=[self.node_x, self.cav_centroid_sum, n], device=dev)
            self.cav_V.zero_()
            wp.launch(
                accumulate_cavity_volume,
                dim=self.ancf.n_elems,
                inputs=[
                    self.node_x,
                    self.ancf.elem_nodes,
                    self.elem_psign,
                    self.cav_centroid_sum,
                    self._inv_n_nodes,
                    self.cav_V,
                    self.ancf.n_elems,
                    n,
                ],
                device=dev,
            )
            wp.launch(
                cavity_gas_law,
                dim=1,
                inputs=[self.cav_Kgas, self.cav_V, self.cav_pbuild, self.cav_p, self.pressure],
                device=dev,
            )
            # Follower pressure force at the current iterate (gauge from gas law).
            self.elem_fp.zero_()
            wp.launch(
                compute_pressure_force_stiffness,
                dim=self.ancf.n_elems,
                inputs=[self.node_x, self.ancf.elem_nodes, self.elem_psign, self.pressure, self.elem_fp],
                device=dev,
            )

            self.global_f_int.zero_()
            wp.launch(
                scatter_forces,
                dim=self.ancf.n_elems,
                inputs=[self.ancf.elem_nodes, self.elem_f, self.global_f_int],
                device=dev,
            )

            self.global_f_ext.zero_()
            wp.launch(
                _assemble_rhs_gravity,
                dim=n_dof,
                inputs=[self.lumped_mass, self._gravity, self.global_f_ext],
                device=dev,
            )
            # External follower-pressure load scattered into f_ext.
            wp.launch(
                scatter_forces,
                dim=self.ancf.n_elems,
                inputs=[self.ancf.elem_nodes, self.elem_fp, self.global_f_ext],
                device=dev,
            )
            wp.launch(zero_contact_diag, dim=n_dof, inputs=[self.K_contact_diag], device=dev)
            wp.launch(
                apply_ground_contact,
                dim=n,
                inputs=[
                    self.node_x,
                    self.node_xd,
                    self.ground_z,
                    self.kn,
                    self.kd,
                    self.mu,
                    self.v_reg,
                    float(gamma / (beta * dt)),
                    self.global_f_ext,
                    self.K_contact_diag,
                ],
                device=dev,
            )
            if self.rim_radius > 0.0 and self._rim_hub_y is not None:
                wp.launch(
                    apply_rim_contact,
                    dim=n,
                    inputs=[
                        self.node_x,
                        self.node_xd,
                        self._rim_hub_y,
                        self._rim_hub_z,
                        self.rim_radius,
                        self.rim_kn,
                        self.rim_kd,
                        self.global_f_ext,
                        self.K_contact_diag,
                    ],
                    device=dev,
                )
            wp.launch(
                _add_node_f_ext_persistent, dim=n, inputs=[self.node_f_ext_persistent, self.global_f_ext], device=dev
            )

            wp.launch(_flatten_vec3_pair, dim=n, inputs=[self.node_xdd, self.node_Ddd, self.a_flat], device=dev)
            wp.launch(_pointwise_mul, dim=n_dof, inputs=[self.lumped_mass, self.a_flat, self.M_a], device=dev)

            wp.launch(
                _build_residual,
                dim=n_dof,
                inputs=[self.M_a, self.global_f_int, self.global_f_int0, self.global_f_ext, alpha, self.residual],
                device=dev,
            )
            if self._dirichlet_idx is not None:
                wp.launch(_mask_dof, dim=n_dof, inputs=[self._dirichlet_dof_mask, self.residual], device=dev)

            if self.debug_residuals:
                wp.launch(_zero_nr_slot, dim=1, inputs=[self._nr_res_hist, _nr, self.nr_max_iter], device=dev)
                wp.launch(
                    _accum_nr_residual_sq,
                    dim=n_dof,
                    inputs=[self.residual, self._nr_res_hist, _nr, self.nr_max_iter, n_dof],
                    device=dev,
                )

            self._update_K_eff_inplace(scale_K_Keff)

            wp.launch(
                add_diag_to_bsr_values,
                dim=n_dof,
                inputs=[self.K_contact_diag, self.K_eff.offsets, self.K_eff.columns, self.K_eff.values],
                device=dev,
            )
            if self._dirichlet_idx is not None:
                wp.launch(
                    _apply_dirichlet_to_bsr,
                    dim=self._nnz,
                    inputs=[self._dirichlet_nnz_mask, self.K_eff.values],
                    device=dev,
                )

            self.da.zero_()
            wp.launch(_negate, dim=n_dof, inputs=[self.residual, self.neg_R], device=dev)
            self.pcg.solve(self.K_eff.offsets, self.K_eff.columns, self.K_eff.values, self.neg_R, self.da)

            wp.launch(
                _apply_acceleration_update,
                dim=n,
                inputs=[self.da, float(1.0 / (beta * dt * dt)), self.node_xdd, self.node_Ddd],
                device=dev,
            )

            if self._dirichlet_idx is not None:
                wp.launch(
                    _zero_dirichlet_acc,
                    dim=self._dirichlet_idx.shape[0],
                    inputs=[self._dirichlet_idx, self.node_xdd, self.node_Ddd],
                    device=dev,
                )

        wp.launch(
            _hht_kinematic,
            dim=n,
            inputs=[self.x_pred, self.xd_pred, self.node_xdd, dt, beta, gamma],
            outputs=[self.node_x, self.node_xd],
            device=dev,
        )
        wp.launch(
            _hht_kinematic,
            dim=n,
            inputs=[self.D_pred, self.Dd_pred, self.node_Ddd, dt, beta, gamma],
            outputs=[self.node_D, self.node_Dd],
            device=dev,
        )

    # ------------------------------------------------------------------
    def _step_batched(self, dt: float) -> None:
        """Multi-env (N>1) HHT step — all arrays flat [N*per_env]."""
        dev = self.device
        n = self.ancf.n_nodes
        ne = self.ancf.n_elems
        n_dof = n * 6
        N = self.n_envs
        Nn = N * n
        Nne = N * ne
        Nndof = N * n_dof
        alpha = _HHT_ALPHA
        beta = _HHT_BETA
        gamma = _HHT_GAMMA

        wp.copy(self.global_f_int0, self.global_f_int)

        wp.launch(_zero_eas_alpha, dim=Nne, inputs=[self.ancf.elem_eas_alpha], device=dev)

        wp.launch(
            _hht_predict,
            dim=Nn,
            inputs=[self.node_x, self.node_xd, self.node_xdd, dt, beta, gamma],
            outputs=[self.x_pred, self.xd_pred],
            device=dev,
        )
        wp.launch(
            _hht_predict,
            dim=Nn,
            inputs=[self.node_D, self.node_Dd, self.node_Ddd, dt, beta, gamma],
            outputs=[self.D_pred, self.Dd_pred],
            device=dev,
        )
        # Dirichlet handling in the batched path is opt-in (see
        # _fix_dirichlet_in_batched) so ancf_rim_shell keeps its behaviour.
        _dirichlet = self._fix_dirichlet_in_batched and self._dirichlet_idx is not None
        if _dirichlet:
            self._launch_dirichlet_pred_override()

        scale_K = 1.0 + alpha
        _alpha_d = getattr(self, "_alpha_damp", 0.0)
        scale_K_Keff = scale_K * (1.0 + _alpha_d * gamma / (beta * dt))

        for _nr in range(self.nr_max_iter):
            wp.launch(
                _hht_kinematic,
                dim=Nn,
                inputs=[self.x_pred, self.xd_pred, self.node_xdd, dt, beta, gamma],
                outputs=[self.node_x, self.node_xd],
                device=dev,
            )
            wp.launch(
                _hht_kinematic,
                dim=Nn,
                inputs=[self.D_pred, self.Dd_pred, self.node_Ddd, dt, beta, gamma],
                outputs=[self.node_D, self.node_Dd],
                device=dev,
            )

            self.elem_f.zero_()
            self.elem_K.zero_()
            wp.launch(
                _zero_eas_and_accum_batched,
                dim=Nne,
                inputs=[self.ancf.elem_eas_alpha, self.elem_HE, self.elem_KA],
                device=dev,
            )
            wp.launch(
                compute_element_forces_stiffness_batched_gp,
                dim=Nne * self.n_gp_total,
                inputs=[
                    self.node_x,
                    self.node_D,
                    self.node_xd,
                    self.node_Dd,
                    self.ancf.node_x0,
                    self.ancf.node_D0,
                    self.ancf.elem_nodes,
                    self.ancf.elem_h,
                    self.ancf.elem_mat,
                    self.elem_f,
                    self.elem_K,
                    self.gp_bd,
                    self.gp_bs0,
                    self.gp_b13,
                    self.gp_b23,
                    self.gp_w,
                    self.ancf.elem_eas_alpha,
                    self.ancf.elem_fiber_cos,
                    self.ancf.elem_fiber_sin,
                    self.elem_HE,
                    self.elem_KA,
                    ne,
                    n,
                    self.n_gp_total,
                    self.thickness_gp,
                    self.elem_det_J0c,
                    self.elem_T0c0_d,
                    self.elem_T0c0_s,
                    self.elem_T0c1_d,
                    self.elem_T0c1_s,
                    self.elem_T0c2_d,
                    self.elem_T0c2_s,
                    self.elem_T0c3_d,
                    self.elem_T0c3_s,
                    self.elem_J0inv_a,
                    self.elem_J0inv_b,
                    self.elem_J0inv_cc,
                    self.elem_J0inv_d,
                    self.elem_J0inv_tA,
                    self.elem_J0inv_tB,
                    self.elem_J0inv_tC,
                    self.elem_J0inv_tD,
                ],
                device=dev,
            )
            wp.launch(
                compute_element_K_from_B,
                dim=Nne * 576,
                inputs=[
                    self.gp_bd,
                    self.gp_bs0,
                    self.gp_b13,
                    self.gp_b23,
                    self.gp_w,
                    self.ancf.elem_mat,
                    self.elem_K,
                    self.n_gp_total,
                ],
                device=dev,
            )
            wp.launch(
                _eas_solve_damping_batched,
                dim=Nne,
                inputs=[
                    self.elem_HE,
                    self.elem_KA,
                    self.ancf.elem_eas_alpha,
                    self.ancf.elem_mat,
                    self.node_xd,
                    self.node_Dd,
                    self.ancf.elem_nodes,
                    self.elem_K,
                    self.elem_f,
                    ne,
                    n,
                ],
                device=dev,
            )
            # --- Sealed-gas cavity (per env): V(x) → p = K_gas/V → gauge load.
            self.cav_centroid_sum.zero_()
            wp.launch(accumulate_centroid, dim=Nn, inputs=[self.node_x, self.cav_centroid_sum, n], device=dev)
            self.cav_V.zero_()
            wp.launch(
                accumulate_cavity_volume,
                dim=Nne,
                inputs=[
                    self.node_x,
                    self.ancf.elem_nodes,
                    self.elem_psign,
                    self.cav_centroid_sum,
                    self._inv_n_nodes,
                    self.cav_V,
                    ne,
                    n,
                ],
                device=dev,
            )
            wp.launch(
                cavity_gas_law,
                dim=N,
                inputs=[self.cav_Kgas, self.cav_V, self.cav_pbuild, self.cav_p, self.pressure],
                device=dev,
            )
            # Follower pressure force at the current iterate (gauge from gas law).
            self.elem_fp.zero_()
            wp.launch(
                compute_pressure_force_stiffness_batched_gp,
                dim=Nne * 4,
                inputs=[
                    self.node_x,
                    self.ancf.elem_nodes,
                    self.elem_psign,
                    self.pressure,
                    self.elem_fp,
                    ne,
                    n,
                ],
                device=dev,
            )

            self.global_f_int.zero_()
            wp.launch(
                scatter_forces_batched,
                dim=Nne,
                inputs=[self.ancf.elem_nodes, self.elem_f, self.global_f_int, ne, n],
                device=dev,
            )

            self.global_f_ext.zero_()
            wp.launch(
                _assemble_rhs_gravity,
                dim=Nndof,
                inputs=[self.lumped_mass_tiled, self._gravity, self.global_f_ext],
                device=dev,
            )
            # External follower-pressure load scattered into f_ext.
            wp.launch(
                scatter_forces_batched,
                dim=Nne,
                inputs=[self.ancf.elem_nodes, self.elem_fp, self.global_f_ext, ne, n],
                device=dev,
            )
            wp.launch(zero_contact_diag, dim=Nndof, inputs=[self.K_contact_diag], device=dev)
            wp.launch(
                apply_ground_contact_batched,
                dim=Nn,
                inputs=[
                    self.node_x,
                    self.node_xd,
                    self.ground_z,
                    self.kn,
                    self.kd,
                    self.mu,
                    self.v_reg,
                    float(gamma / (beta * dt)),
                    self.global_f_ext,
                    self.K_contact_diag,
                    n,
                ],
                device=dev,
            )
            if self.rim_radius > 0.0 and self._rim_hub_y is not None:
                wp.launch(
                    apply_rim_contact_batched,
                    dim=Nn,
                    inputs=[
                        self.node_x,
                        self.node_xd,
                        self._rim_hub_y,
                        self._rim_hub_z,
                        self.rim_radius,
                        self.rim_kn,
                        self.rim_kd,
                        self.global_f_ext,
                        self.K_contact_diag,
                        n,
                    ],
                    device=dev,
                )
            wp.launch(
                _add_node_f_ext_persistent, dim=Nn, inputs=[self.node_f_ext_persistent, self.global_f_ext], device=dev
            )

            wp.launch(_flatten_vec3_pair, dim=Nn, inputs=[self.node_xdd, self.node_Ddd, self.a_flat], device=dev)
            wp.launch(_pointwise_mul, dim=Nndof, inputs=[self.lumped_mass_tiled, self.a_flat, self.M_a], device=dev)

            wp.launch(
                _build_residual,
                dim=Nndof,
                inputs=[self.M_a, self.global_f_int, self.global_f_int0, self.global_f_ext, alpha, self.residual],
                device=dev,
            )
            if _dirichlet:
                wp.launch(_mask_dof, dim=Nndof, inputs=[self._dirichlet_dof_mask, self.residual], device=dev)

            if self.debug_residuals:
                wp.launch(_zero_nr_slot, dim=N, inputs=[self._nr_res_hist, _nr, self.nr_max_iter], device=dev)
                wp.launch(
                    _accum_nr_residual_sq,
                    dim=Nndof,
                    inputs=[self.residual, self._nr_res_hist, _nr, self.nr_max_iter, n_dof],
                    device=dev,
                )

            self._update_K_eff_inplace_batched(scale_K_Keff)

            wp.launch(
                add_diag_to_bsr_values_batched,
                dim=Nndof,
                inputs=[
                    self.K_contact_diag,
                    self.K_eff.offsets,
                    self.K_eff.columns,
                    self.bsr_values_batched,
                    n_dof,
                    self._nnz,
                ],
                device=dev,
            )
            if _dirichlet:
                wp.launch(
                    _apply_dirichlet_to_bsr,
                    dim=N * self._nnz,
                    inputs=[self._dirichlet_nnz_mask, self.bsr_values_batched],
                    device=dev,
                )

            self.da.zero_()
            wp.launch(_negate, dim=Nndof, inputs=[self.residual, self.neg_R], device=dev)
            self.pcg.solve(self.K_eff.offsets, self.K_eff.columns, self.bsr_values_batched, self.neg_R, self.da)

            wp.launch(
                _apply_acceleration_update,
                dim=Nn,
                inputs=[self.da, float(1.0 / (beta * dt * dt)), self.node_xdd, self.node_Ddd],
                device=dev,
            )

            # ── Dirichlet bead nodes: zero NR update (opt-in, for DW) ─────────
            # With the rows eliminated above the solve already returns 0 here;
            # this keeps the pinned accelerations exactly zero regardless.
            if _dirichlet:
                wp.launch(
                    _zero_dirichlet_acc,
                    dim=self._dirichlet_idx.shape[0],
                    inputs=[self._dirichlet_idx, self.node_xdd, self.node_Ddd],
                    device=dev,
                )

        wp.launch(
            _hht_kinematic,
            dim=Nn,
            inputs=[self.x_pred, self.xd_pred, self.node_xdd, dt, beta, gamma],
            outputs=[self.node_x, self.node_xd],
            device=dev,
        )
        wp.launch(
            _hht_kinematic,
            dim=Nn,
            inputs=[self.D_pred, self.Dd_pred, self.node_Ddd, dt, beta, gamma],
            outputs=[self.node_D, self.node_Dd],
            device=dev,
        )

    # ------------------------------------------------------------------
    def recompute_f_int(self) -> None:
        """Re-evaluate ``global_f_int`` at the current ``node_x`` without an NR step.

        ``_step_batched`` (n_envs > 1) does not call ``_zero_dirichlet_acc``
        inside the NR loop, so Dirichlet (bead) nodes accumulate acceleration
        corrections and can drift from
        their prescribed positions.  After a post-step ``prescribe_fn()`` call
        resets the bead positions, call this method before reading
        ``global_f_int`` to get forces at the corrected bead locations.

        ``_step_single`` (n_envs == 1) already zeroes Dirichlet accelerations
        inside the NR loop; this method is a no-op for that path.
        """
        if self.n_envs == 1:
            return
        dev = self.device
        n = self.ancf.n_nodes
        ne = self.ancf.n_elems
        N = self.n_envs
        Nne = N * ne

        self.elem_f.zero_()
        self.elem_K.zero_()
        wp.launch(
            _zero_eas_and_accum_batched,
            dim=Nne,
            inputs=[self.ancf.elem_eas_alpha, self.elem_HE, self.elem_KA],
            device=dev,
        )
        wp.launch(
            compute_element_forces_stiffness_batched_gp,
            dim=Nne * self.n_gp_total,
            inputs=[
                self.node_x,
                self.node_D,
                self.node_xd,
                self.node_Dd,
                self.ancf.node_x0,
                self.ancf.node_D0,
                self.ancf.elem_nodes,
                self.ancf.elem_h,
                self.ancf.elem_mat,
                self.elem_f,
                self.elem_K,
                self.gp_bd,
                self.gp_bs0,
                self.gp_b13,
                self.gp_b23,
                self.gp_w,
                self.ancf.elem_eas_alpha,
                self.ancf.elem_fiber_cos,
                self.ancf.elem_fiber_sin,
                self.elem_HE,
                self.elem_KA,
                ne,
                n,
                self.n_gp_total,
                self.thickness_gp,
                self.elem_det_J0c,
                self.elem_T0c0_d,
                self.elem_T0c0_s,
                self.elem_T0c1_d,
                self.elem_T0c1_s,
                self.elem_T0c2_d,
                self.elem_T0c2_s,
                self.elem_T0c3_d,
                self.elem_T0c3_s,
                self.elem_J0inv_a,
                self.elem_J0inv_b,
                self.elem_J0inv_cc,
                self.elem_J0inv_d,
                self.elem_J0inv_tA,
                self.elem_J0inv_tB,
                self.elem_J0inv_tC,
                self.elem_J0inv_tD,
            ],
            device=dev,
        )
        wp.launch(
            compute_element_K_from_B,
            dim=Nne * 576,
            inputs=[
                self.gp_bd,
                self.gp_bs0,
                self.gp_b13,
                self.gp_b23,
                self.gp_w,
                self.ancf.elem_mat,
                self.elem_K,
                self.n_gp_total,
            ],
            device=dev,
        )
        wp.launch(
            _eas_solve_damping_batched,
            dim=Nne,
            inputs=[
                self.elem_HE,
                self.elem_KA,
                self.ancf.elem_eas_alpha,
                self.ancf.elem_mat,
                self.node_xd,
                self.node_Dd,
                self.ancf.elem_nodes,
                self.elem_K,
                self.elem_f,
                ne,
                n,
            ],
            device=dev,
        )
        self.global_f_int.zero_()
        wp.launch(
            scatter_forces_batched,
            dim=Nne,
            inputs=[self.ancf.elem_nodes, self.elem_f, self.global_f_int, ne, n],
            device=dev,
        )

    # ------------------------------------------------------------------
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance ANCF state by one HHT-alpha step with fixed-count NR.

        The method is graph-capturable: no host-device syncs, no allocations,
        fixed iteration counts.  Call ``capture_graph()`` once after warm-up,
        then replace ``step()`` calls with ``graph_step()`` for best performance.
        """
        if self.n_envs > 1:
            self._step_batched(dt)
        else:
            self._step_single(dt)

    # ------------------------------------------------------------------
    def residual_report(self, env: int = 0) -> str:
        """One-line convergence summary for the last completed step.

        Requires :attr:`debug_residuals` to have been ``True`` before
        :meth:`capture_graph`.  Call from the frame loop *outside* the CUDA
        graph -- it reads device memory back to the host.

        Reports, per Newton-Raphson iteration, ``||R||`` (the HHT residual the
        NR step is driving to zero) and the ratio to the previous iteration.
        A converging Newton solve shows the ratio falling well below 1; ratios
        near or above 1 mean the iteration is not converging and the step being
        taken is not a descent direction.

        Also reports the PCG reduction ``||r||_final / ||r||_initial`` for the
        last linear solve of the step -- how well the *linear* system was
        solved, independent of whether Newton itself is converging.

        Args:
            env: Environment index to report (batched solves hold one history
                per environment).

        Returns:
            Formatted single-line report.
        """
        if not self.debug_residuals:
            return "[ancf] residual_report: set solver.debug_residuals = True before capture_graph()"

        nr = self.nr_max_iter
        h = self._nr_res_hist.numpy()[env * nr : (env + 1) * nr]
        norms = [float(v) ** 0.5 for v in h]

        parts, prev = [], None
        for i, v in enumerate(norms):
            if prev is not None and prev > 0.0:
                parts.append(f"NR{i}={v:.3e}({v / prev:.2f}x)")
            else:
                parts.append(f"NR{i}={v:.3e}")
            prev = v

        rz_i = float(self.pcg.rz_init.numpy()[env])
        rz_f = float(self.pcg.rz_last.numpy()[env])
        pcg = (rz_f / rz_i) ** 0.5 if rz_i > 0.0 else float("nan")
        return f"[ancf env{env}] " + "  ".join(parts) + f"  |  PCG {pcg:.2e}"

    def capture_graph(self, dt: float) -> None:
        """Capture one ``step(dt)`` as a replayable CUDA graph.

        Args:
            dt: Integration time step [s].  Must stay constant between capture
                and all subsequent ``graph_step()`` calls.

        Call once after construction (kernels are already compiled by
        ``_init_keff_structure``).  Subsequent simulation frames should call
        ``graph_step()`` instead of ``step()``.
        """
        dev = self.device
        n_dof = self.ancf.n_nodes * 6
        N = self.n_envs

        # Read alpha_damp from the first element (uniform for Polaris tire).
        # Used to add the damping Jacobian ∂f_damp/∂a = α_d·K·γ·dt to K_eff.
        if not hasattr(self, "_alpha_damp"):
            self._alpha_damp = float(self.ancf.elem_mat.numpy()[0, 10])

        # Precompute lumped_mass_scaled = lumped_mass / (β·dt²).
        # Constant for fixed dt — outside the captured region (read-only inside).
        scale_M = 1.0 / (_HHT_BETA * dt * dt)
        if N > 1:
            wp.launch(
                _pointwise_scale,
                dim=N * n_dof,
                inputs=[self.lumped_mass_tiled, float(scale_M), self.lumped_mass_scaled_tiled],
                device=dev,
            )
        else:
            wp.launch(
                _pointwise_scale,
                dim=n_dof,
                inputs=[self.lumped_mass, float(scale_M), self.lumped_mass_scaled],
                device=dev,
            )
        wp.synchronize_device(dev)

        label = f"CUDA graph capture  ({N} env × {self.nr_max_iter} NR × {self.pcg.max_iters} PCG, dt={dt:.2e} s)"
        with _CompileProgress(label):
            wp.capture_begin(device=dev)
            self.step(None, None, None, None, dt)
            self._graph = wp.capture_end(device=dev)

    def graph_step(self) -> None:
        """Replay the captured CUDA graph for one simulation step."""
        if self._graph is None:
            raise RuntimeError("Call capture_graph() before graph_step().")
        wp.capture_launch(self._graph)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def model_gravity(model: Model) -> wp.vec3:
    """Read gravity vector from Newton Model (returns (0,-9.81,0) if not set)."""
    if model.gravity is not None:
        g_np = model.gravity.numpy()
        if g_np.shape[0] > 0:
            return wp.vec3(float(g_np[0, 0]), float(g_np[0, 1]), float(g_np[0, 2]))
    return wp.vec3(0.0, -9.81, 0.0)


@wp.kernel
def _pointwise_mul(
    a: wp.array[float],
    b: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    out[i] = a[i] * b[i]
