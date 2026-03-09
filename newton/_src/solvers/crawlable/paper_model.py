# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Inchworm crawling paper model (Gamus et al., arXiv:1911.05227).
# Kinematics, normal forces, slippage criterion, and hybrid stick-slip state machine.
#
# Relation to the solver (SolverCrawlable):
#   The solver has no built-in paper constants; M, L, beta, mu, g come from set_gait_params().
#   Each step the solver: (1) gets phi1, phi2 from the mesh via joint_angles_from_positions,
#   (2) calls crawl_state_step_simple(phi1, phi2, d_prev, dt, M, L, beta, mu, g, ...) which uses
#   all geometry/force helpers below to decide which foot slips and with what force,
#   (3) passes the returned slip_leg, ft_mag, slip_dir to the contact kernel and uses d for
#   kinematic displacement. For d_prev and d_current the solver also calls compute_l, compute_theta,
#   compute_d in _crawl_d_from_state().
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Literal


# -----------------------------------------------------------------------------
# GROUP 1: Paper geometry — link length and angles (l, θ, d, xc)
# Used by crawl_state_step_simple and by the solver's _crawl_d_from_state to compute
# contact distance d and to feed the normal-force and slip-criterion formulas.
# -----------------------------------------------------------------------------

def compute_l(M: float, L: float, beta: float) -> float:
    """Link length l = L / (2 + beta). Solver: used inside crawl_state_step_simple and in
    _crawl_d_from_state (with compute_theta, compute_d) to get current contact distance d."""
    return L / (2.0 + beta)


def compute_theta(phi1: float, phi2: float, beta: float) -> float:
    """
    Central link orientation from paper Eq. (3):
    tan θ = (sin φ₁ − sin φ₂) / (cos φ₁ + cos φ₂ − β).
    Solver: used by compute_d, compute_xc, compute_delta_direct, compute_d_dot; not called
    directly by the solver except via these helpers and _crawl_d_from_state (theta for d).
    """
    num = np.sin(phi1) - np.sin(phi2)
    den = np.cos(phi1) + np.cos(phi2) - beta
    return np.arctan2(num, den)


def compute_d(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Horizontal distance between the two foot contacts. Solver: crawl_state_step_simple uses
    d for fn and for (d - d_prev)/dt when angular velocities are not given; _crawl_d_from_state
    returns this d so the solver can compute kinematic displacement Δd = d_new - d_prev."""
    return l * (
        beta * np.cos(theta)
        - np.cos(phi1 - theta)
        - np.cos(phi2 + theta)
    )


def compute_xc(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Horizontal distance of CoM from left contact. Solver: only used inside
    crawl_state_step_simple as input to compute_fn (paper Eq. (6b) for normal forces)."""
    return l / (2.0 * (2.0 + beta)) * (
        (2.0 + beta) * beta * np.cos(theta)
        - (3.0 + 2.0 * beta) * np.cos(phi1 - theta)
        - np.cos(phi2 + theta)
    )


# -----------------------------------------------------------------------------
# GROUP 2: Normal forces and slip criterion
# compute_fn: paper Eq. (6b); solver uses fn1, fn2 for slip force magnitude and CSV.
# compute_delta_direct: Δ > 0 ⇒ left slips, Δ < 0 ⇒ right slips; solver uses inside
# crawl_state_step_simple only.
# -----------------------------------------------------------------------------

def compute_fn(xc: float, d: float, M: float, g: float) -> tuple[float, float]:
    """
    Normal forces at left (1) and right (2) contacts (paper Eq. (6b)).
    Solver: called inside crawl_state_step_simple; fn1, fn2 are returned and used for
    slip force magnitude (μ·f_n of the slipping foot) and exposed via get_crawl_contact_forces
    for CSV (fn_left_raw, fn_right_raw).
    """
    if d <= 1e-12:
        return M * g * 0.5, M * g * 0.5
    fn1 = (1.0 - xc / d) * M * g
    fn2 = (xc / d) * M * g
    fn1 = max(0.0, fn1)
    fn2 = max(0.0, fn2)
    return fn1, fn2


def compute_delta_direct(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Slippage criterion Δ = (1+β)/(2(2+β)) * l * [cos(φ2+θ) - cos(φ1-θ)]. Δ > 0 ⇒ left
    slips (SLIP_STICK), Δ < 0 ⇒ right slips (STICK_SLIP). Solver: only used inside
    crawl_state_step_simple to set state and which f_n to use for ft_mag."""
    return (1.0 + beta) / (2.0 * (2.0 + beta)) * l * (
        np.cos(phi2 + theta) - np.cos(phi1 - theta)
    )


# -----------------------------------------------------------------------------
# GROUP 3: Rate of change of θ and d (for slip direction and hysteresis)
# Solver: the solver passes phi1_dot, phi2_dot from (φ - φ_prev)/dt; when provided,
# crawl_state_step_simple uses compute_d_dot_from_angular_velocities to get ḋ = d_dot.
# sign(ḋ) sets the slip force direction and hysteresis (avoid single-frame flips).
# -----------------------------------------------------------------------------

def compute_theta_dot(
    phi1: float,
    phi2: float,
    phi1_dot: float,
    phi2_dot: float,
    beta: float,
) -> float:
    """θ̇ from chain rule on θ(φ₁, φ₂). Solver: only used inside
    compute_d_dot_from_angular_velocities to evaluate paper Eq. (8) for ḋ."""
    u = np.sin(phi1) - np.sin(phi2)
    v = np.cos(phi1) + np.cos(phi2) - beta
    denom = u * u + v * v
    if denom < 1e-18:
        return 0.0
    d_theta_d_phi1 = (v * np.cos(phi1) + u * np.sin(phi1)) / denom
    d_theta_d_phi2 = (-v * np.cos(phi2) + u * np.sin(phi2)) / denom
    return float(d_theta_d_phi1 * phi1_dot + d_theta_d_phi2 * phi2_dot)


def compute_d_dot(
    phi1: float,
    phi2: float,
    theta: float,
    phi1_dot: float,
    phi2_dot: float,
    theta_dot: float,
    l: float,
    beta: float,
) -> float:
    """Time derivative of contact distance (paper Eq. (8)). Solver: called only from
    compute_d_dot_from_angular_velocities; ḋ is used to set slip_dir = sign(ḋ) and
    hysteresis (when |ḋ| < d_dot_eps keep previous slip_dir to avoid chattering)."""
    return l * (
        np.sin(phi2 + theta) * (phi2_dot + theta_dot)
        + np.sin(phi1 - theta) * (phi1_dot - theta_dot)
        - beta * np.sin(theta) * theta_dot
    )


def compute_d_dot_from_angular_velocities(
    phi1: float,
    phi2: float,
    phi1_dot: float,
    phi2_dot: float,
    l: float,
    beta: float,
) -> float:
    """Accurate ḋ from paper Eq. (8) using θ, θ̇ and φ̇₁, φ̇₂. Solver: called from
    crawl_state_step_simple when the solver passes phi1_dot, phi2_dot (from mesh angles
    (φ−φ_prev)/dt); otherwise crawl_state_step_simple uses (d - d_prev)/dt."""
    theta = compute_theta(phi1, phi2, beta)
    theta_dot = compute_theta_dot(phi1, phi2, phi1_dot, phi2_dot, beta)
    return compute_d_dot(phi1, phi2, theta, phi1_dot, phi2_dot, theta_dot, l, beta)


# -----------------------------------------------------------------------------
# GROUP 4: Mesh state → paper angles (φ₁, φ₂)
# Solver: called at the start of each step in eval_particle_ground_contact_forces and in
# _crawl_d_from_state. The solver passes mean positions of left/right contact and
# left/right joint particles (from set_crawl_contact_groups). Result (phi1, phi2) feeds
# crawl_state_step_simple and _crawl_d_from_state.
# -----------------------------------------------------------------------------

def joint_angles_from_positions(
    left_contact: np.ndarray,
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    right_contact: np.ndarray,
    crawl_axis: int = 0,
    vertical_axis: int = 2,
) -> tuple[float, float]:
    """
    Compute joint angles φ1, φ2 from 4 node positions in the 2D plane (crawl_axis × vertical_axis).
    Solver: provides the only link from mesh particle positions to the paper model; solver
    uses crawl_axis from set_gait_params (e.g. 1 for Y) and vertical_axis 2 (Z). Interior
    angle at rest (flat) = π.
    """
    # Work in 2D (crawl_axis, vertical_axis)
    def p2(u: np.ndarray) -> tuple[float, float]:
        return float(u[crawl_axis]), float(u[vertical_axis])

    p1 = p2(left_contact)
    p2_ = p2(left_joint)
    p3 = p2(right_joint)
    p4 = p2(right_contact)

    # Vectors from joint along each link (pointing outward from joint)
    # Left joint: link to left = p2 - p1, link to right = p3 - p2
    v1 = np.array([p2_[0] - p1[0], p2_[1] - p1[1]], dtype=np.float64)
    v2 = np.array([p3[0] - p2_[0], p3[1] - p2_[1]], dtype=np.float64)
    n1 = v1 / (np.linalg.norm(v1) + 1e-12)
    n2 = v2 / (np.linalg.norm(v2) + 1e-12)
    dot1 = np.clip(np.dot(n1, n2), -1.0, 1.0)
    # Interior angle at left joint: π when flat (vectors opposite in sense from joint)
    phi1 = np.pi - np.arccos(dot1)

    # Right joint: link to left = p3 - p2, link to right = p4 - p3
    w1 = np.array([p3[0] - p2_[0], p3[1] - p2_[1]], dtype=np.float64)
    w2 = np.array([p4[0] - p3[0], p4[1] - p3[1]], dtype=np.float64)
    nw1 = w1 / (np.linalg.norm(w1) + 1e-12)
    nw2 = w2 / (np.linalg.norm(w2) + 1e-12)
    dot2 = np.clip(np.dot(nw1, nw2), -1.0, 1.0)
    phi2 = np.pi - np.arccos(dot2)

    return float(phi1), float(phi2)


# -----------------------------------------------------------------------------
# GROUP 5: Stick-slip state machine (one-step decision)
# Solver: called once per step from eval_particle_ground_contact_forces. Uses all of the
# above (l, θ, d, xc, fn, Δ, ḋ) to decide: which foot slips (slip_leg), slip force magnitude
# and sign (ft_signed, slip_dir), current d, fn1/fn2, and whether slip_dir flipped (for
# min-dwell). The solver then: passes slip_leg, ft_mag, slip_dir to the contact kernel,
# stores d as d_prev for next step, applies kinematic displacement ±Δd using d and slip_dir.
# -----------------------------------------------------------------------------

def crawl_state_step_simple(
    phi1: float,
    phi2: float,
    d_prev: float,
    dt: float,
    *,
    M: float,
    L: float,
    beta: float,
    mu: float,
    g: float,
    phi1_dot: float | None = None,
    phi2_dot: float | None = None,
    previous_slip_dir: int | None = None,
    period: float | None = None,
    phase_in_cycle: float | None = None,
    last_flip_phase: float | None = None,
    min_dwell_phase: float = 0.05,
) -> tuple[Literal["SLIP_STICK", "STICK_SLIP"], float, int, float, float, float, bool]:
    """
    One-step stick-slip decision: which foot slips and with what signed tangential force.

    Returns:
        state: "SLIP_STICK" (left slips) or "STICK_SLIP" (right slips).
        ft_signed: magnitude × slip_dir for the slipping foot.
        slip_dir: +1 or -1 from sign(ḋ), used by solver for force direction and body displacement.
        d: current contact distance (solver uses for d_prev next step and for Δd = d - d_prev).
        fn1, fn2: normal forces at left/right (solver logs as fn_left_raw, fn_right_raw).
        flip_occurred: True if slip_dir changed this step (solver uses for min-dwell / last_flip_phase).

    Solver usage: called from eval_particle_ground_contact_forces with (phi1, phi2) from
    joint_angles_from_positions, d_prev from last step, dt, and M,L,beta,mu,g from set_gait_params.
    Solver passes phi1_dot, phi2_dot when available for accurate ḋ; else (d-d_prev)/dt is used.
    period, phase_in_cycle, last_flip_phase, min_dwell_phase come from the example (gait timing).
    """
    l = compute_l(M, L, beta)
    theta = compute_theta(phi1, phi2, beta)
    d = compute_d(phi1, phi2, theta, l, beta)
    xc = compute_xc(phi1, phi2, theta, l, beta)
    fn1, fn2 = compute_fn(xc, d, M, g)
    delta = compute_delta_direct(phi1, phi2, theta, l, beta)
    if phi1_dot is not None and phi2_dot is not None:
        d_dot = compute_d_dot_from_angular_velocities(
            phi1, phi2, phi1_dot, phi2_dot, l, beta
        )
    else:
        d_dot = (d - d_prev) / dt if dt > 0 else 0.0

    if delta > 0:
        state = "SLIP_STICK"
        ft_mag = mu * fn1
    else:
        state = "STICK_SLIP"
        ft_mag = mu * fn2

    # Hysteresis band: only flip sign when ḋ clearly crosses zero so f_t is ~one rect per half-cycle (no chattering).
    # Numerical (φ−φ_prev)/dt is noisy; too small a band causes many sign flips per quarter cycle.
    _d_dot_eps_fallback = 5e-2
    if period is not None and period > 0:
        # Characteristic |ḋ| ~ l*ω = l*2π/period. Use ~35% as dead band to avoid chattering from noise.
        alpha = 0.35
        d_dot_eps = alpha * l * (2.0 * np.pi) / period
        d_dot_eps = max(d_dot_eps, 1e-5)  # avoid zero
    else:
        d_dot_eps = _d_dot_eps_fallback
    if np.abs(d_dot) < d_dot_eps:
        slip_dir = previous_slip_dir if previous_slip_dir is not None else 1
        flip_occurred = False
    else:
        new_slip = 1 if d_dot >= 0 else -1
        current = previous_slip_dir if previous_slip_dir is not None else 1
        # Minimum dwell: don't flip again until at least min_dwell_phase (e.g. 5%) of cycle has passed.
        # Also: at start of cycle (no prior flip), don't allow flip to -1 before phase >= min_dwell_phase,
        # so the first positive rect isn't a sliver (e.g. 0 to 0.02).
        if phase_in_cycle is not None and min_dwell_phase > 0:
            if last_flip_phase is not None:
                delta_phase = (phase_in_cycle - last_flip_phase) % 1.0
                allow_flip = delta_phase >= min_dwell_phase
            else:
                # First flip: allow flip to -1 only after min_dwell_phase so first rect is at least 5%
                allow_flip = (phase_in_cycle >= min_dwell_phase) or (new_slip == 1)
            if new_slip != current and not allow_flip:
                slip_dir = current
                flip_occurred = False
            else:
                slip_dir = new_slip
                flip_occurred = new_slip != current
        else:
            slip_dir = new_slip
            flip_occurred = slip_dir != current
    ft_signed = ft_mag * slip_dir
    return state, ft_signed, slip_dir, d, fn1, fn2, flip_occurred
