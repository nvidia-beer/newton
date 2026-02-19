# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Inchworm crawling paper model (Gamus et al., arXiv:1911.05227).
# Kinematics, normal forces, slippage criterion, and hybrid stick-slip state machine.

from __future__ import annotations

import numpy as np
from typing import Literal

# Paper parameters (Table I)
PAPER_M = 0.052  # kg
PAPER_L = 0.120  # m
PAPER_BETA = 2.0
PAPER_K = 0.1677  # Nm/rad
PAPER_MU = 0.389
PAPER_G = 9.81

ContactState = Literal["STICK_STICK", "SLIP_STICK", "STICK_SLIP"]


def compute_l(M: float, L: float, beta: float) -> float:
    """Link length l = L / (2 + beta)."""
    return L / (2.0 + beta)


def compute_theta(phi1: float, phi2: float, beta: float) -> float:
    """Central link orientation: tan θ = (sin φ1 - sin φ2) / (cos φ1 + cos φ2 - β)."""
    num = np.sin(phi1) - np.sin(phi2)
    den = np.cos(phi1) + np.cos(phi2) - beta
    return np.arctan2(num, den)


def compute_d(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Horizontal distance between contact points."""
    return l * (
        beta * np.cos(theta)
        - np.cos(phi1 - theta)
        - np.cos(phi2 + theta)
    )


def compute_xc(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Horizontal distance of CoM from left contact."""
    return l / (2.0 * (2.0 + beta)) * (
        (2.0 + beta) * beta * np.cos(theta)
        - (3.0 + 2.0 * beta) * np.cos(phi1 - theta)
        - np.cos(phi2 + theta)
    )


def compute_fn(xc: float, d: float, M: float, g: float) -> tuple[float, float]:
    """Normal forces at left (1) and right (2) contacts."""
    if d <= 1e-12:
        return M * g * 0.5, M * g * 0.5
    fn1 = (1.0 - xc / d) * M * g
    fn2 = (xc / d) * M * g
    fn1 = max(0.0, fn1)
    fn2 = max(0.0, fn2)
    return fn1, fn2


def compute_delta(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Slippage criterion: Δ = x_c - d/2. Δ > 0 ⇒ left slips, Δ < 0 ⇒ right slips."""
    xc = compute_xc(phi1, phi2, theta, l, beta)
    d = compute_d(phi1, phi2, theta, l, beta)
    return xc - d * 0.5


def compute_delta_direct(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Δ from paper formula: (1+β)/(2(2+β)) * l * [cos(φ2+θ) - cos(φ1-θ)]."""
    return (1.0 + beta) / (2.0 * (2.0 + beta)) * l * (
        np.cos(phi2 + theta) - np.cos(phi1 - theta)
    )


def reference_angles(t: float, gamma: float, A: float, omega: float, psi: float) -> tuple[float, float]:
    """φ1_ref, φ2_ref from harmonic gait."""
    phi1_ref = gamma + A * np.sin(omega * t + psi / 2.0)
    phi2_ref = gamma + A * np.sin(omega * t - psi / 2.0)
    return phi1_ref, phi2_ref


def actuation_torques(t: float, k: float, gamma: float, A: float, omega: float, psi: float) -> np.ndarray:
    """τ_i = k(φ_i^ref - π)."""
    phi1_ref, phi2_ref = reference_angles(t, gamma, A, omega, psi)
    return np.array([
        k * (phi1_ref - np.pi),
        k * (phi2_ref - np.pi),
    ], dtype=np.float64)


def joint_angles_from_positions(
    left_contact: np.ndarray,
    left_joint: np.ndarray,
    right_joint: np.ndarray,
    right_contact: np.ndarray,
    crawl_axis: int = 0,
    vertical_axis: int = 2,
) -> tuple[float, float]:
    """
    Compute joint angles φ1, φ2 from 4 node positions (XZ plane).
    crawl_axis: horizontal axis along the beam (0 = x).
    vertical_axis: vertical axis (2 = z).
    Interior angle at rest (flat) = π.
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


def ft_from_stick_stick(
    phi1: float,
    phi2: float,
    tau: np.ndarray,
    k: float,
    l: float,
    beta: float,
    m: float,
    g: float,
) -> float:
    """
    In stick-stick, solve for f_t from first torque balance equation.
    F1 = τ1 - k(φ1-π) - l*sin(φ1-θ)*f_t + grav_term1 = 0  =>  f_t = (τ1 - k(φ1-π) + grav_term1) / (l*sin(φ1-θ))
    """
    theta = compute_theta(phi1, phi2, beta)
    d = compute_d(phi1, phi2, theta, l, beta)
    if d < 1e-12:
        return 0.0
    grav_term1 = (1.0 + beta) * l * l * m * g / (2.0 * d) * np.cos(phi1 - theta) * (
        beta * np.cos(theta) - 2.0 * np.cos(phi2 + theta)
    )
    denom = l * np.sin(phi1 - theta)
    if np.abs(denom) < 1e-12:
        return 0.0
    ft = (tau[0] - k * (phi1 - np.pi) + grav_term1) / denom
    return float(ft)


def crawl_state_step(
    phi1: float,
    phi2: float,
    d_prev: float,
    dt: float,
    t: float,
    state: ContactState,
    slip_direction: int,
    *,
    M: float,
    L: float,
    beta: float,
    k: float,
    mu: float,
    g: float,
    gamma: float,
    A: float,
    omega: float,
    psi: float,
    hysteresis: float = 0.0,
) -> tuple[ContactState, float, int, float]:
    """
    One step of the hybrid stick-slip state machine.
    Returns (new_state, f_t_magnitude, slip_direction, d_current).
    slip_direction: +1 or -1 (direction of ḋ when slipping).
    """
    l = compute_l(M, L, beta)
    m = M / (2.0 + beta)
    theta = compute_theta(phi1, phi2, beta)
    d = compute_d(phi1, phi2, theta, l, beta)
    xc = compute_xc(phi1, phi2, theta, l, beta)
    fn1, fn2 = compute_fn(xc, d, M, g)
    delta = compute_delta_direct(phi1, phi2, theta, l, beta)
    d_dot = (d - d_prev) / dt if dt > 0 else 0.0

    tau = actuation_torques(t, k, gamma, A, omega, psi)

    if state == "STICK_STICK":
        ft = ft_from_stick_stick(phi1, phi2, tau, k, l, beta, m, g)
        fn_min = min(fn1, fn2)
        if fn_min < 1e-12:
            fn_min = 1e-12
        # Ease transition to slip so worm moves (threshold fraction of Coulomb limit)
        slip_threshold = 0.85 * mu * fn_min + hysteresis
        if np.abs(ft) > slip_threshold:
            if fn1 < fn2:
                new_state: ContactState = "SLIP_STICK"
                slip_dir = 1 if ft > 0 else -1
                ft_mag = mu * fn1
            else:
                new_state = "STICK_SLIP"
                slip_dir = 1 if ft > 0 else -1
                ft_mag = mu * fn2
            return new_state, ft_mag, slip_dir, d
        return "STICK_STICK", ft, slip_direction, d

    # Slip state
    if state == "SLIP_STICK":
        ft_mag = mu * fn1
        slip_dir = 1 if d_dot >= 0 else -1
        if np.abs(d_dot) < 1e-9 or (d_dot * slip_direction < 0 and hysteresis <= 0):
            return "STICK_STICK", 0.0, slip_dir, d
        return "SLIP_STICK", ft_mag, slip_dir, d
    else:
        assert state == "STICK_SLIP"
        ft_mag = mu * fn2
        slip_dir = 1 if d_dot >= 0 else -1
        if np.abs(d_dot) < 1e-9 or (d_dot * slip_direction < 0 and hysteresis <= 0):
            return "STICK_STICK", 0.0, slip_dir, d
        return "STICK_SLIP", ft_mag, slip_dir, d


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
) -> tuple[Literal["SLIP_STICK", "STICK_SLIP"], float, int, float]:
    """
    Simplified state machine: use only Δ to decide which foot slips (no stick-stick solve).
    Returns (slip_leg_state, f_t_magnitude, slip_direction, d_current).
    slip_leg_state: "SLIP_STICK" = left slips, "STICK_SLIP" = right slips.
    """
    l = compute_l(M, L, beta)
    theta = compute_theta(phi1, phi2, beta)
    d = compute_d(phi1, phi2, theta, l, beta)
    xc = compute_xc(phi1, phi2, theta, l, beta)
    fn1, fn2 = compute_fn(xc, d, M, g)
    delta = compute_delta_direct(phi1, phi2, theta, l, beta)
    d_dot = (d - d_prev) / dt if dt > 0 else 0.0

    if delta > 0:
        state = "SLIP_STICK"
        ft_mag = mu * fn1
    else:
        state = "STICK_SLIP"
        ft_mag = mu * fn2

    slip_dir = 1 if d_dot >= 0 else -1
    if np.abs(d_dot) < 1e-12:
        slip_dir = 1
    return state, ft_mag, slip_dir, d
