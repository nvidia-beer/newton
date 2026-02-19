# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Crawlable soft body solver: inflatable + paper stick-slip ground contact (Gamus et al.).

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.inflatable import SolverInflatable
from newton._src.solvers.soft.kernels import eval_particle_ground_contacts

from .paper_model import (
    PAPER_BETA,
    PAPER_G,
    PAPER_K,
    PAPER_L,
    PAPER_M,
    PAPER_MU,
    compute_d,
    compute_l,
    compute_theta,
    crawl_state_step,
    joint_angles_from_positions,
)
from .kernels_crawlable import (
    apply_crawl_kinematic_displacement,
    eval_particle_ground_contacts_crawl,
)


class SolverCrawlable(SolverInflatable):
    """
    Crawlable solver: inflatable + paper stick-slip ground contact (arXiv:1911.05227).

    Call set_crawl_contact_groups() and set_gait_params() / set_crawl_time() from the
    example. When crawl is enabled, ground contact uses the hybrid stick-slip model
    (Δ-based slip leg, f_t = μ f_n,sign(ḋ)) so the robot can crawl.

    How crawl motion is produced (paper kinematics)
    -----------------------------------------------
    The paper (Gamus et al.) uses a quasistatic hybrid model: at each step they solve
    F(Φ, τ, f_t) = 0 and then "update contact positions: sticking contact unchanged,
    slipping contact position changes by Δd in slip direction". So the body displacement
    comes from kinematics (Δd = d_new - d_prev), not from integrating slip forces.
    We apply that same rule: after the implicit step we displace all particles by
    ±Δd along the crawl axis (left slip → body_disp = -Δd, right slip → +Δd). This
    is the fundamental physics from the paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crawl_enabled = False
        self._crawl_left_indices: list[int] | None = None
        self._crawl_right_indices: list[int] | None = None
        self._crawl_left_joint_indices: list[int] | None = None
        self._crawl_right_joint_indices: list[int] | None = None
        self._crawl_particle_in_left: wp.array | None = None
        self._crawl_particle_in_right: wp.array | None = None
        self._crawl_d_prev = 0.0
        self._crawl_state: str = "STICK_STICK"
        self._crawl_slip_direction = 1
        self._crawl_time = 0.0
        self._crawl_gamma = np.pi / 2.0
        self._crawl_A = np.pi / 6.0
        self._crawl_omega = 2.0 * np.pi * 0.1
        self._crawl_psi = np.pi / 4.0
        self._crawl_k = PAPER_K
        self._crawl_M: float | None = None
        self._crawl_L: float | None = None
        self._crawl_beta = PAPER_BETA
        self._crawl_mu = PAPER_MU
        self._crawl_g = PAPER_G
        self._crawl_axis = 0
        self._crawl_use_full_state_machine = True
        self._crawl_slip_force_scale = 1.0
        self._crawl_direction = 1.0  # +1 = Y+ (or +X if axis 0), -1 = Y- (or -X)
        self._crawl_last_slip_leg: int = 2  # 0=left slip, 1=right slip, 2=both stick
        self._crawl_apply_kinematic_disp: bool = True  # apply paper's ±Δd position update when slipping
        # Paper: Δ = x_c − d/2; when Δ=0 switching occurs (equal f_n). When flat (φ₁=φ₂=π), Δ=0.
        self._crawl_kick_min_bend: float = 0.08  # rad; when max|φ−π| < this, treat as stick-stick (Δ≈0), no slip force/displacement

    def set_crawl_contact_groups(
        self,
        model: Model,
        left_contact_indices: list[int],
        right_contact_indices: list[int],
        left_joint_indices: list[int],
        right_joint_indices: list[int],
    ) -> None:
        """Set which particles belong to left/right contact and joint (for φ₁, φ₂)."""
        self._crawl_left_indices = list(left_contact_indices)
        self._crawl_right_indices = list(right_contact_indices)
        self._crawl_left_joint_indices = list(left_joint_indices)
        self._crawl_right_joint_indices = list(right_joint_indices)
        n = model.particle_count
        in_left = np.zeros(n, dtype=np.int32)
        in_right = np.zeros(n, dtype=np.int32)
        for i in left_contact_indices:
            if 0 <= i < n:
                in_left[i] = 1
        for i in right_contact_indices:
            if 0 <= i < n:
                in_right[i] = 1
        self._crawl_particle_in_left = wp.array(in_left, dtype=wp.int32, device=model.device)
        self._crawl_particle_in_right = wp.array(in_right, dtype=wp.int32, device=model.device)
        self._crawl_enabled = True
        self._crawl_d_prev = 0.0
        self._crawl_state = "STICK_STICK"
        self._crawl_slip_direction = 1

    def set_crawl_time(self, t: float) -> None:
        """Set current time for gait reference angles (call before each step)."""
        self._crawl_time = t

    def set_gait_params(
        self,
        gamma: float = np.pi / 2.0,
        A: float = np.pi / 6.0,
        omega: float | None = None,
        psi: float = np.pi / 4.0,
        freq_hz: float | None = None,
        k: float = PAPER_K,
        M: float = PAPER_M,
        L: float = PAPER_L,
        beta: float = PAPER_BETA,
        mu: float = PAPER_MU,
        g: float = PAPER_G,
        crawl_axis: int = 0,
        use_full_state_machine: bool = True,
        slip_force_scale: float = 1.0,
        crawl_direction: float = 1.0,
    ) -> None:
        """Set paper gait and physical parameters. crawl_direction: +1 = move toward +crawl_axis (e.g. Y+), -1 = toward -axis (e.g. Y-)."""
        self._crawl_gamma = gamma
        self._crawl_A = A
        if omega is not None:
            self._crawl_omega = omega
        elif freq_hz is not None:
            self._crawl_omega = 2.0 * np.pi * freq_hz
        self._crawl_psi = psi
        self._crawl_k = k
        self._crawl_M = M
        self._crawl_L = L
        self._crawl_beta = beta
        self._crawl_mu = mu
        self._crawl_g = g
        self._crawl_axis = crawl_axis
        self._crawl_use_full_state_machine = use_full_state_machine
        self._crawl_slip_force_scale = max(0.01, float(slip_force_scale))
        self._crawl_direction = 1.0 if float(crawl_direction) >= 0 else -1.0

    def eval_particle_ground_contact_forces(self, model: Model, control: Control, state: State):
        """Ground contact: paper stick-slip when crawl enabled, else default Coulomb."""
        if self._ground_plane is None:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if not self._crawl_enabled or self._crawl_particle_in_left is None or self._crawl_particle_in_right is None:
            return self._eval_default_ground_forces(model, state)

        q = state.particle_q.numpy()
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        n = model.particle_count

        left_idx = self._crawl_left_indices
        right_idx = self._crawl_right_indices
        left_j = self._crawl_left_joint_indices
        right_j = self._crawl_right_joint_indices
        if not left_idx or not right_idx or not left_j or not right_j:
            return self._eval_default_ground_forces(model, state)

        left_contact = np.mean(q[left_idx], axis=0)
        right_contact = np.mean(q[right_idx], axis=0)
        left_joint = np.mean(q[left_j], axis=0)
        right_joint = np.mean(q[right_j], axis=0)

        phi1, phi2 = joint_angles_from_positions(
            left_contact, left_joint, right_joint, right_contact,
            crawl_axis=self._crawl_axis, vertical_axis=2,
        )

        M = self._crawl_M if self._crawl_M is not None else getattr(self, "mass", PAPER_M)
        L = self._crawl_L if self._crawl_L is not None else PAPER_L
        dt = getattr(self, "_step_dt", 1.0 / 60.0)

        if self._crawl_use_full_state_machine:
            state_out, ft_mag, slip_dir, d = crawl_state_step(
                phi1, phi2,
                self._crawl_d_prev, dt, self._crawl_time,
                self._crawl_state, self._crawl_slip_direction,
                M=M, L=L, beta=self._crawl_beta, k=self._crawl_k,
                mu=self._crawl_mu, g=self._crawl_g,
                gamma=self._crawl_gamma, A=self._crawl_A,
                omega=self._crawl_omega, psi=self._crawl_psi,
            )
            self._crawl_state = state_out
            self._crawl_slip_direction = slip_dir
        else:
            from .paper_model import crawl_state_step_simple
            state_out, ft_mag, slip_dir, d = crawl_state_step_simple(
                phi1, phi2, self._crawl_d_prev, dt,
                M=M, L=L, beta=self._crawl_beta, mu=self._crawl_mu, g=self._crawl_g,
            )
        self._crawl_d_prev = d

        if state_out == "STICK_STICK":
            slip_leg = 2
        elif state_out == "SLIP_STICK":
            slip_leg = 0
        else:
            slip_leg = 1
        n_left = len(left_idx)
        n_right = len(right_idx)

        # When worm is flat (Δ ≈ 0), paper gives stick-stick: no slip force, no kinematic displacement.
        bend = max(abs(phi1 - np.pi), abs(phi2 - np.pi))
        min_bend = getattr(self, "_crawl_kick_min_bend", 0.08)
        if bend < min_bend:
            slip_leg = 2
            ft_mag = 0.0

        g = self._get_gravity_vec3(model)
        gravity_mag = float((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5)
        if gravity_mag < 1e-9:
            gravity_mag = 9.81

        self._crawl_last_slip_leg = slip_leg
        self._crawl_apply_kinematic_disp = (slip_leg != 2 and bend >= min_bend)

        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=eval_particle_ground_contacts_crawl,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_inv_mass,
                model.particle_flags,
                self._crawl_particle_in_left,
                self._crawl_particle_in_right,
                self._ground_ke,
                self._ground_kd,
                self._ground_kf,
                self._ground_mu,
                self._ground_plane,
                gravity_mag,
                slip_leg,
                float(ft_mag),
                float(self._crawl_direction),
                n_left,
                n_right,
                self._crawl_axis,
                self._crawl_slip_force_scale,
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def _eval_default_ground_forces(self, model: Model, state: State):
        """Default Coulomb ground contact (parent behavior)."""
        g = self._get_gravity_vec3(model)
        gravity_mag = float((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5)
        if gravity_mag < 1e-9:
            gravity_mag = 9.81
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=eval_particle_ground_contacts,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.particle_radius,
                model.particle_inv_mass,
                model.particle_flags,
                self._ground_ke,
                self._ground_kd,
                self._ground_kf,
                self._ground_mu,
                self._ground_plane,
                gravity_mag,
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """Same as parent; when crawl enabled, apply paper kinematic displacement ±Δd then optional velocity kick."""
        self._step_dt = dt
        state_out = super().step(state_in, state_out, control, contacts, dt)

        if not self._crawl_enabled or self._crawl_left_indices is None or self._crawl_right_indices is None:
            return state_out
        if not self._crawl_left_joint_indices or not self._crawl_right_joint_indices:
            return state_out

        # Paper kinematics: "Update contact positions: sticking contact unchanged,
        # slipping contact position changes by Δd in slip direction." So body displaces by ±Δd.
        apply_disp = getattr(self, "_crawl_apply_kinematic_disp", True)
        if apply_disp and abs((d_new := self._crawl_d_from_state(state_out)) - self._crawl_d_prev) > 1e-9:
            delta_d = d_new - self._crawl_d_prev
            slip_leg = self._crawl_last_slip_leg
            if slip_leg == 0:
                body_disp = -delta_d * float(self._crawl_direction)
            elif slip_leg == 1:
                body_disp = delta_d * float(self._crawl_direction)
            else:
                body_disp = 0.0
            if abs(body_disp) > 1e-12:
                wp.launch(
                    kernel=apply_crawl_kinematic_displacement,
                    dim=self.model.particle_count,
                    inputs=[
                        state_out.particle_q,
                        self._crawl_axis,
                        float(body_disp),
                    ],
                    device=self.model.device,
                )
        return state_out

    def _crawl_d_from_state(self, state: State) -> float:
        """Contact distance d from current geometry (paper formula)."""
        q = state.particle_q.numpy()
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        left_contact = np.mean(q[self._crawl_left_indices], axis=0)
        right_contact = np.mean(q[self._crawl_right_indices], axis=0)
        left_joint = np.mean(q[self._crawl_left_joint_indices], axis=0)
        right_joint = np.mean(q[self._crawl_right_joint_indices], axis=0)
        phi1, phi2 = joint_angles_from_positions(
            left_contact, left_joint, right_joint, right_contact,
            crawl_axis=self._crawl_axis, vertical_axis=2,
        )
        M = self._crawl_M if self._crawl_M is not None else PAPER_M
        L = self._crawl_L if self._crawl_L is not None else PAPER_L
        beta = self._crawl_beta
        l = compute_l(M, L, beta)
        theta = compute_theta(phi1, phi2, beta)
        return float(compute_d(phi1, phi2, theta, l, beta))
