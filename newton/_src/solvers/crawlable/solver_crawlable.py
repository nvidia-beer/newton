# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Crawlable soft body solver: inflatable + paper stick-slip ground contact (Gamus et al.).

from __future__ import annotations

import numpy as np
import warp as wp

from warp.optim.linear import preconditioner

from newton._src.sim import Contacts, Control, Model, State
from newton._src.solvers.inflatable import SolverInflatable
from newton._src.solvers.soft.kernels import eval_particle_ground_contacts

# Defaults from paper Table I; override via set_gait_params(M=..., L=..., ...).
_DEFAULT_M = 0.052  # kg
_DEFAULT_L = 0.120  # m
_DEFAULT_BETA = 2.0
_DEFAULT_K = 0.1677  # Nm/rad
_DEFAULT_MU = 0.389
_DEFAULT_G = 9.81
# Dimensionless k/(M·g·L) from paper; when use_paper_ratios we set k = this * M * g * L.
_K_OVER_MGL = 0.1677 / (0.052 * 9.81 * 0.120)

from .paper_model import (
    compute_d,
    compute_l,
    compute_theta,
    crawl_state_step_simple,
    joint_angles_from_positions,
)
from .kernels_crawlable import (
    apply_crawl_kinematic_displacement,
    apply_crawl_kinematic_from_buf,
    crawl_joint_angles_from_means,
    crawl_paper_state_step,
    crawl_reduce_init,
    crawl_reduce_positions,
    eval_particle_ground_contacts_crawl,
    eval_particle_ground_contacts_crawl_from_buf,
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
        self._crawl_left_indices: list[int] | None = None
        self._crawl_right_indices: list[int] | None = None
        super().__init__(*args, **kwargs)
        self._crawl_enabled = False
        self._crawl_left_joint_indices: list[int] | None = None
        self._crawl_right_joint_indices: list[int] | None = None
        self._crawl_particle_in_left: wp.array | None = None
        self._crawl_particle_in_right: wp.array | None = None
        self._crawl_d_prev = 0.0
        self._crawl_time = 0.0
        self._crawl_gamma = np.pi / 2.0
        self._crawl_A = np.pi / 6.0
        self._crawl_omega = 2.0 * np.pi * 0.1
        self._crawl_psi = np.pi / 4.0
        self._crawl_k = _DEFAULT_K
        self._crawl_M: float | None = None
        self._crawl_L: float | None = None
        self._crawl_beta = _DEFAULT_BETA
        self._crawl_mu = _DEFAULT_MU
        self._crawl_g = _DEFAULT_G
        self._crawl_axis = 0
        self._crawl_slip_force_scale = 1.0
        self._crawl_direction = 1.0  # +1 = Y+ (or +X if axis 0), -1 = Y- (or -X)
        self._crawl_last_slip_leg: int = 2  # 0=left slip, 1=right slip, 2=both stick
        self._crawl_apply_kinematic_disp: bool = True  # apply paper's ±Δd position update when slipping
        self._crawl_last_fn_left_raw: float = 0.0  # sum F·n left (for CSV only)
        self._crawl_last_fn_right_raw: float = 0.0  # sum F·n right (for CSV only)
        self._crawl_last_ft_signed: float = 0.0  # tangential force for CSV (Fig. 6)
        self._crawl_last_slip_dir: int | None = None  # ±1; hysteresis when d_dot ≈ 0 to avoid single-frame flips
        self._crawl_phase_in_cycle: float | None = None  # 0..1 for min-dwell (set by example before step)
        self._crawl_last_flip_phase: float | None = None  # phase when slip_dir last changed
        self._crawl_phi1_prev: float | None = None  # previous step joint angles for ḋ from angular velocities
        self._crawl_phi2_prev: float | None = None
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
        in_left_j = np.zeros(n, dtype=np.int32)
        in_right_j = np.zeros(n, dtype=np.int32)
        for i in left_contact_indices:
            if 0 <= i < n:
                in_left[i] = 1
        for i in right_contact_indices:
            if 0 <= i < n:
                in_right[i] = 1
        for i in left_joint_indices:
            if 0 <= i < n:
                in_left_j[i] = 1
        for i in right_joint_indices:
            if 0 <= i < n:
                in_right_j[i] = 1
        self._crawl_particle_in_left = wp.array(in_left, dtype=wp.int32, device=model.device)
        self._crawl_particle_in_right = wp.array(in_right, dtype=wp.int32, device=model.device)
        self._crawl_in_left_contact = wp.array(in_left, dtype=wp.int32, device=model.device)
        self._crawl_in_right_contact = wp.array(in_right, dtype=wp.int32, device=model.device)
        self._crawl_in_left_joint = wp.array(in_left_j, dtype=wp.int32, device=model.device)
        self._crawl_in_right_joint = wp.array(in_right_j, dtype=wp.int32, device=model.device)
        # state_buf: [slip_leg, ft_mag, slip_direction_sign, d_current, d_prev] (5 for device-full + displacement)
        self._crawl_state_buf = wp.array([2.0, 0.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=model.device)
        self._crawl_reduce_buf = wp.zeros(16, dtype=wp.float32, device=model.device)
        self._crawl_angles_buf = wp.zeros(2, dtype=wp.float32, device=model.device)
        # persistent_buf: [d_prev, phi1_prev, phi2_prev, last_slip_dir, last_flip_phase]
        self._crawl_persistent_buf = wp.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=wp.float32, device=model.device)
        # params_buf: [period, phase_in_cycle] (set by example before each frame/launch); last_flip_phase in persistent_buf[4]
        self._crawl_params_buf = wp.array([1.0, 0.0], dtype=wp.float32, device=model.device)
        self._crawl_use_state_buf = False
        self._crawl_enabled = True
        self._crawl_d_prev = 0.0
        self._crawl_last_slip_dir = None
        self._crawl_phase_in_cycle = None
        self._crawl_last_flip_phase = None
        self._crawl_phi1_prev = None
        self._crawl_phi2_prev = None

    def set_crawl_time(self, t: float) -> None:
        """Set current time for gait reference angles (call before each step)."""
        self._crawl_time = t

    def set_crawl_phase(self, phase_0_to_1: float) -> None:
        """Set phase in current cycle (0..1) for min-dwell; call before each step to avoid ft chattering."""
        self._crawl_phase_in_cycle = float(phase_0_to_1)

    def set_crawl_device_params(self, period: float, phase_in_cycle: float) -> None:
        """Set period, phase_in_cycle on device (for graph capture; call before each frame/launch). last_flip_phase lives in persistent_buf[4]."""
        if getattr(self, "_crawl_params_buf", None) is not None:
            arr = wp.array([float(period), float(phase_in_cycle)], dtype=wp.float32, device=self._crawl_params_buf.device)
            self._crawl_params_buf.assign(arr)

    def set_crawl_use_state_buf(self, use: bool) -> None:
        """When True, ground contact reads slip state from _crawl_state_buf (no host sync, graph-capture safe)."""
        self._crawl_use_state_buf = use

    def set_gait_params(
        self,
        gamma: float = np.pi / 2.0,
        A: float = np.pi / 6.0,
        omega: float | None = None,
        psi: float = np.pi / 4.0,
        freq_hz: float | None = None,
        k: float | None = None,
        M: float = _DEFAULT_M,
        L: float = _DEFAULT_L,
        beta: float = _DEFAULT_BETA,
        mu: float = _DEFAULT_MU,
        g: float = _DEFAULT_G,
        crawl_axis: int = 0,
        slip_force_scale: float = 1.0,
        crawl_direction: float = 1.0,
        use_paper_ratios: bool = True,
        contact_constraint_stiffness: float = 0.0,
    ) -> None:
        """Set paper gait and physical parameters. L = total length along crawl axis (mesh extent).
        If use_paper_ratios is True (default), k is set from the paper ratio k/(M·g·L) so dynamics match at any scale.
        crawl_direction: +1 = toward +crawl_axis, -1 = toward -axis.
        contact_constraint_stiffness: if > 0, add implicit (matrix) penalty so left and right contact mean height match (paper Eq. 2)."""
        self._crawl_gamma = gamma
        self._crawl_A = A
        if omega is not None:
            self._crawl_omega = omega
        elif freq_hz is not None:
            self._crawl_omega = 2.0 * np.pi * freq_hz
        self._crawl_psi = psi
        if use_paper_ratios:
            self._crawl_k = _K_OVER_MGL * M * g * L
        else:
            self._crawl_k = k if k is not None else _DEFAULT_K
        self._crawl_M = M
        self._crawl_L = L
        self._crawl_beta = beta
        self._crawl_mu = mu
        self._crawl_g = g
        self._crawl_axis = crawl_axis
        self._crawl_slip_force_scale = max(0.01, float(slip_force_scale))
        self._crawl_direction = 1.0 if float(crawl_direction) >= 0 else -1.0
        self._crawl_contact_constraint_stiffness = max(0.0, float(contact_constraint_stiffness))

    def eval_particle_ground_contact_forces(self, model: Model, control: Control, state: State):
        """Ground contact: paper stick-slip when crawl enabled, else default Coulomb."""
        if self._ground_plane is None:
            return wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        if not self._crawl_enabled or self._crawl_particle_in_left is None or self._crawl_particle_in_right is None:
            return self._eval_default_ground_forces(model, state)

        if getattr(self, "_crawl_use_state_buf", False) and getattr(self, "_crawl_state_buf", None) is not None:
            if getattr(self, "_crawl_reduce_buf", None) is not None:
                self._update_crawl_state_on_device(model, state)
            return self._eval_crawl_forces_from_buf(model, state)

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

        M = self._crawl_M if self._crawl_M is not None else getattr(self, "mass", _DEFAULT_M)
        L = self._crawl_L if self._crawl_L is not None else _DEFAULT_L
        dt = getattr(self, "_step_dt", 1.0 / 60.0)
        period = (2.0 * np.pi) / self._crawl_omega if self._crawl_omega > 1e-12 else None

        # Use angular velocities for ḋ so slip_dir flips once per half-cycle (ft shows one rectangle per cycle).
        phi1_dot = (phi1 - self._crawl_phi1_prev) / dt if self._crawl_phi1_prev is not None else None
        phi2_dot = (phi2 - self._crawl_phi2_prev) / dt if self._crawl_phi2_prev is not None else None

        state_out, ft_signed, slip_dir, d, fn1, fn2, flip_occurred = crawl_state_step_simple(
            phi1, phi2, self._crawl_d_prev, dt,
            M=M, L=L, beta=self._crawl_beta, mu=self._crawl_mu, g=self._crawl_g,
            phi1_dot=phi1_dot,
            phi2_dot=phi2_dot,
            previous_slip_dir=self._crawl_last_slip_dir,
            period=period,
            phase_in_cycle=getattr(self, "_crawl_phase_in_cycle", None),
            last_flip_phase=getattr(self, "_crawl_last_flip_phase", None),
            min_dwell_phase=0.05,
        )
        self._crawl_d_prev = d
        self._crawl_phi1_prev = float(phi1)
        self._crawl_phi2_prev = float(phi2)
        self._crawl_last_slip_dir = slip_dir
        if flip_occurred and self._crawl_phase_in_cycle is not None:
            self._crawl_last_flip_phase = self._crawl_phase_in_cycle

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
            ft_signed = 0.0

        ft_mag = abs(ft_signed)
        g = self._get_gravity_vec3(model)
        gravity_mag = float((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5)
        if gravity_mag < 1e-9:
            gravity_mag = 9.81

        self._crawl_last_slip_leg = slip_leg
        self._crawl_apply_kinematic_disp = (slip_leg != 2 and bend >= min_bend)
        # CSV "ft" column: discrete slip direction from model (-1, 0, 1). Log model slip_dir so plot shows waveform
        # even when we force stick-stick for flat worm (slip_leg=2); use 0 only when model says stick-stick.
        self._crawl_last_ft_signed = float(slip_dir) if state_out != "STICK_STICK" else 0.0

        # Paper: f_t = μ f_{n,s} sign(ḋ); apply force in slip direction (sign(d_dot)) × crawl axis
        slip_direction_sign = float(slip_dir * self._crawl_direction)

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
                slip_direction_sign,
                n_left,
                n_right,
                self._crawl_axis,
                self._crawl_slip_force_scale,
            ],
            outputs=[forces],
            device=model.device,
        )
        # Sum F·n per group (raw normal force magnitude) for CSV logging.
        if hasattr(self._ground_plane, "numpy"):
            plane = np.array(self._ground_plane.numpy(), dtype=np.float64)
        else:
            plane = np.array(self._ground_plane, dtype=np.float64)
        n_vec = np.array([float(plane[0]), float(plane[1]), float(plane[2])], dtype=np.float64)
        n_sq = float(np.dot(n_vec, n_vec))
        if n_sq < 1e-18:
            n_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n_sq = 1.0
        forces_np = np.array(forces.numpy(), dtype=np.float64)
        if forces_np.ndim == 1:
            forces_np = forces_np.reshape(-1, 3)
        fn_left_raw = sum(float(np.dot(forces_np[i], n_vec)) for i in left_idx) / n_sq
        fn_right_raw = sum(float(np.dot(forces_np[i], n_vec)) for i in right_idx) / n_sq
        self._crawl_last_fn_left_raw = float(fn_left_raw)
        self._crawl_last_fn_right_raw = float(fn_right_raw)
        return forces

    def _eval_crawl_forces_from_buf(self, model: Model, state: State):
        """Crawl ground forces reading slip state from _crawl_state_buf (no host sync, graph-capture safe)."""
        gravity_mag = 9.81
        n_left = len(self._crawl_left_indices)
        n_right = len(self._crawl_right_indices)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=eval_particle_ground_contacts_crawl_from_buf,
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
                self._crawl_state_buf,
                n_left,
                n_right,
                self._crawl_axis,
                self._crawl_slip_force_scale,
            ],
            outputs=[forces],
            device=model.device,
        )
        return forces

    def _update_crawl_state_on_device(self, model: Model, state: State) -> None:
        """Run reduce -> joint angles -> paper state on device (no host sync; graph-capture safe)."""
        dev = model.device
        wp.launch(kernel=crawl_reduce_init, dim=16, inputs=[self._crawl_reduce_buf], device=dev)
        wp.launch(
            kernel=crawl_reduce_positions,
            dim=model.particle_count,
            inputs=[
                state.particle_q,
                self._crawl_in_left_contact,
                self._crawl_in_right_contact,
                self._crawl_in_left_joint,
                self._crawl_in_right_joint,
                self._crawl_reduce_buf,
            ],
            device=dev,
        )
        wp.launch(
            kernel=crawl_joint_angles_from_means,
            dim=1,
            inputs=[self._crawl_reduce_buf, self._crawl_angles_buf, self._crawl_axis],
            device=dev,
        )
        M = self._crawl_M if self._crawl_M is not None else _DEFAULT_M
        L = self._crawl_L if self._crawl_L is not None else _DEFAULT_L
        g = self._crawl_g
        dt = getattr(self, "_step_dt", 1.0 / 60.0)
        min_dwell = 0.05
        min_bend = getattr(self, "_crawl_kick_min_bend", 0.08)
        wp.launch(
            kernel=crawl_paper_state_step,
            dim=1,
            inputs=[
                self._crawl_angles_buf,
                self._crawl_persistent_buf,
                self._crawl_params_buf,
                self._crawl_state_buf,
                M, L, self._crawl_beta, self._crawl_mu, g,
                dt, min_dwell, min_bend, float(self._crawl_direction),
            ],
            device=dev,
        )

    def prepare_crawl_state_from_state(self, model: Model, state: State) -> None:
        """
        Compute crawl slip state from current state (uses host sync), update internal state and _crawl_state_buf.
        Call this before each capture_launch when using graph capture with stick-slip so the replayed graph uses the current frame's crawl state.
        """
        if not self._crawl_enabled or self._crawl_state_buf is None:
            return
        q = state.particle_q.numpy()
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        n = model.particle_count
        left_idx = self._crawl_left_indices
        right_idx = self._crawl_right_indices
        left_j = self._crawl_left_joint_indices
        right_j = self._crawl_right_joint_indices
        if not left_idx or not right_idx or not left_j or not right_j:
            return
        left_contact = np.mean(q[left_idx], axis=0)
        right_contact = np.mean(q[right_idx], axis=0)
        left_joint = np.mean(q[left_j], axis=0)
        right_joint = np.mean(q[right_j], axis=0)
        phi1, phi2 = joint_angles_from_positions(
            left_contact, left_joint, right_joint, right_contact,
            crawl_axis=self._crawl_axis, vertical_axis=2,
        )
        M = self._crawl_M if self._crawl_M is not None else getattr(self, "mass", _DEFAULT_M)
        L = self._crawl_L if self._crawl_L is not None else _DEFAULT_L
        dt = getattr(self, "_step_dt", 1.0 / 60.0)
        period = (2.0 * np.pi) / self._crawl_omega if self._crawl_omega > 1e-12 else None
        phi1_dot = (phi1 - self._crawl_phi1_prev) / dt if self._crawl_phi1_prev is not None else None
        phi2_dot = (phi2 - self._crawl_phi2_prev) / dt if self._crawl_phi2_prev is not None else None
        state_out, ft_signed, slip_dir, d, fn1, fn2, flip_occurred = crawl_state_step_simple(
            phi1, phi2, self._crawl_d_prev, dt,
            M=M, L=L, beta=self._crawl_beta, mu=self._crawl_mu, g=self._crawl_g,
            phi1_dot=phi1_dot, phi2_dot=phi2_dot,
            previous_slip_dir=self._crawl_last_slip_dir,
            period=period,
            phase_in_cycle=getattr(self, "_crawl_phase_in_cycle", None),
            last_flip_phase=getattr(self, "_crawl_last_flip_phase", None),
            min_dwell_phase=0.05,
        )
        self._crawl_d_prev = d
        self._crawl_phi1_prev = float(phi1)
        self._crawl_phi2_prev = float(phi2)
        self._crawl_last_slip_dir = slip_dir
        if flip_occurred and self._crawl_phase_in_cycle is not None:
            self._crawl_last_flip_phase = self._crawl_phase_in_cycle
        if state_out == "STICK_STICK":
            slip_leg = 2
        elif state_out == "SLIP_STICK":
            slip_leg = 0
        else:
            slip_leg = 1
        bend = max(abs(phi1 - np.pi), abs(phi2 - np.pi))
        min_bend = getattr(self, "_crawl_kick_min_bend", 0.08)
        if bend < min_bend:
            slip_leg = 2
            ft_signed = 0.0
        ft_mag = abs(ft_signed)
        slip_direction_sign = float(slip_dir * self._crawl_direction)
        self._crawl_last_slip_leg = slip_leg
        self._crawl_apply_kinematic_disp = (slip_leg != 2 and bend >= min_bend)
        self._crawl_last_ft_signed = float(slip_dir) if state_out != "STICK_STICK" else 0.0
        buf = np.array([float(slip_leg), ft_mag, slip_direction_sign], dtype=np.float32)
        self._crawl_state_buf.assign(wp.array(buf, dtype=wp.float32, device=model.device))

    def apply_crawl_kinematic_after_step(self, model: Model, state_out: State) -> None:
        """Apply paper kinematic displacement ±Δd after a step (uses host sync). Call after sync when using graph + stick-slip."""
        if not getattr(self, "_crawl_use_state_buf", False):
            return
        if not self._crawl_enabled or self._crawl_left_indices is None or self._crawl_right_indices is None:
            return
        if not self._crawl_left_joint_indices or not self._crawl_right_joint_indices:
            return
        apply_disp = getattr(self, "_crawl_apply_kinematic_disp", True)
        if not apply_disp:
            return
        d_new = self._crawl_d_from_state(state_out)
        if abs(d_new - self._crawl_d_prev) <= 1e-9:
            return
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
                inputs=[state_out.particle_q, self._crawl_axis, float(body_disp)],
                device=self.model.device,
            )

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
        """Same as parent; when crawl enabled, apply paper kinematic displacement ±Δd."""
        self._step_dt = dt
        state_out = super().step(state_in, state_out, control, contacts, dt)

        if not self._crawl_enabled or self._crawl_left_indices is None or self._crawl_right_indices is None:
            return state_out
        if not self._crawl_left_joint_indices or not self._crawl_right_joint_indices:
            return state_out
        if getattr(self, "_crawl_use_state_buf", False) and getattr(self, "_crawl_reduce_buf", None) is not None:
            # Device-full mode: apply kinematic displacement from state_buf (already computed this substep)
            wp.launch(
                kernel=apply_crawl_kinematic_from_buf,
                dim=self.model.particle_count,
                inputs=[
                    state_out.particle_q,
                    self._crawl_axis,
                    self._crawl_state_buf,
                    float(self._crawl_direction),
                ],
                device=self.model.device,
            )
            return state_out
        if getattr(self, "_crawl_use_state_buf", False):
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

    def get_crawl_contact_forces(self) -> tuple[float, float, float] | None:
        """
        Return (fn_left_raw, fn_right_raw, ft_signed) from the last ground-contact evaluation.
        Raw normal force sums (F·n) per foot and tangential force [N]; for CSV logging (Fig. 6).
        Returns None if crawl contact is not enabled.
        """
        if not self._crawl_enabled:
            return None
        return (self._crawl_last_fn_left_raw, self._crawl_last_fn_right_raw, self._crawl_last_ft_signed)

    def init_crawl_persistent_from_state(self, model: Model, state: State) -> None:
        """
        Initialize device _crawl_persistent_buf from current state (d, phi1, phi2, slip_dir=1, last_flip=-1).
        Call after _reset_state() when using device-side crawl so the first substep has valid d_prev, phi1_prev, phi2_prev
        instead of zeros (which would produce huge d_dot and NaNs).
        """
        if not self._crawl_enabled or getattr(self, "_crawl_persistent_buf", None) is None:
            return
        q = state.particle_q.numpy()
        if q.ndim == 1:
            q = q.reshape(-1, 3)
        left_idx = self._crawl_left_indices
        right_idx = self._crawl_right_indices
        left_j = self._crawl_left_joint_indices
        right_j = self._crawl_right_joint_indices
        if not left_idx or not right_idx or not left_j or not right_j:
            return
        left_contact = np.mean(q[left_idx], axis=0)
        right_contact = np.mean(q[right_idx], axis=0)
        left_joint = np.mean(q[left_j], axis=0)
        right_joint = np.mean(q[right_j], axis=0)
        phi1, phi2 = joint_angles_from_positions(
            left_contact, left_joint, right_joint, right_contact,
            crawl_axis=self._crawl_axis, vertical_axis=2,
        )
        M = self._crawl_M if self._crawl_M is not None else _DEFAULT_M
        L = self._crawl_L if self._crawl_L is not None else _DEFAULT_L
        beta = self._crawl_beta
        l = compute_l(M, L, beta)
        theta = compute_theta(phi1, phi2, beta)
        d = float(compute_d(phi1, phi2, theta, l, beta))
        # persistent_buf: [d_prev, phi1_prev, phi2_prev, last_slip_dir, last_flip_phase]
        arr = wp.array(
            [d, float(phi1), float(phi2), 1.0, -1.0],
            dtype=wp.float32,
            device=self._crawl_persistent_buf.device,
        )
        self._crawl_persistent_buf.assign(arr)

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
        M = self._crawl_M if self._crawl_M is not None else _DEFAULT_M
        L = self._crawl_L if self._crawl_L is not None else _DEFAULT_L
        beta = self._crawl_beta
        l = compute_l(M, L, beta)
        theta = compute_theta(phi1, phi2, beta)
        return float(compute_d(phi1, phi2, theta, l, beta))
