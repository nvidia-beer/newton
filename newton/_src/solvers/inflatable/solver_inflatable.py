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

"""Inflatable soft-body solver: FEM + rest-configuration pressure control.

Overview
========

:class:`SolverInflatable` is an implicit Backward-Euler finite-element solver
for soft bodies that supports both tetrahedral (tet) and hexahedral (hex) FEM.
The element type is determined automatically from the model: when
``model.hex_count > 0`` hex elements are active; when ``model.tet_count > 0``
tet elements are active; both may be non-zero simultaneously.

Three features distinguish it from a generic FEM integrator:

* **Self-contact** — both particle/particle (hash-grid) and particle/rigid
  (penalty-based), handled by the base solver.

* **Pressure control** — bodies inflate or deflate by scaling their *rest
  configuration*, never by applying an explicit pressure force.  See §Inflation
  below for the mechanism.

* **Dirichlet pin** — a hook that lets an external rigid solver drive a subset
  of soft particles (the kinematic glue interface).

Inflation by rest-configuration scaling
========================================

Pressure ``p`` (a dimensionless volume-ratio target ``≥ 1``) rescales the
per-element rest data::

    Tet:  D_m       ←  D_m_orig / ∛p
    Hex:  inv_J0    ←  original_inv_J0 / ∛p
          det_J0_w  ←  original_det_J0_w × p
    Both: L₀        ←  L₀_orig · ∛p   (spring rest lengths)

Single vs. multi-chamber
-------------------------

* :meth:`set_pressure` — whole-body uniform pressure.
* :meth:`set_chamber_mask` + :meth:`set_chamber_pressures` — independent
  per-chamber control (e.g. two-chamber bending actuator).
* :meth:`set_target_pressure` + :meth:`apply_pressure_control` — smooth ramp.
"""

from __future__ import annotations

import math

import warp as wp

from newton._src.sim import Model, State

from .kernels import (
    compute_hex_volume_kernel,
    compute_volume_kernel,
    scale_hex_gauss_kernel,
    scale_hex_gauss_per_chamber_kernel,
    scale_spring_rest_lengths_kernel,
    scale_spring_rest_lengths_per_chamber_kernel,
    scale_tet_poses_kernel,
    scale_tet_poses_per_chamber_kernel,
)
from .solver_implicit_soft import SolverImplicitSoft


class SolverInflatable(SolverImplicitSoft):
    """Inflatable soft-body solver with rest-configuration pressure control.

    Supports tet and/or hex FEM automatically from the model.  All
    pressure-control and chamber methods work regardless of element type.

    Inflation API:

    * :meth:`set_pressure` — whole-body uniform pressure.
    * :meth:`set_chamber_mask` + :meth:`set_chamber_pressures` — independent
      per-chamber control.
    * :meth:`set_target_pressure` + :meth:`apply_pressure_control` — smooth
      ramp ``p ← p + rate · (p_target − p)``.

    Args:
        model: The Newton model owning the particles.
        fem: Accepted for backward compatibility but ignored — the active
            element types are determined by ``model.tet_count`` and
            ``model.hex_count``.
        max_volume_ratio: Upper bound for the inflation pressure ratio.
        dt: Default timestep [s].
        mass: Uniform per-particle mass [kg] (kept for API compatibility).
        preconditioner_type: Jacobi preconditioner type (``"diag"``).
        solver_type: Iterative solver — ``"cg"``, ``"bicgstab"``,
            ``"gmres"``, or ``"cr"``.
        linear_solver_maxiter: Maximum solver iterations per substep.
        min_stretch: Lower element principal-stretch clamp (dimensionless).
        max_stretch: Upper element principal-stretch clamp (dimensionless).
        ground_plane: ``(nx, ny, nz, d)`` analytic ground plane or ``None``.
        ground_ke: Ground normal stiffness [N/m].
        ground_kd: Ground normal damping [N·s/m].
        ground_kf: Ground tangential friction stiffness [N/m].
        ground_mu: Ground Coulomb friction coefficient.
        torque_stiffness: Spring bend-torque stiffness [N·m/rad].
        torque_damping: Spring bend-torque damping [N·m·s/rad].
        spring_rest_direction: Per-spring rest-frame direction vectors,
            shape ``[spring_count, 3]``.
        self_contact_ke: Particle-particle self-contact penalty stiffness [N/m].
        self_contact_kd: Particle-particle self-contact damping [N·s/m].
        linear_damping: Mass-proportional Rayleigh damping coefficient α [1/s].
    """

    def __init__(
        self,
        model: Model,
        fem: str = "auto",
        max_volume_ratio: float = 3.0,
        **kwargs,
    ):
        # Accept fem param for backward compat but ignore it — model decides.
        kwargs.pop("fem", None)
        super().__init__(model=model, **kwargs)
        self._init_inflation(model, max_volume_ratio)

    def _init_inflation(self, model: Model, max_volume_ratio: float) -> None:
        """Initialise all inflation state after base-class construction."""
        self.max_volume_ratio = float(max_volume_ratio)
        self.current_pressure = 1.0
        self.target_pressure = 1.0
        self.num_chambers = 0
        self._chamber_pressures_array = None
        self.element_chamber_mask = None
        self._tet_chamber_mask = None
        self._hex_chamber_mask = None
        self.spring_chamber_mask = None

        # Tet snapshot
        if model.tet_count > 0:
            self.original_tet_poses = wp.clone(model.tet_poses)
            self.tet_volumes = wp.zeros(model.tet_count, dtype=wp.float32, device=model.device)
        else:
            self.original_tet_poses = None
            self.tet_volumes = None

        # Hex snapshot — replace base views with writable clones so pressure
        # scaling never mutates the original model data.
        if self.hex_count > 0:
            self.original_hex_inv_J0 = self.hex_inv_J0
            self.hex_inv_J0 = wp.clone(self.original_hex_inv_J0)
            self.original_hex_det_J0_w = self.hex_det_J0_w
            self.hex_det_J0_w = wp.clone(self.original_hex_det_J0_w)
        else:
            self.original_hex_inv_J0 = None
            self.original_hex_det_J0_w = None

        # Spring snapshot
        if model.spring_count > 0:
            self.original_spring_rest_length = wp.clone(model.spring_rest_length)
        else:
            self.original_spring_rest_length = None

    # ------------------------------------------------------------------
    # Element pressure scaling
    # ------------------------------------------------------------------

    def set_pressure(self, pressure: float) -> None:
        """Set inflation volume ratio for the whole body.

        Scales tet poses (if tet elements are present), hex Gauss data (if hex
        elements are present), and spring rest lengths uniformly.

        Args:
            pressure: Target volume ratio (dimensionless, ≥ 1.0).
        """
        if self.num_chambers > 0 and (self._tet_chamber_mask is not None or self._hex_chamber_mask is not None):
            self.set_chamber_pressures([pressure] * self.num_chambers)
            self.current_pressure = float(pressure)
            return
        model = self.model
        pressure = max(1.0, min(float(pressure), self.max_volume_ratio))
        if abs(pressure - self.current_pressure) < 1e-9:
            return
        self.current_pressure = pressure
        linear_scale = pressure ** (1.0 / 3.0)

        if model.tet_count > 0 and self.original_tet_poses is not None:
            wp.launch(
                kernel=scale_tet_poses_kernel,
                dim=model.tet_count,
                inputs=[self.original_tet_poses, wp.float32(linear_scale)],
                outputs=[model.tet_poses],
                device=model.device,
            )
        if self.hex_count > 0:
            wp.launch(
                kernel=scale_hex_gauss_kernel,
                dim=self.hex_count * 8,
                inputs=[
                    self.original_hex_inv_J0,
                    self.original_hex_det_J0_w,
                    wp.float32(linear_scale),
                ],
                outputs=[self.hex_inv_J0, self.hex_det_J0_w],
                device=model.device,
            )
        self._scale_springs(linear_scale)

    def set_chamber_mask(
        self,
        element_chamber_mask: wp.array | None = None,
        spring_chamber_mask: wp.array | None = None,
        num_chambers: int = 0,
        *,
        tet_chamber_mask: wp.array | None = None,
        hex_chamber_mask: wp.array | None = None,
    ) -> None:
        """Mark spatially separate inflation regions for per-chamber control.

        For pure-tet or pure-hex models pass ``element_chamber_mask``.
        For mixed tet+hex models, pass ``tet_chamber_mask`` (shape ``[tet_count]``)
        and ``hex_chamber_mask`` (shape ``[hex_count]``) separately so each can
        map into independent (or shared) chamber-id ranges.

        Args:
            element_chamber_mask: Per-element chamber id, int32, for single-type
                models.  Pass ``None`` when using the explicit ``tet_chamber_mask``
                / ``hex_chamber_mask`` kwargs.
            spring_chamber_mask: Per-spring chamber id or ``None``.
            num_chambers: Total number of distinct chamber ids across all masks.
            tet_chamber_mask: Per-tet chamber id ``[tet_count]`` (keyword-only).
                Overrides the tet portion of ``element_chamber_mask``.
            hex_chamber_mask: Per-hex chamber id ``[hex_count]`` (keyword-only).
                Overrides the hex portion of ``element_chamber_mask``.
        """
        # Resolve masks: explicit kwargs take priority over the unified arg.
        self._tet_chamber_mask = tet_chamber_mask if tet_chamber_mask is not None else element_chamber_mask
        self._hex_chamber_mask = hex_chamber_mask if hex_chamber_mask is not None else element_chamber_mask
        # Keep element_chamber_mask as a convenience alias for the common single-type case.
        self.element_chamber_mask = element_chamber_mask
        self.spring_chamber_mask = spring_chamber_mask
        self.num_chambers = int(num_chambers)
        if self.num_chambers > 0:
            self._chamber_pressures_array = wp.array(
                [1.0] * self.num_chambers, dtype=wp.float32, device=self.model.device
            )

    def set_chamber_pressures(self, pressure_list: list[float]) -> None:
        """Set per-chamber pressure, clamped to ``[1.0, max_volume_ratio]``.

        Args:
            pressure_list: Per-chamber pressures, length ``≥ num_chambers``.
        """
        if self.num_chambers <= 0:
            return
        model = self.model
        pressures = [max(1.0, min(float(p), self.max_volume_ratio)) for p in pressure_list]
        while len(pressures) < self.num_chambers:
            pressures.append(1.0)
        pressures = pressures[: self.num_chambers]
        self._chamber_pressures_array.assign(pressures)

        if model.tet_count > 0 and self.original_tet_poses is not None and self._tet_chamber_mask is not None:
            wp.launch(
                kernel=scale_tet_poses_per_chamber_kernel,
                dim=model.tet_count,
                inputs=[
                    self.original_tet_poses,
                    self._tet_chamber_mask,
                    self._chamber_pressures_array,
                    self.num_chambers,
                ],
                outputs=[model.tet_poses],
                device=model.device,
            )
        if self.hex_count > 0 and self._hex_chamber_mask is not None:
            wp.launch(
                kernel=scale_hex_gauss_per_chamber_kernel,
                dim=self.hex_count * 8,
                inputs=[
                    self.original_hex_inv_J0,
                    self.original_hex_det_J0_w,
                    self._hex_chamber_mask,
                    self._chamber_pressures_array,
                    self.num_chambers,
                ],
                outputs=[self.hex_inv_J0, self.hex_det_J0_w],
                device=model.device,
            )
        self._scale_springs_per_chamber(pressures)

    # ------------------------------------------------------------------
    # Shared spring-scaling helpers
    # ------------------------------------------------------------------

    def _scale_springs(self, linear_scale: float) -> None:
        model = self.model
        if model.spring_count > 0 and self.original_spring_rest_length is not None:
            wp.launch(
                kernel=scale_spring_rest_lengths_kernel,
                dim=model.spring_count,
                inputs=[self.original_spring_rest_length, wp.float32(linear_scale)],
                outputs=[model.spring_rest_length],
                device=model.device,
            )

    def _scale_springs_per_chamber(self, pressures: list[float]) -> None:
        model = self.model
        if model.spring_count == 0 or self.original_spring_rest_length is None:
            return
        if self.spring_chamber_mask is not None:
            wp.launch(
                kernel=scale_spring_rest_lengths_per_chamber_kernel,
                dim=model.spring_count,
                inputs=[
                    self.original_spring_rest_length,
                    self.spring_chamber_mask,
                    self._chamber_pressures_array,
                    self.num_chambers,
                ],
                outputs=[model.spring_rest_length],
                device=model.device,
            )
        else:
            avg_s = (sum(pressures) / len(pressures)) ** (1.0 / 3.0)
            wp.launch(
                kernel=scale_spring_rest_lengths_kernel,
                dim=model.spring_count,
                inputs=[self.original_spring_rest_length, wp.float32(avg_s)],
                outputs=[model.spring_rest_length],
                device=model.device,
            )

    # ------------------------------------------------------------------
    # Pressure ramp
    # ------------------------------------------------------------------

    def set_target_pressure(self, target: float) -> None:
        """Store a target pressure; call :meth:`apply_pressure_control` to chase it.

        Args:
            target: Target volume ratio, clamped to ``[1.0, max_volume_ratio]``.
        """
        self.target_pressure = max(1.0, min(float(target), self.max_volume_ratio))

    def apply_pressure_control(self, rate: float = 0.1) -> None:
        """Step current pressure toward :attr:`target_pressure` at ``rate`` per call.

        Args:
            rate: Blending fraction in ``[0, 1]``.
        """
        new_p = self.current_pressure + rate * (self.target_pressure - self.current_pressure)
        self.set_pressure(new_p)

    # ------------------------------------------------------------------
    # Volume telemetry
    # ------------------------------------------------------------------

    def compute_volume(self, state: State | None = None) -> float:
        """Current volume [m³] from tet particle positions and/or hex Gauss data.

        For tet elements, ``state`` is required (volumes are computed from
        current particle positions).  For hex elements, ``state`` is not used
        (volume is computed analytically from the current Gauss data which
        reflects the inflation scale).

        Args:
            state: Current simulation state. Required when tet elements are
                present; ignored (but accepted for API parity) for hex-only.
        """
        model = self.model
        total = 0.0

        if model.tet_count > 0 and self.tet_volumes is not None and state is not None:
            wp.launch(
                kernel=compute_volume_kernel,
                dim=model.tet_count,
                inputs=[state.particle_q, model.tet_indices],
                outputs=[self.tet_volumes],
                device=model.device,
            )
            total += float(self.tet_volumes.numpy().sum())

        if self.hex_count > 0:
            vols = wp.zeros(self.hex_count, dtype=wp.float32, device=model.device)
            wp.launch(
                kernel=compute_hex_volume_kernel,
                dim=self.hex_count,
                inputs=[self.hex_det_J0_w],
                outputs=[vols],
                device=model.device,
            )
            total += float(vols.numpy().sum())

        return total

    def get_initial_volume(self, state: State | None = None) -> float:
        """Rest-config volume at ``p = 1`` [m³].

        Always available after construction (set by the base-class
        ``__init__``).  The ``state`` argument is accepted for API parity but
        is not used.

        Args:
            state: Ignored.
        """
        return self._initial_volume if self._initial_volume > 1e-12 else 1.0

    def get_volume_ratio(self, state: State | None = None) -> float:
        """current_volume / initial_volume, clamped to a sane range.

        Args:
            state: Current simulation state (required for tet; ignored for hex).
        """
        initial = self.get_initial_volume(state)
        if initial <= 1.0e-12:
            return 1.0
        current = self.compute_volume(state)
        if not math.isfinite(current) or current <= 0.0:
            return 1.0
        ratio = current / initial
        if not math.isfinite(ratio):
            return 1.0
        return max(0.0, min(ratio, max(10.0, self.max_volume_ratio * 2.0)))

    def get_inflation_info(self, state: State | None = None) -> dict:
        """Dictionary of inflation state for telemetry / UI overlays.

        Keys: ``initial_volume``, ``current_volume``, ``max_volume`` [m³];
        ``current_ratio``, ``target_ratio``, ``max_ratio`` (dimensionless);
        ``pressure``.

        Args:
            state: Current simulation state (required for tet; ignored for hex).
        """
        initial = self.get_initial_volume(state)
        current = self.compute_volume(state)
        if not math.isfinite(current):
            current_ratio = float("nan")
        elif initial > 1.0e-12 and math.isfinite(initial):
            current_ratio = float(current / initial)
            if not math.isfinite(current_ratio):
                current_ratio = float("nan")
        else:
            current_ratio = 1.0
        return {
            "initial_volume": initial,
            "current_volume": current,
            "max_volume": initial * self.max_volume_ratio,
            "current_ratio": current_ratio,
            "target_ratio": self.target_pressure,
            "max_ratio": self.max_volume_ratio,
            "pressure": self.current_pressure,
        }
