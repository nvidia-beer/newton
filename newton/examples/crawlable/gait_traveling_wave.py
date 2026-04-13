# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Traveling-wave chamber pressures for soft crawling (open-loop gait).
# Same mathematical structure as typical traveling-wave / phase-shifted inflation controllers:
# per-chamber p_k from a sinusoid along spatial chamber order, with symmetric clipping.
#
# This module is intentionally self-contained (NumPy only) so examples can drive
# SolverInflatable.set_chamber_pressures without the paper stick-slip contact model.

from __future__ import annotations

from typing import Any, Generic, Protocol, Sequence, TypeVar

import numpy as np

_CHAMBER_X_SORT_DECIMALS = 3


def smoothstep01(t: float) -> float:
    """Cubic Hermite s(t)=t²(3−2t) on [0,1] after clamping."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def symmetric_radius(baseline: float, min_p: float, max_p: float) -> float | None:
    lo = float(baseline - min_p)
    hi = float(max_p - baseline)
    if lo <= 0.0 or hi <= 0.0:
        return None
    return min(lo, hi)


def clip_pressure_symmetric(baseline: float, raw_p: float, min_p: float, max_p: float) -> float:
    r = symmetric_radius(baseline, min_p, max_p)
    if r is None:
        return float(np.clip(raw_p, min_p, max_p))
    return float(np.clip(raw_p, baseline - r, baseline + r))


def symmetric_gait_amplitude_cap(baseline: float, min_p: float, max_p: float) -> float:
    r = symmetric_radius(baseline, min_p, max_p)
    return 0.0 if r is None else r


def active_chambers_ordered_along_world_x(model: Any, pos_np: np.ndarray, active_ids: list) -> list:
    """
    Sort active chamber ids so wave index k=0 is at the "start" of the crawl direction.

    Uses triangle centroids per chamber when ``tri_chamber_mask`` exists. The primary
    sort axis follows the chamber grid: ``+X`` if ``num_chambers_x`` > 1, else ``+Y``
    if ``num_chambers_y`` > 1, else ``+Z`` if ``num_chambers_z`` > 1 (matches multi-row
    inchworm 1×N×2 and Soft worm N×M layouts). If centroid data are incomplete, falls
    back to lexicographic ``(ix, iy, iz)`` from the 3D index convention
    ``ch = ix * (ny * nz) + iy * nz + iz``.
    """
    if not active_ids:
        return []
    nx_m = int(getattr(model, "num_chambers_x", 1) or 1)
    ny_m = int(getattr(model, "num_chambers_y", 1) or 1)
    nz_m = int(getattr(model, "num_chambers_z", 1) or 1)

    def _centroid_axis_key(mean_xyz: np.ndarray) -> float:
        if nx_m > 1:
            return float(mean_xyz[0])
        if ny_m > 1:
            return float(mean_xyz[1])
        if nz_m > 1:
            return float(mean_xyz[2])
        return float(mean_xyz[0])

    n_tri_m = int(getattr(model, "tri_count", 0) or 0)
    if n_tri_m > 0 and hasattr(model, "tri_chamber_mask") and hasattr(model, "tri_indices"):
        tri = np.asarray(model.tri_indices.numpy()).reshape(-1, 3)
        tcm = np.asarray(model.tri_chamber_mask.numpy())
        n_tri = tri.shape[0]
        if tcm.size >= n_tri:
            means: dict[int, np.ndarray] = {}
            for c in active_ids:
                mask = tcm[:n_tri] == c
                if not np.any(mask):
                    continue
                verts = tri[mask].astype(np.int64).ravel()
                means[int(c)] = np.mean(pos_np[verts], axis=0)
            if len(means) == len(active_ids):
                rd = _CHAMBER_X_SORT_DECIMALS
                return sorted(
                    active_ids,
                    key=lambda cid: (round(_centroid_axis_key(means[int(cid)]), rd), int(cid)),
                )
    nx = getattr(model, "num_chambers_x", None)
    ny = getattr(model, "num_chambers_y", None)
    nz = getattr(model, "num_chambers_z", None)
    nch = int(getattr(model, "num_chambers", 0) or 0)
    if nx is not None and ny is not None and nz is not None and nch == int(nx) * int(ny) * int(nz):

        def _ix_iy_iz(c: int) -> tuple[int, int, int]:
            c = int(c)
            nnz = int(nz)
            nny = int(ny)
            iz = c % nnz
            t = c // nnz
            iy = t % nny
            ix = t // nny
            return (ix, iy, iz)

        return sorted(active_ids, key=_ix_iy_iz)
    if nx is not None and ny is not None and nch == int(nx) * int(ny):
        return sorted(active_ids, key=lambda c: (int(c) % int(nx), int(c) // int(nx)))
    return list(active_ids)


def wave_phase_chamber_order(active_order: list[int]) -> list[int]:
    return list(active_order)


def traveling_wave_active_layout(
    num_chambers: int,
    active_inflation: Sequence[int],
    active_chamber_order: list[int] | None,
    phase_span_rad: float,
) -> tuple[list[int], list[int], list[int], int, float]:
    active_raw = [
        i
        for i in range(min(len(active_inflation), num_chambers))
        if int(active_inflation[i]) == 1
    ]
    active_order = list(active_chamber_order) if active_chamber_order else active_raw
    if set(active_order) != set(active_raw) or len(active_order) != len(active_raw):
        active_order = active_raw
    wave_order = wave_phase_chamber_order(active_order)
    na = len(wave_order)
    dphi = float(phase_span_rad) / (na - 1) if na > 1 else 0.0
    return active_raw, active_order, wave_order, na, dphi


def traveling_wave_fill_pressures(
    wave_order: Sequence[int],
    *,
    wave_dir: float,
    dphi: float,
    baseline: float,
    amplitude: float,
    amp_scale: float,
    phase_at_k0: float,
    num_chambers: int,
    min_pressure: float,
    max_pressure: float,
) -> tuple[list[float], list[float], list[float]]:
    pressures = [float(baseline)] * int(num_chambers)
    sin_phases: list[float] = []
    raw_active: list[float] = []
    for k, idx in enumerate(wave_order):
        ph = float(phase_at_k0 - float(wave_dir) * float(k) * dphi)
        sin_phases.append(ph)
        raw = float(baseline + float(amplitude) * float(amp_scale) * float(np.sin(ph)))
        raw_active.append(raw)
        pressures[int(idx)] = clip_pressure_symmetric(
            float(baseline), raw, float(min_pressure), float(max_pressure)
        )
    return pressures, sin_phases, raw_active


class TravelingWaveGait:
    """
    Open-loop traveling inflation wave: per-chamber pressure from a phase-shifted sinusoid
    along spatial order (min world +x → max +x).
    """

    def __init__(
        self,
        freq_hz: float,
        baseline: float,
        amplitude: float,
        phase_rad: float,
        num_chambers: int,
        min_pressure: float,
        max_pressure: float,
        settle_time: float = 1.0,
        active_inflation: list[int] | None = None,
        amplitude_rise_s: float = 0.45,
    ):
        self.omega = 2.0 * np.pi * float(freq_hz)
        self.baseline = float(baseline)
        self.amplitude = float(amplitude)
        self.phase = float(phase_rad)
        self.num_chambers = int(num_chambers)
        self.min_pressure = float(min_pressure)
        self.max_pressure = float(max_pressure)
        self.settle_time = float(settle_time)
        self.active_inflation = active_inflation if active_inflation is not None else [1] * self.num_chambers
        self.amplitude_rise_s = max(0.0, float(amplitude_rise_s))
        self.active_chamber_order: list[int] | None = None
        self._freq_hz = float(freq_hz)

    @property
    def freq_hz(self) -> float:
        return float(self._freq_hz)

    def set_freq_hz(self, f: float) -> None:
        self._freq_hz = max(1e-3, float(f))
        self.omega = 2.0 * np.pi * self._freq_hz

    def set_amplitude(self, a: float) -> None:
        self.amplitude = float(a)

    def compute_pressures(self, t: float, dt: float | None = None) -> list[float]:
        d = 1.0
        _active_raw, active_order, wave_order, na, dphi = traveling_wave_active_layout(
            self.num_chambers,
            self.active_inflation,
            self.active_chamber_order,
            float(self.phase),
        )

        if t < self.settle_time:
            return [float(self.baseline)] * self.num_chambers

        tau = float(t - self.settle_time)
        omega_tau = float(self.omega * tau)
        rise_s = max(float(self.amplitude_rise_s), 1e-3)
        amp_scale = float(smoothstep01(tau / rise_s))

        pressures, _sin_phases, _raw_active = traveling_wave_fill_pressures(
            wave_order,
            wave_dir=float(d),
            dphi=float(dphi),
            baseline=float(self.baseline),
            amplitude=float(self.amplitude),
            amp_scale=float(amp_scale),
            phase_at_k0=float(omega_tau),
            num_chambers=int(self.num_chambers),
            min_pressure=float(self.min_pressure),
            max_pressure=float(self.max_pressure),
        )
        return pressures


class TravelingWavePressureController(Protocol):
    """Contract for gait objects driven by ``CrawlPressureOrchestratorBase``."""

    active_inflation: list[int]
    active_chamber_order: list[int] | None

    def compute_pressures(self, t: float, dt: float | None = None) -> list[float]: ...


TWaveCtl = TypeVar("TWaveCtl", bound=TravelingWavePressureController)


class CrawlPressureOrchestratorBase(Generic[TWaveCtl]):
    """
    Uniform 1→baseline ramp, then controller ``compute_pressures`` — same structure as Soft worm
    ``crawlable_control.controllers.gait.orchestrator_base``.
    """

    def __init__(
        self,
        *,
        controller: TWaveCtl,
        num_chambers: int,
        gait_baseline: float,
        startup_ramp_s: float,
        dt: float,
    ):
        self._controller = controller
        self._num_chambers = int(num_chambers)
        self._gait_baseline = float(gait_baseline)
        self._startup_ramp_s = float(startup_ramp_s)
        self._dt = float(dt)
        self._ramp_steps = max(1, int(round(self._startup_ramp_s / max(self._dt, 1e-9))))
        self._sigma = 1.0
        self._last_spatial_order: list[int] | None = None

    @property
    def gait_controller(self) -> TWaveCtl:
        return self._controller

    @property
    def ramp_step_count(self) -> int:
        return self._ramp_steps

    @property
    def effective_sigma(self) -> float:
        """Traveling-wave direction along +x: +1 (diagnostics; CSV ``sigma``)."""
        return float(self._sigma)

    def sync_chamber_order(self, model: Any, pos_np: np.ndarray) -> None:
        _ai = self._controller.active_inflation
        _raw = [i for i in range(min(len(_ai), self._num_chambers)) if _ai[i] == 1]
        spatial = active_chambers_ordered_along_world_x(model, pos_np, _raw)
        prev = self._controller.active_chamber_order
        if prev is not None and list(prev) == spatial:
            return
        self._controller.active_chamber_order = spatial
        if spatial != _raw and tuple(spatial) != tuple(self._last_spatial_order or ()):
            self._last_spatial_order = list(spatial)
            print(
                f"  Gait wave: active chambers in spatial order (index order {_raw} → spatial {spatial})",
                flush=True,
            )

    def step_pressures(self, step: int, sim_time: float, dt: float) -> list[float]:
        """Ramp then gait; advances gait state every step for telemetry."""
        c = self._controller
        n = self._ramp_steps
        if step < n:
            u = smoothstep01((step + 1) / n)
            p = 1.0 + (self._gait_baseline - 1.0) * u
            pressures = [float(p)] * self._num_chambers
            c.compute_pressures(sim_time, dt=dt)
        else:
            pressures = list(c.compute_pressures(sim_time, dt=dt))
        self._sigma = 1.0
        return pressures


__all__ = [
    "CrawlPressureOrchestratorBase",
    "TravelingWaveGait",
    "TravelingWavePressureController",
    "TWaveCtl",
    "active_chambers_ordered_along_world_x",
    "clip_pressure_symmetric",
    "smoothstep01",
    "symmetric_gait_amplitude_cap",
    "traveling_wave_active_layout",
    "traveling_wave_fill_pressures",
]
