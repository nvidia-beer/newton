# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Driver (policy) interface for the track examples.

The example produces a :class:`TrackObservation` once per frame and asks a
:class:`TrackDriver` for a :class:`DriveCommand`; nothing else crosses the boundary.
Commands are normalized in the ``newton.vehicles`` convention (drive and steer in
[-1, 1], ``+steer`` = left turn, ``+drive`` = forward); the example maps them to its
actuators and joint signs.

Implementations:

* :class:`ManualDriver` - returns whatever the GUI sliders last set.
* :class:`PurePursuitDriver` - lookahead steering on the centerline with a
  curvature-limited speed (a ChPathFollowerDriver-style baseline, not a planner).
* a learned policy: subclass :class:`TrackDriver`, read the :class:`TrackObservation` fields
  and return a :class:`DriveCommand`. Register it in :data:`DRIVERS` to select it with
  ``--driver``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriveCommand:
    """Normalized command: ``drive`` in [-1, 1] (+ = forward), ``steer`` in [-1, 1] (+ = left)."""

    drive: float = 0.0
    steer: float = 0.0

    def clipped(self) -> DriveCommand:
        return DriveCommand(max(-1.0, min(1.0, float(self.drive))), max(-1.0, min(1.0, float(self.steer))))


@dataclass
class TrackObservation:
    """What a driver may look at, in world units.

    Attributes:
        x, y: Chassis position [m].
        yaw: Chassis heading [rad].
        speed: Planar chassis speed [m/s].
        yaw_rate: [rad/s].
        roll, pitch: [rad].
        s: Arc length of the nearest centerline point [m].
        lateral: Signed lateral offset from the centerline [m], + = left of the route.
        heading_error: Route tangent heading minus chassis yaw, wrapped [rad], + = route bends left.
        border_distance: Signed distance to the corridor border [m], negative inside.
        half_width: Corridor half width [m].
        curvature_ahead: Centerline curvature [1/m] sampled at :attr:`CURVATURE_LOOKAHEADS` metres ahead.
        route: The route object (``point_at(s)``, ``curvature_at(s)``, ``length``) for drivers
            that want geometry beyond the scalars; ``None`` for a policy that only reads the scalars.
    """

    CURVATURE_LOOKAHEADS = (5.0, 10.0, 20.0, 40.0)  # [m]

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    s: float = 0.0
    lateral: float = 0.0
    heading_error: float = 0.0
    border_distance: float = 0.0
    half_width: float = 1.0
    curvature_ahead: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 4)
    route: object | None = None


class TrackDriver:
    """Policy interface: one :class:`DriveCommand` per frame from a :class:`TrackObservation`.

    Subclasses must implement :meth:`act`; the other hooks are optional.
    """

    name: str = "driver"

    def reset(self) -> None:
        """Called at spawn / episode start."""

    def act(self, obs: TrackObservation) -> DriveCommand:
        """Return the command for this frame."""
        raise NotImplementedError

    def gui(self, ui) -> None:
        """Optional parameter panel (ImGui); default none."""

    def trajectory(self, obs: TrackObservation) -> np.ndarray | None:
        """Optional intended path for display: ``(K, 2)`` plan-view points from the car onward
        (the example drapes them on the terrain and draws a curved arrow). Default none."""
        return None


TRAJECTORY_SAMPLES = 24
"""Points a :meth:`TrackDriver.trajectory` should return (fixed so render buffers are static)."""


def pursuit_arc(x: float, y: float, yaw: float, target_xy, n: int = TRAJECTORY_SAMPLES) -> np.ndarray:
    """Circular arc tangent to the heading at (x, y) through ``target_xy``: the pure-pursuit path.

    Chord ``L_d`` at angle ``alpha`` to the heading gives radius ``R = L_d / (2 sin alpha)``; a
    straight segment when ``alpha`` is ~0.
    """
    tx, ty = float(target_xy[0]) - x, float(target_xy[1]) - y
    ld = math.hypot(tx, ty)
    alpha = _wrap_pi(math.atan2(ty, tx) - yaw)
    t = np.linspace(0.0, 1.0, n)
    if ld < 1e-6 or abs(math.sin(alpha)) < 1e-4:
        return np.stack([x + t * tx, y + t * ty], axis=1)
    R = ld / (2.0 * math.sin(alpha))  # signed: + = left turn
    theta = 2.0 * alpha * t  # heading change along the arc
    # arc in the heading frame: forward s = R sin(theta), left l = R (1 - cos(theta))
    fwd = R * np.sin(theta)
    left = R * (1.0 - np.cos(theta))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * fwd - s * left, y + s * fwd + c * left], axis=1)


class ManualDriver(TrackDriver):
    """Passes through the values the GUI sliders wrote into :attr:`command`."""

    name = "manual"

    def __init__(self) -> None:
        self.command = DriveCommand()

    def act(self, obs: TrackObservation) -> DriveCommand:
        return self.command.clipped()


class PurePursuitDriver(TrackDriver):
    """Lookahead steering on the centerline + curvature-limited speed.

    Steering: the bicycle-model pure-pursuit law ``delta = atan(2 L sin(alpha) / L_d)`` toward
    the centerline point ``L_d`` metres ahead, with ``L_d = max(lookahead_min, lookahead_time * v)``.
    Speed: ``min(target_speed, sqrt(a_lat_max / kappa_ahead))`` from the tightest bend within
    ``curvature_horizon`` metres, plus a proportional term on the speed error; the drive command
    is that wheel speed as a fraction of ``max_wheel_speed``.

    Args:
        wheelbase: [m].
        max_steer: Steering angle at ``steer = 1`` [rad].
        tire_radius: [m].
        max_wheel_speed: Wheel speed at ``drive = 1`` [rad/s].
        target_speed: Cruise speed [m/s].
        lookahead_min: [m].
        lookahead_time: [s].
        a_lat_max: Lateral-acceleration budget for the curvature cap [m/s²]; ``<= 0`` disables the cap.
        curvature_horizon: How far past the lookahead the cap looks [m].
        speed_kp: Drive gain on the speed error [1/s].
    """

    name = "pure_pursuit"

    def __init__(
        self,
        wheelbase: float,
        max_steer: float,
        tire_radius: float,
        max_wheel_speed: float,
        target_speed: float = 4.0,
        lookahead_min: float = 6.0,
        lookahead_time: float = 1.2,
        a_lat_max: float = 2.5,
        curvature_horizon: float = 12.0,
        speed_kp: float = 0.6,
        lookahead_max: float = 40.0,
    ) -> None:
        self.wheelbase = float(wheelbase)
        self.max_steer = float(max_steer)
        self.tire_radius = float(tire_radius)
        self.max_wheel_speed = float(max_wheel_speed)
        self.target_speed = float(target_speed)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_time = float(lookahead_time)
        # Cap on the speed-scaled lookahead: a vehicle that has blown up (non-finite or huge
        # speed after a solver failure) must not turn into a multi-GiB curvature scan below.
        self.lookahead_max = float(lookahead_max)
        # The cap is a toggle plus a value: a_lat_max <= 0 at construction starts with it off but
        # keeps a sensible value behind the checkbox so it can be switched on in the GUI.
        self.cap_enabled = float(a_lat_max) > 0.0
        self.a_lat_max = float(a_lat_max) if self.cap_enabled else 2.5
        self.curvature_horizon = float(curvature_horizon)
        self.speed_kp = float(speed_kp)
        self.lookahead = 0.0
        self.v_set = 0.0
        self._target_xy: np.ndarray | None = None

    def act(self, obs: TrackObservation) -> DriveCommand:
        route = obs.route
        if route is None:
            raise ValueError("PurePursuitDriver needs obs.route")
        speed = obs.speed if math.isfinite(obs.speed) else 0.0
        look = min(self.lookahead_max, max(self.lookahead_min, self.lookahead_time * speed))
        target = route.point_at(obs.s + look)
        self._target_xy = np.asarray(target[:2], dtype=np.float64)
        alpha = _wrap_pi(math.atan2(float(target[1]) - obs.y, float(target[0]) - obs.x) - obs.yaw)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), look)
        steer = delta / self.max_steer
        kappa = max(route.curvature_at(obs.s + d) for d in np.arange(0.0, look + self.curvature_horizon, 2.0))
        # a_lat_max <= 0 disables the curvature cap: the controller then runs at target_speed
        # into every bend, which is how it is made to fail on purpose.
        v_allowed = (
            math.sqrt(self.a_lat_max / kappa)
            if (self.cap_enabled and kappa > 1e-4 and self.a_lat_max > 0.0)
            else self.target_speed
        )
        v_set = min(self.target_speed, v_allowed)
        drive = (v_set + self.speed_kp * (v_set - obs.speed)) / (self.tire_radius * self.max_wheel_speed)
        self.lookahead, self.v_set = look, v_set
        return DriveCommand(max(0.0, drive), steer).clipped()

    def gui(self, ui) -> None:
        # upper bound = the no-slip speed at full throttle, so the whole throttle range is reachable
        v_max = self.tire_radius * self.max_wheel_speed
        _c, self.target_speed = ui.slider_float(f"Target speed [m/s] (max {v_max:.0f})", self.target_speed, 0.5, v_max)
        _c, self.lookahead_min = ui.slider_float("Lookahead min [m]", self.lookahead_min, 3.0, 15.0)
        _c, self.cap_enabled = ui.checkbox("Curvature speed cap", self.cap_enabled)
        if self.cap_enabled:
            _c, self.a_lat_max = ui.slider_float("Lateral accel cap [m/s^2]", self.a_lat_max, 0.5, 8.0)
        else:
            ui.text("  cap OFF: full target speed into every bend")
        ui.text(f"  lookahead {self.lookahead:.1f} m   v_set {self.v_set:.2f} m/s")

    def trajectory(self, obs: TrackObservation) -> np.ndarray | None:
        if self._target_xy is None:
            return None
        return pursuit_arc(obs.x, obs.y, obs.yaw, self._target_xy)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


DRIVERS: dict[str, type[TrackDriver]] = {
    ManualDriver.name: ManualDriver,
    PurePursuitDriver.name: PurePursuitDriver,
}
"""Selectable drivers (``--driver``); add a learned policy here."""
