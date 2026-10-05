# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiate a steady skid-steer response fit to observed vehicle velocities."""

import math

import numpy as np
import warp as wp


@wp.kernel
def _response_loss(
    gains: wp.array[float],
    inputs: wp.array[wp.vec2],
    measured: wp.array[wp.vec2],
    scale: wp.vec2,
    loss: wp.array[float],
):
    i = wp.tid()
    error_v = (gains[0] * inputs[i][0] - measured[i][0]) / scale[0]
    yaw = inputs[i][1]
    error_w = (gains[1] * yaw + gains[2] * yaw * yaw * yaw - measured[i][1]) / scale[1]
    wp.atomic_add(loss, 0, 0.5 * (error_v * error_v + error_w * error_w) / float(inputs.shape[0]))


class SkidResponse:
    """Fit speed gain and an odd cubic yaw response to fixed MuJoCo observations.

    The differentiable model predicts body speed [m/s] and yaw rate [rad/s]
    from left/right motor targets [rad/s]. Its gains include tire slip, loaded
    rolling radius and drive response. They are not identified material constants.
    For nominal rates v and w: speed = g0*v, yaw = g1*w + g2*w**3.
    g0 and g1 are dimensionless; g2 is in seconds squared. Nonnegative yaw
    coefficients keep the response monotone and preserve left/right symmetry.
    """

    def __init__(self, radius: float, track: float, device="cpu"):
        if not np.isfinite([radius, track]).all() or min(radius, track) <= 0:
            raise ValueError("Rolling radius and track must be finite and positive")
        self.radius = radius
        self.track = track
        self.gains = wp.array([1.0, 1.0, 0.0], dtype=float, requires_grad=True, device=device)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        self.inputs = self.measured = None
        self.history = []

    def features(self, wheel_targets: np.ndarray) -> np.ndarray:
        """Convert left/right targets [rad/s] into nominal speed [m/s] and yaw [rad/s]."""
        wheels = np.asarray(wheel_targets, dtype=float)
        if wheels.ndim != 2 or wheels.shape[1] != 2 or not np.isfinite(wheels).all():
            raise ValueError("Wheel targets must be a finite (N, 2) array")
        return np.column_stack(
            (self.radius * wheels.mean(axis=1), self.radius * (wheels[:, 1] - wheels[:, 0]) / self.track)
        )

    def set_samples(self, wheel_targets: np.ndarray, measured: np.ndarray) -> None:
        """Set measured body velocities [m/s, rad/s] for motor targets [rad/s]."""
        inputs = self.features(wheel_targets)
        measured = np.asarray(measured, dtype=float)
        if len(inputs) < 3 or measured.shape != inputs.shape or not np.isfinite(measured).all():
            raise ValueError("Need at least three finite, paired velocity samples")
        scale = np.sqrt(np.mean(inputs**2, axis=0))
        if np.any(scale < 1e-5) or not (np.any(inputs[:, 1] > 0) and np.any(inputs[:, 1] < 0)):
            raise ValueError("Calibration requires forward motion and both left and right turns")
        yaw_basis = np.column_stack((inputs[:, 1], inputs[:, 1] ** 3)) / scale[1]
        if np.linalg.matrix_rank(yaw_basis) < 2:
            raise ValueError("Calibration requires at least two different turning strengths")
        # Precondition the correlated linear/cubic terms without changing the loss.
        self._hessian = np.eye(3)
        self._hessian[1:, 1:] = yaw_basis.T @ yaw_basis / len(inputs)
        self.scale = wp.vec2(*scale)
        self.inputs = wp.array(inputs, dtype=wp.vec2, device=self.gains.device)
        self.measured = wp.array(measured, dtype=wp.vec2, device=self.gains.device)
        self.history = [self.loss_gradient()[0]]

    def loss_gradient(self) -> tuple[float, np.ndarray]:
        """Return normalized velocity loss and its Warp autodiff gradient."""
        if self.inputs is None:
            raise RuntimeError("Collect calibration maneuvers before learning")
        self.loss.zero_()
        with wp.Tape() as tape:
            wp.launch(
                _response_loss,
                dim=len(self.inputs),
                inputs=[self.gains, self.inputs, self.measured, self.scale],
                outputs=[self.loss],
                device=self.gains.device,
            )
        tape.backward(self.loss)
        value, gradient = float(self.loss.numpy()[0]), self.gains.grad.numpy().copy()
        tape.zero()
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError("Non-finite skid-response gradient")
        return value, gradient

    def step(self) -> float:
        """Take one bounded gradient step and return the normalized velocity loss."""
        _value, gradient = self.loss_gradient()
        direction = np.linalg.solve(self._hessian, gradient)
        self.gains.assign(np.maximum(self.gains.numpy() - 0.35 * direction, [0.001, 0.001, 0.0]))
        value = self.loss_gradient()[0]
        self.history.append(value)
        return value

    def wheel_commands(self, speed: float, yaw_rate: float, *, learned: bool = True) -> np.ndarray:
        """Invert the fitted response for speed [m/s] and yaw rate [rad/s].

        Return independent left/right motor targets [rad/s], without clipping.
        The caller must enforce the vehicle's drive limits.
        """
        if not np.isfinite([speed, yaw_rate]).all():
            raise ValueError("Speed and yaw rate must be finite")
        gv, gw, cubic = self.gains.numpy() if learned else (1.0, 1.0, 0.0)
        mean = speed / (self.radius * gv)
        lo, hi = 0.0, abs(yaw_rate) / gw
        for _ in range(40):
            mid = (lo + hi) / 2
            if gw * mid + cubic * mid**3 < abs(yaw_rate):
                lo = mid
            else:
                hi = mid
        nominal_yaw = math.copysign((lo + hi) / 2, yaw_rate)
        difference = nominal_yaw * self.track / self.radius
        return np.array([mean - 0.5 * difference, mean + 0.5 * difference])
