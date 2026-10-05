# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit dynamics and trajectory sensitivities for a freely translating ANCF wheel.

This opt-in, dense float64 experiment reuses the ANCF3423 elastic/EAS and cavity
equations of ANCFTireEquilibrium. The rim has vertical and longitudinal inertia;
its spin is prescribed by an ideal velocity motor. Forward travel is an unknown.
Backward Euler integrates the shell and rim together. Contact uses the production
plane's one-sided normal damping and tanh Coulomb force, with its exact derivative
(not the production solver's positive iteration majorizer). Structural Rayleigh
damping is omitted; this is not the MuJoCo/HHT production stepping path.

For one unknown mu, a forward sensitivity is cheaper than a reverse tape. Each
converged step solves J dz_next/dmu = -dR/dmu including the previous position and
velocity sensitivities. Thus gradients include the entire dynamic history.
Finite differences are used only by tests.
"""

from dataclasses import dataclass

import numpy as np

from .differentiation import ANCFTireEquilibrium


class ANCFTireTraction:
    """Single-wheel dynamics with an analytic friction trajectory derivative.

    Args:
        equilibrium: Source single-tire elastic, pressure and contact equations.
        pressure: Fixed nominal cavity pressure [Pa].
        rim_mass: Translating rim and carried load mass [kg]. Its weight must
            equal ``equilibrium.rim_load``; shell mass is counted separately.
        dt: Integration time step [s].
        contact_damping: Closing-only normal contact damping [N s/m] per node.
        friction_velocity: Tanh friction regularization speed [m/s].
    """

    @dataclass
    class State:
        """Dynamic state and derivative with respect to dimensionless friction.

        ``q`` stores nodal positions [m] and directors; ``velocity`` their rates.
        ``z`` stores free shell coordinates followed by rim height and travel [m].
        ``zd`` stores their rates. ``sensitivity`` and ``velocity_sensitivity``
        have the units of z and zd, respectively. ``angle`` is rim spin [rad].
        """

        q: np.ndarray
        velocity: np.ndarray
        z: np.ndarray
        zd: np.ndarray
        sensitivity: np.ndarray
        velocity_sensitivity: np.ndarray
        angle: float = 0.0
        residual: float = 0.0
        iterations: int = 0

    def __init__(
        self,
        equilibrium: ANCFTireEquilibrium,
        pressure: float,
        rim_mass: float,
        dt: float = 0.01,
        contact_damping: float = 0.0,
        friction_velocity: float = 0.01,
    ):
        values = [pressure, rim_mass, dt, contact_damping, friction_velocity]
        if not np.isfinite(values).all() or min(pressure, rim_mass, dt, friction_velocity) <= 0 or contact_damping < 0:
            raise ValueError(
                "Finite positive pressure, mass, dt and friction velocity, and nonnegative damping required"
            )
        if not np.isclose(-rim_mass * equilibrium.gravity[1], equilibrium.rim_load, rtol=1e-6):
            raise ValueError("Translating mass and equilibrium rim load must describe the same load")
        self.eq = equilibrium
        self.pressure, self.rim_mass, self.dt = pressure, rim_mass, dt
        self.kd, self.v_reg = contact_damping, friction_velocity
        self.free = equilibrium.free
        self.rim_dofs = [6 * equilibrium.beads + 1, 6 * equilibrium.beads + 2]
        self.size = len(self.free) + 2
        self.mass = self._reduce(equilibrium.mass)
        self.mass[-2:] += rim_mass

    def _reduce(self, full):
        return np.r_[full[self.free], [full[dofs].sum() for dofs in self.rim_dofs]]

    def _reduce_matrix(self, full):
        n = len(self.free)
        result = np.empty((self.size, self.size))
        result[:n, :n] = full[np.ix_(self.free, self.free)]
        for i, dofs in enumerate(self.rim_dofs):
            result[:n, n + i] = full[np.ix_(self.free, dofs)].sum(axis=1)
            result[n + i, :n] = full[np.ix_(dofs, self.free)].sum(axis=0)
            for j, other in enumerate(self.rim_dofs):
                result[n + i, n + j] = full[np.ix_(dofs, other)].sum()
        return result

    def _full(self, z, angle):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        q = (self.eq.rest @ rotation.T).reshape(-1)
        q[self.free] = z[:-2]
        q[self.rim_dofs[0]] += z[-2]
        q[self.rim_dofs[1]] += z[-1]
        return q.reshape(self.eq.n, 2, 3)

    def initial_state(self, static: ANCFTireEquilibrium.Result) -> State:
        """Start at a static equilibrium with zero velocity and friction sensitivity."""
        if not np.isclose(static.pressure, self.pressure):
            raise ValueError("Initial equilibrium must use the fixed experiment pressure")
        z = np.r_[static.coordinates[:-1], static.height, 0.0]
        return self.State(
            self._full(z, 0),
            np.zeros_like(self.eq.rest),
            z,
            np.zeros(self.size),
            np.zeros(self.size),
            np.zeros(self.size),
        )

    def contact(self, q, velocity, mu, tangent=True):
        """Return resisting contact force and derivatives with respect to q, velocity and mu."""
        eq = self.eq
        n = eq.n * 6
        force, dmu = np.zeros(n), np.zeros(n)
        k, damping = (np.zeros((n, n)), np.zeros((n, n))) if tangent else (None, None)
        for i in np.flatnonzero(q[:, 0, 1] < eq.ground):
            v = velocity[i, 0]
            fn = eq.kn * (eq.ground - q[i, 0, 1]) + self.kd * max(-v[1], 0.0)
            vt = v[[0, 2]]
            speed = np.sqrt(vt @ vt + 1e-12)
            g = np.tanh(speed / self.v_reg)
            h = g / speed
            tangent_dofs = np.array([6 * i, 6 * i + 2])
            normal = 6 * i + 1
            unit = fn * h * vt
            force[tangent_dofs] = mu * unit
            force[normal] = -fn
            dmu[tangent_dofs] = unit
            if tangent:
                dh = ((1 - g * g) / self.v_reg - h) / speed**2
                damping[np.ix_(tangent_dofs, tangent_dofs)] = mu * fn * (h * np.eye(2) + dh * np.outer(vt, vt))
                k[tangent_dofs, normal] = -mu * eq.kn * h * vt
                k[normal, normal] = eq.kn
                if v[1] < 0:
                    damping[tangent_dofs, normal] = -mu * self.kd * h * vt
                    damping[normal, normal] = self.kd
        return force, k, damping, dmu

    def evaluate(self, z, previous: State, angle: float, mu: float, tangent=True):
        """Evaluate one backward-Euler residual, exact step Jacobian, and friction partial."""
        eq, dt = self.eq, self.dt
        q = self._full(z, angle)
        velocity = (q - previous.q) / dt
        elastic, ke = eq._elastic(q, tangent)
        volume, dv, pressure_force, kp = eq._pressure(q[:, 0], tangent)
        gauge = self.pressure * eq.vref / volume - eq.build_pressure
        force, k, damping, dmu = self.contact(q, velocity, mu, tangent)
        np.add.at(force, eq.element_dofs, elastic - gauge * pressure_force)
        force.reshape(eq.n, 6)[:, :3] -= eq.mass.reshape(eq.n, 6)[:, :3] * eq.gravity
        inertia = eq.mass * ((velocity - previous.velocity) / dt).ravel()
        residual = self._reduce(force + inertia)
        residual[-2:] += self.rim_mass * (z[-2:] - previous.z[-2:] - dt * previous.zd[-2:]) / dt**2
        residual[-2] += eq.rim_load
        if not tangent:
            return residual, None, None, self._reduce(dmu)
        np.add.at(k, (eq._rows, eq._cols), (ke - gauge * kp).ravel())
        unit_pressure = np.zeros(eq.n * 6)
        np.add.at(unit_pressure, eq.element_dofs, pressure_force)
        dv_full = np.zeros((eq.n, 6))
        dv_full[:, :3] = dv
        k += np.outer(unit_pressure, self.pressure * eq.vref / volume**2 * dv_full.ravel())
        damping = self._reduce_matrix(damping)
        jac = self._reduce_matrix(k) + damping / dt
        jac[np.diag_indices(self.size)] += self.mass / dt**2
        return residual, jac, damping, self._reduce(dmu)

    def step(self, previous: State, omega: float, mu: float, *, sensitivity=True, max_iterations=25) -> State:
        """Advance with prescribed rim speed [rad/s] and friction coefficient.

        Raises on a failed nonlinear or sensitivity solve. No partial trajectory
        is returned as a successful differentiable step.
        """
        if not np.isfinite([omega, mu]).all() or mu <= 0:
            raise ValueError("Wheel speed must be finite and friction positive")
        dt = self.dt
        angle = previous.angle + dt * omega
        z = previous.z + dt * previous.zd
        for _iteration in range(max_iterations):
            residual, jac, damping, dmu = self.evaluate(z, previous, angle, mu)
            norm = np.linalg.norm(residual, np.inf)
            if norm < 1e-6:
                break
            direction = self.eq._solve_scaled(jac, -residual)
            merit = np.linalg.norm(residual)
            step = 1.0
            for _ in range(16):
                try:
                    r = self.evaluate(z + step * direction, previous, angle, mu, False)[0]
                    if np.linalg.norm(r) < (1 - 1e-4 * step) * merit:
                        z += step * direction
                        break
                except ValueError:
                    pass
                step *= 0.5
            else:
                raise RuntimeError(f"Traction Newton line search stalled at residual {norm:.3g}")
        else:
            raise RuntimeError(f"Traction Newton solve failed: residual {norm:.3g}")
        ds = np.zeros(self.size)
        if sensitivity:
            rhs = self.mass * (previous.sensitivity / dt**2 + previous.velocity_sensitivity / dt)
            rhs += damping @ previous.sensitivity / dt - dmu
            ds = self.eq._solve_scaled(jac, rhs)
            if not np.isfinite(ds).all() or np.linalg.norm(jac @ ds - rhs) / max(np.linalg.norm(rhs), 1e-20) > 1e-7:
                raise RuntimeError("Traction sensitivity solve failed")
        q = self._full(z, angle)
        if not np.isfinite(q).all() or q[:, 0, 1].min() < self.eq.ground - 0.05:
            raise RuntimeError("Invalid traction state or more than 50 mm ground penetration")
        return self.State(
            q,
            (q - previous.q) / dt,
            z,
            (z - previous.z) / dt,
            ds,
            (ds - previous.sensitivity) / dt,
            angle,
            float(norm),
            _iteration,
        )

    @staticmethod
    def loss_gradient(states, reference, position_scale=0.25, speed_scale=1.0):
        """Return normalized trajectory loss, d(loss)/dmu and scalar Gauss-Newton curvature.

        Position and speed scales are in [m] and [m/s], respectively. Reference
        states are fixed observations and never supply parameter derivatives.
        """
        if len(states) != len(reference) or not states or min(position_scale, speed_scale) <= 0:
            raise ValueError("Paired nonempty trajectories and positive measurement scales required")
        errors = np.array(
            [
                [(s.z[-1] - r.z[-1]) / position_scale, (s.zd[-1] - r.zd[-1]) / speed_scale]
                for s, r in zip(states, reference, strict=True)
            ]
        )
        jac = np.array([[s.sensitivity[-1] / position_scale, s.velocity_sensitivity[-1] / speed_scale] for s in states])
        return (
            float(0.5 * np.mean(np.sum(errors**2, axis=1))),
            float(np.mean(np.sum(errors * jac, axis=1))),
            float(np.mean(np.sum(jac**2, axis=1))),
        )
