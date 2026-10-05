# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact Jacobians, full-trajectory sensitivities, and friction recovery."""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from newton.examples.ancf.diffsim.example_diffsim_ancf_tire_traction import Example, make_experiment, motor_commands
from newton.solvers import ANCFTireTraction
from newton.tests.ancf_diffsim_checks import check_tire_traction_result


class TestTractionContact(unittest.TestCase):
    def setUp(self):
        self.solver = ANCFTireTraction.__new__(ANCFTireTraction)
        self.solver.eq = SimpleNamespace(n=2, ground=0.0, kn=2000.0)
        self.solver.kd, self.solver.v_reg = 3.0, 0.05
        self.q = np.zeros((2, 2, 3))
        self.q[:, 0, 1] = [-0.01, 0.02]
        self.v = np.zeros_like(self.q)
        self.v[0, 0] = [0.2, -0.04, -0.4]

    def test_contact_derivatives_match_central_differences(self):
        _f, k, damping, dmu = self.solver.contact(self.q, self.v, 0.6)
        eps = 1e-6
        for j in range(12):
            delta = np.eye(12)[j].reshape(2, 2, 3) * eps
            numeric_q = (
                self.solver.contact(self.q + delta, self.v, 0.6)[0]
                - self.solver.contact(self.q - delta, self.v, 0.6)[0]
            ) / (2 * eps)
            numeric_v = (
                self.solver.contact(self.q, self.v + delta, 0.6)[0]
                - self.solver.contact(self.q, self.v - delta, 0.6)[0]
            ) / (2 * eps)
            np.testing.assert_allclose(k[:, j], numeric_q, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(damping[:, j], numeric_v, rtol=1e-5, atol=1e-7)
        numeric_mu = (
            self.solver.contact(self.q, self.v, 0.6 + eps)[0] - self.solver.contact(self.q, self.v, 0.6 - eps)[0]
        ) / (2 * eps)
        np.testing.assert_allclose(dmu, numeric_mu, rtol=1e-8, atol=1e-8)

    def test_contact_sticking_limit_and_separation(self):
        self.v.fill(0)
        f, _k, damping, dmu = self.solver.contact(self.q, self.v, 0.6)
        self.assertAlmostEqual(f[1], -20)
        self.assertAlmostEqual(damping[0, 0], 0.6 * 20 / 0.05, places=5)
        np.testing.assert_array_equal(dmu, 0)
        np.testing.assert_array_equal(f[6:], 0)
        self.q[:, 0, 1] = 0.02
        for value in self.solver.contact(self.q, self.v, 0.6):
            np.testing.assert_array_equal(value, 0)

    def test_optimizer_uses_observed_motion_and_sensitivity(self):
        example = Example.__new__(Example)
        example.phase = "Learning"
        example.mu = 0.2
        example._validation = False
        example._accepted_loss = float("inf")
        example.history, example._paths, example.results = [], {}, {}
        example.args = SimpleNamespace(train=True)
        example.dynamics = ANCFTireTraction.__new__(ANCFTireTraction)
        # A local linear physical response gives an exact known update direction.
        example._trace = [
            SimpleNamespace(
                z=np.array([0, 0.2]),
                zd=np.array([0, 0.2]),
                sensitivity=np.array([0, 1]),
                velocity_sensitivity=np.array([0, 1]),
            )
        ]
        example.references = {False: [SimpleNamespace(z=np.array([0, 0.6]), zd=np.array([0, 0.6]))]}
        example._begin = lambda phase: setattr(example, "phase", phase)
        example._finish()
        self.assertGreater(example.mu, 0.2)
        self.assertLess(example.mu, 0.6)
        accepted = example.mu
        example._accepted_loss = 1e-10
        example._finish()
        self.assertLess(example.mu, accepted)


@unittest.skipUnless(
    os.environ.get("ANCF_TRACTION_TESTS") == "1", "Opt-in full ANCF trajectory solves (USD assets required)"
)
class TestTractionDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver, cls.initial, _mesh, _hub = make_experiment()

    def rollout(self, mu, validation=False, sensitivity=True):
        state, states = self.initial, []
        for speed in motor_commands(validation):
            state = self.solver.step(state, float(speed), mu, sensitivity=sensitivity)
            states.append(state)
        return states

    def test_trajectory_gradient_matches_resolved_rollouts(self):
        reference = self.rollout(0.65, sensitivity=False)
        states = self.rollout(0.4)
        eps = 1e-4
        plus = self.rollout(0.4 + eps, sensitivity=False)
        minus = self.rollout(0.4 - eps, sensitivity=False)
        numeric = (np.array([s.z[-1] for s in plus]) - np.array([s.z[-1] for s in minus])) / (2 * eps)
        analytic = np.array([s.sensitivity[-1] for s in states])
        np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-5)
        _loss, gradient, information = self.solver.loss_gradient(states, reference)
        numeric_loss = (
            self.solver.loss_gradient(plus, reference)[0] - self.solver.loss_gradient(minus, reference)[0]
        ) / (2 * eps)
        self.assertAlmostEqual(gradient, numeric_loss, delta=max(1e-6, abs(gradient) * 1e-3))
        self.assertGreater(information, 0.1)
        np.testing.assert_array_equal(self.initial.sensitivity, 0)
        np.testing.assert_array_equal(self.initial.zd, 0)

    def test_exact_step_jacobian_and_repeatable_reset(self):
        previous = self.initial
        for speed in motor_commands()[:5]:
            previous = self.solver.step(previous, float(speed), 0.4)
        current = self.solver.step(previous, 6.0, 0.4)
        residual, jac, _damping, dmu = self.solver.evaluate(current.z, previous, current.angle, 0.4)
        direction = np.random.default_rng(8).normal(size=self.solver.size)
        direction /= np.linalg.norm(direction)
        eps = 1e-7
        plus = self.solver.evaluate(current.z + eps * direction, previous, current.angle, 0.4, False)[0]
        minus = self.solver.evaluate(current.z - eps * direction, previous, current.angle, 0.4, False)[0]
        np.testing.assert_allclose(jac @ direction, (plus - minus) / (2 * eps), rtol=2e-4, atol=0.002)
        plus_mu = self.solver.evaluate(current.z, previous, current.angle, 0.4 + eps, False)[0]
        minus_mu = self.solver.evaluate(current.z, previous, current.angle, 0.4 - eps, False)[0]
        np.testing.assert_allclose(dmu, (plus_mu - minus_mu) / (2 * eps), rtol=1e-5, atol=1e-7)
        self.assertLess(np.linalg.norm(residual, np.inf), 1e-6)
        again = self.solver.step(previous, 6.0, 0.4)
        np.testing.assert_array_equal(again.q, current.q)
        np.testing.assert_array_equal(again.sensitivity, current.sensitivity)

    def test_friction_recovery_and_independent_validation(self):
        example = Example(None, Example.create_parser().parse_args([]))
        for _ in range(1600):
            example.step()
            if example.phase in ("Ready", "Stopped"):
                break
        check_tire_traction_result(example)
        self.assertEqual(example.phase, "Ready")
        self.assertIn("validation_after", example.results)

    def test_smaller_time_step_preserves_travel(self):
        coarse = self.rollout(0.65, sensitivity=False)[-1]
        dt = self.solver.dt
        try:
            self.solver.dt = dt / 2
            fine = self.initial
            for speed in motor_commands(dt=dt / 2):
                fine = self.solver.step(fine, float(speed), 0.65, sensitivity=False)
        finally:
            self.solver.dt = dt
        self.assertLess(abs(fine.z[-1] - coarse.z[-1]), 0.05)
        self.assertLess(abs(fine.z[-2] - coarse.z[-2]), 0.005)


if __name__ == "__main__":
    unittest.main()
