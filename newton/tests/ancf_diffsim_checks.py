# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Acceptance checks used by the ANCF diffsim examples with ``--test``.

The examples keep only the hooks required by the example runner. Tolerances,
assertions, and diagnostic readbacks belong here, shared by the examples'
``test_final`` hooks and the opt-in physical tests. These checks are numerical
screens on synthetic experiments; they do not establish agreement with a real vehicle.
"""

import math

import numpy as np


def check_skid_steer_result(example) -> None:
    """Require a completed calibration to improve held-out physical motion."""
    example._observe()
    for name in ("node_x", "node_D", "node_xd", "node_Dd", "global_f_int"):
        if not np.isfinite(getattr(example.rig.ancf_solver, name).numpy()).all():
            raise AssertionError(f"Non-finite skid-steer {name}")
    if example.args.train and example._frame >= 3972 and not example.calibrated:
        raise AssertionError("Calibration did not finish")
    if not example.calibrated or "Learned" not in example.results:
        return
    if example.response.history[-1] >= 0.02 * example.response.history[0]:
        raise AssertionError("Response learning did not reduce velocity loss")
    before, after = example.results["Before"], example.results["Learned"]
    if abs(after["speed_error_m_s"]) > 0.04:
        raise AssertionError(f"Learned speed misses target: {after}")
    if abs(after["yaw_error_rad_s"]) > math.radians(0.6):
        raise AssertionError(f"Learned turn misses target: {after}")
    if after["position_error_m"] >= before["position_error_m"]:
        raise AssertionError(f"Learned drive did not improve the trajectory: {example.results}")


def check_tire_lift_forward(example) -> None:
    """Check the coupled forward state independently of the training objective."""
    rig = example.rig
    for name in ("node_x", "node_D", "node_xd", "node_Dd", "global_f_int"):
        if not np.isfinite(getattr(rig.ancf_solver, name).numpy()).all():
            raise AssertionError(f"Non-finite {name} in tire-lift preview")
    if not np.isfinite(rig.state_0.body_q.numpy()).all():
        raise AssertionError("Non-finite rigid state in tire-lift preview")
    if not np.isfinite([example.height, example.loss]).all() or example.height <= 0.0:
        raise AssertionError("Invalid rim height/loss in tire-lift preview")
    if rig._n_envs != 1 or rig._current_rpm != [0.0]:
        raise AssertionError("The tire-lift experiment requires one non-spinning tire")


def check_tire_lift_result(example) -> None:
    """A full training run must reach its physical target, not just stay finite."""
    check_tire_lift_forward(example)
    if example._training_requested and example._measurements and example.rig._frame >= 120:
        if not example._fit_done:
            raise AssertionError(f"Static stiffness fitting failed: {example.training_status}")
    if example._training_requested and example.rig._frame >= 960 and not example._measurements:
        if example.training_status != "Converged" or abs(example.height_error) >= 2.5e-4:
            raise AssertionError(f"Pressure learning failed: {example.training_status}, error={example.height_error} m")


def check_tire_traction_result(example) -> None:
    """Check convergence and an independently driven validation trajectory."""
    if not np.isfinite(example.current.q).all() or example.current.residual > 1e-6:
        raise AssertionError("Invalid or unconverged tire state")
    if example.phase == "Stopped":
        raise AssertionError(example.status)
    if not example.args.train or (example._frame < 1600 and example.phase != "Ready"):
        return
    if "validation_after" not in example.results:
        raise AssertionError("Friction learning/validation did not complete")
    if abs(example.mu - example.reference_mu) > 0.005:
        raise AssertionError("Failed to recover the synthetic friction coefficient")
    before, after = example.results["validation_before"], example.results["validation_after"]
    if after["position_rmse_m"] > 0.002 or after["loss"] > max(1e-7, 0.01 * before["loss"]):
        raise AssertionError(f"Independent validation failed: {before} -> {after}")
    if np.any(np.diff([r["loss"] for r in example.history]) > 1e-8):
        raise AssertionError("An accepted learning step increased loss")
