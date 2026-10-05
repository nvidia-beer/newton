# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wheel-only identification experiment using the actual example 01 implicit solver.

Synthetic recovery verifies the calibration machinery, not real tire accuracy.
All measurements and parameter searches stay outside the interactive example.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton.examples.ancf.example_ancf_shell_drop import Example
from newton.solvers import load_ancf_tire_usd
from newton.tests.ancf_example_probe import ASSETS


def make_wheel(
    young_pa=None,
    *,
    clearance_m=1.0,
    density=None,
    n_envs=1,
    substeps=6,
    stiffness_scale=1.0,
    nr_iters=2,
    pcg_iters=10,
):
    """Construct uncoupled Warthog tires with a known initial clearance [m]."""
    if not np.isfinite([clearance_m, stiffness_scale]).all() or clearance_m <= 0 or stiffness_scale <= 0:
        raise ValueError("Clearance and stiffness scale must be finite and positive")
    cfg = {"position": [0.0, clearance_m - 1.0, 0.0], "pressure": 3.0 * 6894.757}
    if young_pa is not None:
        cfg.update(E=float(young_pa), nu=0.3, rho=1100.0 if density is None else density)
    elif density is not None:
        cfg["rho"] = density

    def load_material(*args, **kwargs):
        model, meta = load_ancf_tire_usd(*args, **kwargs)
        material = model.elem_mat.numpy()
        material[:, :9] *= stiffness_scale
        model.elem_mat.assign(material)
        return model, meta

    with patch("newton.examples.ancf.example_ancf_shell_drop.load_ancf_tire_usd", load_material):
        return Example(
            None,
            SimpleNamespace(
                tire_asset="warthog_ancf_tire_simple.usda",
                shell_tires=[cfg],
                n_envs=n_envs,
                substeps=substeps,
                nr_iters=nr_iters,
                pcg_iters=pcg_iters,
                kn=2.0e7,
                kd=None,
                contact_penetration=0.01,
                diag_period=1000000,
            ),
        )


def observe(wheel, frames=90):
    """Sample height and vertical extent [m], velocity [m/s], and penetration [m]."""
    solver = wheel.solver
    mass = solver.lumped_mass.numpy().reshape(wheel._n_nodes, 6)[:, 0].astype(float)
    weights = mass / mass.sum()
    rows = []
    for frame in range(frames + 1):
        x, v = wheel._node_x().numpy(), wheel._node_xd().numpy()
        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            raise ValueError(f"Non-finite wheel state at frame {frame}")
        rows.append([weights @ x[:, 1], np.ptp(x[:, 1]), weights @ v[:, 1], max(0.0, -x[:, 1].min())])
        if frame < frames:
            wheel.step()
    return np.asarray(rows)


def fit_stiffness(measured, rollout, *, bounds=(0.6, 1.8), noise_m=0.0005, refinements=8):
    """Fit a positive stiffness multiplier to height/extent observations [m].

    A coarse scan brackets the best basin before a bounded golden-section search
    in log space. This deliberately small one-parameter experiment needs no new
    optimizer dependency. ``rollout(scale)`` must return independent predictions;
    neither generating parameters nor validation observations enter the search.
    """
    measured = np.asarray(measured, dtype=float)
    lower, upper = bounds
    if measured.ndim != 2 or measured.shape[1] != 2 or len(measured) < 2 or not np.isfinite(measured).all():
        raise ValueError("Expected finite (samples, 2) height/extent observations")
    if not np.isfinite([lower, upper, noise_m]).all() or not 0 < lower < upper or noise_m <= 0:
        raise ValueError("Positive ordered bounds and measurement noise are required")
    if refinements < 0:
        raise ValueError("Refinements must be nonnegative")
    trials, predictions = {}, {}

    def evaluate(log_scale):
        scale = float(np.exp(log_scale))
        if scale not in trials:
            try:
                predicted = np.asarray(rollout(scale), dtype=float)
                if predicted.shape != measured.shape or not np.isfinite(predicted).all():
                    raise ValueError("Invalid predicted observations")
                loss = float(np.mean(((predicted - measured) / noise_m) ** 2))
                trials[scale] = {"scale": scale, "loss": loss, "failure": None}
                predictions[scale] = predicted
            except (ValueError, FloatingPointError) as error:
                trials[scale] = {"scale": scale, "loss": None, "failure": str(error)}
        loss = trials[scale]["loss"]
        return float("inf") if loss is None else loss

    grid = np.linspace(np.log(lower), np.log(upper), 7)
    losses = [evaluate(x) for x in grid]
    if not predictions:
        raise ValueError("All calibration trials failed")
    spread = float(np.sqrt(np.mean(np.ptp(np.array(list(predictions.values())), axis=0) ** 2)))
    if spread <= noise_m:
        raise ValueError("Stiffness response is below measurement noise; use a contact/deformation experiment")
    best = int(np.argmin(losses))
    lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f_left, f_right = evaluate(left), evaluate(right)
    for _ in range(refinements):
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = evaluate(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = evaluate(right)
    valid = [trial for trial in trials.values() if trial["loss"] is not None]
    winner = min(valid, key=lambda trial: trial["loss"])
    return {
        "stiffness_scale": winner["scale"],
        "normalized_mse": winner["loss"],
        "response_spread_m": spread,
        "bound_hit": bool(min(abs(np.log(winner["scale"] / lower)), abs(np.log(winner["scale"] / upper))) < 0.01),
        "trials": list(trials.values()),
    }


def experiment(output, *, frames=48):
    """Recover a hidden synthetic stiffness, then validate at another height."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    truth, nominal, noise = 1.23, 1.0, 0.0005

    def calibrated_wheel(**kwargs):
        return make_wheel(substeps=20, nr_iters=4, pcg_iters=40, **kwargs)

    target = observe(calibrated_wheel(stiffness_scale=truth), frames)
    measured = target[:, :2] + np.random.default_rng(20261003).normal(0.0, noise, target[:, :2].shape)
    cache = {}

    def rollout(scale):
        trace = observe(calibrated_wheel(stiffness_scale=scale), frames)
        cache[scale] = trace
        print(f"[wheel calibration] trial scale={scale:.6f}", flush=True)
        return trace[:, :2]

    result = fit_stiffness(measured, rollout, noise_m=noise)
    fitted = result["stiffness_scale"]
    baseline = observe(calibrated_wheel(stiffness_scale=nominal), frames)
    prediction = cache[fitted]
    # These different-height data are first generated after fitting is finished.
    held_truth = observe(calibrated_wheel(stiffness_scale=truth, clearance_m=0.35), frames)
    held_fit = observe(calibrated_wheel(stiffness_scale=fitted, clearance_m=0.35), frames)
    held_base = observe(calibrated_wheel(stiffness_scale=nominal, clearance_m=0.35), frames)
    refined = observe(make_wheel(stiffness_scale=truth, substeps=40, nr_iters=4, pcg_iters=40), frames)
    production = observe(make_wheel(stiffness_scale=truth), frames)

    def rms(a, b):
        return float(np.sqrt(np.mean((a[:, :2] - b[:, :2]) ** 2)))

    result.update(
        synthetic=True,
        physical_calibration=False,
        parameter="uniform multiplier of the baked orthotropic stiffness tensor; density and damping unchanged",
        truth_scale=truth,
        parameter_relative_error=abs(fitted / truth - 1),
        train_nominal_rmse_m=rms(baseline, target),
        train_fitted_rmse_m=rms(prediction, target),
        validation_nominal_rmse_m=rms(held_base, held_truth),
        validation_fitted_rmse_m=rms(held_fit, held_truth),
        time_refinement_rmse_m=rms(target, refined),
        noise_std_m=noise,
        frames=frames,
        coupled=False,
        solver_budget=[20, 4, 40],
        refinement_budget=[40, 4, 40],
        production_budget=[6, 2, 10],
        production_reference_rmse_m=rms(production, target),
        contact_kn_n_m=2.0e7,
        pressure_pa=3.0 * 6894.757,
        training_clearance_m=1.0,
        validation_clearance_m=0.35,
        max_frame_sampled_penetration_m=float(
            max(trace[:, 3].max() for trace in [target, baseline, held_truth, held_fit, refined, *cache.values()])
        ),
    )
    asset = ASSETS / "warthog_ancf_tire_simple.usda"
    result["asset_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    np.savez_compressed(
        output / "traces.npz",
        t=np.arange(frames + 1) / 60.0,
        measured=measured,
        target=target,
        nominal=baseline,
        fitted=prediction,
        held_truth=held_truth,
        held_fit=held_fit,
        held_nominal=held_base,
        refined=refined,
        production=production,
    )
    result["recovery_pass"] = bool(result["parameter_relative_error"] < 0.02 and not result["bound_hit"])
    result["validation_pass"] = bool(
        result["validation_fitted_rmse_m"] < 0.001
        and result["validation_fitted_rmse_m"] < 0.3 * result["validation_nominal_rmse_m"]
    )
    result["reference_time_refinement_pass"] = bool(result["time_refinement_rmse_m"] < 0.001)
    result["production_accuracy_pass"] = bool(result["production_reference_rmse_m"] < 0.001)
    result["ready_for_coupling"] = all(
        result[key]
        for key in ("recovery_pass", "validation_pass", "reference_time_refinement_pass", "production_accuracy_pass")
    )
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    experiment(args.output)
