# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Subprocess measurements for test_ancf_example_physics.

Uses the launcher's existing config resolver. Only the outer runner is replaced;
example construction, controls, and physics are unchanged. Test-only readbacks
occur every six frames. Run through docker/test-ancf-examples.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples
from newton.examples.ancf._vehicle_usd import find_body

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "newton/examples/ancf/assets"
CASES = {
    "01_free_fall": ("01_ancf_shell_drop", 6),
    "01_drop": ("01_ancf_shell_drop", 300),
    "02_loaded": ("02_ancf_rigid_mujoco_tires", 300),
    "03_stationary": ("03_vehicle_ancf_tires", 300),
    "03_driving": ("03_vehicle_ancf_tires", 360),
    "03_braking": ("03_vehicle_ancf_tires", 540),
    "03_warthog_motion": ("03_vehicle_ancf_tires", 720),
    "03_vehicle_motion": ("03_vehicle_ancf_tires", 1500),
    "05_flat": ("05_vehicle_terrain", 300),
    "05_relief": ("05_vehicle_terrain", 300),
    "05_driving": ("05_vehicle_terrain", 540),
    "07_replay": ("07_vehicle_telemetry", 2400),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_launcher():
    """Import docker/run-example.py, which lives outside the package, as a module."""
    spec = importlib.util.spec_from_file_location("ancf_test_config", ROOT / "docker/run-example.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _configuration(case: str, extra: dict | None = None) -> tuple[str, list[str], Path]:
    name, frames = CASES[case]
    config = ROOT / "docker/config" / f"{name}.json"
    helper = load_launcher()
    overrides = {
        "viewer": "null",
        "headless": True,
        "num-frames": frames,
        "diag-period": 120,
        "substeps": 6,
        "nr-iters": 2,
        "pcg-iters": 10,
        "tire-asset": "warthog_ancf_tire_simple.usda",
    }
    if case.startswith("01"):
        # Independent, identical tires translated by 2 m. Zero pressure isolates
        # gravity; 3 psi is a controlled drop input, not measured RELLIS pressure.
        pressure = 0.0 if case == "01_free_fall" else 3.0 * 6894.757
        overrides["n-envs"] = 2
        overrides["shell-tires"] = [{"position": [offset, 0.0, 0.0], "pressure": pressure} for offset in (0.0, 2.0)]
    else:
        overrides["vehicle-asset"] = "warthog_vehicle.usdc"
        if case.startswith("02"):
            cfg = json.loads(config.read_text())["args"]["shell-tires"][0]
            overrides.update({"n-envs": 1, "shell-tires": [cfg], "rpm": 0.0, "f-load": 50.0})
        else:
            overrides["wheel-speed"] = 3.0 if case in ("03_driving", "03_braking") else 0.0
            overrides["steer-angle"] = 0.0
            if case.startswith(("05", "07")):
                overrides.update(
                    {
                        "terrain": "rellis_00000",
                        "difficulty": 0.0 if case == "05_flat" else 1.0,
                        "controller": "track" if case.startswith("07") else "manual",
                    }
                )
            if case.startswith("07"):
                overrides["replay"] = True
    overrides.update(extra or {})
    raw_sets = [
        f"{key}={json.dumps(value) if not isinstance(value, str) else value}" for key, value in overrides.items()
    ]
    cli = helper.resolve(config, False, raw_sets)
    # --test changes example 05's terrain and manual commands. Use the real
    # scene settings here and make independent assertions in the parent.
    return json.loads(config.read_text())["example"], cli, config


def _rotate(quaternion: np.ndarray, points: np.ndarray) -> np.ndarray:
    vector, scalar = quaternion[:3], quaternion[3]
    return points + 2.0 * np.cross(vector, np.cross(vector, points) + scalar * points)


def _hull_surface_points(example) -> np.ndarray:
    """Sample collision boxes in body coordinates, including face centres."""
    model = example.model
    chassis = find_body(model, "chassis")
    bodies = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    types = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    transforms = model.shape_transform.numpy()
    unit = np.array([p for p in itertools.product((-1.0, 0.0, 1.0), repeat=3) if any(p)])
    samples = []
    for i in np.flatnonzero(bodies == chassis):
        if not flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if types[i] != int(newton.GeoType.BOX):
            raise AssertionError("Hull-clearance fixture currently supports box colliders only")
        samples.append(_rotate(transforms[i, 3:], unit * scales[i]) + transforms[i, :3])
    if not samples:
        raise AssertionError("No chassis collision boxes found")
    return np.concatenate(samples)


def _measure(example, case: str, frame: int, hull_points=None) -> dict:
    drop = case.startswith("01")
    ancf = example.solver if drop else example.ancf_solver
    x = example._node_x().numpy() if drop else ancf.node_x.numpy()
    xd = example._node_xd().numpy() if drop else ancf.node_xd.numpy()
    if not (np.isfinite(x).all() and np.isfinite(xd).all()):
        raise AssertionError(f"{case}: non-finite tire state at frame {frame}")
    for field in ("node_D", "node_Dd", "node_xdd", "node_Ddd", "global_f_int", "global_f_ext"):
        array = getattr(ancf, field, None)
        if array is not None and not np.isfinite(array.numpy()).all():
            raise AssertionError(f"{case}: non-finite {field} at frame {frame}")
    sample = {"t": frame / 60.0, "node_y_min": float(x[:, 1].min())}
    if drop:
        n = example._n_nodes
        weights = ancf.lumped_mass.numpy().reshape(n, 6)[:, 0].astype(float)
        weights /= weights.sum()
        positions = x.reshape(example._n_envs, n, 3)
        velocities = xd.reshape(example._n_envs, n, 3)
        sample["com"] = np.einsum("n,enp->ep", weights, positions).tolist()
        sample["com_velocity"] = np.einsum("n,enp->ep", weights, velocities).tolist()
        sample["vertical_extent"] = np.ptp(positions[:, :, 1], axis=1).tolist()
        if example._n_envs > 1:
            difference = positions[1] - positions[0] - np.array([2.0, 0.0, 0.0])
            sample["translation_error"] = float(np.linalg.norm(difference, axis=1).max())
        return sample

    q = example.state_0.body_q.numpy()
    qd = example.state_0.body_qd.numpy()
    if not (np.isfinite(q).all() and np.isfinite(qd).all()):
        raise AssertionError(f"{case}: non-finite rigid state at frame {frame}")
    staging = np.array([a.numpy()[0] for a in ancf._xfrc_stg_per_tire])
    sample["support_force"] = (staging[:, 4] + example._fz_tare).tolist()
    if case.startswith("02"):
        sample["pose"] = q[example._spindle_newton_idx].tolist()
        sample["velocity"] = qd[example._spindle_newton_idx].tolist()
        return sample

    chassis = find_body(example.model, "chassis")
    sample["pose"] = q[chassis].tolist()
    sample["velocity"] = qd[chassis].tolist()
    dofs = example.vehicle._axle_dofs.numpy()
    signs = example.vehicle._axle_sign.numpy()
    sample["wheel_velocity"] = (example.state_0.joint_qd.numpy()[dofs] * signs).tolist()
    sample["motor_torque"] = example.control.joint_f.numpy()[dofs].tolist()
    sample["tire_moment"] = staging[:, [2, 0, 1]].tolist()  # Y-up -> Z-up
    sample["applied_moment"] = example.solver.xfrc_applied.numpy()[0, example._spindle_mj_arr.numpy(), 3:].tolist()
    if example._gs_coupler is not None:
        sample["coupling_velocity_residual"] = example._gs_coupler.interface_residual.numpy().tolist()
        sample["coupling_iterations_used"] = int(example._gs_coupler.interface_iterations.numpy()[0])
        sample["coupling_totals"] = example._gs_coupler.interface_totals.numpy().tolist()
        sample["coupling_probes"] = int(example._gs_coupler.interface_probe_count.numpy()[0])
    sample["wheel_target"] = (example.control.joint_target_qd.numpy()[dofs] * signs).tolist()
    sample["fault_count"] = int(getattr(example, "fault_count", 0))
    sample["fault"] = str(getattr(example, "_fault", ""))
    sample["episode"] = int(getattr(example, "episode", 0))
    if case.startswith(("05", "07")):
        points = _rotate(q[chassis, 3:], hull_points) + q[chassis, :3]
        sample["sampled_hull_clearance"] = min(
            float(p[2]) - float(example.terrain.height_at(float(p[0]), float(p[1]))) for p in points
        )
        data = example.solver.mjw_data
        count = int(data.nacon.numpy()[0])
        sample["rigid_constraint_generalized_force"] = data.qfrc_constraint.numpy()[0].tolist()
        sample["rigid_contacts"] = []
        if count:
            pairs = data.contact.geom.numpy()[:count]
            positions = data.contact.pos.numpy()[:count]
            distances = data.contact.dist.numpy()[:count]
            shapes = example.solver.mjc_geom_to_newton_shape.numpy()[0]
            sample["rigid_contacts"] = [
                {
                    "shapes": [example.model.shape_label[int(shapes[g])] for g in pair],
                    "position": position.tolist(),
                    "distance": float(distance),
                }
                for pair, position, distance in zip(pairs, positions, distances, strict=True)
            ]
    if case.startswith("07"):
        sample["recording_time"] = example._ghost_time()
        sample["replay_error"] = list(example._ghost_err)
        sample["command"] = [example.v_cmd, example.yaw_cmd]
    return sample


def probe(case: str, output: Path, overrides: dict | None = None, *, benchmark: bool = False) -> None:
    name, cli, config = _configuration(case, overrides)
    module = f"newton.examples.ancf.example_{name}"
    report = {
        "case": case,
        "argv": cli,
        "config_sha256": _sha256(config),
        "example_sha256": _sha256(ROOT / Path(*module.split(".")).with_suffix(".py")),
        "shared_source_sha256": {
            name: _sha256(ROOT / "newton/examples/ancf" / name)
            for name in (
                "_vehicle_config.py",
                "_vehicle_kernels.py",
                "_vehicle_simulation.py",
                "_vehicle_terrain.py",
                "_vehicle_usd.py",
                "_terrain_common.py",
            )
        },
        "tire_sha256": _sha256(ASSETS / cli[cli.index("--tire-asset") + 1]),
        "overrides": overrides or {},
        "samples": [],
        "complete": False,
    }

    def observe(example, args):
        hull_points = _hull_surface_points(example) if case.startswith(("05", "07")) else None
        collider = importlib.import_module("mujoco_warp._src.collision_convex")
        report["collision_source_sha256"] = _sha256(Path(collider.__file__))
        report["heightfield_prism_capacity"] = getattr(collider, "MJ_MAXHFPRISM", None)
        report["gravity"] = example.model.gravity.numpy().tolist()
        report["solver_budget"] = [args.substeps, args.nr_iters, args.pcg_iters]
        shell = example.solver if case.startswith("01") else example.ancf_solver
        if benchmark:
            report["setup_seconds"] = time.perf_counter() - setup_start
            report["device"] = wp.get_device("cuda:0").name
            report["warp_version"] = wp.__version__
            report["solver_source_sha256"] = {
                p.name: _sha256(p) for p in sorted((ROOT / "newton/_src/solvers/ancf_shell").glob("*.py"))
            }
            graph = getattr(example, "_substep_graph", getattr(example, "_simulation_graph", None))
            graph_substeps = getattr(example, "_graph_substeps", args.substeps) if graph is not None else 0
            report["captured_substeps"] = graph_substeps
            report["frame_graph_captured"] = graph_substeps == args.substeps
            report["timings"] = []
            report["warmup_frames"] = 30 if case.startswith("01") else 120
            report["physics"] = {key: float(getattr(shell, key)) for key in ("kn", "kd", "mu", "v_reg", "ground_z")}
            report["physics"]["n_envs"] = shell.n_envs
            report["physics"]["thickness_gp"] = shell.thickness_gp
            for key, array in {
                "material": example.ancf_model.elem_mat,
                "thickness": example.ancf_model.elem_h,
                "mass": shell.lumped_mass,
                "gas_amount": shell.cav_Kgas,
                "build_pressure": shell.cav_pbuild,
            }.items():
                report["physics"][key + "_sha256"] = hashlib.sha256(array.numpy().tobytes()).hexdigest()
            if not case.startswith("01"):
                report["physics"]["torque_alpha"] = shell.torque_alpha
                report["physics"]["rigid_mass_sha256"] = hashlib.sha256(
                    example.model.body_mass.numpy().tobytes()
                ).hexdigest()
            start_event = wp.Event("cuda:0", enable_timing=True)
            end_event = wp.Event("cuda:0", enable_timing=True)
        report["fem_tire_mass_kg"] = float(shell.lumped_mass.numpy().reshape(-1, 6)[:, 0].sum())
        if not case.startswith("01"):
            report["vehicle_sha256"] = _sha256(ASSETS / cli[cli.index("--vehicle-asset") + 1])
            if case.startswith("02"):
                report["expected_support_force"] = example._m_rigid * 9.81 + example._f_load
                report["requested_preload"] = 50.0
                report["effective_preload"] = example._f_load
            else:
                report["expected_support_force"] = float(example.model.body_mass.numpy().sum()) * 9.81
                report["model_total_mass_kg"] = (
                    float(example.model.body_mass.numpy().sum()) + 4 * report["fem_tire_mass_kg"]
                )
                limits = example.model.joint_effort_limit.numpy()[example.vehicle._axle_dofs.numpy()]
                report["effort_limit"] = float(limits.max()) if np.isfinite(limits).all() else None
                report["rolling_radius"] = example.vehicle.r_roll
                report["torque_alpha"] = example.ancf_solver.torque_alpha
                report["coupling_iterations"] = example._gs_coupler._n_iters if example._gs_coupler else 1
                report["coupling_method"] = getattr(args, "coupling_method", "aitken")
                report["coupling_graph_captured"] = example._substep_graph is not None
                report["response_reuse"] = bool(example._gs_coupler and example._gs_coupler.reuse_response)
                combined = example._gs_coupler._coupled_solver if example._gs_coupler else None
                if not report["coupling_graph_captured"]:
                    combined = None
                report["coupled_newton_active"] = combined is not None
                report["coupled_linear_iterations"] = combined.linear_iterations if combined else None
                report["coupling_max_evaluations"] = combined.maximum + 1 if combined else report["coupling_iterations"]
                report["coupling_work_unit"] = "joint Newton evaluation" if combined else "complete shell step"

                report["joint_velocity_dofs"] = example.state_0.joint_qd.shape[0]
                report["tire_nodes"] = example.ancf_model.n_nodes
                report["max_wheel_speed"] = example.spec.max_wheel_speed
        if case.startswith(("05", "07")):
            report["terrain_stage"] = float(example.terrain.w)
            report["rigid_terrain"] = bool(example.terrain_scm.rigid)
        if case.startswith("07"):
            report["replay"] = bool(example.replay)
            report["command_end_time"] = float(example.telemetry.cmd_t[-1])
            report["trajectory_end_time"] = example.telemetry.duration
        if case == "01_drop" and not benchmark:
            # Inspect every substep, so a short impact cannot fall between 0.1 s samples.
            # Readbacks are confined to this independent acceptance probe.
            example._substep_graph = None
            advance = example.solver.graph_step
            report["peak_penetration"] = 0.0

            def measured_substep():
                advance()
                minimum = float(example.solver.node_x.numpy()[:, 1].min())
                report["peak_penetration"] = max(report["peak_penetration"], -minimum)

            example.solver.graph_step = measured_substep
            report["contact_stiffness"] = example.solver.kn
        report["samples"].append(_measure(example, case, 0, hull_points))
        for frame in range(1, args.num_frames + 1):
            if benchmark and (frame - 1) % 6 == 0:
                block_start = time.perf_counter()
                wp.record_event(start_event)
            if case in ("03_braking", "05_driving"):
                example._target_wheel_speed = 3.0 if frame <= 180 else (0.0 if frame <= 360 else -3.0)
            if case == "03_warthog_motion":
                # Include the 6 rad/s driving case missed by the original 3 rad/s test.
                example._target_wheel_speed = 0.0 if frame <= 120 or frame > 600 else (3.0 if frame <= 240 else 6.0)
                example.steer_angle = 0.5 if 480 < frame <= 600 else 0.0
            if case == "03_vehicle_motion":
                # Exercise the same controls on skid-steered and articulated vehicles.
                speed, steer = 0.0, 0.0
                if 120 < frame <= 300:
                    speed = 3.0
                elif 420 < frame <= 600:
                    speed = -3.0
                elif 720 < frame <= 1080:
                    speed = 3.0
                    steer = 0.5 * example.spec.max_steer * (1 if frame <= 900 else -1)
                elif 1080 < frame <= 1320:
                    speed = 6.0
                example._target_wheel_speed = speed
                example.steer_angle = steer
            example.step()
            if frame % 6 == 0 or frame == args.num_frames:
                if benchmark:
                    wp.record_event(end_event)
                    gpu_ms = wp.get_event_elapsed_time(start_event, end_event)
                    wall_seconds = time.perf_counter() - block_start
                    report["timings"].append(
                        {
                            "end_frame": frame,
                            "frames": frame % 6 or 6,
                            "wall_seconds": wall_seconds,
                            "gpu_ms": gpu_ms,
                        }
                    )
                sample = _measure(example, case, frame, hull_points)
                report["samples"].append(sample)
                if benchmark and (sample.get("fault_count", 0) or sample.get("fault") or sample.get("episode", 0)):
                    raise AssertionError(f"{case}: simulation fault/reset at frame {frame}")
        if case.startswith("03") and not benchmark:
            example.test_final()
            report["builtin_test_passed"] = True
        report["complete"] = True

    setup_start = time.perf_counter()
    try:
        with patch.object(sys, "argv", [module, *cli]), patch.object(newton.examples, "run", observe):
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overrides", type=json.loads, default=None)
    parser.add_argument("--benchmark", action="store_true")
    options = parser.parse_args()
    probe(options.case, options.output, options.overrides, benchmark=options.benchmark)
