# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Acceptance checks used by full-vehicle examples with ``--test`` and probes.

The examples keep only the hooks required by the example runner. Physical
tolerances, assertions, and diagnostic readbacks belong here. These checks are
numerical screens; they do not establish agreement with a real vehicle.
"""

import math

import numpy as np

from newton.examples.ancf import _vehicle_config as vehicle_config


def check_vehicle_step(example) -> None:
    x_np = example.ancf_solver.node_x.numpy()
    if not np.all(np.isfinite(x_np)):
        raise AssertionError(f"non-finite ANCF node_x at frame {example._frame}, t={example._t:.3f}s")


def check_vehicle_final(example) -> None:
    x_np = example.ancf_solver.node_x.numpy()
    xd_np = example.ancf_solver.node_xd.numpy()
    assert np.all(np.isfinite(x_np)), "non-finite node_x at test_final"
    assert np.all(np.isfinite(xd_np)), "non-finite node_xd at test_final"

    stg_per_wheel = [example.ancf_solver._xfrc_stg_per_tire[w].numpy()[0] for w in range(vehicle_config.N_TIRES)]
    # MuJoCo's cached xpos/xquat precede integration; body_q is at t_{n+1}.
    spindle_poses = example.state_0.body_q.numpy()[example._spindle_body_indices_np]
    rest_np = example._bead_rest_np
    nn = example._n_nodes
    # Single-wheel rig metadata excludes the full articulated chassis load.
    fz_exp = float(example.model.body_mass.numpy().sum()) * vehicle_config.GRAVITY / vehicle_config.N_TIRES

    for e in range(vehicle_config.N_TIRES):
        x_e = x_np[e * nn : (e + 1) * nn]
        pos_zu = spindle_poses[e, :3]
        qx, qy, qz, qw = (float(value) for value in spindle_poses[e, 3:])
        u = np.array([qx, qy, qz])
        r_mj = np.stack([rest_np[:, 2], rest_np[:, 0], rest_np[:, 1]], axis=1)
        r_rot = (
            r_mj * (qw * qw - u @ u)
            + 2.0 * (r_mj @ u)[:, None] * u[None, :]
            + 2.0 * qw * np.cross(np.broadcast_to(u, r_mj.shape), r_mj)
        )
        p_mj = r_rot + pos_zu
        expected = np.stack([p_mj[:, 1], p_mj[:, 2], p_mj[:, 0]], axis=1)
        max_drift = float(np.max(np.linalg.norm(x_e[example._bead_idx_np] - expected, axis=1)))
        assert max_drift < 1e-3, f"FAIL tire {e}: bead drift {max_drift * 1e3:.2f} mm > 1 mm"
        fz = abs(float(stg_per_wheel[e][4]) + example._fz_tare)
        rel = abs(fz - fz_exp) / max(fz_exp, 1.0)
        assert rel < 0.50, f"FAIL tire {e}: F_z={fz:.1f} N  expected~{fz_exp:.1f} N  err={rel * 100:.1f}% > 50%"

    assert np.all(np.isfinite(example.ancf_solver.global_f_int.numpy())), "non-finite global_f_int at test_final"
    print(f"[PASS] frame={example._frame}  t={example._t:.1f}s  all 4 tires OK")


def check_terrain_final(example) -> None:
    # The base F_z check assumes four tires on a plane; on rocks the loads are anything.
    x_np = example.ancf_solver.node_x.numpy()
    assert np.all(np.isfinite(x_np)), "non-finite node_x at test_final"
    assert example.fault_count == 0, f"FAIL: {example.fault_count} simulation faults"
    q = example._pose
    h = float(q[2]) - example.terrain.height_at(float(q[0]), float(q[1]))
    assert h > 0.0, f"FAIL: chassis {h:.2f} m below the terrain"
    dist = math.hypot(float(q[0]) - example.spawn[0], float(q[1]) - example.spawn[1])
    assert dist > 1.0, f"FAIL: the vehicle drove only {dist:.1f} m from the spawn"
    if example._mode == "track":
        assert abs(example.lateral) < example._half_width + 1.0, (
            f"FAIL: vehicle left the corridor: {example.lateral:+.1f} m off the track"
        )
        example._close_turn()
        lat = [d["lat_rms"] for d in example.turns]
        assert all(math.isfinite(v) for v in lat) and (not lat or max(lat) < example._half_width), (
            f"FAIL: lateral RMS per turn {[round(v, 1) for v in lat]} m exceeds the {example._half_width:g} m corridor half width"
        )
    print(
        f"[PASS] {example.terrain.name} w={example.terrain.w:.2f} {example._mode}: drove {dist:.1f} m, chassis {h:.2f} m above the terrain, "
        f"{len(example.turns)} turns, off-track {example.lateral:+.2f} m"
    )


def check_sand_final(example) -> None:
    check_vehicle_final(example)
    sand_q = example.sand_state.particle_q.numpy()[: example._n_sand]
    assert np.all(np.isfinite(sand_q)), "non-finite sand particle positions"
    xpos_all = example.solver.xpos.numpy()
    smj = example._spindle_mj_arr.numpy()
    for e in range(vehicle_config.N_TIRES):
        drop = example.spec.tire_R_outer + example._sand_h - float(xpos_all[0, int(smj[e])][2])
        assert 0.005 < drop < 0.3, f"FAIL tire {e}: hub drop {drop * 1e3:.1f} mm not in (5, 300) mm"
    print(f"[PASS] sand: {example._n_sand:,} particles finite, all 4 tires supported by the bed")


def check_telemetry_final(example) -> None:
    check_terrain_final(example)
    assert np.all(np.isfinite(example._ghost_points.numpy())), "FAIL: non-finite ghost points"
    rms = math.sqrt(example._err_sq / example._err_n) if example._err_n else math.nan
    print(
        f"[PASS] telemetry {example.telemetry.sequence}: RMS position error vs the recording {rms:.2f} m ({example._err_n} frames)"
    )
