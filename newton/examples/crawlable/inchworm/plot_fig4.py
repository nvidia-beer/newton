#!/usr/bin/env python3
"""
Plot Fig. 4 style (paper arXiv:1911.05227): simulation solutions for the three-link robot.

(a) Angles φ1 (blue) and φ2 (purple): dashed = paper’s prescribed angles, solid = actual simulated angles.
(b) Position of left x1 (blue) and right x2 (purple) contacts; gray = robot snapshots.
Convention (simulation/CSV): x1/φ1 = left (bottom_y_minus), x2/φ2 = right (bottom_y_plus). Plot: left=blue, right=purple.
Output: <base>_a.png (angles), <base>_b.png (contacts), and <base>_combined.png ((a)+(b) with caption), where base is -o without extension.

All data is from simulation CSV. Pass --csv (main run) and optionally --csv_low (lower torque_stiffness) for dotted curves.
(b) is scaled to paper range -50..50 mm by default (--no_paper_scale to disable).

Usage:
  python3 plot_fig4.py --csv run1.csv --csv_low run2.csv -o fig4.png
  python3 plot_fig4.py --csv run1.csv --params inchworm_params.json --save_csv derived.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# #region agent log
def _debug_log(message: str, data: dict, hypothesis_id: str = "") -> None:
    payload = {"message": message, "data": data, "timestamp": time.time()}
    if hypothesis_id:
        payload["hypothesisId"] = hypothesis_id
    line = json.dumps(payload) + "\n"
    for p in [
        Path(__file__).resolve().parent / "debug_fig4a.log",
        Path("/home/beer/Dev/Robotics/IsaacLab/Newton/isaac-lab/.cursor/debug.log"),
    ]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
# #endregion


def _parse_arr(v: str) -> list[float]:
    if not v or not str(v).strip():
        return []
    return [float(x) for x in str(v).split() if x.strip()]


def _arr_mean(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0


def load_csv(path: str) -> list[dict]:
    """Load inchworm CSV; parse array columns as space-separated floats."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = {}
            for k, v in row.items():
                k = k.strip()
                if k == "frame":
                    try:
                        out[k] = int(float(v))
                    except (ValueError, TypeError):
                        out[k] = 0
                elif k == "t":
                    try:
                        out[k] = float(v)
                    except (ValueError, TypeError):
                        out[k] = 0.0
                elif k in (
                    "t_norm", "phi1_deg", "phi2_deg", "x1_mm", "x2_mm",
                    "fn_left_raw", "fn_right_raw", "ft",
                ):
                    try:
                        out[k] = float(v) if (v is not None and str(v).strip() != "") else None
                    except (ValueError, TypeError):
                        out[k] = None
                elif k in (
                    "y_left_ground", "z_left_ground", "y_right_ground", "z_right_ground",
                    "y_link_left", "z_link_left", "y_link_right", "z_link_right",
                ):
                    out[k] = _parse_arr(v) if isinstance(v, str) else (v if isinstance(v, list) else [])
                else:
                    try:
                        out[k] = float(v)
                    except (ValueError, TypeError):
                        out[k] = v
            rows.append(out)
    # #region agent log
    _debug_log("load_csv exit", {"row_count": len(rows), "path": path}, "H_csv_load")
    # #endregion
    return rows


def four_points_from_row(row: dict) -> tuple[list[float], list[float]]:
    """Return (ys, zs) for [left_ground, link_left, link_right, right_ground]. Same logic as render_inchworm_gif."""
    def pt(y_arr, z_arr):
        if not y_arr or not z_arr:
            return 0.0, 0.0
        return _arr_mean(y_arr), min(z_arr) if z_arr else 0.0
    if isinstance(row.get("y_left_ground"), list):
        y1, z1 = pt(row["y_left_ground"], row["z_left_ground"])
        y2, z2 = pt(row["y_link_left"], row["z_link_left"])
        y3, z3 = pt(row["y_link_right"], row["z_link_right"])
        y4, z4 = pt(row["y_right_ground"], row["z_right_ground"])
        return ([y1, y2, y3, y4], [z1, z2, z3, z4])
    y1 = float(row.get("y_left_ground", -0.015))
    z1 = float(row.get("z_left_ground", 0))
    y2 = float(row.get("y_link_left", -0.005))
    z2 = float(row.get("z_link_left", 0))
    y3 = float(row.get("y_link_right", 0.005))
    z3 = float(row.get("z_link_right", 0))
    y4 = float(row.get("y_right_ground", 0.015))
    z4 = float(row.get("z_right_ground", 0))
    return ([y1, y2, y3, y4], [z1, z2, z3, z4])


def joint_angles_from_four_points(ys: list[float], zs: list[float]) -> tuple[float, float]:
    """Compute φ1, φ2 (rad) from 4 points [left_ground, link_left, link_right, right_ground]. Interior angle π when flat."""
    y1, y2, y3, y4 = ys[0], ys[1], ys[2], ys[3]
    z1, z2, z3, z4 = zs[0], zs[1], zs[2], zs[3]
    # Left joint: v1 = link_left - left_ground, v2 = link_right - link_left
    v1x, v1y = y2 - y1, z2 - z1
    v2x, v2y = y3 - y2, z3 - z2
    n1 = (v1x * v1x + v1y * v1y) ** 0.5
    n2 = (v2x * v2x + v2y * v2y) ** 0.5
    if n1 < 1e-12 or n2 < 1e-12:
        phi1 = math.pi
    else:
        dot1 = (v1x * v2x + v1y * v2y) / (n1 * n2)
        dot1 = max(-1.0, min(1.0, dot1))
        phi1 = math.pi - math.acos(dot1)
    # Right joint
    w1x, w1y = y3 - y2, z3 - z2
    w2x, w2y = y4 - y3, z4 - z3
    nw1 = (w1x * w1x + w1y * w1y) ** 0.5
    nw2 = (w2x * w2x + w2y * w2y) ** 0.5
    if nw1 < 1e-12 or nw2 < 1e-12:
        phi2 = math.pi
    else:
        dot2 = (w1x * w2x + w1y * w2y) / (nw1 * nw2)
        dot2 = max(-1.0, min(1.0, dot2))
        phi2 = math.pi - math.acos(dot2)
    return phi1, phi2


def paper_theta(phi1: float, phi2: float, beta: float) -> float:
    """Central link angle: tan θ = (sin φ1 - sin φ2) / (cos φ1 + cos φ2 - β)."""
    num = math.sin(phi1) - math.sin(phi2)
    den = math.cos(phi1) + math.cos(phi2) - beta
    return math.atan2(num, den)


def paper_d(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Contact distance d (horizontal between feet)."""
    return l * (
        beta * math.cos(theta)
        - math.cos(phi1 - theta)
        - math.cos(phi2 + theta)
    )


def paper_xc(phi1: float, phi2: float, theta: float, l: float, beta: float) -> float:
    """Horizontal distance of CoM from left contact (paper)."""
    return l / (2.0 * (2.0 + beta)) * (
        (2.0 + beta) * beta * math.cos(theta)
        - (3.0 + 2.0 * beta) * math.cos(phi1 - theta)
        - math.cos(phi2 + theta)
    )


def series_from_csv(csv_path: str, period: float) -> tuple[list[float], list[float], list[float], list[float], list[float], list[list[float]], list[list[float]]]:
    """From CSV rows get t_norm, phi1_deg, phi2_deg, x1_mm, x2_mm, snap_ys, snap_zs."""
    rows = load_csv(csv_path)
    t_norm, phi1_deg, phi2_deg, x1_mm, x2_mm = [], [], [], [], []
    snap_ys, snap_zs = [], []
    for r in rows:
        t = r.get("t", 0.0)
        ys, zs = four_points_from_row(r)
        # Prefer CSV columns when present (from simulation log)
        tn = r.get("t_norm")
        if tn is not None:
            t_norm.append(float(tn))
        else:
            t_norm.append(t / period if period > 0 else 0.0)
        p1 = r.get("phi1_deg")
        p2 = r.get("phi2_deg")
        if p1 is not None and p2 is not None:
            phi1_deg.append(float(p1))
            phi2_deg.append(float(p2))
        else:
            phi1, phi2 = joint_angles_from_four_points(ys, zs)
            phi1_deg.append(math.degrees(phi1))
            phi2_deg.append(math.degrees(phi2))
        x1_val = r.get("x1_mm")
        x2_val = r.get("x2_mm")
        if x1_val is not None and x2_val is not None:
            x1_mm.append(float(x1_val))
            x2_mm.append(float(x2_val))
        else:
            x1_m = _arr_mean(r.get("y_left_ground", [ys[0]])) if isinstance(r.get("y_left_ground"), list) else ys[0]
            x2_m = _arr_mean(r.get("y_right_ground", [ys[3]])) if isinstance(r.get("y_right_ground"), list) else ys[3]
            x1_mm.append(x1_m * 1000.0)
            x2_mm.append(x2_m * 1000.0)
        snap_ys.append(ys)
        snap_zs.append(zs)
    # #region agent log
    first_tn = t_norm[0] if t_norm else None
    first_p1 = phi1_deg[0] if phi1_deg else None
    first_p2 = phi2_deg[0] if phi2_deg else None
    _debug_log(
        "series_from_csv exit",
        {"period": period, "n_rows": len(rows), "first_t_norm": first_tn, "first_phi1_deg": first_p1, "first_phi2_deg": first_p2, "phi1_min": min(phi1_deg) if phi1_deg else None, "phi1_max": max(phi1_deg) if phi1_deg else None},
        "H_series",
    )
    # #endregion
    return t_norm, phi1_deg, phi2_deg, x1_mm, x2_mm, snap_ys, snap_zs


def _interp_series_to_grid(
    t_norm: list[float],
    phi1_deg: list[float],
    phi2_deg: list[float],
    x1_mm: list[float],
    x2_mm: list[float],
    t_fine: list[float],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Interpolate one-cycle CSV series onto a fine t_norm grid so plotted curves look smooth (paper-style)."""
    import numpy as np
    t_arr = np.asarray(t_norm)
    # Restrict to first cycle for Fig. 4
    mask = (t_arr >= 0) & (t_arr <= 1.0)
    if mask.sum() < 2:
        return t_norm, phi1_deg, phi2_deg, x1_mm, x2_mm
    t_1 = t_arr[mask].tolist()
    p1 = [phi1_deg[i] for i in range(len(t_norm)) if mask[i]]
    p2 = [phi2_deg[i] for i in range(len(t_norm)) if mask[i]]
    x1 = [x1_mm[i] for i in range(len(t_norm)) if mask[i]]
    x2 = [x2_mm[i] for i in range(len(t_norm)) if mask[i]]
    t_f = np.asarray(t_fine)
    return (
        t_fine,
        np.interp(t_f, t_1, p1).tolist(),
        np.interp(t_f, t_1, p2).tolist(),
        np.interp(t_f, t_1, x1).tolist(),
        np.interp(t_f, t_1, x2).tolist(),
    )


def find_period_from_csv(rows: list[dict], gait_freq: float) -> float:
    """Period T = 1/gait_freq (one cycle in seconds)."""
    if gait_freq > 0:
        return 1.0 / gait_freq
    if len(rows) >= 2:
        ts = [r.get("t", 0) for r in rows]
        return max(ts) - min(ts) if max(ts) > min(ts) else 1.0
    return 1.0


def plot_fig4(
    csv: str | None,
    csv_low: str | None,
    params_path: str | None,
    out_path: str,
    save_csv_path: str | None,
    dpi: int = 150,
    paper_scale_b: bool = True,
    center_b: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Gait params (paper: gamma=π/2, A=π/6 → angles 60–120°)
    gamma = math.pi / 2.0
    A = math.pi / 6.0  # stick_slip_amplitude (rad), paper Fig.4 range
    gait_freq = 0.15
    omega = 2.0 * math.pi * gait_freq
    psi = 1.57  # gait_phase
    L = 1.5   # width (m)
    beta = 2.0
    if params_path and os.path.isfile(params_path):
        with open(params_path, encoding="utf-8") as f:
            p = json.load(f)
        gamma = math.pi / 2.0
        A = float(p.get("stick_slip_amplitude", A))
        gait_freq = float(p.get("gait_freq", gait_freq))
        omega = 2.0 * math.pi * gait_freq
        psi = float(p.get("gait_phase", psi))
        L = float(p.get("width", L))
        beta = float(p.get("paper_beta", beta))

    period = 1.0 / gait_freq

    # From CSVs: use the last full cycle. t_norm in CSV is gait_time/period (can be >1). All data is simulation.
    cycle_index = 0
    data_sim = None
    if csv and os.path.isfile(csv):
        raw = series_from_csv(csv, period)
        t_raw, snap_ys, snap_zs = raw[0], raw[5], raw[6]
        if t_raw:
            cycle_index = max(0, int(max(t_raw)) - 1)
        t_lo, t_hi = float(cycle_index), float(cycle_index + 1)
        mask_1c = [t_lo <= t < t_hi for t in t_raw]
        n_1c = sum(mask_1c)
        # #region agent log
        _debug_log("cycle filter", {"csv_path": csv, "period": period, "cycle_index": cycle_index, "n_total": len(t_raw), "n_cycle": n_1c, "t_raw_sample": t_raw[:3] if len(t_raw) >= 3 else t_raw}, "H_filter")
        # #endregion
        if n_1c >= 2:
            tr_raw = [t_raw[i] for i in range(len(t_raw)) if mask_1c[i]]
            # Normalize to [0, 1] for this cycle so plot y-axis is always 0..1
            tr = [t - t_lo for t in tr_raw]
            phi1r = [raw[1][i] for i in range(len(t_raw)) if mask_1c[i]]
            phi2r = [raw[2][i] for i in range(len(t_raw)) if mask_1c[i]]
            x1r = [raw[3][i] for i in range(len(t_raw)) if mask_1c[i]]
            x2r = [raw[4][i] for i in range(len(t_raw)) if mask_1c[i]]
            t_snap = tr
            snap_ys_1c = [snap_ys[i] for i in range(len(t_raw)) if mask_1c[i]]
            snap_zs_1c = [snap_zs[i] for i in range(len(t_raw)) if mask_1c[i]]
            data_sim = (tr, phi1r, phi2r, x1r, x2r, t_snap, snap_ys_1c, snap_zs_1c)
            # #region agent log
            _debug_log("data_sim filled", {"len_tr": len(tr), "tr_min": min(tr) if tr else None, "tr_max": max(tr) if tr else None, "phi1r_min": min(phi1r) if phi1r else None, "phi1r_max": max(phi1r) if phi1r else None, "phi2r_min": min(phi2r) if phi2r else None, "phi2r_max": max(phi2r) if phi2r else None}, "H_units")
            # #endregion
        else:
            data_sim = ([], [], [], [], [], [], [], [])
    data_low = None
    if csv_low and os.path.isfile(csv_low):
        raw = series_from_csv(csv_low, period)
        t_raw, snap_ys, snap_zs = raw[0], raw[5], raw[6]
        if not (csv and os.path.isfile(csv)) and t_raw:
            cycle_index = max(0, int(max(t_raw)) - 1)
        t_lo, t_hi = float(cycle_index), float(cycle_index + 1)
        mask_1c = [t_lo <= t < t_hi for t in t_raw]
        if sum(mask_1c) >= 2:
            tl_raw = [t_raw[i] for i in range(len(t_raw)) if mask_1c[i]]
            tl = [t - t_lo for t in tl_raw]
            phi1l = [raw[1][i] for i in range(len(t_raw)) if mask_1c[i]]
            phi2l = [raw[2][i] for i in range(len(t_raw)) if mask_1c[i]]
            x1l = [raw[3][i] for i in range(len(t_raw)) if mask_1c[i]]
            x2l = [raw[4][i] for i in range(len(t_raw)) if mask_1c[i]]
            t_snap = tl
            snap_ys_1c = [snap_ys[i] for i in range(len(t_raw)) if mask_1c[i]]
            snap_zs_1c = [snap_zs[i] for i in range(len(t_raw)) if mask_1c[i]]
            data_low = (tl, phi1l, phi2l, x1l, x2l, t_snap, snap_ys_1c, snap_zs_1c)
        else:
            data_low = ([], [], [], [], [], [], [], [])
    if (data_sim is not None or data_low is not None) and data_low is None:
        print("Tip: pass --csv_low <low_stiffness.csv> for dotted (low stiffness) curves. Run the example with lower torque_stiffness to get that CSV.", file=sys.stderr)

    # (a) Angles: φ1 = left (blue), φ2 = right (purple). Simulation only.
    fig_a, ax_a = plt.subplots(figsize=(5, 5))
    ax_a.set_xlabel(r"$t/T$ (normalized time, one cycle)")
    ax_a.set_ylabel(r"$\varphi_i(t)$ [deg]")
    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_xticklabels(["0", r"1/4", r"1/2", r"3/4", "1"])
    y_vals = []
    if data_sim is not None and len(data_sim[0]) >= 2:
        tr, phi1r, phi2r = data_sim[0], data_sim[1], data_sim[2]
        ax_a.plot(tr, phi1r, "-", color="C0", linewidth=2.5, label=r"$\phi_1$")
        ax_a.plot(tr, phi2r, "-", color="purple", linewidth=2.5, label=r"$\phi_2$")
        y_vals += phi1r + phi2r
    if data_low is not None and len(data_low[0]) >= 2:
        tl, phi1l, phi2l = data_low[0], data_low[1], data_low[2]
        ax_a.plot(tl, phi1l, ":", color="C0", linewidth=1.5, label=r"$\phi_1$ (low stiff.)")
        ax_a.plot(tl, phi2l, ":", color="purple", linewidth=1.5, label=r"$\phi_2$ (low stiff.)")
        y_vals += phi1l + phi2l
    if y_vals:
        y_lo = min(y_vals) - 5.0
        y_hi = max(y_vals) + 5.0
        ax_a.set_ylim(y_lo, y_hi)
    else:
        ax_a.set_ylim(60, 120)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="best", fontsize=8)

    # (b) Contact positions: x1 = left (blue), x2 = right (purple). Gray snapshots. Paper uses -50 to 50 mm.
    fig_b, ax_b = plt.subplots(figsize=(5, 5))
    # Scale to [-50, 50] mm. If center_b: subtract cycle-mean so (b) shows relative oscillation (no net drift). If not: raw positions so crawl direction is visible.
    x_center_mm = None
    if paper_scale_b and data_sim is not None and len(data_sim[0]) >= 2:
        x1r, x2r = data_sim[3], data_sim[4]
        if center_b:
            x_center_mm = sum((a + b) / 2.0 for a, b in zip(x1r, x2r)) / len(x1r)
            sim_x_c = [x - x_center_mm for x in x1r] + [x - x_center_mm for x in x2r]
            max_abs = max(abs(x) for x in sim_x_c) if sim_x_c else 1.0
        else:
            sim_x_c = x1r + x2r
            max_abs = max(abs(x) for x in sim_x_c) if sim_x_c else 1.0
        scale_b = 50.0 / max_abs if max_abs > 1e-6 else 1.0
    elif paper_scale_b:
        sim_x = []
        if data_sim is not None:
            sim_x += data_sim[3] + data_sim[4]
        if data_low is not None:
            sim_x += data_low[3] + data_low[4]
        if sim_x:
            max_abs = max(abs(x) for x in sim_x)
            scale_b = 50.0 / max_abs if max_abs > 1e-6 else 1.0
        else:
            scale_b = 1.0
    else:
        scale_b = 1.0

    ax_b.set_xlabel(r"$x_1(t)$, $x_2(t)$ [mm]")
    ax_b.set_ylabel(r"$t/T$ (normalized time, one cycle)")
    ax_b.set_ylim(0, 1)
    # Y-axis ticks at 0, 1/4, 1/2, 3/4, 1
    ax_b.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_yticklabels(["0", r"1/4", r"1/2", r"3/4", "1"])
    ax_b.grid(True, alpha=0.3)
    # Paper Fig. 4b: Position of left x1 (blue) and right x2 (purple) contacts; gray snapshots at 0, 1/4, 1/2, 3/4 (not at 1).
    SNAPSHOT_T = (0.0, 0.25, 0.5, 0.75)  # gray robot silhouettes; omit t/T=1
    snapshot_height = 0.2  # normalized height of gray silhouettes in t/T so base is visible
    use_csv_outline = (
        data_sim is not None
        and len(data_sim) >= 8
        and len(data_sim[6]) > 0
        and len(data_sim[7]) > 0
    )
    if use_csv_outline:
        t_snap = data_sim[5]
        snap_ys_1c = data_sim[6]
        snap_zs_1c = data_sim[7]
        x1r, x2r = data_sim[3], data_sim[4]
        x_off = x_center_mm if x_center_mm is not None else 0.0
        for T_val in SNAPSHOT_T:
            idx = min(range(len(t_snap)), key=lambda i: abs(t_snap[i] - T_val))
            ys, zs = snap_ys_1c[idx], snap_zs_1c[idx]
            T_snap = t_snap[idx]
            x1_plt = (ys[0] * 1000.0 - x_off) * scale_b
            x2_plt = (ys[1] * 1000.0 - x_off) * scale_b
            x3_plt = (ys[2] * 1000.0 - x_off) * scale_b
            x4_plt = (ys[3] * 1000.0 - x_off) * scale_b
            # Use actual z (height) so leg lengths and middle angle are correct. Map z into [T_snap, T_snap+snapshot_height].
            z_ref = min(zs)
            z_span = max(zs) - z_ref
            if z_span < 1e-9:
                z_span = 1e-9
            V = [T_snap + snapshot_height * (z - z_ref) / z_span for z in zs]
            # Three line segments (skeleton): left leg, middle, right leg. Middle slopes if z2 != z3.
            ax_b.plot([x1_plt, x2_plt], [V[0], V[1]], color="0.35", linewidth=1.2, zorder=0)
            ax_b.plot([x2_plt, x3_plt], [V[1], V[2]], color="0.5", linewidth=1.2, zorder=0)
            ax_b.plot([x3_plt, x4_plt], [V[2], V[3]], color="0.65", linewidth=1.2, zorder=0)
    # Contact curves: simulation only. Low stiffness (dotted).
    if data_sim is not None and len(data_sim[0]) >= 2:
        x_off = x_center_mm if x_center_mm is not None else 0.0
        x1r_b = [(x - x_off) * scale_b for x in data_sim[3]]
        x2r_b = [(x - x_off) * scale_b for x in data_sim[4]]
        ax_b.plot(x1r_b, data_sim[0], "-", color="C0", linewidth=2.5, label=r"$x_1$", zorder=3)  # left = blue
        ax_b.plot(x2r_b, data_sim[0], "-", color="purple", linewidth=2.5, label=r"$x_2$", zorder=3)  # right = purple
    if data_low is not None and len(data_low[0]) >= 2:
        x1l, x2l = data_low[3], data_low[4]
        if x_center_mm is not None:
            x1l_b = [(x - x_center_mm) * scale_b for x in x1l]
            x2l_b = [(x - x_center_mm) * scale_b for x in x2l]
        else:
            x1l_b = [x * scale_b for x in x1l]
            x2l_b = [x * scale_b for x in x2l]
        ax_b.plot(x1l_b, data_low[0], ":", color="C0", linewidth=1.5, zorder=2)
        ax_b.plot(x2l_b, data_low[0], ":", color="purple", linewidth=1.5, zorder=2)

    # Axis range (b): original paper Fig. 4(b) range — x in [-50, 50] mm, T in [0, 1]
    ax_b.set_xlim(-55, 55)
    ax_b.legend(loc="best", fontsize=8)

    # Save two separate figures: (a) angles, (b) contact positions + snapshots
    base = os.path.splitext(out_path)[0]
    out_a = base + "_a.png"
    out_b = base + "_b.png"
    fig_a.tight_layout()
    fig_b.tight_layout()
    fig_a.savefig(out_a, dpi=dpi)
    fig_b.savefig(out_b, dpi=dpi)
    plt.close(fig_a)
    plt.close(fig_b)
    print(f"Saved: {os.path.abspath(out_a)}")
    print(f"Saved: {os.path.abspath(out_b)}")

    # Combined figure (a) and (b) with caption
    out_combined = base + "_combined.png"
    img_a = plt.imread(out_a)
    img_b = plt.imread(out_b)
    fig_comb, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_a)
    axes[0].axis("off")
    axes[1].imshow(img_b)
    axes[1].axis("off")
    # Reserve bottom for panel labels (a)/(b) and caption: layout so caption is below (a)/(b)
    fig_comb.tight_layout(rect=[0, 0.22, 1, 1])
    # Panel labels (a) and (b) below the two panels, above the caption
    fig_comb.text(0.25, 0.19, "(a)", ha="center", va="top", fontsize=11, fontfamily="serif", fontweight="bold")
    fig_comb.text(0.75, 0.19, "(b)", ha="center", va="top", fontsize=11, fontfamily="serif", fontweight="bold")
    # Caption: line 1 (black); line 2 with blue/purple/gray for curve labels (no mathtext \color)
    caption_line1 = (
        "Simulation solutions for the three-link robot's configuration – angles and contact positions from simulation. "
        "Dotted: low stiffness."
    )
    fig_comb.text(0.5, 0.08, textwrap.fill(caption_line1, width=100), ha="center", va="bottom", fontsize=9, fontfamily="serif")
    # Line 2: segments with colors; position by measuring (mathtext \color not supported)
    caption_segments = [
        ("(a) Angles ", "black"),
        ("ϕ₁ ", "black"),
        ("(blue)", "C0"),
        (" and ", "black"),
        ("ϕ₂ ", "black"),
        ("(purple)", "purple"),
        (". (b) Position of left ", "black"),
        ("x₁ ", "black"),
        ("(blue)", "C0"),
        (" and right ", "black"),
        ("x₂ ", "black"),
        ("(purple)", "purple"),
        (" contacts and snapshots of the robot ", "black"),
        ("(gray).", "gray"),
    ]
    x_cap = 0.05
    try:
        fig_comb.draw_without_rendering()
    except Exception:
        pass
    try:
        r = fig_comb.canvas.get_renderer()
    except Exception:
        try:
            r = fig_comb.get_renderer()
        except Exception:
            r = None
    for seg_text, seg_color in caption_segments:
        t = fig_comb.text(
            x_cap, 0.02, seg_text, ha="left", va="bottom", fontsize=9, color=seg_color, fontfamily="serif"
        )
        if r is not None:
            try:
                bbox = t.get_window_extent(r)
                bbox_fig = bbox.transformed(fig_comb.transFigure.inverted())
                x_cap = bbox_fig.x1
            except Exception:
                x_cap += 0.012 * len(seg_text)  # fallback: rough char width
        else:
            x_cap += 0.012 * len(seg_text)
    fig_comb.savefig(out_combined, dpi=dpi)
    plt.close(fig_comb)
    print(f"Saved: {os.path.abspath(out_combined)}")

    if save_csv_path:
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["condition", "t_norm", "phi1_deg", "phi2_deg", "x1_mm", "x2_mm"])
            if data_sim is not None:
                tr, phi1r, phi2r, x1r, x2r = data_sim[0], data_sim[1], data_sim[2], data_sim[3], data_sim[4]
                for i in range(len(tr)):
                    w.writerow(["simulation", tr[i], phi1r[i], phi2r[i], x1r[i], x2r[i]])
            if data_low is not None:
                tl, phi1l, phi2l, x1l, x2l = data_low[0], data_low[1], data_low[2], data_low[3], data_low[4]
                for i in range(len(tl)):
                    w.writerow(["low_stiffness", tl[i], phi1l[i], phi2l[i], x1l[i], x2l[i]])
        print(f"Saved derived CSV: {os.path.abspath(save_csv_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Fig. 4 style: angles and contact positions vs t/T.")
    parser.add_argument("--csv", type=str, default=None, help="Simulation CSV (main run)")
    parser.add_argument("--csv_low", type=str, default=None, help="Simulation CSV with low torque_stiffness (dotted curves)")
    parser.add_argument("--params", type=str, default=None, help="Params JSON (gait_freq, stick_slip_amplitude, gait_phase, width)")
    parser.add_argument("-o", "--output", type=str, default="fig4_simulation.png", help="Output base path: saves <base>_a.png, <base>_b.png, and <base>_combined.png")
    parser.add_argument("--save_csv", type=str, default=None, help="Save derived t_norm, phi1, phi2, x1, x2 to CSV")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no_paper_scale", action="store_true", help="Disable paper scale for (b): use data range instead of -50..50 mm")
    parser.add_argument("--no_center", action="store_true", help="(b) only: do not center x1/x2 by cycle mean; plot world positions so net crawl direction is visible.")
    args = parser.parse_args()

    try:
        plot_fig4(
            csv=args.csv,
            csv_low=args.csv_low,
            params_path=args.params,
            out_path=args.output,
            save_csv_path=args.save_csv,
            dpi=args.dpi,
            paper_scale_b=not args.no_paper_scale,
            center_b=not args.no_center,
        )
    except ImportError as e:
        if "matplotlib" in str(e).lower():
            print("matplotlib required: pip install matplotlib", file=sys.stderr)
        else:
            print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
