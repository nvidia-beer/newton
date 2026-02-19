#!/usr/bin/env python3
"""
Analyze an inchworm run CSV to compute paper-aligned metrics.

Usage (from repo root, in container):
  python .../crawlable/inchworm/analyze_run_csv.py .../crawlable/inchworm/inchworm_2026-02-18_16-12-45.csv

Output: summary of displacement, time, CoM height, and derived metrics for tuning.
"""

import csv
import sys
from pathlib import Path


def _parse_list(s: str) -> list[float]:
    """Parse space-separated floats; return empty list on failure."""
    out = []
    for x in s.strip().split():
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def analyze(csv_path: str) -> dict:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        return {"error": "No rows"}

    t_vals = []
    y_left_ground = []
    z_left_ground = []
    y_right_ground = []
    z_right_ground = []
    y_link_left = []
    z_link_left = []
    y_link_right = []
    z_link_right = []

    for r in rows:
        t_vals.append(float(r["t"]))
        yl = _parse_list(r["y_left_ground"])
        zl = _parse_list(r["z_left_ground"])
        yr = _parse_list(r["y_right_ground"])
        zr = _parse_list(r["z_right_ground"])
        y_left_ground.append(_mean(yl) if yl else 0.0)
        z_left_ground.append(_mean(zl) if zl else 0.0)
        y_right_ground.append(_mean(yr) if yr else 0.0)
        z_right_ground.append(_mean(zr) if zr else 0.0)
        yll = _parse_list(r["y_link_left"])
        zll = _parse_list(r["z_link_left"])
        ylr = _parse_list(r["y_link_right"])
        zlr = _parse_list(r["z_link_right"])
        y_link_left.append(_mean(yll) if yll else 0.0)
        z_link_left.append(_mean(zll) if zll else 0.0)
        y_link_right.append(_mean(ylr) if ylr else 0.0)
        z_link_right.append(_mean(zlr) if zlr else 0.0)

    t_start, t_end = t_vals[0], t_vals[-1]
    duration = t_end - t_start
    # Center of mass (approximate): midpoint of left and right ground contacts
    y_center_start = 0.5 * (y_left_ground[0] + y_right_ground[0])
    y_center_end = 0.5 * (y_left_ground[-1] + y_right_ground[-1])
    net_displacement_y = y_center_end - y_center_start
    avg_velocity = net_displacement_y / duration if duration > 0 else 0.0

    # CoM height: use mean of link z (or ground z) over time
    z_ground_left_min = min(z_left_ground)
    z_ground_left_max = max(z_left_ground)
    z_ground_right_min = min(z_right_ground)
    z_ground_right_max = max(z_right_ground)
    z_link_min = min(z_link_left + z_link_right)
    z_link_max = max(z_link_left + z_link_right)

    # Lift: differential between feet (one end lifted) and arch above lowest foot
    z_ground_diff_per_row = [
        abs(z_left_ground[i] - z_right_ground[i]) for i in range(len(z_left_ground))
    ]
    z_ground_min_per_row = [min(z_left_ground[i], z_right_ground[i]) for i in range(len(z_left_ground))]
    lift_differential_max = max(z_ground_diff_per_row) if z_ground_diff_per_row else 0.0
    arch_above_lowest = [
        (z_link_left[i] + z_link_right[i]) / 2.0 - z_ground_min_per_row[i]
        for i in range(len(z_link_left))
    ]
    arch_height_min = min(arch_above_lowest) if arch_above_lowest else 0.0
    arch_height_max = max(arch_above_lowest) if arch_above_lowest else 0.0

    # Gait: paper uses ~0.2 Hz → period 5 s
    gait_period = 5.0  # seconds (0.2 Hz)
    num_cycles = duration / gait_period if gait_period > 0 else 0
    distance_per_cycle = net_displacement_y / num_cycles if num_cycles > 0 else 0.0

    return {
        "num_frames": len(rows),
        "t_start": t_start,
        "t_end": t_end,
        "duration_s": duration,
        "y_center_start": y_center_start,
        "y_center_end": y_center_end,
        "net_displacement_m": net_displacement_y,
        "avg_velocity_m_s": avg_velocity,
        "z_ground_left_min": z_ground_left_min,
        "z_ground_left_max": z_ground_left_max,
        "z_ground_right_min": z_ground_right_min,
        "z_ground_right_max": z_ground_right_max,
        "z_link_min": z_link_min,
        "z_link_max": z_link_max,
        "lift_differential_max_m": lift_differential_max,
        "arch_height_min_m": arch_height_min,
        "arch_height_max_m": arch_height_max,
        "gait_period_s": gait_period,
        "num_cycles_approx": num_cycles,
        "distance_per_cycle_m": distance_per_cycle,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_run_csv.py <path_to_inchworm.csv>", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    m = analyze(csv_path)
    if "error" in m:
        print(m["error"], file=sys.stderr)
        sys.exit(1)
    print("Inchworm run metrics (paper-aligned)")
    print("=====================================")
    print(f"  Frames:        {m['num_frames']}")
    print(f"  Time:         {m['t_start']:.3f} s → {m['t_end']:.3f} s  (duration {m['duration_s']:.3f} s)")
    print(f"  Y center:     {m['y_center_start']:.4f} → {m['y_center_end']:.4f} m")
    print(f"  Net displacement: {m['net_displacement_m']:.4f} m")
    print(f"  Avg velocity:     {m['avg_velocity_m_s']:.6f} m/s")
    print(f"  Z ground left:    [{m['z_ground_left_min']:.4f}, {m['z_ground_left_max']:.4f}] m")
    print(f"  Z ground right:   [{m['z_ground_right_min']:.4f}, {m['z_ground_right_max']:.4f}] m")
    print(f"  Z link (min/max): {m['z_link_min']:.4f} / {m['z_link_max']:.4f} m")
    print(f"  Lift differential (max |left−right| foot height): {m['lift_differential_max_m']*1000:.2f} mm")
    print(f"  Arch above lowest foot: {m['arch_height_min_m']*1000:.2f}–{m['arch_height_max_m']*1000:.2f} mm")
    print(f"  Gait @ 0.2 Hz:    ~{m['num_cycles_approx']:.1f} cycles, ~{m['distance_per_cycle_m']*1000:.2f} mm/cycle")
    print()
    print("Use these in INCHWORM_TUNING.md to compare with paper and adjust simulation.")


if __name__ == "__main__":
    main()
