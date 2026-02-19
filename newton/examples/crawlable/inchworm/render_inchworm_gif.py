#!/usr/bin/env python3
"""
Standalone script to render inchworm GIF from CSV. No newton/warp dependency.
Only needs: pip install matplotlib (and standard library: csv, math, os, sys).

Usage:
  python3 render_inchworm_gif.py inchworm_2026-02-18_15-34-25.csv
  python3 render_inchworm_gif.py inchworm_2026-02-18_15-34-25.csv -o out.gif --fps 10
"""
from __future__ import annotations

import csv
import math
import os
import sys


def _val(row: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in row:
            v = row[k]
            if isinstance(v, list):
                return (sum(v) / len(v)) if v else default
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _parse_arr(v: str) -> list[float]:
    """Parse space-separated floats; return [] if empty or invalid."""
    if not v or not str(v).strip():
        return []
    out = []
    for x in str(v).split():
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            pass
    return out


def arrays_to_yz_point(y_arr: list[float], z_arr: list[float], use_min_z: bool = True) -> tuple[float, float]:
    """Reduce one array to a single (y,z) point: mean(y), min(z) or mean(z) for 2D rendering."""
    if not y_arr or not z_arr:
        return (0.0, 0.0)
    y_pt = sum(y_arr) / len(y_arr)
    z_pt = min(z_arr) if use_min_z else (sum(z_arr) / len(z_arr))
    return (y_pt, z_pt)


def four_arrays_to_four_points(row: dict) -> tuple[list[float], list[float]]:
    """
    Convert 4 arrays (left_ground, right_ground, link_left, link_right) to 4 (y,z) points in 2D.

    Order: [left_ground, link_left, link_right, right_ground].
    Each point from one array as (mean(y), min(z)). Returns (ys, zs) for the 4 points.
    """
    # New format: 4 arrays → one (y,z) per array
    if "y_left_ground" in row and isinstance(row.get("y_left_ground"), list):
        y1, z1 = arrays_to_yz_point(row["y_left_ground"], row["z_left_ground"])
        y2, z2 = arrays_to_yz_point(row["y_link_left"], row["z_link_left"])
        y3, z3 = arrays_to_yz_point(row["y_link_right"], row["z_link_right"])
        y4, z4 = arrays_to_yz_point(row["y_right_ground"], row["z_right_ground"])
        return ([y1, y2, y3, y4], [z1, z2, z3, z4])
    # Legacy: single-value columns
    y1 = _val(row, "y_left_ground", "y_contact_Ym", "y_Ym", default=-0.015)
    z1 = _val(row, "z_left_ground", "z_contact_Ym", "z_Ym", "z_blue_Ym", "max_z_blue_Ym")
    y2 = _val(row, "y_link_left", "y_phi1", default=-0.005)
    z2 = _val(row, "z_link_left", "z_phi1", "z_green_phi1", "mean_z_phi1")
    y3 = _val(row, "y_link_right", "y_phi2", default=0.005)
    z3 = _val(row, "z_link_right", "z_phi2", "z_green_phi2", "mean_z_phi2")
    y4 = _val(row, "y_right_ground", "y_contact_Yp", "y_Yp", default=0.015)
    z4 = _val(row, "z_right_ground", "z_contact_Yp", "z_Yp", "z_blue_Yp", "max_z_blue_Yp")
    return ([y1, y2, y3, y4], [z1, z2, z3, z4])


def draw_links_and_joints(
    line_links: "matplotlib.lines.Line2D",
    line_contacts: "matplotlib.lines.Line2D",
    line_joints: "matplotlib.lines.Line2D",
    ys: list[float],
    zs: list[float],
    time_text: "matplotlib.text.Text" | None = None,
    t: float | None = None,
) -> None:
    """
    Draw one frame: links (line through 4 points), contacts (points 0 and 3), joints (points 1 and 2).

    ys, zs: length-4 lists for [left_ground, link_left, link_right, right_ground].
    """
    line_links.set_data(ys, zs)
    line_contacts.set_data([ys[0], ys[3]], [zs[0], zs[3]])
    line_joints.set_data([ys[1], ys[2]], [zs[1], zs[2]])
    if time_text is not None and t is not None:
        time_text.set_text(f"t = {t:.2f} s")


ARRAY_COLUMNS = (
    "y_left_ground", "z_left_ground", "y_right_ground", "z_right_ground",
    "y_link_left", "z_link_left", "y_link_right", "z_link_right",
)


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
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
                elif k in ARRAY_COLUMNS:
                    out[k] = _parse_arr(v) if isinstance(v, str) else (v if isinstance(v, list) else [])
                else:
                    try:
                        out[k] = float(v)
                    except (ValueError, TypeError):
                        out[k] = v
            rows.append(out)
    return rows


def frames_to_animation_gif(
    rows: list[dict],
    out_path: str,
    fps: float = 10,
    dpi: int = 120,
    aspect: str = "paper",
) -> None:
    """
    Plot 4 points per frame (from four_arrays_to_four_points), draw links and joints, save animated GIF.

    rows: list of CSV row dicts (with 4 arrays or legacy columns).
    aspect: "equal" = 1:1 data scale (arch looks flat); "paper" = exaggerate Z so φ₁, φ₂ bend is visible (paper Fig. 3).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as mplanim
    except ImportError as e:
        raise ImportError("matplotlib is required: pip install matplotlib") from e

    if not rows:
        raise ValueError("No rows to animate")

    # Compute 4 points for every frame; set x-axis (Y) to full CSV range so inchworm walking is visible
    y_vals, z_vals = [], []
    for r in rows:
        ys, zs = four_arrays_to_four_points(r)
        y_vals.extend(ys)
        z_vals.extend(zs)
    if not y_vals:
        y_vals = [-0.02, 0.02]
    y_min_csv, y_max_csv = min(y_vals), max(y_vals)
    y_range = y_max_csv - y_min_csv or 0.04
    y_margin = max(0.005, 0.03 * y_range)
    y_lim = (y_min_csv - y_margin, y_max_csv + y_margin)
    z_min = min(z_vals) - 0.005
    z_max = max(z_vals) + 0.01
    z_lim = (min(0.0, z_min), z_max)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # Axes in simulation units (CSV not necessarily in meters). Y–Z slice: bending only; forward motion (X) not shown.
    ax.set_xlabel("Y (sim. units)")
    ax.set_ylabel("Z (sim. units)")
    ax.set_title("Inchworm Y–Z slice (paper Fig. 3)\nForward motion along X not shown")
    ax.set_xlim(y_lim)
    ax.set_ylim(z_lim)
    # Paper Fig. 3 uses a stretched Z scale so the arch and φ₁, φ₂ joints are clearly visible; "equal" flattens the bend.
    if aspect == "paper":
        y_span = y_lim[1] - y_lim[0]
        z_span = z_lim[1] - z_lim[0] or 0.01
        # Exaggerate Z so ~1 Z unit has same display length as ~(y_span/4) in Y (makes arch/angles visible).
        ax.set_aspect((y_span / 4.0) / z_span, adjustable="datalim")
    else:
        ax.set_aspect("equal")
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="-")
    ax.fill_between([y_lim[0], y_lim[1]], 0, z_lim[0], color="0.85", zorder=0)

    # Artists: links (line), contacts (end points), joints (middle points)
    line_links, = ax.plot([], [], "k-", linewidth=4, solid_capstyle="round", zorder=2)
    line_contacts, = ax.plot([], [], "o", color="C0", markersize=10, markeredgecolor="k", markeredgewidth=0.5, zorder=3)
    line_joints, = ax.plot([], [], "o", color="C2", markersize=10, markeredgecolor="k", markeredgewidth=0.5, zorder=3)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=9)
    # Paper parameters φ₁, φ₂ at joint positions (LaTeX)
    label_offset = 0.003
    phi1_text = ax.text(0, 0, r"$\phi_1$", fontsize=11, ha="center", va="bottom", zorder=4)
    phi2_text = ax.text(0, 0, r"$\phi_2$", fontsize=11, ha="center", va="bottom", zorder=4)

    def init():
        line_links.set_data([], [])
        line_contacts.set_data([], [])
        line_joints.set_data([], [])
        time_text.set_text("")
        phi1_text.set_position((0, 0))
        phi2_text.set_position((0, 0))
        return (line_links, line_contacts, line_joints, time_text, phi1_text, phi2_text)

    def animate(i: int):
        r = rows[i]
        ys, zs = four_arrays_to_four_points(r)
        draw_links_and_joints(line_links, line_contacts, line_joints, ys, zs, time_text, r.get("t", i))
        # Place φ₁ and φ₂ labels just above the joint points
        if len(ys) >= 4 and len(zs) >= 4:
            phi1_text.set_position((ys[1], zs[1] + label_offset))
            phi2_text.set_position((ys[2], zs[2] + label_offset))
        return (line_links, line_contacts, line_joints, time_text, phi1_text, phi2_text)

    anim = mplanim.FuncAnimation(
        fig, animate, init_func=init, frames=len(rows), interval=1000.0 / fps, blit=True
    )
    anim.save(out_path, writer=mplanim.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"GIF saved: {os.path.abspath(out_path)}")


def render_gif(
    csv_path: str,
    out_path: str,
    fps: float = 10,
    dpi: int = 120,
    aspect: str = "paper",
) -> None:
    """Load CSV, convert each row to 4 points, draw links and joints, save animated GIF."""
    rows = load_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    frames_to_animation_gif(rows, out_path, fps=fps, dpi=dpi, aspect=aspect)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Render inchworm CSV to GIF (no newton/warp).")
    p.add_argument("csv", help="Path to inchworm CSV")
    p.add_argument("-o", "--output", default=None, help="Output GIF path (default: <csv_base>_movement.gif)")
    p.add_argument("--fps", type=float, default=10, help="FPS (default: 10)")
    p.add_argument("--dpi", type=int, default=120, help="DPI (default: 120)")
    p.add_argument(
        "--aspect",
        choices=("paper", "equal"),
        default="paper",
        help="Y:Z aspect: 'paper' = exaggerate Z so φ₁,φ₂ bend visible (default); 'equal' = 1:1",
    )
    args = p.parse_args()
    if not os.path.isfile(args.csv):
        print(f"Not a file: {args.csv}", file=sys.stderr)
        return 1
    out = args.output or f"{os.path.splitext(args.csv)[0]}_movement.gif"
    try:
        render_gif(args.csv, out, fps=args.fps, dpi=args.dpi, aspect=args.aspect)
    except ImportError as e:
        if "matplotlib" in str(e).lower():
            print("Could not create animation: matplotlib is required.", file=sys.stderr)
            print("Install with:  pip install matplotlib", file=sys.stderr)
        else:
            print(f"Import error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Could not create animation: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
