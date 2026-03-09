# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Paper-related inchworm logic (arXiv:1911.05227): CSV format and 4 vertex arrays.

Saves frame, t, and the 4 arrays (left_ground, right_ground, link_left, link_right) as space-separated
y,z values per row. Post-process for floor contacts, 4-point model, etc.
"""

import csv
import os
from datetime import datetime
from typing import Any

import numpy as np


def _round5(v: float) -> float:
    """Round to 5 decimals for CSV."""
    return round(float(v), 5)


def _arr_to_str(arr: list[float]) -> str:
    """Space-separated values for one column."""
    return " ".join(str(_round5(x)) for x in arr)


# frame, t, t_norm (= gait_time/period, so 0..1 = 1st cycle, 1..2 = 2nd, etc.), then angles/contacts, contact forces, 4 arrays.
# Plot scripts (fig4, fig6) use last full cycle.
CSV_HEADER = [
    "frame", "t", "t_norm",
    "phi1_deg", "phi2_deg", "x1_mm", "x2_mm",
    "fn_left_raw", "fn_right_raw", "ft",
    "y_left_ground", "z_left_ground", "y_right_ground", "z_right_ground",
    "y_link_left", "z_link_left", "y_link_right", "z_link_right",
]


def _pt(y_arr: list[float], z_arr: list[float]) -> tuple[float, float]:
    if not y_arr or not z_arr:
        return 0.0, 0.0
    return float(np.mean(y_arr)), float(min(z_arr))


def angles_and_contacts_from_metrics(m: dict[str, Any]) -> tuple[float, float, float, float]:
    """From get_paper_metrics output compute phi1_deg, phi2_deg, x1_mm, x2_mm (same convention as plot)."""
    y1, z1 = _pt(m.get("y_left_ground", []), m.get("z_left_ground", []))
    y2, z2 = _pt(m.get("y_link_left", []), m.get("z_link_left", []))
    y3, z3 = _pt(m.get("y_link_right", []), m.get("z_link_right", []))
    y4, z4 = _pt(m.get("y_right_ground", []), m.get("z_right_ground", []))
    # Interior angles φ1, φ2 (rad) from 4 points
    v1x, v1y = y2 - y1, z2 - z1
    v2x, v2y = y3 - y2, z3 - z2
    n1 = (v1x * v1x + v1y * v1y) ** 0.5
    n2 = (v2x * v2x + v2y * v2y) ** 0.5
    phi1 = np.pi if (n1 < 1e-12 or n2 < 1e-12) else np.pi - np.arccos(np.clip((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0))
    w1x, w1y = y3 - y2, z3 - z2
    w2x, w2y = y4 - y3, z4 - z3
    nw1, nw2 = (w1x * w1x + w1y * w1y) ** 0.5, (w2x * w2x + w2y * w2y) ** 0.5
    phi2 = np.pi if (nw1 < 1e-12 or nw2 < 1e-12) else np.pi - np.arccos(np.clip((w1x * w2x + w1y * w2y) / (nw1 * nw2), -1.0, 1.0))
    x1_mm = y1 * 1000.0
    x2_mm = y4 * 1000.0
    return float(np.degrees(phi1)), float(np.degrees(phi2)), x1_mm, x2_mm


class InchwormValidation:
    """CSV logger: saves frame, t, and the 4 vertex arrays (2 blue, 2 green) per row."""

    @staticmethod
    def csv_log_path(csv_log_arg: str | None, csv_log_dir: str | None = None) -> str | None:
        """Resolve CSV path from CLI: explicit path, or inchworm_<time>.csv (optionally under csv_log_dir). Use --csv_log '' to disable."""
        if csv_log_arg is not None and csv_log_arg.strip():
            return csv_log_arg.strip()
        if csv_log_arg is not None:
            return None
        name = f"inchworm_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        if csv_log_dir and csv_log_dir.strip():
            return os.path.join(csv_log_dir.strip(), name)
        return name

    def __init__(self, path: str | None, log_interval: int = 120):
        self.path = path
        self.log_interval = log_interval
        self._file: Any = None
        self._writer: csv.writer | None = None
        if path and path.strip():
            self._file = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(CSV_HEADER)
            self._file.flush()
            print(f"   CSV log: {os.path.abspath(path)} (4 arrays: left_ground, right_ground, link_left, link_right; every {log_interval} frames)", flush=True)

    def log_row(
        self,
        frame: int,
        sim_time: float,
        y_left_ground: list[float],
        z_left_ground: list[float],
        y_right_ground: list[float],
        z_right_ground: list[float],
        y_link_left: list[float],
        z_link_left: list[float],
        y_link_right: list[float],
        z_link_right: list[float],
        t_norm: float | None = None,
        phi1_deg: float | None = None,
        phi2_deg: float | None = None,
        x1_mm: float | None = None,
        x2_mm: float | None = None,
        fn_left_raw: float | None = None,
        fn_right_raw: float | None = None,
        ft: float | None = None,
    ) -> None:
        """Write one row: frame, t, t_norm, angles/contacts, fn_left_raw/fn_right_raw/ft, and the 4 arrays."""
        if self._writer is None:
            return
        row = [
            frame,
            round(sim_time, 4),
            _round5(t_norm) if t_norm is not None else "",
            _round5(phi1_deg) if phi1_deg is not None else "",
            _round5(phi2_deg) if phi2_deg is not None else "",
            _round5(x1_mm) if x1_mm is not None else "",
            _round5(x2_mm) if x2_mm is not None else "",
            _round5(fn_left_raw) if fn_left_raw is not None else "",
            _round5(fn_right_raw) if fn_right_raw is not None else "",
            _round5(ft) if ft is not None else "",
            _arr_to_str(y_left_ground), _arr_to_str(z_left_ground),
            _arr_to_str(y_right_ground), _arr_to_str(z_right_ground),
            _arr_to_str(y_link_left), _arr_to_str(z_link_left),
            _arr_to_str(y_link_right), _arr_to_str(z_link_right),
        ]
        self._writer.writerow(row)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            if self.path:
                print(f"   CSV log saved: {os.path.abspath(self.path)}", flush=True)
            self._file = None
            self._writer = None

    def __enter__(self) -> "InchwormValidation":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def is_logging(self) -> bool:
        return self._writer is not None


def get_paper_metrics(
    particle_q: Any,
    bottom_y_plus_indices: list[int],
    bottom_y_minus_indices: list[int],
    joint_left_indices: list[int],
    joint_right_indices: list[int],
) -> dict[str, Any]:
    """
    Return the 4 vertex arrays (left_ground, right_ground, link_left, link_right) from particle positions.

    particle_q: (n, 3) positions (numpy or object with .numpy()).
    Returns dict with: y_left_ground, z_left_ground, y_right_ground, z_right_ground,
    y_link_left, z_link_left, y_link_right, z_link_right (lists), and contact_ok.
    """
    if hasattr(particle_q, "numpy"):
        q = np.array(particle_q.numpy(), dtype=np.float64)
    else:
        q = np.asarray(particle_q, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(-1, 3)
    y, z = q[:, 1], q[:, 2]
    idx_ym = bottom_y_minus_indices or []   # left_ground
    idx_yp = bottom_y_plus_indices or []    # right_ground
    idx_phi1 = joint_left_indices or []     # link_left
    idx_phi2 = joint_right_indices or []    # link_right

    y_left_ground = [float(y[i]) for i in idx_ym]
    z_left_ground = [float(z[i]) for i in idx_ym]
    y_right_ground = [float(y[i]) for i in idx_yp]
    z_right_ground = [float(z[i]) for i in idx_yp]
    y_link_left = [float(y[i]) for i in idx_phi1]
    z_link_left = [float(z[i]) for i in idx_phi1]
    y_link_right = [float(y[i]) for i in idx_phi2]
    z_link_right = [float(z[i]) for i in idx_phi2]

    # contact_ok: both ground patches have at least one vert at z <= 0; threshold in post-process
    th = 0.0
    contact_ok = bool(
        (z_left_ground and min(z_left_ground) <= th) and (z_right_ground and min(z_right_ground) <= th)
    )

    return {
        "contact_ok": contact_ok,
        "y_left_ground": y_left_ground,
        "z_left_ground": z_left_ground,
        "y_right_ground": y_right_ground,
        "z_right_ground": z_right_ground,
        "y_link_left": y_link_left,
        "z_link_left": z_link_left,
        "y_link_right": y_link_right,
        "z_link_right": z_link_right,
    }
