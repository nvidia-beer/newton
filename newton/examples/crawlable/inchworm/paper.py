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


# frame, t, then 4 arrays: left_ground, right_ground, link_left, link_right (y and z each = space-separated).
CSV_HEADER = [
    "frame", "t",
    "y_left_ground", "z_left_ground", "y_right_ground", "z_right_ground",
    "y_link_left", "z_link_left", "y_link_right", "z_link_right",
]


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
    ) -> None:
        """Write one row: frame, t, and the 4 arrays (space-separated)."""
        if self._writer is None:
            return
        self._writer.writerow([
            frame,
            round(sim_time, 4),
            _arr_to_str(y_left_ground), _arr_to_str(z_left_ground),
            _arr_to_str(y_right_ground), _arr_to_str(z_right_ground),
            _arr_to_str(y_link_left), _arr_to_str(z_link_left),
            _arr_to_str(y_link_right), _arr_to_str(z_link_right),
        ])
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
