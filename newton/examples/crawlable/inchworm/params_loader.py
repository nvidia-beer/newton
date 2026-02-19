# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Load/save inchworm physics and calibration parameters to JSON.

Use from the inchworm_crawling example:
  --params crawlable/inchworm/inchworm_params.json   # load defaults from file (CLI still overrides)
  --save_params crawlable/inchworm/my_calibration.json  # write effective params at end of run
"""

from __future__ import annotations

import json
import os
from typing import Any

# Keys accepted in the params JSON (same as Example __init__ + run kwargs).
# Order and presence here define what we save; extra keys in JSON are ignored when applying.
INCHWORM_PARAM_KEYS = [
    "length",
    "width",
    "height",
    "subdivisions_x",
    "subdivisions_y",
    "subdivisions_z",
    "num_chambers_x",
    "num_chambers_y",
    "num_chambers_z",
    "initial_height",
    "total_mass",
    "k_mu",
    "k_lambda",
    "k_damp",
    "spring_ke",
    "spring_kd",
    "gravity",
    "max_pressure",
    "substeps",
    "anisotropy_x",
    "anisotropy_y",
    "anisotropy_z",
    "torque_stiffness",
    "torque_damping",
    "chamber_stiffness_scale",
    "chamber_active_inflation",
    "ground_friction",
    "contact_offset",
    "contact_iterations",
    "ground_ke",
    "particle_radius",
    "gait_enabled",
    "gait_freq",
    "gait_amplitude",
    "gait_phase",
    "gait_baseline",
    "gait_pressure_min",
    "gait_pressure_max",
    "settle_seconds",
    "start_at_ground_level",
    "use_crawlable_stick_slip",
    "stick_slip_scale",
    "stick_slip_amplitude",
    "crawl_direction",
    "num_frames",
    "validate_contact",
    "stop_on_lost_contact",
    "csv_log_interval",
]


def load_params(path: str | os.PathLike[str]) -> dict[str, Any]:
    """
    Load inchworm parameters from a JSON file.
    Returns a dict of key -> value; only keys in INCHWORM_PARAM_KEYS are included.
    Keys starting with '_' are ignored (e.g. _comment).
    """
    path = os.path.expanduser(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, Any] = {}
    for k in INCHWORM_PARAM_KEYS:
        if k in raw:
            out[k] = raw[k]
    return out


def save_params(
    path: str | os.PathLike[str],
    params: dict[str, Any],
    comment: str | None = None,
) -> None:
    """
    Save inchworm parameters to a JSON file.
    Only keys in INCHWORM_PARAM_KEYS are written; keys are written in INCHWORM_PARAM_KEYS order.
    Optional top-level _comment can be set.
    """
    path = os.path.expanduser(path)
    out: dict[str, Any] = {}
    if comment is not None:
        out["_comment"] = comment
    for k in INCHWORM_PARAM_KEYS:
        if k in params:
            out[k] = params[k]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return None
