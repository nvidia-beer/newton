# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static tire measurements and a scalar, analytic modulus fit for the lift example."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Measurement:
    """One flat-ground experiment; pressure [Pa], additional spindle load [N], heights [m]."""

    nominal_pressure_pa: float
    additional_load_n: float
    axle_height_m: float
    height_std_m: float
    split: str


def read_measurements(path: Path) -> list[Measurement]:
    """Read explicit model-nominal pressures, loads excluding spindle weight, and axle heights."""
    fields = tuple(Measurement.__dataclass_fields__)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(fields).issubset(reader.fieldnames):
            raise ValueError(f"Calibration CSV requires columns: {', '.join(fields)}")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                values = [float(row[name]) for name in fields[:-1]]
                if not np.isfinite(values).all() or min(values[0], values[2], values[3]) <= 0 or values[1] < 0:
                    raise ValueError("pressure, height and uncertainty must be positive; additional load nonnegative")
                split = (row["split"] or "").strip()
                if split not in ("train", "validation"):
                    raise ValueError("split must be train or validation")
                rows.append(Measurement(*values, split))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
    experiments = {(r.nominal_pressure_pa, r.additional_load_n) for r in rows if r.split == "train"}
    if len(experiments) < 2:
        raise ValueError("Provide at least two distinct training load/pressure cases")
    return rows


def fit_stiffness(equilibrium, initial, measurements, bounds=(0.25, 4.0), max_iterations=20):
    """Fit a positive uniform modulus multiplier; validation rows never affect updates."""
    lower, upper = bounds
    if not np.isfinite(bounds).all() or not 0 < lower < 1.0 < upper:
        raise ValueError("Stiffness bounds must be finite and contain the starting multiplier 1")
    training = [row for row in measurements if row.split == "train"]
    original_load = equilibrium.rim_load
    # The caller records the spindle weight separately; additional loads exclude it.
    spindle_weight = equilibrium.spindle_weight

    def evaluate(scale, rows, starts):
        results = []
        for row, start in zip(rows, starts, strict=True):
            equilibrium.rim_load = spindle_weight + row.additional_load_n
            result = equilibrium.solve(row.nominal_pressure_pa, start, stiffness_scale=scale)
            if equilibrium.ground - equilibrium._full(result.coordinates)[:, 0, 1].min() > 0.025:
                raise RuntimeError("Static tire penetration exceeds the supported 25 mm limit")
            results.append(result)
        residual = np.array([(r.height - m.axle_height_m) / m.height_std_m for r, m in zip(results, rows, strict=True)])
        return results, residual

    try:
        scale = 1.0
        results, residual = evaluate(scale, training, [initial] * len(training))
        for _iteration in range(max_iterations):
            jac = np.array([r.dh_dscale * scale / m.height_std_m for r, m in zip(results, training, strict=True)])
            information = float(jac @ jac)
            if information < 1e-12:
                raise RuntimeError("Measurements do not constrain tire stiffness at this equilibrium")
            step = float(np.clip(-(jac @ residual) / information, -0.5, 0.5))
            if abs(step) < 1e-5:
                break
            for _ in range(12):
                candidate_scale = float(np.clip(scale * np.exp(step), lower, upper))
                if abs(candidate_scale - scale) < 1e-10:
                    raise RuntimeError("Stiffness fit reached its bounds; check measurements and model assumptions")
                try:
                    candidate, candidate_residual = evaluate(
                        candidate_scale, training, [r.coordinates for r in results]
                    )
                    if candidate_residual @ candidate_residual < residual @ residual:
                        scale, results, residual = candidate_scale, candidate, candidate_residual
                        break
                except (RuntimeError, ValueError):
                    pass
                step *= 0.5
            else:
                raise RuntimeError("No improving stiffness step")
        else:
            raise RuntimeError("Stiffness fit exceeded its iteration budget")

        predictions, _ = evaluate(scale, measurements, [results[0].coordinates] * len(measurements))
        rows = [
            {
                **vars(row),
                "predicted_height_m": float(result.height),
                "error_m": float(result.height - row.axle_height_m),
                "dh_dscale_m": float(result.dh_dscale),
                "equilibrium_residual": float(result.residual_norm),
            }
            for row, result in zip(measurements, predictions, strict=True)
        ]
        rms = {
            split: float(np.sqrt(np.mean([r["error_m"] ** 2 for r in rows if r["split"] == split])))
            for split in ("train", "validation")
            if any(r["split"] == split for r in rows)
        }
        return {"stiffness_scale": scale, "updates": _iteration, "height_rms_m": rms, "cases": rows}
    finally:
        equilibrium.rim_load = original_load
