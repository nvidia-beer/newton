# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared full-vehicle defaults and command-line options.

All vehicle assets and tire resolutions use these options; geometry, drive limits,
and recommended tire parameters are read from the USD assets at construction.
"""

import json
from pathlib import Path

import newton.examples

# All supported assets have four tires, ordered in the driver's frame (+Y left).
N_TIRES = 4
WHEEL_ORDER = (("FL", 0), ("FR", 1), ("RL", 2), ("RR", 3))

# Material fallbacks. Mesh topology and asset-specific recommendations come from USD.
H_SHELL = 0.010  # [m]
E_TIRE = 5.0e7  # [Pa]
NU_TIRE = 0.45
RHO_TIRE = 700.0  # [kg/m^3]
ALPHA_D = 0.15  # stiffness-proportional damping [s]
PRESSURE = 30_000.0  # [Pa]
BUILD_PRESSURE = 0.0  # [Pa]; applied pressure is PRESSURE - BUILD_PRESSURE

# Default time integration and interface budgets; the launcher can override them.
SIM_SUBSTEPS = 10
NR_ITERS = 2
PCG_ITERS = 10
FRAME_DT = 1.0 / 60.0  # [s]
GS_ITERS = 6

# Contact fallbacks and full tire reaction moment transfer.
KN = 20_000.0  # [N/m]
KD = 42.0  # [N*s/m]
MU = 0.9
TORQUE_ALPHA = 1.0

GRAVITY = 9.81  # [m/s^2]
WHEEL_SPEED_RATE = 0.2  # [rad/s per frame]
RIDE_DROP = 0.030  # initial chassis lowering [m]

# The launcher preset the diffsim preparation examples read their physical setup from
# (docker/config at the repository root; the tests resolve the root the same way).
TELEMETRY_PRESET = Path(__file__).resolve().parents[3] / "docker/config/07_vehicle_telemetry.json"


def load_telemetry_preset(path: Path | None = None) -> dict:
    """Load a ``vehicle_telemetry`` launcher preset (default :data:`TELEMETRY_PRESET`).

    Returns the parsed JSON: ``example``, ``description`` and the kebab-case ``args`` dict.
    """
    path = TELEMETRY_PRESET if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"vehicle_telemetry preset not found: {path} (docker/config/ of a source checkout, or --telemetry-config)"
        )
    config = json.loads(path.read_text())
    if config.get("example") != "vehicle_telemetry" or not isinstance(config.get("args"), dict):
        raise ValueError("--telemetry-config must name a vehicle_telemetry preset")
    return config


def create_parser():
    """Build options shared by all full-vehicle scenes without resolving assets."""
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--vehicle-asset",
        type=str,
        default=None,
        help="Vehicle USD (assets/ name or absolute path, baked by newton-tire-tool): articulated "
        "(e.g. the super jeep) or rigid hull (e.g. the Sherp). Required.",
    )
    parser.add_argument(
        "--tire-asset",
        type=str,
        default=None,
        help="Baked ANCF tire USD (assets/ name or absolute path); default: the vehicle asset's defaultTireAsset.",
    )
    parser.add_argument("--wheel-speed", type=float, default=0.0, help="Throttle angular speed [rad/s].")
    parser.add_argument(
        "--steer-angle", type=float, default=0.0, help="Steering [rad] (within the vehicle asset's maxSteer)."
    )
    parser.add_argument("--substeps", type=int, default=SIM_SUBSTEPS, help="Substeps per frame.")
    parser.add_argument("--nr-iters", type=int, default=NR_ITERS, help="Newton-Raphson iterations per substep.")
    parser.add_argument(
        "--pcg-iters",
        type=int,
        default=None,
        help="PCG iterations per NR step (default: the tire asset's recommendation).",
    )
    parser.add_argument(
        "--kn",
        type=float,
        default=None,
        help="Contact normal stiffness [N/m] (default: the tire asset's recommendation).",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=None,
        help="Contact damping [N·s/m] (default: kn x 42/20000, the baseline ratio).",
    )
    parser.add_argument("--mu", type=float, default=MU, help="Friction coefficient.")
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.0,
        help="Height of the analytic ANCF ground plane [m]; far below the tires disables it.",
    )
    parser.add_argument(
        "--world-z-offset",
        type=float,
        default=0.0,
        help="Lift the whole vehicle (chassis, suspension, its ground plane) and the tires by this height [m].",
    )
    parser.add_argument(
        "--shell-tires",
        type=str,
        default=None,
        help="JSON array of per-tire configs (same format as ancf_rigid_mujoco_tires); "
        "first entry used for all 4 tires, overrides the flat --e-tire / --nu-tire / --rho-tire / "
        "--h-shell / --pressure args. Geometry keys (n-circ, sec-divs) are not read — mesh topology "
        "comes from the baked tire asset (newton-tire-tool).",
    )
    parser.add_argument(
        "--pressure", type=float, default=PRESSURE, help="Nominal cavity pressure [Pa] (without --shell-tires)."
    )
    parser.add_argument(
        "--build-pressure",
        type=float,
        default=BUILD_PRESSURE,
        help="Pressure the rest shape was meshed at [Pa]; the shell sees pressure - build_pressure.",
    )
    parser.add_argument(
        "--e-tire", type=float, default=E_TIRE, help="Shell Young's modulus [Pa] (without --shell-tires)."
    )
    parser.add_argument("--nu-tire", type=float, default=NU_TIRE, help="Poisson ratio (without --shell-tires).")
    parser.add_argument("--rho-tire", type=float, default=RHO_TIRE, help="Density [kg/m^3] (without --shell-tires).")
    parser.add_argument("--h-shell", type=float, default=H_SHELL, help="Shell thickness [m] (without --shell-tires).")
    parser.add_argument("--thickness-gp", type=int, default=3, choices=[3, 5], help="Through-thickness Gauss points.")
    parser.add_argument(
        "--ride-drop",
        type=float,
        default=RIDE_DROP,
        help="Lower the chassis by this many metres at t=0 so the tires start pre-compressed.",
    )
    parser.add_argument(
        "--torque-alpha",
        type=float,
        default=TORQUE_ALPHA,
        help="Scale on the tire reaction torque fed back to the spindle (0 = force only).",
    )
    parser.add_argument(
        "--gs-iters",
        type=int,
        default=GS_ITERS,
        help="Interface iterations per substep (default 6, with Aitken acceleration). Each tire solve "
        "keeps the configured NR/PCG budget. 1 selects the legacy explicit pass.",
    )
    parser.add_argument(
        "--coupling-method",
        choices=["auto", "adaptive", "coupled-newton"],
        default="auto",
        help="Wheel/tire interface coupling. Auto (default) selects coupled-newton (joint shell/interface Newton "
        "corrections with a reused condensed tangent) on supported implicit vehicles with even substeps and at "
        "least three gs-iters; other models, odd substeps and large "
        "interfaces use adaptive Aitken relaxation.",
    )
    parser.add_argument("--fast-math", action="store_true", default=False, help="Enable Warp fast-math.")
    parser.add_argument(
        "--diag-period", type=int, default=30, help="Print per-wheel diagnostics every N frames (host readback)."
    )
    parser.add_argument(
        "--debug-residuals",
        action="store_true",
        default=False,
        help="Report per-NR-iteration ||R|| and the final PCG reduction per tire each --diag-period frames.",
    )
    return parser
