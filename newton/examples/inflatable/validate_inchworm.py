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
Validation script for inchworm simulation (arXiv:1911.05227).

Runs the inchworm headless at paper scale (or custom params), records
center-of-mass (COM) x-position over time, and reports:
  - Net displacement and step distance per gait period (S).
  - Normalized step distance S/L for comparison to the paper's theory.

Optional limit-case test: with --test_symmetric (ψ=0), asserts that net
displacement is small (symmetric gait should yield ~zero progression).

Usage (from repo root, or inside Newton devcontainer):
  python -m newton.examples.inflatable.validate_inchworm
  python -m newton.examples.inflatable.validate_inchworm --test_symmetric
  python -m newton.examples.inflatable.validate_inchworm --length 0.12 --mass 0.052 --num_periods 5
"""

import argparse
import sys
import warp as wp
import numpy as np

# Import the inchworm Example (same process as example_inchworm.main)
from newton.examples.inflatable.example_inchworm import Example


def com_x(state) -> float:
    """Center-of-mass x position (assuming equal particle masses)."""
    q = np.array(state.particle_q.numpy(), dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(-1, 3)
    return float(np.mean(q[:, 0]))


def run_validation(
    length: float = 0.12,
    width: float = 0.03,
    height: float = 0.01,
    mass: float = 0.052,
    ground_friction: float = 0.39,
    gait_freq: float = 0.2,
    gait_phase: float = 1.57,
    gait_amplitude: float = 0.8,
    gait_baseline: float = 1.2,
    num_periods: int = 5,
    substeps: int = 8,
    device: str | None = None,
    test_symmetric: bool = False,
) -> dict:
    """
    Run inchworm headless for num_periods gait periods; return metrics.
    If test_symmetric, use gait_phase=0 and assert small net displacement.
    """
    period = 1.0 / gait_freq
    fps = 60
    frame_dt = 1.0 / fps
    num_frames = max(1, int(np.ceil(num_periods * period / frame_dt)))

    if test_symmetric:
        gait_phase = 0.0

    wp.init()
    with wp.ScopedDevice(device):
        ex = Example(
            viewer=None,
            length=length,
            width=width,
            height=height,
            subdivisions_x=8,
            subdivisions_y=10,
            subdivisions_z=3,
            num_chambers_x=1,
            num_chambers_y=2,
            num_chambers_z=2,
            initial_height=height * 2,
            mass=mass,
            ground_friction=ground_friction,
            gait_enabled=True,
            gait_freq=gait_freq,
            gait_amplitude=gait_amplitude,
            gait_phase=gait_phase,
            gait_baseline=gait_baseline,
            substeps=substeps,
        )
        L = length
        com_x0 = com_x(ex.state_0)
        com_x_per_period = [com_x0]
        steps_done = 0
        steps_per_period_actual = int(round(period / ex.sim_dt))

        for frame in range(num_frames):
            ex.step()
            steps_done += ex.substeps
            if steps_done >= steps_per_period_actual * (len(com_x_per_period)):
                com_x_per_period.append(com_x(ex.state_0))

        com_x_final = com_x(ex.state_0)
        net_disp = com_x_final - com_x0
        n_periods_actual = len(com_x_per_period) - 1
        if n_periods_actual < 1:
            n_periods_actual = 1
        step_distance = net_disp / n_periods_actual  # per period
        step_distance_normalized = step_distance / L if L > 0 else 0.0

        metrics = {
            "com_x0": com_x0,
            "com_x_final": com_x_final,
            "net_displacement_m": net_disp,
            "num_periods": n_periods_actual,
            "step_distance_m": step_distance,
            "step_distance_S_over_L": step_distance_normalized,
            "length_m": L,
        }

        if test_symmetric:
            # Paper: ψ=0 ⇒ symmetric gait ⇒ zero progression
            tol = 0.02 * L  # allow 2% of body length as numerical drift
            if abs(net_disp) > tol:
                raise AssertionError(
                    f"Symmetric gait (ψ=0) test failed: |net_displacement|={abs(net_disp):.6f} m > tol={tol:.6f} m (2% of L)"
                )
            metrics["symmetric_test_passed"] = True

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Validate inchworm sim: paper-scale run, step distance S, optional ψ=0 test."
    )
    parser.add_argument("--length", type=float, default=0.12, help="Body length (m); paper L=0.12")
    parser.add_argument("--width", type=float, default=0.03)
    parser.add_argument("--height", type=float, default=0.01)
    parser.add_argument("--mass", type=float, default=0.052, help="kg; paper M=0.052")
    parser.add_argument("--ground_friction", type=float, default=0.39, help="Paper µ≈0.389")
    parser.add_argument("--gait_freq", type=float, default=0.2)
    parser.add_argument("--gait_phase", type=float, default=1.57, help="rad; π/2 for inchworm")
    parser.add_argument("--gait_amplitude", type=float, default=0.8)
    parser.add_argument("--gait_baseline", type=float, default=1.2)
    parser.add_argument("--num_periods", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--test_symmetric",
        action="store_true",
        help="Run with ψ=0 and assert net displacement < 2%% of L",
    )
    args = parser.parse_args()

    try:
        metrics = run_validation(
            length=args.length,
            width=args.width,
            height=args.height,
            mass=args.mass,
            ground_friction=args.ground_friction,
            gait_freq=args.gait_freq,
            gait_phase=args.gait_phase,
            gait_amplitude=args.gait_amplitude,
            gait_baseline=args.gait_baseline,
            num_periods=args.num_periods,
            substeps=args.substeps,
            device=args.device,
            test_symmetric=args.test_symmetric,
        )
    except AssertionError as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Inchworm validation metrics (paper scale)")
    print(f"  Length L = {metrics['length_m']} m")
    print(f"  COM x(0) = {metrics['com_x0']:.6f} m")
    print(f"  COM x(end) = {metrics['com_x_final']:.6f} m")
    print(f"  Net displacement = {metrics['net_displacement_m']:.6f} m")
    print(f"  Step distance S (per period) = {metrics['step_distance_m']:.6f} m")
    print(f"  S/L (dimensionless) = {metrics['step_distance_S_over_L']:.4f}")
    print(f"  Periods = {metrics['num_periods']}")
    if metrics.get("symmetric_test_passed"):
        print("  Symmetric (ψ=0) test: PASSED")
    print("\nCompare S/L to paper Fig 7/8 (e.g. ψ=π/4, γ=π/2).")


if __name__ == "__main__":
    main()
