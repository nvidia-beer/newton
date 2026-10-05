#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Independent checks of examples 01, 02, 03, 05, 07, 08, 09 and 10, on rigid ground.
# Usage: bash docker/test-ancf-examples.sh [unittest selectors/options]
# Example: bash docker/test-ancf-examples.sh -k test_02
# ANCF_TEST_OUTPUT_DIR selects a host directory for JSON and console logs.
# Sources, assets, configs, and the vendored dependency are read-only.
set -euo pipefail

NEWTON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ANCF_TEST_OUTPUT_DIR:-${TMPDIR:-/tmp}/newton-ancf-tests-$(date +%Y%m%d-%H%M%S)}"
WARP_CACHE_DIR="${WARP_CACHE_DIR:-$HOME/.cache/newton-warp}"
NEWTON_CACHE_DIR="${NEWTON_CACHE_DIR:-$HOME/.cache/newton}"
mkdir -p "$OUTPUT_DIR" "$WARP_CACHE_DIR" "$NEWTON_CACHE_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
echo "ANCF test artifacts: $OUTPUT_DIR"

docker run --rm --gpus all --shm-size=16g --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONDONTWRITEBYTECODE=1 -e ANCF_TEST_OUTPUT_DIR=/test-results -e OPENBLAS_NUM_THREADS=1 \
    -e ANCF_WHEEL_RECOVERY="${ANCF_WHEEL_RECOVERY:-0}" \
    -e ANCF_SKID_CALIBRATION="${ANCF_SKID_CALIBRATION:-0}" -e ANCF_TRACTION_TESTS="${ANCF_TRACTION_TESTS:-0}" \
    -v "$NEWTON_DIR/newton:/workspace/newton/newton:ro" \
    -v "$NEWTON_DIR/docker:/workspace/newton/docker:ro" \
    -v "$NEWTON_DIR/third_party/mujoco_warp:/workspace/newton/third_party/mujoco_warp:ro" \
    -v "$WARP_CACHE_DIR:/root/.cache/warp" \
    -v "$NEWTON_CACHE_DIR:/root/.cache/newton" \
    -v "$OUTPUT_DIR:/test-results" \
    -w /workspace/newton newton:latest \
    python -m unittest newton.tests.test_ancf_shell_formulation newton.tests.test_ancf_schur newton.tests.test_ancf_coupled_newton newton.tests.test_ancf_coupling newton.tests.test_ancf_vehicle_examples newton.tests.test_ancf_solver_regressions newton.tests.test_ancf_terrain_contact newton.tests.test_ancf_wheel_calibration newton.tests.test_ancf_solver_comparison newton.tests.test_ancf_example_physics newton.tests.test_ancf_differentiation newton.tests.test_diffsim_ancf_tire_lift newton.tests.test_diffsim_ancf_skid_steer newton.tests.test_diffsim_ancf_tire_traction -v "$@"
