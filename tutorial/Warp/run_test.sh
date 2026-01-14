#!/bin/bash
# Run the stability test script inside Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=============================================="
echo "  Running Stability Test in Docker"
echo "=============================================="

# Build Docker image if needed
if [[ "$(docker images -q newton-jupyter 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t newton-jupyter "${NEWTON_ROOT}/.devcontainer"
fi

# Run the test script in Docker
docker run --rm \
    --gpus=all \
    --shm-size=16g \
    -v "${NEWTON_ROOT}:/workspaces/newton" \
    -w /workspaces/newton \
    newton-jupyter \
    bash -c '
        cd /workspaces/newton
        
        # Sync dependencies
        uv sync --extra sim --extra examples --dev 2>/dev/null || true
        
        # Run the test script
        uv run python tutorial/DLI/test_stability.py
    '
