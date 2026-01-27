#!/bin/bash
# Rigid-Soft Interaction Example
# Usage: ./run_rigid_soft_interaction.sh [xpbd|mujoco]

set -e

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

SOLVER="${1:-xpbd}"  # Default to xpbd

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEWTON_ROOT="${SCRIPT_DIR}"
IMAGE_NAME="newton-rigid-soft:latest"
CONTAINER_NAME="newton-rigid-soft"

echo -e "${BLUE}==== Rigid-Soft Interaction (${SOLVER^^} solver) ====${NC}"
echo -e "  ${GREEN}GREEN ball:${NC} Rigid body (${SOLVER^^})"
echo -e "  ${BLUE}BLUE ball:${NC} Soft/FEM body"
echo ""

# Build if needed
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo -e "${BLUE}Building Docker image...${NC}"
    docker build -t $IMAGE_NAME -f "${NEWTON_ROOT}/.devcontainer/Dockerfile" "${NEWTON_ROOT}/.devcontainer/"
fi

docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true

docker run --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --shm-size=16g \
    --network=host \
    --privileged \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NEWTON_DISABLE_CUDA_INTEROP=1 \
    -e DISPLAY=:1 \
    -e XAUTHORITY=/root/.Xauthority \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -e SOLVER=${SOLVER} \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /run/user/1000/dcv/session.xauth:/root/.Xauthority:ro \
    -v "${NEWTON_ROOT}":/workspaces/newton:rw \
    -w /workspaces/newton \
    $IMAGE_NAME \
    bash -c '
        echo "📦 Installing newton..."
        uv sync --extra sim --extra examples --dev 2>&1 | tail -3
        echo ""
        echo "🔴🔵 Running rigid-soft interaction ($SOLVER solver)..."
        .venv/bin/python -m newton.examples.soft.example_rigid_soft_interaction \
            --solver $SOLVER \
            --ball-radius 0.3 \
            --drop-height 1.5 \
            --substeps 8 \
            --num-frames 600
    '

echo -e "${GREEN}Done!${NC}"
