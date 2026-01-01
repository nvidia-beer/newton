#!/bin/bash

# 🏀 Simple Bouncing Ball - Drop and watch it bounce!
# Uses FEM soft body simulation with implicit integration

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEWTON_ROOT="${SCRIPT_DIR}"
IMAGE_NAME="newton-bouncing-ball:latest"
CONTAINER_NAME="newton-bouncing-ball"

echo -e "${BLUE}==== 🏀 Newton Bouncing Ball ====${NC}"
echo "Newton root: ${NEWTON_ROOT}"

# Build from main devcontainer if needed
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo -e "${BLUE}Building Docker image from newton/.devcontainer...${NC}"
    docker build -t $IMAGE_NAME -f "${NEWTON_ROOT}/.devcontainer/Dockerfile" "${NEWTON_ROOT}/.devcontainer/"
fi

# Remove existing container
docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true

echo -e "${BLUE}Dropping the ball...${NC}"

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
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /run/user/1000/dcv/session.xauth:/root/.Xauthority:ro \
    -v "${NEWTON_ROOT}":/workspaces/newton:rw \
    -w /workspaces/newton \
    $IMAGE_NAME \
    bash -c "
        echo '📦 Installing newton and rerun-sdk...'
        uv sync --extra sim --extra examples --dev 2>&1 | tail -3
        .venv/bin/pip install rerun-sdk -q
        echo ''
        echo '🏀 Running bouncing ball simulation with OpenGL viewer...'
        echo '   Window will open in DCV desktop'
        echo ''
        .venv/bin/python -m newton.examples.soft.example_bouncing_ball \
            --radius 0.3 \
            --subdivisions 2 \
            --initial_height 0.9 \
            --mass 1.0 \
            --k_mu 4.0e4 \
            --k_lambda 4.0e4 \
            --k_damp 3.0 \
            --gravity 9.81 \
            --substeps 12 \
            --num_frames 600
    "

echo -e "${GREEN}==== 🏀 Done! ====${NC}"

