#!/bin/bash

# 🎎 Bouncing Mesh - Drop a soft body mesh and watch it bounce!
# Uses FEM soft body simulation with implicit integration
# Loads tetrahedral meshes from .mesh (Medit) format files

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEWTON_ROOT="${SCRIPT_DIR}"
IMAGE_NAME="newton-bouncing-mesh:latest"
CONTAINER_NAME="newton-bouncing-mesh"

# Default mesh file
MESH_FILE="${1:-tutorial/spot.mesh}"

echo -e "${BLUE}==== 🎎 Newton Bouncing Mesh ====${NC}"
echo "Newton root: ${NEWTON_ROOT}"
echo "Mesh file: ${MESH_FILE}"

# Build from main devcontainer if needed
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo -e "${BLUE}Building Docker image from newton/.devcontainer...${NC}"
    docker build -t $IMAGE_NAME -f "${NEWTON_ROOT}/.devcontainer/Dockerfile" "${NEWTON_ROOT}/.devcontainer/"
fi

# Remove existing container
docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true

echo -e "${BLUE}Dropping the mesh...${NC}"

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
        echo '🎎 Running bouncing mesh simulation with OpenGL viewer...'
        echo '   Window will open in DCV desktop'
        echo ''
        .venv/bin/python -m newton.examples.soft.example_bouncing_mesh \
            --mesh_file ${MESH_FILE} \
            --scale 1.0 \
            --initial_height 1.0 \
            --mass 2.0 \
            --k_mu 1.0e6 \
            --k_lambda 1.0e6 \
            --k_damp 1.0 \
            --spring_ke 1.0e5 \
            --spring_kd 1.0 \
            --gravity 9.81 \
            --substeps 5 \
            --num_frames 300
    "

echo -e "${GREEN}==== 🎎 Done! ====${NC}"

