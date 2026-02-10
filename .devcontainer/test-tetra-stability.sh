#!/bin/bash
# Test Tetrahedral Mesh Stability
# This utility iteratively adds tetrahedra to a soft body simulation one by one,
# testing stability after each addition. It helps identify problematic tetrahedra
# that cause simulation instability.
# This script runs via Docker, similar to run-examples.sh
#
# Usage: ./test-tetra-stability.sh [mesh_file] [extra_args...]
# Examples:
#   ./test-tetra-stability.sh examples/assets/spot.mesh
#   ./test-tetra-stability.sh examples/assets/spot.mesh --test_steps 100 --max_position 10.0
#   ./test-tetra-stability.sh examples/assets/ball.mesh --scale 0.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_DIR="$(dirname "$SCRIPT_DIR")"

echo "═══════════════════════════════════════════════════════════════"
echo "              Tetrahedral Mesh Stability Tester"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Parse arguments - first arg is mesh file, rest are passed to Python script
# Default: use spot_fixed.mesh if available, else spot.mesh
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    # First arg is mesh file
    MESH_FILE="$1"
    shift  # Remove first arg, keep rest for Python script
    EXTRA_ARGS=("$@")
else
    # No mesh file provided, use default (script will auto-detect spot_fixed.mesh)
    MESH_FILE=""
    EXTRA_ARGS=("$@")
fi

# The newton directory is mounted at /workspace/newton/newton
# Python runs from /workspace, so use path: newton/newton/examples/assets/filename.mesh
if [ -z "$MESH_FILE" ]; then
    # No mesh file specified - Python script will auto-detect spot_fixed.mesh
    MESH_REL_PATH=""
    echo "No mesh file specified - will use spot_fixed.mesh if available, else spot.mesh"
elif [[ "$MESH_FILE" == examples/assets/* ]]; then
    MESH_REL_PATH="newton/newton/$MESH_FILE"
elif [[ "$MESH_FILE" == *"/examples/assets/"* ]]; then
    # Extract filename and build path
    MESH_REL_PATH="newton/newton/examples/assets/$(basename "$MESH_FILE")"
else
    # Assume it's just a filename
    MESH_REL_PATH="newton/newton/examples/assets/$MESH_FILE"
fi

if [ -n "$MESH_FILE" ]; then
    echo "Mesh file: $MESH_FILE"
    echo "Using path in container: $MESH_REL_PATH"
else
    echo "Using default mesh (auto-detected)"
fi
echo ""
echo "Note: spot_fixed.mesh is recommended (pre-processed, more stable)"
echo "      Create it with: ./create-spot-fixed-mesh.sh"
echo ""

# Check if Docker image exists, if not build it automatically
if ! docker image inspect newton:latest >/dev/null 2>&1; then
    echo "Docker image 'newton:latest' not found."
    echo "Building Docker image automatically..."
    echo ""
    "$SCRIPT_DIR/build-docker.sh"
    echo ""
    echo "Build complete! Starting stability test..."
    echo ""
fi

# Check if GPU is available
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected - enabling GPU acceleration"
    GPU_ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all -e __GLX_VENDOR_LIBRARY_NAME=nvidia"
else
    echo "⚠ No NVIDIA GPU detected - running in CPU mode"
    echo "  (Performance will be slower, but tests will still work)"
fi
echo ""

# Ensure DISPLAY is set (not needed for headless, but doesn't hurt)
if [ -z "$DISPLAY" ]; then
    # Try to auto-detect
    if [ -e "/tmp/.X11-unix/X0" ]; then
        export DISPLAY=":0"
    elif [ -e "/tmp/.X11-unix/X1" ]; then
        export DISPLAY=":1"
    else
        export DISPLAY=":0"  # Default fallback
    fi
fi

# Build docker command with X11 support (for potential future GUI)
DOCKER_X11_ARGS=(
    -e "DISPLAY=$DISPLAY"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
)

# Add XAUTHORITY if available
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
    DOCKER_X11_ARGS+=(
        -e "XAUTHORITY=$XAUTHORITY"
        -v "$XAUTHORITY:$XAUTHORITY:ro"
    )
fi

echo "Running stability test in Docker container..."
echo ""

# Run inside Docker container (headless, no -it flag)
# Build command - only add --mesh_file if specified
if [ -n "$MESH_REL_PATH" ]; then
    docker run --rm \
        $GPU_ARGS \
        --shm-size=16g \
        --network=host \
        --privileged \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e NEWTON_DISABLE_CUDA_INTEROP=1 \
        "${DOCKER_X11_ARGS[@]}" \
        -v "$NEWTON_DIR/newton:/workspace/newton/newton" \
        newton:latest \
        python -m newton.examples.utils.test_tetra_stability \
            --mesh_file "$MESH_REL_PATH" \
            "${EXTRA_ARGS[@]}"
else
    docker run --rm \
        $GPU_ARGS \
        --shm-size=16g \
        --network=host \
        --privileged \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e NEWTON_DISABLE_CUDA_INTEROP=1 \
        "${DOCKER_X11_ARGS[@]}" \
        -v "$NEWTON_DIR/newton:/workspace/newton/newton" \
        newton:latest \
        python -m newton.examples.utils.test_tetra_stability \
            "${EXTRA_ARGS[@]}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "              Stability test complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
