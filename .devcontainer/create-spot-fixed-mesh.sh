#!/bin/bash
# Create spot_fixed.mesh from spot.mesh with proper centering, fixed tetrahedra, and filtering
# This processes spot.mesh (centers Z, flips inverted tetrahedra, removes degenerate tetrahedra) 
# and saves it as spot_fixed.mesh for direct loading without calculations in example_bouncing_mesh.py
# This script runs via Docker, similar to run-examples.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_DIR="$(dirname "$SCRIPT_DIR")"

echo "═══════════════════════════════════════════════════════════════"
echo "              Creating spot_fixed.mesh from spot.mesh"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if Docker image exists, if not build it automatically
if ! docker image inspect newton:latest >/dev/null 2>&1; then
    echo "Docker image 'newton:latest' not found."
    echo "Building Docker image automatically..."
    echo ""
    "$SCRIPT_DIR/build-docker.sh"
    echo ""
    echo "Build complete! Creating spot_fixed.mesh..."
    echo ""
fi

# Check if GPU is available
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected - enabling GPU acceleration"
    GPU_ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all -e __GLX_VENDOR_LIBRARY_NAME=nvidia"
else
    echo "⚠ No NVIDIA GPU detected - running in CPU mode"
fi
echo ""

# Ensure DISPLAY is set
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

# Build docker command with X11 support
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

# Run inside Docker container
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
    python -m newton.examples.soft.create_spot_fixed_mesh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "              spot_fixed.mesh created successfully!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "You can now run the bouncing_mesh example with spot_fixed.mesh:"
echo "  python -m newton.examples.soft.example_bouncing_mesh --mesh_file examples/assets/spot_fixed.mesh"
echo ""
echo "Or use the run-examples.sh script (modify to use spot_fixed.mesh):"
echo "  ./run-examples.sh bouncing_mesh"
echo ""
