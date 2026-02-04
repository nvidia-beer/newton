#!/bin/bash
# Newton Example Runner
# Usage: ./run-examples.sh [example_name|number] [extra_args...]
# Examples:
#   ./run-examples.sh                                  # Interactive menu
#   ./run-examples.sh 1                                # Run example 1
#   ./run-examples.sh bouncing_ball
#   ./run-examples.sh rigid_soft_interaction           # Default: XPBD solver
#   ./run-examples.sh rigid_soft_interaction --solver xpbd
#   ./run-examples.sh rigid_soft_interaction --solver mujoco
#   ./run-examples.sh rigid_soft_interaction --solver mujoco --use-mujoco-cpu

set -e

# Define available examples
declare -a EXAMPLES=(
    "bouncing_ball"
    "rigid_soft_interaction"
    "bouncing_mesh"
    "inflatable"
    "inflatable_rigid"
    "inflatable_table"
)

declare -A EXAMPLE_DESCRIPTIONS=(
    ["bouncing_ball"]="Soft ball bouncing on the ground"
    ["rigid_soft_interaction"]="Rigid-soft interaction (supports --solver xpbd|mujoco)"
    ["bouncing_mesh"]="Mesh-based bouncing simulation"
    ["inflatable"]="Inflatable soft body with pressure control (Press I/K/O)"
    ["inflatable_rigid"]="Inflatable with rigid plate on top (supports --solver xpbd|mujoco)"
    ["inflatable_table"]="4 inflatable supports with rigid table top (Press I/K/O)"
)

declare -A EXAMPLE_SOLVER_OPTIONS=(
    ["rigid_soft_interaction"]="xpbd mujoco"
    ["inflatable_rigid"]="xpbd mujoco"
)

# If no argument provided, show menu
if [ $# -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Newton Examples - Select a Simulation"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    for i in "${!EXAMPLES[@]}"; do
        example="${EXAMPLES[$i]}"
        desc="${EXAMPLE_DESCRIPTIONS[$example]}"
        printf "  %d) %-30s - %s\n" $((i+1)) "$example" "$desc"
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    read -p "Select example (1-${#EXAMPLES[@]}): " choice
    echo ""
    
    # Validate choice
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#EXAMPLES[@]}" ]; then
        echo "Error: Invalid choice. Please select a number between 1 and ${#EXAMPLES[@]}"
        exit 1
    fi
    
    EXAMPLE="${EXAMPLES[$((choice-1))]}"
else
    # Check if first argument is a number
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        if [ "$1" -lt 1 ] || [ "$1" -gt "${#EXAMPLES[@]}" ]; then
            echo "Error: Invalid example number. Please select between 1 and ${#EXAMPLES[@]}"
            exit 1
        fi
        EXAMPLE="${EXAMPLES[$(($1-1))]}"
        shift
    else
        EXAMPLE="$1"
        shift
    fi
fi

# Show selected example
echo "Running example: $EXAMPLE"
echo ""

# Check if this example supports solver selection and user hasn't specified --solver
SOLVER_ARG=""
if [[ -n "${EXAMPLE_SOLVER_OPTIONS[$EXAMPLE]}" ]] && [[ ! "$*" =~ --solver ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Select Rigid Body Solver"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  1) xpbd     - XPBD solver (default, unified model)"
    echo "  2) mujoco   - MuJoCo solver (hybrid: MuJoCo for rigid, Newton for soft)"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    read -p "Select solver (1-2, or press Enter for default XPBD): " solver_choice
    echo ""
    
    case "$solver_choice" in
        1|"")
            SOLVER_ARG="--solver xpbd"
            echo "Selected: XPBD solver"
            ;;
        2)
            SOLVER_ARG="--solver mujoco"
            echo "Selected: MuJoCo solver"
            ;;
        *)
            echo "Invalid choice, using default XPBD solver"
            SOLVER_ARG="--solver xpbd"
            ;;
    esac
    echo ""
fi

# Default stable parameters for examples that need them
EXTRA_ARGS="$*"
if [ -z "$EXTRA_ARGS" ]; then
    case "$EXAMPLE" in
        bouncing_ball)
            # Stable and fast: fewer substeps, coarser mesh
            EXTRA_ARGS="--radius 0.3 --initial_height 0.9 --k_mu 4e4 --k_lambda 4e4 --k_damp 3.0 --substeps 10 --subdivisions 1 --interior_layers 1"
            ;;
        rigid_soft_interaction)
            # Rigid ball drops onto soft ball (SolverSoft - pure FEM, no pressure control)
            # Solver will be selected interactively or via --solver xpbd|mujoco
            EXTRA_ARGS="--ball-radius 0.3 --drop-height 1.5 --substeps 16"
            ;;
        inflatable)
            # Inflatable soft body with manual pressure control
            # Press I to inflate, K to deflate, O to reset
            EXTRA_ARGS="--radius 0.3 --initial_height 0.4 --k_mu 5e4 --k_lambda 5e4 --k_damp 2.0 --max_pressure 5.0 --substeps 8 --subdivisions 2 --interior_layers 2"
            ;;
        inflatable_rigid)
            # Inflatable soft body with large flat rigid plate on top
            # Plate is 5x the soft body diameter (3.0m for 0.3m radius soft)
            # Note: Using heavier mass (0.01kg) and smaller particle radius (0.015m) for better MuJoCo interaction
            EXTRA_ARGS="--radius 0.3 --rigid_width 3.0 --rigid_mass 0.01 --particle_radius 0.015 --k_mu 1e5 --k_lambda 1e5 --k_damp 5.0 --spring_ke 5e4 --spring_kd 5.0 --max_pressure 5.0 --substeps 16 --subdivisions 2 --interior_layers 2 --num_frames 800"
            ;;
        inflatable_table)
            # Inflatable table: 4 soft bodies at corners supporting a rigid plate
            # Press I to inflate, K to deflate, O to reset
            # Note: plate must be ultra-light (0.004kg) for contact forces to work
            EXTRA_ARGS="--radius 0.25 --rigid_width 3.0 --rigid_mass 0.004 --particle_radius 0.03 --k_mu 5e4 --k_lambda 5e4 --k_damp 50.0 --spring_ke 2e4 --spring_kd 20.0 --max_pressure 5.0 --substeps 32 --subdivisions 2 --interior_layers 2 --num_frames 800"
            ;;
    esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_DIR="$(dirname "$SCRIPT_DIR")"

# Check if Docker image exists, if not build it automatically
if ! docker image inspect newton:latest >/dev/null 2>&1; then
    echo "Docker image 'newton:latest' not found."
    echo "Building Docker image automatically..."
    echo ""
    "$SCRIPT_DIR/build-docker.sh"
    echo ""
    echo "Build complete! Starting example..."
    echo ""
fi

# Check if GPU is available
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected - enabling GPU acceleration"
    GPU_ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all -e __GLX_VENDOR_LIBRARY_NAME=nvidia"
else
    echo "⚠ No NVIDIA GPU detected - running in CPU mode"
    echo "  (Performance will be slower, but examples will still work)"
fi
echo ""

docker run --rm -it \
    $GPU_ARGS \
    --shm-size=16g \
    --network=host \
    --privileged \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NEWTON_DISABLE_CUDA_INTEROP=1 \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$NEWTON_DIR/newton:/workspace/newton/newton" \
    newton:latest \
    python -m newton.examples "$EXAMPLE" $SOLVER_ARG $EXTRA_ARGS
