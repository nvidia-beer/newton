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
#   ./run-examples.sh soft_on_box                      # With constraint-based contacts
#   ./run-examples.sh soft_on_box --no-constraint-contacts  # With force-based contacts

set -e

# Define available examples
declare -a EXAMPLES=(
    "bouncing_sphere"
    "bouncing_cylinder"
    "bouncing_box"
    "rigid_soft_interaction"
    "soft_on_box"
    "bouncing_mesh"
    "inflatable_box"
    "inflatable_sphere"
    "inflatable_rigid_box"
    "inflatable_rigid_sphere"
    "inflatable_table_box"
    "inflatable_table_sphere"
)

declare -A EXAMPLE_DESCRIPTIONS=(
    ["bouncing_sphere"]="Soft sphere bouncing on the ground"
    ["bouncing_cylinder"]="Soft cylinder bouncing and tumbling"
    ["bouncing_box"]="Soft box bouncing and tumbling"
    ["rigid_soft_interaction"]="Rigid-soft interaction (supports --solver xpbd|mujoco)"
    ["soft_on_box"]="Soft object sitting on rigid XPBD box (supports --constraint-contacts)"
    ["bouncing_mesh"]="Mesh-based bouncing simulation"
    ["inflatable_box"]="Inflatable soft body box with pressure control (Press I/K/O)"
    ["inflatable_sphere"]="Inflatable soft body sphere with pressure control (Press I/K/O)"
    ["inflatable_rigid_box"]="Inflatable box with rigid plate on top (supports --solver xpbd|mujoco)"
    ["inflatable_rigid_sphere"]="Inflatable sphere with rigid plate on top (supports --solver xpbd|mujoco)"
    ["inflatable_table_box"]="4 inflatable boxes at corners with rigid table top (Press I/K/O)"
    ["inflatable_table_sphere"]="4 inflatable spheres at corners with rigid table top (Press I/K/O)"
)

declare -A EXAMPLE_SOLVER_OPTIONS=(
    ["rigid_soft_interaction"]="xpbd mujoco"
    ["inflatable_rigid_box"]="xpbd mujoco"
    ["inflatable_rigid_sphere"]="xpbd mujoco"
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

# Interactive parameter selection for bouncing_box
BOX_ARGS=""
if [ "$EXAMPLE" == "bouncing_box" ] && [ -z "$*" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Bouncing Box - Box Parameters"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    read -p "Box size - width height depth (default: 0.6 0.8 1.0): " box_size
    if [ -z "$box_size" ]; then
        box_size="0.6 0.8 1.0"
    fi
    
    read -p "Segments per axis - width height depth (default: 3 5 7): " box_segments
    if [ -z "$box_segments" ]; then
        box_segments="3 5 7"
    fi
    
    echo ""
    echo "Using box size: $box_size, segments: $box_segments"
    echo ""
    
    # Use default values for other parameters
    BOX_ARGS="--size $box_size --initial_height 2.5 --mass 0.5 --k_mu 2e5 --k_lambda 2e5 --k_damp 0.5 --substeps 8 --segments $box_segments --num_frames 600"
fi

# Default stable parameters for examples that need them
EXTRA_ARGS="$*"
if [ -z "$EXTRA_ARGS" ]; then
    case "$EXAMPLE" in
        bouncing_sphere)
            # Stable and fast: fewer substeps, coarser mesh
            EXTRA_ARGS="--radius 0.3 --initial_height 0.9 --k_mu 4e4 --k_lambda 4e4 --k_damp 3.0 --substeps 10 --subdivisions 1 --interior_layers 1"
            ;;
        bouncing_cylinder)
            # Cylinder bouncing and tumbling
            EXTRA_ARGS="--radius 0.2 --height 0.6 --initial_height 2.5 --k_mu 2e5 --k_lambda 2e5 --k_damp 0.5 --substeps 8 --radial_subdivisions 16 --height_subdivisions 1 --interior_layers 2"
            ;;
        bouncing_box)
            # Use interactive selection if available, otherwise use default
            if [ -n "$BOX_ARGS" ]; then
                EXTRA_ARGS="$BOX_ARGS"
            else
                # Box bouncing and tumbling - subdivided into multiple cubic elements
                # Larger non-uniform box: size=(0.6, 0.8, 1.0), segments=(3, 5, 7)
                # Creates 105 cubic elements (630 tetrahedra) with non-uniform subdivision
                # Each cube uses stable 6-tet pattern (all share v6 corner vertex)
                EXTRA_ARGS="--size 0.6 0.8 1.0 --initial_height 2.5 --mass 0.5 --k_mu 2e5 --k_lambda 2e5 --k_damp 0.5 --substeps 8 --segments 3 5 7 --num_frames 600"
            fi
            ;;
        rigid_soft_interaction)
            # Rigid ball drops onto soft ball (SolverSoft - pure FEM, no pressure control)
            # Solver will be selected interactively or via --solver xpbd|mujoco
            EXTRA_ARGS="--ball-radius 0.3 --drop-height 1.5 --substeps 16"
            ;;
        soft_on_box)
            # Soft sphere sitting on top of rigid XPBD box
            # Demonstrates constraint-based contact handling (prevents penetration)
            # Use --constraint-contacts to enable constraint-based contacts
            # Note: substeps is hardcoded to 16 in the example (not a CLI argument)
            EXTRA_ARGS="--constraint-contacts"
            ;;
        bouncing_mesh)
            # Mesh-based bouncing simulation - automatically uses spot_fixed.mesh if available, else spot.mesh
            # Parameters match old working example for stability, with increased substeps
            # spot_fixed.mesh is normalized to 10m, use --scale 0.1 to get 1m mesh
            # To create spot_fixed.mesh: ./create-spot-fixed-mesh.sh
            EXTRA_ARGS="--scale 0.1 --initial_height 0.2 --mass 1.0 --k_mu 5.0 --k_lambda 5.0 --k_damp 40.0 --spring_ke 50.0 --spring_kd 40.0 --substeps 20"
            ;;
        inflatable_box)
            # Inflatable soft body box with manual pressure control
            # Press I to inflate, K to deflate, O to reset
            # Using 5x5x5 segments for more particles (125 cubes = 750 tetrahedra)
            EXTRA_ARGS="--size 0.4 0.4 0.4 --initial_height 0.5 --k_mu 1e5 --k_lambda 1e5 --k_damp 1.0 --max_pressure 5.0 --substeps 5 --segments 5 5 5 --num_frames 1800"
            ;;
        inflatable_sphere)
            # Inflatable soft body sphere with manual pressure control
            # Press I to inflate, K to deflate, O to reset
            EXTRA_ARGS="--radius 0.3 --initial_height 0.4 --k_mu 5e4 --k_lambda 5e4 --k_damp 2.0 --max_pressure 5.0 --substeps 8 --subdivisions 2 --interior_layers 2 --num_frames 1800"
            ;;
        inflatable_rigid_box)
            # Inflatable soft body box with large flat rigid plate on top
            # Plate is 5x the soft body size (3.0m for 0.3m box)
            # Note: Using heavier mass (0.01kg) and smaller particle radius (0.015m) for better MuJoCo interaction
            EXTRA_ARGS="--size 0.3 0.3 0.3 --rigid_width 3.0 --rigid_mass 0.01 --particle_radius 0.015 --k_mu 1e5 --k_lambda 1e5 --k_damp 5.0 --spring_ke 5e4 --spring_kd 5.0 --max_pressure 5.0 --substeps 16 --segments 3 3 3 --num_frames 800"
            ;;
        inflatable_rigid_sphere)
            # Inflatable soft body sphere with large flat rigid plate on top
            # Plate is 5x the soft body diameter (3.0m for 0.3m radius soft)
            # Note: Using heavier mass (0.01kg) and smaller particle radius (0.015m) for better MuJoCo interaction
            EXTRA_ARGS="--radius 0.3 --rigid_width 3.0 --rigid_mass 0.01 --particle_radius 0.015 --k_mu 1e5 --k_lambda 1e5 --k_damp 5.0 --spring_ke 5e4 --spring_kd 5.0 --max_pressure 5.0 --substeps 16 --subdivisions 2 --interior_layers 2 --num_frames 800"
            ;;
        inflatable_table_box)
            # Inflatable table: 4 soft body boxes at corners supporting a rigid plate
            # Press I to inflate, K to deflate, O to reset
            # Note: plate is ultra-light (0.001kg) with high friction to prevent sliding
            EXTRA_ARGS="--size 0.25 0.25 0.25 --rigid_width 3.0 --rigid_mass 0.001 --particle_radius 0.03 --k_mu 5e4 --k_lambda 5e4 --k_damp 50.0 --spring_ke 2e4 --spring_kd 20.0 --max_pressure 5.0 --substeps 32 --segments 3 3 3 --num_frames 800"
            ;;
        inflatable_table_sphere)
            # Inflatable table: 4 soft body spheres at corners supporting a rigid plate
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

echo "Using DISPLAY=$DISPLAY"
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
    echo "Using XAUTHORITY=$XAUTHORITY"
fi
echo ""

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

docker run --rm -it \
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
    python -m newton.examples "$EXAMPLE" $SOLVER_ARG $EXTRA_ARGS
