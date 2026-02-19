#!/bin/bash
# Newton Example Runner
# Usage: ./run-examples.sh [example_name|number] [extra_args...]
# Examples:
#   ./run-examples.sh                                  # Interactive menu
#   ./run-examples.sh 1                                # Run example 1
#   ./run-examples.sh bouncing_sphere
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
    "bouncing_box"
    "rigid_soft_interaction"
    "soft_on_box"
    "inflatable_box"
    "inflatable_box_sand"
    "inflatable_sphere"
    "inflatable_rigid_box"
    "inflatable_rigid_sphere"
    "chambers"
    "inflatable_glue"
    "disabled_chambers"
    "worm"
    "inchworm"
    "inchworm_crawling"
    "inchworm_crawling_sand"
    "mpm_worm_sand"
    "rigid_carpet"
    "inflatable_table_glue"
)

declare -A EXAMPLE_DESCRIPTIONS=(
    ["bouncing_sphere"]="Soft sphere bouncing on the ground"
    ["bouncing_box"]="Soft box bouncing and tumbling"
    ["rigid_soft_interaction"]="Rigid-soft interaction (supports --solver xpbd|mujoco)"
    ["soft_on_box"]="Soft object sitting on rigid XPBD box (supports --constraint-contacts)"
    ["inflatable_box"]="Inflatable soft body box with pressure control (Press I/K/O)"
    ["inflatable_box_sand"]="Inflatable box on MPM sand (two-way coupling; Press I/K/O)"
    ["inflatable_sphere"]="Inflatable soft body sphere with pressure control (Press I/K/O)"
    ["inflatable_rigid_box"]="Inflatable box with rigid plate on top (supports --solver xpbd|mujoco)"
    ["inflatable_rigid_sphere"]="Inflatable sphere with rigid plate on top (supports --solver xpbd|mujoco)"
    ["chambers"]="N-chamber anisotropic inflatable box (Press I/K inflate/deflate, C chamber, N num chambers, A anisotropy)"
    ["inflatable_glue"]="Rigid + Inflatable glued by proximity springs (--top soft|rigid; Press I/K/O, G/F glue)"
    ["disabled_chambers"]="Same as chambers (N-chamber inflatable); Press I/K inflate/deflate, C cycle chamber"
    ["worm"]="Worm with torque (bend about one axis); interactive: stiff_axes, torque, length/width/height, subdivisions; [I]/[K] [C]"
    ["inchworm"]="Inchworm (arXiv:1911.05227): left/right chambers, phase-shifted harmonic gait; --no_gait for manual [I]/[K] [C]"
    ["inchworm_crawling"]="Inchworm crawling (SolverCrawlable): same as inchworm; menu asks Normal vs Debug (stick-slip)"
    ["inchworm_crawling_sand"]="Inchworm crawling on MPM sand (two-way coupling; --no-sand = same as inchworm_crawling)"
    ["mpm_worm_sand"]="Worm-shaped rigid body on MPM sand (two-way coupling; sand deforms, worm gets reaction forces)"
    ["rigid_carpet"]="DEBUG: Rigid plates on ground + glue (minimal, no soft body)"
    ["inflatable_table_glue"]="Table: 4 soft legs glued to rigid plate (SurfaceBox); Press I/K/O, G/F"
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
if [ "$EXAMPLE" == "rigid_carpet" ]; then
    echo "  (To add params, pass them after the example: ./run-examples.sh rigid_carpet --glue_ke_rr 1e5 --glue_kd_rr 500)"
fi
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
    
    read -p "Subdivisions per axis - width height depth (default: 3 5 7): " box_subdivisions
    if [ -z "$box_subdivisions" ]; then
        box_subdivisions="3 5 7"
    fi
    
    echo ""
    echo "Using box size: $box_size, subdivisions: $box_subdivisions"
    echo ""
    
    # Use default values for other parameters
    BOX_ARGS="--size $box_size --initial_height 2.5 --mass 0.5 --k_mu 2e5 --k_lambda 2e5 --k_damp 0.5 --substeps 8 --subdivisions $box_subdivisions --num_frames 600"
fi

# Interactive parameter selection for inflatable_glue (position + all params)
GLUE_ARGS=""
if [ "$EXAMPLE" == "inflatable_glue" ] && [ -z "$*" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Inflatable Glue - Parameters"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Which object on top?"
    echo "  1) soft    - Inflatable soft body on top (default)"
    echo "  2) rigid   - Rigid box on top"
    echo ""
    read -p "Select (1-2, Enter for soft): " glue_top_choice
    case "$glue_top_choice" in
        2)
            glue_top_arg="--top rigid"
            echo "Selected: rigid on top"
            ;;
        *)
            glue_top_arg="--top soft"
            echo "Selected: soft on top"
            ;;
    esac
    echo ""
    echo "Mesh dimensions (width=X, height=Y, depth=Z in meters):"
    read -p "  Width (X) in m (default: 1): " glue_width
    if [ -z "$glue_width" ]; then glue_width=1; fi
    read -p "  Height (Y) in m (default: 1): " glue_height
    if [ -z "$glue_height" ]; then glue_height=1; fi
    read -p "  Depth (Z) in m (default: 1): " glue_depth
    if [ -z "$glue_depth" ]; then glue_depth=1; fi
    glue_size="$glue_width $glue_height $glue_depth"

    read -p "Subdivisions X Y Z (default: 5 5 5): " glue_subdivisions
    if [ -z "$glue_subdivisions" ]; then glue_subdivisions="5 5 5"; fi

    # Default positions (stacked) from dimensions and top choice
    if [ "$glue_top_arg" = "--top rigid" ]; then
        rigid_z_def=$(awk "BEGIN {printf \"%.3f\", $glue_depth + $glue_depth/2}")
        inflatable_z_def=$(awk "BEGIN {printf \"%.3f\", $glue_depth/2}")
    else
        rigid_z_def=$(awk "BEGIN {printf \"%.3f\", $glue_depth/2}")
        inflatable_z_def=$(awk "BEGIN {printf \"%.3f\", $glue_depth + $glue_depth/2}")
    fi

    echo ""
    echo "Positions (x y z in meters, center of each box):"
    read -p "  Rigid position (default: 0 0 $rigid_z_def): " glue_rigid_pos
    if [ -z "$glue_rigid_pos" ]; then glue_rigid_pos="0 0 $rigid_z_def"; fi
    read -p "  Inflatable position (default: 0 0 $inflatable_z_def): " glue_inflatable_pos
    if [ -z "$glue_inflatable_pos" ]; then glue_inflatable_pos="0 0 $inflatable_z_def"; fi
    glue_pos_args="--rigid_pos $glue_rigid_pos --inflatable_pos $glue_inflatable_pos"

    echo ""
    read -p "Mass (default: 1.0): " glue_mass
    if [ -z "$glue_mass" ]; then glue_mass=1.0; fi
    read -p "Rigid mass (default: 0.2): " glue_rigid_mass
    if [ -z "$glue_rigid_mass" ]; then glue_rigid_mass=0.2; fi
    read -p "Glue epsilon in m (default: 0.05): " glue_epsilon
    if [ -z "$glue_epsilon" ]; then glue_epsilon=0.05; fi
    read -p "Glue stiffness ke (default: 5e4): " glue_ke
    if [ -z "$glue_ke" ]; then glue_ke=5e4; fi
    read -p "Glue damping kd (default: 200): " glue_kd
    if [ -z "$glue_kd" ]; then glue_kd=200; fi
    read -p "Max pressure (default: 5.0): " glue_max_pressure
    if [ -z "$glue_max_pressure" ]; then glue_max_pressure=5.0; fi
    read -p "Substeps (default: 8): " glue_substeps
    if [ -z "$glue_substeps" ]; then glue_substeps=8; fi
    read -p "XPBD iterations (default: 10): " glue_xpbd_iter
    if [ -z "$glue_xpbd_iter" ]; then glue_xpbd_iter=10; fi
    read -p "Num frames (default: 1800): " glue_num_frames
    if [ -z "$glue_num_frames" ]; then glue_num_frames=1800; fi

    echo ""
    echo "Gravity: 0 = disabled, 9.81 = Earth default"
    read -p "Gravity in m/s² (default: 9.81, 0 to disable): " glue_gravity
    if [ -z "$glue_gravity" ]; then glue_gravity=9.81; fi

    echo ""
    echo "Using size=$glue_size (w=$glue_width h=$glue_height d=$glue_depth) subdivisions=$glue_subdivisions"
    echo "  rigid_pos=$glue_rigid_pos  inflatable_pos=$glue_inflatable_pos"
    echo "  mass=$glue_mass rigid_mass=$glue_rigid_mass glue_epsilon=$glue_epsilon"
    echo "  glue_ke=$glue_ke glue_kd=$glue_kd max_pressure=$glue_max_pressure"
    echo "  gravity=$glue_gravity substeps=$glue_substeps xpbd_iterations=$glue_xpbd_iter num_frames=$glue_num_frames"
    echo ""
    GLUE_ARGS="--size $glue_size --subdivisions $glue_subdivisions $glue_top_arg $glue_pos_args --mass $glue_mass --rigid_mass $glue_rigid_mass --glue_epsilon $glue_epsilon --glue_ke $glue_ke --glue_kd $glue_kd --max_pressure $glue_max_pressure --gravity $glue_gravity --substeps $glue_substeps --xpbd_iterations $glue_xpbd_iter --num_frames $glue_num_frames"
fi

# Interactive parameter selection for soft_on_box (constraint vs force-based contacts)
SOFT_ON_BOX_ARGS=""
if [ "$EXAMPLE" == "soft_on_box" ] && [ -z "$*" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              soft_on_box - Contact mode"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  Constraint-based: position corrections after integration (no penetration)"
    echo "  Force-based:     penalty forces only (may sink into box slightly)"
    echo ""
    read -p "Use constraint-based contacts? (y/n, default y): " soft_contact_choice
    if [ "$soft_contact_choice" = "n" ] || [ "$soft_contact_choice" = "N" ]; then
        SOFT_ON_BOX_ARGS="--no-constraint-contacts"
        echo "Selected: force-based contacts"
    else
        SOFT_ON_BOX_ARGS="--constraint-contacts"
        echo "Selected: constraint-based contacts"
    fi
    echo ""
fi

# Interactive parameter selection for chambers and disabled_chambers (same parameters)
CHAMBERS_ARGS=""
if { [ "$EXAMPLE" == "chambers" ] || [ "$EXAMPLE" == "disabled_chambers" ]; } && [ -z "$*" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              $EXAMPLE - Parameters (chambers per axis)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    read -p "Chambers along X (default: 1): " num_chambers_x
    if [ -z "$num_chambers_x" ]; then num_chambers_x=1; fi
    read -p "Chambers along Y (default: 2): " num_chambers_y
    if [ -z "$num_chambers_y" ]; then num_chambers_y=2; fi
    if [ "$EXAMPLE" == "disabled_chambers" ]; then
        read -p "Chambers along Z (default: 2): " num_chambers_z
        if [ -z "$num_chambers_z" ]; then num_chambers_z=2; fi
    else
        read -p "Chambers along Z (default: 1): " num_chambers_z
        if [ -z "$num_chambers_z" ]; then num_chambers_z=1; fi
    fi
    read -p "Length (X) in m (default: 1): " length
    if [ -z "$length" ]; then
        length=1
    fi
    read -p "Width (Y) in m (default: 2): " width
    if [ -z "$width" ]; then
        width=2
    fi
    if [ "$EXAMPLE" == "disabled_chambers" ]; then
        read -p "Height (Z) in m (default: 0.1): " height
        if [ -z "$height" ]; then height=0.1; fi
    else
        read -p "Height (Z) in m (default: 0.06): " height
        if [ -z "$height" ]; then height=0.06; fi
    fi
    read -p "Subdivisions X (default: 10): " subdivisions_x
    if [ -z "$subdivisions_x" ]; then
        subdivisions_x=10
    fi
    read -p "Subdivisions Y (default: 30): " subdivisions_y
    if [ -z "$subdivisions_y" ]; then
        subdivisions_y=30
    fi
    if [ "$EXAMPLE" == "disabled_chambers" ]; then
        read -p "Subdivisions Z (default: 4): " subdivisions_z
        if [ -z "$subdivisions_z" ]; then subdivisions_z=4; fi
    else
        read -p "Subdivisions Z (default: 2): " subdivisions_z
        if [ -z "$subdivisions_z" ]; then subdivisions_z=2; fi
    fi
    read -p "Anisotropy X (default: 1.0): " anisotropy_x
    if [ -z "$anisotropy_x" ]; then
        anisotropy_x=1.0
    fi
    if [ "$EXAMPLE" == "disabled_chambers" ]; then
        read -p "Anisotropy Y for worm movements (default: 1.4): " anisotropy_y
        if [ -z "$anisotropy_y" ]; then
            anisotropy_y=1.4
        fi
    else
        read -p "Anisotropy Y (default: 1.0): " anisotropy_y
        if [ -z "$anisotropy_y" ]; then
            anisotropy_y=1.0
        fi
    fi
    read -p "Anisotropy Z (default: 1.0): " anisotropy_z
    if [ -z "$anisotropy_z" ]; then
        anisotropy_z=1.0
    fi
    chamber_stiffness_scale_arg=""
    chamber_inflation_disabled_arg=""
    if { [ "$EXAMPLE" == "chambers" ] || [ "$EXAMPLE" == "disabled_chambers" ]; }; then
        total_chambers=$((num_chambers_x * num_chambers_y * num_chambers_z))
        echo "Chamber layout (index = ix*(ny*nz)+iy*nz+iz, ${num_chambers_x}x${num_chambers_y}x${num_chambers_z} = $total_chambers chambers):"
        for iz in $(seq 0 $((num_chambers_z - 1))); do
            echo "  Z=$iz (layer):"
            for ix in $(seq 0 $((num_chambers_x - 1))); do
                line="    "
                for iy in $(seq 0 $((num_chambers_y - 1))); do
                    ch=$((ix * num_chambers_y * num_chambers_z + iy * num_chambers_z + iz))
                    line="${line}[$ch] "
                done
                echo "$line"
            done
        done
        echo ""
        default_stiffness=""
        for ((i=0; i<total_chambers; i++)); do
            if [ "$i" -eq 1 ] || [ "$i" -eq $((total_chambers - 1)) ]; then
                default_stiffness="${default_stiffness}1,"
            else
                default_stiffness="${default_stiffness}4,"
            fi
        done
        default_stiffness="${default_stiffness%,}"
        read -p "Stiffness scale array ($total_chambers values, comma-separated). Enter for default ($default_stiffness): " chamber_stiffness_scale_input
        if [ -n "$chamber_stiffness_scale_input" ]; then
            chamber_stiffness_scale_arg="--chamber_stiffness_scale $chamber_stiffness_scale_input"
        fi
        default_inflation=""
        inflate_idx=$((total_chambers - 1))
        for ((i=0; i<total_chambers; i++)); do
            if [ "$i" -eq 1 ] || [ "$i" -eq "$inflate_idx" ]; then
                default_inflation="${default_inflation}1,"
            else
                default_inflation="${default_inflation}0,"
            fi
        done
        default_inflation="${default_inflation%,}"
        default_disabled_list=""
        for ((i=0; i<total_chambers; i++)); do
            if [ "$i" -ne 1 ] && [ "$i" -ne "$inflate_idx" ]; then
                default_disabled_list="${default_disabled_list}${i},"
            fi
        done
        default_disabled_list="${default_disabled_list%,}"
        read -p "Inflation array ($total_chambers values: 0=inactive 1=active, comma-separated). Enter for default ($default_inflation), or 'none' for all inflatable: " chamber_inflation_input
        if [ -z "$chamber_inflation_input" ]; then
            chamber_inflation_disabled_arg="--chamber_inflation_disabled $default_disabled_list"
            chamber_inflation_disabled_input="$default_disabled_list"
        elif [ "$chamber_inflation_input" = "none" ] || [ "$chamber_inflation_input" = "all" ]; then
            chamber_inflation_disabled_arg="--chamber_inflation_disabled none"
            chamber_inflation_disabled_input="none"
        else
            chamber_inflation_disabled_list=""
            idx=0
            IFS=',' read -ra INVALS <<< "$chamber_inflation_input"
            for val in "${INVALS[@]}"; do
                val=$(echo "$val" | tr -d ' ')
                if [ "$val" = "0" ]; then
                    chamber_inflation_disabled_list="${chamber_inflation_disabled_list}${idx},"
                fi
                idx=$((idx + 1))
            done
            chamber_inflation_disabled_list="${chamber_inflation_disabled_list%,}"
            if [ -n "$chamber_inflation_disabled_list" ]; then
                chamber_inflation_disabled_arg="--chamber_inflation_disabled $chamber_inflation_disabled_list"
                chamber_inflation_disabled_input="$chamber_inflation_disabled_list"
            else
                chamber_inflation_disabled_arg="--chamber_inflation_disabled none"
                chamber_inflation_disabled_input="none"
            fi
        fi
    fi
    echo ""
    echo "Using: chambers X=$num_chambers_x Y=$num_chambers_y Z=$num_chambers_z length=$length width=$width height=$height subdivisions_x=$subdivisions_x subdivisions_y=$subdivisions_y subdivisions_z=$subdivisions_z anisotropy_x=$anisotropy_x anisotropy_y=$anisotropy_y anisotropy_z=$anisotropy_z"
    if [ -n "$chamber_inflation_disabled_arg" ]; then
        echo "  inflatable chambers: disabled=${chamber_inflation_disabled_input:-0,2}"
    fi
    echo ""
    CHAMBERS_ARGS="--num_chambers_x $num_chambers_x --num_chambers_y $num_chambers_y --num_chambers_z $num_chambers_z --length $length --width $width --height $height --subdivisions_x $subdivisions_x --subdivisions_y $subdivisions_y --subdivisions_z $subdivisions_z --anisotropy_x $anisotropy_x --anisotropy_y $anisotropy_y --anisotropy_z $anisotropy_z --initial_height 0.3 --k_mu 1e5 --k_lambda 1e5 --max_pressure 5.0 --substeps 5 --num_frames 14400"
    if [ "$EXAMPLE" == "disabled_chambers" ]; then
        if [ -n "$chamber_stiffness_scale_arg" ]; then
            CHAMBERS_ARGS="$CHAMBERS_ARGS $chamber_stiffness_scale_arg"
        fi
        if [ -n "$chamber_inflation_disabled_arg" ]; then
            CHAMBERS_ARGS="$CHAMBERS_ARGS $chamber_inflation_disabled_arg"
        fi
    fi
fi

# Interactive parameter selection for worm (single axis stays straight, whole object)
TORQUE_WORM_ARGS=""
if [ "$EXAMPLE" == "worm" ] && [ -z "$*" ] && [ -t 0 ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Worm - Which axis stays straight?"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "  One axis gets torque on the whole object (all chambers); that axis stays straight."
    echo "  1) X stays straight (bend about Y, Z)"
    echo "  2) Y stays straight (bend about X, Z)"
    echo "  3) Z stays straight (bend about X, Y)"
    echo ""
    read -p "  Choice (1-3) [1]: " stiff_choice
    stiff_choice=${stiff_choice:-1}
    case "$stiff_choice" in
        1) stiff_axes="x" ;;
        2) stiff_axes="y" ;;
        3) stiff_axes="z" ;;
        *) stiff_axes="x" ;;
    esac
    read -p "  Torque stiffness (default: 100): " torque_stiffness
    torque_stiffness=${torque_stiffness:-100}
    read -p "  Torque damping (default: 2): " torque_damping
    torque_damping=${torque_damping:-2}
    read -p "  Ground friction mu (0=slippery, 0.8=default for stick-slip, ~1=grip) [0.8]: " ground_friction
    ground_friction=${ground_friction:-0.8}
    read -p "  Chamber inflation disabled, comma-separated (default: 0,2): " chamber_disabled
    chamber_disabled=${chamber_disabled:-0,2}
    echo ""
    echo "  Worm geometry (size and subdivisions):"
    read -p "    Length X (m) [1]: " worm_length
    worm_length=${worm_length:-1}
    read -p "    Width Y (m) [1.5]: " worm_width
    worm_width=${worm_width:-1.5}
    read -p "    Height Z (m) [0.1]: " worm_height
    worm_height=${worm_height:-0.1}
    read -p "    Subdivisions X [10]: " worm_sub_x
    worm_sub_x=${worm_sub_x:-10}
    read -p "    Subdivisions Y [15]: " worm_sub_y
    worm_sub_y=${worm_sub_y:-15}
    read -p "    Subdivisions Z [4]: " worm_sub_z
    worm_sub_z=${worm_sub_z:-4}
    echo "  → stiff_axes=$stiff_axes, ground_friction=$ground_friction, geometry ${worm_length}x${worm_width}x${worm_height}, sub ${worm_sub_x}x${worm_sub_y}x${worm_sub_z}"
    echo ""
    TORQUE_WORM_ARGS="--stiff_axes $stiff_axes --torque_stiffness $torque_stiffness --torque_damping $torque_damping --ground_friction $ground_friction --chamber_inflation_disabled $chamber_disabled --length $worm_length --width $worm_width --height $worm_height --subdivisions_x $worm_sub_x --subdivisions_y $worm_sub_y --subdivisions_z $worm_sub_z --num_chambers_x 1 --num_chambers_y 2 --num_chambers_z 2 --initial_height 0.3 --mass 1.0 --k_mu 1e5 --k_lambda 1e5 --k_damp 1.0 --spring_ke 5e4 --spring_kd 1.0 --gravity 9.81 --max_pressure 5.0 --anisotropy_x 1.2 --anisotropy_y 1.4 --anisotropy_z 1.2 --substeps 5 --num_frames 14400"
fi

# Inchworm crawling: use_crawlable_stick_slip comes from JSON (--params); no prompt so JSON wins.
# To override from CLI, pass --normal or --stick-slip after the example name.
CRAWL_EXTRA=""

# Inchworm crawling on sand: optional --no-sand (same as inchworm_crawling)
CRAWL_SAND_EXTRA=""
if [ "$EXAMPLE" == "inchworm_crawling_sand" ] && [ -z "$*" ] && [ -t 0 ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Inchworm Crawling Sand - Options"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    read -p "Enable MPM sand (two-way coupling)? [Y/n]: " crawl_sand_choice
    if [ "$crawl_sand_choice" = "n" ] || [ "$crawl_sand_choice" = "N" ]; then
        CRAWL_SAND_EXTRA="--no-sand"
        echo "  → Sand disabled (same as inchworm_crawling)"
    else
        echo "  → Sand enabled"
    fi
    echo ""
fi

# Inflatable box on sand: option to disable sand (box only)
INFLATABLE_BOX_SAND_ARGS=""
if [ "$EXAMPLE" == "inflatable_box_sand" ] && [ -z "$*" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Inflatable Box Sand - Options"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    read -p "Enable MPM sand (two-way coupling)? [Y/n]: " sand_choice
    if [ "$sand_choice" = "n" ] || [ "$sand_choice" = "N" ]; then
        INFLATABLE_BOX_SAND_ARGS="--no-sand"
        echo "  → Sand disabled (inflatable box only)"
    else
        echo "  → Sand enabled"
    fi
    echo ""
fi

# Default stable parameters for examples that need them
# User-passed args ($*) are merged with defaults so you can override individual params
DEFAULT_ARGS=""
case "$EXAMPLE" in
        bouncing_sphere)
            # Stable and fast: fewer substeps, coarser mesh
            DEFAULT_ARGS="--radius 0.3 --initial_height 0.9 --k_mu 4e4 --k_lambda 4e4 --k_damp 3.0 --substeps 10 --subdivisions 1 --interior_layers 1"
            ;;
        bouncing_box)
            # Use interactive selection if available, otherwise use default
            if [ -n "$BOX_ARGS" ]; then
                DEFAULT_ARGS="$BOX_ARGS"
            else
                # Box bouncing and tumbling - subdivided into multiple cubic elements
                # Larger non-uniform box: size=(0.6, 0.8, 1.0), subdivisions=(3, 5, 7)
                # Creates 105 cubic elements (630 tetrahedra) with non-uniform subdivision
                # Each cube uses stable 6-tet pattern (all share v6 corner vertex)
                DEFAULT_ARGS="--size 0.6 0.8 1.0 --initial_height 2.5 --mass 0.5 --k_mu 2e5 --k_lambda 2e5 --k_damp 0.5 --substeps 8 --subdivisions 3 5 7 --num_frames 600"
            fi
            ;;
        rigid_soft_interaction)
            # Rigid ball drops onto soft ball (SolverSoft - pure FEM, no pressure control)
            # Solver will be selected interactively or via --solver xpbd|mujoco
            DEFAULT_ARGS="--ball-radius 0.3 --drop-height 1.5 --substeps 16"
            ;;
        soft_on_box)
            # Soft sphere sitting on top of rigid XPBD box
            # SOFT_ON_BOX_ARGS set by interactive prompt (constraint vs force-based), or pass --constraint-contacts / --no-constraint-contacts
            if [ -n "$SOFT_ON_BOX_ARGS" ]; then
                DEFAULT_ARGS="$SOFT_ON_BOX_ARGS"
            else
                DEFAULT_ARGS="--constraint-contacts"
            fi
            ;;
        inflatable_box)
            # Inflatable soft body box with manual pressure control
            # Press I to inflate, K to deflate, O to reset
            # Using 5x5x5 subdivisions for more particles (125 cells = 750 tetrahedra)
            DEFAULT_ARGS="--size 0.4 0.4 0.4 --initial_height 0.5 --k_mu 1e5 --k_lambda 1e5 --k_damp 1.0 --max_pressure 5.0 --substeps 5 --subdivisions 5 5 5 --num_frames 1800"
            ;;
        inflatable_box_sand)
            # Inflatable box on MPM sand; use --no-sand for box only (set via INFLATABLE_BOX_SAND_ARGS)
            DEFAULT_ARGS="$INFLATABLE_BOX_SAND_ARGS"
            ;;
        inflatable_sphere)
            # Inflatable soft body sphere with manual pressure control
            # Press I to inflate, K to deflate, O to reset
            DEFAULT_ARGS="--radius 0.3 --initial_height 0.4 --k_mu 5e4 --k_lambda 5e4 --k_damp 2.0 --max_pressure 5.0 --substeps 8 --subdivisions 2 --interior_layers 2 --num_frames 1800"
            ;;
        inflatable_rigid_box)
            # Inflatable soft body box with large flat rigid plate on top
            # Plate is 5x the soft body size (3.0m for 0.3m box)
            # Note: Using heavier mass (0.01kg) and smaller particle radius (0.015m) for better MuJoCo interaction
            DEFAULT_ARGS="--size 0.3 0.3 0.3 --rigid_width 3.0 --rigid_mass 0.01 --particle_radius 0.015 --k_mu 1e5 --k_lambda 1e5 --k_damp 5.0 --spring_ke 5e4 --spring_kd 5.0 --max_pressure 5.0 --substeps 16 --subdivisions 3 3 3 --num_frames 800"
            ;;
        inflatable_rigid_sphere)
            # Inflatable soft body sphere with large flat rigid plate on top
            # Plate is 5x the soft body diameter (3.0m for 0.3m radius soft)
            # Note: Using heavier mass (0.01kg) and smaller particle radius (0.015m) for better MuJoCo interaction
            DEFAULT_ARGS="--radius 0.3 --rigid_width 3.0 --rigid_mass 0.01 --particle_radius 0.015 --k_mu 1e5 --k_lambda 1e5 --k_damp 5.0 --spring_ke 5e4 --spring_kd 5.0 --max_pressure 5.0 --substeps 16 --subdivisions 2 --interior_layers 2 --num_frames 800"
            ;;
        inflatable_glue)
            # Stacked by default (rigid bottom, inflatable on top); use GLUE_ARGS if interactive selection ran
            if [ -n "$GLUE_ARGS" ]; then
                DEFAULT_ARGS="$GLUE_ARGS"
            else
                DEFAULT_ARGS="--size 0.4 0.4 0.4 --subdivisions 5 5 5 --rigid_z_base 0 --stack_gap 0 --mass 1.0 --rigid_mass 0.2 --glue_epsilon 0.05 --glue_ke 5e4 --glue_kd 200 --max_pressure 5.0 --gravity 9.81 --substeps 8 --xpbd_iterations 10 --num_frames 1800"
            fi
            ;;
        inflatable_table_glue)
            # Table: 4 soft legs (same size as inflatable_glue), plate not too light or it explodes
            DEFAULT_ARGS="--leg_size 0.4 0.4 0.4 --subdivisions 5 5 5 --plate_width 2.0 --plate_height 0.05 --mass 1.0 --plate_mass 0.05 --rigid_leg_mass 10000000.0 --particle_radius 0.008 --glue_epsilon 0.05 --glue_ke 3e4 --glue_kd 300 --glue_max_vel 1.0 --max_pressure 5.0 --gravity 9.81 --substeps 8 --xpbd_iterations 12 --num_frames 1800"
            ;;
        worm)
            # Single axis stays straight (X, Y, or Z); torque on whole object. Pass --ground_friction 0.1 for slippery.
            if [ -n "$TORQUE_WORM_ARGS" ]; then
                DEFAULT_ARGS="$TORQUE_WORM_ARGS"
            else
                DEFAULT_ARGS="--stiff_axes x --ground_friction 0.8 --length 1 --width 1.5 --height 0.1 --subdivisions_x 10 --subdivisions_y 15 --subdivisions_z 4 --num_chambers_x 1 --num_chambers_y 2 --num_chambers_z 2 --initial_height 0.3 --mass 1.0 --k_mu 1e5 --k_lambda 1e5 --k_damp 1.0 --spring_ke 5e4 --spring_kd 1.0 --gravity 9.81 --max_pressure 5.0 --anisotropy_x 1.2 --anisotropy_y 1.4 --anisotropy_z 1.2 --torque_stiffness 100 --torque_damping 2 --chamber_inflation_disabled 0,2 --substeps 5 --num_frames 14400"
            fi
            ;;
        inchworm)
            # Inchworm: start on ground (no --initial_height: default 0 = bottom at z=0), then settle_seconds before gait.
            # --csv_log_dir so CSV lands in newton/inchworm/ when run via Docker.
            DEFAULT_ARGS="--csv_log_dir /workspace/inchworm --csv_log_interval 10 --ground_friction 1.0 --length 1 --width 1.5 --height 0.1 --subdivisions_x 10 --subdivisions_y 15 --subdivisions_z 4 --num_chambers_x 1 --num_chambers_y 2 --num_chambers_z 2 --mass 1.0 --k_mu 1e5 --k_lambda 1e5 --k_damp 1.0 --spring_ke 5e4 --spring_kd 1.0 --gravity 9.81 --max_pressure 5.0 --anisotropy_x 1.2 --anisotropy_y 1.5 --anisotropy_z 1.4 --torque_stiffness 100 --torque_damping 2 --chamber_inflation_disabled 0,2 --substeps 5 --num_frames 14400"
            ;;
        inchworm_crawling)
            # Params from JSON (newton/inchworm/inchworm_params.json). CRAWL_EXTRA adds --normal or --stick-slip from menu.
            DEFAULT_ARGS="--params /workspace/inchworm/inchworm_params.json --csv_log_dir /workspace/inchworm --csv_log_interval 10"
            ;;
        inchworm_crawling_sand)
            # Same params as inchworm_crawling; CRAWL_EXTRA = --normal/--stick-slip, CRAWL_SAND_EXTRA = optional --no-sand.
            DEFAULT_ARGS="--params /workspace/inchworm/inchworm_params.json --csv_log_dir /workspace/inchworm --csv_log_interval 10 ${CRAWL_SAND_EXTRA:-}"
            ;;
        chambers)
            # Flat rectangular slab, 2 chambers side-by-side (bends with differential pressure)
            if [ -n "$CHAMBERS_ARGS" ]; then
                DEFAULT_ARGS="$CHAMBERS_ARGS"
            else
                DEFAULT_ARGS="--length 1 --width 2 --height 0.06 --subdivisions_x 10 --subdivisions_y 30 --subdivisions_z 2 --num_chambers_y 2 --pos 0 0 0.3 --anisotropy_x 1.2 --anisotropy_z 1.2 --initial_height 0.3 --k_mu 1e5 --k_lambda 1e5 --max_pressure 5.0 --substeps 5 --num_frames 14400"
            fi
            ;;
        disabled_chambers)
            # Disabled chambers: same as chambers but default height 0.1, subdivisions_z 4
            if [ -n "$CHAMBERS_ARGS" ]; then
                DEFAULT_ARGS="$CHAMBERS_ARGS"
            else
                DEFAULT_ARGS="--length 1 --width 2 --height 0.1 --subdivisions_x 10 --subdivisions_y 30 --subdivisions_z 4 --num_chambers_y 2 --num_chambers_z 2 --pos 0 0 0.3 --chamber_inflation_disabled 0,2 --anisotropy_x 1.2 --anisotropy_z 1.2 --initial_height 0.3 --k_mu 1e5 --k_lambda 1e5 --max_pressure 5.0 --substeps 5 --num_frames 14400"
            fi
            ;;
        rigid_carpet)
            # Override any param: ./run-examples.sh rigid_carpet --length 2 --rigid_height 0.2 --subdivisions_x 8
            DEFAULT_ARGS="--glue_axis y --length 1.0 --width 2 --rigid_subdivision 30 --subdivisions_x 12 --subdivisions_z 1 --rigid_height 0.01 --rigid_mass 0.005 --glue_epsilon 0.02 --glue_ke_rr 6e3 --glue_kd_rr 80 --substeps 5 --xpbd_iterations 10 --num_frames 3600 --drop_height 0.3"
            if [ -z "$*" ] && [ -t 0 ]; then
                echo "═══════════════════════════════════════════════════════════════"
                echo "              Rigid Carpet - Parameters (Enter = default)"
                echo "═══════════════════════════════════════════════════════════════"
                echo ""
                read -p "  Glue axis: y=Y+↔Y- (along Y), x=X+↔X-, z=Z+↔Z- [y]: " rc_axis
                read -p "  Length (X) in m [1.0]: " rc_length
                read -p "  Width (Y) in m [2]: " rc_width
                read -p "  Rigid height (Z) in m [0.01]: " rc_height
                read -p "  Number of rigid bodies [30]: " rc_plates
                echo "  (In-plane: axis y = X,Z; axis x = Y,Z; axis z = X,Y)"
                read -p "  Subdivisions X (in-plane 1) [12]: " rc_subx
                read -p "  Subdivision within each rigid (Y axis) [1]: " rc_suby
                read -p "  Subdivisions Z (in-plane 2) [1]: " rc_subz
                read -p "  Drop height in m (0=on ground) [0.3]: " rc_drop
                read -p "  Glue stiffness ke_rr (N/m) [6e3]: " rc_glue_ke_rr
                read -p "  Glue damping kd_rr [80]: " rc_glue_kd_rr
                echo ""
                rc_axis=${rc_axis:-y}
                rc_length=${rc_length:-1.0}
                rc_width=${rc_width:-2}
                rc_height=${rc_height:-0.01}
                rc_plates=${rc_plates:-30}
                rc_subx=${rc_subx:-12}
                rc_suby=${rc_suby:-1}
                rc_subz=${rc_subz:-1}
                rc_drop=${rc_drop:-0.3}
                rc_glue_ke_rr=${rc_glue_ke_rr:-6e3}
                rc_glue_kd_rr=${rc_glue_kd_rr:-80}
                DEFAULT_ARGS="--glue_axis $rc_axis --length $rc_length --width $rc_width --rigid_subdivision $rc_plates --subdivisions_x $rc_subx --subdivisions_y $rc_suby --subdivisions_z $rc_subz --rigid_height $rc_height --rigid_mass 0.005 --glue_epsilon 0.02 --glue_ke_rr $rc_glue_ke_rr --glue_kd_rr $rc_glue_kd_rr --substeps 5 --xpbd_iterations 10 --num_frames 3600 --drop_height $rc_drop"
            fi
            ;;
        *)
            DEFAULT_ARGS=""
            ;;
    esac

# Merge defaults with user overrides (user args go last, so they override)
EXTRA_ARGS="$DEFAULT_ARGS ${CRAWL_EXTRA:-} $*"

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
    -e PYTHONPATH=/workspace \
    "${DOCKER_X11_ARGS[@]}" \
    -v "$NEWTON_DIR/newton:/workspace/newton/newton" \
    -v "$NEWTON_DIR/newton/examples/crawlable/inchworm:/workspace/inchworm" \
    newton:latest \
    python -m newton.examples "$EXAMPLE" $SOLVER_ARG $EXTRA_ARGS
