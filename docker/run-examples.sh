#!/bin/bash
# Newton Example Runner
#
# Discovers examples from `config/*.json` (each JSON file = one example) and
# runs the selected one inside the `newton:latest` Docker image.
#
# Delegates JSON parsing / parameter editing to `run-example.py`. See
# `config/README.md` for the file format.
#
# Usage:
#   ./run-examples.sh                              # Menu (pick one, run with JSON defaults)
#   ./run-examples.sh -e                           # Menu + edit parameters before running
#   ./run-examples.sh 1                            # Run item #1 from the menu
#   ./run-examples.sh basic_pendulum               # By name (uses JSON defaults)
#   ./run-examples.sh basic_pendulum -e            # By name + edit interactively
#   ./run-examples.sh basic_pendulum --set num-frames=500
#   ./run-examples.sh basic_pendulum -- --num-frames 500   # Raw Newton args after --
#   ./run-examples.sh basic_pendulum --profile     # Profile with Nsight Systems
#   ./run-examples.sh basic_pendulum --profile --profile-dir ~/my-profiles
#
# Interactive hints:
#   At the menu prompt, suffix your choice with 'e' to edit (e.g. '3e').
#
# Profiling:
#   --profile saves an .nsys-rep file to PROFILE_DIR (default: $HOME/newton-profiles).
#   Open results with: ./view-profile.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# docker/ lives at <repo>/docker; repo root is one level up
NEWTON_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$SCRIPT_DIR/config"
HELPER="$SCRIPT_DIR/run-example.py"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: config dir not found: $CONFIG_DIR" >&2
    exit 1
fi
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required on the host (to parse config/*.json)." >&2
    exit 1
fi

# ─── CLI parsing: pull out our own flags, stash raw args for after '--' ───────
EDIT=0
PROFILE=0
PROFILE_DIR="${PROFILE_DIR:-$SCRIPT_DIR/nsys}"
SETS=()       # --set KEY=VAL overrides (pre-'--')
RAW_ARGS=()   # anything after '--', passed verbatim to the example
POSITIONAL=() # example name/number and leftover args

seen_dashdash=0
while [ $# -gt 0 ]; do
    if [ "$seen_dashdash" -eq 1 ]; then
        RAW_ARGS+=("$1"); shift; continue
    fi
    case "$1" in
        --)              seen_dashdash=1 ;;
        -e|--edit)       EDIT=1 ;;
        --profile)       PROFILE=1 ;;
        --no-profile)    PROFILE=0 ;;
        --profile-dir)
            [ $# -lt 2 ] && { echo "error: --profile-dir expects a path" >&2; exit 2; }
            PROFILE_DIR="$2"; shift
            ;;
        --profile-dir=*) PROFILE_DIR="${1#--profile-dir=}" ;;
        --set)
            [ $# -lt 2 ] && { echo "error: --set expects KEY=VAL" >&2; exit 2; }
            SETS+=("$2"); shift
            ;;
        --set=*)   SETS+=("${1#--set=}") ;;
        -h|--help)
            sed -n '2,/^set -e$/p' "$0" | sed 's/^# \{0,1\}//;/^set -e$/d'
            exit 0
            ;;
        *)         POSITIONAL+=("$1") ;;
    esac
    shift
done
set -- "${POSITIONAL[@]}"

# ─── Load example list from config/*.json ────────────────────────────────────
mapfile -t LINES < <(python3 "$HELPER" list --config-dir "$CONFIG_DIR")
if [ "${#LINES[@]}" -eq 0 ]; then
    echo "Error: no JSON configs found in $CONFIG_DIR" >&2
    echo "       add one like config/<name>.json (see config/README.md)" >&2
    exit 1
fi

declare -a NAMES=()
declare -A DESCRIPTIONS=()
for line in "${LINES[@]}"; do
    name="${line%%$'\t'*}"
    desc="${line#*$'\t'}"
    [ "$name" = "$desc" ] && desc=""
    NAMES+=("$name")
    DESCRIPTIONS["$name"]="$desc"
done

in_names() {
    local needle="$1"
    for n in "${NAMES[@]}"; do [ "$n" = "$needle" ] && return 0; done
    return 1
}

# ─── Select example (interactive menu or CLI arg) ────────────────────────────
INTERACTIVE=0
if [ $# -eq 0 ]; then
    INTERACTIVE=1
    echo "═══════════════════════════════════════════════════════════════"
    echo "              Newton Examples - Select a Simulation"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    for i in "${!NAMES[@]}"; do
        name="${NAMES[$i]}"
        desc="${DESCRIPTIONS[$name]}"
        if [ -n "$desc" ]; then
            printf "  %3d) %-34s - %s\n" $((i+1)) "$name" "$desc"
        else
            printf "  %3d) %s\n" $((i+1)) "$name"
        fi
    done
    echo ""
    echo "Tip: append 'e' to your choice (e.g. 3e) to edit parameters before running."
    echo "═══════════════════════════════════════════════════════════════"
    # Default: basic_pendulum if present, else first
    DEFAULT_NUM=1
    for i in "${!NAMES[@]}"; do
        if [ "${NAMES[$i]}" = "basic_pendulum" ]; then DEFAULT_NUM=$((i+1)); break; fi
    done
    read -p "Select example (1-${#NAMES[@]}, Enter for ${DEFAULT_NUM}=${NAMES[$((DEFAULT_NUM-1))]}): " choice
    choice="${choice:-$DEFAULT_NUM}"
    # Accept "3" or "3e"
    if [[ "$choice" == *[eE] ]]; then
        EDIT=1
        choice="${choice%[eE]}"
    fi
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#NAMES[@]}" ]; then
        echo "Error: invalid choice." >&2
        exit 1
    fi
    EXAMPLE="${NAMES[$((choice-1))]}"
    echo ""
else
    sel="$1"; shift || true
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        if [ "$sel" -lt 1 ] || [ "$sel" -gt "${#NAMES[@]}" ]; then
            echo "Error: invalid example number: $sel" >&2
            exit 1
        fi
        EXAMPLE="${NAMES[$((sel-1))]}"
    else
        EXAMPLE="$sel"
        if ! in_names "$EXAMPLE"; then
            echo "Error: no config for '$EXAMPLE' in $CONFIG_DIR" >&2
            echo "Available:" >&2
            for n in "${NAMES[@]}"; do echo "  $n" >&2; done
            exit 1
        fi
    fi
    # Remaining POSITIONAL go as raw args too
    for extra in "$@"; do RAW_ARGS+=("$extra"); done
fi

# A config may delegate to a different example module via the optional
# top-level "example" field — lets multiple configs share one Python
# example (e.g. baymax_demo.json → inflatable_demo with --shape baymax).
INVOKE_EXAMPLE=$(python3 -c "import json; d=json.load(open('$CONFIG_DIR/$EXAMPLE.json')); print(d.get('example') or '$EXAMPLE')")

echo "Running: $EXAMPLE (module: $INVOKE_EXAMPLE)"
echo ""

# ─── Resolve CLI args from the JSON (optionally edit interactively) ──────────
RESOLVE_CMD=(python3 "$HELPER" resolve --config "$CONFIG_DIR/$EXAMPLE.json")
[ "$EDIT" -eq 1 ] && RESOLVE_CMD+=(--edit)
for s in "${SETS[@]}"; do RESOLVE_CMD+=(--set "$s"); done

RESOLVED=$("${RESOLVE_CMD[@]}")
# Raw args (after '--') go last and win per argparse's last-value rule.
RAW_STR=""
if [ "${#RAW_ARGS[@]}" -gt 0 ]; then
    RAW_STR=$(printf ' %q' "${RAW_ARGS[@]}")
fi
EXTRA_ARGS="$RESOLVED$RAW_STR"

# ─── Profile output directory ────────────────────────────────────────────────
PROFILE_MOUNT_ARGS=()
PROFILE_CMD_PREFIX=""
if [ "$PROFILE" -eq 1 ]; then
    mkdir -p "$PROFILE_DIR"
    REPORT_NAME="${EXAMPLE}_$(date +%Y%m%d_%H%M%S)"
    PROFILE_MOUNT_ARGS=(-v "$PROFILE_DIR:/profiles")
    PROFILE_CMD_PREFIX="nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        -o /profiles/${REPORT_NAME}"
    echo "Profiling enabled → $PROFILE_DIR/${REPORT_NAME}.nsys-rep"
    echo ""
fi

# ─── Auto-build image if missing ─────────────────────────────────────────────
if ! docker image inspect newton:latest >/dev/null 2>&1; then
    echo "Docker image 'newton:latest' not found. Building it now..."
    echo ""
    "$SCRIPT_DIR/build-docker.sh"
    echo ""
fi

# ─── GPU detection ───────────────────────────────────────────────────────────
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected - enabling GPU acceleration"
    GPU_ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all -e __GLX_VENDOR_LIBRARY_NAME=nvidia"
else
    echo "⚠ No NVIDIA GPU detected - running in CPU mode"
fi

# ─── X11 ─────────────────────────────────────────────────────────────────────
if [ -z "$DISPLAY" ]; then
    if   [ -e "/tmp/.X11-unix/X0" ]; then export DISPLAY=":0"
    elif [ -e "/tmp/.X11-unix/X1" ]; then export DISPLAY=":1"
    else                                 export DISPLAY=":0"
    fi
fi
echo "Using DISPLAY=$DISPLAY"
DOCKER_X11_ARGS=(
    -e "DISPLAY=$DISPLAY"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
)
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
    echo "Using XAUTHORITY=$XAUTHORITY"
    DOCKER_X11_ARGS+=(
        -e "XAUTHORITY=$XAUTHORITY"
        -v "$XAUTHORITY:$XAUTHORITY:ro"
    )
fi
echo ""

echo "+ python -m newton.examples $INVOKE_EXAMPLE $EXTRA_ARGS"
echo ""

# Mount the in-tree `newton` package over the image's copy so host edits are
# picked up without rebuilding.
# Persist the Warp kernel cache on the host so unchanged kernels skip
# recompilation across runs (Warp hashes kernel source and only rebuilds
# when it changes).
WARP_CACHE_DIR="${WARP_CACHE_DIR:-$HOME/.cache/newton-warp}"
NEWTON_CACHE_DIR="${NEWTON_CACHE_DIR:-$HOME/.cache/newton}"
mkdir -p "$WARP_CACHE_DIR" "$NEWTON_CACHE_DIR"

PROFILE_CAP=""
[ "$PROFILE" -eq 1 ] && PROFILE_CAP="--cap-add SYS_ADMIN"

docker run --rm -it \
    $GPU_ARGS \
    $PROFILE_CAP \
    --shm-size=16g \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${DOCKER_X11_ARGS[@]}" \
    "${PROFILE_MOUNT_ARGS[@]}" \
    -v "$NEWTON_DIR/newton:/workspace/newton/newton" \
    -v "$NEWTON_DIR/docker/config:/workspace/newton/docker/config" \
    -v "$WARP_CACHE_DIR:/root/.cache/warp" \
    -v "$NEWTON_CACHE_DIR:/root/.cache/newton" \
    newton:latest \
    bash -lc "rm -f /workspace/newton/.git && $PROFILE_CMD_PREFIX python -m newton.examples $INVOKE_EXAMPLE $EXTRA_ARGS"

if [ "$PROFILE" -eq 1 ]; then
    echo ""
    echo "Profile saved: $PROFILE_DIR/${REPORT_NAME}.nsys-rep"
    echo "Open with:     $(dirname "$SCRIPT_DIR")/docker/view-profile.sh"
fi
