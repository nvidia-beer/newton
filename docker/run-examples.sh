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
#   ./run-examples.sh basic_pendulum --set num-frames=500   (flags go AFTER the example name/number)
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

# ─── Read the config once: module, prompt keys, env ─────────────────────────
# A config may delegate to a different example module via the optional
# top-level "example" field — lets multiple configs share one Python
# example (e.g. baymax_demo.json → inflatable with --shape baymax).
# Every fact the prompts below need is emitted as shlex-quoted shell
# assignments from a single read of the JSON, then eval'd.
DESCRIBE=$(python3 - "$CONFIG_DIR/$EXAMPLE.json" "$EXAMPLE" <<'PY'
import json, shlex, sys
d = json.load(open(sys.argv[1])); a = d.get("args", {})
def out(name, value): print(f"{name}={shlex.quote(str(value))}")
out("INVOKE_EXAMPLE", d.get("example") or sys.argv[2])
out("HAS_VEHICLE_KEY", int("vehicle-asset" in a and a.get("vehicle-asset") is None))
out("HAS_TERRAIN_KEY", int("terrain" in a))
out("CONFIG_TERRAIN", a.get("terrain") or "")
out("TELEMETRY_VEHICLE", a.get("telemetry-vehicle") or "")
out("HAS_TIRE_KEY", int("tire-asset" in a and a.get("tire-asset") is None and "vehicle-asset" not in a))
out("HAS_SUBSTEPS_KEY", int("substeps" in a))
out("CONFIG_SUBSTEPS", a.get("substeps"))
print("CONFIG_ENV=(" + " ".join(shlex.quote(f"{k}={v}") for k, v in d.get("env", {}).items()) + ")")
PY
)
eval "$DESCRIBE"

echo "Running: $EXAMPLE (module: $INVOKE_EXAMPLE)"
echo ""

# ─── Resolve CLI args from the JSON (optionally edit interactively) ──────────
RESOLVE_CMD=(python3 "$HELPER" resolve --config "$CONFIG_DIR/$EXAMPLE.json")
[ "$EDIT" -eq 1 ] && RESOLVE_CMD+=(--edit)
for s in "${SETS[@]}"; do RESOLVE_CMD+=(--set "$s"); done

# given KEY → true when --set KEY=... or a raw --KEY... argument was passed
given() {
    local s r
    for s in "${SETS[@]}"; do [[ "$s" == "$1="* ]] && return 0; done
    for r in "${RAW_ARGS[@]}"; do [[ "$r" == "--$1"* ]] && return 0; done
    return 1
}

# prompt_choice VAR noun "Title" default_name NAMES [LABELS]
# Lists the NAMES array (shown as LABELS when given), reads a 1-based choice —
# Enter picks default_name, or item 1 when it is absent — validates it and
# stores the chosen name in VAR.
prompt_choice() {
    local out="$1" noun="$2" title="$3" default_name="$4"
    local -n names_ref="$5"
    local -n labels_ref="${6:-$5}"
    local default=1 i choice
    for i in "${!names_ref[@]}"; do [ "${names_ref[$i]}" = "$default_name" ] && default=$((i+1)); done
    echo "$title"
    for i in "${!names_ref[@]}"; do printf "  %3d) %s\n" $((i+1)) "${labels_ref[$i]}"; done
    read -p "Select ${noun} (1-${#names_ref[@]}, Enter for ${default}=${names_ref[$((default-1))]}): " choice
    choice="${choice:-$default}"
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#names_ref[@]}" ]; then
        echo "Error: invalid ${noun} choice." >&2
        exit 1
    fi
    printf -v "$out" '%s' "${names_ref[$((choice-1))]}"
}

# ─── Vehicle prompt: configs with a "vehicle-asset" key run any vehicle USD ──
# (newton/examples/ancf/assets/*_vehicle.usd*, baked by newton-tire-tool). Ask which one
# unless --set vehicle-asset=... / a raw --vehicle-asset was given; default: the Warthog.
# The tire follows the vehicle (defaultTireAsset in the vehicle USD) unless overridden.
# (a non-null vehicle-asset in the config pins the vehicle and skips the prompt, like tire-asset)
if [ "$HAS_VEHICLE_KEY" = "1" ] && ! given vehicle-asset; then
    mapfile -t VEHICLES < <(cd "$NEWTON_DIR/newton/examples/ancf/assets" && ls *_vehicle.usd* 2>/dev/null)
    if [ "${#VEHICLES[@]}" -eq 0 ]; then
        echo "Error: no vehicle assets (*_vehicle.usd*) in newton/examples/ancf/assets — run newton-tire-tool/scripts/regenerate_all.sh" >&2
        exit 1
    fi
    prompt_choice VEHICLE vehicle "Vehicle (USD asset):" warthog_vehicle.usdc VEHICLES
    RESOLVE_CMD+=(--set "vehicle-asset=${VEHICLE}")
    echo ""
fi

# ─── Tire resolution prompt: vehicles baked with a low-resolution tire (simpleTireAsset) ─────
# A vehicle <name>_vehicle.usdc that ships assets/<name>_ancf_tire_simple.usda (the super-jeep
# tire's 240-node layout: smooth, no lugs) asks which tire mesh to run (default low); high = the
# vehicle's defaultTireAsset (no override). Skipped when --set tire-asset=... / a raw --tire-asset was given.
VEH_NAME=""
for s in "${SETS[@]}"; do [[ "$s" == vehicle-asset=* ]] && VEH_NAME="${s#vehicle-asset=}"; done
for i in "${!RAW_ARGS[@]}"; do
    case "${RAW_ARGS[$i]}" in
        --vehicle-asset=*) VEH_NAME="${RAW_ARGS[$i]#--vehicle-asset=}" ;;
        --vehicle-asset) VEH_NAME="${RAW_ARGS[$((i+1))]:-}" ;;
    esac
done
[ -n "${VEHICLE:-}" ] && VEH_NAME="$VEHICLE"
if [ -n "$VEH_NAME" ] && ! given tire-asset; then
    VEH_BASE=$(basename "$VEH_NAME"); VEH_BASE="${VEH_BASE%%_vehicle.usd*}"
    SIMPLE_TIRE=$(cd "$NEWTON_DIR/newton/examples/ancf/assets" && ls "${VEH_BASE}"_*tire_simple.usda 2>/dev/null | head -1)
    if [ -n "$SIMPLE_TIRE" ]; then
        echo "Tire resolution for ${VEH_BASE}:"
        echo "    1) high   the vehicle's own tire mesh (lugs, full detail)"
        echo "    2) low    ${SIMPLE_TIRE} (smooth, 240 nodes / 224 elems, like the super-jeep tire)"
        read -p "Select resolution (1-2, Enter for 2=low): " rchoice
        case "${rchoice:-}" in
            1|high) ;;
            ""|2|low) RESOLVE_CMD+=(--set "tire-asset=${SIMPLE_TIRE}") ;;
            *) echo "Error: invalid resolution choice." >&2; exit 1 ;;
        esac
        echo ""
    fi
fi

# ─── Terrain prompt: configs with a "terrain" key run any terrain bundle ─────────────────
# (newton/examples/ancf/assets/terrain/<name>/<name>_terrain.json: boulders / craters from
# newton-terrain-tool, rellis_0000N from newton-rellis-3d-tool — one format). Ask which one
# unless --set terrain=... / a raw --terrain was given; Enter keeps the config's value.
if [ "$HAS_TERRAIN_KEY" = "1" ] && ! given terrain; then
    TERRAIN_DIR="$NEWTON_DIR/newton/examples/ancf/assets/terrain"
    mapfile -t TERRAINS < <(cd "$TERRAIN_DIR" 2>/dev/null && for d in */; do d="${d%/}"; [ -f "$d/${d}_terrain.json" ] && echo "$d"; done)
    # configs with a "telemetry-vehicle" key (vehicle_telemetry) need a recording of that vehicle on the terrain
    if [ -n "$TELEMETRY_VEHICLE" ]; then
        TELEMETRY_DIR="$NEWTON_DIR/newton/examples/ancf/assets/vehicle_telemetry/$TELEMETRY_VEHICLE"
        FILTERED=()
        for d in "${TERRAINS[@]}"; do [ -f "$TELEMETRY_DIR/$d/metadata.json" ] && FILTERED+=("$d"); done
        TERRAINS=("${FILTERED[@]}")
        if [ "${#TERRAINS[@]}" -eq 0 ]; then
            echo "Error: no terrain has $TELEMETRY_VEHICLE telemetry under $TELEMETRY_DIR" >&2
            exit 1
        fi
    fi
    if [ "${#TERRAINS[@]}" -eq 0 ]; then
        echo "Error: no terrain bundles in newton/examples/ancf/assets/terrain — run newton-terrain-tool/regenerate_all.sh --no-trackgen" >&2
        exit 1
    fi
    TERRAIN_LABELS=()
    for i in "${!TERRAINS[@]}"; do
        # one line per bundle: grid size, reference track (length, open/closed) or none
        info=$(python3 - "$TERRAIN_DIR/${TERRAINS[$i]}/${TERRAINS[$i]}_terrain.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); g = m["grid"]; rt = m.get("reference_track")
s = f"{g['size'][0]:.0f} x {g['size'][1]:.0f} m, cell {g['cell']} m"
s += f", track {rt['length_m']:.0f} m {'loop' if rt.get('closed') else 'open'}" if rt else ", no track"
print(s)
PY
)
        printf -v "TERRAIN_LABELS[$i]" "%-14s %s" "${TERRAINS[$i]}" "$info"
    done
    prompt_choice TERRAIN terrain "Terrain (bundle under assets/terrain/):" "$CONFIG_TERRAIN" TERRAINS TERRAIN_LABELS
    RESOLVE_CMD+=(--set "terrain=${TERRAIN}")
    echo ""
fi

# ─── Tire prompt: tire-only configs (a "tire-asset" key, no vehicle) run any ANCF tire USD ──
# (newton/examples/ancf/assets/*.usda — tires and the minimal ball; vehicles are .usdc). Default: the super-jeep tire
# (the validated one; the Sherp bake is not stable yet). Skipped when the config pins a non-null tire-asset, or when given on the CLI.
if [ "$HAS_TIRE_KEY" = "1" ] && ! given tire-asset; then
    mapfile -t TIRES < <(cd "$NEWTON_DIR/newton/examples/ancf/assets" && ls *.usda 2>/dev/null)
    if [ "${#TIRES[@]}" -eq 0 ]; then
        echo "Error: no ANCF shell assets (*.usda) in newton/examples/ancf/assets — run newton-tire-tool/scripts/regenerate_all.sh" >&2
        exit 1
    fi
    prompt_choice TIRE tire "Tire (ANCF USD asset):" superjeep_tire.usda TIRES
    RESOLVE_CMD+=(--set "tire-asset=${TIRE}")
    echo ""
fi

# ─── Substeps prompt: configs with a "substeps" key ─────────────────────────────────────────
# The implicit solver retains the config timestep; fewer substeps need an accuracy check.
# Skipped when --set substeps=... or a raw --substeps was given; Enter keeps
# the config's value.
if [ "$HAS_SUBSTEPS_KEY" = "1" ] && ! given substeps; then
    echo "Substeps per 60 Hz frame (implicit solver):"
    read -p "Substeps (positive integer, Enter for ${CONFIG_SUBSTEPS}): " nchoice
    case "${nchoice:-}" in
        "")  ;;
        ''|*[!0-9]*) echo "Error: substeps must be a positive integer." >&2; exit 1 ;;
        0) echo "Error: substeps must be a positive integer." >&2; exit 1 ;;
        *) RESOLVE_CMD+=(--set "substeps=${nchoice}") ;;
    esac
    echo ""
fi

# Validate the final argument selection, including raw arguments, before launching Docker.
# Raw arguments remain last and retain argparse's last-value precedence.
RESOLVE_CMD+=(--raw-args "${RAW_ARGS[@]}")
EXTRA_ARGS=$("${RESOLVE_CMD[@]}")

# Optional per-example environment (config "env"), passed as array arguments.
CONFIG_ENV_ARGS=()
for assignment in "${CONFIG_ENV[@]}"; do CONFIG_ENV_ARGS+=(-e "$assignment"); done

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
    "${CONFIG_ENV_ARGS[@]}" \
    -v "$NEWTON_DIR/newton:/workspace/newton/newton" \
    -v "$NEWTON_DIR/third_party/mujoco_warp:/workspace/newton/third_party/mujoco_warp:ro" \
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
