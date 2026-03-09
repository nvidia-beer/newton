#!/usr/bin/env bash
# Plot Fig. 4 style (paper arXiv:1911.05227) via Docker (same image as run-examples).
# CSVs and output PNG live in this folder (crawlable/inchworm/). By default uses latest inchworm_*.csv.
# All CSV data is from simulation.
#
# Usage:
#   ./plot-fig4.sh                         # latest simulation CSV, last full cycle
#   ./plot-fig4.sh run.csv                 # one simulation CSV
#   ./plot-fig4.sh run.csv run_low.csv     # main + low-stiffness simulation CSVs
#   ./plot-fig4.sh run1.csv -o fig4.png
# Output: <base>_a.png (angles), <base>_b.png (contacts)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_DIR="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
INCHWORM_DIR="$SCRIPT_DIR"
PARAMS="$INCHWORM_DIR/inchworm_params.json"

# Parse positional CSV args (optional: first = main simulation CSV, second = low stiffness)
CSV_R_PATH=""
CSV_L_PATH=""
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -o|--output)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --params|--save_csv|--dpi)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --cycle)
      shift 2
      ;;
    *)
      if [[ "$1" == *.csv ]]; then
        if [ -z "$CSV_R_PATH" ]; then
          CSV_R_PATH="$1"
        elif [ -z "$CSV_L_PATH" ]; then
          CSV_L_PATH="$1"
        else
          EXTRA_ARGS+=("$1")
        fi
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

# Resolve CSV paths (like render-inchworm-gif.sh): filename in INCHWORM_DIR or full path
resolve_csv() {
  local arg="$1"
  if [ -z "$arg" ]; then
    return
  fi
  if [ -f "$arg" ]; then
    echo "$arg"
    return
  fi
  if [ -f "$INCHWORM_DIR/$arg" ]; then
    echo "$INCHWORM_DIR/$arg"
    return
  fi
  echo "Not a file: $arg" >&2
  exit 1
}

# Default: latest inchworm_*.csv if no CSV given
if [ -z "$CSV_R_PATH" ]; then
  CSV_R_PATH="$(ls -t "$INCHWORM_DIR"/inchworm_*.csv 2>/dev/null | head -1)"
  if [ -n "$CSV_R_PATH" ]; then
    echo "Using latest CSV: $(basename "$CSV_R_PATH")"
  fi
fi

CSV_R_RESOLVED=""
CSV_L_RESOLVED=""
[ -n "$CSV_R_PATH" ] && CSV_R_RESOLVED="$(resolve_csv "$CSV_R_PATH")"
[ -n "$CSV_L_PATH" ] && CSV_L_RESOLVED="$(resolve_csv "$CSV_L_PATH")"

# Require CSVs under INCHWORM_DIR so container sees them
for p in "$CSV_R_RESOLVED" "$CSV_L_RESOLVED"; do
  [ -z "$p" ] && continue
  case "$p" in
    "$INCHWORM_DIR"/*) ;;
    *)
      echo "CSV must be in $INCHWORM_DIR: $p" >&2
      exit 1
      ;;
  esac
done

# Default output if -o not in EXTRA_ARGS
HAS_O=0
for a in "${EXTRA_ARGS[@]}"; do
  [ "$a" = "-o" ] || [ "$a" = "--output" ] && HAS_O=1 && break
done
OUT_BASENAME="fig4_simulation.png"
[ $HAS_O -eq 0 ] && EXTRA_ARGS+=("-o" "$OUT_BASENAME")

if ! docker image inspect newton:latest >/dev/null 2>&1; then
  echo "Docker image newton:latest not found. Build with: $NEWTON_DIR/.devcontainer/build-docker.sh" >&2
  exit 1
fi

# Build Python args with container paths (/workspace/inchworm/...)
PY_ARGS=()
[ -n "$CSV_R_RESOLVED" ] && PY_ARGS+=("--csv" "/workspace/inchworm/$(basename "$CSV_R_RESOLVED")")
[ -n "$CSV_L_RESOLVED" ] && PY_ARGS+=("--csv_low" "/workspace/inchworm/$(basename "$CSV_L_RESOLVED")")
[ -f "$PARAMS" ] && PY_ARGS+=("--params" "/workspace/inchworm/inchworm_params.json")

# Rewrite -o and --save_csv paths to container paths (so output lands in mounted dir)
i=0
while [ $i -lt ${#EXTRA_ARGS[@]} ]; do
  a="${EXTRA_ARGS[$i]}"
  if [ "$a" = "-o" ] || [ "$a" = "--output" ]; then
    i=$((i + 1))
    val="${EXTRA_ARGS[$i]}"
    PY_ARGS+=("$a" "/workspace/inchworm/$(basename "$val")")
    OUT_BASENAME="$(basename "$val")"
  elif [ "$a" = "--save_csv" ]; then
    i=$((i + 1))
    val="${EXTRA_ARGS[$i]}"
    PY_ARGS+=("$a" "/workspace/inchworm/$(basename "$val")")
  else
    PY_ARGS+=("$a")
  fi
  i=$((i + 1))
done

docker run --rm \
  -v "$INCHWORM_DIR:/workspace/inchworm" \
  newton:latest \
  bash -c "python3 -m pip install matplotlib -q 2>/dev/null; python3 /workspace/inchworm/plot_fig4.py \"\$@\"" _ "${PY_ARGS[@]}"

BASE="${OUT_BASENAME%.*}"
echo "Plots: $INCHWORM_DIR/${BASE}_a.png (angles), $INCHWORM_DIR/${BASE}_b.png (contacts)"
