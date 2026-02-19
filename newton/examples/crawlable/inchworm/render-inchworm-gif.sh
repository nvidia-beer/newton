#!/usr/bin/env bash
# Render inchworm GIF from CSV via Docker (same image as run-examples).
# CSVs and GIFs live in this folder (crawlable/inchworm/). By default uses latest CSV.
#
# Usage:
#   ./newton/newton/examples/crawlable/inchworm/render-inchworm-gif.sh   # latest inchworm_*.csv
#   ./newton/newton/examples/crawlable/inchworm/render-inchworm-gif.sh file.csv
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWTON_DIR="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
INCHWORM_DIR="$SCRIPT_DIR"

CSV_ARG="$1"
if [ -z "$CSV_ARG" ]; then
  CSV_PATH="$(ls -t "$INCHWORM_DIR"/inchworm_*.csv 2>/dev/null | head -1)"
  if [ -z "$CSV_PATH" ]; then
    echo "Usage: $0 [inchworm_csv]" >&2
    echo "  No CSV given and no inchworm_*.csv in $INCHWORM_DIR" >&2
    exit 1
  fi
  echo "Using latest CSV: $(basename "$CSV_PATH")"
else
  # Allow filename only (in inchworm dir) or full path
  if [ -f "$CSV_ARG" ]; then
    CSV_PATH="$CSV_ARG"
  elif [ -f "$INCHWORM_DIR/$CSV_ARG" ]; then
    CSV_PATH="$INCHWORM_DIR/$CSV_ARG"
  else
    echo "Not a file: $CSV_ARG" >&2
    exit 1
  fi
fi

CSV_BASENAME="$(basename "$CSV_PATH")"
OUT_BASENAME="${CSV_BASENAME%.csv}_movement.gif"

# CSV must be in inchworm dir so the container sees it
case "$CSV_PATH" in
  "$INCHWORM_DIR"/*) ;;
  *)
    echo "CSV must be in $INCHWORM_DIR" >&2
    exit 1
    ;;
esac

if ! docker image inspect newton:latest >/dev/null 2>&1; then
  echo "Docker image newton:latest not found. Build with: $NEWTON_DIR/.devcontainer/build-docker.sh" >&2
  exit 1
fi

# Matplotlib is in the image if you rebuild (pyproject.toml [examples]). Else install at run time:
docker run --rm \
  -v "$INCHWORM_DIR:/workspace/inchworm" \
  newton:latest \
  bash -c "python3 -m pip install matplotlib -q; python3 /workspace/inchworm/render_inchworm_gif.py \
    \"/workspace/inchworm/$CSV_BASENAME\" \
    -o \"/workspace/inchworm/$OUT_BASENAME\" \
    --fps 10"

echo "Animation: $INCHWORM_DIR/$OUT_BASENAME"
