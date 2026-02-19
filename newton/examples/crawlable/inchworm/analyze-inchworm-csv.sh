#!/usr/bin/env bash
# Analyze inchworm run CSV via Docker (paper-aligned metrics: displacement, velocity, CoM height).
# Same image as run-examples. By default uses latest inchworm_*.csv in this folder.
#
# Usage:
#   ./newton/newton/examples/crawlable/inchworm/analyze-inchworm-csv.sh   # latest inchworm_*.csv
#   ./newton/newton/examples/crawlable/inchworm/analyze-inchworm-csv.sh file.csv
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root (where .devcontainer lives)
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

docker run --rm \
  -v "$INCHWORM_DIR:/workspace/inchworm" \
  newton:latest \
  python3 "/workspace/inchworm/analyze_run_csv.py" "/workspace/inchworm/$CSV_BASENAME"
