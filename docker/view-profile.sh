#!/bin/bash
# Newton Profile Viewer
#
# Opens an Nsight Systems .nsys-rep file in the host GUI.
# Searches PROFILE_DIR (default: $HOME/newton-profiles) for reports.
#
# Usage:
#   ./view-profile.sh                        # Pick from saved reports (newest first)
#   ./view-profile.sh path/to/report.nsys-rep  # Open specific file
#   ./view-profile.sh --dir ~/my-profiles    # Search a different directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="${PROFILE_DIR:-$SCRIPT_DIR/nsys}"

# ─── CLI ─────────────────────────────────────────────────────────────────────
EXPLICIT_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)
            [ $# -lt 2 ] && { echo "error: --dir expects a path" >&2; exit 2; }
            PROFILE_DIR="$2"; shift
            ;;
        --dir=*) PROFILE_DIR="${1#--dir=}" ;;
        -h|--help)
            sed -n '2,/^set -e$/p' "$0" | sed 's/^# \{0,1\}//;/^set -e$/d'
            exit 0
            ;;
        *.nsys-rep) EXPLICIT_FILE="$1" ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# ─── Find nsys-ui ────────────────────────────────────────────────────────────
find_nsys_ui() {
    # 1. Already on PATH
    if command -v nsys-ui &>/dev/null; then
        echo "$(command -v nsys-ui)"
        return
    fi
    # 2. Common NVIDIA install locations (newest version first)
    for dir in \
        /opt/nvidia/nsight-systems/*/host-linux-x64 \
        /opt/nvidia/nsight-systems/*/host-linux-armv8 \
        /usr/local/nsight-systems/*/host-linux-x64 \
        /usr/local/nsight-systems/*/host-linux-armv8 \
        "$HOME/nsight-systems/*/host-linux-x64" \
        "$HOME/nsight-systems/*/host-linux-armv8"
    do
        # glob expansion — pick last match (newest version)
        for candidate in $dir/nsys-ui; do
            [ -x "$candidate" ] && echo "$candidate" && return
        done
    done
    echo ""
}

NSYS_UI=$(find_nsys_ui)
if [ -z "$NSYS_UI" ]; then
    echo "Error: nsys-ui not found on this host." >&2
    echo "" >&2
    echo "Install Nsight Systems from:" >&2
    echo "  https://developer.nvidia.com/nsight-systems" >&2
    echo "" >&2
    echo "Or copy the .nsys-rep file to a machine that has it." >&2
    exit 1
fi
echo "Using: $NSYS_UI"
echo ""

# ─── Resolve target file ─────────────────────────────────────────────────────
if [ -n "$EXPLICIT_FILE" ]; then
    if [ ! -f "$EXPLICIT_FILE" ]; then
        echo "Error: file not found: $EXPLICIT_FILE" >&2
        exit 1
    fi
    TARGET="$EXPLICIT_FILE"
else
    if [ ! -d "$PROFILE_DIR" ]; then
        echo "Error: profile directory not found: $PROFILE_DIR" >&2
        echo "Run an example with --profile first." >&2
        exit 1
    fi

    # List reports, newest first
    mapfile -t REPORTS < <(find "$PROFILE_DIR" -maxdepth 2 -name "*.nsys-rep" -printf "%T@ %p\n" 2>/dev/null \
        | sort -rn | awk '{print $2}')

    if [ "${#REPORTS[@]}" -eq 0 ]; then
        echo "No .nsys-rep files found in $PROFILE_DIR" >&2
        echo "Run an example with --profile first." >&2
        exit 1
    fi

    if [ "${#REPORTS[@]}" -eq 1 ]; then
        TARGET="${REPORTS[0]}"
        echo "Opening: $TARGET"
    else
        echo "Available profiles (newest first):"
        echo ""
        for i in "${!REPORTS[@]}"; do
            printf "  %3d) %s\n" $((i+1)) "$(basename "${REPORTS[$i]}")"
        done
        echo ""
        read -p "Select profile (1-${#REPORTS[@]}, Enter for 1): " choice
        choice="${choice:-1}"
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#REPORTS[@]}" ]; then
            echo "Error: invalid choice." >&2
            exit 1
        fi
        TARGET="${REPORTS[$((choice-1))]}"
    fi
fi

echo ""
echo "Opening: $TARGET"
"$NSYS_UI" "$TARGET" &
