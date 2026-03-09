#!/usr/bin/env bash
# Build full documentation PDF (LaTeX) from Markdown.
# Usage: run from repo root or from documentation/scripts:
#   ./newton/newton/documentation/scripts/build_documentation_pdf.sh
# Or (from documentation/scripts): ./build_documentation_pdf.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python3 build_documentation_pdf.py
