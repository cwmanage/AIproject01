#!/usr/bin/env bash
# =====================================================================
#  One-click launcher for macOS / Linux
#  Usage:  ./start.sh        (first time:  chmod +x start.sh)
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --- find a Python interpreter ------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    exec python3 start.py
elif command -v python >/dev/null 2>&1; then
    exec python start.py
else
    echo "[start.sh] Python 3 was not found on PATH."
    echo "Please install Python 3.9+ (https://www.python.org/downloads/)"
    exit 1
fi
