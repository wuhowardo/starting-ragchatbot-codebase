#!/usr/bin/env bash
# Run code quality checks for the project.
# Usage: ./scripts/check_quality.sh [--fix]
#
#   --fix   Auto-format with black instead of just checking

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FIX=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX=true
fi

echo "=== Black (formatting) ==="
if $FIX; then
    uv run black backend/ main.py
else
    uv run black --check backend/ main.py
fi

echo ""
echo "=== Pytest ==="
cd backend && uv run pytest tests/ -v
