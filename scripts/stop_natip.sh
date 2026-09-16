#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${NATIP_PORT:-8501}"
PID_FILE="${PROJECT_ROOT}/.natip/streamlit.pid"

pid=""
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" || true)"
fi
if [[ -z "$pid" ]]; then
  pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
fi

if [[ -z "$pid" ]]; then
  echo "NATIP is not running on port ${PORT}."
  exit 0
fi

echo "Stopping NATIP pid ${pid}..."
kill "$pid" 2>/dev/null || true
sleep 2
if kill -0 "$pid" 2>/dev/null; then
  kill -9 "$pid" 2>/dev/null || true
fi
rm -f "$PID_FILE"
echo "NATIP stopped."
