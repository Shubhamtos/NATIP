#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

HOST="${NATIP_HOST:-127.0.0.1}"
PORT="${NATIP_PORT:-8501}"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR" "${PROJECT_ROOT}/.natip"

if [[ ! -x "${PROJECT_ROOT}/venv/bin/streamlit" ]]; then
  echo "Missing ${PROJECT_ROOT}/venv/bin/streamlit" >&2
  exit 1
fi

exec "${PROJECT_ROOT}/venv/bin/streamlit" run app/dashboard/streamlit_app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none
