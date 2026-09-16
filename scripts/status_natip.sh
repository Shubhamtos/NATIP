#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${NATIP_HOST:-127.0.0.1}"
PORT="${NATIP_PORT:-8501}"
URL="http://${HOST}:${PORT}"
LOG_FILE="${PROJECT_ROOT}/logs/streamlit.log"

pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
if [[ -z "$pid" ]]; then
  echo "NATIP is not running on ${URL}."
  [[ -f "$LOG_FILE" ]] && echo "Last log: ${LOG_FILE}"
  exit 1
fi

if "${PROJECT_ROOT}/venv/bin/python" - "$URL" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
then
  echo "NATIP is running and healthy: ${URL} (pid ${pid})"
else
  echo "NATIP process exists on ${URL}, but health check failed (pid ${pid})."
  echo "Check log: ${LOG_FILE}"
  exit 2
fi
