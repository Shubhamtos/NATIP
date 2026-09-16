#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

HOST="${NATIP_HOST:-127.0.0.1}"
PORT="${NATIP_PORT:-8501}"
URL="http://${HOST}:${PORT}"
PID_DIR="${PROJECT_ROOT}/.natip"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${PID_DIR}/streamlit.pid"
LOG_FILE="${LOG_DIR}/streamlit.log"

mkdir -p "$PID_DIR" "$LOG_DIR"

if [[ ! -x "${PROJECT_ROOT}/venv/bin/python" || ! -x "${PROJECT_ROOT}/venv/bin/streamlit" ]]; then
  echo "NATIP venv is missing or incomplete."
  echo "Run: cd \"$PROJECT_ROOT\" && python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

if ! "${PROJECT_ROOT}/venv/bin/python" - <<'PY' >/dev/null 2>&1
import streamlit
raise SystemExit(0 if streamlit.__version__ == "1.36.0" else 1)
PY
then
  echo "Streamlit version mismatch. NATIP local UI is pinned to Streamlit 1.36.0."
  echo "Run: cd \"$PROJECT_ROOT\" && venv/bin/pip install -r requirements.txt"
  exit 1
fi

existing_pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
if [[ -n "$existing_pid" ]]; then
  echo "$existing_pid" > "$PID_FILE"
  echo "NATIP already appears to be running on ${URL} (pid ${existing_pid})."
  echo "Open: ${URL}"
  exit 0
fi

echo "Starting NATIP on ${URL}..."
nohup "${PROJECT_ROOT}/venv/bin/streamlit" run app/dashboard/streamlit_app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  >>"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

for _ in $(seq 1 60); do
  if "${PROJECT_ROOT}/venv/bin/python" - "$URL" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

url = sys.argv[1]
with urlopen(url, timeout=1.5) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    echo "NATIP is ready: ${URL}"
    echo "Log file: ${LOG_FILE}"
    exit 0
  fi
  sleep 1
done

echo "NATIP did not become ready within 60 seconds."
echo "Check log: ${LOG_FILE}"
exit 1
