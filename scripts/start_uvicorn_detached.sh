#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/var/www/CRM_SAAS"
LOG_FILE="${APP_DIR}/uvicorn.log"
PID_FILE="${APP_DIR}/uvicorn.pid"
UVICORN_BIN="${APP_DIR}/.venv/bin/uvicorn"
PYTHON_BIN="${APP_DIR}/.venv/bin/python3"

cd "$APP_DIR"

# Stop existing CRM_SAAS uvicorn (port 8100)
PIDS=$(pgrep -f "${APP_DIR}/\\.venv/bin/uvicorn src\\.api\\.main:app .*--port 8100" || true)
if [ -n "${PIDS:-}" ]; then
  echo "Stopping old uvicorn PIDs: ${PIDS}..."
  kill ${PIDS} 2>/dev/null || true
  sleep 0.6
fi

: > "$LOG_FILE"

# Detach a background uvicorn in a new session so it survives the shell.
APP_DIR="$APP_DIR" LOG_FILE="$LOG_FILE" PID_FILE="$PID_FILE" UVICORN_BIN="$UVICORN_BIN" "$PYTHON_BIN" - <<'PY'
import os
import subprocess

app_dir = os.environ["APP_DIR"]
log_path = os.environ["LOG_FILE"]
pid_path = os.environ["PID_FILE"]
uvicorn_bin = os.environ["UVICORN_BIN"]

env = os.environ.copy()
env["PYTHONPATH"] = app_dir

cmd = [
  uvicorn_bin,
  "src.api.main:app",
  "--host", "127.0.0.1",
  "--port", "8100",
  "--log-level", "info",
  "--access-log",
]

# `--reload` est pratique en dev, mais peut être bloqué dans certains environnements sandboxés.
if os.environ.get("UVICORN_RELOAD", "0") == "1":
  cmd.extend(["--reload", "--reload-dir", app_dir])

with open(log_path, "ab", buffering=0) as log:
  p = subprocess.Popen(
    cmd,
    cwd=app_dir,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
  )

with open(pid_path, "w", encoding="utf-8") as f:
  f.write(str(p.pid))

print(p.pid)
PY

sleep 0.4

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "${PID:-}" ] || ! kill -0 "$PID" 2>/dev/null; then
  echo "uvicorn n'est pas resté en vie. Derniers logs:"
  tail -n 80 "$LOG_FILE" || true
  exit 1
fi

echo "Uvicorn détaché OK (PID $PID) — logs: $LOG_FILE"
