#!/usr/bin/env bash
set -euo pipefail

SESSION="crm"
LOG_FILE="/var/www/CRM_SAAS/uvicorn.log"
PID_FILE="/var/www/CRM_SAAS/uvicorn.pid"
UVICORN_CMD="cd /var/www/CRM_SAAS && : > \"$LOG_FILE\" && PYTHONPATH=/var/www/CRM_SAAS /var/www/CRM_SAAS/.venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8100 --reload >> \"$LOG_FILE\" 2>&1"

# Tuer l’ancien serveur s’il existe (sans dépendre de `ss`, parfois restreint)
PIDS=$(pgrep -f "/var/www/CRM_SAAS/\\.venv/bin/uvicorn src\\.api\\.main:app .*--port 8100" || true)
if [ -n "${PIDS:-}" ]; then
  echo "Stopping old uvicorn PIDs: ${PIDS}..."
  kill ${PIDS} 2>/dev/null || true
  sleep 0.5
fi

# Si tmux n'est pas utilisable (sandbox), fallback vers un démarrage détaché.
if ! tmux ls >/dev/null 2>&1; then
  echo "tmux indisponible (sandbox). Fallback: start_uvicorn_detached.sh"
  bash /var/www/CRM_SAAS/scripts/start_uvicorn_detached.sh
  exit 0
fi

# Créer la session tmux si elle n’existe pas
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
fi

# Lancer la commande dans tmux (remplace la fenêtre 1)
tmux send-keys -t "$SESSION" C-c
tmux send-keys -t "$SESSION" "$UVICORN_CMD" Enter

sleep 0.4
NEW_PID=$(pgrep -f "/var/www/CRM_SAAS/\\.venv/bin/uvicorn src\\.api\\.main:app .*--port 8100" | head -n1 || true)
if [ -n "${NEW_PID:-}" ]; then
  echo "${NEW_PID}" > "$PID_FILE"
fi

echo "Uvicorn démarré dans tmux session \"$SESSION\"."
echo "Logs: $LOG_FILE"
echo "PID (si détecté): $(cat "$PID_FILE" 2>/dev/null || echo 'n/a')"
echo "Attache: tmux attach -t $SESSION"
