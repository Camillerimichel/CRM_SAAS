#!/usr/bin/env bash
set -euo pipefail

SESSION="crm"
UVICORN_CMD="PYTHONPATH=/var/www/CRM_SAAS /var/www/CRM_SAAS/.venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8100 --reload"

# Tuer l’ancien serveur s’il existe
OLD_PID=$(ss -tulpn | awk '/127\.0\.0\.1:8100/ { print $6 }' | cut -d',' -f2 | head -n1 || true)
if [ -n "$OLD_PID" ]; then
  echo "Stopping old uvicorn (PID $OLD_PID)..."
  kill "$OLD_PID" || true
fi

# Créer la session tmux si elle n’existe pas
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION"
fi

# Lancer la commande dans tmux (remplace la fenêtre 1)
tmux send-keys -t "$SESSION" C-c
tmux send-keys -t "$SESSION" "cd /var/www/CRM_SAAS" Enter
tmux send-keys -t "$SESSION" "$UVICORN_CMD" Enter

echo "Uvicorn démarré dans tmux session \"$SESSION\". Attache avec : tmux attach -t $SESSION"
