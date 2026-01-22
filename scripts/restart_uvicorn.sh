#!/usr/bin/env bash
set -euo pipefail

cd /var/www/CRM_SAAS

# Arrêter tous les uvicorn CRM_SAAS encore en écoute
if [[ -f uvicorn.pid ]]; then
  kill "$(cat uvicorn.pid)" 2>/dev/null || true
fi
pkill -f "uvicorn src.api.main:app --host 127.0.0.1 --port 8101" || true

# Attendre la fin réelle des processus
for _ in {1..10}; do
  if ! pgrep -f "uvicorn src.api.main:app --host 127.0.0.1 --port 8101" >/dev/null; then
    break
  fi
  sleep 0.2
done

# Lancer le nouveau serveur (log dans uvicorn.log, PID dans uvicorn.pid)
PYTHONPATH=/var/www/CRM_SAAS nohup .venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8101 > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
echo "Processus uvicorn CRM_SAAS actifs :"
ps -ef | grep "[u]vicorn src.api.main:app --host 127.0.0.1 --port 8101"
echo "Derniers logs uvicorn.log :"
tail -n 10 uvicorn.log
