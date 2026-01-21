#!/usr/bin/env bash
set -euo pipefail

cd /var/www/CRM_SAAS

# Arrêter tous les uvicorn CRM_SAAS encore en écoute
pkill -f ".venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8101" || true

# Lancer le nouveau serveur (log dans uvicorn.log, PID dans uvicorn.pid)
PYTHONPATH=/var/www/CRM_SAAS nohup .venv/bin/uvicorn src.api.main:app --host 127.0.0.1 --port 8101 > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
echo "Processus uvicorn CRM_SAAS actifs :"
ps -ef | grep "[u]vicorn src.api.main:app --host 127.0.0.1 --port 8101"
echo "Derniers logs uvicorn.log :"
tail -n 10 uvicorn.log
