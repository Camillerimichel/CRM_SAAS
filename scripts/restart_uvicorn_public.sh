#!/usr/bin/env bash
set -euo pipefail

cd /var/www/CRM_SAAS

# Stop any existing uvicorn for this app
pkill -f "uvicorn src.api.main:app" || true

# Start on public port 8100 with logs
PYTHONPATH=/var/www/CRM_SAAS nohup /var/www/CRM_SAAS/.venv/bin/uvicorn \
  src.api.main:app --host 0.0.0.0 --port 8100 --log-level info --access-log \
  > /var/www/CRM_SAAS/uvicorn.log 2>&1 &

echo $! > /var/www/CRM_SAAS/uvicorn.pid

echo "Processus uvicorn CRM_SAAS actifs :"
ps -ef | grep "[u]vicorn src.api.main:app --host 0.0.0.0 --port 8100"

echo "Derniers logs uvicorn.log :"
tail -n 10 /var/www/CRM_SAAS/uvicorn.log
