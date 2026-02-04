#!/usr/bin/env bash
set -euo pipefail

cd /var/www/CRM_SAAS

echo "Restart uvicorn CRM_SAAS (127.0.0.1:8100)..."

# Démarrage détaché (tmux n'est pas toujours autorisé dans cet environnement).
bash /var/www/CRM_SAAS/scripts/start_uvicorn_detached.sh

echo "Derniers logs uvicorn.log :"
tail -n 30 /var/www/CRM_SAAS/uvicorn.log 2>/dev/null || true
