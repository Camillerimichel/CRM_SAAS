#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 CLIENT_ID"
  exit 1
fi

CLIENT_ID="$1"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PYTHONPATH="${PYTHONPATH:-/var/www/CRM_SAAS}"

echo "Seeding SRI metrics for client ${CLIENT_ID}..."
PYTHONPATH="${PYTHONPATH}" "$PYTHON_BIN" scripts/seed_client_sri.py "${CLIENT_ID}"

PDF_URL="http://127.0.0.1:8100/dashboard/clients/${CLIENT_ID}/kyc/rapport?style=conformite&pdf=1"
PDF_FILE="rapport_kyc_${CLIENT_ID}.pdf"

echo "Fetching PDF report to ${PDF_FILE}..."
curl -fSL -o "${PDF_FILE}" "${PDF_URL}"

echo "Done. Open ${PDF_FILE} to verify the SRI block."
