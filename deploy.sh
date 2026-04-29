#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/var/www/CRM_SAAS"
FRONTEND_DIR="${APP_DIR}/frontend"
PUBLIC_DOC_HTML="${FRONTEND_DIR}/public/bibliotheque-documents/index.html"
SOURCE_DOC_HTML="${APP_DIR}/src/api/templates/bibliotheque_documents.html"
STATIC_TARGET="/var/www/html/crm_saas"
SERVICE_NAME="crm-saas.service"
LOG_FILE="${APP_DIR}/deploy.log"
CHECK_URL="https://crmsaas.eu/dashboard/bibliotheque-documents/"
CHECK_MARKER="Nouveau modèle documentaire"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

fail() {
  log "ERREUR: $*"
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "commande manquante: $1"
}

cd "$APP_DIR"
: > "$LOG_FILE"

log "Déploiement CRM_SAAS lancé"
log "Répertoire application: $APP_DIR"

need_cmd rsync
need_cmd curl
need_cmd systemctl
need_cmd git
need_cmd npm

log "Mise à jour du dépôt Git"
git pull --rebase --autostash origin main

if [ -f "$SOURCE_DOC_HTML" ]; then
  log "Synchronisation du template bibliothèque documentaire vers le frontend public"
  mkdir -p "$(dirname "$PUBLIC_DOC_HTML")"
  cp "$SOURCE_DOC_HTML" "$PUBLIC_DOC_HTML"
fi

log "Build frontend"
(
  cd "$FRONTEND_DIR"
  npm run build
)

if [ -d "$FRONTEND_DIR/build" ]; then
  log "Synchronisation du build frontend vers $STATIC_TARGET"
  mkdir -p "$STATIC_TARGET"
  rsync -a --delete \
    "$FRONTEND_DIR/build/" \
    "$STATIC_TARGET/"
else
  fail "répertoire build frontend introuvable après compilation"
fi

log "Redémarrage du service $SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
sleep 2

if ! systemctl is-active --quiet "$SERVICE_NAME"; then
  systemctl status "$SERVICE_NAME" --no-pager | tee -a "$LOG_FILE" || true
  fail "le service $SERVICE_NAME n'est pas actif après redémarrage"
fi

log "Vérification HTTP de $CHECK_URL"
page=""
for attempt in 1 2 3 4 5; do
  page="$(curl -ksL "$CHECK_URL" || true)"
  if printf '%s' "$page" | grep -q "$CHECK_MARKER"; then
    break
  fi
  sleep 2
done
if ! printf '%s' "$page" | grep -q "$CHECK_MARKER"; then
  fail "le marqueur '$CHECK_MARKER' n'est pas présent dans la page publiée"
fi

log "Déploiement terminé avec succès"
