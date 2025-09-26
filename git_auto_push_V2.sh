#!/bin/bash

# =============================================================================
# Script pour pousser les modifications sur GitHub (sans fichiers SQL)
# Version améliorée avec gestion d'erreurs et optimisations
# =============================================================================

set -euo pipefail  # Arrêt immédiat en cas d'erreur

# Configuration
LOG_FILE="git_auto_push.log"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Fonction pour afficher l'heure
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Vérification des prérequis
check_prerequisites() {
    log "${BLUE}[$(timestamp)] Vérification des prérequis...${NC}"
    
    # Vérifier que nous sommes dans un repo Git
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log "${RED}❌ Ce dossier n'est pas un repository Git${NC}"
        exit 1
    fi
    
    log "${GREEN}✅ Tous les prérequis sont OK${NC}"
}

# Configuration Git pour les gros fichiers
configure_git() {
    log "${BLUE}[$(timestamp)] Configuration Git...${NC}"
    
    # Augmenter la taille du buffer
    git config http.postBuffer 524288000  # 500MB
    git config http.maxRequestBuffer 100m
    git config core.compression 9
    
    log "${GREEN}✅ Configuration Git mise à jour${NC}"
}

# Se placer dans le dossier du script
cd "$(dirname "$0")"

log "${BLUE}[$(timestamp)] === Début du script de push Git ===${NC}"
log "${BLUE}Dossier de travail: $(pwd)${NC}"

# Vérifications
check_prerequisites
configure_git

# Vérifier le statut Git
log "${BLUE}[$(timestamp)] Vérification des changements Git...${NC}"

# Exclure tous les fichiers SQL du tracking Git
if ! grep -q "*.sql" .gitignore 2>/dev/null; then
    echo "*.sql" >> .gitignore
    echo "*.sql.gz" >> .gitignore
    echo "schema.sql*" >> .gitignore
    echo "dump.sql" >> .gitignore
    log "${GREEN}✅ Fichiers SQL ajoutés au .gitignore${NC}"
fi

# Supprimer les fichiers SQL de l'index Git s'ils y sont déjà
if git ls-files | grep -q "\.sql$"; then
    log "${YELLOW}⚠️ Fichiers SQL détectés dans Git, suppression en cours...${NC}"
    git rm --cached *.sql 2>/dev/null || true
    git rm --cached dump.sql 2>/dev/null || true
    git rm --cached schema.sql* 2>/dev/null || true
    log "${GREEN}✅ Fichiers SQL supprimés de l'index Git${NC}"
fi

# Ajouter tous les fichiers (sauf les SQL grâce au .gitignore)
git add -A

# Vérifier s'il y a des changements
if git diff --cached --quiet; then
    log "${YELLOW}⚠️ Aucune modification détectée - rien à commiter${NC}"
    exit 0
fi

# Afficher les changements
log "${BLUE}📋 Changements détectés:${NC}"
git diff --cached --stat | while read line; do
    log "  $line"
done

# Message de commit avec informations détaillées
commit_message="Auto-commit du $(timestamp)

- Modifications du code source
- Fichiers modifiés: $(git diff --cached --numstat | wc -l | tr -d ' ')
"

# Commit
log "${BLUE}[$(timestamp)] Création du commit...${NC}"
if git commit -m "$commit_message"; then
    log "${GREEN}✅ Commit créé avec succès${NC}"
else
    log "${RED}❌ Erreur lors du commit${NC}"
    exit 1
fi

# Push avec retry
log "${BLUE}[$(timestamp)] Push vers GitHub...${NC}"
retry_count=0
max_retries=3

while [ $retry_count -lt $max_retries ]; do
    if git push origin main 2>&1 | tee -a "$LOG_FILE"; then
        log "${GREEN}✅ Push réussi vers GitHub${NC}"
        break
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log "${YELLOW}⚠️ Échec du push (tentative $retry_count/$max_retries). Nouvelle tentative dans 5s...${NC}"
            sleep 5
            
            # Essayer de résoudre les problèmes courants
            if [ $retry_count -eq 2 ]; then
                log "${BLUE}🔧 Tentative de résolution automatique...${NC}"
                git pull --rebase origin main 2>/dev/null || true
            fi
        else
            log "${RED}❌ Échec définitif du push après $max_retries tentatives${NC}"
            
            # Diagnostic
            log "${BLUE}🔍 Diagnostic:${NC}"
            log "  - Taille du dernier commit: $(git show --stat HEAD | tail -1)"
            log "  - Connexion réseau: $(ping -c 1 github.com > /dev/null 2>&1 && echo 'OK' || echo 'ÉCHEC')"
            log "  - Configuration remote: $(git remote get-url origin)"
            
            # Suggestions
            log "${YELLOW}💡 Suggestions:${NC}"
            log "  1. Vérifiez votre connexion internet"
            log "  2. Vérifiez vos identifiants GitHub"
            log "  3. Le fichier est peut-être trop volumineux"
            log "  4. Essayez: git push origin main --force-with-lease"
            
            exit 1
        fi
    fi
done

# Nettoyage et résumé
log "${BLUE}[$(timestamp)] === Résumé de l'opération ===${NC}"
log "${GREEN}✅ Push terminé avec succès${NC}"
log "📁 Dossier: $(pwd)"
log "📝 Message: Auto-commit du $(timestamp)"
log "🚀 Repository mis à jour sur GitHub"

# Rotation des logs (garder les 10 derniers)
if [ -f "$LOG_FILE" ]; then
    tail -n 1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
fi

log "${BLUE}[$(timestamp)] === Fin du script ===${NC}"