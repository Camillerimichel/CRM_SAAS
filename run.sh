#!/bin/bash
# Activer l'environnement
source .venv/bin/activate

# Mettre à jour pip et outils de build
pip install --upgrade pip setuptools wheel -i https://pypi.org/simple

# Installer dépendances en utilisant cache local et miroir rapide
pip install -r requirements.txt --cache-dir=.pip_cache -i https://mirror.inria.fr/pypi/web/simple

# Lancer le serveur FastAPI
uvicorn src.api.main:app --reload
