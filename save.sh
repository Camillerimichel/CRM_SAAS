#!/bin/bash

# Vérifie qu'un message est fourni
if [ -z "$1" ]; then
  echo "❌ Merci de donner un message de commit."
  echo "Usage: ./save.sh \"Ton message de commit\""
  exit 1
fi

# Sauvegarde avec le message
git add .
git commit -m "$1"
git push

echo "✅ Sauvegarde envoyée avec le message : $1"
