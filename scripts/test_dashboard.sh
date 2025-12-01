#!/bin/bash
set -euo pipefail
BASE_URL="http://localhost:8100"
COOKIE_JAR="/tmp/dashboard_cookies.txt"

read -p "Login (email ou user) : " LOGIN
read -s -p "Mot de passe : " PASSWORD
echo

curl -c "$COOKIE_JAR" -X POST \
  -d "username=${LOGIN}&password=${PASSWORD}" \
  "$BASE_URL/dashboard/login"

echo -e "\n--- Session obtenue, cookies stockés dans $COOKIE_JAR ---"

read -p "ID relation à mettre à jour : " ROW_ID
read -p "ID société (societe_id) : " SOC_ID

curl -b "$COOKIE_JAR" -X POST \
  -F "societe_id=${SOC_ID}" \
  "$BASE_URL/dashboard/parametres/courtier/relation/${ROW_ID}"
