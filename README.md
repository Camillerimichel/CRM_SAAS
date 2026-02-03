# 📌 CRM SAAS – État du projet

## ✅ Ce qui est déjà fait

### 1. Base de données
- Base MariaDB/MySQL
- Tables recréées et propres :  
  - `mariadb_clients`  
  - `mariadb_affaires`  
  - `Documents`  
  - `Documents_client`  
  - `mariadb_support`  
  - `mariadb_historique_affaire_w`  
  - `mariadb_historique_personne_w`  
  - `mariadb_historique_support_w`  
  - `Allocations`

### 2. Backend (FastAPI)
- Modèles SQLAlchemy alignés sur MariaDB/MySQL  
- Schémas Pydantic complets (lecture + création)  
- Services refactorisés avec injection `db: Session = Depends(get_db)`  
- Routes unifiées dans `main.py` (API claire et sans doublons)  
- Tests `GET` et `POST` validés avec `curl`  

---

## 📂 Endpoints disponibles

### 🔹 Clients
- `GET /clients`
- `GET /clients/{id}`
- `POST /clients`

### 🔹 Affaires
- `GET /affaires`
- `GET /affaires/{id}`
- `POST /affaires`

### 🔹 Documents
- `GET /documents`
- `GET /documents/{id}`
- `POST /documents`

### 🔹 Documents par client
- `GET /documents-client/{client_id}`
- `GET /documents-client/{client_id}/{document_id}`
- `POST /documents-client`

### 🔹 Supports
- `GET /supports`
- `GET /supports/{id}`
- `POST /supports`

### 🔹 Historiques
- `GET /historiques/personne/{id}`
- `POST /historiques/personne`
- `GET /historiques/affaire/{id}`
- `POST /historiques/affaire`
- `GET /historiques/support/{id}`
- `POST /historiques/support`

### 🔹 Reporting
- `GET /reporting/clients`
- `GET /reporting/top-clients?limit={n}`
- `GET /reporting/affaires`
- `GET /reporting/allocations`
- `GET /reporting/supports`

---

## 🧰 Maintenance

Guide rapide nginx + uvicorn :
- `config/README_maintenance.md`

---

## ⏸️ Reste à faire

1. **Nettoyage** : supprimer reliquats/doublons éventuels dans `src/services`
2. **Tests Reporting** : vérifier cohérence des résultats avec des données réelles
3. **Endpoints complémentaires** : ajouter `PUT` et `DELETE` si CRUD complet souhaité
4. **Documentation** : écrire un `README.md` avec guide lancement + exemples `curl`

---

## 🚀 Exemple de lancement

```bash
./run.sh

#!/bin/bash
# Activer l'environnement
source .venv/bin/activate

# Installer dépendances (optionnel si déjà faites une fois)
pip install -r requirements.txt

# Lancer le serveur
uvicorn src.api.main:app --host 0.0.0.0 --port 8100 --reload


```

API disponible sur : [http://72.61.94.45:8100/docs](http://72.61.94.45:8100/docs)

---

## 🖥️ Frontend (React) – Port 8101

```bash
cd frontend
npm install          # première installation
npm run build        # génère le dossier build/
PORT=8101 npm run serve
```

Le serveur statique Node intégré à `npm run serve` diffuse `build/` sur [http://72.61.94.45:8101](http://72.61.94.45:8101). L’API et l’app partagent désormais les IP/ports réservés 8100/8101.

---

## 📊 Exemple de tests Reporting avec `curl`

```bash
# Tous les clients
curl -X GET "http://72.61.94.45:8100/reporting/clients"

# Top 5 clients
curl -X GET "http://72.61.94.45:8100/reporting/top-clients?limit=5"

# Toutes les affaires
curl -X GET "http://72.61.94.45:8100/reporting/affaires"

# Toutes les allocations
curl -X GET "http://72.61.94.45:8100/reporting/allocations"

# Tous les supports
curl -X GET "http://72.61.94.45:8100/reporting/supports"
```

---

✅ L’API est **fonctionnelle et stable**.  
Il ne reste que les points optionnels pour finaliser un produit complet (CRUD + documentation).
