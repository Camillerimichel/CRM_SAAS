# API Import — Inventaires et Mouvements de portefeuilles

## Vue d'ensemble

Deux APIs distinctes permettent d'alimenter les tables historiques du CRM :

| API | Endpoint | Tables alimentées |
|-----|----------|------------------|
| **Inventaire** | `POST /api/import/inventaire/preview` / `.../commit` | `mariadb_historique_support_w`, `mariadb_historique_affaire_w`, `mariadb_historique_personne_w` |
| **Mouvements** | `POST /api/import/mouvements/preview` / `.../commit` | `mouvement`, `avis`, + recalcul des tables historiques ci-dessus |

Chaque endpoint existe en deux variantes :
- **`/preview`** : analyse le fichier, retourne un rapport de validation sans toucher la base
- **`/commit`** : exécute l'import réel avec le pipeline de recalcul complet

---

## API 1 — Inventaire

### Sémantique

Un inventaire est un **snapshot de positions** à une date donnée : pour chaque affaire et chaque support (code ISIN), on fournit le nombre d'unités de compte détenues (`nbuc`) et la valeur liquidative (`vl`) à cette date.

- Le champ `valo` est calculé automatiquement : `valo = nbuc × vl`
- Le champ `prmp` est recalculé depuis l'historique complet des mouvements (voir pipeline)
- L'inventaire **ne génère pas** de mouvement de trésorerie — le champ `mouvement` de `historique_affaire` n'est pas touché

### Format CSV

**En-tête obligatoire :**
```
id_affaire,date,code_isin,nbuc,vl
```

**Règles :**
- Séparateur : virgule `,`
- Encodage : UTF-8
- Dates : `YYYY-MM-DD` ou `DD/MM/YYYY`
- Décimaux : point `.` comme séparateur décimal
- Une ligne par position `(id_affaire, code_isin, date)`

**Exemple :**
```csv
id_affaire,date,code_isin,nbuc,vl
123,2024-01-31,FR0010135103,100.5,98.20
123,2024-01-31,FR0013283075,250.0,115.30
456,2024-01-31,FR0010135103,75.2,98.20
456,2024-01-31,LU0823421333,310.0,204.55
789,2024-01-31,FR0010135103,0,98.20
```

> La ligne avec `nbuc=0` représente une position liquidée — elle sera enregistrée avec `valo=0`.

### Format JSON

**Structure racine :**

```json
{
  "date": "2024-01-31",
  "id_societe_gestion": 1,
  "positions": [
    {
      "id_affaire": 123,
      "code_isin": "FR0010135103",
      "nbuc": 100.5,
      "vl": 98.20
    },
    {
      "id_affaire": 123,
      "code_isin": "FR0013283075",
      "nbuc": 250.0,
      "vl": 115.30
    },
    {
      "id_affaire": 456,
      "code_isin": "FR0010135103",
      "nbuc": 75.2,
      "vl": 98.20
    }
  ]
}
```

**Variante : date par ligne** (si les positions couvrent plusieurs dates) :

```json
{
  "id_societe_gestion": 1,
  "positions": [
    {
      "id_affaire": 123,
      "date": "2024-01-31",
      "code_isin": "FR0010135103",
      "nbuc": 100.5,
      "vl": 98.20
    },
    {
      "id_affaire": 123,
      "date": "2024-02-07",
      "code_isin": "FR0010135103",
      "nbuc": 102.0,
      "vl": 99.10
    }
  ]
}
```

> Si `date` est présente à la fois à la racine et sur la ligne, la date de la ligne est prioritaire.

### Champs

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `id_affaire` | entier | oui | Identifiant de l'affaire (`mariadb_affaires.id`) |
| `date` | date | oui | Date du snapshot (`YYYY-MM-DD`) |
| `code_isin` | chaîne | oui | Code ISIN du support (résolu vers `mariadb_support.id`) |
| `nbuc` | décimal | oui | Nombre d'unités de compte détenues (0 = position soldée) |
| `vl` | décimal | oui | Valeur liquidative à la date |
| `id_societe_gestion` | entier | non | Périmètre société de gestion (filtrage droits) |

---

## API 2 — Mouvements

### Sémantique

Un mouvement est une **transaction** sur un support : souscription, rachat, arbitrage, etc. Chaque ligne correspond à une opération sur un `(id_affaire, code_isin, date)`.

- `nbuc_mvt` : signé — **positif = souscription/achat**, **négatif = rachat/vente**
- `montant_ope` est calculé automatiquement : `montant_ope = nbuc_mvt × vl`
- Le `code_mouvement` détermine le type d'opération (voir table de référence ci-dessous)
- Chaque groupe `(id_affaire, date)` génère un **avis d'opération** dans la table `avis`

### Table de référence des codes mouvement

| Code | Titre | Sens | Affecte l'investi | Affecte le PRMP |
|------|-------|------|-------------------|-----------------|
| `VI`  | Versement initial            | +1  | oui | oui |
| `VC`  | Versement complémentaire     | +1  | oui | oui |
| `VM`  | Versement mensuel            | +1  | oui | oui |
| `AE`  | Arbitrage : entrée           | +1  | non | oui |
| `AS`  | Arbitrage : sortie           | -1  | non | non |
| `RP`  | Rachat partiel               | -1  | oui | non |
| `RT`  | Rachat total                 | -1  | oui | non |
| `TRF` | Transfert de fichier         | +1  | oui | oui |
| `TCE` | Transfert de contrat entrée  | +1  | oui | oui |
| `TCS` | Transfert contrat sortie     | -1  | oui | non |
| `CCE` | Changement contrat entrée    | +1  | oui | oui |
| `CCS` | Changement contrat sortie    | -1  | oui | non |
| `DS`  | Distribution de parts        | +1  | non | non |
| `RET` | Rétrocession                 | +1  | non | non |
| `FG`  | Frais de gestion             | -1  | non | non |
| `FR`  | Frais de rachat              | -1  | non | non |
| `FA`  | Frais sur avances            | -1  | non | non |
| `AV`  | Avances                      |  0  | non | non |
| `RA`  | Remboursement d'avance       |  0  | non | non |
| `RV`  | Revenu                       | -1  | oui | non |

> **Cohérence signe** : `nbuc_mvt > 0` doit correspondre à un code de `sens = +1`, et inversement. Une incohérence est signalée comme erreur de validation.

### Format CSV

**En-tête obligatoire :**
```
id_affaire,date,code_isin,code_mouvement,nbuc_mvt,vl
```

**Règles :**
- Séparateur : virgule `,`
- Encodage : UTF-8
- Dates : `YYYY-MM-DD` ou `DD/MM/YYYY`
- Décimaux : point `.`
- `nbuc_mvt` : signé (positif = entrée, négatif = sortie)
- Plusieurs lignes possibles pour un même `(id_affaire, date)` → elles seront regroupées dans un seul avis

**Exemple :**
```csv
id_affaire,date,code_isin,code_mouvement,nbuc_mvt,vl
123,2024-01-31,FR0010135103,VI,50.0,97.50
123,2024-01-31,FR0013283075,AE,30.0,115.00
123,2024-01-31,FR0010135103,AS,-20.0,97.50
456,2024-01-31,FR0010135103,RP,-75.2,98.10
789,2024-02-15,LU0823421333,VC,100.0,205.00
```

> Les lignes 2 et 3 concernent la même affaire (123) à la même date (2024-01-31) : elles produiront un seul avis d'opération avec deux mouvements.

### Format JSON

**Structure racine :**

```json
{
  "date": "2024-01-31",
  "id_societe_gestion": 1,
  "mouvements": [
    {
      "id_affaire": 123,
      "code_isin": "FR0010135103",
      "code_mouvement": "VI",
      "nbuc_mvt": 50.0,
      "vl": 97.50
    },
    {
      "id_affaire": 123,
      "code_isin": "FR0013283075",
      "code_mouvement": "AE",
      "nbuc_mvt": 30.0,
      "vl": 115.00
    },
    {
      "id_affaire": 456,
      "code_isin": "FR0010135103",
      "code_mouvement": "RP",
      "nbuc_mvt": -75.2,
      "vl": 98.10
    }
  ]
}
```

**Variante : date par ligne** (multi-dates dans un même fichier) :

```json
{
  "id_societe_gestion": 1,
  "mouvements": [
    {
      "id_affaire": 123,
      "date": "2024-01-31",
      "code_isin": "FR0010135103",
      "code_mouvement": "VI",
      "nbuc_mvt": 50.0,
      "vl": 97.50
    },
    {
      "id_affaire": 789,
      "date": "2024-02-15",
      "code_isin": "LU0823421333",
      "code_mouvement": "VC",
      "nbuc_mvt": 100.0,
      "vl": 205.00
    }
  ]
}
```

### Champs

| Champ | Type | Obligatoire | Description |
|-------|------|-------------|-------------|
| `id_affaire` | entier | oui | Identifiant de l'affaire |
| `date` | date | oui | Date de la transaction |
| `code_isin` | chaîne | oui | Code ISIN du support |
| `code_mouvement` | chaîne | oui | Code de la règle mouvement (VI, VC, RP, RT, AE, AS…) |
| `nbuc_mvt` | décimal | oui | Nombre d'UC échangés, signé (+= entrée, -= sortie) |
| `vl` | décimal | oui | Valeur liquidative au jour de la transaction |
| `frais` | décimal | non | Frais éventuels (défaut : 0) |
| `id_societe_gestion` | entier | non | Périmètre société de gestion |

---

## Génération de l'avis d'opération

Pour chaque groupe `(id_affaire, date)` d'un import de mouvements, le système crée automatiquement :

### Table `avis` (un enregistrement par groupe)

| Champ | Valeur |
|-------|--------|
| `reference` | `IMP-{YYYYMMDD}-{id_affaire}` |
| `date` | date des mouvements |
| `id_affaire` | id_affaire |
| `id_etape` | `5` (état "exécuté") |
| `etat` | `importé` |
| `entree` | `SUM(nbuc_mvt × vl)` pour les lignes de sens `+1` |
| `sortie` | `SUM(ABS(nbuc_mvt × vl))` pour les lignes de sens `-1` |
| `commentaire` | `Import automatique {nom_fichier}` |

### Table `mouvement` (un enregistrement par ligne)

| Champ | Valeur |
|-------|--------|
| `id_affaire` | id_affaire |
| `id_mouvement_regle` | résolu depuis `code_mouvement` |
| `id_support` | résolu depuis `code_isin` |
| `id_avis` | FK vers l'avis créé ci-dessus |
| `nb_uc` | `nbuc_mvt` (signé) |
| `vl` | vl de la transaction |
| `montant_ope` | `nbuc_mvt × vl` |
| `frais` | frais (0 si absent) |
| `vl_date` | date de la transaction |
| `date_sp` | date de la transaction |
| `etat` | `importé` |
| `modif_quand` | horodatage serveur |

---

## Pipeline de recalcul post-import

Déclenché automatiquement à la fin de chaque `/commit` (inventaire ou mouvements).

### Étape A — Recalcul PRMP depuis l'historique complet

Pour chaque `(id_affaire, id_support)`, l'ensemble des mouvements de la table `mouvement` est relu en ordre chronologique. Le PRMP est recalculé selon la règle du **Coût Unitaire Moyen Pondéré (CUMP)** :

```
prmp ← 0 ; nbuc ← 0
Pour chaque mouvement ORDER BY date ASC :
  si mouvement_regle.prmp = 1 :
    si nbuc + nbuc_mvt > 0 :
      prmp ← (nbuc × prmp + |nbuc_mvt| × vl) / (nbuc + nbuc_mvt)
    sinon :
      prmp ← 0
  nbuc ← nbuc + nbuc_mvt
  → UPDATE historique_support SET prmp = prmp_courant
    WHERE id_source = id_affaire AND id_support = id_support AND date = date_mvt
```

> Si aucun mouvement n'existe pour une position d'inventaire (première entrée) : `prmp = vl` (initialisation au coût du jour, log d'alerte généré).

### Étape B — Agrégation vers `historique_affaire`

```sql
-- Valuation depuis les positions
UPSERT historique_affaire (id_affaire, date)
SET valo     = SUM(h.nbuc * h.vl)           -- depuis historique_support
    annee    = YEAR(date)

-- Mouvements depuis les transactions (investi ≠ 0)
SET mouvement = SUM(m.montant_ope)
    WHERE mouvement_regle.investi != 0       -- depuis table mouvement
```

### Étape C — Agrégation vers `historique_personne`

```sql
UPSERT historique_personne (id_personne, date)
SET valo      = SUM(ha.valo)      -- sur toutes les affaires du client
    mouvement = SUM(ha.mouvement)
    annee     = YEAR(date)
-- id_personne résolu via mariadb_affaires.id_personne
```

### Étape D — Recalcul SRRI, volatilité et performance 52 semaines

Formule **Modified Dietz** par période :
```
r_t = (V_t - CF_t - V_{t-1}) / (V_{t-1} + 0.5 × CF_t)
```

Dietz cumulé :
```
dietz_t = 1 + SUM(r_1 … r_{t-1})
```

Performance 52 semaines glissantes (ajout dans les CTEs existantes) :
```sql
(dietz_t - LAG(dietz, 52) OVER (PARTITION BY id ORDER BY date))
/ NULLIF(LAG(dietz, 52) OVER (PARTITION BY id ORDER BY date), 0)
  AS perf_sicav_52
```

Volatilité annualisée (52 semaines) :
```sql
STDDEV_SAMP(perf_dietz) OVER (
  PARTITION BY id ORDER BY date
  ROWS BETWEEN 51 PRECEDING AND CURRENT ROW
) * SQRT(52) AS volat
```

Fonctions appelées :
- `_recompute_srri_affaires(db)` → met à jour `historique_affaire.volat`, `historique_affaire.perf_sicav_52`
- `_recompute_srri_clients(db)` → met à jour `historique_personne.volat`, `historique_personne.perf_sicav_52`

### Étape E — Calcul SRI (niveau de risque final)

```python
_store_sri_metrics(db, entity_type="affaire", tempo_table="tempo_hist_affaire_w", source="vev")
_update_sri_current(db, entity_type="affaire")
_store_sri_metrics(db, entity_type="client",  tempo_table="tempo_hist_personne_w", source="vev")
_update_sri_current(db, entity_type="client")
```

Bandes SRI (VeV → niveau 1–7) :
```
volat < 0.5 %  → SRI 1
volat < 2 %    → SRI 2
volat < 5 %    → SRI 3
volat < 10 %   → SRI 4
volat < 15 %   → SRI 5
volat < 25 %   → SRI 6
volat ≥ 25 %   → SRI 7
```

---

## Comportement sur les cas particuliers

| Cas | Comportement |
|-----|-------------|
| `code_isin` inconnu | Création automatique dans `mariadb_support` + entrée dans le log d'alerte |
| Conflit de date (ligne déjà existante) | Écrasement (UPSERT) + entrée dans le log d'alerte |
| `code_mouvement` inconnu | Erreur de validation — ligne rejetée |
| Incohérence signe `nbuc_mvt` / `sens` du code | Erreur de validation — ligne rejetée |
| `id_affaire` inconnu | Erreur de validation — ligne rejetée |
| Première position sans historique mouvement | `prmp = vl` (initialisation) + log d'alerte |
| `nbuc_mvt` amène `nbuc_total` à 0 | Position soldée enregistrée avec `valo = 0` |
| `nbuc_mvt` amènerait `nbuc_total` négatif | Avertissement dans le log, valeur conservée |

---

## Format des réponses API

### `/preview`

```json
{
  "rows_total": 42,
  "rows_valid": 40,
  "rows_error": 2,
  "warnings": [
    "ISIN FR0000000001 inconnu — sera créé automatiquement (ligne 7)",
    "Position (affaire=123, FR0010135103, 2024-01-31) déjà existante — sera écrasée"
  ],
  "errors": [
    "Ligne 15 : code_mouvement 'XX' inconnu",
    "Ligne 22 : id_affaire 9999 introuvable"
  ],
  "preview": {
    "historique_support": [
      {"id_affaire": 123, "code_isin": "FR0010135103", "date": "2024-01-31",
       "nbuc": 100.5, "vl": 98.20, "valo": 9868.10, "prmp": null, "action": "insert"}
    ],
    "historique_affaire": [
      {"id_affaire": 123, "date": "2024-01-31", "valo": 38693.10, "mouvement": 500.00, "action": "upsert"}
    ],
    "avis": [
      {"id_affaire": 123, "date": "2024-01-31", "reference": "IMP-20240131-123",
       "entree": 4875.00, "sortie": 0, "nb_mouvements": 2}
    ]
  }
}
```

### `/commit`

```json
{
  "status": "ok",
  "inserted": {
    "historique_support": 38,
    "historique_affaire": 4,
    "historique_personne": 3,
    "mouvement": 12,
    "avis": 4
  },
  "updated": {
    "historique_support": 2,
    "historique_affaire": 1,
    "historique_personne": 1
  },
  "skipped": 2,
  "recalcul": {
    "affaires_impactees": 4,
    "personnes_impactees": 3,
    "duree_secondes": 1.42
  },
  "alerts": [
    "ISIN FR0000000001 créé automatiquement (support id=187)",
    "Position (affaire=456, FR0010135103, 2024-01-31) écrasée"
  ],
  "errors": []
}
```

---

## Structure des fichiers à créer

| Fichier | Rôle |
|---------|------|
| `src/schemas/import_portefeuille.py` | Pydantic : `InventaireRow`, `MouvementRow`, `PreviewResult`, `CommitResult` |
| `src/services/import_inventaire.py` | Parsing CSV/JSON, validation, UPSERT `historique_support` |
| `src/services/import_mouvements.py` | Parsing CSV/JSON, CUMP, INSERT `mouvement` + `avis` |
| `src/services/recalcul_portefeuille.py` | Pipeline A→E : PRMP, agrégation, SRRI, `perf_52`, SRI |
| `src/api/routes/import_portefeuille.py` | 4 routes FastAPI (preview + commit × 2) |

**Modifications de `dashboard.py` :**
- `_recompute_srri_affaires(db, ids=None)` : ajout de `perf_sicav_52` dans la CTE + paramètre `ids` pour restreindre le recalcul aux affaires impactées
- `_recompute_srri_clients(db, ids=None)` : idem pour les clients

---

## Appel API — exemples

### Upload CSV inventaire

```bash
curl -X POST https://crmsaas.eu/api/import/inventaire/commit \
  -H "Cookie: session=..." \
  -F "file=@inventaire_2024-01-31.csv"
```

### Upload JSON mouvements

```bash
curl -X POST https://crmsaas.eu/api/import/mouvements/preview \
  -H "Cookie: session=..." \
  -H "Content-Type: application/json" \
  -d @mouvements_2024-01-31.json
```

### Paramètres query optionnels

| Paramètre | Type | Description |
|-----------|------|-------------|
| `date` | `YYYY-MM-DD` | Écrase la date contenue dans le fichier |
| `id_societe_gestion` | entier | Filtre le périmètre (écrase la valeur du fichier) |
| `dry_run` | bool | Alias de `/preview` sur les endpoints `/commit` |
