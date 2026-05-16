# Correction des imports fournisseurs — Détection et traitement des variations de valorisation

## Vue d'ensemble

Ce module assure un contrôle hebdomadaire automatisé des historiques de valorisation des affaires clients.
Il détecte les variations de valorisation supérieures à 10%, en recherche les causes probables, applique les corrections possibles, et génère une tâche de contrôle réglementaire par client concerné.

---

## 1. Déclenchement et périmètre

- **Fréquence** : contrôle hebdomadaire, déclenché manuellement depuis l'interface superadmin ou automatiquement
- **Source principale** : `mariadb_historique_affaire_w` — une ligne par affaire par semaine avec `valo`, `mouvement`
- **Source granulaire** : `mariadb_historique_support_w` — valorisation par support (fonds) par semaine avec `nbuc`, `vl`, `valo`
- **Seuil de déclenchement** : variation nette > 10% en valeur absolue (paramètre configurable)
- **Scope** : filtré par `id_societe_gestion`, comme l'ensemble des routes superadmin

---

## 2. Calcul de la variation nette

La variation brute entre deux semaines consécutives est neutralisée des mouvements (versements/retraits) pour mesurer uniquement la variation de valorisation pure :

```
variation_nette = (valo_N - valo_{N-1} - mouvement_N) / valo_{N-1}
```

Si `abs(variation_nette) > seuil` → anomalie détectée sur l'affaire à la date N.

---

## 3. Causes probables et modules de correction

### 3.1 Cause 1 — Nombre de parts incorrect (décalage temporel)

**Origine** : décalage entre la date réelle d'un avis d'opération, la date d'enregistrement du mouvement, et la date du snapshot inventaire. Le `nbuc` dans `historique_support_w` est incorrect pour une ou plusieurs semaines.

**Détection — Module `module_nbuc`** :

1. Reconstituer le `nbuc` théorique pour chaque affaire + support à partir des mouvements enregistrés :
   ```
   nbuc_théorique(date) = nbuc_initial + Σ(mouvements en parts jusqu'à cette date)
   ```
2. Comparer avec le `nbuc` stocké dans `historique_support_w`
3. Si écart non nul et qu'un mouvement existe dans la fenêtre ±2 semaines autour de la date anormale → décalage temporel probable

**Correction** :

- Identifier le mouvement "orphelin" dans la fenêtre ±2 semaines
- Recalculer le `valo` de la semaine affectée avec le `nbuc` corrigé (celui d'avant le mouvement si le mouvement réel est postérieur, et inversement)
- Enregistrer la correction dans `mariadb_correction_historique` avec motif et traçabilité complète
- Ne jamais écraser les données sources sans trace

---

### 3.2 Cause 2 — Valeur liquidative absente ou aberrante

**Origine** : absence de cotation (jour férié, fonds illiquide, défaut de flux fournisseur), ou VL à 0/NULL, ou valeur aberrante sur une seule semaine avant retour à la normale.

**Détection — Module `module_vl`** :

- `vl IS NULL` ou `vl = 0` dans `historique_support_w` pour un support actif (`nbuc > 0`)
- Ou : VL chute à quasi-zéro sur une semaine isolée puis remonte (spike aberrant)

**Hiérarchie de correction** (par ordre de fiabilité décroissante) :

| Priorité | Source | Action |
|---|---|---|
| 1 | `mariadb_support_val` | VL officielle à la date exacte → utilisation directe |
| 2 | `historique_support_w` semaine précédente | Forward-fill (carry forward) — convient aux fonds peu volatils |
| 3 | Interpolation linéaire | Entre la semaine -1 et la semaine +1 si les deux encadrent la lacune |
| 4 | Aucune source disponible | VL non corrigible — flagué `vl_estimee = NULL` dans le log, tâche créée |

---

### 3.3 Analyse contextuelle — Fenêtre semaine -1 / semaine / semaine +1

Pour chaque anomalie détectée sur une affaire à la date D, le module d'analyse :

1. Récupère toutes les lignes de `mariadb_historique_support_w` sur les semaines D-7, D, D+7 pour cette affaire
2. Calcule pour chaque support sa variation de VL entre D-7 et D :
   ```
   delta_vl = (vl_D - vl_{D-7}) / vl_{D-7}
   ```
3. Identifie les supports dont la VL ou le nbuc a le plus varié → supports "coupables"
4. Retourne : support incriminé, VL avant/après, nbuc avant/après, contribution à la variation totale de l'affaire

---

## 4. Recalcul post-correction — Module `module_recalcul`

Après application des corrections issues des modules 3.1 et 3.2 :

1. Recalculer `valo = nbuc_corrigé × vl_corrigée` pour chaque support + date concernée
2. Ré-agréger à l'affaire : `valo_affaire = Σ(valo_support)`
3. Recalculer la variation nette avec les valeurs corrigées
4. Décision :
   - **Variation corrigée ≤ seuil** → anomalie auto-résolue, log "résolu par [motif]", pas de tâche créée
   - **Variation corrigée > seuil** → anomalie persistante, création de la tâche réglementaire

---

## 5. Génération des tâches de contrôle

### 5.1 Type d'événement

- **Catégorie** : `Réglementaire`
- **Libellé** : `Variation de cours importante`
- Créé via `ensure_type(libelle="Variation de cours importante", categorie="Réglementaire")` — créé s'il n'existe pas, récupéré sinon

### 5.2 Règle de regroupement

**Une seule tâche par client**, même si plusieurs affaires sont touchées. Le détail de chaque affaire est inclus dans le commentaire. Objectif : éviter le spam de tâches pour un même client.

### 5.3 Contenu de la tâche

| Champ | Valeur |
|---|---|
| `client_id` | `id_personne` du client |
| `affaire_id` | `null` (ou l'affaire la plus significative si une seule) |
| `statut` | `à faire` |
| `date_evenement` | date d'exécution du contrôle |
| `commentaire` | Détail des anomalies par affaire (voir format ci-dessous) |

**Format du commentaire** :
```
Variation de cours détectée > 10% sur les affaires suivantes :
- [REF-001] Semaine 2025-04-07 : +14,3% (98 500 € → 112 600 €) — Support: Fonds XYZ (VL aberrante, corrigée par forward-fill)
- [REF-002] Semaine 2025-03-24 : -11,2% (75 000 € → 66 600 €) — Cause non résolue automatiquement
Contrôle manuel requis.
```

---

## 6. Log interactif (SSE)

Le service expose un générateur SSE pour un retour en temps réel dans l'interface. Format des événements :

```
→ [SCAN]    Affaire #1234 (REF-001) — Client: Dupont Jean
→ [ANOMALIE] Semaine 2025-04-07 : variation nette +14,3%  (98 500 → 112 600 €)
→ [CAUSE]   Support "Fonds XYZ FR0012345678" — VL 105,20 → 120,10 (+14,2%) — nbuc stable
→ [CORRECTION] VL corrigée par forward-fill (source: mariadb_historique_support_w S-1)
→ [RÉSOLU]  Variation post-correction : +0,8% — sous le seuil, aucune tâche créée
---
→ [SCAN]    Affaire #1235 (REF-002) — Client: Dupont Jean
→ [ANOMALIE] Semaine 2025-03-24 : variation nette -11,2%
→ [CAUSE]   nbuc anormal : 150 parts au lieu de 120 — mouvement trouvé en S+1 (décalage)
→ [CORRECTION] nbuc recalculé à 120 — valo corrigée 66 600 → 75 200 €
→ [RÉSOLU]  Variation post-correction : -0,4% — sous le seuil
---
→ [TÂCHE]   Créée pour client Dupont Jean (2 affaires, 0 anomalie persistante) — résumé inclus
→ [OK]      Affaire #1236 — aucune anomalie
...
→ [RÉSUMÉ]  5 affaires scannées — 2 anomalies détectées — 2 auto-résolues — 0 tâche créée
```

---

## 7. Traçabilité — Table `mariadb_correction_historique`

Toutes les corrections appliquées automatiquement sont enregistrées pour audit :

```sql
CREATE TABLE mariadb_correction_historique (
  id               INT AUTO_INCREMENT PRIMARY KEY,
  id_affaire       INT,
  id_support       INT,
  date_semaine     TIMESTAMP,
  motif            ENUM('nbuc_decalage', 'vl_manquante', 'vl_aberrante'),
  champ_corrige    ENUM('nbuc', 'vl', 'valo'),
  valeur_avant     DECIMAL(18,6),
  valeur_apres     DECIMAL(18,6),
  source_correction VARCHAR(100),  -- 'forward_fill', 'support_val', 'interpolation', 'recalcul_mouvement'
  corrige_quand    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  id_societe_gestion INT
);
```

---

## 8. Structure des fichiers

| Fichier | Rôle |
|---|---|
| `src/services/detect_variation_cours.py` | Orchestrateur principal — générateur SSE, boucle de détection |
| `src/services/module_nbuc.py` | Détection et correction des décalages de parts |
| `src/services/module_vl.py` | Détection et correction des VL manquantes ou aberrantes |
| `src/services/module_recalcul.py` | Recalcul post-correction, décision tâche ou résolu |
| `src/api/???` | Route API (à confirmer) |
| `src/api/templates/???` | Interface log interactif (à confirmer) |

---

## 9. Logique de décision synthétique

```
Pour chaque affaire avec variation > seuil :
  ├── module_vl    → VL NULL / aberrante ?
  │     └── Oui : corriger (hiérarchie support_val → forward-fill → interpolation) + recalcul
  ├── module_nbuc  → nbuc incohérent vs mouvements ?
  │     └── Oui : corriger nbuc dans fenêtre ±2 semaines + recalcul
  └── module_recalcul → variation post-correction ?
        ├── ≤ seuil → log "auto-résolu [motif]"
        └── > seuil → log "non résolu" + tâche Réglementaire / Variation de cours importante
```

---

## 10. Points ouverts

- Emplacement de la route API et du template HTML (à confirmer avant développement)
- Déclenchement : manuel uniquement ou aussi automatique (cron hebdomadaire) ?
- Que faire si aucune VL n'est récupérable (priorité 4) : bloquer la correction ou créer la tâche immédiatement ?
- Seuil configurable par société de gestion ou global ?
