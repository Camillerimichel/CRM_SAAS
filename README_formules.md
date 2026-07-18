# Formules de calcul

## 1. Scénarios SRI avec frais (lettre d'adéquation — chapitre 7)

### Contexte

La lettre d'adéquation présente deux tableaux de scénarios de performance :
- **Tableau brut** : valeur sans frais, issue des scénarios SRI (Cornish-Fisher)
- **Tableau avec frais** : valeur nette après déduction des frais d'entrée et de gestion

### Paramètres

| Paramètre | Notation | Source |
|---|---|---|
| Capital de référence | `base = 10 000 €` | Fixe (convention SRI) |
| Frais d'entrée | `fe` (%) | Saisie KYC — table `KYC_frais_entree` |
| Frais de gestion moyen | `mg` (% / an) | Moyenne des `total_frais` des contrats sélectionnés |
| Facteur de croissance brut | `f` | Issu des scénarios SRI pour l'horizon considéré |
| Nombre d'années | `N` | Label de l'horizon (1, 3 ou 5 ans) |
| Facteur annuel | `r = f^(1/N)` | Taux annualisé du scénario |

### Calcul itératif année par année

Les frais de gestion sont prélevés **annuellement** sur la **base moyenne Jan–Déc** de chaque année.
La base de l'année N = fin de l'année (N−1) − frais prélevés en (N−1).

**Initialisation :**
```
C₀ = base × (1 − fe)          # capital investi après frais d'entrée
```

**Pour chaque année k = 1 … N :**
```
fin_brute_k  = C_{k-1} × r
moyenne_k    = (C_{k-1} + fin_brute_k) / 2
frais_k      = moyenne_k × mg
C_k          = fin_brute_k − frais_k   # base de l'année suivante
```

**Résultats :**
```
valeur_nette    = C_N
frais_gestion   = Σ frais_k  (k = 1 … N)
frais_totaux    = base × fe + frais_gestion
taux_rendement  = (valeur_nette − base) / base × 100
```

### Propriétés

- Les frais de gestion **varient par scénario** : un portefeuille favorable génère une base plus élevée → frais plus élevés.
- La valeur nette `C_N ≠ valeur_brute − frais_totaux` car les frais réduisent la base de croissance des années suivantes (effet de capitalisation).
- Pour N = 1 : `frais_gestion = (base × (1 − fe) + base × (1 − fe) × f) / 2 × mg`

### Exemple — Client 1869, Scénario Favorable sur 5 ans

- `fe = 2,5 %`, `mg = 1 %/an`, `f = 1,4598`, `r = 1,07859`, `C₀ = 9 750 €`

| Année | Base début | Fin brute | Moyenne | Frais | Base suivante |
|---|---|---|---|---|---|
| 1 | 9 750 € | 10 516 € | 10 133 € | 101 € | 10 415 € |
| 2 | 10 415 € | 11 233 € | 10 824 € | 108 € | 11 125 € |
| 3 | 11 125 € | 11 999 € | 11 562 € | 116 € | 11 884 € |
| 4 | 11 884 € | 12 818 € | 12 351 € | 124 € | 12 694 € |
| 5 | 12 694 € | 13 692 € | 13 193 € | 132 € | 13 560 € |

→ **Net : 13 560 €** (+35,6 %) | Frais gestion : 581 € + Entrée : 250 € = **831 € total**
