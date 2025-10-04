---

## Pages suivantes

### En-têtes de section

* Police Helvetica, **gras**, taille 16.

### Bas de page

* Ligne horizontale.
* Une ligne de texte centrée, Helvetica normale, taille 11.
* Pagination : `Page X / Y`.

## Feuille de styles

### Listes numérotées (sommaire et sections)

* Helvetica, **gras**, taille 14.

### Titre niveau 1

- Helvetica, normale, taille 16, alignement **gauche**.
- Ligne horizontale complète

### Titre niveau 2

- Helvetica, normale, taille 14, alignement **gauche**.

### Titre niveau 3

- Helvetica, normale, taille 12, alignement **gauche**.

### Paragraphes

* Helvetica, normale, taille 12, alignement **justifié**.

### Tableaux

* **En-tête** : Helvetica, **gras**, taille 12, fond bleu clair, police bleue (couleur du site).
* **Cellules** : Helvetica, normale, taille 12, fond blanc.

---

# Rapport de prise de connaissance

* **Logo** : centré, placé au tiers supérieur de la page.
* **Titre** :
  * Texte : `Compte rendu d’entretien avec {{ client.prenom }} {{ client.nom }}`
  * Police Helvetica, **gras**, taille 16, centré.
* **Date** :
  * Format `jj/mm/aaaa` (date du jour).
  * Helvetica, **gras italique**, taille 12, centré.
* **Bas de page** :
  * Ligne horizontale.
  * 4 lignes de texte centrées, Helvetica normale, taille 11 (par ex. mentions légales ou coordonnées du cabinet).

---

## Devoir de connaisance du client {Titre niveau 1}

Conformément à la réglementation applicable aux courtiers en assurance et en investissements financiers, le professionnel a l’obligation de recueillir et d’analyser les informations nécessaires pour s’assurer que les conseils fournis correspondent à la situation, aux objectifs et au profil de risque de son client.

Cette obligation découle :

- du Code monétaire et financier, notamment les articles L.541-8-1 et suivants ;
- de la Directive européenne MIFID II (Directive 2014/65/UE) ;
- du Code des assurances, articles L.520-1 et L.521-4 ;
- des textes d’application précisant le recueil d’informations sur la situation financière, les connaissances, l’expérience en matière d’investissement, les objectifs patrimoniaux, la tolérance au risque et l’horizon d’investissement.

Les informations fournies par le client constituent la base indispensable à la formulation de toute recommandation personnalisée. Le rapport d’entretien qui suit s’inscrit dans ce cadre légal.

---

## Sommaire {Titre niveau 2}

1. État civil
2. Patrimoine et revenus
3. Connaissances financières
4. Sensibilité ESG
5. Objectifs
6. Page de fin

---

### 1. État civil {Titre niveau 1}

Issu de Récapitulatif => récapitulatif enregistré du bloc "Etat civil"

* Adresse principale : {{ client.adresse }}
* Situation matrimoniale : {{ client.matrimonial }}
* Situation professionnelle : {{ client.profession }}

### 2. Patrimoine et revenus {Titre niveau 1}

Issus de Récapitulatif => récapitulatif enregistré du bloc "Patrimoine et revenus"

* Tableaux des actifs / passifs / recettes / charges
* Graphiques Pie-Charts détail actifs / passifs / recettes / charges

### 3. Connaissances financières {Titre niveau 1}

Issues de Récapitulatif => récapitulatif enregistré du bloc "Connaissance client"

* Récapitulatif enregistré le {{ date_test_connaissance }}
* Texte de l’offre financière adaptée
* Graphique du profil de risque

### 4. Sensibilité ESG {Titre niveau 1}

Issue de Récapitulatif => récapitulatif enregistré du bloc "Sensibilité ESG"

* Sensibilité : Réponses au bloc "Sensibilité ESG"
* Exclusions : Réponses au bloc "Exclusions ""
* Indicateurs de suivi : Réponses au bloc "Indicateurs "

### 5. Objectifs {Titre niveau 1}

Issus de Récapitulatif => récapitulatif enregistré du bloc "Objectifs"

* Tableau listant les objectifs avec : priorité, titre, horizon, commentaire

### Page de fin

* Logo centré
* Informations du cabinet en dessous (nom, coordonnées, numéro ORIAS, mentions légales)

---

👉 Ce squelette est prêt à être décliné en **modèle Word (docxtpl)** ou **modèle HTML (WeasyPrint)**.

Je veux que tu génères directement la **version HTML complète avec CSS inline** (donc utilisable immédiatement avec Jinja2 + WeasyPrint), et un **modèle Word (.docx) avec placeholders** ?