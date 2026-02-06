# Champs à renseigner (zones grises) – Autocertification FATCA / EAI

Ce fichier sert de **guide** (nom de champ → quoi mettre → exemple).  
Les valeurs d'exemple sont **fictives** : remplace-les par les données du client.

## En-tête

- `interlocuteur_commercial` : nom/prénom du conseiller — ex. `Camille Martin`
- `produit_nom` : libellé du produit — ex. `LA MONDIALE Partenaires – Harmonis Vie`
- `compagnie_assurance` : nom de la compagnie — ex. `La Mondiale Partenaires`
- `date_operation` : date (format libre) — ex. `06/02/2026`
- `operation_souscription` : checkbox (true/false)
- `operation_autre` : checkbox (true/false)

## Souscripteur 1 (personne physique)

Préfixe champs : `souscripteur1_`

- `civilite` : `M.` / `Mme` / `Mlle`
- `nom`, `nom_jeune_fille`
- `prenom_usage`, `prenom_etat_civil`
- `date_naissance` : ex. `01/01/1980`
- `ville_naissance`, `pays_naissance`, `code_postal_naissance`
- `adresse_rue`, `adresse_cp_ville`, `adresse_pays`
- `est_residence_fiscale_oui`, `est_residence_fiscale_non` : checkboxes (true/false)
- `residence_fiscale_si_differente` : adresse (si différente)
- `residence_fiscale_pays` : pays (si différent)

## Autre souscripteur (personne physique)

Préfixe champs : `autre_souscripteur_`  
Même liste de champs que pour `souscripteur1_`.

## Souscripteur (personne morale)

- `personne_morale_raison_sociale`
- `personne_morale_forme_juridique`
- `personne_morale_siret`
- `personne_morale_ape`
- `personne_morale_represente_par` : nom/prénom du représentant
- `personne_morale_agissant_en_qualite_de` : qualité/fonction

## Bénéficiaire

Préfixe champs : `beneficiaire_`

- `civilite`
- `nom`, `nom_jeune_fille`
- `prenom_usage`, `prenom_etat_civil`
- `date_naissance`, `ville_naissance`, `pays_naissance`, `code_postal_naissance`

## EAI (domiciliation fiscale)

- `eai_souscripteur_resident_france_oui`, `eai_souscripteur_resident_france_non` : checkboxes
- `eai_souscripteur_pays` : pays de résidence fiscale
- `eai_souscripteur_nif` : NIF (numéro d'identification fiscale)
- `eai_autre_souscripteur_resident_france_oui`, `eai_autre_souscripteur_resident_france_non` : checkboxes
- `eai_autre_souscripteur_pays`
- `eai_autre_souscripteur_nif`

## FATCA

- `fatca_souscripteur_us_person_oui`, `fatca_souscripteur_us_person_non` : checkboxes
- `fatca_beneficiaire_us_person_oui`, `fatca_beneficiaire_us_person_non` : checkboxes
- `fatca_autre_souscripteur_us_person_oui`, `fatca_autre_souscripteur_us_person_non` : checkboxes

