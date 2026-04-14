# Recette V2

Ce document prepare une recette fonctionnelle et technique de la V2 hierarchique.

## Objectif

Verifier en environnement de dev ou de recette que les quatre niveaux suivants se comportent correctement :

- `co_courtier`
- `master_courtier`
- `delegation_regionale`
- `superadmin`

La V2 doit permettre :

- une hierarchie explicite entre societes de gestion
- une consolidation descendante des KPI
- une administration de cette hierarchie
- des vues adaptees selon le niveau actif

## Prerequis

- disposer d'un environnement non productif
- avoir applique les migrations jusqu'a `migrations/20260414_societe_hierarchy_v2.sql`
- avoir un acces applicatif a `/dashboard/superadmin`
- avoir un compte `superadmin` capable de creer des comptes d'authentification et d'assigner des roles

## Donnees de demonstration

Le script [data/recette_v2_hierarchy_seed.sql](/var/www/CRM_SAAS/data/recette_v2_hierarchy_seed.sql:1) cree une hierarchie de test non destructive prefixee `RECETTE V2 -` :

- `RECETTE V2 - Delegation Nord`
- `RECETTE V2 - Master Lille`
- `RECETTE V2 - Master Amiens`
- `RECETTE V2 - Co Lille Centre`
- `RECETTE V2 - Co Lille Est`
- `RECETTE V2 - Co Amiens Centre`

Il reconstruit ensuite `mariadb_societe_hierarchy`.

Le script [data/recette_v2_business_seed.sql](/var/www/CRM_SAAS/data/recette_v2_business_seed.sql:1) ajoute un jeu minimal de donnees metier rattache a cette hierarchie :

- 4 clients
- 5 affaires
- 3 co-courtiers sources

Volumes attendus apres seed metier :

- `RECETTE V2 - Co Lille Centre` : 2 clients, 2 affaires
- `RECETTE V2 - Co Lille Est` : 1 client, 1 affaire
- `RECETTE V2 - Co Amiens Centre` : 1 client, 2 affaires
- `RECETTE V2 - Master Lille` : 3 clients, 3 affaires
- `RECETTE V2 - Master Amiens` : 1 client, 2 affaires
- `RECETTE V2 - Delegation Nord` : 4 clients, 5 affaires

Execution type :

```bash
mysql -u <user> -p <database> < data/recette_v2_hierarchy_seed.sql
mysql -u <user> -p <database> < data/recette_v2_business_seed.sql
```

## Mise en place de la recette

1. Appliquer la migration V2.
2. Executer le seed de hierarchie.
3. Executer le seed metier.
4. Ouvrir `/dashboard/superadmin`.
5. Verifier dans la liste des societes que les six societes `RECETTE V2 - ...` sont presentes avec le bon parent.
6. Creer ou reutiliser quatre utilisateurs RH de test.
7. Creer leurs comptes d'authentification depuis `superadmin`.
8. Assigner les roles et portees suivantes :

- utilisateur A : role `commercial` sur `RECETTE V2 - Co Lille Centre`
- utilisateur B : role `directeur_commercial` sur `RECETTE V2 - Master Lille`
- utilisateur C : role `dirigeant` ou `directeur_commercial` sur `RECETTE V2 - Delegation Nord`
- utilisateur D : role `superadmin` en portee globale

## Parcours a verifier

### 1. Co-courtier

Connexion avec l'utilisateur A.

Points attendus :

- la sidebar affiche `co-courtier`
- l'accueil reste oriente portefeuille et operationnel
- `clients` et `affaires` ne remontent que le perimetre du co-courtier
- `RECETTE V2 - Co Lille Centre` remonte 2 clients et 2 affaires
- la page `pilotage reseau` n'apparait pas

### 2. Master courtier

Connexion avec l'utilisateur B.

Points attendus :

- la sidebar affiche `master courtier`
- l'accueil affiche le contexte reseau
- la page `pilotage reseau` est accessible
- les KPI consolidés couvrent le master et ses co-courtiers descendants
- `RECETTE V2 - Master Lille` remonte 3 clients et 3 affaires
- dans `clients` et `affaires`, la colonne `Societe source` est visible
- depuis `pilotage reseau`, les liens de detail ouvrent les listes filtrees par `societe_source_id`

### 3. Delegation regionale

Connexion avec l'utilisateur C.

Points attendus :

- la sidebar affiche `delegation regionale`
- l'accueil masque les blocs trop operationnels
- `pilotage reseau` est accessible
- les KPI sont consolides sur tous les masters et co-courtiers descendants
- `RECETTE V2 - Delegation Nord` remonte 4 clients et 5 affaires
- la lecture doit rester orientee pilotage, sans surcharge detaillee inutile

### 4. Superadmin

Connexion avec l'utilisateur D.

Points attendus :

- acces complet a `/dashboard/superadmin`
- creation, modification et suppression d'une societe de gestion
- impossibilite de creer une boucle parent/enfant
- impossibilite de supprimer une societe ayant encore des filles
- creation d'un compte d'authentification RH
- affectation et suppression d'un role sur une portee societe ou globale

## Controles SQL utiles

Verifier la hierarchie :

```sql
SELECT
  child.id,
  child.nom,
  child.organisation_level,
  parent.nom AS parent_nom
FROM mariadb_societe_gestion child
LEFT JOIN mariadb_societe_gestion parent ON parent.id = child.parent_societe_id
WHERE child.nom LIKE 'RECETTE V2 - %'
ORDER BY child.nom;
```

Verifier la fermeture :

```sql
SELECT
  a.nom AS ancestor_nom,
  d.nom AS descendant_nom,
  h.depth
FROM mariadb_societe_hierarchy h
JOIN mariadb_societe_gestion a ON a.id = h.ancestor_societe_id
JOIN mariadb_societe_gestion d ON d.id = h.descendant_societe_id
WHERE a.nom LIKE 'RECETTE V2 - %'
  AND d.nom LIKE 'RECETTE V2 - %'
ORDER BY a.nom, h.depth, d.nom;
```

Verifier les roles staff :

```sql
SELECT
  au.login,
  ar.code AS role_code,
  sg.nom AS societe_nom
FROM auth_user_roles aur
JOIN auth_users au ON au.id = aur.user_id
JOIN auth_roles ar ON ar.id = aur.role_id
LEFT JOIN mariadb_societe_gestion sg ON sg.id = aur.societe_id
WHERE au.user_type = 'staff'
ORDER BY au.login, ar.code, sg.nom;
```

Verifier les volumes metier de recette :

```sql
SELECT
  sg.nom,
  (
    SELECT COUNT(DISTINCT cs.client_id)
    FROM mariadb_client_societe cs
    WHERE cs.societe_id IN (
      SELECT h.descendant_societe_id
      FROM mariadb_societe_hierarchy h
      WHERE h.ancestor_societe_id = sg.id
    )
      AND cs.date_fin IS NULL
  ) AS clients_total,
  (
    SELECT COUNT(DISTINCT afs.affaire_id)
    FROM mariadb_affaire_societe afs
    WHERE afs.societe_id IN (
      SELECT h.descendant_societe_id
      FROM mariadb_societe_hierarchy h
      WHERE h.ancestor_societe_id = sg.id
    )
      AND afs.date_fin IS NULL
  ) AS affaires_total
FROM mariadb_societe_gestion sg
WHERE sg.nom LIKE 'RECETTE V2 - %'
ORDER BY sg.nom;
```

## Resultat attendu

La recette est consideree comme bonne si :

- chaque niveau voit le bon perimetre
- les KPI consolides varient selon la racine selectionnee
- `superadmin` garde une vision transverse complete
- les liens `pilotage reseau -> clients/affaires` restent coherents
- aucune fuite de donnees n'apparait entre branches soeurs de l'arbre

## Limites connues

- le seed metier couvre les clients et affaires, pas encore les taches ouvertes ni les reclamations
- la V2 livre deja le socle, mais la recette doit encore couvrir les ecrans secondaires non passes en revue manuellement
