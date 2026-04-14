# README evol V2

## Objet

Ce document pose le cadre de la V2 du CRM pour supporter une organisation commerciale a 4 niveaux :

- co-courtier
- master courtier
- delegation regionale
- superadministrateur

L'objectif est de faire evoluer la base de donnees, les regles d'acces et les ecrans de pilotage sans casser le fonctionnement actuel.

## Contexte actuel

L'application dispose deja des briques suivantes :

- une table `mariadb_societe_gestion` pour representer les societes
- des liens `mariadb_client_societe` et `mariadb_affaire_societe`
- un RBAC via `auth_roles`, `auth_permissions`, `auth_user_roles`
- un filtrage par `societe_id`
- un acces global pour le superadmin

Le point bloquant actuel est que le produit est concu autour d'une societe active unique. Il n'existe pas encore de hierarchie metier entre societes, ni de consolidation native par niveau parent.

## Principe directeur de la V2

La V2 doit conserver integralement la structure actuelle et l'etendre.

Le modele cible n'est donc pas une remise a plat. Il repose sur :

- la conservation des courtiers actuels
- l'ajout des niveaux `master courtier` et `delegation regionale`
- le maintien du `superadministrateur` comme role global transverse

Le superadministrateur ne doit pas etre traite comme un niveau commercial. Il doit rester un role global permettant d'acceder a l'integralite des donnees consolidees pour preparer les operations commerciales et piloter l'ensemble.

## Cible fonctionnelle

### 1. Co-courtier

Le co-courtier voit :

- ses clients
- ses affaires
- ses KPI propres

### 2. Master courtier

Le master courtier voit :

- ses co-courtiers rattaches
- les clients de ses co-courtiers
- les KPI consolides de son perimetre

### 3. Delegation regionale

La delegation regionale voit :

- ses master courtiers rattaches
- les KPI consolides de ses master courtiers

Par defaut, ce niveau doit etre oriente pilotage et consolidation. L'acces au detail client doit rester maitrise et explicitement arbitre par les regles metier.

### 4. Superadministrateur

Le superadministrateur voit :

- tous les niveaux
- tous les KPI consolides
- l'integralite des donnees utiles a l'administration et au pilotage transverse

## Cible technique

### 1. Hierarchie organisationnelle

La table `mariadb_societe_gestion` reste la table centrale.

Elle devra etre enrichie avec :

- un identifiant de parent, par exemple `parent_societe_id`
- un niveau d'organisation, par exemple `organisation_level`

Valeurs cibles possibles pour `organisation_level` :

- `co_courtier`
- `master_courtier`
- `delegation_regionale`

Le `superadmin` ne sera pas porte par cette colonne, car il s'agit d'un role RBAC global et non d'un noeud de la hierarchie.

### 2. Rattachement des clients et affaires

Les clients et affaires resteront rattaches au niveau operationnel, c'est-a-dire au co-courtier.

La visibilite des niveaux superieurs sera derivee par la hierarchie :

- un master voit les donnees de ses descendants
- une delegation voit les donnees de ses descendants
- un superadmin voit tout

### 3. Permissions

Le RBAC actuel doit etre conserve et enrichi.

Il faudra introduire au minimum :

- des roles adaptes aux nouveaux niveaux
- une notion de consultation consolidee des KPI
- une gestion de perimetre par descendants et non plus uniquement par `societe_id` exact

### 4. Filtrage applicatif

Le filtrage actuel repose sur une societe active unique.

La V2 devra faire evoluer ce principe vers :

- une societe racine selectionnee
- un ensemble de societes autorisees calcule a partir de la hierarchie

## Impacts attendus

### Base de donnees

- ajout de colonnes de hierarchie sur `mariadb_societe_gestion`
- eventuellement ajout d'une table de fermeture hierarchique si necessaire pour les performances
- migration des societes existantes sans rupture

### Backend

- evolution des helpers RBAC
- evolution du contexte de societe courant
- refonte des filtres pour gerer les descendants
- adaptation des endpoints de dashboard et de reporting

### Frontend / templates

- adaptation des ecrans selon le niveau connecte
- ajout de vues consolidees pour master et delegation
- maintien des vues actuelles pour les niveaux deja en place

## Strategie de mise en oeuvre

### Phase 1. Documentation et cadrage

- formaliser l'etat actuel
- formaliser la cible V2
- cadrer les impacts schema, securite et ecrans

### Phase 2. Evolution du schema

- ajouter la hierarchie organisationnelle
- preparer les scripts de migration et de backfill
- garantir la compatibilite ascendante

### Phase 3. Evolution du controle d'acces

- etendre les roles
- gerer les scopes descendants
- conserver le mode global du superadmin

### Phase 4. Evolution des KPI et des listes

- consolider les indicateurs par perimetre
- mettre a jour les requetes de listing clients, affaires et utilisateurs
- eviter les doublons et les ruptures de perimetre

### Phase 5. Evolution des ecrans

- ajouter les vues master
- ajouter les vues delegation regionale
- conserver les ecrans actuels pour les courtiers existants

### Phase 6. Validation

- tests de droits
- tests de non regression
- tests de consolidation
- verification des parcours de connexion et de navigation

## Decision de conception retenue

La V2 sera construite en extension de l'existant, sans suppression de la structure actuelle.

Decision retenue :

- on conserve la structure actuelle integralement
- on ajoute les deux niveaux manquants
- on maintient le superadmin en role global transverse
- on fait porter la logique de consolidation par une vraie hierarchie organisationnelle

## Prochaines etapes

Les prochaines etapes de travail sont :

1. creer la migration SQL de hierarchie
2. faire evoluer les modeles SQLAlchemy
3. refondre le calcul du scope de visibilite
4. adapter les dashboards et KPI
5. ajouter les tests de droits et de consolidation

## Etat de livraison V2

La V2 a maintenant un premier socle fonctionnel livre dans le depot.

### Ce qui est en place

- migration de schema pour ajouter `organisation_level` et `parent_societe_id`
- table de fermeture `mariadb_societe_hierarchy`
- modeles SQLAlchemy alignes sur la hierarchie
- extension du RBAC pour gerer les descendants
- filtre ORM multi-societes a partir d'un scope descendant
- middleware enrichi avec le niveau organisationnel courant
- administration de la hierarchie depuis `superadmin`
- vues d'accueil adaptees au niveau actif
- listes `clients` et `affaires` enrichies avec la `societe source` en vue reseau
- page dediee de `pilotage reseau`
- drill-down depuis le pilotage vers les listes filtrees

### Ecrans V2 deja adaptes

- `superadmin`
- `dashboard` / accueil
- `clients`
- `affaires`
- `taches`
- `parametres`

### Logique metier actuellement livree

- `co_courtier`
  - vue portefeuille et operationnelle
- `master_courtier`
  - vue reseau avec detail et societe source
- `delegation_regionale`
  - vue pilotage allegee + page dediee de pilotage reseau
- `superadministrateur`
  - role global transverse

### Fichiers structurants de la V2

- `migrations/20260414_societe_hierarchy_v2.sql`
- `src/models/societe_gestion.py`
- `src/models/societe_hierarchy.py`
- `src/security/rbac.py`
- `src/api/utils/societe_context.py`
- `src/api/main.py`
- `src/api/dashboard.py`
- `src/api/templates/dashboard_superadmin.html`
- `src/api/templates/dashboard.html`
- `src/api/templates/dashboard_clients.html`
- `src/api/templates/dashboard_affaires.html`
- `src/api/templates/dashboard_pilotage.html`

## Validation realisee

### Verification technique effectuee ici

- compilation Python via `python3 -m py_compile` sur les fichiers modifies
- verification des diffs et des commits intermediaires
- test unitaire cible sur le scope V2 dans `tests/test_rbac_v2_scope.py`

### Limites de validation dans cet environnement

- `pytest` n'est pas disponible dans l'environnement shell utilise ici
- l'application complete n'a pas ete lancee avec sa base et ses dependances runtime
- les parcours navigateur n'ont pas ete verifies en execution reelle ici

## Reste a faire

Les points suivants restent a traiter pour une cloture V2 plus complete :

- ajouter une campagne de tests d'integration et de non regression
- verifier tous les ecrans secondaires qui reposent encore sur des requetes SQL directes
- enrichir la page `pilotage reseau` avec des comparatifs temporels si besoin
- eventuellement restreindre encore davantage le niveau `delegation_regionale` au detail client selon arbitrage metier
- documenter les scenarios de migration / backfill en environnement de recette et production

## Conclusion

La V2 n'est plus seulement cadree : elle dispose maintenant d'un socle livre, navigable et pousse sur le depot.

Le produit supporte deja :

- une hierarchie metier explicite
- des droits descendants
- une administration de la hierarchie
- des vues adaptees par niveau
- un pilotage reseau avec descente vers le detail

La suite releve maintenant surtout de la stabilisation, de la recette et des arbitrages metier de finition.
