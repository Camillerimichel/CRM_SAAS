-- Seed de recette V2
-- Usage: environnement de dev / recette uniquement
-- Objectif: creer une hierarchie de demonstration prefixee "RECETTE V2 -"

START TRANSACTION;

-- Nettoyage limite au jeu de demonstration
DELETE FROM mariadb_societe_gestion
WHERE nom IN (
  'RECETTE V2 - Co Lille Centre',
  'RECETTE V2 - Co Lille Est',
  'RECETTE V2 - Co Amiens Centre',
  'RECETTE V2 - Master Lille',
  'RECETTE V2 - Master Amiens',
  'RECETTE V2 - Delegation Nord'
);

-- Creation de la delegation
INSERT INTO mariadb_societe_gestion (
  nom,
  nature,
  organisation_level,
  parent_societe_id,
  contact,
  telephone,
  email,
  adresse,
  actif
)
VALUES (
  'RECETTE V2 - Delegation Nord',
  'courtier',
  'delegation_regionale',
  NULL,
  'Equipe recette V2',
  '0100000000',
  'recette-delegation@example.test',
  '1 rue de la Recette, Lille',
  1
);

SET @delegation_nord_id := (
  SELECT id
  FROM mariadb_societe_gestion
  WHERE nom = 'RECETTE V2 - Delegation Nord'
  ORDER BY id DESC
  LIMIT 1
);

-- Creation des masters
INSERT INTO mariadb_societe_gestion (
  nom, nature, organisation_level, parent_societe_id,
  contact, telephone, email, adresse, actif
)
VALUES
(
  'RECETTE V2 - Master Lille',
  'courtier',
  'master_courtier',
  @delegation_nord_id,
  'Master Lille',
  '0100000001',
  'recette-master-lille@example.test',
  '10 place de Lille',
  1
),
(
  'RECETTE V2 - Master Amiens',
  'courtier',
  'master_courtier',
  @delegation_nord_id,
  'Master Amiens',
  '0100000002',
  'recette-master-amiens@example.test',
  '20 place d Amiens',
  1
);

SET @master_lille_id := (
  SELECT id
  FROM mariadb_societe_gestion
  WHERE nom = 'RECETTE V2 - Master Lille'
  ORDER BY id DESC
  LIMIT 1
);

SET @master_amiens_id := (
  SELECT id
  FROM mariadb_societe_gestion
  WHERE nom = 'RECETTE V2 - Master Amiens'
  ORDER BY id DESC
  LIMIT 1
);

-- Creation des co-courtiers
INSERT INTO mariadb_societe_gestion (
  nom, nature, organisation_level, parent_societe_id,
  contact, telephone, email, adresse, actif
)
VALUES
(
  'RECETTE V2 - Co Lille Centre',
  'co-courtier',
  'co_courtier',
  @master_lille_id,
  'Co Lille Centre',
  '0100000011',
  'recette-co-lille-centre@example.test',
  '11 avenue du Centre, Lille',
  1
),
(
  'RECETTE V2 - Co Lille Est',
  'co-courtier',
  'co_courtier',
  @master_lille_id,
  'Co Lille Est',
  '0100000012',
  'recette-co-lille-est@example.test',
  '12 avenue de l Est, Lille',
  1
),
(
  'RECETTE V2 - Co Amiens Centre',
  'co-courtier',
  'co_courtier',
  @master_amiens_id,
  'Co Amiens Centre',
  '0100000013',
  'recette-co-amiens-centre@example.test',
  '13 avenue du Centre, Amiens',
  1
);

-- Reconstruction complete de la fermeture hierarchique
DELETE FROM mariadb_societe_hierarchy;

INSERT IGNORE INTO mariadb_societe_hierarchy (ancestor_societe_id, descendant_societe_id, depth)
SELECT sg.id, sg.id, 0
FROM mariadb_societe_gestion sg;

INSERT INTO mariadb_societe_hierarchy (ancestor_societe_id, descendant_societe_id, depth)
SELECT child.parent_societe_id, child.id, 1
FROM mariadb_societe_gestion child
WHERE child.parent_societe_id IS NOT NULL
ON DUPLICATE KEY UPDATE depth = VALUES(depth);

INSERT INTO mariadb_societe_hierarchy (ancestor_societe_id, descendant_societe_id, depth)
SELECT h1.ancestor_societe_id, h2.descendant_societe_id, h1.depth + h2.depth
FROM mariadb_societe_hierarchy h1
JOIN mariadb_societe_hierarchy h2
  ON h1.descendant_societe_id = h2.ancestor_societe_id
WHERE h1.depth > 0
  AND h2.depth > 0
ON DUPLICATE KEY UPDATE
  depth = LEAST(mariadb_societe_hierarchy.depth, VALUES(depth));

COMMIT;
