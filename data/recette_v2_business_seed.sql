-- Seed metier de recette V2
-- Usage: environnement de dev / recette uniquement
-- Prerequis: data/recette_v2_hierarchy_seed.sql deja execute

START TRANSACTION;

-- Resolution des societes de recette
SET @co_lille_centre_id := (
  SELECT id FROM mariadb_societe_gestion
  WHERE nom = 'RECETTE V2 - Co Lille Centre'
  ORDER BY id DESC
  LIMIT 1
);

SET @co_lille_est_id := (
  SELECT id FROM mariadb_societe_gestion
  WHERE nom = 'RECETTE V2 - Co Lille Est'
  ORDER BY id DESC
  LIMIT 1
);

SET @co_amiens_centre_id := (
  SELECT id FROM mariadb_societe_gestion
  WHERE nom = 'RECETTE V2 - Co Amiens Centre'
  ORDER BY id DESC
  LIMIT 1
);

-- Nettoyage strictement limite au jeu de recette
DELETE FROM mariadb_affaire_societe
WHERE affaire_id IN (
  SELECT id FROM mariadb_affaires WHERE ref LIKE 'RECETTE-V2-%'
);

DELETE FROM mariadb_affaires
WHERE ref LIKE 'RECETTE-V2-%';

DELETE FROM mariadb_client_societe
WHERE client_id IN (
  SELECT id FROM mariadb_clients WHERE email LIKE 'recette-v2-%@example.test'
);

DELETE FROM mariadb_clients
WHERE email LIKE 'recette-v2-%@example.test';

-- Creation des clients
SET @client_id_1 := (SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_clients);
SET @client_id_2 := @client_id_1 + 1;
SET @client_id_3 := @client_id_2 + 1;
SET @client_id_4 := @client_id_3 + 1;

INSERT INTO mariadb_clients (
  id,
  nom,
  prenom,
  srri,
  telephone,
  adresse_postale,
  email,
  commercial_id,
  id_societe_gestion
)
VALUES
(
  @client_id_1,
  'Martin',
  'Alice',
  3,
  '0600000001',
  '101 rue de la Recette, Lille',
  'recette-v2-alice.martin@example.test',
  NULL,
  @co_lille_centre_id
),
(
  @client_id_2,
  'Durand',
  'Bruno',
  4,
  '0600000002',
  '102 rue de la Recette, Lille',
  'recette-v2-bruno.durand@example.test',
  NULL,
  @co_lille_centre_id
),
(
  @client_id_3,
  'Petit',
  'Claire',
  2,
  '0600000003',
  '103 rue de la Recette, Lille',
  'recette-v2-claire.petit@example.test',
  NULL,
  @co_lille_est_id
),
(
  @client_id_4,
  'Bernard',
  'David',
  5,
  '0600000004',
  '104 rue de la Recette, Amiens',
  'recette-v2-david.bernard@example.test',
  NULL,
  @co_amiens_centre_id
);

INSERT INTO mariadb_client_societe (
  client_id,
  societe_id,
  role,
  date_debut,
  date_fin,
  commentaire
)
VALUES
(@client_id_1, @co_lille_centre_id, 'courtier', CURRENT_DATE, NULL, 'Jeu de recette V2'),
(@client_id_2, @co_lille_centre_id, 'courtier', CURRENT_DATE, NULL, 'Jeu de recette V2'),
(@client_id_3, @co_lille_est_id, 'courtier', CURRENT_DATE, NULL, 'Jeu de recette V2'),
(@client_id_4, @co_amiens_centre_id, 'courtier', CURRENT_DATE, NULL, 'Jeu de recette V2');

-- Creation des affaires
SET @affaire_id_1 := (SELECT COALESCE(MAX(id), 0) + 1 FROM mariadb_affaires);
SET @affaire_id_2 := @affaire_id_1 + 1;
SET @affaire_id_3 := @affaire_id_2 + 1;
SET @affaire_id_4 := @affaire_id_3 + 1;
SET @affaire_id_5 := @affaire_id_4 + 1;

INSERT INTO mariadb_affaires (
  id,
  id_personne,
  ref,
  date_debut,
  date_cle,
  srri,
  SRI,
  frais_negocies,
  id_affaire_generique,
  id_societe_gestion
)
VALUES
(@affaire_id_1, @client_id_1, 'RECETTE-V2-ALICE-01', NOW(), NOW(), 3, 3, 0.50, NULL, @co_lille_centre_id),
(@affaire_id_2, @client_id_2, 'RECETTE-V2-BRUNO-01', NOW(), NOW(), 4, 4, 0.65, NULL, @co_lille_centre_id),
(@affaire_id_3, @client_id_3, 'RECETTE-V2-CLAIRE-01', NOW(), NOW(), 2, 2, 0.45, NULL, @co_lille_est_id),
(@affaire_id_4, @client_id_4, 'RECETTE-V2-DAVID-01', NOW(), NOW(), 5, 5, 0.80, NULL, @co_amiens_centre_id),
(@affaire_id_5, @client_id_4, 'RECETTE-V2-DAVID-02', NOW(), NOW(), 4, 4, 0.75, NULL, @co_amiens_centre_id);

INSERT INTO mariadb_affaire_societe (
  affaire_id,
  societe_id,
  role,
  date_debut,
  date_fin,
  commentaire
)
SELECT
  a.id,
  CASE
    WHEN a.ref IN ('RECETTE-V2-ALICE-01', 'RECETTE-V2-BRUNO-01') THEN @co_lille_centre_id
    WHEN a.ref = 'RECETTE-V2-CLAIRE-01' THEN @co_lille_est_id
    WHEN a.ref IN ('RECETTE-V2-DAVID-01', 'RECETTE-V2-DAVID-02') THEN @co_amiens_centre_id
    ELSE NULL
  END,
  'courtier',
  CURRENT_DATE,
  NULL,
  'Jeu de recette V2'
FROM mariadb_affaires a
WHERE a.ref LIKE 'RECETTE-V2-%';

COMMIT;
