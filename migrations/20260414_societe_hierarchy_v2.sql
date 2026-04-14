-- V2 : hierarchie organisationnelle des societes
-- Objectif :
-- - conserver l'existant
-- - ajouter un niveau d'organisation explicite
-- - permettre une relation parent/enfant entre societes
-- - preparer les consolidations par descendants

-- 1. Ajouter le niveau d'organisation sur la table centrale
SET @has_org_level := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_societe_gestion'
    AND COLUMN_NAME = 'organisation_level'
);
SET @sql_org_level := IF(
  @has_org_level = 0,
  "ALTER TABLE mariadb_societe_gestion ADD COLUMN organisation_level VARCHAR(50) NOT NULL DEFAULT 'co_courtier' AFTER nature;",
  'SELECT 1;'
);
PREPARE stmt_org_level FROM @sql_org_level;
EXECUTE stmt_org_level;
DEALLOCATE PREPARE stmt_org_level;

-- 2. Ajouter le parent hierarchique
SET @has_parent_societe := (
  SELECT COUNT(*)
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_societe_gestion'
    AND COLUMN_NAME = 'parent_societe_id'
);
SET @sql_parent_societe := IF(
  @has_parent_societe = 0,
  'ALTER TABLE mariadb_societe_gestion ADD COLUMN parent_societe_id INT NULL AFTER organisation_level;',
  'SELECT 1;'
);
PREPARE stmt_parent_societe FROM @sql_parent_societe;
EXECUTE stmt_parent_societe;
DEALLOCATE PREPARE stmt_parent_societe;

-- 3. Index et FK sur le parent
SET @has_parent_idx := (
  SELECT COUNT(*)
  FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_societe_gestion'
    AND INDEX_NAME = 'idx_societe_parent'
);
SET @sql_parent_idx := IF(
  @has_parent_idx = 0,
  'ALTER TABLE mariadb_societe_gestion ADD INDEX idx_societe_parent (parent_societe_id);',
  'SELECT 1;'
);
PREPARE stmt_parent_idx FROM @sql_parent_idx;
EXECUTE stmt_parent_idx;
DEALLOCATE PREPARE stmt_parent_idx;

SET @has_parent_fk := (
  SELECT COUNT(*)
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_societe_gestion'
    AND COLUMN_NAME = 'parent_societe_id'
    AND REFERENCED_TABLE_NAME = 'mariadb_societe_gestion'
);
SET @sql_parent_fk := IF(
  @has_parent_fk = 0,
  'ALTER TABLE mariadb_societe_gestion ADD CONSTRAINT fk_societe_parent FOREIGN KEY (parent_societe_id) REFERENCES mariadb_societe_gestion(id);',
  'SELECT 1;'
);
PREPARE stmt_parent_fk FROM @sql_parent_fk;
EXECUTE stmt_parent_fk;
DEALLOCATE PREPARE stmt_parent_fk;

-- 4. Table de fermeture pour les descendants
CREATE TABLE IF NOT EXISTS mariadb_societe_hierarchy (
  ancestor_societe_id INT NOT NULL,
  descendant_societe_id INT NOT NULL,
  depth INT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (ancestor_societe_id, descendant_societe_id),
  KEY idx_societe_hierarchy_descendant (descendant_societe_id),
  KEY idx_societe_hierarchy_depth (depth),
  CONSTRAINT fk_societe_hierarchy_ancestor
    FOREIGN KEY (ancestor_societe_id) REFERENCES mariadb_societe_gestion(id) ON DELETE CASCADE,
  CONSTRAINT fk_societe_hierarchy_descendant
    FOREIGN KEY (descendant_societe_id) REFERENCES mariadb_societe_gestion(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Backfill du niveau organisationnel
-- Les societes courtiers existantes deviennent des co-courtiers par defaut.
UPDATE mariadb_societe_gestion
SET organisation_level = CASE
  WHEN organisation_level IS NULL OR TRIM(organisation_level) = '' THEN
    CASE
      WHEN LOWER(COALESCE(nature, '')) IN ('master courtier', 'master_courtier') THEN 'master_courtier'
      WHEN LOWER(COALESCE(nature, '')) IN ('delegation regionale', 'delegation_regionale') THEN 'delegation_regionale'
      ELSE 'co_courtier'
    END
  WHEN organisation_level NOT IN ('co_courtier', 'master_courtier', 'delegation_regionale') THEN
    CASE
      WHEN LOWER(COALESCE(organisation_level, '')) IN ('master courtier', 'master_courtier') THEN 'master_courtier'
      WHEN LOWER(COALESCE(organisation_level, '')) IN ('delegation regionale', 'delegation_regionale') THEN 'delegation_regionale'
      ELSE 'co_courtier'
    END
  ELSE organisation_level
END;

-- 6. Amorcer la table de fermeture avec les liens reflexifs
INSERT IGNORE INTO mariadb_societe_hierarchy (ancestor_societe_id, descendant_societe_id, depth)
SELECT sg.id, sg.id, 0
FROM mariadb_societe_gestion sg;

-- 7. Amorcer les liens directs parent -> enfant pour les societes deja renseignees
INSERT INTO mariadb_societe_hierarchy (ancestor_societe_id, descendant_societe_id, depth)
SELECT child.parent_societe_id, child.id, 1
FROM mariadb_societe_gestion child
WHERE child.parent_societe_id IS NOT NULL
ON DUPLICATE KEY UPDATE depth = VALUES(depth);

-- 8. Calculer les chemins transitoires existants
INSERT INTO mariadb_societe_hierarchy (ancestor_societe_id, descendant_societe_id, depth)
SELECT h1.ancestor_societe_id, h2.descendant_societe_id, h1.depth + h2.depth
FROM mariadb_societe_hierarchy h1
JOIN mariadb_societe_hierarchy h2
  ON h1.descendant_societe_id = h2.ancestor_societe_id
WHERE h1.depth > 0
  AND h2.depth > 0
ON DUPLICATE KEY UPDATE
  depth = LEAST(mariadb_societe_hierarchy.depth, VALUES(depth));
