-- Ajout de id_societe_gestion sur les historiques pour respecter la multi-société

SET @has_historique_personne := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_historique_personne_w'
    AND COLUMN_NAME = 'id_societe_gestion'
);
SET @sql_historique_personne := IF(
  @has_historique_personne = 0,
  'ALTER TABLE mariadb_historique_personne_w ADD COLUMN id_societe_gestion INT NULL;',
  'SELECT 1;'
);
PREPARE stmt1 FROM @sql_historique_personne;
EXECUTE stmt1;
DEALLOCATE PREPARE stmt1;

SET @has_historique_personne_idx := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_historique_personne_w'
    AND INDEX_NAME = 'idx_historique_personne_societe'
);
SET @sql_historique_personne_idx := IF(
  @has_historique_personne_idx = 0,
  'ALTER TABLE mariadb_historique_personne_w ADD INDEX idx_historique_personne_societe (id_societe_gestion);',
  'SELECT 1;'
);
PREPARE stmt2 FROM @sql_historique_personne_idx;
EXECUTE stmt2;
DEALLOCATE PREPARE stmt2;

SET @has_historique_personne_fk := (
  SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS tc
  JOIN information_schema.KEY_COLUMN_USAGE kcu ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
  WHERE tc.TABLE_SCHEMA = DATABASE()
    AND tc.TABLE_NAME = 'mariadb_historique_personne_w'
    AND kcu.COLUMN_NAME = 'id_societe_gestion'
    AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
);
SET @sql_historique_personne_fk := IF(
  @has_historique_personne_fk = 0,
  'ALTER TABLE mariadb_historique_personne_w ADD CONSTRAINT fk_hist_personne_societe FOREIGN KEY (id_societe_gestion) REFERENCES mariadb_societe_gestion(id);',
  'SELECT 1;'
);
PREPARE stmt3 FROM @sql_historique_personne_fk;
EXECUTE stmt3;
DEALLOCATE PREPARE stmt3;

SET @has_historique_affaire := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_historique_affaire_w'
    AND COLUMN_NAME = 'id_societe_gestion'
);
SET @sql_historique_affaire := IF(
  @has_historique_affaire = 0,
  'ALTER TABLE mariadb_historique_affaire_w ADD COLUMN id_societe_gestion INT NULL;',
  'SELECT 1;'
);
PREPARE stmt4 FROM @sql_historique_affaire;
EXECUTE stmt4;
DEALLOCATE PREPARE stmt4;

SET @has_historique_affaire_idx := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_historique_affaire_w'
    AND INDEX_NAME = 'idx_historique_affaire_societe'
);
SET @sql_historique_affaire_idx := IF(
  @has_historique_affaire_idx = 0,
  'ALTER TABLE mariadb_historique_affaire_w ADD INDEX idx_historique_affaire_societe (id_societe_gestion);',
  'SELECT 1;'
);
PREPARE stmt5 FROM @sql_historique_affaire_idx;
EXECUTE stmt5;
DEALLOCATE PREPARE stmt5;

SET @has_historique_affaire_fk := (
  SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS tc
  JOIN information_schema.KEY_COLUMN_USAGE kcu ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
  WHERE tc.TABLE_SCHEMA = DATABASE()
    AND tc.TABLE_NAME = 'mariadb_historique_affaire_w'
    AND kcu.COLUMN_NAME = 'id_societe_gestion'
    AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
);
SET @sql_historique_affaire_fk := IF(
  @has_historique_affaire_fk = 0,
  'ALTER TABLE mariadb_historique_affaire_w ADD CONSTRAINT fk_hist_affaire_societe FOREIGN KEY (id_societe_gestion) REFERENCES mariadb_societe_gestion(id);',
  'SELECT 1;'
);
PREPARE stmt6 FROM @sql_historique_affaire_fk;
EXECUTE stmt6;
DEALLOCATE PREPARE stmt6;

SET @has_historique_support := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_historique_support_w'
    AND COLUMN_NAME = 'id_societe_gestion'
);
SET @sql_historique_support := IF(
  @has_historique_support = 0,
  'ALTER TABLE mariadb_historique_support_w ADD COLUMN id_societe_gestion INT NULL;',
  'SELECT 1;'
);
PREPARE stmt7 FROM @sql_historique_support;
EXECUTE stmt7;
DEALLOCATE PREPARE stmt7;

SET @has_historique_support_idx := (
  SELECT COUNT(*) FROM information_schema.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'mariadb_historique_support_w'
    AND INDEX_NAME = 'idx_historique_support_societe'
);
SET @sql_historique_support_idx := IF(
  @has_historique_support_idx = 0,
  'ALTER TABLE mariadb_historique_support_w ADD INDEX idx_historique_support_societe (id_societe_gestion);',
  'SELECT 1;'
);
PREPARE stmt8 FROM @sql_historique_support_idx;
EXECUTE stmt8;
DEALLOCATE PREPARE stmt8;

SET @has_historique_support_fk := (
  SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS tc
  JOIN information_schema.KEY_COLUMN_USAGE kcu ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
  WHERE tc.TABLE_SCHEMA = DATABASE()
    AND tc.TABLE_NAME = 'mariadb_historique_support_w'
    AND kcu.COLUMN_NAME = 'id_societe_gestion'
    AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
);
SET @sql_historique_support_fk := IF(
  @has_historique_support_fk = 0,
  'ALTER TABLE mariadb_historique_support_w ADD CONSTRAINT fk_hist_support_societe FOREIGN KEY (id_societe_gestion) REFERENCES mariadb_societe_gestion(id);',
  'SELECT 1;'
);
PREPARE stmt9 FROM @sql_historique_support_fk;
EXECUTE stmt9;
DEALLOCATE PREPARE stmt9;
