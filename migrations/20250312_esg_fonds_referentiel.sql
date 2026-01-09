-- Align esg_fonds with CRM_ESG Referentiel_final columns (idempotent).

SET @has_company_name := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'company_name'
);
SET @sql_company_name := IF(
  @has_company_name = 0,
  'ALTER TABLE esg_fonds ADD COLUMN company_name TEXT NULL;',
  'SELECT 1;'
);
PREPARE stmt_company_name FROM @sql_company_name;
EXECUTE stmt_company_name;
DEALLOCATE PREPARE stmt_company_name;

SET @has_sector1 := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'sector1'
);
SET @sql_sector1 := IF(
  @has_sector1 = 0,
  'ALTER TABLE esg_fonds ADD COLUMN sector1 MEDIUMTEXT NULL;',
  'SELECT 1;'
);
PREPARE stmt_sector1 FROM @sql_sector1;
EXECUTE stmt_sector1;
DEALLOCATE PREPARE stmt_sector1;

SET @has_mcap_usd := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'mcap_usd'
);
SET @sql_mcap_usd := IF(
  @has_mcap_usd = 0,
  'ALTER TABLE esg_fonds ADD COLUMN mcap_usd DOUBLE NULL;',
  'SELECT 1;'
);
PREPARE stmt_mcap_usd FROM @sql_mcap_usd;
EXECUTE stmt_mcap_usd;
DEALLOCATE PREPARE stmt_mcap_usd;

SET @has_revenue_usd := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'revenue_usd'
);
SET @sql_revenue_usd := IF(
  @has_revenue_usd = 0,
  'ALTER TABLE esg_fonds ADD COLUMN revenue_usd DOUBLE NULL;',
  'SELECT 1;'
);
PREPARE stmt_revenue_usd FROM @sql_revenue_usd;
EXECUTE stmt_revenue_usd;
DEALLOCATE PREPARE stmt_revenue_usd;

SET @has_processes_ungc := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'processes_ungc'
);
SET @sql_processes_ungc := IF(
  @has_processes_ungc = 0,
  'ALTER TABLE esg_fonds ADD COLUMN processes_ungc INT(11) NULL;',
  'SELECT 1;'
);
PREPARE stmt_processes_ungc FROM @sql_processes_ungc;
EXECUTE stmt_processes_ungc;
DEALLOCATE PREPARE stmt_processes_ungc;

SET @has_note_esg := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'note_esg'
);
SET @sql_note_esg := IF(
  @has_note_esg = 0,
  'ALTER TABLE esg_fonds ADD COLUMN note_esg DOUBLE NULL;',
  'SELECT 1;'
);
PREPARE stmt_note_esg FROM @sql_note_esg;
EXECUTE stmt_note_esg;
DEALLOCATE PREPARE stmt_note_esg;

SET @has_evic := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'evic'
);
SET @sql_evic := IF(
  @has_evic = 0,
  'ALTER TABLE esg_fonds ADD COLUMN evic DECIMAL(10,0) NULL;',
  'SELECT 1;'
);
PREPARE stmt_evic FROM @sql_evic;
EXECUTE stmt_evic;
DEALLOCATE PREPARE stmt_evic;
