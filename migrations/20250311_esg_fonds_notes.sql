-- Add note_e / note_s / note_g to esg_fonds if missing.
SET @has_note_e := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'note_e'
);
SET @sql_note_e := IF(
  @has_note_e = 0,
  'ALTER TABLE esg_fonds ADD COLUMN note_e DECIMAL(4,2) NULL;',
  'SELECT 1;'
);
PREPARE stmt_note_e FROM @sql_note_e;
EXECUTE stmt_note_e;
DEALLOCATE PREPARE stmt_note_e;

SET @has_note_s := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'note_s'
);
SET @sql_note_s := IF(
  @has_note_s = 0,
  'ALTER TABLE esg_fonds ADD COLUMN note_s DECIMAL(4,2) NULL;',
  'SELECT 1;'
);
PREPARE stmt_note_s FROM @sql_note_s;
EXECUTE stmt_note_s;
DEALLOCATE PREPARE stmt_note_s;

SET @has_note_g := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'esg_fonds' AND COLUMN_NAME = 'note_g'
);
SET @sql_note_g := IF(
  @has_note_g = 0,
  'ALTER TABLE esg_fonds ADD COLUMN note_g DECIMAL(4,2) NULL;',
  'SELECT 1;'
);
PREPARE stmt_note_g FROM @sql_note_g;
EXECUTE stmt_note_g;
DEALLOCATE PREPARE stmt_note_g;
