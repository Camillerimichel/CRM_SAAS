SET @compliance_table := (
  SELECT TABLE_NAME
  FROM INFORMATION_SCHEMA.TABLES
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME IN ('Compliance_rc_pro', 'compliance_rc_pro')
  LIMIT 1
);

SET @has_activite := (
  SELECT COUNT(*)
  FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = @compliance_table
    AND COLUMN_NAME = 'activite'
);

SET @has_libelle := (
  SELECT COUNT(*)
  FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = @compliance_table
    AND COLUMN_NAME = 'libelle'
);

SET @sql_delete := "SELECT 1";

SET @sql_delete := IF(
  @compliance_table IS NOT NULL AND @has_activite > 0,
  CONCAT(
    "DELETE FROM ", @compliance_table, " WHERE activite = 'Garanties financières'"
  ),
  IF(
    @compliance_table IS NOT NULL AND @has_libelle > 0,
    CONCAT(
      "DELETE FROM ", @compliance_table, " WHERE libelle = 'Garanties financières'"
    ),
    "SELECT 1"
  )
);

PREPARE stmt FROM @sql_delete;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
