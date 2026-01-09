-- Dedupe esg_fonds by ISIN and enforce unique index (updates on import).
UPDATE esg_fonds
SET isin = UPPER(TRIM(isin))
WHERE isin IS NOT NULL;

DROP TABLE IF EXISTS esg_fonds_dedup;
CREATE TABLE esg_fonds_dedup LIKE esg_fonds;

SET @has_idx := (
  SELECT COUNT(*)
  FROM INFORMATION_SCHEMA.STATISTICS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'esg_fonds_dedup'
    AND INDEX_NAME = 'idx_esg_fonds_isin'
);
SET @sql_drop := IF(@has_idx > 0, 'ALTER TABLE esg_fonds_dedup DROP INDEX idx_esg_fonds_isin', 'SELECT 1');
PREPARE stmt_drop FROM @sql_drop;
EXECUTE stmt_drop;
DEALLOCATE PREPARE stmt_drop;

ALTER TABLE esg_fonds_dedup ADD UNIQUE KEY idx_esg_fonds_isin (isin(12));

INSERT IGNORE INTO esg_fonds_dedup
SELECT * FROM esg_fonds;

DROP TABLE IF EXISTS esg_fonds_dups;
RENAME TABLE esg_fonds TO esg_fonds_dups, esg_fonds_dedup TO esg_fonds;
DROP TABLE esg_fonds_dups;
