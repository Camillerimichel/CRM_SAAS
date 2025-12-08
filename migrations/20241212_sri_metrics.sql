-- Migration: création de la table d'historisation des métriques SRI
-- Objet : stocker les moments (M0..M4), volatilité, asymétrie, kurtosis,
--         VaR Cornish-Fisher et VeV pour chaque entité (client / affaire)
--         avec historisation par date de calcul (as_of_date).

CREATE TABLE IF NOT EXISTS sri_metrics (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  entity_type VARCHAR(16) NOT NULL,                -- 'client' ou 'affaire'
  entity_id INT NOT NULL,
  as_of_date DATE NOT NULL,                        -- date de référence de la série VL
  calc_at DATETIME DEFAULT CURRENT_TIMESTAMP,      -- date/heure du calcul
  sri INT NULL,                                    -- SRI (1..7)
  m0 INT NULL,                                     -- nombre d'observations
  m1 DECIMAL(38,18) NULL,                          -- moyenne des rendements
  m2 DECIMAL(38,18) NULL,                          -- moment d'ordre 2 (variance)
  m3 DECIMAL(38,18) NULL,                          -- moment d'ordre 3 (centré)
  m4 DECIMAL(38,18) NULL,                          -- moment d'ordre 4 (centré)
  sigma DECIMAL(38,18) NULL,                       -- volatilité (sqrt(m2))
  mu1 DECIMAL(38,18) NULL,                         -- skewness
  mu2 DECIMAL(38,18) NULL,                         -- kurtosis d'excès
  n_periods INT NULL,                              -- nombre de périodes pour VaR
  var_cf DECIMAL(38,18) NULL,                      -- VaR Cornish-Fisher
  vev DECIMAL(38,18) NULL,                         -- équivalent volatilité de la VaR
  UNIQUE KEY uk_entity_date (entity_type, entity_id, as_of_date),
  INDEX idx_entity (entity_type, entity_id),
  INDEX idx_as_of (as_of_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Ajouter une colonne SRI dédiée (distincte du SRRI existant) dans les tables clients et affaires
ALTER TABLE mariadb_clients
  ADD COLUMN IF NOT EXISTS SRI INT NULL AFTER SRRI;

ALTER TABLE mariadb_affaires
  ADD COLUMN IF NOT EXISTS SRI INT NULL AFTER SRRI;
