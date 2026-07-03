-- Migration 20260703: Consolidate the risk questionnaire onto KYC_Client_Risque
--
-- KYC_Client_Risque becomes the single parent/history table for the risk
-- questionnaire (one row per client per day). risque_questionnaire and its
-- child tables (risque_questionnaire_connaissance, risque_questionnaire_objectif,
-- risque_decision_client) are no longer written to by the application; they
-- are left as-is in the database for later cleanup.
--
-- KYC_Client_Risque_Connaissance is fixed and formalized as the single
-- remaining child table (per-product knowledge level), properly linked via FK.

ALTER TABLE KYC_Client_Risque
  ADD COLUMN perte_option_id INT NULL,
  ADD COLUMN patrimoine_part_option_id INT NULL,
  ADD COLUMN disponibilite_option_id INT NULL,
  ADD COLUMN duree_option_id INT NULL,
  ADD COLUMN offre_calculee_niveau_id INT NULL,
  ADD COLUMN objectifs_json TEXT NULL,
  ADD COLUMN objectif_autre_detail TEXT NULL,
  ADD COLUMN decision VARCHAR(16) NULL,
  ADD COLUMN niveau_client_id INT NULL,
  ADD COLUMN motivation_refus TEXT NULL,
  ADD COLUMN created_at DATETIME NULL,
  ADD COLUMN updated_at DATETIME NULL;

-- KYC_Client_Risque_Connaissance already exists (created ad hoc by
-- dashboard.py / scripts/migrate_add_kcr_connaissance.py) but has no
-- PRIMARY KEY / FK and holds a few orphaned rows from deleted snapshots.
-- Clean up orphans, then formalize the constraints.
DELETE k FROM KYC_Client_Risque_Connaissance k
WHERE NOT EXISTS (SELECT 1 FROM KYC_Client_Risque r WHERE r.id = k.risque_id);

ALTER TABLE KYC_Client_Risque_Connaissance
  DROP INDEX idx_kyc_risqueconnaissance;

ALTER TABLE KYC_Client_Risque_Connaissance
  MODIFY risque_id INT NOT NULL,
  MODIFY produit_id INT NOT NULL,
  MODIFY niveau_id INT NOT NULL,
  MODIFY produit_label VARCHAR(255) NULL,
  MODIFY niveau_label VARCHAR(255) NULL,
  ADD PRIMARY KEY (risque_id, produit_id),
  ADD CONSTRAINT fk_kcr_connaissance_risque
    FOREIGN KEY (risque_id) REFERENCES KYC_Client_Risque(id) ON DELETE CASCADE;
