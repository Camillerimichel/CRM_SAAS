ALTER TABLE KYC_Client_Risque
  ADD COLUMN profil_coherence_html TEXT NULL,
  ADD COLUMN age_adequation_html TEXT NULL,
  ADD COLUMN age_adequation_status VARCHAR(32) NULL,
  ADD COLUMN analysis_generated_at DATETIME NULL;
