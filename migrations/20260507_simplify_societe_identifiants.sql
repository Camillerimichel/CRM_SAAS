-- Simplification de mariadb_societe_identifiants_fournisseur :
-- L'identifiant externe (ref CGP / ORIAS) est globalement unique —
-- il identifie une société indépendamment de l'assureur.
-- On supprime la contrainte composée (fournisseur, identifiant_externe)
-- et on rend fournisseur optionnel (metadata uniquement).

ALTER TABLE mariadb_societe_identifiants_fournisseur
  DROP INDEX uq_sif_fournisseur_id;

ALTER TABLE mariadb_societe_identifiants_fournisseur
  DROP INDEX idx_sif_fournisseur;

ALTER TABLE mariadb_societe_identifiants_fournisseur
  MODIFY fournisseur VARCHAR(100) NULL
    COMMENT 'Code fournisseur (metadata optionnel, ex: AES PAT-LUX)';

ALTER TABLE mariadb_societe_identifiants_fournisseur
  ADD CONSTRAINT uq_sif_identifiant UNIQUE (identifiant_externe);
