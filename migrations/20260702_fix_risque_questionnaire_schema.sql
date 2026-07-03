-- Migration 20260702: Fix KYC risk questionnaire schema and restore ID integrity
--
-- This migration ensures that the risk questionnaire and decision tables
-- have proper integer primary keys with AUTO_INCREMENT. It also assigns
-- new IDs to rows where id was NULL or 0, preserving existing history.

SET @next_rq_id = (SELECT IFNULL(MAX(id), 0) FROM risque_questionnaire);
UPDATE risque_questionnaire
SET id = (@next_rq_id := @next_rq_id + 1)
WHERE id IS NULL OR id = 0
ORDER BY client_ref, saisie_at, obsolescence_at,
         offre_calculee_niveau_id, offre_finale_niveau_id,
         IFNULL(objectif_autre_detail, '');

ALTER TABLE risque_questionnaire
  MODIFY id INT NOT NULL AUTO_INCREMENT,
  ADD PRIMARY KEY (id);

SET @next_dc_id = (SELECT IFNULL(MAX(id), 0) FROM risque_decision_client);
UPDATE risque_decision_client
SET id = (@next_dc_id := @next_dc_id + 1)
WHERE id IS NULL OR id = 0
ORDER BY questionnaire_id, saisie_at, obsolescence_at;

ALTER TABLE risque_decision_client
  MODIFY id INT NOT NULL AUTO_INCREMENT,
  ADD PRIMARY KEY (id);
