-- Migration 20260703: Remove orphaned KYC risk questionnaire child rows
-- and enforce that questionnaire_id is always positive.

DELETE FROM risque_questionnaire_connaissance WHERE questionnaire_id = 0;
DELETE FROM risque_questionnaire_objectif WHERE questionnaire_id = 0;
DELETE FROM risque_decision_client WHERE questionnaire_id = 0;

SELECT COUNT(*) AS orphaned_risque_questionnaire_connaissance_after_cleanup
FROM risque_questionnaire_connaissance
WHERE questionnaire_id = 0;

SELECT COUNT(*) AS orphaned_risque_questionnaire_objectif_after_cleanup
FROM risque_questionnaire_objectif
WHERE questionnaire_id = 0;

SELECT COUNT(*) AS orphaned_risque_decision_client_after_cleanup
FROM risque_decision_client
WHERE questionnaire_id = 0;

ALTER TABLE risque_questionnaire_connaissance
  MODIFY questionnaire_id INT NOT NULL;

ALTER TABLE risque_questionnaire_objectif
  MODIFY questionnaire_id INT NOT NULL;

ALTER TABLE risque_decision_client
  MODIFY questionnaire_id INT NOT NULL;
