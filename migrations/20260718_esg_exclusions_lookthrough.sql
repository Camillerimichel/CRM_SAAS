-- Migration 20260718: Colonnes d'exclusions ESG "look-through" par fonds
--
-- Complète esg_fonds (indicateurs PAI au niveau du fonds lui-même) avec des
-- flags binaires par fonds calculés côté CRM_ESG à partir de sa composition
-- réelle (positions sous-jacentes, cf. esgnote.eu/qualification-esg, onglet
-- "Fonds") : True dès qu'au moins une position du fonds déclenche la
-- catégorie. Synchronisées par src/services/esg_import.py::sync_esg_exclusions_holdings.
--
-- Déjà appliquée manuellement en production le 2026-07-18 ; ce fichier documente
-- le changement pour la suite (nouvel environnement, relecture du schéma).

ALTER TABLE esg_fonds
    ADD COLUMN excluded_coal TINYINT(1) NULL,
    ADD COLUMN excluded_oil_gas TINYINT(1) NULL,
    ADD COLUMN excluded_tar_sands TINYINT(1) NULL,
    ADD COLUMN excluded_tobacco TINYINT(1) NULL,
    ADD COLUMN excluded_weapons TINYINT(1) NULL,
    ADD COLUMN excluded_weapons_controversial TINYINT(1) NULL,
    ADD COLUMN excluded_gambling TINYINT(1) NULL,
    ADD COLUMN excluded_alcohol TINYINT(1) NULL,
    ADD COLUMN excluded_nuclear TINYINT(1) NULL,
    ADD COLUMN excluded_pornography TINYINT(1) NULL,
    ADD COLUMN excluded_fossil_power_generation TINYINT(1) NULL,
    ADD COLUMN excluded_corruption TINYINT(1) NULL,
    ADD COLUMN excluded_human_rights_issue TINYINT(1) NULL,
    ADD COLUMN excluded_forced_labour TINYINT(1) NULL,
    ADD COLUMN excluded_environmental_issue TINYINT(1) NULL;
