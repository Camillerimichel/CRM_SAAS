-- Table de correspondance entre identifiants externes fournisseurs et sociétés de gestion CRM.
-- Une même société de gestion (courtier) peut avoir des identifiants différents chez chaque
-- assureur/fournisseur. Cette table est utilisée lors de l'import de fichiers fournisseurs pour
-- résoudre la société de gestion CRM à partir du code transmis dans le fichier.

CREATE TABLE IF NOT EXISTS mariadb_societe_identifiants_fournisseur (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    societe_id           INT          NOT NULL,
    fournisseur          VARCHAR(100) NOT NULL COMMENT 'Code fournisseur normalisé en majuscules (ex: GENERALI, CARDIF, SPIRICA)',
    identifiant_externe  VARCHAR(255) NOT NULL COMMENT 'Identifiant de la société chez ce fournisseur',
    date_creation        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actif                TINYINT(1)   NOT NULL DEFAULT 1,

    CONSTRAINT fk_sif_societe
        FOREIGN KEY (societe_id) REFERENCES mariadb_societe_gestion(id) ON DELETE CASCADE,

    -- Un identifiant externe est unique par fournisseur
    CONSTRAINT uq_sif_fournisseur_id
        UNIQUE (fournisseur, identifiant_externe),

    INDEX idx_sif_societe_id  (societe_id),
    INDEX idx_sif_fournisseur (fournisseur)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Données initiales
INSERT IGNORE INTO mariadb_societe_identifiants_fournisseur (societe_id, fournisseur, identifiant_externe, actif)
VALUES (5,  'AFI ESCA FRANCE', '21079',   1),  -- Majors Courtage chez AFI ESCA France
       (12, 'AFI ESCA FRANCE', '9910386', 1);  -- BCAM chez AFI ESCA France
