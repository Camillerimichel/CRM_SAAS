-- Table de correspondance entre identifiants externes fournisseurs et clients CRM.
-- Un même client physique peut avoir des identifiants différents chez chaque assureur/fournisseur.
-- Cette table est utilisée lors de l'import de fichiers fournisseurs pour résoudre le client CRM
-- à partir de l'identifiant fourni dans le fichier.

CREATE TABLE IF NOT EXISTS mariadb_client_identifiants_fournisseur (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    client_id            INT          NOT NULL,
    fournisseur          VARCHAR(100) NOT NULL COMMENT 'Code fournisseur normalisé en majuscules (ex: GENERALI, AXA, CARDIF)',
    identifiant_externe  VARCHAR(255) NOT NULL COMMENT 'Identifiant du client chez ce fournisseur',
    date_creation        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actif                TINYINT(1)   NOT NULL DEFAULT 1,

    CONSTRAINT fk_cif_client
        FOREIGN KEY (client_id) REFERENCES mariadb_clients(id) ON DELETE CASCADE,

    -- Un identifiant externe est unique par fournisseur
    CONSTRAINT uq_cif_fournisseur_id
        UNIQUE (fournisseur, identifiant_externe),

    INDEX idx_cif_client_id  (client_id),
    INDEX idx_cif_fournisseur (fournisseur)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
