CREATE TABLE IF NOT EXISTS document_controle_document_type (
  id int(11) NOT NULL AUTO_INCREMENT,
  activite_id int(11) NOT NULL,
  code varchar(50) NOT NULL,
  libelle varchar(255) NOT NULL,
  seuil_confiance decimal(5,2) NOT NULL DEFAULT 70.00,
  actif tinyint(1) NOT NULL DEFAULT 1,
  created_at timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_controle_keyword (
  id int(11) NOT NULL AUTO_INCREMENT,
  document_type_id int(11) NOT NULL,
  expression varchar(255) NOT NULL,
  is_regex tinyint(1) NOT NULL DEFAULT 0,
  poids decimal(5,2) NOT NULL DEFAULT 1.00,
  obligatoire tinyint(1) NOT NULL DEFAULT 0,
  commentaire text DEFAULT NULL,
  actif tinyint(1) NOT NULL DEFAULT 1,
  created_at timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (id),
  KEY document_type_id (document_type_id),
  CONSTRAINT document_controle_keyword_ibfk_1 FOREIGN KEY (document_type_id) REFERENCES document_controle_document_type (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_controle_analyse (
  id int(11) NOT NULL AUTO_INCREMENT,
  document_id int(11) NOT NULL,
  rule_version varchar(50) NOT NULL,
  score_global decimal(5,2) NOT NULL,
  statut enum('OK','WARNING','KO') NOT NULL,
  analyse_detail longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(analyse_detail)),
  source enum('AUTO','MANUAL','RECHECK') NOT NULL DEFAULT 'AUTO',
  processing_time_ms int(11) DEFAULT NULL,
  analysed_at timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (id),
  KEY idx_document_id (document_id),
  KEY idx_statut (statut),
  CONSTRAINT fk_document_controle_analyse_document FOREIGN KEY (document_id) REFERENCES courtier_documents_officiels (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

CREATE TABLE IF NOT EXISTS document_controle_validation (
  id int(11) NOT NULL AUTO_INCREMENT,
  analyse_id int(11) NOT NULL,
  decision enum('ACCEPT','REJECT') NOT NULL,
  commentaire text DEFAULT NULL,
  validated_by int(11) NOT NULL,
  validated_at timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (id),
  KEY analyse_id (analyse_id),
  CONSTRAINT document_controle_validation_ibfk_1 FOREIGN KEY (analyse_id) REFERENCES document_controle_analyse (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_controle_audit_log (
  id int(11) NOT NULL AUTO_INCREMENT,
  document_id int(11) NOT NULL,
  action varchar(100) NOT NULL,
  details text DEFAULT NULL,
  user_id int(11) DEFAULT NULL,
  created_at timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (id),
  KEY document_id (document_id),
  CONSTRAINT document_controle_audit_log_ibfk_1 FOREIGN KEY (document_id) REFERENCES courtier_documents_officiels (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_controle_export (
  id int(11) NOT NULL AUTO_INCREMENT,
  document_id int(11) NOT NULL,
  format enum('CSV','PDF') NOT NULL,
  file_path varchar(255) NOT NULL,
  generated_at timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (id),
  KEY document_id (document_id),
  CONSTRAINT document_controle_export_ibfk_1 FOREIGN KEY (document_id) REFERENCES courtier_documents_officiels (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE courtier_documents_officiels
  ADD COLUMN controle_status text,
  ADD COLUMN controle_resume text,
  ADD COLUMN controle_checked_at text,
  ADD COLUMN controle_error text;
