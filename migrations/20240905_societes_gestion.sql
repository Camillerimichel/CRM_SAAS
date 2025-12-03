-- Ajout des sociétés de gestion/courtage et liaisons clients/affaires

-- Normalisation des clés primaires nécessaires aux FK (idempotent)
SET @has_pk_clients := (
  SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'mariadb_clients' AND CONSTRAINT_NAME = 'PRIMARY'
);
SET @sql_pk_clients := IF(
  @has_pk_clients = 0,
  'ALTER TABLE mariadb_clients MODIFY COLUMN id INT(11) NOT NULL, ADD PRIMARY KEY (id);',
  'SELECT 1;'
);
PREPARE stmt FROM @sql_pk_clients;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_pk_affaires := (
  SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'mariadb_affaires' AND CONSTRAINT_NAME = 'PRIMARY'
);
SET @sql_pk_affaires := IF(
  @has_pk_affaires = 0,
  'ALTER TABLE mariadb_affaires MODIFY COLUMN id INT(11) NOT NULL, ADD PRIMARY KEY (id);',
  'SELECT 1;'
);
PREPARE stmt2 FROM @sql_pk_affaires;
EXECUTE stmt2;
DEALLOCATE PREPARE stmt2;

-- Ajouter colonne téléphone dans DER_courtier si absente
SET @has_tel_courtier := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'DER_courtier' AND COLUMN_NAME = 'telephone'
);
SET @sql_tel_courtier := IF(
  @has_tel_courtier = 0,
  'ALTER TABLE DER_courtier ADD COLUMN telephone TEXT NULL;',
  'SELECT 1;'
);
PREPARE stmt3 FROM @sql_tel_courtier;
EXECUTE stmt3;
DEALLOCATE PREPARE stmt3;

-- Ajout des flags/liaisons superadmin sur administration_RH
SET @has_superadmin := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'administration_RH' AND COLUMN_NAME = 'superadmin'
);
SET @sql_superadmin := IF(
  @has_superadmin = 0,
  'ALTER TABLE administration_RH ADD COLUMN superadmin TINYINT(1) NOT NULL DEFAULT 0;',
  'SELECT 1;'
);
PREPARE stmt4 FROM @sql_superadmin;
EXECUTE stmt4;
DEALLOCATE PREPARE stmt4;

SET @has_soc_gest := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'administration_RH' AND COLUMN_NAME = 'societe_gestion_id'
);
SET @sql_soc_gest := IF(
  @has_soc_gest = 0,
  'ALTER TABLE administration_RH ADD COLUMN societe_gestion_id INT NULL;',
  'SELECT 1;'
);
PREPARE stmt5 FROM @sql_soc_gest;
EXECUTE stmt5;
DEALLOCATE PREPARE stmt5;

-- Index/clé étrangère si la colonne existe
SET @has_fk_soc_gest := (
  SELECT COUNT(*) FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'administration_RH' AND COLUMN_NAME = 'societe_gestion_id' AND REFERENCED_TABLE_NAME = 'mariadb_societe_gestion'
);
SET @sql_fk_soc_gest := IF(
  @has_fk_soc_gest = 0,
  'ALTER TABLE administration_RH ADD CONSTRAINT fk_rh_societe_gestion FOREIGN KEY (societe_gestion_id) REFERENCES mariadb_societe_gestion(id);',
  'SELECT 1;'
);
PREPARE stmt6 FROM @sql_fk_soc_gest;
EXECUTE stmt6;
DEALLOCATE PREPARE stmt6;

CREATE TABLE IF NOT EXISTS mariadb_societe_gestion (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  nom VARCHAR(255) NOT NULL,
  nature VARCHAR(50) NOT NULL,
  siret VARCHAR(50) NULL,
  rcs VARCHAR(50) NULL,
  contact VARCHAR(255) NULL,
  telephone VARCHAR(50) NULL,
  email VARCHAR(255) NULL,
  adresse TEXT NULL,
  actif TINYINT(1) NOT NULL DEFAULT 1,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS mariadb_client_societe (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  client_id INT NOT NULL,
  societe_id INT NOT NULL,
  role VARCHAR(50) NOT NULL,
  date_debut DATE NULL,
  date_fin DATE NULL,
  commentaire TEXT NULL,
  KEY idx_client_societe_societe (societe_id),
  KEY idx_client_societe_active (client_id, date_fin),
  CONSTRAINT fk_client_societe_client FOREIGN KEY (client_id) REFERENCES mariadb_clients(id),
  CONSTRAINT fk_client_societe_societe FOREIGN KEY (societe_id) REFERENCES mariadb_societe_gestion(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS mariadb_affaire_societe (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  affaire_id INT NOT NULL,
  societe_id INT NOT NULL,
  role VARCHAR(50) NOT NULL,
  date_debut DATE NULL,
  date_fin DATE NULL,
  commentaire TEXT NULL,
  KEY idx_affaire_societe_societe (societe_id),
  KEY idx_affaire_societe_active (affaire_id, date_fin),
  CONSTRAINT fk_affaire_societe_affaire FOREIGN KEY (affaire_id) REFERENCES mariadb_affaires(id),
  CONSTRAINT fk_affaire_societe_societe FOREIGN KEY (societe_id) REFERENCES mariadb_societe_gestion(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Initialisation d'une société par défaut
INSERT INTO mariadb_societe_gestion (nom, nature, actif)
VALUES ('Majors Courtage', 'courtier', 1)
ON DUPLICATE KEY UPDATE nom = VALUES(nom);

-- RBAC : rôles, permissions et affectations (dirigeant/directeur/responsable/commercial/back office/client/superadmin)
CREATE TABLE IF NOT EXISTS auth_roles (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(64) NOT NULL UNIQUE,
  label VARCHAR(255) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS auth_permissions (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  feature VARCHAR(100) NOT NULL,
  action VARCHAR(50) NOT NULL,
  description VARCHAR(255) NULL,
  UNIQUE KEY uniq_feature_action (feature, action)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Comptes d'authentification (staff/clients) liés aux rôles
CREATE TABLE IF NOT EXISTS auth_users (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_type ENUM('staff','client') NOT NULL,
  login VARCHAR(255) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  actif TINYINT(1) NOT NULL DEFAULT 1,
  client_id INT NULL,
  rh_id INT NULL,
  last_login DATETIME NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_login (login),
  KEY idx_auth_users_type (user_type),
  KEY idx_auth_users_client (client_id),
  KEY idx_auth_users_rh (rh_id),
  CONSTRAINT fk_auth_users_client FOREIGN KEY (client_id) REFERENCES mariadb_clients(id),
  CONSTRAINT fk_auth_users_rh FOREIGN KEY (rh_id) REFERENCES administration_RH(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS auth_role_permissions (
  role_id INT NOT NULL,
  permission_id INT NOT NULL,
  allow TINYINT(1) NOT NULL DEFAULT 1,
  PRIMARY KEY (role_id, permission_id),
  CONSTRAINT fk_auth_role_perm_role FOREIGN KEY (role_id) REFERENCES auth_roles(id),
  CONSTRAINT fk_auth_role_perm_perm FOREIGN KEY (permission_id) REFERENCES auth_permissions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS auth_user_roles (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  user_type ENUM('staff','client') NOT NULL,
  user_id INT NOT NULL,
  role_id INT NOT NULL,
  societe_id INT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_user_role (user_type, user_id, role_id, societe_id),
  KEY idx_user_role_role (role_id),
  KEY idx_user_role_societe (societe_id),
  CONSTRAINT fk_auth_user_roles_role FOREIGN KEY (role_id) REFERENCES auth_roles(id),
  CONSTRAINT fk_auth_user_roles_societe FOREIGN KEY (societe_id) REFERENCES mariadb_societe_gestion(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Seed des rôles
INSERT IGNORE INTO auth_roles (code, label) VALUES
  ('dirigeant', 'Dirigeant'),
  ('directeur_commercial', 'Directeur commercial'),
  ('responsable_service', 'Responsable de service'),
  ('commercial', 'Commercial'),
  ('back_office', 'Back office'),
  ('client', 'Client'),
  ('superadmin', 'Superadministrateur');

-- Seed des permissions clés (consultation/édition + modules spécifiques)
INSERT IGNORE INTO auth_permissions (feature, action, description) VALUES
  ('data', 'read', 'Consultation des données métiers'),
  ('data', 'write', 'Modification des données métiers'),
  ('groups', 'manage', 'Gestion des groupes'),
  ('administration', 'access', 'Accès administration'),
  ('supports', 'access', 'Module supports'),
  ('offres', 'access', 'Module offres'),
  ('logs', 'view', 'Consultation des journaux'),
  ('client_portal', 'view_own', 'Accès portail client restreint');

-- Variables de rôles
SET @role_dirigeant := (SELECT id FROM auth_roles WHERE code = 'dirigeant');
SET @role_directeur := (SELECT id FROM auth_roles WHERE code = 'directeur_commercial');
SET @role_responsable := (SELECT id FROM auth_roles WHERE code = 'responsable_service');
SET @role_commercial := (SELECT id FROM auth_roles WHERE code = 'commercial');
SET @role_backoffice := (SELECT id FROM auth_roles WHERE code = 'back_office');
SET @role_client := (SELECT id FROM auth_roles WHERE code = 'client');
SET @role_superadmin := (SELECT id FROM auth_roles WHERE code = 'superadmin');

-- Variables de permissions
SET @perm_data_read := (SELECT id FROM auth_permissions WHERE feature = 'data' AND action = 'read');
SET @perm_data_write := (SELECT id FROM auth_permissions WHERE feature = 'data' AND action = 'write');
SET @perm_groups_manage := (SELECT id FROM auth_permissions WHERE feature = 'groups' AND action = 'manage');
SET @perm_admin_access := (SELECT id FROM auth_permissions WHERE feature = 'administration' AND action = 'access');
SET @perm_supports := (SELECT id FROM auth_permissions WHERE feature = 'supports' AND action = 'access');
SET @perm_offres := (SELECT id FROM auth_permissions WHERE feature = 'offres' AND action = 'access');
SET @perm_logs_view := (SELECT id FROM auth_permissions WHERE feature = 'logs' AND action = 'view');
SET @perm_client_view := (SELECT id FROM auth_permissions WHERE feature = 'client_portal' AND action = 'view_own');

-- Dirigeant : tous les droits
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_dirigeant, p.id FROM auth_permissions p WHERE @role_dirigeant IS NOT NULL;

-- Directeur commercial : consultation, gestion des groupes, accès supports/offres
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_directeur, @perm_data_read FROM dual WHERE @role_directeur IS NOT NULL AND @perm_data_read IS NOT NULL
UNION ALL SELECT @role_directeur, @perm_groups_manage FROM dual WHERE @role_directeur IS NOT NULL AND @perm_groups_manage IS NOT NULL
UNION ALL SELECT @role_directeur, @perm_supports FROM dual WHERE @role_directeur IS NOT NULL AND @perm_supports IS NOT NULL
UNION ALL SELECT @role_directeur, @perm_offres FROM dual WHERE @role_directeur IS NOT NULL AND @perm_offres IS NOT NULL;

-- Responsable de service : consultation + modification, pas d’administration ni groupes
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_responsable, @perm_data_read FROM dual WHERE @role_responsable IS NOT NULL AND @perm_data_read IS NOT NULL
UNION ALL SELECT @role_responsable, @perm_data_write FROM dual WHERE @role_responsable IS NOT NULL AND @perm_data_write IS NOT NULL
UNION ALL SELECT @role_responsable, @perm_supports FROM dual WHERE @role_responsable IS NOT NULL AND @perm_supports IS NOT NULL
UNION ALL SELECT @role_responsable, @perm_offres FROM dual WHERE @role_responsable IS NOT NULL AND @perm_offres IS NOT NULL;

-- Commercial : consultation + modification + gestion des groupes
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_commercial, @perm_data_read FROM dual WHERE @role_commercial IS NOT NULL AND @perm_data_read IS NOT NULL
UNION ALL SELECT @role_commercial, @perm_data_write FROM dual WHERE @role_commercial IS NOT NULL AND @perm_data_write IS NOT NULL
UNION ALL SELECT @role_commercial, @perm_groups_manage FROM dual WHERE @role_commercial IS NOT NULL AND @perm_groups_manage IS NOT NULL
UNION ALL SELECT @role_commercial, @perm_supports FROM dual WHERE @role_commercial IS NOT NULL AND @perm_supports IS NOT NULL
UNION ALL SELECT @role_commercial, @perm_offres FROM dual WHERE @role_commercial IS NOT NULL AND @perm_offres IS NOT NULL;

-- Back office : pas de supports/offres/admin/groupes/logs
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_backoffice, @perm_data_read FROM dual WHERE @role_backoffice IS NOT NULL AND @perm_data_read IS NOT NULL
UNION ALL SELECT @role_backoffice, @perm_data_write FROM dual WHERE @role_backoffice IS NOT NULL AND @perm_data_write IS NOT NULL;

-- Client : accès portail restreint (consultation uniquement)
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_client, @perm_client_view FROM dual WHERE @role_client IS NOT NULL AND @perm_client_view IS NOT NULL;

-- Superadmin : tous les droits + miroir dirigeant pour consultation des journaux
INSERT IGNORE INTO auth_role_permissions (role_id, permission_id)
SELECT @role_superadmin, p.id FROM auth_permissions p WHERE @role_superadmin IS NOT NULL;

-- Affectation automatique du rôle superadmin pour les RH marqués superadmin=1
INSERT IGNORE INTO auth_user_roles (user_type, user_id, role_id, societe_id)
SELECT 'staff', rh.id, @role_superadmin, NULL
FROM administration_RH rh
WHERE rh.superadmin = 1 AND @role_superadmin IS NOT NULL;
