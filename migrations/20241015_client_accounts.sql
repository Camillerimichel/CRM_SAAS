-- Comptes clients scindés par courtier (mêmes emails possibles sur plusieurs courtiers)
-- À exécuter avant l'implémentation applicative.

CREATE TABLE IF NOT EXISTS auth_client_users (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  client_id INT NOT NULL,
  broker_id INT NOT NULL, -- référence mariadb_societe_gestion (nature = 'courtier')
  login VARCHAR(255) NOT NULL, -- email ou identifiant de connexion saisi par le client
  password_hash VARCHAR(255) NOT NULL,
  status ENUM('active','disabled','pending_reset') NOT NULL DEFAULT 'active',
  legacy_password_hint TEXT NULL, -- pour stocker l'indication existante le temps de la migration (jamais le mot de passe en clair)
  last_login DATETIME NULL,
  password_updated_at DATETIME NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_broker_login (broker_id, login), -- autorise le même email sur deux courtiers différents
  KEY idx_client_users_client (client_id),
  KEY idx_client_users_broker (broker_id),
  CONSTRAINT fk_client_users_client FOREIGN KEY (client_id) REFERENCES mariadb_clients(id),
  CONSTRAINT fk_client_users_broker FOREIGN KEY (broker_id) REFERENCES mariadb_societe_gestion(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Rôles/portées des comptes clients (réutilise auth_roles existants, ex: rôle "client" ou variantes portails)
CREATE TABLE IF NOT EXISTS auth_client_user_roles (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  client_user_id INT NOT NULL,
  role_id INT NOT NULL,
  societe_id INT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_client_role (client_user_id, role_id, societe_id),
  KEY idx_client_role_societe (societe_id),
  CONSTRAINT fk_client_role_user FOREIGN KEY (client_user_id) REFERENCES auth_client_users(id),
  CONSTRAINT fk_client_role_role FOREIGN KEY (role_id) REFERENCES auth_roles(id),
  CONSTRAINT fk_client_role_societe FOREIGN KEY (societe_id) REFERENCES mariadb_societe_gestion(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Index de confort pour les journaux de connexion futurs
CREATE INDEX IF NOT EXISTS idx_client_users_last_login ON auth_client_users (last_login);
