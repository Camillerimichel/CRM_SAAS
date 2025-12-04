-- Journalisation des connexions clients (par courtier) + mise à jour de last_login

CREATE TABLE IF NOT EXISTS auth_client_login_logs (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  client_user_id INT NOT NULL,
  client_id INT NULL,
  broker_id INT NULL,
  ip_address VARCHAR(64) NULL,
  user_agent TEXT NULL,
  success TINYINT(1) NOT NULL DEFAULT 1,
  session_seconds INT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_login_client_user (client_user_id),
  KEY idx_login_client (client_id),
  KEY idx_login_broker (broker_id),
  CONSTRAINT fk_login_client_user FOREIGN KEY (client_user_id) REFERENCES auth_client_users(id),
  CONSTRAINT fk_login_client FOREIGN KEY (client_id) REFERENCES mariadb_clients(id),
  CONSTRAINT fk_login_broker FOREIGN KEY (broker_id) REFERENCES mariadb_societe_gestion(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
