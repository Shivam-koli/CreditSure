-- Runs once, automatically, when the MySQL container is first created (docker-entrypoint-initdb.d).
-- The application user and database come from the compose environment (MYSQL_DATABASE / MYSQL_USER).
-- Table DDL is applied by `python -m lia.bootstrap` from sql/02_tables.sql, views from sql/views/*.sql.

CREATE TABLE IF NOT EXISTS monitoring_drift_runs (
  run_id        INT AUTO_INCREMENT PRIMARY KEY,
  run_time      DATETIME NOT NULL,
  model_kind    VARCHAR(16) NOT NULL,
  window_name   VARCHAR(32) NOT NULL,
  overall       VARCHAR(8)  NOT NULL,
  n_alert       INT,
  n_watch       INT,
  max_psi       DOUBLE,
  top_drifting  VARCHAR(200)
);

CREATE TABLE IF NOT EXISTS model_registry_log (
  log_id     INT AUTO_INCREMENT PRIMARY KEY,
  log_time   DATETIME NOT NULL,
  model_kind VARCHAR(16) NOT NULL,
  event      VARCHAR(24) NOT NULL,
  model_name VARCHAR(80),
  details    TEXT
);
