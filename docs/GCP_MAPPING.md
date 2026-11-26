# How the local stack maps to GCP

Everything runs locally in Docker at zero cost. The same containers move to GCP with minimal changes:

| Local | GCP equivalent | Change needed |
|---|---|---|
| `data/raw/` landing folder | Cloud Storage bucket (`gs://.../landing/`) | read the CSV with `gcsfs` or `gsutil cp` before `lia.etl` |
| `mysql` container | Cloud SQL for MySQL 8, or MySQL on the same Compute Engine VM | point `DATABASE_URL` at it |
| `bootstrap` / `scheduler` containers | cron or Cloud Scheduler on the VM, or Cloud Run jobs | none: same image, same command |
| `api` container (serves both the API and the static dashboard on one port) | `docker compose up` on an e2-medium Compute Engine VM, or Cloud Run | open port 8000 (mapped to 8001 locally) or put it behind a load balancer |
| `models/` + `registry.json` | a Cloud Storage bucket (or Vertex AI Model Registry) | mount or sync the folder |
| `reports/` drift history | the `monitoring_drift_runs` table plus Cloud Monitoring alerts | none |
| MLflow (SQLite, `lia.tracking`) | a managed MLflow tracking server (Vertex AI Experiments is the closer native equivalent) | point `MLFLOW_TRACKING_URI` at the managed server instead of a local file |

The cheapest real deployment is a single VM: install Docker, `git clone`, `cp .env.example .env`, `docker compose up -d`.
