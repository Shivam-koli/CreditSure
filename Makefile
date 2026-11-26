export PYTHONPATH := src
.PHONY: install etl train drift retrain api test up down mlflow-ui
install: ; pip install -r requirements-dev.txt
etl:     ; python -m lia.etl
train:   ; python -m lia.loan_train && python -m lia.insurance_train
drift:   ; python -m lia.drift --kind loan --current new && python -m lia.drift --kind insurance --current new
retrain: ; python -m lia.retrain --kind all
api:     ; uvicorn lia.api:app --port 8000
test:    ; pytest -q
up:      ; docker compose up --build
down:    ; docker compose down
mlflow-ui:  ; docker compose --profile tools run --rm --service-ports mlflow-ui  # http://localhost:5000 (no Docker: mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db)
