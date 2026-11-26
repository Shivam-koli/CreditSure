"""Tiny model registry: artifacts + a JSON history, so retrains are auditable (MLflow-lite, zero cost)."""
import json
from datetime import datetime, timezone

import joblib

from . import config

REG = config.MODELS / "registry.json"


def _read() -> dict:
    return json.loads(REG.read_text()) if REG.exists() else {"champions": {}, "history": []}


def artifact_path(kind: str):
    return config.MODELS / f"{kind}_champion.joblib"


def save_champion(kind: str, model, meta: dict, event: str = "trained") -> dict:
    config.MODELS.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {**meta, "kind": kind, "version": version}
    joblib.dump({"model": model, "meta": meta}, artifact_path(kind))
    reg = _read()
    reg["champions"][kind] = {k: meta.get(k) for k in ("version", "name", "rule", "threshold", "train_window", "metrics")}
    reg["history"].append({"time": version, "kind": kind, "event": event, "name": meta.get("name"),
                           "metrics": meta.get("metrics")})
    REG.write_text(json.dumps(reg, indent=2, default=str))
    return meta


def load_champion(kind: str) -> dict:
    p = artifact_path(kind)
    if not p.exists():
        raise FileNotFoundError(f"No {kind} model at {p}. Run: python -m lia.{kind}_train")
    return joblib.load(p)


def log_event(kind: str, event: str, payload: dict):
    reg = _read()
    reg["history"].append({"time": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "kind": kind,
                           "event": event, **payload})
    REG.write_text(json.dumps(reg, indent=2, default=str))


def read_registry() -> dict:
    return _read()
