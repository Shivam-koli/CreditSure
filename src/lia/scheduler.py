"""Zero-dependency scheduler for the `scheduler` container (cron-style, no extra service).
Every DRIFT_EVERY_HOURS it runs a drift check on recent data and on the newest scoring batch;
every RETRAIN_EVERY_DAYS it runs the gated champion/challenger retrain."""
import json
import os
import time
from datetime import datetime, timezone

from . import config
from .drift import run_drift
from .retrain import main as retrain_main

STATE = config.REPORTS / "scheduler_state.json"


def _now():
    return datetime.now(timezone.utc)


def _load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def tick(state: dict, drift_hours: float, retrain_days: float) -> dict:
    now = _now()
    last_d, last_r = state.get("last_drift"), state.get("last_retrain")
    due = lambda ts, delta_h: ts is None or (now - datetime.fromisoformat(ts)).total_seconds() >= delta_h * 3600
    if due(last_d, drift_hours):
        for kind in ("loan", "insurance"):
            for cur in ("recent", "new"):
                r = run_drift(kind, cur)
                print(f"[scheduler] drift {kind}/{cur}: {r['overall']} (max PSI {r['max_psi']})", flush=True)
        state["last_drift"] = now.isoformat()
    if due(last_r, retrain_days * 24):
        retrain_main("all", force=False)
        state["last_retrain"] = now.isoformat()
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    return state


def main():
    drift_h = float(os.environ.get("DRIFT_EVERY_HOURS", 24))
    retrain_d = float(os.environ.get("RETRAIN_EVERY_DAYS", 30))
    print(f"[scheduler] drift every {drift_h}h, retrain check every {retrain_d}d", flush=True)
    state = _load()
    while True:
        try:
            state = tick(state, drift_h, retrain_d)
        except Exception as exc:                       # never let the loop die
            print(f"[scheduler] error: {exc!r}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
