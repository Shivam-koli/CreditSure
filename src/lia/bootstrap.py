"""One-shot start-up job used by docker compose:  raw CSV -> MySQL -> cleaned -> trained models -> first drift reports.
Idempotent: models are only trained if they do not exist yet.   python -m lia.bootstrap [--retrain]"""
import argparse

from . import config, registry


def main(retrain: bool = False):
    from . import db
    from .etl import run_etl
    if config.DATABASE_URL:
        print("[bootstrap] waiting for MySQL ...")
        db.wait_for_db()
    run_etl(write_db=True)
    if retrain or not registry.artifact_path("loan").exists():
        from .loan_train import train_loan
        train_loan()
    if retrain or not registry.artifact_path("insurance").exists():
        from .insurance_train import train_insurance
        train_insurance()
    from .drift import run_drift
    for kind in ("loan", "insurance"):
        for cur in ("recent", "new"):
            r = run_drift(kind, cur)
            print(f"[bootstrap] drift {kind}/{cur}: {r['overall']} (max PSI {r['max_psi']})")
    print("[bootstrap] ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    main(ap.parse_args().retrain)
