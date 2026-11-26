"""raw CSV (landing zone) -> cleaned parquet (+ MySQL when DATABASE_URL is set).
Run:  python -m lia.etl"""
import json

import pandas as pd

from . import config
from .cleaning import clean_insurance, clean_loan

FILES = {
    "loan": ("loan_applications_raw.csv", clean_loan, "loan_clean.parquet"),
    "loan_new": ("loan_new_applications_raw.csv", clean_loan, "loan_new_clean.parquet"),
    "insurance": ("insurance_policies_raw.csv", clean_insurance, "insurance_clean.parquet"),
    "insurance_new": ("insurance_new_policies_raw.csv", clean_insurance, "insurance_new_clean.parquet"),
}


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_RAW / FILES[name][0], dtype=str, keep_default_na=False)


def run_etl(write_db: bool = True) -> dict:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, (_, fn, out) in FILES.items():
        clean, rep = fn(read_raw(name))
        clean.to_parquet(config.DATA_PROCESSED / out, index=False)
        reports[name] = rep
        print(f"[etl] {name:14s} {rep['rows_in']:>6} -> {rep['rows_out']:>6} rows")
    (config.REPORTS / "data_quality.json").write_text(json.dumps(reports, indent=2, default=str))
    if write_db and config.DATABASE_URL:
        from . import db
        db.load_all()
    return reports


def load_clean(name: str) -> pd.DataFrame:
    """Return cleaned data. With a database configured the feature store (MySQL) is the source of truth."""
    if config.DATABASE_URL:
        from . import db
        return db.read_clean(name)
    path = config.DATA_PROCESSED / FILES[name][2]
    if not path.exists():
        run_etl(write_db=False)
    return pd.read_parquet(path)


if __name__ == "__main__":
    run_etl()
