"""MySQL feature store: loads raw landing tables, cleansed tables and the SQL views.
Works against any SQLAlchemy URL; SQLite is supported as a stand-in for tests (MySQL functions are shimmed)."""
from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, event, text

from . import config
from .etl import FILES, read_raw

SQL_DIR = config.ROOT / "sql"
RAW_TABLES = {"loan": "raw_loan_applications", "loan_new": "raw_loan_new",
              "insurance": "raw_insurance_policies", "insurance_new": "raw_insurance_new"}
CLEAN_TABLES = {"loan": "loan_clean", "loan_new": "loan_new_clean",
                "insurance": "insurance_clean", "insurance_new": "insurance_new_clean"}


# ---- SQLite shims for the few MySQL-only functions used in the views (tests only) ----
def _str_to_date(s, fmt):
    if s is None:
        return None
    py = fmt.replace("%b", "%b")
    try:
        return datetime.strptime(s, py).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _datediff(a, b):
    try:
        return (datetime.strptime(str(a)[:10], "%Y-%m-%d") - datetime.strptime(str(b)[:10], "%Y-%m-%d")).days
    except Exception:
        return None


def _sqlite_shims(dbapi_conn, _):
    if isinstance(dbapi_conn, sqlite3.Connection):
        dbapi_conn.create_function("STR_TO_DATE", 2, _str_to_date)
        dbapi_conn.create_function("DATEDIFF", 2, _datediff)


_ENGINE = None


def get_engine(url: str | None = None):
    global _ENGINE
    url = url or config.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if _ENGINE is None or str(_ENGINE.url) != url:
        _ENGINE = create_engine(url, pool_pre_ping=True)
        if _ENGINE.dialect.name == "sqlite":
            event.listen(_ENGINE, "connect", _sqlite_shims)
    return _ENGINE


def wait_for_db(timeout: int = 120):
    """The MySQL container needs a few seconds to accept connections."""
    end = time.time() + timeout
    while True:
        try:
            with get_engine().connect() as c:
                c.execute(text("SELECT 1"))
            return
        except Exception as e:                      # noqa: BLE001
            if time.time() > end:
                raise RuntimeError(f"database not reachable: {e}") from e
            time.sleep(3)


def run_sql_file(path: Path, engine=None):
    engine = engine or get_engine()
    sql = re.sub(r"--[^\n]*", "", path.read_text())          # strip line comments
    with engine.begin() as conn:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                conn.execute(text(stmt))


def create_tables(engine=None):
    run_sql_file(SQL_DIR / "02_tables.sql", engine)


def create_views(engine=None):
    for f in sorted((SQL_DIR / "views").glob("*.sql")):
        run_sql_file(f, engine)


def _to_db(df: pd.DataFrame, table: str, engine, chunk: int = 5000):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    out = out.astype(object).where(out.notna(), None)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table}"))
        out.to_sql(table, conn, if_exists="append", index=False, chunksize=chunk)


def load_all(engine=None):
    """raw CSV -> raw_* tables; cleaned parquet -> *_clean tables; then (re)create the views."""
    engine = engine or get_engine()
    create_tables(engine)
    for name in FILES:
        _to_db(read_raw(name), RAW_TABLES[name], engine)
        from . import config as cfg
        _to_db(pd.read_parquet(cfg.DATA_PROCESSED / FILES[name][2]), CLEAN_TABLES[name], engine)
        print(f"[db] loaded {name}")
    create_views(engine)


def read_clean(name: str) -> pd.DataFrame:
    df = pd.read_sql(text(f"SELECT * FROM {CLEAN_TABLES[name]}"), get_engine())
    df["application_date"] = pd.to_datetime(df["application_date"])
    return df


def read_view(view: str) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT * FROM {view}"), get_engine())


def log_drift_to_db(kind: str, rep: dict):
    if not config.DATABASE_URL:
        return
    with get_engine().begin() as conn:
        conn.execute(text("INSERT INTO monitoring_drift_runs (run_time, model_kind, window_name, overall, n_alert, n_watch, max_psi, top_drifting) "
                          "VALUES (:t,:k,:w,:o,:a,:wa,:m,:top)"),
                     dict(t=datetime.utcnow(), k=kind, w=rep["name"], o=rep["overall"], a=rep["n_alert"], wa=rep["n_watch"],
                          m=rep["max_psi"], top=",".join(rep["top_drifting"])))
