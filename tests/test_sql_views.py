"""Runs the real SQL files against SQLite (MySQL-only functions are shimmed in lia.db) on a sample of the raw data."""
import pandas as pd
import pytest

from lia import config, db
from lia.cleaning import clean_insurance, clean_loan


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = db.get_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 't.db'}")
    db.create_tables(eng)
    raw_l = pd.read_csv(config.DATA_RAW / "loan_applications_raw.csv", dtype=str, keep_default_na=False, nrows=1500)
    raw_i = pd.read_csv(config.DATA_RAW / "insurance_policies_raw.csv", dtype=str, keep_default_na=False, nrows=1500)
    db._to_db(raw_l, "raw_loan_applications", eng)
    db._to_db(clean_loan(raw_l)[0], "loan_clean", eng)
    db._to_db(raw_i, "raw_insurance_policies", eng)
    db._to_db(clean_insurance(raw_i)[0], "insurance_clean", eng)
    db.create_views(eng)
    return eng


def q(eng, sql):
    return pd.read_sql(sql, eng)


def test_dedup_view_matches_python(engine):
    n_sql = q(engine, "SELECT COUNT(*) n FROM vw_loan_raw_dedup")["n"].iloc[0]
    n_py = q(engine, "SELECT COUNT(*) n FROM loan_clean")["n"].iloc[0]
    assert n_sql == n_py


def test_parsed_view(engine):
    p = q(engine, "SELECT * FROM vw_loan_raw_parsed")
    assert p["application_date"].notna().all() and p["loan_status"].notna().all()
    assert (p["branch_city_code"].str.len() == 3).all()


def test_feature_view_foir_equals_python(engine):
    from lia.features import build_loan_features
    f = q(engine, "SELECT application_id, foir FROM vw_loan_features").set_index("application_id")
    c = build_loan_features(clean_loan(pd.read_csv(config.DATA_RAW / "loan_applications_raw.csv", dtype=str,
                                                   keep_default_na=False, nrows=1500))[0]).set_index("application_id")
    assert (f["foir"] - c.loc[f.index, "foir"]).abs().max() < 1e-9


def test_rank_and_series_views(engine):
    assert len(q(engine, "SELECT * FROM vw_loan_monthly_performance")) >= 1
    g = q(engine, "SELECT * FROM vw_insurance_monthly_gwp")
    assert {"mom_growth_pct", "trailing_12m_premium_inr"} <= set(g.columns)
    assert len(q(engine, "SELECT * FROM vw_insurance_health_features")) > 0
