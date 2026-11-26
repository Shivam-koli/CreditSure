"""Feature engineering (deterministic, no fitting -> safe to share between train and serve).
Protected attributes (gender, marital status) are deliberately NOT model inputs."""
import numpy as np
import pandas as pd

from . import config

SECURED = ["Home Loan", "Auto Loan", "Home Improvement"]

LOAN_NUM = ["age", "years_employed", "dependents", "monthly_income_inr", "co_applicant_income_inr", "total_income",
            "existing_loans_count", "existing_emi_inr", "past_delinquency_24m", "cibil_score", "cibil_new_to_credit",
            "credit_history_flag", "loan_amount_inr", "tenure_months", "interest_rate_pct", "emi_inr", "foir",
            "loan_to_income", "emi_to_income", "ltv", "is_secured", "total_assets", "assets_to_loan",
            "bank_assets_inr", "city_tier"]
LOAN_CAT = ["employment_type", "education", "home_ownership", "property_area", "channel", "loan_purpose"]
LOAN_FEATURES = LOAN_NUM + LOAN_CAT

INS_NUM = ["age", "bmi", "children", "sum_assured_inr", "city_tier", "t_years"]
INS_CAT = ["sex", "smoker", "pre_existing_condition", "zone", "occupation"]
INS_FEATURES = INS_NUM + INS_CAT
INS_ORIGIN = pd.Timestamp("2023-11-01")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def build_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    inc = _num(d, "monthly_income_inr")
    co = _num(d, "co_applicant_income_inr").fillna(0)
    d["total_income"] = inc + 0.6 * co                       # lenders count part of a co-applicant's income
    tot = d["total_income"].replace(0, np.nan)
    amt = _num(d, "loan_amount_inr")
    emi, ex = _num(d, "emi_inr"), _num(d, "existing_emi_inr").fillna(0)
    d["foir"] = (emi + ex) / tot                              # fixed obligations to income ratio
    d["emi_to_income"] = emi / tot
    d["loan_to_income"] = amt / (12 * tot)
    coll = _num(d, "collateral_value_inr")
    d["ltv"] = np.where(coll > 0, amt / coll.where(coll > 0), np.where(coll == 0, 0.0, np.nan))
    d["is_secured"] = d["loan_purpose"].isin(SECURED).astype(int)
    d["total_assets"] = sum(_num(d, c).fillna(0) for c in
                            ["residential_assets_inr", "commercial_assets_inr", "luxury_assets_inr", "bank_assets_inr"])
    d["assets_to_loan"] = d["total_assets"] / amt.replace(0, np.nan)
    for c in LOAN_NUM:
        if c not in d:
            d[c] = np.nan
    return d


def build_insurance_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["t_years"] = (pd.to_datetime(d["application_date"]) - INS_ORIGIN).dt.days / 365.25
    return d


def loan_xy(df: pd.DataFrame):
    d = build_loan_features(df)
    return d[LOAN_FEATURES], d["default_flag"] if "default_flag" in d else None


def insurance_xy(df: pd.DataFrame):
    d = build_insurance_features(df)
    return d[INS_FEATURES], d["annual_premium_inr"] if "annual_premium_inr" in d else None


def time_split(df: pd.DataFrame, train_end=config.TRAIN_END, val_end=config.VAL_END, date_col="application_date"):
    dt = pd.to_datetime(df[date_col])
    return df[dt < train_end], df[(dt >= train_end) & (dt < val_end)], df[dt >= val_end]
