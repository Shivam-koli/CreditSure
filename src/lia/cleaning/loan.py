"""Loan cleaning: deterministic, row-level repairs only. Statistical imputation (median / mode)
happens later inside the sklearn Pipeline so it is fitted on the training fold only (no leakage)."""
import numpy as np
import pandas as pd

from .. import mappings as M
from .common import blank_to_nan, parse_dates, parse_money

NUMERIC = ["city_tier", "age", "years_employed", "existing_loans_count", "existing_emi_inr", "past_delinquency_24m",
           "cibil_score", "credit_history_flag", "tenure_months", "interest_rate_pct", "emi_inr",
           "co_applicant_income_inr", "residential_assets_inr", "commercial_assets_inr", "luxury_assets_inr",
           "bank_assets_inr", "collateral_value_inr", "recovery_amount_inr", "days_past_due_current"]
SECURED = {"Home Loan", "Auto Loan", "Home Improvement"}


def clean_loan(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"rows_in": int(len(raw))}
    df = blank_to_nan(raw)

    # 1. de-duplicate (re-ingested records) -- SQL twin: ROW_NUMBER() OVER (PARTITION BY application_id)
    df = df.drop_duplicates("application_id", keep="first").copy()
    rep["duplicates_removed"] = rep["rows_in"] - len(df)

    # 2. dates and money strings
    df["application_date"] = parse_dates(df["application_date"])
    rep["unparseable_dates"] = int(df["application_date"].isna().sum())
    for c in ("monthly_income_inr", "loan_amount_inr"):
        df[c], fixed = parse_money(df[c])
        rep[f"{c}_currency_strings_parsed"] = fixed
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # 3. canonical labels
    for col, mp, strict in [("gender", M.GENDER, True), ("marital_status", M.MARITAL, True),
                            ("education", M.EDUCATION, True), ("employment_type", M.EMPLOYMENT, True),
                            ("loan_purpose", M.PURPOSE, True), ("city", M.CITY, False)]:
        before = df[col].copy()
        df[col] = M.canonicalize(df[col], mp, strict)
        rep[f"{col}_labels_normalised"] = int(((before != df[col]) & before.notna() & df[col].notna()).sum())
    if "loan_status" in df:
        df["loan_status"] = M.canonicalize(df["loan_status"], M.LOAN_STATUS)
    df["dependents"] = pd.to_numeric(df["dependents"].astype(object).replace({"3+": 3}), errors="coerce")
    df["branch_city_code"] = df["branch_code"].str.split("-").str[0]

    # 4. unit errors and impossible values
    m = df["interest_rate_pct"] > 100
    df.loc[m, "interest_rate_pct"] /= 100                      # basis points -> percent
    rep["interest_rate_bps_fixed"] = int(m.sum())
    m = ~df["interest_rate_pct"].between(5, 30) & df["interest_rate_pct"].notna()
    df.loc[m, "interest_rate_pct"] = np.nan
    m = df["loan_amount_inr"] < 20_000                           # product minimum is Rs 20,000
    df.loc[m, "loan_amount_inr"] *= 1000                         # entered in thousands
    rep["loan_amount_in_thousands_fixed"] = int(m.sum())
    m = df["monthly_income_inr"] < 0
    df.loc[m, "monthly_income_inr"] = df.loc[m, "monthly_income_inr"].abs()
    rep["negative_income_fixed"] = int(m.sum())
    m = df["monthly_income_inr"] > 700_000                       # > Rs 7 lakh/month goes to manual verification
    df.loc[m, "monthly_income_inr"] /= 100                       # extra-zeros entry error
    rep["income_x100_fixed"] = int(m.sum())
    m = ~df["age"].between(18, 70) & df["age"].notna()
    df.loc[m, "age"] = np.nan
    rep["invalid_age_to_null"] = int(m.sum())
    m = df["years_employed"] > (df["age"] - 18)
    df.loc[m, "years_employed"] = (df.loc[m, "age"] - 21).clip(lower=0)
    rep["experience_exceeds_age_fixed"] = int(m.sum())
    m = ~df["tenure_months"].between(6, 360) & df["tenure_months"].notna()
    df.loc[m, "tenure_months"] = np.nan
    rep["invalid_tenure_to_null"] = int(m.sum())

    # 5. bureau score: -1 is a legitimate "new to credit" code, other out-of-range codes are junk
    df["cibil_new_to_credit"] = (df["cibil_score"] == -1).astype(int)
    m = (~df["cibil_score"].between(300, 900)) & df["cibil_score"].notna()
    rep["invalid_cibil_to_null"] = int((m & (df["cibil_score"] != -1)).sum())
    df.loc[m, "cibil_score"] = np.nan

    # 6. target: 1 = Charged Off, 0 = Fully Paid, NaN = Active (no outcome yet)
    if "loan_status" in df:
        df["default_flag"] = df["loan_status"].map({"Charged Off": 1, "Fully Paid": 0})
        rep["status_counts"] = df["loan_status"].value_counts(dropna=False).astype(int).to_dict()

    df = df.sort_values(["application_date", "application_id"]).reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    rep["missing_rate_after_cleaning"] = {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0}
    return df, rep
