"""Insurance cleaning (row-level repairs). BMI is rebuilt from height and weight, which is the
standard fix for the stale / impossible / unit-confused BMI values in the policy-admin extract."""
import numpy as np
import pandas as pd

from .. import mappings as M
from .common import blank_to_nan, parse_dates, parse_money

NUMERIC = ["city_tier", "age", "height_cm", "weight_kg", "bmi", "children", "policy_term_years",
           "premium_paying_term_years", "sum_assured_inr", "modal_premium_inr", "agent_commission_inr"]
LBS = 2.2046


def flag_premium_outliers(df: pd.DataFrame, min_ratio: float = 7.0) -> pd.Series:
    """Model-based outlier test for the target: out-of-fold gradient-boosting prediction of log-premium,
    then flag premiums more than `min_ratio` times above the prediction. An 'extra zero' entry (x10) adds
    ~2.3 to the log value, while genuine noise rarely exceeds a 3-4x ratio."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    ok = df["annual_premium_inr"] > 0
    if ok.sum() < 100:                       # too few rows to learn a reference model (e.g. a single quote)
        return pd.Series(False, index=df.index)
    cols = ["age", "bmi", "children", "annual_income_inr", "sum_assured_inr", "policy_term_years", "city_tier"]
    X = df[cols].copy()
    for c in ("plan_type", "smoker", "pre_existing_condition", "sex"):
        X[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
    y = np.log(df.loc[ok, "annual_premium_inr"])
    oof = cross_val_predict(HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0),
                            X[ok], y, cv=KFold(5, shuffle=True, random_state=0))
    resid = y - oof
    flag = pd.Series(False, index=df.index)
    flag.loc[ok] = resid > np.log(min_ratio)
    return flag


def clean_insurance(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"rows_in": int(len(raw))}
    df = blank_to_nan(raw)
    df = df.drop_duplicates("policy_id", keep="first").copy()
    rep["duplicates_removed"] = rep["rows_in"] - len(df)

    df["application_date"] = parse_dates(df["application_date"])
    rep["unparseable_dates"] = int(df["application_date"].isna().sum())
    money = ["annual_income_inr"] + (["annual_premium_inr"] if "annual_premium_inr" in df else [])
    for c in money:
        df[c], fixed = parse_money(df[c])
        rep[f"{c}_currency_strings_parsed"] = fixed
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for col, mp, strict in [("plan_type", M.PLAN, True), ("sex", M.GENDER, True), ("smoker", M.YES_NO, True),
                            ("pre_existing_condition", M.YES_NO, True), ("payment_frequency", M.FREQUENCY, True),
                            ("city", M.CITY, False)]:
        df[col] = M.canonicalize(df[col], mp, strict)

    # ---- body measurements ----
    m = df["height_cm"] < 3
    df.loc[m, "height_cm"] *= 100                                  # metres -> cm
    rep["height_in_metres_fixed"] = int(m.sum())
    df.loc[~df["height_cm"].between(120, 215), "height_cm"] = np.nan
    df.loc[~df["weight_kg"].between(25, 250), "weight_kg"] = np.nan
    bmi_calc = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    lbs = (df["bmi"].between(12, 60) & ((df["bmi"] - bmi_calc / LBS).abs() < 1.5) & ((df["bmi"] - bmi_calc).abs() > 4))
    df.loc[lbs, "weight_kg"] = (df.loc[lbs, "weight_kg"] / LBS).round(1)
    rep["weight_in_pounds_fixed"] = int(lbs.sum())
    bmi_calc = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    bmi_calc = bmi_calc.where(bmi_calc.between(12, 55))
    given_ok = df["bmi"].between(12, 60)
    rebuild = bmi_calc.notna() & (~given_ok | ((df["bmi"] - bmi_calc).abs() > 2.5))
    rep["bmi_rebuilt_from_height_weight"] = int(rebuild.sum())
    df["bmi"] = np.where(rebuild, bmi_calc, df["bmi"].where(given_ok))
    df["bmi"] = df["bmi"].round(1)

    # ---- other impossible values ----
    m = ~df["age"].between(18, 70) & df["age"].notna()
    df.loc[m, "age"] = np.nan
    rep["invalid_age_to_null"] = int(m.sum())
    m = df["children"] < 0
    df.loc[m, "children"] = np.nan
    rep["negative_children_to_null"] = int(m.sum())
    m = df["annual_income_inr"] > 12_000_000                       # > Rs 1.2 crore: extra-zeros entry error
    df.loc[m, "annual_income_inr"] /= 100
    rep["income_x100_fixed"] = int(m.sum())
    m = df["premium_paying_term_years"] > df["policy_term_years"]
    df.loc[m, "premium_paying_term_years"] = df.loc[m, "policy_term_years"]
    rep["ppt_longer_than_term_fixed"] = int(m.sum())

    # ---- target hygiene (training rows only) ----
    if "annual_premium_inr" in df:
        bad = df["annual_premium_inr"].isna() | (df["annual_premium_inr"] <= 0)
        out = flag_premium_outliers(df.assign(annual_premium_inr=df["annual_premium_inr"].where(~bad)))
        df["target_invalid"] = (bad | out).astype(int)
        rep["target_zero_negative"] = int(bad.sum())
        rep["target_x10_outliers_flagged"] = int(out.sum())

    df = df.sort_values(["application_date", "policy_id"]).reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    rep["missing_rate_after_cleaning"] = {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0}
    return df, rep
