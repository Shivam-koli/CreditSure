"""Human-readable reason codes built from SHAP contributions."""
from __future__ import annotations

import pandas as pd


def rs(v) -> str:
    """Indian digit grouping: 1250000 -> 'Rs. 12,50,000'."""
    s = str(int(round(abs(float(v)))))
    if len(s) > 3:
        head, tail, parts = s[:-3], s[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        s = ",".join(([head] if head else []) + parts + [tail])
    return ("-" if float(v) < 0 else "") + "Rs. " + s

LOAN_TEXT = {
    "foir": lambda v: f"Existing + new EMIs take {v:.0%} of income",
    "cibil_score": lambda v: f"Credit bureau score is {v:.0f}",
    "cibil_new_to_credit": lambda v: "No bureau history (new to credit)" if v == 1 else "Has a bureau history",
    "credit_history_flag": lambda v: "Credit history meets guidelines" if v == 1 else "Credit history does not meet guidelines",
    "past_delinquency_24m": lambda v: f"{v:.0f} late payment(s) in the last 24 months",
    "loan_to_income": lambda v: f"Loan is {v:.1f}x annual income",
    "emi_to_income": lambda v: f"New EMI is {v:.0%} of income",
    "ltv": lambda v: "Unsecured loan" if v == 0 else f"Loan-to-value is {v:.0%}",
    "interest_rate_pct": lambda v: f"Interest rate is {v:.2f}%",
    "tenure_months": lambda v: f"Tenure is {v / 12:.1f} years",
    "monthly_income_inr": lambda v: f"Monthly income is {rs(v)}",
    "total_income": lambda v: f"Total household income is {rs(v)}/month",
    "loan_amount_inr": lambda v: f"Loan amount is {rs(v)}",
    "years_employed": lambda v: f"{v:.0f} years of work experience",
    "age": lambda v: f"Applicant age is {v:.0f}",
    "existing_loans_count": lambda v: f"{v:.0f} existing loan(s)",
    "existing_emi_inr": lambda v: f"Existing EMIs of {rs(v)}/month",
    "employment_type": lambda v: f"Employment type: {v}",
    "loan_purpose": lambda v: f"Loan purpose: {v}",
    "home_ownership": lambda v: f"Home ownership: {v}",
    "total_assets": lambda v: f"Declared assets of {rs(v)}",
    "assets_to_loan": lambda v: f"Assets cover {v:.1f}x the loan",
    "is_secured": lambda v: "Secured by collateral" if v == 1 else "No collateral",
    "city_tier": lambda v: f"City tier {v:.0f}",
    "dependents": lambda v: f"{v:.0f} dependent(s)",
    "education": lambda v: f"Education: {v}",
    "channel": lambda v: f"Applied via {v}",
    "property_area": lambda v: f"Property area: {v}",
    "co_applicant_income_inr": lambda v: f"Co-applicant income {rs(v)}/month",
    "bank_assets_inr": lambda v: f"Bank balances of {rs(v)}",
}
INS_TEXT = {
    "smoker": lambda v: f"Tobacco use: {v}",
    "bmi": lambda v: f"BMI is {v:.1f}",
    "age": lambda v: f"Age is {v:.0f}",
    "children": lambda v: f"{v:.0f} dependent child(ren)",
    "sum_assured_inr": lambda v: f"Cover of {rs(v)}",
    "pre_existing_condition": lambda v: f"Pre-existing condition: {v}",
    "city_tier": lambda v: f"City tier {v:.0f}",
    "zone": lambda v: f"Zone: {v}",
    "occupation": lambda v: f"Occupation: {v}",
    "sex": lambda v: f"Sex: {v}",
    "t_years": lambda v: "Price level for the policy year (medical inflation)",
}


def _describe(templates: dict, feat: str, value) -> str:
    fn = templates.get(feat)
    try:
        return fn(value) if fn else f"{feat} = {value}"
    except Exception:
        return f"{feat} = {value}"


def reason_codes(contrib_row: pd.Series, values_row: pd.Series, templates: dict, top_n: int = 4, min_impact: float = 0.05):
    """Return (risk_raising, risk_lowering). For the loan model a positive SHAP value = more default risk.
    For the premium model a positive value = higher premium."""
    s = contrib_row.sort_values(ascending=False)
    up = [(f, v) for f, v in s.items() if v >= min_impact][:top_n]            # ignore immaterial contributions
    down = [(f, v) for f, v in s.sort_values().items() if v <= -min_impact][:top_n]

    def pack(items):
        return [{"feature": f, "impact": round(float(v), 4), "value": None if pd.isna(values_row[f]) else values_row[f].item()
                 if hasattr(values_row[f], "item") else values_row[f],
                 "text": _describe(templates, f, values_row[f])} for f, v in items]
    return pack(up), pack(down)
