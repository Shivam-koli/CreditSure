"""Canonical category maps. Keys are normalised (lower-case, letters/digits only) so that
'FULLY_PAID', 'Fully paid' and 'fully-paid' all hit the same entry."""
import re

import numpy as np
import pandas as pd


def norm_key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _build(canon_to_aliases: dict) -> dict:
    out = {}
    for canon, aliases in canon_to_aliases.items():
        out[norm_key(canon)] = canon
        for a in aliases:
            out[norm_key(a)] = canon
    return out


GENDER = _build({"Male": ["M"], "Female": ["F", "Femle"]})
MARITAL = _build({"Married": ["Yes", "Y"], "Single": ["No", "N", "Unmarried"]})
EDUCATION = _build({"Graduate": ["Grad"], "Not Graduate": ["Non-Graduate", "NotGraduate", "Under Graduate"]})
EMPLOYMENT = _build({"Salaried": ["Salary"], "Self-Employed": ["Self Employed", "SELF-EMP", "Selfemployed"]})
PURPOSE = _build({
    "Home Loan": ["Housing Loan"], "Personal Loan": ["Personal", "PL"], "Auto Loan": ["Car Loan", "Vehicle Loan"],
    "Business Loan": ["MSME Loan"], "Education Loan": [], "Home Improvement": [],
    "Medical Emergency": [], "Debt Consolidation": [],
})
LOAN_STATUS = _build({"Fully Paid": ["Paid"], "Charged Off": ["Defaulted"], "Active": []})
YES_NO = _build({"Yes": ["Y", "1", "True"], "No": ["N", "0", "False"]})
PLAN = _build({
    "Health": ["Health Plan"], "Nonpar": ["Non-Par", "Non Participating"], "Par": ["Participating"],
    "ULIP": ["Unit Linked"],
})
FREQUENCY = _build({
    "Annual": ["Yearly"], "Half-Yearly": ["Semi-Annual", "Half Yearly"], "Quarterly": ["Qtrly"],
    "Monthly": ["Mthly"],
})
CITY = _build({
    "Bengaluru": ["Bangalore", "Bengaluru City", "BLR"], "Mumbai": ["Bombay"], "Delhi": ["New Delhi"],
    "Kolkata": ["Calcutta"], "Chennai": ["Madras"], "Pune": ["Poona"], "Hyderabad": ["Hydrabad"],
    "Ahmedabad": ["Ahmadabad"], "Visakhapatnam": ["Vizag"], "Mysuru": ["Mysore"], "Kochi": ["Cochin"],
    "Vadodara": ["Baroda"],
})


def canonicalize(s: pd.Series, mapping: dict, strict: bool = True) -> pd.Series:
    """Map messy labels to canonical ones. strict=True -> unknown labels become NaN;
    strict=False -> unknown labels are kept (stripped), used for open-ended fields like city."""
    s = s.astype(object)
    keys = s.map(lambda v: norm_key(v) if isinstance(v, str) and v.strip() else None)
    out = keys.map(lambda k: mapping.get(k) if k is not None else np.nan)
    if not strict:
        fallback = s.map(lambda v: v.strip() if isinstance(v, str) and v.strip() else np.nan)
        out = out.where(out.notna(), fallback)
    return out
