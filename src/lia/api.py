"""FastAPI service: loan decision + reason codes, premium quote + error range, monitoring endpoints.
Run:  uvicorn lia.api:app --host 0.0.0.0 --port 8000
Gender and marital status are NOT accepted inputs (fair-lending design); the model never sees them."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, registry
from .explain import INS_TEXT, LOAN_TEXT, reason_codes
from .features import INS_ORIGIN, SECURED, build_insurance_features, build_loan_features
from .metrics import breakeven_pd, economics
from .modeling import Explainer

STATE: dict = {}


def _load_models():
    for kind in ("loan", "insurance"):
        art = registry.load_champion(kind)
        model, meta = art["model"], art["meta"]
        pipe = model.pipe if kind == "loan" else model.point
        STATE[kind] = {"model": model, "meta": meta, "explainer": Explainer(pipe, meta["features"])}
    ref = pd.read_parquet(config.MODELS / "insurance_reference.parquet")
    STATE["ins_dist"] = np.sort(STATE["insurance"]["model"].predict(ref))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="Loan Eligibility & Insurance Premium API", version="0.1.0", lifespan=lifespan)

# The frontend (web/) is plain static HTML/CSS/JS with no build step; it fetches these same API
# routes directly from the browser. CORS only matters when the page is opened from a different
# origin than the API (e.g. a static file server during development) - when FastAPI serves the
# frontend itself (the StaticFiles mount at the bottom of this file), every call is same-origin
# and CORS is not even consulted. Wide open here on purpose: no auth, no cookies, nothing to leak.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ schemas
class LoanApplication(BaseModel):
    age: int = Field(ge=18, le=70)
    employment_type: Literal["Salaried", "Self-Employed"]
    years_employed: float = Field(0, ge=0, le=50)
    dependents: int = Field(0, ge=0, le=3, description="3 means 3 or more")
    education: Literal["Graduate", "Not Graduate"] = "Graduate"
    home_ownership: Literal["Rented", "Owned", "Mortgaged", "Other"] = "Rented"
    property_area: Literal["Urban", "Semiurban", "Rural"] = "Urban"
    city_tier: Literal[1, 2, 3] = 1
    channel: Literal["Branch", "Online", "DSA", "Partner"] = "Branch"
    monthly_income_inr: float = Field(gt=0)
    co_applicant_income_inr: float = Field(0, ge=0)
    existing_loans_count: int = Field(0, ge=0)
    existing_emi_inr: float = Field(0, ge=0)
    past_delinquency_24m: int = Field(0, ge=0)
    cibil_score: Optional[int] = Field(None, ge=300, le=900, description="omit for new-to-credit applicants")
    credit_history_flag: Optional[Literal[0, 1]] = 1
    loan_purpose: Literal["Home Loan", "Personal Loan", "Auto Loan", "Education Loan", "Business Loan",
                          "Home Improvement", "Medical Emergency", "Debt Consolidation"]
    loan_amount_inr: float = Field(gt=0)
    tenure_months: int = Field(ge=6, le=360)
    interest_rate_pct: Optional[float] = Field(None, ge=5, le=30, description="defaults to the typical rate for the purpose")
    residential_assets_inr: float = Field(0, ge=0)
    commercial_assets_inr: float = Field(0, ge=0)
    luxury_assets_inr: float = Field(0, ge=0)
    bank_assets_inr: float = Field(0, ge=0)
    collateral_value_inr: float = Field(0, ge=0, description="0 for unsecured loans")


class InsuranceRequest(BaseModel):
    age: int = Field(ge=18, le=70)
    sex: Literal["Male", "Female"]
    height_cm: Optional[float] = Field(None, ge=120, le=215)
    weight_kg: Optional[float] = Field(None, ge=25, le=250)
    bmi: Optional[float] = Field(None, ge=12, le=60, description="give BMI or height + weight")
    children: int = Field(0, ge=0, le=6)
    smoker: Literal["Yes", "No"] = "No"
    pre_existing_condition: Literal["Yes", "No"] = "No"
    zone: Literal["North", "South", "East", "West", "Central"] = "West"
    city_tier: Literal[1, 2, 3] = 1
    occupation: Literal["IT / Office", "Business Owner", "Government", "Healthcare", "Manufacturing",
                        "Professional (CA/Lawyer/Architect)"] = "IT / Office"
    sum_assured_inr: Literal[300000, 500000, 1000000, 1500000, 2500000] = 500000
    policy_date: Optional[date] = None


# ------------------------------------------------------------------ helpers
def _emi(p: float, rate: float, n: int) -> float:
    r = rate / 1200
    return p * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _risk_band(pd_: float) -> str:
    return "Low" if pd_ < 0.08 else "Moderate" if pd_ < 0.20 else "Elevated" if pd_ < 0.40 else "High"


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "models": {k: STATE[k]["meta"]["version"] for k in ("loan", "insurance")}}


def _thresholds(rows: pd.DataFrame, meta: dict) -> np.ndarray:
    econ = economics(rows, meta["recovery_ratio"])
    if meta["rule"] == "breakeven":
        return breakeven_pd(econ), econ
    return np.full(len(rows), float(meta["threshold"])), econ


def _policy_flags(row: pd.Series) -> list[str]:
    flags = []
    foir, age = row.get("foir"), row.get("age")
    if pd.notna(foir) and foir > 0.85:
        flags.append(f"Policy: fixed obligations would take {foir:.0%} of income (limit 85%)")
    if pd.notna(age) and pd.notna(row.get("tenure_months")) and age + row["tenure_months"] / 12 > 70:
        flags.append(f"Policy: borrower would be {age + row['tenure_months'] / 12:.0f} at maturity (limit 70)")
    return flags


def _decision(p: float, thr: float, flags: list) -> str:
    if flags:
        return "NOT_ELIGIBLE"
    if p < thr * (1 - config.REVIEW_BAND):
        return "ELIGIBLE"
    if p < thr * (1 + config.REVIEW_BAND):
        return "MANUAL_REVIEW"
    return "NOT_ELIGIBLE"


def _clip01(x) -> float:
    return float(np.clip(0 if pd.isna(x) else x, 0, 1))


def _profile(r: pd.Series) -> dict:
    """0-100 scores for the applicant radar chart (higher = stronger applicant). Descriptive, not model inputs."""
    cibil = r.get("cibil_score")
    credit = 0.35 if pd.isna(cibil) else (cibil - 300) / 600
    return {
        "Credit score": round(100 * _clip01(credit)),
        "Affordability": round(100 * _clip01(1 - (r.get("foir") or 0) / 0.85)),
        "Repayment record": round(100 * _clip01(1 - (r.get("past_delinquency_24m") or 0) / 4) *
                                  (0.6 if r.get("credit_history_flag") == 0 else 1.0)),
        "Job stability": round(100 * _clip01((r.get("years_employed") or 0) / 15)),
        "Asset cover": round(100 * _clip01((r.get("assets_to_loan") or 0) / 2)),
        "Income level": round(100 * _clip01(np.log10(max(r.get("total_income") or 1, 1)) / 1.8 - 2.0)),
    }


def _explain_loan(row: pd.DataFrame, p: float, thr: float, econ_row: pd.Series, extra: dict) -> dict:
    meta, expl = STATE["loan"]["meta"], STATE["loan"]["explainer"]
    r = row.iloc[0]
    flags = _policy_flags(r)
    contrib = expl.contributions(row).iloc[0]
    against, favour = reason_codes(contrib, row[meta["features"]].iloc[0], LOAN_TEXT, top_n=4, min_impact=0.05)
    margin, loss = float(econ_row["margin"]), float(econ_row["loss"])
    num = lambda v: None if pd.isna(v) else float(v)
    return {
        "decision": _decision(p, thr, flags), "probability_of_default": round(p, 4), "approval_threshold": round(thr, 4),
        "decision_rule": "break-even PD of this loan (margin / (margin + loss))" if meta["rule"] == "breakeven" else "global cost-optimal threshold",
        "risk_band": _risk_band(p), "policy_flags": flags,
        "emi_inr": round(float(r["emi_inr"])), "foir": round(float(r["foir"]), 4),
        "interest_rate_pct": float(r["interest_rate_pct"]),
        "cibil_score": num(r.get("cibil_score")), "loan_to_income": num(r.get("loan_to_income")),
        "loan_amount_inr": num(r.get("loan_amount_inr")), "tenure_months": num(r.get("tenure_months")),
        "economics": {"net_interest_if_repaid": round(margin), "loss_if_default": round(loss),
                      "expected_net_value": round((1 - p) * margin - p * loss)},
        "profile": _profile(r),
        "reasons_against": against, "reasons_for": favour,
        "model": {"name": meta["name"], "version": meta["version"]}, **extra,
    }


@app.post("/loan/score")
def score_loan(app_in: LoanApplication):
    meta, model = STATE["loan"]["meta"], STATE["loan"]["model"]
    d = app_in.model_dump()
    rate = d.pop("interest_rate_pct") or meta.get("rate_by_purpose", {}).get(d["loan_purpose"], 12.0)
    d.update(interest_rate_pct=rate, emi_inr=_emi(d["loan_amount_inr"], rate, d["tenure_months"]),
             cibil_new_to_credit=int(d["cibil_score"] is None), application_date=pd.Timestamp.today().normalize())
    if d["cibil_score"] is None:
        d["cibil_score"] = np.nan
    if d["loan_purpose"] in SECURED and not d["collateral_value_inr"]:
        d["collateral_value_inr"] = np.nan                  # secured product without a stated collateral value
    if d["credit_history_flag"] is None:
        d["credit_history_flag"] = np.nan
    row = build_loan_features(pd.DataFrame([d]))
    p = float(model.predict_proba(row)[0])
    thr, econ = _thresholds(row, meta)
    out = _explain_loan(row, p, float(thr[0]), econ.iloc[0], {})
    out["suggestion"] = None if out["decision"] == "ELIGIBLE" else _max_eligible_amount(d, rate)
    return out


def _max_eligible_amount(d: dict, rate: float):
    """What-if: the largest loan amount (same tenure, rate and profile) that the model would approve."""
    meta, model = STATE["loan"]["meta"], STATE["loan"]["model"]
    factors = np.round(np.arange(0.95, 0.04, -0.05), 2)
    grid = pd.DataFrame([d] * len(factors))
    grid["loan_amount_inr"] = d["loan_amount_inr"] * factors
    grid["emi_inr"] = [_emi(a, rate, d["tenure_months"]) for a in grid["loan_amount_inr"]]
    g = build_loan_features(grid)
    pds = model.predict_proba(g)
    thr, _ = _thresholds(g, meta)
    for (_, r), p, t in zip(g.iterrows(), pds, thr):
        if _decision(float(p), float(t), _policy_flags(r)) == "ELIGIBLE":
            amt = int(r["loan_amount_inr"] // 10000 * 10000)
            return {"max_eligible_amount_inr": amt, "emi_inr": round(float(r["emi_inr"])),
                    "probability_of_default": round(float(p), 4)}
    return {"max_eligible_amount_inr": None}


# ------------------------------------------------------------------ applications queue (new, unlabelled batch)
def _queue() -> pd.DataFrame:
    key = STATE["loan"]["meta"]["version"]
    if STATE.get("queue_key") != key:
        from .etl import load_clean
        meta, model = STATE["loan"]["meta"], STATE["loan"]["model"]
        df = build_loan_features(load_clean("loan_new")).reset_index(drop=True)
        df["pd"] = model.predict_proba(df)
        df["threshold"], _ = _thresholds(df, meta)
        df["decision"] = [_decision(p, t, _policy_flags(r)) for p, t, (_, r) in zip(df["pd"], df["threshold"], df.iterrows())]
        STATE["queue"], STATE["queue_key"] = df, key
    return STATE["queue"]


@app.get("/loan/queue")
def loan_queue(limit: int = 600):
    q = _queue()
    cols = ["application_id", "application_date", "loan_purpose", "loan_amount_inr", "monthly_income_inr",
            "employment_type", "city", "cibil_score", "pd", "threshold", "decision"]
    out = q[cols].head(limit).copy()
    out["application_date"] = out["application_date"].dt.strftime("%Y-%m-%d")
    summary = q["decision"].value_counts().to_dict()
    return {"summary": {k: int(v) for k, v in summary.items()}, "n": int(len(q)),
            "applications": json.loads(out.to_json(orient="records"))}


@app.get("/loan/queue/{application_id}")
def loan_queue_item(application_id: str):
    q = _queue()
    hit = q[q["application_id"] == application_id]
    if hit.empty:
        raise HTTPException(404, "unknown application_id")
    meta = STATE["loan"]["meta"]
    row = hit.iloc[[0]].reset_index(drop=True)
    _, econ = _thresholds(row, meta)
    r = row.iloc[0]
    applicant = {k: (None if pd.isna(r[k]) else (r[k].item() if hasattr(r[k], "item") else r[k]))
                 for k in ("application_id", "loan_purpose", "employment_type", "city", "age", "monthly_income_inr",
                           "loan_amount_inr", "tenure_months", "channel")}
    return _explain_loan(row, float(r["pd"]), float(r["threshold"]), econ.iloc[0], {"applicant": applicant})


@app.post("/insurance/quote")
def quote_insurance(req: InsuranceRequest):
    meta, model, expl = STATE["insurance"]["meta"], STATE["insurance"]["model"], STATE["insurance"]["explainer"]
    d = req.model_dump()
    bmi = d.pop("bmi")
    if bmi is None:
        if not (d["height_cm"] and d["weight_kg"]):
            raise HTTPException(422, "Provide bmi, or both height_cm and weight_kg")
        bmi = d["weight_kg"] / (d["height_cm"] / 100) ** 2
    pdate = d.pop("policy_date") or date.today()
    d.update(bmi=round(bmi, 1), application_date=pd.Timestamp(pdate))
    row = build_insurance_features(pd.DataFrame([d]))
    prem = float(model.predict(row)[0])
    lo, hi = (float(x[0]) for x in model.predict_interval(row))
    contrib = expl.contributions(row).iloc[0]
    up, down = reason_codes(contrib, row[meta["features"]].iloc[0], INS_TEXT, top_n=4, min_impact=0.02)
    for item in up + down:
        item["premium_effect_pct"] = round((float(np.exp(item["impact"])) - 1) * 100, 1)
    pct = float(np.searchsorted(STATE["ins_dist"], prem) / len(STATE["ins_dist"]) * 100)
    band = "Underweight" if bmi < 18.5 else "Normal" if bmi < 23 else "Overweight" if bmi < 27.5 else "Obese"
    grid = np.arange(17.0, 41.0, 1.0)
    curves = {}
    for smk in ("No", "Yes"):
        g = pd.concat([row] * len(grid), ignore_index=True)
        g["bmi"], g["smoker"] = grid, smk
        curves[smk] = [round(float(v)) for v in model.predict(g)]
    return {
        "bmi_curve": {"bmi": grid.tolist(), "non_smoker": curves["No"], "smoker": curves["Yes"]},
        "percentile": round(pct, 1), "bmi_band": band, "smoker": d["smoker"],
        "annual_premium_inr": round(prem), "range_low_inr": round(lo), "range_high_inr": round(hi),
        "range_confidence": "80%", "monthly_equivalent_inr": round(prem / 12), "bmi_used": round(bmi, 1),
        "raises_premium": up, "lowers_premium": down,
        "model": {"name": meta["name"], "version": meta["version"], "segment": meta["segment"],
                  "test_rmse_inr": round(meta["metrics"]["test"]["rmse"]),
                  "test_r2": round(meta["metrics"]["test"]["r2"], 3),
                  "linear_baseline_r2": round(meta["metrics"]["test"]["baseline_r2"], 3)},
    }


@app.get("/monitor/models")
def monitor_models():
    return registry.read_registry()


@app.get("/monitor/comparison/{kind}")
def monitor_comparison(kind: Literal["loan", "insurance"]):
    p = config.REPORTS / f"{kind}_model_comparison.csv"
    if not p.exists():
        raise HTTPException(404, "Run training first")
    return json.loads(pd.read_csv(p).to_json(orient="records"))


@app.get("/monitor/drift/{kind}")
def monitor_drift(kind: Literal["loan", "insurance"]):
    p = config.REPORTS / f"drift_{kind}_latest.json"
    if not p.exists():
        raise HTTPException(404, "No drift report yet. Run: python -m lia.drift --kind " + kind)
    return json.loads(p.read_text())


@app.get("/monitor/drift-trend/{kind}")
def monitor_drift_trend(kind: Literal["loan", "insurance"]):
    """PSI of each month vs the training reference, for the top drivers and for the model score."""
    from .drift import psi_categorical, psi_numeric
    from .etl import load_clean
    key = (kind, STATE[kind]["meta"]["version"])
    cache = STATE.setdefault("trend", {})
    if key in cache:
        return cache[key]
    meta, model = STATE[kind]["meta"], STATE[kind]["model"]
    ref = pd.read_parquet(config.MODELS / f"{kind}_reference.parquet")
    if kind == "loan":
        df = build_loan_features(load_clean("loan"))
        score = lambda X: model.predict_proba(X)
        feats = ["interest_rate_pct", "cibil_score", "foir", "monthly_income_inr", "loan_amount_inr", "channel"]
    else:
        df = build_insurance_features(load_clean("insurance"))
        df = df[df["plan_type"] == "Health"]
        score = lambda X: np.log(model.predict(X))
        feats = ["bmi", "age", "smoker", "sum_assured_inr", "zone"]
    ref_score = pd.Series(score(ref))
    df["month"] = df["application_date"].dt.to_period("M").astype(str)
    rows = []
    for m, g in df.groupby("month"):
        rec = {"month": m, "n": int(len(g)), "score": round(psi_numeric(ref_score, pd.Series(score(g))), 4)}
        for f in feats:
            fn = psi_numeric if pd.api.types.is_numeric_dtype(g[f]) else psi_categorical
            rec[f] = round(fn(ref[f], g[f]), 4)
        if kind == "loan":
            rec["default_rate"] = None if g["default_flag"].notna().sum() < 30 else round(float(g["default_flag"].mean()), 4)
        rows.append(rec)
    out = {"features": feats, "months": rows, "warn": config.PSI_WARN, "alert": config.PSI_ALERT}
    cache[key] = out
    return out


@app.get("/monitor/meta/{kind}")
def monitor_meta(kind: Literal["loan", "insurance"]):
    m = STATE[kind]["meta"]
    return {k: m.get(k) for k in ("name", "version", "rule", "threshold", "train_window", "metrics",
                                  "feature_importance", "best_params", "assumptions", "baseline")}


# ------------------------------------------------------------------ frontend (static HTML/CSS/JS)
# Mounted last and at "/" so it only catches requests that none of the routes above already matched
# (Starlette tries routes in registration order) - the dashboard and the API are one server, one port.
_WEB_DIR = config.ROOT / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="frontend")
