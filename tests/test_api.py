"""API contract tests. Need trained models (python -m lia.bootstrap or the two *_train modules)."""
import pytest
from fastapi.testclient import TestClient

from lia import registry

pytestmark = pytest.mark.skipif(not registry.artifact_path("loan").exists() or not registry.artifact_path("insurance").exists(),
                                reason="train the models first")


@pytest.fixture(scope="module")
def client():
    from lia.api import app
    with TestClient(app) as c:
        yield c


STRONG = dict(age=34, employment_type="Salaried", years_employed=10, monthly_income_inr=95000, loan_amount_inr=1800000,
              tenure_months=180, loan_purpose="Home Loan", cibil_score=790, existing_emi_inr=8000, collateral_value_inr=2400000)
RISKY = dict(age=27, employment_type="Salaried", years_employed=3, monthly_income_inr=24000, loan_amount_inr=500000,
             tenure_months=48, loan_purpose="Personal Loan", cibil_score=585, past_delinquency_24m=3, existing_emi_inr=6500, city_tier=3)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_strong_vs_risky(client):
    a, b = client.post("/loan/score", json=STRONG).json(), client.post("/loan/score", json=RISKY).json()
    assert a["decision"] == "ELIGIBLE" and b["decision"] == "NOT_ELIGIBLE"
    assert a["probability_of_default"] < b["probability_of_default"]
    assert b["reasons_against"] and all("text" in r for r in b["reasons_against"])


def test_policy_knockout(client):
    r = client.post("/loan/score", json={**STRONG, "age": 62, "tenure_months": 240}).json()
    assert r["decision"] == "NOT_ELIGIBLE" and r["policy_flags"]


def test_protected_fields_are_not_accepted_inputs(client):
    from lia.api import LoanApplication
    assert "gender" not in LoanApplication.model_fields and "marital_status" not in LoanApplication.model_fields


def test_quote_range_and_smoker_effect(client):
    base = dict(age=35, sex="Male", height_cm=172, weight_kg=78, children=1)
    n, s = client.post("/insurance/quote", json=base).json(), client.post("/insurance/quote", json={**base, "smoker": "Yes"}).json()
    assert n["range_low_inr"] <= n["annual_premium_inr"] <= n["range_high_inr"]
    assert s["annual_premium_inr"] > 2 * n["annual_premium_inr"]
    assert client.post("/insurance/quote", json=dict(age=35, sex="Male")).status_code == 422


def test_validation_rejects_bad_input(client):
    assert client.post("/loan/score", json={**STRONG, "age": 15}).status_code == 422


def test_suggestion_for_borderline(client):
    b = dict(age=38, employment_type="Self-Employed", years_employed=6, monthly_income_inr=42000, loan_amount_inr=900000,
             tenure_months=84, loan_purpose="Business Loan", cibil_score=665, past_delinquency_24m=1, existing_emi_inr=7000, city_tier=2)
    r = client.post("/loan/score", json=b).json()
    assert r["decision"] != "ELIGIBLE" and set(r["profile"]) >= {"Credit score", "Affordability"}
    s = r["suggestion"]["max_eligible_amount_inr"]
    assert s is None or s < b["loan_amount_inr"]


def test_queue_and_detail(client):
    q = client.get("/loan/queue").json()
    assert q["n"] == 600 and sum(q["summary"].values()) == 600
    d = client.get(f"/loan/queue/{q['applications'][0]['application_id']}").json()
    assert d["decision"] in ("ELIGIBLE", "MANUAL_REVIEW", "NOT_ELIGIBLE") and d["applicant"]["application_id"]
    assert client.get("/loan/queue/NOPE").status_code == 404


def test_drift_trend_shows_august_shift(client):
    t = client.get("/monitor/drift-trend/loan").json()
    m = {r["month"]: r for r in t["months"]}
    assert m["2026-09"]["interest_rate_pct"] > 2 * m["2026-05"]["interest_rate_pct"]


def test_quote_extras(client):
    r = client.post("/insurance/quote", json=dict(age=35, sex="Male", height_cm=172, weight_kg=78)).json()
    assert 0 <= r["percentile"] <= 100 and len(r["bmi_curve"]["bmi"]) == len(r["bmi_curve"]["smoker"])
