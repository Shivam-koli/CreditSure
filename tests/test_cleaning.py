import numpy as np
import pandas as pd

from lia.cleaning import clean_insurance, clean_loan
from lia.cleaning.common import parse_dates, parse_money
from lia.mappings import LOAN_STATUS, canonicalize


def test_parse_dates_all_four_layouts():
    s = pd.Series(["2026-03-05", "05/03/2026", "05-Mar-2026", "05.03.2026", None])
    out = parse_dates(s)
    assert (out.iloc[:4] == pd.Timestamp("2026-03-05")).all()
    assert pd.isna(out.iloc[4])


def test_parse_money_indian_formats():
    vals, _ = parse_money(pd.Series(["Rs. 12,50,000", "INR 45,000", "12.5 L", "4 Lakh", "45000", "-1200", None]))
    assert vals.tolist()[:6] == [1250000, 45000, 1250000, 400000, 45000, -1200]
    assert np.isnan(vals.iloc[6])


def test_target_label_variants():
    s = pd.Series(["FULLY_PAID", "fully paid", "Paid", "Charged-Off", "Defaulted", "Active", "garbage"])
    assert canonicalize(s, LOAN_STATUS).tolist()[:6] == ["Fully Paid"] * 3 + ["Charged Off"] * 2 + ["Active"]
    assert pd.isna(canonicalize(s, LOAN_STATUS).iloc[6])


def _loan_row(**kw):
    base = dict(application_id="LA1", application_date="2026-01-02", customer_id="C1", channel="Branch",
                branch_code="PUN-001", city="Poona", state="Maharashtra", city_tier="1", property_area="Urban",
                age="30", gender="M", marital_status="Y", dependents="3+", education="grad", employment_type="Self Employed",
                years_employed="5", home_ownership="Rented", monthly_income_inr="Rs. 50,000", co_applicant_income_inr="0",
                existing_loans_count="0", existing_emi_inr="0", past_delinquency_24m="0", cibil_score="-1",
                credit_history_flag="1", loan_purpose="Car Loan", loan_amount_inr="500", tenure_months="60",
                interest_rate_pct="1050", emi_inr="10000", residential_assets_inr="0", commercial_assets_inr="0",
                luxury_assets_inr="0", bank_assets_inr="0", collateral_value_inr="600000", loan_status="FULLY_PAID",
                recovery_amount_inr="0", days_past_due_current="0", collection_status="None", country="India",
                source_system="LOS-CORE")
    base.update(kw)
    return base


def test_clean_loan_repairs_and_dedupes():
    raw = pd.DataFrame([_loan_row(), _loan_row(), _loan_row(application_id="LA2", age="150", cibil_score="9999")])
    out, rep = clean_loan(raw)
    assert rep["duplicates_removed"] == 1 and len(out) == 2
    r = out.set_index("application_id").loc["LA1"]
    assert r.city == "Pune" and r.gender == "Male" and r.education == "Graduate"
    assert r.employment_type == "Self-Employed" and r.loan_purpose == "Auto Loan"
    assert r.monthly_income_inr == 50000 and r.loan_amount_inr == 500000       # thousands fixed
    assert r.interest_rate_pct == 10.5                                          # basis points fixed
    assert r.cibil_new_to_credit == 1 and np.isnan(r.cibil_score)               # -1 = new to credit
    assert r.dependents == 3 and r.default_flag == 0 and r.branch_city_code == "PUN"
    r2 = out.set_index("application_id").loc["LA2"]
    assert np.isnan(r2.age) and np.isnan(r2.cibil_score) and r2.cibil_new_to_credit == 0


def test_clean_insurance_rebuilds_bmi_and_units():
    row = dict(policy_id="P1", application_date="2026-01-02", channel="Online", city="Bangalore", state="Karnataka",
               zone="South", city_tier="1", age="40", sex="m", marital_status="Married", occupation="IT / Office",
               annual_income_inr="Rs. 9,00,000", height_cm="1.75", weight_kg="70", bmi="99", smoker="1", children="-1",
               pre_existing_condition="N", plan_type="Health Plan", policy_term_years="1", premium_paying_term_years="1",
               sum_assured_inr="500000", payment_frequency="Yearly", annual_premium_inr="Rs. 12,000",
               modal_premium_inr="12000", agent_commission_inr="0", country="India", source_system="X")
    out, _ = clean_insurance(pd.DataFrame([row] * 3))
    r = out.iloc[0]
    assert r.height_cm == 175 and abs(r.bmi - 22.9) < 0.1                       # metres fixed, BMI rebuilt
    assert r.smoker == "Yes" and r.pre_existing_condition == "No" and r.plan_type == "Health"
    assert r.sex == "Male" and r.city == "Bengaluru" and np.isnan(r.children)
    assert r.payment_frequency == "Annual" and r.annual_income_inr == 900000
