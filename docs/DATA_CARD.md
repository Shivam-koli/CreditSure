# Data Card - Loan Eligibility Prediction and Insurance Pricing Forecast (Gradient Boosting / XGBoost, GCP, MySQL)

**What this is:** a *synthetic dataset seeded from public datasets* (Kaggle downloads you supplied). It is not real customer data and must be described that way in the README, CV and interviews. Nothing in it comes from an employer.

Seed: `numpy.random.default_rng(42)` - fully reproducible with `generator/build.py`.

## 1. Files

| File | Rows x cols | Purpose |
|---|---|---|
| `loan_applications_raw.csv` | 24,008 x 40 | Main loan training extract (Nov 2023 - Oct 2026). 23,400 unique applications + 608 re-ingested duplicates |
| `loan_new_applications_raw.csv` | 600 x 36 | Unlabelled scoring batch, 1-5 Nov 2026 (batch inference + drift check + "scheduled retrain" demo) |
| `insurance_policies_raw.csv` | 20,133 x 28 | Main insurance extract (Nov 2023 - Oct 2026). 19,700 unique policies + 433 duplicates |
| `insurance_new_policies_raw.csv` | 500 x 25 | New policies without premium, 1-5 Nov 2026 |
| `data_quality_issue_manifest.csv` | - | Every injected defect: column, issue, rows affected, how to detect/repair |
| `generator/` | - | `build.py`, `gen_loan.py`, `gen_insurance.py`, `corrupt.py`, `validate.py`, `validate_raw.py` |

The files are deliberately **raw** (landing-zone quality). The cleaning and transformation work is part of your project. Every defect is detectable, and the format and label defects are fully reversible (round-trip tested, section 6); outliers, sentinels and unit slips need rules or imputation.

## 2. Loan dictionary (`loan_applications_raw.csv`)

| Column | Role | Notes |
|---|---|---|
| application_id | id | `LA000001`...; duplicates exist |
| application_date | time | 4 mixed formats; used for time-based split |
| customer_id | id | ~10% of applications are repeat customers |
| channel | feature | Branch / Online / DSA / Partner (Online share rises after Aug 2026) |
| branch_code, city, state, city_tier, property_area | feature | `branch_code` like `PUN-004` (parse with SUBSTR/INSTR); city aliases (Bangalore, Bombay...) |
| age, gender, marital_status, dependents, education | feature | `dependents` is 0/1/2/3+ |
| employment_type, years_employed, home_ownership | feature | Salaried / Self-Employed |
| monthly_income_inr, co_applicant_income_inr | feature | Some values are `Rs. 12,50,000` / `12.5 L` strings |
| existing_loans_count, existing_emi_inr, past_delinquency_24m | feature | |
| cibil_score | feature | 300-900; **-1 = new-to-credit (valid)**; 0/999/-99/9999 are invalid sentinels |
| credit_history_flag | feature | 1 = meets credit guidelines (Analytics Vidhya field), ~6% blank |
| loan_purpose, loan_amount_inr, tenure_months, interest_rate_pct, emi_inr | feature | |
| residential/commercial/luxury/bank_assets_inr, collateral_value_inr | feature | Collateral 0 = unsecured |
| **loan_status** | **target** | Fully Paid / Charged Off / **Active** (applications from Sep-Oct 2026 have no outcome yet: label lag, exclude from training) |
| recovery_amount_inr, days_past_due_current, collection_status | **LEAKAGE** | Post-origination. Must be dropped from features. `recovery_amount_inr` is legitimately useful only to estimate loss given default for the cost-based threshold (recoveries average ~25% of the loan, so LGD is ~75%) |
| country, source_system | constant | Drop |

Features you should engineer (not stored on purpose): FOIR = (emi + existing_emi) / (income + 0.6 x co-applicant income), loan-to-income, LTV = loan / collateral, total assets, income per dependent, employment stability, application month/quarter.

## 3. Insurance dictionary (`insurance_policies_raw.csv`)

| Column | Role | Notes |
|---|---|---|
| policy_id, application_date | id, time | |
| channel, city, state, zone, city_tier | feature | |
| age, sex, marital_status, occupation | feature | |
| annual_income_inr | feature | Strong driver of savings-plan premium |
| height_cm, weight_kg, bmi | feature | Collinear by construction; BMI stale/wrong in ~1.5% of rows |
| smoker, children, pre_existing_condition | feature | Y/N, 1/0, True/False variants |
| plan_type | segment | Health (52%) / Nonpar / Par / ULIP (names from the 100-row plan dataset) |
| policy_term_years, premium_paying_term_years | feature | Health = 1 (annual renewable) |
| sum_assured_inr | feature (**careful**) | Health: chosen cover (a legitimate input). Savings plans: 7-15x premium, correlation 0.95 with the target -> **near-leak, exclude from savings-plan models** |
| payment_frequency | feature | |
| **annual_premium_inr** | **target** | Health: risk-based. Savings plans: affordability-based (income share) |
| modal_premium_inr, agent_commission_inr | **LEAKAGE** | Derived from the target |
| country, source_system | constant | Drop |

**Recommended headline model:** Health-only (the risk-priced segment, where the linear-vs-XGBoost story is real). Keep savings plans for the plan-mix/forecast view and an optional second model.

## 4. Injected defects (see manifest for exact counts)

- Duplicates: 2.6% (loan), 2.2% (insurance) exact re-ingested rows
- Mixed date formats: ISO, dd/mm/yyyy, dd-Mon-yyyy, dd.mm.yyyy (~45% non-ISO)
- Inconsistent labels: gender, marital status, education, employment, city, purpose, plan, smoker, frequency, **and the target spelling** (e.g. `FULLY_PAID`, `Defaulted`)
- Currency strings: `Rs. 1,25,000`, `INR 45,000`, `12.5 L`, `4 Lakh`
- Missing values: MCAR on most columns, MAR (income missing more for self-employed / online channel), MNAR (assets, undisclosed pre-existing condition), structural blanks
- Impossible values: age 0/150/999 and under-18, negative income, tenure 0/999, bureau sentinels, BMI 0/5.2/71/99, negative children, PPT > term, zero/negative premium
- Unit errors: interest rate in basis points, height in metres, weight in pounds, loan amount in thousands
- Outliers: income x100, premium x10
- Whitespace padding, constant columns, high-cardinality ids

## 5. Time and drift design

- Window: 1 Nov 2023 - 31 Oct 2026; volume grows ~1.2% a month.
- Drift episode from **1 Aug 2026**: interest rates +0.6 pt (PSI 0.15), Online channel share 30% -> 46% (PSI 0.11), CIBIL slightly lower, self-employed income -10%, default rate 17.8% -> 23.4%. Insurance: BMI and smoker mix shift slightly, health premiums +7%.
- Stable control period: PSI 0.005 (Feb-Apr vs May-Jul 2026), so the monitor does not alarm on noise.
- Suggested split: train Nov 2023 - Jan 2026, validate Feb - Jul 2026, test/monitor Aug - Oct 2026.

## 6. Validation results (re-run with `generator/validate.py`, `validate_raw.py`)

**Loan** (clean version, HistGB default hyper-parameters vs standardised logistic regression):

| Model | Split | ROC-AUC | MCC (best threshold) |
|---|---|---|---|
| Logistic regression | Feb-Jul 2026 | 0.861 | 0.519 |
| Gradient boosting | Feb-Jul 2026 | 0.861 | 0.537 |
| Logistic regression | Aug-Oct 2026 (drift) | 0.861 | 0.568 |
| Gradient boosting | Aug-Oct 2026 (drift) | 0.869 | 0.591 |

So boosting wins modestly, as it should on real credit data: logistic regression is a strong baseline and a fair comparison matters more than a big gap. The best possible AUC given the generating risk model is ~0.877.

**Insurance, Health segment** (test = Feb-Oct 2026, scaled linear regression vs boosting): linear R2 0.70 / RMSE Rs. 9,900; HistGB R2 0.83 / RMSE Rs. 7,600; default XGBoost R2 0.76 (tuning will improve it). Median Health premium Rs. 8,700 (non-smoker Rs. 7,500; smoker Rs. 40,800); smoker x BMI interaction present.

**Round trip:** after de-duplication, dates, currency strings, label variants and whitespace are recovered to the truth at 100% (income 98.0%, loan amount 99.5% - the remainder are the deliberately injected sign/x100/x1000 errors and genuine blanks, which need outlier rules or imputation).

## 7. Honest limitations (say these in interviews)

1. **Synthetic.** The repayment outcome comes from a documented risk model I wrote, so reported AUC measures the pipeline, not real-world credit risk. Never claim it predicts real defaults.
2. **What the seeds really contributed.** Only the Analytics Vidhya loan set (demographics, income, credit-history mix) and `insurance.csv` (charges surface) carry real signal. The CIBIL/asset set is rule-based synthetic (approval flips at CIBIL ~550 with ~10% noise) and the 45k credit-risk set looks synthetic (interest rate uncorrelated with credit score). They were used for schema, asset ratios, age/experience and category mixes only. The 100-row plan set gave plan/term/premium structure.
3. **Smoker loading** (~5x) follows the US-based public seed, not filed Indian tariffs. It keeps the interaction structure the project teaches; do not present it as Indian pricing.
4. **Outcome label:** this is repayment behaviour on booked loans (no reject inference). Eligibility = approve when predicted default probability is below a cost-based threshold.
5. **Architecture images:** the 24 images in the knowledge file are the same six generic reference carousel diagrams (S3 data lake, Kinesis streaming, CNN images, call-centre ARCH/GARCH, LSTM text, SQL DDL/functions), not project-specific. Data requirements were therefore taken from the course outcome text plus the SQL / time-series / imbalance hooks in those diagrams.
6. **Licences - partially verified (2026-11-21).** Checked without a Kaggle login, via mirrors/citations:
   - `mirichoi0218/insurance` (Medical Cost Personal Datasets) - the dataset's own author states all of these source datasets are in the public domain. Low risk.
   - The 614-row Analytics Vidhya loan prediction seed - mirrored on OpenML tagged **CC0: Public Domain**. Low risk.
   - The ~4,269-row CIBIL/assets seed and the ~45,000-row credit-risk seed - could not confirm an explicit licence tag; Kaggle's dataset pages require a logged-in session this check couldn't reach. **Action for you:** open these two dataset pages yourself (30 seconds each, since you have a Kaggle account) and note the licence badge shown there.
   - **Why this is low-risk regardless:** the repo never commits the raw downloaded seed CSVs, only `generator/`'s synthetic output, which contains no verbatim seed rows (incomes and amounts are rescaled and re-drawn). The open item is about citing sources correctly, not about redistribution risk.

## 8. Suggested cleaning checklist (the standard flow your pipeline should show)

1. Profile raw -> data-quality report (nulls, dtypes, duplicates, cardinality).
2. De-duplicate on id (`ROW_NUMBER() OVER (PARTITION BY id ORDER BY ...)`).
3. Parse dates and currency strings; fix units (bps, metres, pounds, thousands).
4. Canonicalise categories with a mapping table; trim whitespace; fix target spelling.
5. Handle sentinels (-1 = new-to-credit flag, other codes -> null) and impossible values.
6. Missing-value strategy per column (indicator + median/mode, or KNN for numeric; BMI recomputed from height and weight).
7. Outlier handling (IQR / ratio rules / winsorise; keep a flag).
8. Drop leakage and constant columns; remove `Active` rows from training.
9. Engineer features (FOIR, LTV, ratios, calendar features), encode (ordinal/one-hot), scale for linear models.
10. Time-based split, SMOTE inside the CV fold only, then modelling, tuning, SHAP, cost-based threshold, drift check.
