-- Analytics layer on the cleansed table. Demonstrates: COALESCE/NULLIF, window functions RANK / DENSE_RANK,
-- conditional aggregation, GROUP BY with HAVING.

DROP VIEW IF EXISTS vw_loan_features;
CREATE VIEW vw_loan_features AS
SELECT
  application_id, application_date, customer_id, branch_city_code, city, city_tier, channel, loan_purpose,
  employment_type, education, home_ownership, property_area, age, years_employed, dependents,
  monthly_income_inr, co_applicant_income_inr,
  monthly_income_inr + 0.6 * COALESCE(co_applicant_income_inr, 0)                                   AS total_income,
  (emi_inr + COALESCE(existing_emi_inr, 0)) /
      NULLIF(monthly_income_inr + 0.6 * COALESCE(co_applicant_income_inr, 0), 0)                     AS foir,
  loan_amount_inr / NULLIF(12 * (monthly_income_inr + 0.6 * COALESCE(co_applicant_income_inr, 0)), 0) AS loan_to_income,
  CASE WHEN collateral_value_inr > 0 THEN loan_amount_inr / collateral_value_inr ELSE 0 END          AS ltv,
  loan_amount_inr, tenure_months, interest_rate_pct, emi_inr, cibil_score, cibil_new_to_credit,
  credit_history_flag, past_delinquency_24m, existing_loans_count, existing_emi_inr,
  COALESCE(residential_assets_inr, 0) + commercial_assets_inr + luxury_assets_inr + bank_assets_inr   AS total_assets,
  loan_status, default_flag
FROM loan_clean;

DROP VIEW IF EXISTS vw_loan_city_rank;
CREATE VIEW vw_loan_city_rank AS
SELECT
  city, applications, avg_loan_inr, default_rate,
  RANK()       OVER (ORDER BY default_rate DESC) AS risk_rank,
  DENSE_RANK() OVER (ORDER BY avg_loan_inr DESC) AS ticket_size_rank
FROM (
  SELECT city,
         COUNT(*)                                   AS applications,
         ROUND(AVG(loan_amount_inr), 0)             AS avg_loan_inr,
         ROUND(AVG(default_flag), 4)                AS default_rate      -- NULL (Active) rows are ignored by AVG
  FROM loan_clean
  GROUP BY city
  HAVING COUNT(*) >= 100
) c;

DROP VIEW IF EXISTS vw_loan_monthly_performance;
CREATE VIEW vw_loan_monthly_performance AS
SELECT
  SUBSTR(CAST(application_date AS CHAR), 1, 7)                    AS month,
  COUNT(*)                                                         AS applications,
  SUM(CASE WHEN loan_status = 'Active' THEN 1 ELSE 0 END)          AS still_active,
  ROUND(AVG(default_flag), 4)                                      AS default_rate,
  ROUND(AVG(interest_rate_pct), 2)                                 AS avg_rate_pct,
  ROUND(AVG(CASE WHEN channel = 'Online' THEN 1.0 ELSE 0.0 END), 4) AS online_share
FROM loan_clean
GROUP BY SUBSTR(CAST(application_date AS CHAR), 1, 7);
