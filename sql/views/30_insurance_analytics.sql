-- Premium prediction features + the monthly premium series used for the forecast view.
-- Demonstrates: LAG, rolling-window SUM, CASE banding.

DROP VIEW IF EXISTS vw_insurance_health_features;
CREATE VIEW vw_insurance_health_features AS
SELECT
  policy_id, application_date, age, sex, bmi, children, smoker, pre_existing_condition, zone, city_tier,
  occupation, sum_assured_inr,
  CASE WHEN bmi < 18.5 THEN 'Underweight' WHEN bmi < 23 THEN 'Normal'            -- Asian-Pacific BMI cut-offs
       WHEN bmi < 27.5 THEN 'Overweight' ELSE 'Obese' END           AS bmi_band,
  CASE WHEN age < 30 THEN '18-29' WHEN age < 40 THEN '30-39' WHEN age < 50 THEN '40-49' ELSE '50+' END AS age_band,
  DATEDIFF(application_date, '2023-11-01') / 365.25                 AS t_years,
  annual_premium_inr
FROM insurance_clean
WHERE plan_type = 'Health' AND target_invalid = 0;

DROP VIEW IF EXISTS vw_insurance_monthly_gwp;
CREATE VIEW vw_insurance_monthly_gwp AS
SELECT
  month, plan_type, policies, written_premium_inr, avg_premium_inr,
  ROUND(100.0 * (written_premium_inr - LAG(written_premium_inr) OVER (PARTITION BY plan_type ORDER BY month))
        / NULLIF(LAG(written_premium_inr) OVER (PARTITION BY plan_type ORDER BY month), 0), 1) AS mom_growth_pct,
  SUM(written_premium_inr) OVER (PARTITION BY plan_type ORDER BY month
                                 ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)                      AS trailing_12m_premium_inr
FROM (
  SELECT SUBSTR(CAST(application_date AS CHAR), 1, 7) AS month,
         plan_type,
         COUNT(*)                                  AS policies,
         ROUND(SUM(annual_premium_inr), 0)         AS written_premium_inr,
         ROUND(AVG(annual_premium_inr), 0)         AS avg_premium_inr
  FROM insurance_clean
  WHERE target_invalid = 0
  GROUP BY SUBSTR(CAST(application_date AS CHAR), 1, 7), plan_type
) m;
