-- SQL twin of the Python cleaning steps 1-3 (de-duplicate, parse dates, parse money, split branch code).
-- Demonstrates: ROW_NUMBER, CASE, STR_TO_DATE, SUBSTR/INSTR, TRIM, UPPER, REPLACE, COALESCE, CAST.

DROP VIEW IF EXISTS vw_loan_raw_dedup;
CREATE VIEW vw_loan_raw_dedup AS
SELECT *
FROM (
  SELECT r.*,
         ROW_NUMBER() OVER (PARTITION BY r.application_id ORDER BY r.application_id) AS rn
  FROM raw_loan_applications r
) t
WHERE rn = 1;

DROP VIEW IF EXISTS vw_loan_raw_parsed;
CREATE VIEW vw_loan_raw_parsed AS
SELECT
  application_id,
  CASE                                                     -- four mixed date layouts, told apart by their shape
    WHEN TRIM(application_date) LIKE '____-__-__'   THEN STR_TO_DATE(TRIM(application_date), '%Y-%m-%d')
    WHEN TRIM(application_date) LIKE '__/__/____'   THEN STR_TO_DATE(TRIM(application_date), '%d/%m/%Y')
    WHEN TRIM(application_date) LIKE '__-___-____'  THEN STR_TO_DATE(TRIM(application_date), '%d-%b-%Y')
    WHEN TRIM(application_date) LIKE '__.__.____'   THEN STR_TO_DATE(TRIM(application_date), '%d.%m.%Y')
  END AS application_date,
  SUBSTR(TRIM(branch_code), 1, INSTR(TRIM(branch_code), '-') - 1) AS branch_city_code,
  TRIM(city) AS city_trimmed,
  CASE UPPER(TRIM(loan_status))
    WHEN 'FULLY PAID' THEN 'Fully Paid' WHEN 'FULLY_PAID' THEN 'Fully Paid' WHEN 'PAID' THEN 'Fully Paid'
    WHEN 'CHARGED OFF' THEN 'Charged Off' WHEN 'CHARGED_OFF' THEN 'Charged Off'
    WHEN 'CHARGED-OFF' THEN 'Charged Off' WHEN 'DEFAULTED' THEN 'Charged Off'
    WHEN 'ACTIVE' THEN 'Active'
  END AS loan_status,
  -- 'Rs. 12,50,000' / 'INR 45,000' / '12.5 L' / '4 Lakh' / '45000'  ->  number (blank stays NULL)
  CASE
    WHEN NULLIF(TRIM(monthly_income_inr), '') IS NULL THEN NULL
    WHEN UPPER(monthly_income_inr) LIKE '%LAKH' THEN CAST(REPLACE(UPPER(monthly_income_inr), 'LAKH', '') AS DECIMAL(18,2)) * 100000
    WHEN UPPER(monthly_income_inr) LIKE '%L'    THEN CAST(REPLACE(UPPER(monthly_income_inr), 'L', '') AS DECIMAL(18,2)) * 100000
    ELSE CAST(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(monthly_income_inr), 'RS.', ''), 'RS', ''), 'INR', ''), ',', '') AS DECIMAL(18,2))
  END AS monthly_income_inr
FROM vw_loan_raw_dedup;
