-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Delta Live Tables - Silver Layer (SQL)
-- MAGIC
-- MAGIC > **IMPORTANT:** All data in this project is **FICTIONAL** and created for demonstration purposes only.
-- MAGIC
-- MAGIC I built this pipeline using SQL to demonstrate that DLT supports both Python and SQL.
-- MAGIC The Silver layer cleans, normalizes, and enriches data from Bronze.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Clean Orders
-- MAGIC
-- MAGIC Transform raw orders with proper data types and business logic.

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE silver_orders
COMMENT "Cleaned orders with proper types. ALL DATA IS FICTIONAL."
TBLPROPERTIES (
  "quality" = "silver",
  "pipelines.autoOptimize.managed" = "true"
)
AS
SELECT
  order_id,
  customer_id,
  product_id,
  CAST(quantity AS INT) AS quantity,
  CAST(unit_price AS DECIMAL(10, 2)) AS unit_price,
  CAST(quantity * unit_price AS DECIMAL(10, 2)) AS total_amount,
  TO_DATE(order_date) AS order_date,
  TRIM(UPPER(status)) AS order_status,
  CASE
    WHEN TRIM(UPPER(status)) = 'COMPLETED' THEN TRUE
    ELSE FALSE
  END AS is_completed,
  _ingestion_timestamp,
  current_timestamp() AS _processed_timestamp,
  TRUE AS _is_fictional_data
FROM LIVE.bronze_orders
WHERE order_id IS NOT NULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Clean Customers
-- MAGIC
-- MAGIC Normalize customer data with standardized formats.

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE silver_customers
COMMENT "Cleaned customer data. ALL DATA IS FICTIONAL - no real persons."
TBLPROPERTIES (
  "quality" = "silver"
)
AS
SELECT
  customer_id,
  INITCAP(TRIM(name)) AS customer_name,
  LOWER(TRIM(email)) AS email,
  TRIM(UPPER(country)) AS country,
  TO_DATE(registration_date) AS registration_date,
  DATEDIFF(current_date(), TO_DATE(registration_date)) AS days_since_registration,
  _ingestion_timestamp,
  current_timestamp() AS _processed_timestamp,
  TRUE AS _is_fictional_data
FROM LIVE.bronze_customers
WHERE customer_id IS NOT NULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Clean Payments
-- MAGIC
-- MAGIC Standardize payment data with proper categorization.

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE silver_payments
COMMENT "Cleaned payment data. ALL DATA IS FICTIONAL."
TBLPROPERTIES (
  "quality" = "silver"
)
AS
SELECT
  payment_id,
  order_id,
  CAST(amount AS DECIMAL(10, 2)) AS amount,
  LOWER(TRIM(payment_method)) AS payment_method,
  CASE
    WHEN LOWER(payment_method) IN ('credit_card', 'debit_card') THEN 'card'
    WHEN LOWER(payment_method) = 'paypal' THEN 'digital_wallet'
    WHEN LOWER(payment_method) = 'bank_transfer' THEN 'bank'
    ELSE 'other'
  END AS payment_category,
  TO_DATE(payment_date) AS payment_date,
  TRIM(UPPER(status)) AS payment_status,
  _ingestion_timestamp,
  current_timestamp() AS _processed_timestamp,
  TRUE AS _is_fictional_data
FROM LIVE.bronze_payments
WHERE payment_id IS NOT NULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Enriched Orders (Join)
-- MAGIC
-- MAGIC Join orders with customers and payments for a complete view.

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE silver_enriched_orders
COMMENT "Orders enriched with customer and payment info. ALL DATA IS FICTIONAL."
TBLPROPERTIES (
  "quality" = "silver"
)
AS
SELECT
  o.order_id,
  o.customer_id,
  c.customer_name,
  c.email AS customer_email,
  c.country AS customer_country,
  o.product_id,
  o.quantity,
  o.unit_price,
  o.total_amount,
  o.order_date,
  o.order_status,
  o.is_completed,
  p.payment_id,
  p.amount AS payment_amount,
  p.payment_method,
  p.payment_category,
  p.payment_date,
  p.payment_status,
  c.days_since_registration AS customer_tenure_days,
  current_timestamp() AS _processed_timestamp,
  TRUE AS _is_fictional_data
FROM LIVE.silver_orders o
LEFT JOIN LIVE.silver_customers c ON o.customer_id = c.customer_id
LEFT JOIN LIVE.silver_payments p ON o.order_id = p.order_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Validated Orders (with Expectations)
-- MAGIC
-- MAGIC Final Silver table with strict data quality checks.

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE silver_validated_orders (
  CONSTRAINT valid_total "total_amount > 0" EXPECT (total_amount > 0),
  CONSTRAINT valid_customer "customer_id IS NOT NULL" EXPECT (customer_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_status "order_status IN ('COMPLETED', 'PENDING', 'CANCELLED')" EXPECT (order_status IN ('COMPLETED', 'PENDING', 'CANCELLED'))
)
COMMENT "Validated orders with strict quality checks. ALL DATA IS FICTIONAL."
TBLPROPERTIES (
  "quality" = "silver",
  "pipelines.autoOptimize.managed" = "true"
)
AS
SELECT
  order_id,
  customer_id,
  customer_name,
  customer_country,
  product_id,
  quantity,
  unit_price,
  total_amount,
  order_date,
  order_status,
  payment_status,
  payment_method,
  customer_tenure_days,
  _processed_timestamp,
  _is_fictional_data
FROM LIVE.silver_enriched_orders
WHERE order_id IS NOT NULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. Quarantine Table
-- MAGIC
-- MAGIC Capture rows that fail validation for investigation.

-- COMMAND ----------

CREATE OR REFRESH LIVE TABLE silver_quarantine
COMMENT "Orders that failed validation. For data quality investigation. FICTIONAL DATA."
TBLPROPERTIES (
  "quality" = "silver"
)
AS
SELECT
  order_id,
  customer_id,
  total_amount,
  order_status,
  'validation_failed' AS quarantine_reason,
  current_timestamp() AS quarantine_timestamp,
  TRUE AS _is_fictional_data
FROM LIVE.silver_enriched_orders
WHERE total_amount <= 0
   OR customer_id IS NULL
   OR order_status NOT IN ('COMPLETED', 'PENDING', 'CANCELLED')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC In this Silver layer SQL pipeline, I demonstrated:
-- MAGIC
-- MAGIC | Feature | SQL Syntax |
-- MAGIC |---------|-----------|
-- MAGIC | **Live Tables** | `CREATE OR REFRESH LIVE TABLE` |
-- MAGIC | **Expectations** | `CONSTRAINT ... EXPECT` |
-- MAGIC | **Drop Invalid** | `ON VIOLATION DROP ROW` |
-- MAGIC | **Joins** | Standard SQL joins with `LIVE.` prefix |
-- MAGIC | **Quarantine** | Separate table for invalid data |
-- MAGIC
-- MAGIC **Why SQL for Silver Layer?**
-- MAGIC - Easier for analysts to understand
-- MAGIC - Familiar syntax for data transformations
-- MAGIC - Self-documenting queries
-- MAGIC - Can be reviewed by non-programmers
-- MAGIC
-- MAGIC > **Reminder:** All data is FICTIONAL and for demonstration only.
