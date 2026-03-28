# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Live Tables - Bronze Layer
# MAGIC
# MAGIC > **IMPORTANT:** All data in this project is **FICTIONAL** and created for demonstration purposes only.
# MAGIC > This pipeline showcases Delta Live Tables (DLT) capabilities for portfolio demonstration.
# MAGIC
# MAGIC I built this pipeline to demonstrate declarative data ingestion with DLT.
# MAGIC The Bronze layer ingests raw data with schema enforcement and data quality expectations.

# COMMAND ----------

import dlt
from pyspark.sql.functions import current_timestamp, input_file_name, lit

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC In production, these paths would come from pipeline parameters.
# MAGIC For this demonstration, I use sample paths.

# COMMAND ----------

# Data paths - All data is FICTIONAL
RAW_PATH = "/mnt/raw/"
BRONZE_PATH = "/mnt/bronze/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Orders - Bronze Table
# MAGIC
# MAGIC I ingest raw orders with data quality expectations.
# MAGIC DLT automatically tracks data quality metrics.

# COMMAND ----------

@dlt.table(
    name="bronze_orders",
    comment="Raw orders data ingested from CSV. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true"
    }
)
@dlt.expect("valid_order_id", "order_id IS NOT NULL")
@dlt.expect("valid_customer_id", "customer_id IS NOT NULL")
@dlt.expect_or_drop("valid_quantity", "quantity > 0")
@dlt.expect_or_drop("valid_unit_price", "unit_price > 0")
def bronze_orders():
    """
    Ingest raw orders from CSV files.

    Expectations:
    - valid_order_id: Order ID must not be null (WARN)
    - valid_customer_id: Customer ID must not be null (WARN)
    - valid_quantity: Quantity must be positive (DROP invalid rows)
    - valid_unit_price: Price must be positive (DROP invalid rows)

    Note: All data is FICTIONAL and for demonstration only.
    """
    return (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{RAW_PATH}/orders/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Customers - Bronze Table
# MAGIC
# MAGIC Customer data with email validation expectations.

# COMMAND ----------

@dlt.table(
    name="bronze_customers",
    comment="Raw customer data. ALL DATA IS FICTIONAL - no real customer information.",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect("valid_customer_id", "customer_id IS NOT NULL")
@dlt.expect("valid_email", "email LIKE '%@%.%'")
@dlt.expect("valid_country", "country IS NOT NULL")
def bronze_customers():
    """
    Ingest raw customer data from CSV.

    Expectations:
    - valid_customer_id: Must not be null
    - valid_email: Must contain @ and . (basic email format)
    - valid_country: Country must not be null

    IMPORTANT: All customer data (names, emails) is FICTIONAL.
    """
    return (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{RAW_PATH}/customers/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Payments - Bronze Table
# MAGIC
# MAGIC Payment data from JSON files with amount validation.

# COMMAND ----------

@dlt.table(
    name="bronze_payments",
    comment="Raw payment data from JSON. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect("valid_payment_id", "payment_id IS NOT NULL")
@dlt.expect("valid_order_id", "order_id IS NOT NULL")
@dlt.expect_or_drop("valid_amount", "amount > 0")
@dlt.expect("valid_status", "status IN ('completed', 'pending', 'failed', 'refunded')")
def bronze_payments():
    """
    Ingest raw payment data from JSON files.

    Expectations:
    - valid_payment_id: Must not be null
    - valid_order_id: Must not be null
    - valid_amount: Must be positive (DROP invalid)
    - valid_status: Must be a known status value

    Note: All payment data is FICTIONAL.
    """
    return (
        spark.read
        .format("json")
        .option("multiLine", "true")
        .load(f"{RAW_PATH}/payments/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Products - Bronze Table (Streaming)
# MAGIC
# MAGIC I demonstrate streaming ingestion with Auto Loader.
# MAGIC This is how I would handle continuous data arrival in production.

# COMMAND ----------

@dlt.table(
    name="bronze_products_streaming",
    comment="Products ingested via Auto Loader (streaming). FICTIONAL DATA.",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect("valid_product_id", "product_id IS NOT NULL")
@dlt.expect_or_drop("valid_price", "price > 0")
def bronze_products_streaming():
    """
    Streaming ingestion using Auto Loader (cloudFiles).

    Auto Loader benefits:
    - Automatically discovers new files
    - Exactly-once processing guarantees
    - Schema inference and evolution
    - Efficient for high-volume data

    Note: All product data is FICTIONAL.
    """
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", f"{BRONZE_PATH}/_schemas/products")
        .load(f"{RAW_PATH}/products/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Events - Bronze Table (Real-time)
# MAGIC
# MAGIC Event data for real-time analytics demonstration.

# COMMAND ----------

@dlt.table(
    name="bronze_events",
    comment="User events for analytics. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "bronze"
    }
)
@dlt.expect("valid_event_id", "event_id IS NOT NULL")
@dlt.expect("valid_event_type", "event_type IS NOT NULL")
@dlt.expect("valid_timestamp", "event_timestamp IS NOT NULL")
def bronze_events():
    """
    Ingest user events (page views, clicks, purchases).

    Event types:
    - page_view: User viewed a page
    - add_to_cart: User added product to cart
    - purchase: User completed purchase
    - search: User performed search

    Note: All event data is FICTIONAL and synthetic.
    """
    return (
        spark.read
        .format("json")
        .load(f"{RAW_PATH}/events/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this Bronze layer pipeline, I demonstrated:
# MAGIC
# MAGIC | Feature | Implementation |
# MAGIC |---------|---------------|
# MAGIC | **Declarative Tables** | `@dlt.table` decorator |
# MAGIC | **Data Quality** | `@dlt.expect`, `@dlt.expect_or_drop` |
# MAGIC | **Batch Ingestion** | CSV and JSON files |
# MAGIC | **Streaming** | Auto Loader with `cloudFiles` |
# MAGIC | **Audit Columns** | `_ingestion_timestamp`, `_source_file` |
# MAGIC
# MAGIC **Why DLT over traditional Spark?**
# MAGIC - Automatic dependency management
# MAGIC - Built-in data quality tracking
# MAGIC - Simplified orchestration
# MAGIC - Production-ready with minimal code
# MAGIC
# MAGIC > **Reminder:** All data processed is FICTIONAL and for demonstration only.
