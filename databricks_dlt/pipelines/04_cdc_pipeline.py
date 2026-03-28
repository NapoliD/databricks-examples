# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Live Tables - CDC Pipeline
# MAGIC
# MAGIC > **IMPORTANT:** All data in this project is **FICTIONAL** and created for demonstration purposes only.
# MAGIC
# MAGIC I built this pipeline to demonstrate Change Data Capture (CDC) with DLT's `apply_changes()`.
# MAGIC This is how I would handle real-time updates from transactional systems.

# COMMAND ----------

import dlt
from pyspark.sql.functions import (
    col, current_timestamp, lit, expr, when,
    to_timestamp, coalesce
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# CDC source path - All data is FICTIONAL
CDC_PATH = "/mnt/cdc/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CDC Source - Customer Changes
# MAGIC
# MAGIC I ingest CDC events that contain INSERT, UPDATE, and DELETE operations.

# COMMAND ----------

@dlt.table(
    name="cdc_customers_raw",
    comment="Raw CDC events for customers. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "bronze"
    }
)
def cdc_customers_raw():
    """
    Ingest raw CDC events from source system.

    CDC event structure:
    - customer_id: Primary key
    - name, email, country: Customer attributes
    - operation: 'INSERT', 'UPDATE', or 'DELETE'
    - updated_at: Timestamp of the change

    Note: All customer data is FICTIONAL.
    """
    # In production, this would read from Kafka, Kinesis, or CDC files
    # For demonstration, I create synthetic CDC data

    return (
        spark.read
        .format("json")
        .option("multiLine", "true")
        .load(f"{CDC_PATH}/customers/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Apply Changes - SCD Type 1
# MAGIC
# MAGIC SCD Type 1 simply overwrites the old value with the new one.
# MAGIC No history is kept - only the current state.

# COMMAND ----------

dlt.create_streaming_table(
    name="customers_scd1",
    comment="Customers with SCD Type 1 (overwrite). ALL DATA IS FICTIONAL."
)

dlt.apply_changes(
    target="customers_scd1",
    source="cdc_customers_raw",
    keys=["customer_id"],
    sequence_by=col("updated_at"),
    apply_as_deletes=expr("operation = 'DELETE'"),
    except_column_list=["operation", "_ingestion_timestamp", "_is_fictional_data"],
    stored_as_scd_type=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Apply Changes - SCD Type 2
# MAGIC
# MAGIC SCD Type 2 keeps full history of changes.
# MAGIC Each change creates a new row with validity timestamps.

# COMMAND ----------

dlt.create_streaming_table(
    name="customers_scd2",
    comment="Customers with SCD Type 2 (full history). ALL DATA IS FICTIONAL."
)

dlt.apply_changes(
    target="customers_scd2",
    source="cdc_customers_raw",
    keys=["customer_id"],
    sequence_by=col("updated_at"),
    apply_as_deletes=expr("operation = 'DELETE'"),
    except_column_list=["operation", "_ingestion_timestamp", "_is_fictional_data"],
    stored_as_scd_type=2
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. CDC Source - Order Changes
# MAGIC
# MAGIC Similar pattern for order updates.

# COMMAND ----------

@dlt.table(
    name="cdc_orders_raw",
    comment="Raw CDC events for orders. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "bronze"
    }
)
def cdc_orders_raw():
    """
    Ingest raw CDC events for orders.

    Typical CDC scenarios:
    - New order placed (INSERT)
    - Order status changed (UPDATE)
    - Order cancelled (UPDATE or soft DELETE)

    Note: All order data is FICTIONAL.
    """
    return (
        spark.read
        .format("json")
        .option("multiLine", "true")
        .load(f"{CDC_PATH}/orders/")
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Apply Changes - Orders SCD Type 1

# COMMAND ----------

dlt.create_streaming_table(
    name="orders_current",
    comment="Current order state (SCD Type 1). ALL DATA IS FICTIONAL."
)

dlt.apply_changes(
    target="orders_current",
    source="cdc_orders_raw",
    keys=["order_id"],
    sequence_by=col("updated_at"),
    apply_as_deletes=expr("operation = 'DELETE'"),
    except_column_list=["operation", "_ingestion_timestamp", "_is_fictional_data"],
    stored_as_scd_type=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. CDC with Custom Logic
# MAGIC
# MAGIC Sometimes we need custom handling for specific scenarios.

# COMMAND ----------

@dlt.table(
    name="cdc_orders_processed",
    comment="Processed CDC orders with custom logic. FICTIONAL DATA."
)
def cdc_orders_processed():
    """
    Process CDC events with custom business logic.

    Custom handling:
    - Soft deletes (mark as deleted instead of removing)
    - Status transitions validation
    - Audit trail creation

    Note: All data is FICTIONAL.
    """
    return (
        dlt.read("cdc_orders_raw")
        .withColumn(
            "is_deleted",
            when(col("operation") == "DELETE", True).otherwise(False)
        )
        .withColumn(
            "change_type",
            when(col("operation") == "INSERT", "new_order")
            .when(col("operation") == "UPDATE", "status_change")
            .when(col("operation") == "DELETE", "cancellation")
            .otherwise("unknown")
        )
        .withColumn("processed_at", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Audit Trail Table
# MAGIC
# MAGIC Keep track of all changes for compliance.

# COMMAND ----------

@dlt.table(
    name="cdc_audit_trail",
    comment="Audit trail of all CDC operations. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "silver"
    }
)
def cdc_audit_trail():
    """
    Create audit trail from CDC events.

    Captures:
    - What changed
    - When it changed
    - What operation was performed

    This is useful for:
    - Compliance reporting
    - Debugging data issues
    - Understanding data lineage

    Note: All audit data is FICTIONAL.
    """
    customers_changes = (
        dlt.read("cdc_customers_raw")
        .select(
            lit("customers").alias("table_name"),
            col("customer_id").alias("record_id"),
            col("operation"),
            col("updated_at").alias("change_timestamp"),
            col("_ingestion_timestamp")
        )
    )

    orders_changes = (
        dlt.read("cdc_orders_raw")
        .select(
            lit("orders").alias("table_name"),
            col("order_id").alias("record_id"),
            col("operation"),
            col("updated_at").alias("change_timestamp"),
            col("_ingestion_timestamp")
        )
    )

    return (
        customers_changes
        .union(orders_changes)
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sample CDC Data Generator
# MAGIC
# MAGIC This function creates sample CDC events for testing.
# MAGIC **All generated data is FICTIONAL.**

# COMMAND ----------

def generate_sample_cdc_data():
    """
    Generate FICTIONAL CDC data for demonstration.

    This is NOT part of the DLT pipeline - it's a utility function
    to create sample data for testing.

    ALL DATA GENERATED IS FICTIONAL.
    """
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType

    # Sample CDC events (FICTIONAL)
    sample_events = [
        # Initial inserts
        ("CUST001", "Juan Garcia", "juan@email.com", "Argentina", "INSERT", "2024-01-15 10:00:00"),
        ("CUST002", "Maria Lopez", "maria@email.com", "Mexico", "INSERT", "2024-01-15 10:05:00"),
        # Updates
        ("CUST001", "Juan Garcia", "juan.garcia@newemail.com", "Argentina", "UPDATE", "2024-01-16 14:30:00"),
        ("CUST002", "Maria Lopez Rodriguez", "maria@email.com", "Mexico", "UPDATE", "2024-01-17 09:15:00"),
        # Delete
        ("CUST003", "Carlos Test", "carlos@test.com", "Colombia", "DELETE", "2024-01-18 11:00:00"),
    ]

    schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("country", StringType(), True),
        StructField("operation", StringType(), False),
        StructField("updated_at", StringType(), False)
    ])

    df = spark.createDataFrame(sample_events, schema)
    return df.withColumn("updated_at", to_timestamp(col("updated_at")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this CDC pipeline, I demonstrated:
# MAGIC
# MAGIC | Feature | Implementation |
# MAGIC |---------|---------------|
# MAGIC | **CDC Ingestion** | `@dlt.table` for raw events |
# MAGIC | **SCD Type 1** | `apply_changes(stored_as_scd_type=1)` |
# MAGIC | **SCD Type 2** | `apply_changes(stored_as_scd_type=2)` |
# MAGIC | **Delete Handling** | `apply_as_deletes=expr(...)` |
# MAGIC | **Custom Logic** | Additional processing for soft deletes |
# MAGIC | **Audit Trail** | Track all changes for compliance |
# MAGIC
# MAGIC **Why `apply_changes()` over manual MERGE?**
# MAGIC - Automatic handling of out-of-order events
# MAGIC - Built-in SCD Type 1 and Type 2 support
# MAGIC - Declarative and easier to maintain
# MAGIC - Automatic schema evolution
# MAGIC
# MAGIC **CDC Sources in Production:**
# MAGIC - Debezium (for database CDC)
# MAGIC - AWS DMS (Database Migration Service)
# MAGIC - Azure Data Factory
# MAGIC - Kafka Connect
# MAGIC
# MAGIC > **Reminder:** All data and CDC events are FICTIONAL and for demonstration only.
