# Databricks notebook source
# MAGIC %md
# MAGIC # Delta Live Tables - Gold Layer
# MAGIC
# MAGIC > **IMPORTANT:** All data in this project is **FICTIONAL** and created for demonstration purposes only.
# MAGIC
# MAGIC I built this pipeline to create business-ready aggregations from Silver data.
# MAGIC The Gold layer contains KPIs, metrics, and dimensional tables for BI consumption.

# COMMAND ----------

import dlt
from pyspark.sql.functions import (
    col, sum as spark_sum, count, avg, max as spark_max, min as spark_min,
    round as spark_round, dense_rank, percent_rank, current_timestamp,
    when, lit, datediff, current_date
)
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Daily Revenue Metrics
# MAGIC
# MAGIC Key business metrics aggregated by day.

# COMMAND ----------

@dlt.table(
    name="gold_daily_revenue",
    comment="Daily revenue metrics for business reporting. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def gold_daily_revenue():
    """
    Calculate daily revenue KPIs.

    Metrics:
    - Total orders
    - Gross revenue
    - Collected revenue
    - Average order value
    - Collection rate

    Note: All data is FICTIONAL.
    """
    return (
        dlt.read("silver_validated_orders")
        .filter(col("order_status") == "COMPLETED")
        .groupBy("order_date")
        .agg(
            count("order_id").alias("total_orders"),
            spark_round(spark_sum("total_amount"), 2).alias("gross_revenue"),
            spark_round(avg("total_amount"), 2).alias("avg_order_value"),
            count("customer_id").alias("unique_customers")
        )
        .withColumn("_processed_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
        .orderBy(col("order_date").desc())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Country Performance
# MAGIC
# MAGIC Revenue and order metrics by country with rankings.

# COMMAND ----------

@dlt.table(
    name="gold_country_performance",
    comment="Country-level performance metrics with rankings. FICTIONAL DATA.",
    table_properties={
        "quality": "gold"
    }
)
def gold_country_performance():
    """
    Calculate country-level KPIs with rankings.

    Includes:
    - Revenue by country
    - Order count
    - Average order value
    - Revenue rank
    - Percentile position

    Note: All country data is FICTIONAL.
    """
    # Base aggregation
    country_metrics = (
        dlt.read("silver_validated_orders")
        .filter(col("order_status") == "COMPLETED")
        .groupBy("customer_country")
        .agg(
            count("order_id").alias("total_orders"),
            spark_round(spark_sum("total_amount"), 2).alias("total_revenue"),
            spark_round(avg("total_amount"), 2).alias("avg_order_value"),
            count("customer_id").alias("total_customers")
        )
    )

    # Add rankings
    window_revenue = Window.orderBy(col("total_revenue").desc())

    return (
        country_metrics
        .withColumn("revenue_rank", dense_rank().over(window_revenue))
        .withColumn("revenue_percentile", spark_round(percent_rank().over(window_revenue) * 100, 2))
        .withColumn("_processed_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product Performance
# MAGIC
# MAGIC Top products by revenue and quantity sold.

# COMMAND ----------

@dlt.table(
    name="gold_product_performance",
    comment="Product performance metrics. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "gold"
    }
)
def gold_product_performance():
    """
    Calculate product-level KPIs.

    Metrics:
    - Units sold
    - Revenue generated
    - Number of orders
    - Average selling price
    - Revenue and units rankings

    Note: Product data is FICTIONAL.
    """
    product_metrics = (
        dlt.read("silver_validated_orders")
        .filter(col("order_status") == "COMPLETED")
        .groupBy("product_id")
        .agg(
            spark_sum("quantity").alias("total_units_sold"),
            spark_round(spark_sum("total_amount"), 2).alias("total_revenue"),
            count("order_id").alias("order_count"),
            spark_round(avg("unit_price"), 2).alias("avg_selling_price")
        )
    )

    # Rankings
    window_revenue = Window.orderBy(col("total_revenue").desc())
    window_units = Window.orderBy(col("total_units_sold").desc())

    return (
        product_metrics
        .withColumn("revenue_rank", dense_rank().over(window_revenue))
        .withColumn("units_rank", dense_rank().over(window_units))
        .withColumn("_processed_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Customer 360 View
# MAGIC
# MAGIC Comprehensive customer metrics for segmentation and analysis.

# COMMAND ----------

@dlt.table(
    name="gold_customer_360",
    comment="Customer 360 view with lifetime metrics. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def gold_customer_360():
    """
    Create Customer 360 view with:
    - Lifetime value (LTV)
    - Order history
    - Recency, Frequency, Monetary (RFM)
    - Customer segment

    IMPORTANT: All customer data is FICTIONAL.
    No real person's information is used.
    """
    customer_metrics = (
        dlt.read("silver_validated_orders")
        .groupBy("customer_id", "customer_name", "customer_country")
        .agg(
            count("order_id").alias("total_orders"),
            spark_sum(when(col("order_status") == "COMPLETED", 1).otherwise(0)).alias("completed_orders"),
            spark_sum(when(col("order_status") == "CANCELLED", 1).otherwise(0)).alias("cancelled_orders"),
            spark_round(spark_sum("total_amount"), 2).alias("lifetime_value"),
            spark_round(avg("total_amount"), 2).alias("avg_order_value"),
            spark_max("order_date").alias("last_order_date"),
            spark_min("order_date").alias("first_order_date"),
            spark_max("customer_tenure_days").alias("tenure_days")
        )
    )

    return (
        customer_metrics
        .withColumn(
            "days_since_last_order",
            datediff(current_date(), col("last_order_date"))
        )
        .withColumn(
            "customer_segment",
            when(col("lifetime_value") >= 500, "Premium")
            .when(col("lifetime_value") >= 200, "Regular")
            .when(col("lifetime_value") >= 50, "Occasional")
            .otherwise("New")
        )
        .withColumn(
            "churn_risk",
            when(col("days_since_last_order") > 30, "High")
            .when(col("days_since_last_order") > 14, "Medium")
            .otherwise("Low")
        )
        .withColumn("_processed_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Payment Analytics
# MAGIC
# MAGIC Payment method analysis for finance team.

# COMMAND ----------

@dlt.table(
    name="gold_payment_analytics",
    comment="Payment method analytics. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "gold"
    }
)
def gold_payment_analytics():
    """
    Analyze payment methods.

    Metrics:
    - Transaction count by method
    - Revenue by method
    - Average transaction size
    - Success rate

    Note: All payment data is FICTIONAL.
    """
    return (
        dlt.read("silver_validated_orders")
        .filter(col("payment_method").isNotNull())
        .groupBy("payment_method")
        .agg(
            count("order_id").alias("transaction_count"),
            spark_round(spark_sum("total_amount"), 2).alias("total_processed"),
            spark_round(avg("total_amount"), 2).alias("avg_transaction"),
            spark_round(
                spark_sum(when(col("payment_status") == "COMPLETED", 1).otherwise(0)) /
                count("order_id") * 100, 2
            ).alias("success_rate_pct")
        )
        .withColumn("_processed_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Executive Dashboard Summary
# MAGIC
# MAGIC High-level KPIs for executive dashboards.

# COMMAND ----------

@dlt.table(
    name="gold_executive_summary",
    comment="Executive KPIs summary. ALL DATA IS FICTIONAL.",
    table_properties={
        "quality": "gold"
    }
)
def gold_executive_summary():
    """
    Create executive summary with key KPIs.

    This is a single-row table with overall metrics:
    - Total revenue
    - Total orders
    - Total customers
    - Average order value
    - Top country
    - Top product

    Note: All data is FICTIONAL.
    """
    orders_df = dlt.read("silver_validated_orders").filter(col("order_status") == "COMPLETED")

    return (
        orders_df
        .agg(
            spark_round(spark_sum("total_amount"), 2).alias("total_revenue"),
            count("order_id").alias("total_orders"),
            spark_round(avg("total_amount"), 2).alias("avg_order_value"),
            spark_min("order_date").alias("first_order_date"),
            spark_max("order_date").alias("last_order_date")
        )
        .withColumn("report_date", current_date())
        .withColumn("_processed_timestamp", current_timestamp())
        .withColumn("_is_fictional_data", lit(True))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this Gold layer pipeline, I created:
# MAGIC
# MAGIC | Table | Purpose | Key Metrics |
# MAGIC |-------|---------|-------------|
# MAGIC | `gold_daily_revenue` | Time-series analysis | Revenue, orders, AOV |
# MAGIC | `gold_country_performance` | Geographic analysis | Revenue by country, rankings |
# MAGIC | `gold_product_performance` | Product analysis | Units sold, revenue rank |
# MAGIC | `gold_customer_360` | Customer analytics | LTV, RFM, segments |
# MAGIC | `gold_payment_analytics` | Finance reporting | Payment methods, success rate |
# MAGIC | `gold_executive_summary` | Executive dashboard | Overall KPIs |
# MAGIC
# MAGIC **DLT Benefits for Gold Layer:**
# MAGIC - Automatic dependency resolution
# MAGIC - Incremental processing
# MAGIC - Built-in lineage tracking
# MAGIC - Easy to maintain and update
# MAGIC
# MAGIC > **Reminder:** All data is FICTIONAL and for demonstration only.
