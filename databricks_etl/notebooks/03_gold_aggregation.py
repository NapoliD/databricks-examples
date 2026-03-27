# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - Business Aggregations
# MAGIC Tablas agregadas listas para consumo de BI y Analytics.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as spark_sum, count, avg, max as spark_max, min as spark_min,
    round as spark_round, dense_rank, percent_rank,
    date_trunc, current_timestamp, lit,
    when, coalesce
)
from pyspark.sql.window import Window

# COMMAND ----------

SILVER_PATH = "/mnt/silver/"
GOLD_PATH = "/mnt/gold/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Revenue por Día

# COMMAND ----------

def create_daily_revenue():
    """Crea tabla de revenue diario."""

    enriched_orders = spark.read.format("delta").load(f"{SILVER_PATH}/enriched_orders")

    daily_revenue = (enriched_orders
        .filter(col("order_status") == "completed")
        .groupBy("order_date")
        .agg(
            count("order_id").alias("total_orders"),
            spark_sum("total_amount").alias("gross_revenue"),
            spark_sum("payment_amount").alias("collected_revenue"),
            avg("total_amount").alias("avg_order_value"),
            spark_sum("quantity").alias("total_items_sold")
        )
        .withColumn("gross_revenue", spark_round(col("gross_revenue"), 2))
        .withColumn("collected_revenue", spark_round(col("collected_revenue"), 2))
        .withColumn("avg_order_value", spark_round(col("avg_order_value"), 2))
        .withColumn("collection_rate",
            spark_round(col("collected_revenue") / col("gross_revenue") * 100, 2))
        .orderBy(col("order_date").desc())
        .withColumn("_processed_timestamp", current_timestamp()))

    (daily_revenue.write
     .format("delta")
     .mode("overwrite")
     .save(f"{GOLD_PATH}/daily_revenue"))

    print(f"Daily Revenue: {daily_revenue.count()} records")
    return daily_revenue

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Revenue por País

# COMMAND ----------

def create_country_revenue():
    """Crea tabla de revenue por país."""

    enriched_orders = spark.read.format("delta").load(f"{SILVER_PATH}/enriched_orders")

    country_revenue = (enriched_orders
        .filter(col("order_status") == "completed")
        .groupBy("customer_country")
        .agg(
            count("order_id").alias("total_orders"),
            count("customer_id").alias("total_customers"),
            spark_sum("total_amount").alias("total_revenue"),
            avg("total_amount").alias("avg_order_value"),
            spark_max("total_amount").alias("max_order_value"),
            spark_min("total_amount").alias("min_order_value")
        )
        .withColumn("total_revenue", spark_round(col("total_revenue"), 2))
        .withColumn("avg_order_value", spark_round(col("avg_order_value"), 2))
        .withColumn("_processed_timestamp", current_timestamp()))

    # Ranking de países
    window = Window.orderBy(col("total_revenue").desc())
    country_revenue_ranked = (country_revenue
        .withColumn("revenue_rank", dense_rank().over(window))
        .withColumn("revenue_percentile", spark_round(percent_rank().over(window) * 100, 2)))

    (country_revenue_ranked.write
     .format("delta")
     .mode("overwrite")
     .save(f"{GOLD_PATH}/country_revenue"))

    print(f"Country Revenue: {country_revenue_ranked.count()} records")
    return country_revenue_ranked

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Top Productos

# COMMAND ----------

def create_top_products():
    """Crea tabla de productos más vendidos."""

    enriched_orders = spark.read.format("delta").load(f"{SILVER_PATH}/enriched_orders")

    product_stats = (enriched_orders
        .filter(col("order_status") == "completed")
        .groupBy("product_id")
        .agg(
            count("order_id").alias("times_ordered"),
            spark_sum("quantity").alias("total_units_sold"),
            spark_sum("total_amount").alias("total_revenue"),
            avg("unit_price").alias("avg_unit_price"),
            count("customer_id").alias("unique_buyers")
        )
        .withColumn("total_revenue", spark_round(col("total_revenue"), 2))
        .withColumn("avg_unit_price", spark_round(col("avg_unit_price"), 2)))

    # Ranking
    window_revenue = Window.orderBy(col("total_revenue").desc())
    window_units = Window.orderBy(col("total_units_sold").desc())

    top_products = (product_stats
        .withColumn("revenue_rank", dense_rank().over(window_revenue))
        .withColumn("units_rank", dense_rank().over(window_units))
        .withColumn("_processed_timestamp", current_timestamp()))

    (top_products.write
     .format("delta")
     .mode("overwrite")
     .save(f"{GOLD_PATH}/top_products"))

    print(f"Top Products: {top_products.count()} records")
    return top_products

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Métricas de Clientes

# COMMAND ----------

def create_customer_metrics():
    """Crea tabla de métricas por cliente (Customer 360)."""

    enriched_orders = spark.read.format("delta").load(f"{SILVER_PATH}/enriched_orders")

    customer_metrics = (enriched_orders
        .groupBy("customer_id", "customer_name", "customer_email", "customer_country")
        .agg(
            count("order_id").alias("total_orders"),
            spark_sum(when(col("order_status") == "completed", 1).otherwise(0)).alias("completed_orders"),
            spark_sum(when(col("order_status") == "cancelled", 1).otherwise(0)).alias("cancelled_orders"),
            spark_sum(when(col("order_status") == "pending", 1).otherwise(0)).alias("pending_orders"),
            spark_sum("total_amount").alias("total_spent"),
            avg("total_amount").alias("avg_order_value"),
            spark_max("order_date").alias("last_order_date"),
            spark_min("order_date").alias("first_order_date")
        )
        .withColumn("total_spent", spark_round(col("total_spent"), 2))
        .withColumn("avg_order_value", spark_round(col("avg_order_value"), 2))
        .withColumn("completion_rate",
            spark_round(col("completed_orders") / col("total_orders") * 100, 2)))

    # Segmentación de clientes
    customer_segmented = customer_metrics.withColumn(
        "customer_segment",
        when(col("total_spent") >= 500, "Premium")
        .when(col("total_spent") >= 200, "Regular")
        .when(col("total_spent") >= 50, "Occasional")
        .otherwise("New")
    ).withColumn("_processed_timestamp", current_timestamp())

    (customer_segmented.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("customer_segment")
     .save(f"{GOLD_PATH}/customer_metrics"))

    print(f"Customer Metrics: {customer_segmented.count()} records")
    return customer_segmented

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Métricas de Pagos

# COMMAND ----------

def create_payment_metrics():
    """Crea tabla de métricas de métodos de pago."""

    enriched_orders = spark.read.format("delta").load(f"{SILVER_PATH}/enriched_orders")

    payment_metrics = (enriched_orders
        .filter(col("payment_status") == "completed")
        .groupBy("payment_method", "payment_category")
        .agg(
            count("payment_id").alias("total_transactions"),
            spark_sum("payment_amount").alias("total_processed"),
            avg("payment_amount").alias("avg_transaction"),
            spark_max("payment_amount").alias("max_transaction"),
            spark_min("payment_amount").alias("min_transaction")
        )
        .withColumn("total_processed", spark_round(col("total_processed"), 2))
        .withColumn("avg_transaction", spark_round(col("avg_transaction"), 2))
        .withColumn("_processed_timestamp", current_timestamp()))

    (payment_metrics.write
     .format("delta")
     .mode("overwrite")
     .save(f"{GOLD_PATH}/payment_metrics"))

    print(f"Payment Metrics: {payment_metrics.count()} records")
    return payment_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecución

# COMMAND ----------

daily_revenue = create_daily_revenue()
country_revenue = create_country_revenue()
top_products = create_top_products()
customer_metrics = create_customer_metrics()
payment_metrics = create_payment_metrics()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualización de Resultados

# COMMAND ----------

print("=== DAILY REVENUE ===")
daily_revenue.show(10, truncate=False)

print("\n=== COUNTRY REVENUE ===")
country_revenue.show(10, truncate=False)

print("\n=== TOP PRODUCTS ===")
top_products.orderBy("revenue_rank").show(10, truncate=False)

print("\n=== CUSTOMER SEGMENTS ===")
customer_metrics.groupBy("customer_segment").count().show()

print("\n=== PAYMENT METHODS ===")
payment_metrics.show(truncate=False)
