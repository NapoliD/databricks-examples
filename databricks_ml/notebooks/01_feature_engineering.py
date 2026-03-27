# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering for Churn Prediction
# MAGIC
# MAGIC This notebook creates features for customer churn prediction using Databricks Feature Store.
# MAGIC I designed this feature pipeline to capture three key aspects of customer behavior:
# MAGIC - **RFM Analysis**: Recency, Frequency, and Monetary value
# MAGIC - **Behavioral Patterns**: Cancellation rates, basket composition
# MAGIC - **Temporal Dynamics**: Purchase velocity and lifecycle stage

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, max as spark_max, min as spark_min,
    datediff, current_date, lit, when, countDistinct,
    round as spark_round, coalesce
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, DateType, TimestampType
)
from datetime import datetime, timedelta
import mlflow
from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths - adjust according to your environment
SILVER_PATH = "/mnt/silver/"
GOLD_PATH = "/mnt/gold/"
FEATURE_STORE_DB = "churn_features"
FEATURE_TABLE_NAME = f"{FEATURE_STORE_DB}.customer_churn_features"

# Churn definition: customers who haven't purchased in last N days
CHURN_THRESHOLD_DAYS = 30

# Reference date for calculating recency (in production, use current_date())
REFERENCE_DATE = "2024-01-28"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Silver Data
# MAGIC
# MAGIC I'm using the enriched orders from the Silver layer which already contains
# MAGIC customer information joined with orders and payments.

# COMMAND ----------

def load_source_data():
    """Load enriched orders from Silver layer."""

    # In a real scenario, read from Delta tables
    # For demonstration, I'll create sample data that matches our schema

    orders_schema = StructType([
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("product_id", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("order_date", DateType(), True),
        StructField("order_status", StringType(), True),
        StructField("customer_name", StringType(), True),
        StructField("customer_email", StringType(), True),
        StructField("customer_country", StringType(), True),
        StructField("registration_date", DateType(), True),
        StructField("payment_status", StringType(), True),
        StructField("payment_method", StringType(), True)
    ])

    try:
        # Try to read from Silver layer
        enriched_orders = spark.read.format("delta").load(f"{SILVER_PATH}/enriched_orders")
        print(f"Loaded {enriched_orders.count()} records from Silver layer")
    except Exception as e:
        print(f"Silver layer not available, creating sample data: {e}")
        # Create sample data for demonstration
        enriched_orders = create_sample_data()

    return enriched_orders


def create_sample_data():
    """Create sample data for demonstration purposes."""

    from pyspark.sql.functions import to_date

    # Sample enriched orders data
    data = [
        ("ORD001", "CUST001", "PROD001", 2, 29.99, 59.98, "2024-01-15", "completed", "Juan Garcia", "juan@email.com", "Argentina", "2023-06-15", "completed", "credit_card"),
        ("ORD002", "CUST002", "PROD002", 1, 149.99, 149.99, "2024-01-15", "completed", "Maria Lopez", "maria@email.com", "Mexico", "2023-07-20", "completed", "debit_card"),
        ("ORD003", "CUST001", "PROD003", 3, 9.99, 29.97, "2024-01-16", "completed", "Juan Garcia", "juan@email.com", "Argentina", "2023-06-15", "completed", "credit_card"),
        ("ORD004", "CUST003", "PROD001", 1, 29.99, 29.99, "2024-01-16", "pending", "Carlos Rodriguez", "carlos@email.com", "Colombia", "2023-08-10", "pending", "bank_transfer"),
        ("ORD005", "CUST004", "PROD004", 2, 79.99, 159.98, "2024-01-17", "completed", "Ana Martinez", "ana@email.com", "Chile", "2023-09-05", "completed", "credit_card"),
        ("ORD006", "CUST002", "PROD005", 1, 199.99, 199.99, "2024-01-17", "completed", "Maria Lopez", "maria@email.com", "Mexico", "2023-07-20", "completed", "paypal"),
        ("ORD007", "CUST005", "PROD002", 2, 149.99, 299.98, "2024-01-18", "cancelled", "Pedro Sanchez", "pedro@email.com", "Peru", "2023-09-15", "cancelled", "credit_card"),
        ("ORD008", "CUST003", "PROD003", 5, 9.99, 49.95, "2024-01-18", "completed", "Carlos Rodriguez", "carlos@email.com", "Colombia", "2023-08-10", "completed", "debit_card"),
        ("ORD009", "CUST006", "PROD006", 1, 299.99, 299.99, "2024-01-19", "completed", "Laura Fernandez", "laura@email.com", "Argentina", "2023-10-01", "completed", "credit_card"),
        ("ORD010", "CUST001", "PROD004", 1, 79.99, 79.99, "2024-01-19", "pending", "Juan Garcia", "juan@email.com", "Argentina", "2023-06-15", "pending", "bank_transfer"),
        ("ORD011", "CUST007", "PROD001", 4, 29.99, 119.96, "2024-01-20", "completed", "Diego Gonzalez", "diego@email.com", "Uruguay", "2023-10-20", "completed", "debit_card"),
        ("ORD012", "CUST008", "PROD007", 1, 449.99, 449.99, "2024-01-20", "completed", "Sofia Ramirez", "sofia@email.com", "Ecuador", "2023-11-05", "completed", "credit_card"),
        ("ORD013", "CUST004", "PROD002", 1, 149.99, 149.99, "2024-01-21", "completed", "Ana Martinez", "ana@email.com", "Chile", "2023-09-05", "completed", "paypal"),
        ("ORD014", "CUST009", "PROD003", 2, 9.99, 19.98, "2024-01-21", "pending", "Miguel Torres", "miguel@email.com", "Venezuela", "2023-11-15", "pending", "bank_transfer"),
        ("ORD015", "CUST010", "PROD008", 1, 599.99, 599.99, "2024-01-22", "completed", "Carmen Ruiz", "carmen@email.com", "Bolivia", "2023-12-01", "completed", "credit_card"),
    ]

    df = spark.createDataFrame(data, [
        "order_id", "customer_id", "product_id", "quantity", "unit_price",
        "total_amount", "order_date", "order_status", "customer_name",
        "customer_email", "customer_country", "registration_date",
        "payment_status", "payment_method"
    ])

    # Convert date strings to date type
    df = df.withColumn("order_date", to_date(col("order_date"))) \
           .withColumn("registration_date", to_date(col("registration_date")))

    return df

# COMMAND ----------

# Load data
enriched_orders = load_source_data()
display(enriched_orders.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. RFM Features
# MAGIC
# MAGIC RFM (Recency, Frequency, Monetary) analysis is a classic approach to customer segmentation.
# MAGIC These features are highly predictive of churn because they capture the core dimensions
# MAGIC of customer engagement.

# COMMAND ----------

def calculate_rfm_features(df, reference_date: str):
    """
    Calculate RFM features for each customer.

    Args:
        df: DataFrame with order data
        reference_date: Date to calculate recency from

    Returns:
        DataFrame with RFM features per customer
    """

    ref_date = lit(reference_date).cast("date")

    # Filter only completed orders for RFM calculation
    completed_orders = df.filter(col("order_status") == "completed")

    rfm_features = (completed_orders
        .groupBy("customer_id")
        .agg(
            # Recency: days since last purchase
            datediff(ref_date, spark_max("order_date")).alias("recency_days"),

            # Frequency: number of purchases
            count("order_id").alias("frequency"),

            # Monetary: total amount spent
            spark_round(spark_sum("total_amount"), 2).alias("monetary_value"),

            # Additional RFM metrics
            spark_round(avg("total_amount"), 2).alias("avg_order_value"),
            spark_max("total_amount").alias("max_order_value"),
            spark_min("total_amount").alias("min_order_value")
        ))

    return rfm_features

# COMMAND ----------

rfm_features = calculate_rfm_features(enriched_orders, REFERENCE_DATE)
print("RFM Features created:")
display(rfm_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Behavioral Features
# MAGIC
# MAGIC Beyond RFM, I capture behavioral patterns that indicate customer satisfaction
# MAGIC and engagement. High cancellation rates or low product diversity often
# MAGIC precede churn.

# COMMAND ----------

def calculate_behavioral_features(df):
    """
    Calculate behavioral features for each customer.

    These features capture patterns that may indicate dissatisfaction:
    - High cancellation rate
    - Low basket diversity
    - Payment method preferences
    """

    behavioral = (df
        .groupBy("customer_id")
        .agg(
            # Cancellation rate
            spark_round(
                spark_sum(when(col("order_status") == "cancelled", 1).otherwise(0)) /
                count("order_id") * 100, 2
            ).alias("cancellation_rate"),

            # Pending orders ratio
            spark_round(
                spark_sum(when(col("order_status") == "pending", 1).otherwise(0)) /
                count("order_id") * 100, 2
            ).alias("pending_rate"),

            # Product diversity
            countDistinct("product_id").alias("unique_products"),

            # Average basket size
            spark_round(avg("quantity"), 2).alias("avg_basket_size"),

            # Total items purchased
            spark_sum("quantity").alias("total_items"),

            # Payment method diversity (more methods = more engaged)
            countDistinct("payment_method").alias("payment_methods_used"),

            # Preferred payment method
            spark_max("payment_method").alias("preferred_payment_method")
        ))

    return behavioral

# COMMAND ----------

behavioral_features = calculate_behavioral_features(enriched_orders)
print("Behavioral Features created:")
display(behavioral_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Temporal Features
# MAGIC
# MAGIC Time-based features capture the customer lifecycle stage and purchase velocity.
# MAGIC New customers behave differently than long-term customers.

# COMMAND ----------

def calculate_temporal_features(df, reference_date: str):
    """
    Calculate temporal features for each customer.

    These features capture customer lifecycle:
    - How long they've been a customer
    - Their purchase velocity over time
    - Time between first and last order
    """

    ref_date = lit(reference_date).cast("date")

    temporal = (df
        .groupBy("customer_id", "registration_date")
        .agg(
            # Customer tenure
            datediff(ref_date, col("registration_date")).alias("days_since_registration"),

            # First order timing
            datediff(ref_date, spark_min("order_date")).alias("days_since_first_order"),

            # Last order timing
            datediff(ref_date, spark_max("order_date")).alias("days_since_last_order"),

            # Active period
            datediff(spark_max("order_date"), spark_min("order_date")).alias("active_period_days"),

            # Order count for velocity calculation
            count("order_id").alias("order_count")
        )
        .withColumn(
            # Purchase velocity: orders per month of tenure
            "purchase_velocity",
            spark_round(
                col("order_count") / (col("days_since_registration") / 30.0 + 1),
                3
            )
        )
        .withColumn(
            # Time to first order after registration
            "days_to_first_order",
            col("days_since_registration") - col("days_since_first_order")
        )
        .drop("registration_date", "order_count"))

    return temporal

# COMMAND ----------

temporal_features = calculate_temporal_features(enriched_orders, REFERENCE_DATE)
print("Temporal Features created:")
display(temporal_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Churn Label
# MAGIC
# MAGIC I define churn as customers who haven't made a purchase in the last N days.
# MAGIC This is a business decision that should be validated with stakeholders.

# COMMAND ----------

def create_churn_label(df, reference_date: str, threshold_days: int):
    """
    Create binary churn label based on purchase recency.

    Args:
        df: DataFrame with customer data
        reference_date: Date to calculate from
        threshold_days: Days without purchase to consider as churned

    Returns:
        DataFrame with customer_id and is_churned label
    """

    ref_date = lit(reference_date).cast("date")

    # Get last purchase date per customer
    last_purchase = (df
        .filter(col("order_status") == "completed")
        .groupBy("customer_id")
        .agg(spark_max("order_date").alias("last_purchase_date"))
        .withColumn(
            "days_since_purchase",
            datediff(ref_date, col("last_purchase_date"))
        )
        .withColumn(
            "is_churned",
            when(col("days_since_purchase") > threshold_days, 1).otherwise(0)
        )
        .select("customer_id", "is_churned"))

    return last_purchase

# COMMAND ----------

churn_labels = create_churn_label(enriched_orders, REFERENCE_DATE, CHURN_THRESHOLD_DAYS)
print(f"Churn labels created (threshold: {CHURN_THRESHOLD_DAYS} days):")
display(churn_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Combine All Features
# MAGIC
# MAGIC Now I join all feature groups into a single feature table.

# COMMAND ----------

def combine_features(rfm, behavioral, temporal, labels):
    """
    Combine all feature groups into a single feature table.
    """

    # Get unique customers with basic info
    customer_base = enriched_orders.select(
        "customer_id",
        "customer_name",
        "customer_country"
    ).dropDuplicates(["customer_id"])

    # Join all features
    features = (customer_base
        .join(rfm, "customer_id", "left")
        .join(behavioral.drop("preferred_payment_method"), "customer_id", "left")
        .join(temporal, "customer_id", "left")
        .join(labels, "customer_id", "left"))

    # Fill nulls with appropriate defaults
    features = features.fillna({
        "recency_days": 999,
        "frequency": 0,
        "monetary_value": 0.0,
        "avg_order_value": 0.0,
        "cancellation_rate": 0.0,
        "pending_rate": 0.0,
        "unique_products": 0,
        "avg_basket_size": 0.0,
        "purchase_velocity": 0.0,
        "is_churned": 1  # Assume churned if no data
    })

    return features

# COMMAND ----------

customer_features = combine_features(rfm_features, behavioral_features, temporal_features, churn_labels)
print(f"Final feature table: {customer_features.count()} customers")
display(customer_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Register in Feature Store
# MAGIC
# MAGIC Using Databricks Feature Store allows me to:
# MAGIC - Version features automatically
# MAGIC - Track feature lineage
# MAGIC - Reuse features across models
# MAGIC - Serve features for online inference

# COMMAND ----------

def register_feature_table(df, table_name: str, primary_key: str, description: str):
    """
    Register features in Databricks Feature Store.
    """

    fs = FeatureStoreClient()

    # Create database if not exists
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {FEATURE_STORE_DB}")

    try:
        # Try to get existing table
        fs.get_table(table_name)
        print(f"Feature table {table_name} exists, updating...")

        # Write to existing table
        fs.write_table(
            name=table_name,
            df=df,
            mode="overwrite"
        )
    except Exception:
        print(f"Creating new feature table: {table_name}")

        # Create new feature table
        fs.create_table(
            name=table_name,
            primary_keys=[primary_key],
            df=df,
            description=description
        )

    print(f"Feature table {table_name} registered successfully")
    return fs

# COMMAND ----------

# Select only numeric features for ML (excluding customer_name, customer_country)
ml_features = customer_features.select(
    "customer_id",
    "recency_days",
    "frequency",
    "monetary_value",
    "avg_order_value",
    "max_order_value",
    "min_order_value",
    "cancellation_rate",
    "pending_rate",
    "unique_products",
    "avg_basket_size",
    "total_items",
    "payment_methods_used",
    "days_since_registration",
    "days_since_first_order",
    "days_since_last_order",
    "active_period_days",
    "purchase_velocity",
    "days_to_first_order",
    "is_churned"
)

# COMMAND ----------

# Register in Feature Store (commented for local execution)
# Uncomment when running in Databricks

# fs = register_feature_table(
#     df=ml_features,
#     table_name=FEATURE_TABLE_NAME,
#     primary_key="customer_id",
#     description="Customer features for churn prediction including RFM, behavioral, and temporal metrics"
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save to Delta Lake
# MAGIC
# MAGIC As a backup and for easy access, I also save the features to a Delta table.

# COMMAND ----------

# Save to Delta Lake
output_path = f"{GOLD_PATH}/ml_features/customer_churn_features"

(ml_features.write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .save(output_path))

print(f"Features saved to {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Feature Summary Statistics
# MAGIC
# MAGIC Understanding the distribution of features helps identify data quality issues
# MAGIC and inform feature engineering decisions.

# COMMAND ----------

# Summary statistics
print("Feature Statistics:")
display(ml_features.describe())

# COMMAND ----------

# Churn distribution
print("\nChurn Distribution:")
display(ml_features.groupBy("is_churned").count())

# COMMAND ----------

# Feature correlations preview
print("\nSample of features with churn label:")
display(ml_features.select(
    "customer_id", "recency_days", "frequency", "monetary_value",
    "cancellation_rate", "purchase_velocity", "is_churned"
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, I created a comprehensive feature engineering pipeline that:
# MAGIC
# MAGIC 1. **RFM Features**: Captured core customer value metrics
# MAGIC 2. **Behavioral Features**: Identified patterns indicating satisfaction
# MAGIC 3. **Temporal Features**: Tracked customer lifecycle and velocity
# MAGIC 4. **Churn Label**: Created target variable based on business rules
# MAGIC
# MAGIC **Features Created:**
# MAGIC - `recency_days`: Days since last purchase
# MAGIC - `frequency`: Number of completed orders
# MAGIC - `monetary_value`: Total amount spent
# MAGIC - `cancellation_rate`: Percentage of cancelled orders
# MAGIC - `purchase_velocity`: Orders per month of tenure
# MAGIC - And 14 more features...
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Train and compare multiple ML models (Notebook 02)
# MAGIC - Track experiments with MLflow (Notebook 03)
