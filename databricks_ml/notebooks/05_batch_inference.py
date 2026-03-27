# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference - Production Scoring
# MAGIC
# MAGIC > **Note:** All data in this project is **fictional** and created for demonstration purposes only.
# MAGIC
# MAGIC In this notebook, I demonstrate batch inference at scale:
# MAGIC - Loading production models from the registry
# MAGIC - Scoring large datasets efficiently
# MAGIC - Saving predictions to Delta Lake
# MAGIC - Monitoring prediction distributions
# MAGIC
# MAGIC This is how I would deploy churn predictions in a production environment.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, current_timestamp, pandas_udf, struct,
    count, avg, min as spark_min, max as spark_max,
    percentile_approx, when, round as spark_round
)
from pyspark.sql.types import DoubleType, StructType, StructField, StringType

import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

MODEL_NAME = "churn_prediction_model"
GOLD_PATH = "/mnt/gold/"
PREDICTIONS_PATH = f"{GOLD_PATH}/ml_predictions/churn_scores"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Production Model
# MAGIC
# MAGIC I load the model from the Production stage of the registry.
# MAGIC This ensures I always use the validated, approved model.

# COMMAND ----------

def load_production_model(model_name: str):
    """
    Load the production model from MLflow Registry.

    In production, always load by stage (not version) so that
    model updates don't require code changes.
    """

    model_uri = f"models:/{model_name}/Production"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded production model: {model_name}")

        # Get model metadata
        client = MlflowClient()
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if prod_versions:
            v = prod_versions[0]
            print(f"  Version: {v.version}")
            print(f"  Run ID: {v.run_id[:8]}...")
            print(f"  Description: {v.description[:50] if v.description else 'N/A'}...")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a fallback model for demonstration...")
        return create_fallback_model()


def create_fallback_model():
    """Create a simple model if production model is not available."""
    from sklearn.ensemble import RandomForestClassifier

    # Train on sample data
    np.random.seed(42)
    X_sample = np.random.randn(100, 5)
    y_sample = (np.random.random(100) > 0.5).astype(int)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_sample, y_sample)

    print("Created fallback model for demonstration")
    return model

# COMMAND ----------

production_model = load_production_model(MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Customer Data for Scoring
# MAGIC
# MAGIC **All data is fictional and for demonstration purposes only.**
# MAGIC
# MAGIC In production, I would read from Feature Store or Silver tables.

# COMMAND ----------

def create_customers_to_score(n_customers: int = 1000):
    """
    Create fictional customer data for batch scoring demonstration.

    In production, this would read from Feature Store:
    ```python
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    features = fs.read_table(FEATURE_TABLE_NAME)
    ```
    """

    np.random.seed(42)

    # Generate synthetic customer features
    data = []
    for i in range(n_customers):
        customer = {
            'customer_id': f'CUST{str(i+1).zfill(5)}',
            'recency_days': max(1, int(np.random.exponential(20))),
            'frequency': max(1, np.random.poisson(3)),
            'monetary_value': round(np.random.lognormal(4, 1), 2),
            'cancellation_rate': round(np.random.beta(2, 10) * 100, 2),
            'purchase_velocity': round(np.random.lognormal(-1, 0.8), 3)
        }
        data.append(customer)

    df = spark.createDataFrame(data)

    print(f"Created {n_customers} customers for scoring (FICTIONAL DATA)")
    print(f"Schema: {df.columns}")

    return df

# COMMAND ----------

customers_df = create_customers_to_score(1000)
display(customers_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch Scoring with Pandas UDF
# MAGIC
# MAGIC For large datasets, I use Pandas UDFs which allow efficient
# MAGIC distributed scoring using the Spark engine.

# COMMAND ----------

# Feature columns expected by the model
FEATURE_COLUMNS = ['recency_days', 'frequency', 'monetary_value', 'cancellation_rate', 'purchase_velocity']

# Broadcast the model to all workers
broadcast_model = spark.sparkContext.broadcast(production_model)

# COMMAND ----------

@pandas_udf(DoubleType())
def predict_churn_probability(recency: pd.Series, frequency: pd.Series,
                               monetary: pd.Series, cancellation: pd.Series,
                               velocity: pd.Series) -> pd.Series:
    """
    Pandas UDF for distributed churn prediction.

    This function runs on each Spark partition, enabling
    efficient scoring of millions of records.
    """
    # Combine features into DataFrame
    features_df = pd.DataFrame({
        'recency_days': recency,
        'frequency': frequency,
        'monetary_value': monetary,
        'cancellation_rate': cancellation,
        'purchase_velocity': velocity
    })

    # Get model from broadcast variable
    model = broadcast_model.value

    # Predict churn probability
    probabilities = model.predict_proba(features_df)[:, 1]

    return pd.Series(probabilities)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Execute Batch Scoring
# MAGIC
# MAGIC I score all customers and add metadata for auditing.

# COMMAND ----------

def score_customers(df, model_name: str):
    """
    Score all customers with churn probability.
    """

    # Add prediction column using Pandas UDF
    scored_df = df.withColumn(
        "churn_probability",
        predict_churn_probability(
            col("recency_days"),
            col("frequency"),
            col("monetary_value"),
            col("cancellation_rate"),
            col("purchase_velocity")
        )
    )

    # Add metadata columns for auditing
    scored_df = scored_df \
        .withColumn("churn_probability", spark_round(col("churn_probability"), 4)) \
        .withColumn("prediction_timestamp", current_timestamp()) \
        .withColumn("model_name", lit(model_name)) \
        .withColumn("model_stage", lit("Production")) \
        .withColumn("is_fictional_data", lit(True))

    # Add risk category
    scored_df = scored_df.withColumn(
        "churn_risk",
        when(col("churn_probability") >= 0.7, "HIGH")
        .when(col("churn_probability") >= 0.4, "MEDIUM")
        .otherwise("LOW")
    )

    return scored_df

# COMMAND ----------

# Execute scoring
scored_customers = score_customers(customers_df, MODEL_NAME)

print(f"Scored {scored_customers.count()} customers")
display(scored_customers.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Save Predictions to Delta Lake
# MAGIC
# MAGIC I save predictions to the Gold layer for consumption by BI tools and downstream systems.

# COMMAND ----------

def save_predictions(df, output_path: str, mode: str = "overwrite"):
    """
    Save predictions to Delta Lake.

    Args:
        df: Scored DataFrame
        output_path: Delta Lake path
        mode: 'overwrite' or 'append'
    """

    (df.write
     .format("delta")
     .mode(mode)
     .option("overwriteSchema", "true")
     .partitionBy("churn_risk")
     .save(output_path))

    print(f"Predictions saved to: {output_path}")
    print(f"Mode: {mode}")
    print(f"Partitioned by: churn_risk")

# COMMAND ----------

# Save predictions
save_predictions(scored_customers, PREDICTIONS_PATH)

# COMMAND ----------

# Verify saved data
saved_predictions = spark.read.format("delta").load(PREDICTIONS_PATH)
print(f"Verified: {saved_predictions.count()} records saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Prediction Distribution Analysis
# MAGIC
# MAGIC I analyze the distribution of predictions to monitor model behavior
# MAGIC and detect potential drift.

# COMMAND ----------

def analyze_prediction_distribution(df):
    """
    Analyze and visualize prediction distribution.
    """

    print("\n" + "="*60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*60)

    # Basic statistics
    stats = df.select(
        count("*").alias("total_customers"),
        avg("churn_probability").alias("avg_probability"),
        spark_min("churn_probability").alias("min_probability"),
        spark_max("churn_probability").alias("max_probability"),
        percentile_approx("churn_probability", 0.5).alias("median_probability")
    ).collect()[0]

    print(f"\nTotal Customers: {stats['total_customers']}")
    print(f"Average Churn Probability: {stats['avg_probability']:.3f}")
    print(f"Median Churn Probability: {stats['median_probability']:.3f}")
    print(f"Min: {stats['min_probability']:.3f}, Max: {stats['max_probability']:.3f}")

    # Risk distribution
    print("\n" + "-"*40)
    print("RISK DISTRIBUTION:")
    print("-"*40)

    risk_dist = df.groupBy("churn_risk").count().orderBy("churn_risk")
    display(risk_dist)

    return stats

# COMMAND ----------

distribution_stats = analyze_prediction_distribution(scored_customers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. High-Risk Customer Report
# MAGIC
# MAGIC I create a report of customers most likely to churn for the retention team.

# COMMAND ----------

def create_high_risk_report(df, top_n: int = 100):
    """
    Create a report of highest-risk customers.
    """

    print(f"\n{'='*60}")
    print(f"TOP {top_n} HIGH-RISK CUSTOMERS")
    print(f"{'='*60}")
    print("(All data is FICTIONAL - for demonstration only)")

    high_risk = (df
        .filter(col("churn_risk") == "HIGH")
        .orderBy(col("churn_probability").desc())
        .select(
            "customer_id",
            "churn_probability",
            "recency_days",
            "frequency",
            "monetary_value",
            "cancellation_rate"
        )
        .limit(top_n))

    display(high_risk)

    # Save report
    report_path = f"{GOLD_PATH}/ml_reports/high_risk_customers"
    (high_risk.write
     .format("delta")
     .mode("overwrite")
     .save(report_path))

    print(f"\nReport saved to: {report_path}")

    return high_risk

# COMMAND ----------

high_risk_report = create_high_risk_report(scored_customers, 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Monitoring Dashboard Metrics
# MAGIC
# MAGIC I calculate key metrics for monitoring dashboards.

# COMMAND ----------

def calculate_monitoring_metrics(df):
    """
    Calculate metrics for production monitoring.
    """

    metrics = {
        "scoring_timestamp": datetime.now().isoformat(),
        "total_scored": df.count(),
        "model_name": MODEL_NAME,
        "is_fictional_data": True
    }

    # Risk counts
    risk_counts = df.groupBy("churn_risk").count().collect()
    for row in risk_counts:
        metrics[f"count_{row['churn_risk'].lower()}"] = row['count']

    # Probability statistics
    prob_stats = df.agg(
        avg("churn_probability").alias("avg_prob"),
        percentile_approx("churn_probability", [0.25, 0.5, 0.75]).alias("percentiles")
    ).collect()[0]

    metrics["avg_churn_probability"] = round(prob_stats['avg_prob'], 4)
    metrics["p25_probability"] = round(prob_stats['percentiles'][0], 4)
    metrics["p50_probability"] = round(prob_stats['percentiles'][1], 4)
    metrics["p75_probability"] = round(prob_stats['percentiles'][2], 4)

    # Calculate high-risk rate
    total = metrics["total_scored"]
    high_risk = metrics.get("count_high", 0)
    metrics["high_risk_rate"] = round(high_risk / total * 100, 2)

    print("\n" + "="*60)
    print("MONITORING METRICS")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return metrics

# COMMAND ----------

monitoring_metrics = calculate_monitoring_metrics(scored_customers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Schedule Information
# MAGIC
# MAGIC In production, I would schedule this notebook to run daily using Databricks Workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Workflow Configuration (Example)
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "name": "churn_batch_scoring",
# MAGIC   "schedule": {
# MAGIC     "quartz_cron_expression": "0 0 6 * * ?",
# MAGIC     "timezone_id": "America/Buenos_Aires"
# MAGIC   },
# MAGIC   "tasks": [
# MAGIC     {
# MAGIC       "task_key": "score_customers",
# MAGIC       "notebook_task": {
# MAGIC         "notebook_path": "/Repos/demo/databricks_ml/notebooks/05_batch_inference"
# MAGIC       },
# MAGIC       "new_cluster": {
# MAGIC         "num_workers": 2,
# MAGIC         "spark_version": "13.3.x-scala2.12",
# MAGIC         "node_type_id": "i3.xlarge"
# MAGIC       }
# MAGIC     }
# MAGIC   ],
# MAGIC   "email_notifications": {
# MAGIC     "on_failure": ["alerts@company.com"]
# MAGIC   }
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC > **Reminder:** All data, predictions, and metrics shown are **fictional**
# MAGIC > and for demonstration purposes only.
# MAGIC
# MAGIC In this notebook, I implemented production batch inference:
# MAGIC
# MAGIC | Component | Description |
# MAGIC |-----------|-------------|
# MAGIC | **Model Loading** | Load from Registry by stage (Production) |
# MAGIC | **Pandas UDF** | Distributed scoring for large datasets |
# MAGIC | **Delta Lake** | Save predictions with partitioning |
# MAGIC | **Risk Categories** | Classify customers (HIGH/MEDIUM/LOW) |
# MAGIC | **Monitoring** | Calculate metrics for dashboards |
# MAGIC | **Reports** | Generate high-risk customer lists |
# MAGIC
# MAGIC **Production Best Practices:**
# MAGIC 1. Load models by stage, not version
# MAGIC 2. Use Pandas UDFs for scalable scoring
# MAGIC 3. Add audit columns (timestamp, model version)
# MAGIC 4. Partition predictions by risk category
# MAGIC 5. Monitor prediction distributions for drift
# MAGIC 6. Schedule with Databricks Workflows
# MAGIC
# MAGIC **Pipeline Complete!**
# MAGIC
# MAGIC This concludes the MLflow demonstration:
# MAGIC - **01**: Feature Engineering with Feature Store
# MAGIC - **02**: Model Training with comparison
# MAGIC - **03**: MLflow Experiment Tracking
# MAGIC - **04**: Model Registry lifecycle
# MAGIC - **05**: Batch Inference (this notebook)
