# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Model Registry
# MAGIC
# MAGIC > **Note:** All data in this project is **fictional** and created for demonstration purposes only.
# MAGIC
# MAGIC In this notebook, I demonstrate the MLflow Model Registry lifecycle:
# MAGIC - Registering models
# MAGIC - Version management
# MAGIC - Stage transitions (Staging → Production → Archived)
# MAGIC - Model deployment preparation
# MAGIC
# MAGIC The Model Registry is essential for production ML because it provides versioning,
# MAGIC approval workflows, and clear deployment history.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

EXPERIMENT_NAME = "/Users/demo/churn_prediction_experiment"
MODEL_NAME = "churn_prediction_model"
RANDOM_STATE = 42

# Initialize client
client = MlflowClient()
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Sample Data
# MAGIC
# MAGIC **All data is fictional and for demonstration purposes only.**

# COMMAND ----------

def create_demo_data(n_samples=500):
    """Create fictional demo data."""
    np.random.seed(RANDOM_STATE)

    X = pd.DataFrame({
        'recency_days': np.random.exponential(15, n_samples),
        'frequency': np.random.poisson(3, n_samples) + 1,
        'monetary_value': np.random.lognormal(4, 1, n_samples),
        'cancellation_rate': np.random.beta(2, 10, n_samples) * 100,
        'purchase_velocity': np.random.lognormal(-1, 0.8, n_samples)
    })

    churn_prob = (
        0.3 * (X['recency_days'] / X['recency_days'].max()) +
        0.3 * (X['cancellation_rate'] / 100) +
        0.4 * np.random.random(n_samples)
    )
    y = (churn_prob > 0.45).astype(int)

    return X, y

X, y = create_demo_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Data prepared: {len(X_train)} training, {len(X_test)} test samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train and Register Model
# MAGIC
# MAGIC I train a model and register it directly to the Model Registry.

# COMMAND ----------

def train_and_register_model(X_train, X_test, y_train, y_test, model_name: str, version_description: str):
    """
    Train a model and register it in MLflow Model Registry.
    """

    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metrics({
            "roc_auc": auc,
            "accuracy": (y_pred == y_test).mean()
        })

        # Log model parameters
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 8,
            "data_note": "fictional_demo_data"
        })

        # Create model signature
        signature = infer_signature(X_train, y_pred)

        # Log and register model
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=model_name
        )

        run_id = run.info.run_id

    # Add version description
    latest_version = get_latest_model_version(model_name)
    client.update_model_version(
        name=model_name,
        version=latest_version,
        description=version_description
    )

    print(f"\nModel registered: {model_name}")
    print(f"Version: {latest_version}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Run ID: {run_id}")

    return run_id, latest_version, auc


def get_latest_model_version(model_name: str) -> str:
    """Get the latest version number of a registered model."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max([int(v.version) for v in versions])
        return "1"
    except Exception:
        return "1"

# COMMAND ----------

run_id, version, auc = train_and_register_model(
    X_train, X_test, y_train, y_test,
    model_name=MODEL_NAME,
    version_description="Initial model version - Random Forest with default parameters. Trained on fictional demo data."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Versioning
# MAGIC
# MAGIC Each time I register a model with the same name, MLflow creates a new version.
# MAGIC I can train different configurations and compare them.

# COMMAND ----------

def train_improved_model(X_train, X_test, y_train, y_test, model_name: str):
    """
    Train an improved model and register as new version.
    """

    with mlflow.start_run(run_name="improved_model") as run:
        # Train with different hyperparameters
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=3,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metrics({"roc_auc": auc})
        mlflow.log_params({
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_split": 3,
            "improvement": "balanced_class_weight"
        })

        signature = infer_signature(X_train, y_pred)

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=model_name
        )

    version = get_latest_model_version(model_name)
    client.update_model_version(
        name=model_name,
        version=version,
        description="Improved model with balanced class weights and deeper trees. Fictional demo data."
    )

    print(f"\nNew version registered: {version}")
    print(f"ROC-AUC: {auc:.4f}")

    return version, auc

# COMMAND ----------

new_version, new_auc = train_improved_model(X_train, X_test, y_train, y_test, MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Stage Transitions
# MAGIC
# MAGIC Models in the registry go through stages:
# MAGIC - **None**: Just registered
# MAGIC - **Staging**: Being tested
# MAGIC - **Production**: Live deployment
# MAGIC - **Archived**: Deprecated
# MAGIC
# MAGIC I manage these transitions programmatically.

# COMMAND ----------

def transition_model_stage(model_name: str, version: str, stage: str, archive_existing: bool = False):
    """
    Transition a model version to a new stage.

    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage (Staging, Production, Archived)
        archive_existing: If True, archive existing models in target stage
    """

    # Valid stages
    valid_stages = ["Staging", "Production", "Archived", "None"]
    if stage not in valid_stages:
        raise ValueError(f"Stage must be one of: {valid_stages}")

    # Archive existing models in production if requested
    if archive_existing and stage == "Production":
        existing_prod = client.get_latest_versions(model_name, stages=["Production"])
        for mv in existing_prod:
            print(f"Archiving existing production model v{mv.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived"
            )

    # Transition to new stage
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage
    )

    print(f"\nModel {model_name} v{version} transitioned to: {stage}")

# COMMAND ----------

# Promote version 1 to Staging
transition_model_stage(MODEL_NAME, "1", "Staging")

# COMMAND ----------

# After validation, promote to Production
transition_model_stage(MODEL_NAME, "1", "Production", archive_existing=True)

# COMMAND ----------

# Promote improved model (version 2) to Staging
transition_model_stage(MODEL_NAME, new_version, "Staging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. List All Model Versions
# MAGIC
# MAGIC I can query all versions of a model and their stages.

# COMMAND ----------

def list_model_versions(model_name: str):
    """List all versions of a registered model with their stages."""

    print(f"\nModel: {model_name}")
    print("=" * 60)

    try:
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            print("No versions found")
            return

        data = []
        for v in versions:
            data.append({
                'Version': v.version,
                'Stage': v.current_stage,
                'Status': v.status,
                'Run ID': v.run_id[:8] + "...",
                'Created': v.creation_timestamp
            })

        df = pd.DataFrame(data)
        print(df.to_string(index=False))

        return df

    except Exception as e:
        print(f"Error: {e}")

# COMMAND ----------

versions_df = list_model_versions(MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load Model by Stage
# MAGIC
# MAGIC In production, I load models by stage rather than specific version.
# MAGIC This allows seamless updates without changing code.

# COMMAND ----------

def load_model_by_stage(model_name: str, stage: str = "Production"):
    """
    Load a model from the registry by its stage.
    This is the recommended way to load models in production.
    """

    model_uri = f"models:/{model_name}/{stage}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model: {model_name} ({stage})")
        return model
    except Exception as e:
        print(f"Could not load model: {e}")
        return None

# COMMAND ----------

# Load production model
prod_model = load_model_by_stage(MODEL_NAME, "Production")

# COMMAND ----------

# Make predictions with production model
if prod_model:
    sample_prediction = prod_model.predict_proba(X_test.head(5))[:, 1]
    print("Sample churn probabilities:")
    for i, prob in enumerate(sample_prediction):
        print(f"  Customer {i+1}: {prob:.2%} churn probability")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Comparison Before Promotion
# MAGIC
# MAGIC Before promoting a new model to Production, I compare it with the current one.

# COMMAND ----------

def compare_staging_vs_production(model_name: str, X_test, y_test):
    """
    Compare Staging model performance against Production.
    """

    print(f"\n{'='*60}")
    print("STAGING vs PRODUCTION COMPARISON")
    print(f"{'='*60}")

    results = {}

    for stage in ["Production", "Staging"]:
        try:
            model = load_model_by_stage(model_name, stage)
            if model:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                results[stage] = auc
                print(f"\n{stage}:")
                print(f"  ROC-AUC: {auc:.4f}")
        except Exception as e:
            print(f"\n{stage}: Not available ({e})")

    if len(results) == 2:
        improvement = results["Staging"] - results["Production"]
        print(f"\n{'='*60}")
        if improvement > 0:
            print(f"Staging is BETTER by {improvement:.4f} ({improvement/results['Production']*100:.1f}%)")
            print("RECOMMENDATION: Promote Staging to Production")
        else:
            print(f"Production is BETTER by {-improvement:.4f}")
            print("RECOMMENDATION: Keep current Production model")
        print(f"{'='*60}")

    return results

# COMMAND ----------

comparison = compare_staging_vs_production(MODEL_NAME, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Promote Staging to Production
# MAGIC
# MAGIC If Staging performs better, I promote it to Production.

# COMMAND ----------

def promote_staging_to_production(model_name: str):
    """
    Promote the Staging model to Production, archiving the old Production model.
    """

    # Get staging version
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model in Staging")
        return

    staging_version = staging_versions[0].version

    # Archive current production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        print(f"Archiving Production v{pv.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=pv.version,
            stage="Archived"
        )

    # Promote staging to production
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version,
        stage="Production"
    )

    print(f"\nPromoted v{staging_version} to Production")

# COMMAND ----------

# Uncomment to promote (if staging is better)
# promote_staging_to_production(MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Add Model Tags and Descriptions
# MAGIC
# MAGIC Tags help with model organization and searchability.

# COMMAND ----------

def add_model_metadata(model_name: str, version: str, tags: dict, description: str = None):
    """
    Add tags and description to a model version.
    """

    # Add tags
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)
        print(f"Added tag: {key}={value}")

    # Update description
    if description:
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print(f"Updated description")

# COMMAND ----------

# Add metadata to production model
add_model_metadata(
    MODEL_NAME,
    "1",
    tags={
        "data_version": "demo_v1",
        "training_date": datetime.now().strftime("%Y-%m-%d"),
        "author": "portfolio_demo",
        "validated": "true",
        "is_fictional_data": "true"
    },
    description="Production churn model. Trained on fictional demo data for portfolio demonstration."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC > **Reminder:** All models and data in this demonstration are **fictional**.
# MAGIC
# MAGIC In this notebook, I demonstrated the MLflow Model Registry:
# MAGIC
# MAGIC | Operation | Description |
# MAGIC |-----------|-------------|
# MAGIC | **Register** | Log model with `registered_model_name` |
# MAGIC | **Version** | Each registration creates a new version |
# MAGIC | **Stage Transition** | None → Staging → Production → Archived |
# MAGIC | **Load by Stage** | `models:/{name}/{stage}` for production code |
# MAGIC | **Comparison** | Compare Staging vs Production before promoting |
# MAGIC | **Metadata** | Add tags and descriptions for organization |
# MAGIC
# MAGIC **Model Registry Best Practices:**
# MAGIC 1. Always validate in Staging before Production
# MAGIC 2. Use meaningful version descriptions
# MAGIC 3. Tag models with data versions and training dates
# MAGIC 4. Archive (don't delete) old production models
# MAGIC 5. Load by stage in production code for seamless updates
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Deploy model for batch inference (Notebook 05)
