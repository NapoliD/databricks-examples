# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Tracking
# MAGIC
# MAGIC > **Note:** All data in this project is **fictional** and created for demonstration purposes only.
# MAGIC
# MAGIC In this notebook, I demonstrate advanced MLflow tracking capabilities:
# MAGIC - Experiment organization
# MAGIC - Custom metrics and artifacts
# MAGIC - Run comparison
# MAGIC - Hyperparameter search tracking
# MAGIC
# MAGIC This is how I organize ML experiments in production to ensure reproducibility.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

EXPERIMENT_NAME = "/Users/demo/churn_prediction_experiment"
RANDOM_STATE = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MLflow Client Setup
# MAGIC
# MAGIC The MlflowClient provides programmatic access to the tracking server.
# MAGIC I use it to query runs, compare experiments, and manage models.

# COMMAND ----------

# Initialize client
client = MlflowClient()

# Set experiment
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Lifecycle Stage: {experiment.lifecycle_stage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Sample Data for Demonstrations
# MAGIC
# MAGIC **Note:** This is fictional data created for educational purposes.

# COMMAND ----------

def create_demo_data(n_samples=500):
    """Create fictional demo data for MLflow demonstrations."""

    np.random.seed(RANDOM_STATE)

    # Create synthetic features
    X = pd.DataFrame({
        'recency_days': np.random.exponential(15, n_samples),
        'frequency': np.random.poisson(3, n_samples) + 1,
        'monetary_value': np.random.lognormal(4, 1, n_samples),
        'cancellation_rate': np.random.beta(2, 10, n_samples) * 100,
        'purchase_velocity': np.random.lognormal(-1, 0.8, n_samples),
        'avg_basket_size': np.random.lognormal(0.5, 0.5, n_samples),
        'days_since_registration': np.random.randint(30, 400, n_samples)
    })

    # Create target based on features
    churn_prob = (
        0.3 * (X['recency_days'] / X['recency_days'].max()) +
        0.2 * (1 - X['frequency'] / X['frequency'].max()) +
        0.2 * (X['cancellation_rate'] / 100) +
        0.3 * np.random.random(n_samples)
    )
    y = (churn_prob > 0.45).astype(int)

    return X, y

# COMMAND ----------

X, y = create_demo_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.columns.tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Advanced Experiment Tracking
# MAGIC
# MAGIC I demonstrate how to track:
# MAGIC - Custom metrics over training iterations
# MAGIC - Confusion matrices as artifacts
# MAGIC - ROC curves as figures
# MAGIC - Model metadata and tags

# COMMAND ----------

def train_with_advanced_tracking(X_train, X_test, y_train, y_test, run_name: str):
    """
    Train model with comprehensive MLflow tracking.
    Demonstrates best practices for experiment tracking.
    """

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ===== 1. LOG TAGS =====
        # Tags help organize and filter runs
        mlflow.set_tags({
            "project": "churn_prediction",
            "dataset": "demo_fictional_data",
            "author": "portfolio_demo",
            "environment": "development",
            "data_version": "v1.0"
        })

        # ===== 2. LOG PARAMETERS =====
        params = {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        }
        mlflow.log_params(params)

        # Also log data characteristics
        mlflow.log_params({
            "n_training_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "churn_rate_train": round(y_train.mean(), 3)
        })

        # ===== 3. TRAIN MODEL =====
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # ===== 4. LOG METRICS =====
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Standard metrics
        metrics = {
            "test_roc_auc": roc_auc_score(y_test, y_prob),
            "test_accuracy": (y_pred == y_test).mean(),
            "test_precision": (y_pred[y_pred == 1] == y_test[y_pred == 1]).mean() if sum(y_pred) > 0 else 0,
            "test_recall": (y_pred[y_test == 1] == y_test[y_test == 1]).mean() if sum(y_test) > 0 else 0
        }
        mlflow.log_metrics(metrics)

        # ===== 5. LOG CONFUSION MATRIX AS ARTIFACT =====
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        }

        # Save as JSON artifact
        with open("/tmp/confusion_matrix.json", "w") as f:
            json.dump(cm_dict, f, indent=2)
        mlflow.log_artifact("/tmp/confusion_matrix.json")

        # ===== 6. LOG ROC CURVE AS FIGURE =====
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {metrics["test_roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Churn Prediction')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close()

        # ===== 7. LOG PRECISION-RECALL CURVE =====
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='green', lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "precision_recall_curve.png")
        plt.close()

        # ===== 8. LOG FEATURE IMPORTANCE =====
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save as CSV
        importance_df.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.plot(x='feature', y='importance', kind='barh', ax=ax, color='forestgreen')
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # ===== 9. LOG MODEL =====
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None  # Don't register yet
        )

        # ===== 10. LOG CUSTOM DICTIONARY AS ARTIFACT =====
        run_summary = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "parameters": params,
            "top_features": importance_df.head(5).to_dict('records'),
            "data_info": {
                "is_fictional": True,
                "purpose": "demonstration"
            }
        }

        with open("/tmp/run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)
        mlflow.log_artifact("/tmp/run_summary.json")

        print(f"\nRun completed: {run_id}")
        print(f"ROC-AUC: {metrics['test_roc_auc']:.4f}")

        return model, run_id, metrics

# COMMAND ----------

model, run_id, metrics = train_with_advanced_tracking(
    X_train, X_test, y_train, y_test,
    run_name="advanced_tracking_demo"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Hyperparameter Search with Tracking
# MAGIC
# MAGIC MLflow can track all runs from a hyperparameter search automatically.
# MAGIC This makes it easy to compare different configurations.

# COMMAND ----------

def hyperparameter_search_with_tracking(X_train, X_test, y_train, y_test):
    """
    Perform hyperparameter search with MLflow tracking for each combination.
    """

    # Define search space
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    # Track parent run
    with mlflow.start_run(run_name="hyperparameter_search") as parent_run:
        mlflow.set_tag("search_type", "grid_search")
        mlflow.log_param("param_grid", str(param_grid))

        best_auc = 0
        best_params = None
        all_results = []

        # Nested runs for each parameter combination
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:

                    params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_split': min_split,
                        'random_state': RANDOM_STATE
                    }

                    run_name = f"n{n_est}_d{depth}_s{min_split}"

                    with mlflow.start_run(run_name=run_name, nested=True):
                        mlflow.log_params(params)

                        # Train
                        model = RandomForestClassifier(**params)
                        model.fit(X_train, y_train)

                        # Evaluate
                        y_prob = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_prob)

                        mlflow.log_metric("roc_auc", auc)

                        all_results.append({**params, 'roc_auc': auc})

                        if auc > best_auc:
                            best_auc = auc
                            best_params = params.copy()

        # Log best results to parent
        mlflow.log_metric("best_roc_auc", best_auc)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Save all results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("/tmp/search_results.csv", index=False)
        mlflow.log_artifact("/tmp/search_results.csv")

        print(f"\nHyperparameter Search Complete")
        print(f"Best ROC-AUC: {best_auc:.4f}")
        print(f"Best Parameters: {best_params}")

        return results_df, best_params

# COMMAND ----------

search_results, best_params = hyperparameter_search_with_tracking(
    X_train, X_test, y_train, y_test
)
display(search_results.sort_values('roc_auc', ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Query and Compare Runs
# MAGIC
# MAGIC I use the MLflow Client to programmatically query and compare runs.

# COMMAND ----------

def query_experiment_runs(experiment_name: str):
    """
    Query and analyze all runs in an experiment.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Search all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_roc_auc DESC"]
    )

    if len(runs) == 0:
        print("No runs found")
        return None

    print(f"Found {len(runs)} runs in experiment")

    # Display summary
    summary_cols = ['run_id', 'status', 'start_time']
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    param_cols = [col for col in runs.columns if col.startswith('params.')]

    # Show top runs by AUC
    if 'metrics.test_roc_auc' in runs.columns:
        print("\nTop 5 Runs by ROC-AUC:")
        display_cols = ['run_id', 'metrics.test_roc_auc', 'params.n_estimators', 'params.max_depth']
        available_cols = [col for col in display_cols if col in runs.columns]
        print(runs[available_cols].head(5).to_string(index=False))

    return runs

# COMMAND ----------

all_runs = query_experiment_runs(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Load and Compare Models
# MAGIC
# MAGIC I can load any logged model by its run ID and compare predictions.

# COMMAND ----------

def load_and_compare_models(run_ids: list, X_test, y_test):
    """
    Load models from multiple runs and compare their predictions.
    """

    predictions = {}
    metrics = {}

    for run_id in run_ids:
        try:
            # Load model
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            # Get predictions
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)

            predictions[run_id[:8]] = y_prob
            metrics[run_id[:8]] = auc

            print(f"Run {run_id[:8]}: ROC-AUC = {auc:.4f}")
        except Exception as e:
            print(f"Could not load run {run_id[:8]}: {e}")

    return predictions, metrics

# COMMAND ----------

# Compare last few runs if available
if all_runs is not None and len(all_runs) > 0:
    recent_runs = all_runs['run_id'].head(3).tolist()
    predictions, metrics = load_and_compare_models(recent_runs, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Artifacts Exploration
# MAGIC
# MAGIC I can list and download artifacts from any run.

# COMMAND ----------

def explore_run_artifacts(run_id: str):
    """
    List and explore artifacts from a specific run.
    """

    print(f"Artifacts for run: {run_id[:8]}")
    print("-" * 40)

    try:
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            print(f"  {artifact.path} ({artifact.file_size} bytes)")
    except Exception as e:
        print(f"Error listing artifacts: {e}")

# COMMAND ----------

# Explore artifacts from latest run
if run_id:
    explore_run_artifacts(run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC > **Reminder:** All data and results shown are based on **fictional data** for demonstration.
# MAGIC
# MAGIC In this notebook, I demonstrated advanced MLflow tracking:
# MAGIC
# MAGIC | Feature | Description |
# MAGIC |---------|-------------|
# MAGIC | **Tags** | Organize runs with metadata (project, author, environment) |
# MAGIC | **Parameters** | Log all hyperparameters and data characteristics |
# MAGIC | **Metrics** | Track evaluation metrics (AUC, accuracy, etc.) |
# MAGIC | **Artifacts** | Store figures, CSVs, JSONs with each run |
# MAGIC | **Nested Runs** | Organize hyperparameter search |
# MAGIC | **Model Logging** | Save trained models for later use |
# MAGIC | **Run Comparison** | Query and compare runs programmatically |
# MAGIC
# MAGIC **MLflow Tracking Best Practices:**
# MAGIC 1. Use meaningful run names
# MAGIC 2. Tag runs with project metadata
# MAGIC 3. Log data characteristics (not just model params)
# MAGIC 4. Save confusion matrices and curves as artifacts
# MAGIC 5. Use nested runs for hyperparameter search
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Register best model in Model Registry (Notebook 04)
