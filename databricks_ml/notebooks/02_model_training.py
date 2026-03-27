# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training - Churn Prediction
# MAGIC
# MAGIC > **Note:** All data in this project is **fictional** and created for demonstration purposes only.
# MAGIC > This notebook showcases MLflow capabilities and ML best practices.
# MAGIC
# MAGIC In this notebook, I train and compare three different algorithms to predict customer churn:
# MAGIC - **Logistic Regression**: Interpretable baseline
# MAGIC - **Random Forest**: Robust ensemble method
# MAGIC - **XGBoost**: State-of-the-art gradient boosting
# MAGIC
# MAGIC I chose these three models because they represent different approaches and complexity levels,
# MAGIC which is exactly what I would do in a real production scenario.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths
GOLD_PATH = "/mnt/gold/"
FEATURES_PATH = f"{GOLD_PATH}/ml_features/customer_churn_features"

# MLflow experiment name
EXPERIMENT_NAME = "/Users/demo/churn_prediction_experiment"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features
# MAGIC
# MAGIC I load the features created in the previous notebook. In production,
# MAGIC I would use Feature Store lookup to ensure consistency.

# COMMAND ----------

def load_features():
    """Load features from Delta Lake or create sample data."""

    try:
        # Try to read from Delta
        df = spark.read.format("delta").load(FEATURES_PATH)
        print(f"Loaded {df.count()} records from Feature Store")
        return df.toPandas()
    except Exception as e:
        print(f"Loading sample data for demonstration: {e}")
        return create_sample_features()


def create_sample_features():
    """
    Create sample feature data for demonstration.
    All data is FICTIONAL and for educational purposes only.
    """

    np.random.seed(RANDOM_STATE)
    n_customers = 200

    # Generate synthetic features that mimic real patterns
    data = {
        'customer_id': [f'CUST{str(i).zfill(3)}' for i in range(1, n_customers + 1)],
        'recency_days': np.random.exponential(15, n_customers).astype(int) + 1,
        'frequency': np.random.poisson(3, n_customers) + 1,
        'monetary_value': np.random.lognormal(4, 1, n_customers).round(2),
        'avg_order_value': np.random.lognormal(3.5, 0.8, n_customers).round(2),
        'max_order_value': np.random.lognormal(4.5, 1, n_customers).round(2),
        'min_order_value': np.random.lognormal(2.5, 0.5, n_customers).round(2),
        'cancellation_rate': np.random.beta(2, 10, n_customers).round(3) * 100,
        'pending_rate': np.random.beta(1, 8, n_customers).round(3) * 100,
        'unique_products': np.random.poisson(2, n_customers) + 1,
        'avg_basket_size': np.random.lognormal(0.5, 0.5, n_customers).round(2),
        'total_items': np.random.poisson(5, n_customers) + 1,
        'payment_methods_used': np.random.choice([1, 2, 3], n_customers, p=[0.6, 0.3, 0.1]),
        'days_since_registration': np.random.randint(30, 400, n_customers),
        'days_since_first_order': np.random.randint(5, 350, n_customers),
        'days_since_last_order': np.random.exponential(20, n_customers).astype(int) + 1,
        'active_period_days': np.random.randint(0, 300, n_customers),
        'purchase_velocity': np.random.lognormal(-1, 0.8, n_customers).round(3),
        'days_to_first_order': np.random.randint(0, 60, n_customers)
    }

    df = pd.DataFrame(data)

    # Create realistic churn label based on features
    # Higher recency, lower frequency, high cancellation = more likely to churn
    churn_probability = (
        0.3 * (df['recency_days'] / df['recency_days'].max()) +
        0.2 * (1 - df['frequency'] / df['frequency'].max()) +
        0.2 * (df['cancellation_rate'] / 100) +
        0.15 * (1 - df['purchase_velocity'] / df['purchase_velocity'].max()) +
        0.15 * np.random.random(n_customers)
    )

    df['is_churned'] = (churn_probability > 0.45).astype(int)

    print(f"Created sample dataset with {n_customers} customers (FICTIONAL DATA)")
    print(f"Churn rate: {df['is_churned'].mean():.1%}")

    return df

# COMMAND ----------

# Load data
features_df = load_features()
print(f"\nDataset shape: {features_df.shape}")
print(f"Churn distribution:\n{features_df['is_churned'].value_counts(normalize=True)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare Data for Training
# MAGIC
# MAGIC I split the data and scale features. Scaling is important for Logistic Regression
# MAGIC but not strictly necessary for tree-based models.

# COMMAND ----------

def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for model training.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """

    # Feature columns (exclude customer_id and target)
    feature_cols = [col for col in df.columns if col not in ['customer_id', 'is_churned']]

    X = df[feature_cols]
    y = df['is_churned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, X_train, X_test

# COMMAND ----------

X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, X_train_raw, X_test_raw = prepare_data(
    features_df, TEST_SIZE, RANDOM_STATE
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set Up MLflow Experiment
# MAGIC
# MAGIC I create a dedicated experiment to track all training runs.

# COMMAND ----------

# Set or create experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train Model 1: Logistic Regression
# MAGIC
# MAGIC Logistic Regression serves as my baseline. It's interpretable and provides
# MAGIC a good benchmark to compare against more complex models.

# COMMAND ----------

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Logistic Regression with hyperparameter tuning.
    """

    with mlflow.start_run(run_name="logistic_regression"):
        # Log parameters
        params = {
            "model_type": "LogisticRegression",
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "C": 1.0,
            "random_state": RANDOM_STATE
        }
        mlflow.log_params(params)

        # Train model
        model = LogisticRegression(**{k: v for k, v in params.items() if k != "model_type"})
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)

        # Log feature importance (coefficients)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)

        # Save feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.head(10).plot(x='feature', y='importance', kind='barh', ax=ax)
        plt.title('Logistic Regression - Feature Importance (Coefficients)')
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        print(f"\nLogistic Regression Results:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        return model, metrics

# COMMAND ----------

lr_model, lr_metrics = train_logistic_regression(
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Model 2: Random Forest
# MAGIC
# MAGIC Random Forest is more robust to outliers and doesn't require scaling.
# MAGIC I use it to capture non-linear relationships.

# COMMAND ----------

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Random Forest with hyperparameter tuning.
    """

    with mlflow.start_run(run_name="random_forest"):
        # Log parameters
        params = {
            "model_type": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        mlflow.log_params(params)

        # Train model (using raw features, no scaling needed)
        model = RandomForestClassifier(**{k: v for k, v in params.items() if k != "model_type"})
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)

        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.head(10).plot(x='feature', y='importance', kind='barh', ax=ax, color='forestgreen')
        plt.title('Random Forest - Feature Importance')
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        print(f"\nRandom Forest Results:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        return model, metrics, importance_df

# COMMAND ----------

rf_model, rf_metrics, rf_importance = train_random_forest(
    X_train_raw, X_test_raw, y_train, y_test, feature_names
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train Model 3: XGBoost
# MAGIC
# MAGIC XGBoost is often the winning algorithm in ML competitions. It handles
# MAGIC imbalanced data well with the `scale_pos_weight` parameter.

# COMMAND ----------

def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """
    Train XGBoost with hyperparameter tuning.
    """

    with mlflow.start_run(run_name="xgboost"):
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Log parameters
        params = {
            "model_type": "XGBoost",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
            "eval_metric": "auc"
        }
        mlflow.log_params(params)

        # Train model
        model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "model_type"})
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)

        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.head(10).plot(x='feature', y='importance', kind='barh', ax=ax, color='darkorange')
        plt.title('XGBoost - Feature Importance')
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()

        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.xgboost.log_model(model, "model", signature=signature)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='roc_auc')
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

        print(f"\nXGBoost Results:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        return model, metrics, importance_df

# COMMAND ----------

xgb_model, xgb_metrics, xgb_importance = train_xgboost(
    X_train_raw, X_test_raw, y_train, y_test, feature_names
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Comparison
# MAGIC
# MAGIC Now I compare all models side by side to select the best one.

# COMMAND ----------

def compare_models(models_metrics: dict):
    """
    Create comparison visualization of all models.
    """

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(models_metrics).T
    comparison_df.index.name = 'model'

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(comparison_df.round(4).to_string())

    # Find best model
    best_model = comparison_df['roc_auc'].idxmax()
    print(f"\nBest Model (by ROC-AUC): {best_model}")
    print("="*60)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Metrics comparison
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc']
    comparison_df[metrics_to_plot].plot(kind='bar', ax=axes[0], rot=0)
    axes[0].set_title('Model Comparison - Key Metrics')
    axes[0].set_ylabel('Score')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0, 1)

    # Plot 2: ROC-AUC highlight
    colors = ['blue', 'green', 'orange']
    comparison_df['roc_auc'].plot(kind='bar', ax=axes[1], color=colors, rot=0)
    axes[1].set_title('ROC-AUC Comparison')
    axes[1].set_ylabel('ROC-AUC Score')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random Classifier')

    plt.tight_layout()
    plt.show()

    return comparison_df, best_model

# COMMAND ----------

models_metrics = {
    'Logistic Regression': lr_metrics,
    'Random Forest': rf_metrics,
    'XGBoost': xgb_metrics
}

comparison_df, best_model = compare_models(models_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Top Feature Analysis
# MAGIC
# MAGIC I analyze which features are most important across all models.

# COMMAND ----------

def analyze_top_features(rf_importance, xgb_importance):
    """
    Analyze and visualize top features across models.
    """

    # Combine importance from both tree-based models
    combined = rf_importance.merge(
        xgb_importance,
        on='feature',
        suffixes=('_rf', '_xgb')
    )
    combined['avg_importance'] = (combined['importance_rf'] + combined['importance_xgb']) / 2
    combined = combined.sort_values('avg_importance', ascending=False)

    print("\nTop 10 Most Important Features (Average):")
    print(combined.head(10)[['feature', 'importance_rf', 'importance_xgb', 'avg_importance']].to_string(index=False))

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(10)
    width = 0.35

    top_10 = combined.head(10)
    ax.bar(x - width/2, top_10['importance_rf'], width, label='Random Forest', color='forestgreen')
    ax.bar(x + width/2, top_10['importance_xgb'], width, label='XGBoost', color='darkorange')

    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Comparison: Random Forest vs XGBoost')
    ax.set_xticks(x)
    ax.set_xticklabels(top_10['feature'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

    return combined

# COMMAND ----------

feature_analysis = analyze_top_features(rf_importance, xgb_importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC > **Reminder:** All results shown are based on **fictional data** created for demonstration purposes.
# MAGIC
# MAGIC In this notebook, I trained and compared three ML models for churn prediction:
# MAGIC
# MAGIC | Model | ROC-AUC | Precision | Recall | F1-Score |
# MAGIC |-------|---------|-----------|--------|----------|
# MAGIC | Logistic Regression | Baseline | Interpretable | - | - |
# MAGIC | Random Forest | Better | Feature importance | - | - |
# MAGIC | XGBoost | Best | Production-ready | - | - |
# MAGIC
# MAGIC **Key Findings:**
# MAGIC 1. **Top predictive features:** recency_days, cancellation_rate, purchase_velocity
# MAGIC 2. **XGBoost** typically performs best for tabular data
# MAGIC 3. **Feature importance** is consistent across models
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Explore MLflow tracking UI (Notebook 03)
# MAGIC - Register best model in Model Registry (Notebook 04)
# MAGIC - Deploy for batch inference (Notebook 05)
