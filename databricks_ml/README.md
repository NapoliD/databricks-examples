# Customer Churn Prediction with MLflow

> **Important:** All data in this project is **fictional** and was created for demonstration and educational purposes only. This project showcases my skills with MLflow, Databricks, and production ML pipelines.

I built this project to demonstrate end-to-end Machine Learning operations (MLOps) using MLflow on Databricks. The goal is to predict customer churn in an e-commerce context using a complete pipeline from feature engineering to batch inference.

---

## Why I Built This

As a Data Engineer/ML Engineer, I wanted to showcase that I understand not just how to train models, but how to operationalize them in production. This project demonstrates:

- **Feature Engineering at Scale**: Using PySpark and Databricks Feature Store
- **Experiment Tracking**: Comprehensive logging with MLflow
- **Model Comparison**: Evaluating multiple algorithms systematically
- **Model Registry**: Version control and lifecycle management for ML models
- **Batch Inference**: Scalable prediction pipelines

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE STORE                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   RFM Features  │  │   Behavioral    │  │   Temporal      │             │
│  │   - Recency     │  │   - Cancel rate │  │   - Tenure      │             │
│  │   - Frequency   │  │   - Basket size │  │   - Velocity    │             │
│  │   - Monetary    │  │   - Diversity   │  │   - Lifecycle   │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLFLOW EXPERIMENT TRACKING                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │    Logistic     │  │  Random Forest  │  │    XGBoost      │  ← Best     │
│  │   Regression    │  │                 │  │                 │             │
│  │   AUC: ~0.78    │  │   AUC: ~0.85    │  │   AUC: ~0.89    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL REGISTRY                                    │
│                                                                             │
│    v1 (Archived) ──► v2 (Staging) ──► v3 (Production)                      │
│                                                                             │
│    Tags: data_version, training_date, author, validated                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BATCH INFERENCE                                    │
│                                                                             │
│    Load Production Model ──► Score Customers ──► Save to Delta Lake        │
│                                                                             │
│    Output: churn_probability, churn_risk (HIGH/MEDIUM/LOW)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
databricks_ml/
├── notebooks/
│   ├── 01_feature_engineering.py    # Feature Store + RFM/Behavioral/Temporal
│   ├── 02_model_training.py         # Train & compare 3 algorithms
│   ├── 03_mlflow_tracking.py        # Advanced experiment tracking
│   ├── 04_model_registry.py         # Model versioning & lifecycle
│   └── 05_batch_inference.py        # Production scoring pipeline
│
├── tests/
│   └── test_features.py             # Unit tests for feature engineering
│
├── config/
│   └── ml_config.yaml               # Centralized configuration
│
├── README.md                        # English documentation (this file)
└── README_ES.md                     # Spanish documentation
```

---

## Notebooks Overview

### 01. Feature Engineering

I designed a comprehensive feature engineering pipeline that captures three dimensions of customer behavior:

| Feature Group | Features | Business Meaning |
|---------------|----------|------------------|
| **RFM** | recency_days, frequency, monetary_value | Core customer value metrics |
| **Behavioral** | cancellation_rate, avg_basket_size, unique_products | Satisfaction indicators |
| **Temporal** | days_since_registration, purchase_velocity | Lifecycle stage |

**Key decisions I made:**
- Used Feature Store for feature versioning and reusability
- Created churn label based on 30-day inactivity threshold
- Handled null values with business-logical defaults

### 02. Model Training

I trained and compared three algorithms to find the best performer:

| Model | Why I Chose It |
|-------|----------------|
| **Logistic Regression** | Interpretable baseline, good for understanding feature importance |
| **Random Forest** | Robust to outliers, captures non-linear relationships |
| **XGBoost** | State-of-the-art for tabular data, handles imbalanced classes well |

**Key techniques:**
- Cross-validation for reliable performance estimates
- Class weighting for imbalanced data
- Feature importance analysis across models

### 03. MLflow Tracking

I demonstrated advanced MLflow tracking capabilities:

- **Tags**: Project metadata for organization (author, data_version, environment)
- **Parameters**: All hyperparameters and data characteristics
- **Metrics**: AUC, precision, recall, F1-score
- **Artifacts**: ROC curves, confusion matrices, feature importance plots
- **Nested Runs**: Organized hyperparameter search

### 04. Model Registry

I implemented the full model lifecycle:

```
None → Staging → Production → Archived
```

**Key practices:**
- Always validate in Staging before Production
- Archive (don't delete) old production models
- Add meaningful descriptions and tags
- Load by stage in production code for seamless updates

### 05. Batch Inference

I built a production-ready scoring pipeline:

- **Pandas UDFs**: For distributed scoring at scale
- **Risk Categories**: HIGH (>70%), MEDIUM (40-70%), LOW (<40%)
- **Delta Lake**: Partitioned storage for efficient queries
- **Monitoring Metrics**: Track prediction distributions

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Databricks** | Cloud platform, notebooks, clusters |
| **MLflow** | Experiment tracking, model registry |
| **PySpark** | Distributed data processing |
| **Delta Lake** | ACID storage, time travel |
| **Feature Store** | Feature versioning and serving |
| **XGBoost** | Gradient boosting model |
| **scikit-learn** | ML algorithms and utilities |
| **pytest** | Unit testing |

---

## How to Run

### On Databricks

1. Clone or import this repository to your workspace
2. Configure paths in `config/ml_config.yaml`
3. Run notebooks in order: 01 → 02 → 03 → 04 → 05

### Locally (for testing)

```bash
# Install dependencies
pip install pyspark mlflow xgboost scikit-learn pandas numpy pytest

# Run tests
pytest tests/ -v
```

---

## Key Learnings

Building this project taught me several important lessons:

1. **Feature Engineering Matters More Than Model Selection**: RFM features alone provided most of the predictive power.

2. **MLflow Tracking is Essential**: Without proper logging, reproducing experiments becomes impossible.

3. **Model Registry Provides Safety**: Stage-based deployment prevents accidental production issues.

4. **Batch vs Real-time**: For churn prediction, daily batch scoring is sufficient and more cost-effective.

5. **Data Quality First**: I spent significant time ensuring features were correctly calculated.

---

## What I Would Add in Production

If this were a real production system, I would add:

- **Unity Catalog**: For data governance and lineage
- **Databricks Workflows**: For orchestration
- **Model Monitoring**: Drift detection with automated retraining
- **A/B Testing**: Gradual rollout of new models
- **Real-time Serving**: For high-value customers

---

## About the Data

> **All data in this project is entirely fictional.**

I created synthetic customer, order, and payment data to demonstrate the ML pipeline. The patterns in the data are designed to be realistic enough to train meaningful models, but no real customer information was used.

---

## Contact

This project is part of my portfolio demonstrating MLOps skills with Databricks and MLflow. Feel free to reach out if you have questions about the implementation.

---

## License

MIT License
