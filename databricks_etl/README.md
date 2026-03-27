# Databricks ETL Pipeline - E-commerce

End-to-end data pipeline implementing Medallion Architecture (Bronze → Silver → Gold) with Delta Lake and PySpark. Includes solutions for common production challenges.

> **Note:** All data included in this repository is **fictional** and was created for demonstration and educational purposes only.

---

## Architecture

```
┌──────────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│       RAW        │────▶│   BRONZE    │────▶│   SILVER    │────▶│    GOLD     │
│  CSV/JSON/Parquet│     │  Delta Lake │     │  Delta Lake │     │  Delta Lake │
│                  │     │  Raw + Audit│     │   Cleaned   │     │  Aggregated │
└──────────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                │                   │                   │
                                ▼                   ▼                   ▼
                         ┌─────────────────────────────────────────────────────┐
                         │              DATA QUALITY FRAMEWORK                 │
                         │         Validations + Alerts + Reports              │
                         └─────────────────────────────────────────────────────┘
```

### Layers

| Layer | Description | Transformations |
|-------|-------------|-----------------|
| **Bronze** | Raw data as-is | Schema enforcement, audit metadata |
| **Silver** | Clean and normalized data | Deduplication, type casting, joins, null handling |
| **Gold** | Business-level aggregated tables | Metrics, KPIs, reports |

### Supported Ingestion Formats

| Format | Recommended Use | Advantages |
|--------|-----------------|------------|
| **CSV** | Small datasets, manual exports | Simple, universal |
| **JSON** | APIs, logs, semi-structured data | Flexible, supports nesting |
| **Parquet** | Large volumes (+100GB), data lakes | Columnar, compressed, predicate pushdown |

---

## Project Structure

```
databricks_etl/
├── notebooks/
│   │
│   │  # ── Main Pipeline (Medallion) ──
│   ├── 01_bronze_ingestion.py          # Ingest CSV, JSON, Parquet → Delta
│   ├── 02_silver_transformation.py     # Cleaning, joins, deduplication
│   ├── 03_gold_aggregation.py          # Business metrics, KPIs
│   │
│   │  # ── Advanced Patterns ──
│   ├── 04_cdc_incremental.py           # Change Data Capture with MERGE
│   ├── 05_streaming.py                 # Structured Streaming + watermarks
│   ├── 06_data_quality.py              # Validation framework
│   ├── 07_optimization.py              # Partitioning, Z-ORDER, caching
│   │
│   │  # ── Production Solutions ──
│   ├── 08_handling_data_skew.py        # Skew diagnosis and resolution
│   ├── 09_large_dataset_processing.py  # Strategies for +100GB datasets
│   └── 10_cost_optimization.py         # Cluster cost reduction
│
├── data/
│   ├── orders.csv                      # 25 orders (fictional)
│   ├── customers.csv                   # 15 LATAM customers (fictional)
│   └── payments.json                   # 18 payments (fictional)
│
├── tests/
│   └── test_transformations.py         # Unit tests with pytest
│
├── config/
│   └── pipeline_config.yaml            # Centralized configuration
│
├── LICENSE
└── README.md
```

---

## Notebooks

### Main Pipeline

| # | Notebook | Description | Key Techniques |
|---|----------|-------------|----------------|
| 01 | `bronze_ingestion` | Raw data ingestion with audit metadata | `input_file_name()`, schema enforcement, multi-format |
| 02 | `silver_transformation` | Cleaning and normalization | Deduplication, type casting, joins, null handling |
| 03 | `gold_aggregation` | Business-level aggregated tables | Window functions, metrics, segmentation |

### Advanced Patterns

| # | Notebook | Description | Key Techniques |
|---|----------|-------------|----------------|
| 04 | `cdc_incremental` | Incremental loads | `MERGE INTO`, SCD Type 2, soft deletes, time travel |
| 05 | `streaming` | Real-time processing | Structured Streaming, watermarks, window aggregations |
| 06 | `data_quality` | Validation framework | Configurable rules, quarantine, automated reports |
| 07 | `optimization` | Performance tuning | Partitioning, Z-ORDER, broadcast joins, AQE |

### Production Solutions

| # | Notebook | Problem Solved | Key Techniques |
|---|----------|----------------|----------------|
| 08 | `handling_data_skew` | Slow joins due to hot keys | Salting, broadcast, distribution diagnosis |
| 09 | `large_dataset_processing` | Datasets over 100GB | Incremental CDC, smart partitioning, anti-patterns |
| 10 | `cost_optimization` | Excessive cluster costs | Auto-scaling, Spot instances, right-sizing, lifecycle |

---

## Detailed Features

### 1. Medallion Architecture
- Batch ingestion of CSV, JSON, and Parquet
- Parquet recommended for large volumes (better compression and performance)
- Audit metadata: `_ingestion_timestamp`, `_source_file`, `_batch_id`
- Delta Lake storage with `mergeSchema`

### 2. CDC (Change Data Capture)
```python
# MERGE example for upserts
deltaTable.alias("target").merge(
    source_df.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```
- Soft delete pattern with `_is_deleted` column
- SCD Type 2 for historical tracking
- Time travel for auditing

### 3. Streaming
```python
# Example with watermark for late data
df_stream \
    .withWatermark("event_time", "30 minutes") \
    .groupBy(window("event_time", "5 minutes")) \
    .agg(sum("amount"))
```
- Window aggregations (tumbling, sliding)
- Checkpointing for fault tolerance
- Output modes: append, complete, update

### 4. Data Quality
```python
# Validation framework
rules = [
    {"column": "customer_id", "rule": "not_null"},
    {"column": "amount", "rule": "positive"},
    {"column": "email", "rule": "regex", "pattern": r".*@.*\..*"}
]
validate(df, rules)
```
- Quarantine for invalid records
- Automated quality reports
- Validation history

### 5. Optimization
- **Partitioning**: By date for temporal queries
- **Z-ORDER**: By frequently filtered columns
- **Broadcast**: For small tables (<10MB)
- **AQE**: Adaptive Query Execution enabled

### 6. Production Solutions

#### Data Skew
```python
# Skew diagnosis
def diagnose_skew(df, key_column):
    stats = df.groupBy(key_column).count()
    # If max/median > 10, there's skew
```

#### Large Datasets
```python
# Incremental processing with Change Data Feed
spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", last_version) \
    .load(source_path)
```

#### Cost Optimization
```python
# Optimized cluster configuration
{
    "autoscale": {"min_workers": 2, "max_workers": 10},
    "autotermination_minutes": 15,
    "aws_attributes": {
        "availability": "SPOT_WITH_FALLBACK",
        "first_on_demand": 1
    }
}
```

---

## How to Run

### On Databricks

1. Clone the repository or import notebooks to your workspace
2. Upload sample data to `/mnt/raw/` (or DBFS)
3. Run notebooks in order:
   ```
   01_bronze → 02_silver → 03_gold
   ```
4. Explore advanced notebooks as needed

### Local (with PySpark)

```bash
# Install dependencies
pip install pyspark delta-spark pytest

# Run tests
pytest tests/ -v

# Run specific notebook (requires Jupyter)
jupyter notebook notebooks/01_bronze_ingestion.py
```

---

## Configuration

Edit `config/pipeline_config.yaml`:

```yaml
paths:
  raw: "/mnt/raw/"
  bronze: "/mnt/bronze/"
  silver: "/mnt/silver/"
  gold: "/mnt/gold/"

processing:
  shuffle_partitions: 200
  max_records_per_file: 1000000

data_quality:
  quarantine_path: "/mnt/quarantine/"
  alert_threshold: 0.05  # 5% invalid records
```

---

## Gold Tables Generated

| Table | Description | Granularity |
|-------|-------------|-------------|
| `daily_revenue` | Aggregated revenue | Per day |
| `country_revenue` | Revenue with ranking | Per country |
| `top_products` | Best-selling products | Per product |
| `customer_metrics` | Customer 360 | Per customer |
| `payment_metrics` | Payment analysis | Per payment method |

---

## Technical Decisions

### Why Delta Lake?
| Feature | Benefit |
|---------|---------|
| ACID transactions | Consistency in concurrent writes |
| Time travel | Auditing and rollback |
| Schema evolution | Add columns without rewriting |
| MERGE | Efficient CDC |
| OPTIMIZE + Z-ORDER | Fast queries |

### Why Medallion Architecture?
- **Separation of concerns**: Each layer has a clear purpose
- **Debugging**: Easy to identify where the pipeline fails
- **Reprocessing**: Re-run from Bronze without re-ingesting
- **Industry standard**: Adopted by Databricks, Azure, AWS

### Partitioning Strategy

| Table | Partition | Reason |
|-------|-----------|--------|
| Orders | `year/month` | 90% of queries filter by date |
| Customers | `country` | Frequent geographic analysis |
| Payments | `payment_date` | Monthly financial reports |

---

## Common Problems and Solutions

### Data Skew (`08_handling_data_skew.py`)

| Problem | Symptom | Solution |
|---------|---------|----------|
| Hot keys in joins | Tasks taking 2h while others take 10s | Salting: artificially distribute keys |
| Broadcast failure | OOM in driver | Check size, use salting instead |
| Undetected skew | Job is "slow" for no apparent reason | `diagnose_skew()` to identify distribution |

### Large Datasets (`09_large_dataset_processing.py`)

| Problem | Symptom | Solution |
|---------|---------|----------|
| Full scan of 500GB | Queries taking 45+ min | Partitioning + Z-ORDER |
| Daily full refresh | $4,500/month on a single job | Incremental with Change Data Feed |
| `collect()` on large data | OOM | Process in Spark, only `collect()` aggregated results |

### Cluster Costs (`10_cost_optimization.py`)

| Problem | Symptom | Estimated Savings |
|---------|---------|-------------------|
| 24/7 cluster with 35% usage | $14,400/month | Auto-scaling → $4,000/month (72%) |
| On-demand for batch | 3x necessary cost | Spot instances → 60-70% less |
| i3.2xlarge with 30% CPU | Over-provisioned | r5.xlarge → 35% less |

---

## Production Scalability

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Databricks │────▶│   Unity     │────▶│  Downstream │
│  Workflows  │     │   Catalog   │     │   (BI/ML)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  Alerting   │     │  Lineage    │
│  (PagerDuty)│     │  Tracking   │
└─────────────┘     └─────────────┘
```

| Component | Recommended Tool |
|-----------|------------------|
| Orchestration | Databricks Workflows / Airflow |
| Monitoring | Datadog / CloudWatch / Databricks SQL Alerts |
| Secrets | Azure Key Vault / AWS Secrets Manager |
| CI/CD | GitHub Actions + Databricks CLI |
| Governance | Unity Catalog |
| Lineage | Unity Catalog / OpenLineage |

---

## License

MIT License
