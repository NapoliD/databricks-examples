# Delta Live Tables - E-commerce Pipeline

> **Important:** All data in this project is **FICTIONAL** and was created for demonstration and educational purposes only. This project showcases my skills with Delta Live Tables on Databricks.

I built this project to demonstrate declarative data pipelines using Delta Live Tables (DLT). DLT represents the evolution of data engineering on Databricks - moving from imperative Spark code to declarative pipeline definitions.

---

## Why I Built This

After building traditional Spark ETL pipelines (see `databricks_etl/`), I wanted to show that I understand the modern, declarative approach with DLT. This project demonstrates:

- **Declarative Tables**: Define WHAT you want, not HOW to build it
- **Built-in Data Quality**: Expectations that automatically track quality metrics
- **CDC Handling**: `apply_changes()` for real-time data synchronization
- **Both Python and SQL**: Showing versatility in pipeline development

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DELTA LIVE TABLES PIPELINE                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         BRONZE LAYER (Python)                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Orders  │  │ Customers│  │ Payments │  │ Products │            │   │
│  │  │  (CSV)   │  │  (CSV)   │  │  (JSON)  │  │(Streaming)│           │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │   │
│  │       │             │             │             │                    │   │
│  │       ▼             ▼             ▼             ▼                    │   │
│  │  @dlt.expect  @dlt.expect  @dlt.expect_or_drop  cloudFiles          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SILVER LAYER (SQL)                          │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │   │
│  │  │  silver_orders   │  │ silver_customers │  │ silver_payments │   │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘   │   │
│  │           │                     │                      │            │   │
│  │           └──────────┬──────────┴──────────────────────┘            │   │
│  │                      ▼                                              │   │
│  │           ┌──────────────────────┐     ┌─────────────────┐         │   │
│  │           │ silver_enriched_orders│     │ silver_quarantine│        │   │
│  │           │   (Joined + Clean)    │     │  (Failed rows)  │         │   │
│  │           └──────────┬────────────┘     └─────────────────┘         │   │
│  └──────────────────────┼──────────────────────────────────────────────┘   │
│                         │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         GOLD LAYER (Python)                          │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │   Daily    │ │  Country   │ │  Product   │ │ Customer   │       │   │
│  │  │  Revenue   │ │Performance │ │Performance │ │    360     │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CDC PIPELINE (Python)                        │   │
│  │                                                                      │   │
│  │  cdc_raw ──► apply_changes() ──► customers_scd1 (current state)     │   │
│  │                              ──► customers_scd2 (full history)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
databricks_dlt/
├── pipelines/
│   ├── 01_bronze_ingestion.py        # Raw data with expectations
│   ├── 02_silver_transformations.sql # SQL-based cleaning
│   ├── 03_gold_aggregations.py       # Business metrics
│   └── 04_cdc_pipeline.py            # CDC with apply_changes()
│
├── config/
│   └── pipeline_settings.json        # DLT pipeline configuration
│
├── README.md                         # English documentation
└── README_ES.md                      # Spanish documentation
```

---

## Pipeline Components

### 01. Bronze Ingestion (Python)

I implemented raw data ingestion with data quality expectations:

| Table | Source | Expectations |
|-------|--------|--------------|
| `bronze_orders` | CSV | NOT NULL, quantity > 0 |
| `bronze_customers` | CSV | Valid email format |
| `bronze_payments` | JSON | Amount > 0 |
| `bronze_products_streaming` | Auto Loader | Price > 0 |

**Key Features:**
- `@dlt.expect`: Warn on violation
- `@dlt.expect_or_drop`: Remove invalid rows
- Auto Loader for streaming ingestion
- Audit columns for traceability

### 02. Silver Transformations (SQL)

I chose SQL for the Silver layer to demonstrate versatility:

```sql
CREATE OR REFRESH LIVE TABLE silver_orders
COMMENT "Cleaned orders. FICTIONAL DATA."
AS
SELECT
  order_id,
  CAST(quantity AS INT) AS quantity,
  TRIM(UPPER(status)) AS order_status
FROM LIVE.bronze_orders
```

**SQL DLT Features:**
- `CREATE OR REFRESH LIVE TABLE`
- `CONSTRAINT ... EXPECT` for quality
- `ON VIOLATION DROP ROW` for enforcement
- Standard SQL joins with `LIVE.` prefix

### 03. Gold Aggregations (Python)

Business-ready tables with KPIs:

| Table | Purpose |
|-------|---------|
| `gold_daily_revenue` | Time-series revenue analysis |
| `gold_country_performance` | Geographic performance with rankings |
| `gold_product_performance` | Product analytics |
| `gold_customer_360` | Customer lifetime value and segmentation |
| `gold_payment_analytics` | Payment method analysis |
| `gold_executive_summary` | High-level KPIs |

### 04. CDC Pipeline (Python)

Change Data Capture with `apply_changes()`:

```python
dlt.apply_changes(
    target="customers_scd2",
    source="cdc_customers_raw",
    keys=["customer_id"],
    sequence_by=col("updated_at"),
    apply_as_deletes=expr("operation = 'DELETE'"),
    stored_as_scd_type=2
)
```

**CDC Features:**
- SCD Type 1 (overwrite current state)
- SCD Type 2 (full history with timestamps)
- Automatic delete handling
- Out-of-order event handling

---

## Data Quality Expectations

| Layer | Expectation Type | Action |
|-------|------------------|--------|
| Bronze | `@dlt.expect` | Warn and track |
| Bronze | `@dlt.expect_or_drop` | Remove invalid rows |
| Silver | `CONSTRAINT EXPECT` | SQL-based validation |
| Silver | Quarantine table | Capture failures |

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Delta Live Tables** | Declarative pipeline framework |
| **Python DLT** | Complex transformations |
| **SQL DLT** | Simple transformations |
| **Auto Loader** | Streaming file ingestion |
| **apply_changes()** | CDC processing |
| **Expectations** | Data quality |
| **Unity Catalog** | Governance (optional) |

---

## How to Run

### On Databricks

1. Import this repository to your Databricks workspace
2. Create a new DLT pipeline:
   - Go to **Workflows** → **Delta Live Tables** → **Create Pipeline**
   - Use the configuration from `config/pipeline_settings.json`
3. Add the notebooks from `pipelines/` folder
4. Configure source and target paths
5. Start the pipeline

### Pipeline Modes

| Mode | Use Case |
|------|----------|
| **Development** | Testing with full refresh |
| **Production** | Incremental processing |
| **Continuous** | Real-time streaming |

---

## DLT vs Traditional Spark

| Aspect | Traditional Spark | Delta Live Tables |
|--------|-------------------|-------------------|
| **Approach** | Imperative (HOW) | Declarative (WHAT) |
| **Dependencies** | Manual management | Automatic resolution |
| **Data Quality** | Custom code | Built-in expectations |
| **CDC** | Complex MERGE logic | `apply_changes()` |
| **Orchestration** | External (Airflow, etc.) | Built-in |
| **Lineage** | Manual tracking | Automatic |

---

## Key Learnings

Building this project taught me:

1. **Declarative is Powerful**: Focus on business logic, not plumbing
2. **Expectations are Essential**: Built-in quality tracking saves time
3. **SQL and Python Together**: Use each where it makes sense
4. **CDC Made Simple**: `apply_changes()` handles complexity automatically
5. **Streaming is Natural**: Auto Loader makes streaming as easy as batch

---

## What I Would Add in Production

- **Unity Catalog Integration**: For data governance
- **Materialized Views**: For real-time dashboards
- **Event Hubs/Kafka Sources**: For true streaming
- **Alerts and Monitoring**: Custom notifications
- **Multiple Environments**: Dev, staging, production pipelines

---

## About the Data

> **All data in this project is entirely FICTIONAL.**

I created synthetic e-commerce data (orders, customers, payments) to demonstrate DLT capabilities. No real customer or business information was used.

---

## Related Projects

- `databricks_etl/` - Traditional Spark ETL (Medallion Architecture)
- `databricks_ml/` - MLflow for ML operations

---

## License

MIT License
