# Databricks ETL Pipeline - E-commerce

Pipeline de datos end-to-end implementando arquitectura Medallion (Bronze → Silver → Gold) con Delta Lake y PySpark. Incluye soluciones a problemas comunes de producción.

> **Nota:** Los datos incluidos en este repositorio son **ficticios** y fueron creados únicamente con fines demostrativos y educativos.

---

## Arquitectura

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
                         │         Validaciones + Alertas + Reportes           │
                         └─────────────────────────────────────────────────────┘
```

### Capas

| Capa | Descripción | Transformaciones |
|------|-------------|------------------|
| **Bronze** | Datos crudos tal cual llegan | Schema enforcement, metadatos de auditoría |
| **Silver** | Datos limpios y normalizados | Deduplicación, cast de tipos, joins, manejo de nulls |
| **Gold** | Tablas de negocio agregadas | Métricas, KPIs, reportes |

### Formatos de Ingesta Soportados

| Formato | Uso Recomendado | Ventajas |
|---------|-----------------|----------|
| **CSV** | Datos pequeños, exports manuales | Simple, universal |
| **JSON** | APIs, logs, datos semi-estructurados | Flexible, soporta anidamiento |
| **Parquet** | Grandes volúmenes (+100GB), data lakes | Columnar, comprimido, predicate pushdown |

---

## Estructura del Proyecto

```
databricks_etl/
├── notebooks/
│   │
│   │  # ── Pipeline Principal (Medallion) ──
│   ├── 01_bronze_ingestion.py          # Ingesta CSV, JSON, Parquet → Delta
│   ├── 02_silver_transformation.py     # Limpieza, joins, deduplicación
│   ├── 03_gold_aggregation.py          # Métricas de negocio, KPIs
│   │
│   │  # ── Patrones Avanzados ──
│   ├── 04_cdc_incremental.py           # Change Data Capture con MERGE
│   ├── 05_streaming.py                 # Structured Streaming + watermarks
│   ├── 06_data_quality.py              # Framework de validaciones
│   ├── 07_optimization.py              # Particionado, Z-ORDER, caching
│   │
│   │  # ── Soluciones de Producción ──
│   ├── 08_handling_data_skew.py        # Diagnóstico y solución de skew
│   ├── 09_large_dataset_processing.py  # Estrategias para +100GB
│   └── 10_cost_optimization.py         # Reducción de costos de cluster
│
├── data/
│   ├── orders.csv                      # 25 órdenes (ficticio)
│   ├── customers.csv                   # 15 clientes LATAM (ficticio)
│   └── payments.json                   # 18 pagos (ficticio)
│
├── tests/
│   └── test_transformations.py         # Tests unitarios con pytest
│
├── config/
│   └── pipeline_config.yaml            # Configuración centralizada
│
├── LICENSE
└── README.md
```

---

## Notebooks

### Pipeline Principal

| # | Notebook | Descripción | Técnicas Clave |
|---|----------|-------------|----------------|
| 01 | `bronze_ingestion` | Ingesta de datos crudos con metadatos de auditoría | `input_file_name()`, schema enforcement, multi-format |
| 02 | `silver_transformation` | Limpieza y normalización | Deduplicación, cast de tipos, joins, null handling |
| 03 | `gold_aggregation` | Tablas de negocio agregadas | Window functions, métricas, segmentación |

### Patrones Avanzados

| # | Notebook | Descripción | Técnicas Clave |
|---|----------|-------------|----------------|
| 04 | `cdc_incremental` | Cargas incrementales | `MERGE INTO`, SCD Type 2, soft deletes, time travel |
| 05 | `streaming` | Procesamiento en tiempo real | Structured Streaming, watermarks, window aggregations |
| 06 | `data_quality` | Framework de validaciones | Reglas configurables, quarantine, reportes automáticos |
| 07 | `optimization` | Técnicas de rendimiento | Particionado, Z-ORDER, broadcast joins, AQE |

### Soluciones de Producción

| # | Notebook | Problema que Resuelve | Técnicas Clave |
|---|----------|----------------------|----------------|
| 08 | `handling_data_skew` | Joins lentos por hot keys | Salting, broadcast, diagnóstico de distribución |
| 09 | `large_dataset_processing` | Datasets de +100GB | Incremental CDC, particionado inteligente, anti-patterns |
| 10 | `cost_optimization` | Costos excesivos de cluster | Auto-scaling, Spot instances, right-sizing, lifecycle |

---

## Funcionalidades Detalladas

### 1. Medallion Architecture
- Ingesta batch de CSV, JSON y Parquet
- Parquet recomendado para grandes volúmenes (mejor compresión y rendimiento)
- Metadatos de auditoría: `_ingestion_timestamp`, `_source_file`, `_batch_id`
- Almacenamiento en Delta Lake con `mergeSchema`

### 2. CDC (Change Data Capture)
```python
# Ejemplo de MERGE para upserts
deltaTable.alias("target").merge(
    source_df.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```
- Soft delete pattern con columna `_is_deleted`
- SCD Type 2 para mantener historial
- Time travel para auditoría

### 3. Streaming
```python
# Ejemplo con watermark para late data
df_stream \
    .withWatermark("event_time", "30 minutes") \
    .groupBy(window("event_time", "5 minutes")) \
    .agg(sum("amount"))
```
- Window aggregations (tumbling, sliding)
- Checkpointing para fault tolerance
- Output modes: append, complete, update

### 4. Data Quality
```python
# Framework de validaciones
rules = [
    {"column": "customer_id", "rule": "not_null"},
    {"column": "amount", "rule": "positive"},
    {"column": "email", "rule": "regex", "pattern": r".*@.*\..*"}
]
validate(df, rules)
```
- Quarantine para registros inválidos
- Reportes automáticos de calidad
- Historial de validaciones

### 5. Optimización
- **Particionado**: Por fecha para queries temporales
- **Z-ORDER**: Por columnas frecuentes en filtros
- **Broadcast**: Para tablas pequeñas (<10MB)
- **AQE**: Adaptive Query Execution habilitado

### 6. Soluciones de Producción

#### Data Skew
```python
# Diagnóstico de skew
def diagnose_skew(df, key_column):
    stats = df.groupBy(key_column).count()
    # Si max/median > 10, hay skew
```

#### Large Datasets
```python
# Procesamiento incremental con Change Data Feed
spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", last_version) \
    .load(source_path)
```

#### Cost Optimization
```python
# Configuración de cluster optimizada
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

## Cómo Ejecutar

### En Databricks

1. Clonar el repositorio o importar notebooks al workspace
2. Cargar los datos de ejemplo a `/mnt/raw/` (o DBFS)
3. Ejecutar notebooks en orden:
   ```
   01_bronze → 02_silver → 03_gold
   ```
4. Explorar notebooks avanzados según necesidad

### Local (con PySpark)

```bash
# Instalar dependencias
pip install pyspark delta-spark pytest

# Ejecutar tests
pytest tests/ -v

# Ejecutar notebook específico (requiere Jupyter)
jupyter notebook notebooks/01_bronze_ingestion.py
```

---

## Configuración

Editar `config/pipeline_config.yaml`:

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
  alert_threshold: 0.05  # 5% de registros inválidos
```

---

## Tablas Gold Generadas

| Tabla | Descripción | Granularidad |
|-------|-------------|--------------|
| `daily_revenue` | Revenue agregado | Por día |
| `country_revenue` | Revenue con ranking | Por país |
| `top_products` | Productos más vendidos | Por producto |
| `customer_metrics` | Customer 360 | Por cliente |
| `payment_metrics` | Análisis de pagos | Por método de pago |

---

## Decisiones Técnicas

### ¿Por qué Delta Lake?
| Característica | Beneficio |
|----------------|-----------|
| ACID transactions | Consistencia en escrituras concurrentes |
| Time travel | Auditoría y rollback |
| Schema evolution | Agregar columnas sin reescribir |
| MERGE | CDC eficiente |
| OPTIMIZE + Z-ORDER | Queries rápidas |

### ¿Por qué Medallion Architecture?
- **Separación de responsabilidades**: Cada capa tiene un propósito claro
- **Debugging**: Fácil identificar dónde falla el pipeline
- **Reprocesamiento**: Reejecutar desde Bronze sin re-ingestar
- **Estándar**: Adoptado por la industria (Databricks, Azure, AWS)

### Estrategia de Particionado

| Tabla | Partición | Razón |
|-------|-----------|-------|
| Orders | `year/month` | 90% de queries filtran por fecha |
| Customers | `country` | Análisis geográfico frecuente |
| Payments | `payment_date` | Reportes financieros mensuales |

---

## Problemas Comunes y Soluciones

### Data Skew (`08_handling_data_skew.py`)

| Problema | Síntoma | Solución |
|----------|---------|----------|
| Hot keys en joins | Tasks de 2h mientras otras tardan 10s | Salting: distribuir keys artificialmente |
| Broadcast fallido | OOM en driver | Verificar tamaño, usar salting en su lugar |
| Skew no detectado | Job "lento" sin razón aparente | `diagnose_skew()` para identificar distribución |

### Large Datasets (`09_large_dataset_processing.py`)

| Problema | Síntoma | Solución |
|----------|---------|----------|
| Full scan de 500GB | Queries de 45+ min | Particionado + Z-ORDER |
| Full refresh diario | $4,500/mes en un solo job | Incremental con Change Data Feed |
| `collect()` en datos grandes | OOM | Procesar en Spark, solo `collect()` resultados agregados |

### Costos de Cluster (`10_cost_optimization.py`)

| Problema | Síntoma | Ahorro Estimado |
|----------|---------|-----------------|
| Cluster 24/7 con 35% uso | $14,400/mes | Auto-scaling → $4,000/mes (72%) |
| On-demand para batch | 3x costo necesario | Spot instances → 60-70% menos |
| i3.2xlarge con CPU 30% | Over-provisioned | r5.xlarge → 35% menos |

---

## Escalabilidad en Producción

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

| Componente | Herramienta Recomendada |
|------------|------------------------|
| Orquestación | Databricks Workflows / Airflow |
| Monitoreo | Datadog / CloudWatch / Databricks SQL Alerts |
| Secrets | Azure Key Vault / AWS Secrets Manager |
| CI/CD | GitHub Actions + Databricks CLI |
| Governance | Unity Catalog |
| Lineage | Unity Catalog / OpenLineage |

---

## Licencia

MIT License
