# Delta Live Tables - Pipeline E-commerce

> **Importante:** Todos los datos en este proyecto son **FICTICIOS** y fueron creados únicamente con fines demostrativos y educativos. Este proyecto muestra mis habilidades con Delta Live Tables en Databricks.

Construí este proyecto para demostrar pipelines de datos declarativos usando Delta Live Tables (DLT). DLT representa la evolución de la ingeniería de datos en Databricks - pasando de código Spark imperativo a definiciones de pipeline declarativas.

---

## Por Qué Construí Este Proyecto

Después de construir pipelines ETL tradicionales con Spark (ver `databricks_etl/`), quería mostrar que entiendo el enfoque moderno y declarativo con DLT. Este proyecto demuestra:

- **Tablas Declarativas**: Definir QUÉ querés, no CÓMO construirlo
- **Calidad de Datos Integrada**: Expectations que rastrean métricas automáticamente
- **Manejo de CDC**: `apply_changes()` para sincronización de datos en tiempo real
- **Python y SQL**: Mostrando versatilidad en desarrollo de pipelines

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE DELTA LIVE TABLES                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CAPA BRONZE (Python)                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Órdenes │  │ Clientes │  │  Pagos   │  │ Productos│            │   │
│  │  │  (CSV)   │  │  (CSV)   │  │  (JSON)  │  │(Streaming)│           │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘            │   │
│  │       │             │             │             │                    │   │
│  │       ▼             ▼             ▼             ▼                    │   │
│  │  @dlt.expect  @dlt.expect  @dlt.expect_or_drop  cloudFiles          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CAPA SILVER (SQL)                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │   │
│  │  │  silver_orders   │  │ silver_customers │  │ silver_payments │   │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘   │   │
│  │           │                     │                      │            │   │
│  │           └──────────┬──────────┴──────────────────────┘            │   │
│  │                      ▼                                              │   │
│  │           ┌──────────────────────┐     ┌─────────────────┐         │   │
│  │           │ silver_enriched_orders│     │ silver_quarantine│        │   │
│  │           │   (Unido + Limpio)    │     │  (Filas fallidas)│        │   │
│  │           └──────────┬────────────┘     └─────────────────┘         │   │
│  └──────────────────────┼──────────────────────────────────────────────┘   │
│                         │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CAPA GOLD (Python)                           │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │  Revenue   │ │ Performance│ │ Performance│ │ Customer   │       │   │
│  │  │   Diario   │ │   País     │ │  Producto  │ │    360     │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         PIPELINE CDC (Python)                        │   │
│  │                                                                      │   │
│  │  cdc_raw ──► apply_changes() ──► customers_scd1 (estado actual)     │   │
│  │                              ──► customers_scd2 (historial completo) │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
databricks_dlt/
├── pipelines/
│   ├── 01_bronze_ingestion.py        # Datos crudos con expectations
│   ├── 02_silver_transformations.sql # Limpieza basada en SQL
│   ├── 03_gold_aggregations.py       # Métricas de negocio
│   └── 04_cdc_pipeline.py            # CDC con apply_changes()
│
├── config/
│   └── pipeline_settings.json        # Configuración del pipeline DLT
│
├── README.md                         # Documentación en inglés
└── README_ES.md                      # Documentación en español
```

---

## Componentes del Pipeline

### 01. Ingesta Bronze (Python)

Implementé ingesta de datos crudos con expectations de calidad:

| Tabla | Fuente | Expectations |
|-------|--------|--------------|
| `bronze_orders` | CSV | NOT NULL, quantity > 0 |
| `bronze_customers` | CSV | Formato de email válido |
| `bronze_payments` | JSON | Amount > 0 |
| `bronze_products_streaming` | Auto Loader | Price > 0 |

**Características Clave:**
- `@dlt.expect`: Advertir en violación
- `@dlt.expect_or_drop`: Eliminar filas inválidas
- Auto Loader para ingesta streaming
- Columnas de auditoría para trazabilidad

### 02. Transformaciones Silver (SQL)

Elegí SQL para la capa Silver para demostrar versatilidad:

```sql
CREATE OR REFRESH LIVE TABLE silver_orders
COMMENT "Órdenes limpias. DATOS FICTICIOS."
AS
SELECT
  order_id,
  CAST(quantity AS INT) AS quantity,
  TRIM(UPPER(status)) AS order_status
FROM LIVE.bronze_orders
```

**Características SQL DLT:**
- `CREATE OR REFRESH LIVE TABLE`
- `CONSTRAINT ... EXPECT` para calidad
- `ON VIOLATION DROP ROW` para enforcement
- Joins SQL estándar con prefijo `LIVE.`

### 03. Agregaciones Gold (Python)

Tablas listas para negocio con KPIs:

| Tabla | Propósito |
|-------|-----------|
| `gold_daily_revenue` | Análisis de revenue por tiempo |
| `gold_country_performance` | Performance geográfico con rankings |
| `gold_product_performance` | Analytics de productos |
| `gold_customer_360` | Valor de cliente y segmentación |
| `gold_payment_analytics` | Análisis de métodos de pago |
| `gold_executive_summary` | KPIs de alto nivel |

### 04. Pipeline CDC (Python)

Change Data Capture con `apply_changes()`:

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

**Características CDC:**
- SCD Tipo 1 (sobrescribir estado actual)
- SCD Tipo 2 (historial completo con timestamps)
- Manejo automático de deletes
- Manejo de eventos fuera de orden

---

## Expectations de Calidad de Datos

| Capa | Tipo de Expectation | Acción |
|------|---------------------|--------|
| Bronze | `@dlt.expect` | Advertir y rastrear |
| Bronze | `@dlt.expect_or_drop` | Eliminar filas inválidas |
| Silver | `CONSTRAINT EXPECT` | Validación basada en SQL |
| Silver | Tabla quarantine | Capturar fallos |

---

## Tecnologías Utilizadas

| Tecnología | Propósito |
|------------|-----------|
| **Delta Live Tables** | Framework de pipelines declarativos |
| **Python DLT** | Transformaciones complejas |
| **SQL DLT** | Transformaciones simples |
| **Auto Loader** | Ingesta de archivos streaming |
| **apply_changes()** | Procesamiento CDC |
| **Expectations** | Calidad de datos |
| **Unity Catalog** | Governance (opcional) |

---

## Cómo Ejecutar

### En Databricks

1. Importar este repositorio a tu workspace de Databricks
2. Crear un nuevo pipeline DLT:
   - Ir a **Workflows** → **Delta Live Tables** → **Create Pipeline**
   - Usar la configuración de `config/pipeline_settings.json`
3. Agregar los notebooks de la carpeta `pipelines/`
4. Configurar paths de origen y destino
5. Iniciar el pipeline

### Modos del Pipeline

| Modo | Caso de Uso |
|------|-------------|
| **Development** | Testing con refresh completo |
| **Production** | Procesamiento incremental |
| **Continuous** | Streaming en tiempo real |

---

## DLT vs Spark Tradicional

| Aspecto | Spark Tradicional | Delta Live Tables |
|---------|-------------------|-------------------|
| **Enfoque** | Imperativo (CÓMO) | Declarativo (QUÉ) |
| **Dependencias** | Gestión manual | Resolución automática |
| **Calidad de Datos** | Código custom | Expectations integrados |
| **CDC** | Lógica MERGE compleja | `apply_changes()` |
| **Orquestación** | Externa (Airflow, etc.) | Integrada |
| **Lineage** | Tracking manual | Automático |

---

## Aprendizajes Clave

Construir este proyecto me enseñó:

1. **Declarativo es Poderoso**: Enfocarse en lógica de negocio, no en plomería
2. **Expectations son Esenciales**: Tracking de calidad integrado ahorra tiempo
3. **SQL y Python Juntos**: Usar cada uno donde tiene sentido
4. **CDC Simplificado**: `apply_changes()` maneja la complejidad automáticamente
5. **Streaming es Natural**: Auto Loader hace streaming tan fácil como batch

---

## Qué Agregaría en Producción

- **Integración Unity Catalog**: Para governance de datos
- **Materialized Views**: Para dashboards en tiempo real
- **Fuentes Event Hubs/Kafka**: Para streaming verdadero
- **Alertas y Monitoreo**: Notificaciones personalizadas
- **Múltiples Ambientes**: Pipelines dev, staging, producción

---

## Sobre los Datos

> **Todos los datos en este proyecto son completamente FICTICIOS.**

Creé datos sintéticos de e-commerce (órdenes, clientes, pagos) para demostrar las capacidades de DLT. No se utilizó información real de clientes o negocios.

---

## Proyectos Relacionados

- `databricks_etl/` - ETL tradicional con Spark (Arquitectura Medallion)
- `databricks_ml/` - MLflow para operaciones de ML

---

## Licencia

MIT License
