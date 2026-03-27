# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer - Raw Data Ingestion
# MAGIC Este notebook ingesta datos crudos desde múltiples fuentes y los almacena en formato Delta Lake.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, input_file_name
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuración

# COMMAND ----------

# Rutas de datos
RAW_DATA_PATH = "/mnt/raw/"
BRONZE_PATH = "/mnt/bronze/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schemas

# COMMAND ----------

orders_schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("product_id", StringType(), False),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("order_date", StringType(), True),
    StructField("status", StringType(), True)
])

customers_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("country", StringType(), True),
    StructField("registration_date", StringType(), True)
])

payments_schema = StructType([
    StructField("payment_id", StringType(), False),
    StructField("order_id", StringType(), False),
    StructField("amount", DoubleType(), True),
    StructField("payment_method", StringType(), True),
    StructField("payment_date", StringType(), True),
    StructField("status", StringType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones de Ingesta

# COMMAND ----------

def ingest_csv_to_bronze(source_path: str, target_path: str, schema: StructType, table_name: str):
    """
    Ingesta archivos CSV a la capa Bronze con metadatos de auditoría.
    """
    df = (spark.read
          .format("csv")
          .option("header", "true")
          .option("inferSchema", "false")
          .schema(schema)
          .load(source_path))

    # Agregar metadatos de auditoría
    df_with_metadata = (df
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_batch_id", lit(dbutils.widgets.get("batch_id") if "batch_id" in dbutils.widgets.getAll() else "manual")))

    # Escribir en Delta con merge schema
    (df_with_metadata.write
     .format("delta")
     .mode("append")
     .option("mergeSchema", "true")
     .save(f"{target_path}/{table_name}"))

    print(f"Ingested {df_with_metadata.count()} records to {table_name}")
    return df_with_metadata

# COMMAND ----------

def ingest_json_to_bronze(source_path: str, target_path: str, schema: StructType, table_name: str):
    """
    Ingesta archivos JSON a la capa Bronze con metadatos de auditoría.
    """
    df = (spark.read
          .format("json")
          .schema(schema)
          .option("multiLine", "true")
          .load(source_path))

    df_with_metadata = (df
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_batch_id", lit(dbutils.widgets.get("batch_id") if "batch_id" in dbutils.widgets.getAll() else "manual")))

    (df_with_metadata.write
     .format("delta")
     .mode("append")
     .option("mergeSchema", "true")
     .save(f"{target_path}/{table_name}"))

    print(f"Ingested {df_with_metadata.count()} records to {table_name}")
    return df_with_metadata

# COMMAND ----------

def ingest_parquet_to_bronze(source_path: str, target_path: str, schema: StructType, table_name: str):
    """
    Ingesta archivos Parquet a la capa Bronze con metadatos de auditoría.

    Parquet es ideal para grandes volúmenes de datos debido a:
    - Formato columnar: mejor compresión y lectura selectiva de columnas
    - Predicate pushdown: filtros se aplican a nivel de archivo
    - Schema embebido: no requiere definir schema manualmente (opcional)
    - Compatibilidad nativa con Spark: mejor rendimiento que CSV/JSON
    """
    # Si se proporciona schema, usarlo; sino inferir del Parquet
    if schema:
        df = (spark.read
              .format("parquet")
              .schema(schema)
              .load(source_path))
    else:
        df = (spark.read
              .format("parquet")
              .load(source_path))

    df_with_metadata = (df
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_file", input_file_name())
        .withColumn("_batch_id", lit(dbutils.widgets.get("batch_id") if "batch_id" in dbutils.widgets.getAll() else "manual")))

    (df_with_metadata.write
     .format("delta")
     .mode("append")
     .option("mergeSchema", "true")
     .save(f"{target_path}/{table_name}"))

    print(f"Ingested {df_with_metadata.count()} records to {table_name}")
    return df_with_metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecución de Ingesta

# COMMAND ----------

# Ingesta de órdenes
orders_bronze = ingest_csv_to_bronze(
    source_path=f"{RAW_DATA_PATH}/orders/",
    target_path=BRONZE_PATH,
    schema=orders_schema,
    table_name="orders"
)

# COMMAND ----------

# Ingesta de clientes
customers_bronze = ingest_csv_to_bronze(
    source_path=f"{RAW_DATA_PATH}/customers/",
    target_path=BRONZE_PATH,
    schema=customers_schema,
    table_name="customers"
)

# COMMAND ----------

# Ingesta de pagos (JSON)
payments_bronze = ingest_json_to_bronze(
    source_path=f"{RAW_DATA_PATH}/payments/",
    target_path=BRONZE_PATH,
    schema=payments_schema,
    table_name="payments"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ingesta de Parquet (Ejemplo)
# MAGIC Parquet es el formato recomendado para grandes volúmenes de datos por su eficiencia en compresión y lectura.

# COMMAND ----------

# Ejemplo: Ingesta de datos históricos en Parquet (descomentar para usar)
# Este formato es ideal para datasets de millones de registros

# historical_orders_bronze = ingest_parquet_to_bronze(
#     source_path=f"{RAW_DATA_PATH}/historical_orders/",
#     target_path=BRONZE_PATH,
#     schema=None,  # Parquet tiene schema embebido, no es necesario definirlo
#     table_name="historical_orders"
# )

# Ejemplo con schema explícito para validación estricta
# transactions_bronze = ingest_parquet_to_bronze(
#     source_path=f"{RAW_DATA_PATH}/transactions/",
#     target_path=BRONZE_PATH,
#     schema=transactions_schema,
#     table_name="transactions"
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validación de Ingesta

# COMMAND ----------

def validate_bronze_table(table_path: str, table_name: str):
    """Valida que la tabla Bronze se haya creado correctamente."""
    df = spark.read.format("delta").load(table_path)

    print(f"\n=== {table_name} ===")
    print(f"Total records: {df.count()}")
    print(f"Schema:")
    df.printSchema()
    print(f"Sample data:")
    df.show(5, truncate=False)

    return df

# COMMAND ----------

validate_bronze_table(f"{BRONZE_PATH}/orders", "Orders Bronze")
validate_bronze_table(f"{BRONZE_PATH}/customers", "Customers Bronze")
validate_bronze_table(f"{BRONZE_PATH}/payments", "Payments Bronze")
