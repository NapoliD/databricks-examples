# Databricks notebook source
# MAGIC %md
# MAGIC # CDC - Change Data Capture con Delta Lake
# MAGIC Pipeline incremental con MERGE para manejar inserts, updates y deletes.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, lit, when,
    max as spark_max, hash, concat_ws
)
from delta.tables import DeltaTable

# COMMAND ----------

SILVER_PATH = "/mnt/silver/"
CDC_PATH = "/mnt/cdc/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulación de Datos CDC
# MAGIC En producción estos datos vendrían de Kafka, Kinesis, o archivos CDC.

# COMMAND ----------

def generate_cdc_batch():
    """Genera un batch de cambios simulados."""

    cdc_data = [
        # Nuevos registros (INSERT)
        ("ORD026", "CUST016", "PROD001", 3, 29.99, "2024-01-28", "pending", "I"),
        ("ORD027", "CUST017", "PROD002", 1, 149.99, "2024-01-28", "completed", "I"),
        ("ORD028", "CUST018", "PROD005", 2, 199.99, "2024-01-29", "pending", "I"),

        # Actualizaciones (UPDATE)
        ("ORD004", "CUST003", "PROD001", 1, 29.99, "2024-01-16", "completed", "U"),  # pending -> completed
        ("ORD010", "CUST001", "PROD004", 1, 79.99, "2024-01-19", "completed", "U"),  # pending -> completed
        ("ORD014", "CUST009", "PROD003", 2, 9.99, "2024-01-21", "cancelled", "U"),   # pending -> cancelled

        # Eliminaciones (DELETE)
        ("ORD007", "CUST005", "PROD002", 2, 149.99, "2024-01-18", "deleted", "D"),   # registro eliminado
    ]

    schema = ["order_id", "customer_id", "product_id", "quantity", "unit_price", "order_date", "status", "operation"]

    return spark.createDataFrame(cdc_data, schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MERGE: Upsert Pattern

# COMMAND ----------

def apply_cdc_merge(source_df, target_path, key_column):
    """
    Aplica CDC usando MERGE de Delta Lake.

    Operaciones:
    - I (Insert): Inserta nuevo registro
    - U (Update): Actualiza registro existente
    - D (Delete): Elimina registro (soft delete o hard delete)
    """

    # Verificar si la tabla target existe
    if DeltaTable.isDeltaTable(spark, target_path):
        target_table = DeltaTable.forPath(spark, target_path)

        # Preparar source con timestamp
        source_prepared = source_df.withColumn("_cdc_timestamp", current_timestamp())

        # MERGE
        (target_table.alias("target")
         .merge(
             source_prepared.alias("source"),
             f"target.{key_column} = source.{key_column}"
         )
         # DELETE cuando operation = 'D'
         .whenMatchedDelete(
             condition="source.operation = 'D'"
         )
         # UPDATE cuando operation = 'U'
         .whenMatchedUpdate(
             condition="source.operation = 'U'",
             set={
                 "customer_id": "source.customer_id",
                 "product_id": "source.product_id",
                 "quantity": "source.quantity",
                 "unit_price": "source.unit_price",
                 "order_date": "source.order_date",
                 "status": "source.status",
                 "_last_updated": "source._cdc_timestamp"
             }
         )
         # INSERT cuando operation = 'I' y no existe
         .whenNotMatchedInsert(
             condition="source.operation = 'I'",
             values={
                 "order_id": "source.order_id",
                 "customer_id": "source.customer_id",
                 "product_id": "source.product_id",
                 "quantity": "source.quantity",
                 "unit_price": "source.unit_price",
                 "order_date": "source.order_date",
                 "status": "source.status",
                 "_created_at": "source._cdc_timestamp",
                 "_last_updated": "source._cdc_timestamp"
             }
         )
         .execute())

        print(f"CDC MERGE completed successfully")

    else:
        # Primera carga - crear tabla
        print(f"Creating initial table at {target_path}")
        initial_df = (source_df
            .filter(col("operation") == "I")
            .drop("operation")
            .withColumn("_created_at", current_timestamp())
            .withColumn("_last_updated", current_timestamp()))

        (initial_df.write
         .format("delta")
         .mode("overwrite")
         .save(target_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Soft Delete Pattern

# COMMAND ----------

def apply_cdc_soft_delete(source_df, target_path, key_column):
    """
    CDC con Soft Delete - marca registros como eliminados en lugar de borrarlos.
    """

    if DeltaTable.isDeltaTable(spark, target_path):
        target_table = DeltaTable.forPath(spark, target_path)

        source_prepared = source_df.withColumn("_cdc_timestamp", current_timestamp())

        (target_table.alias("target")
         .merge(
             source_prepared.alias("source"),
             f"target.{key_column} = source.{key_column}"
         )
         # Soft delete: marcar como deleted
         .whenMatchedUpdate(
             condition="source.operation = 'D'",
             set={
                 "status": lit("deleted"),
                 "_is_deleted": lit(True),
                 "_deleted_at": "source._cdc_timestamp",
                 "_last_updated": "source._cdc_timestamp"
             }
         )
         .whenMatchedUpdate(
             condition="source.operation = 'U'",
             set={
                 "customer_id": "source.customer_id",
                 "product_id": "source.product_id",
                 "quantity": "source.quantity",
                 "unit_price": "source.unit_price",
                 "order_date": "source.order_date",
                 "status": "source.status",
                 "_last_updated": "source._cdc_timestamp"
             }
         )
         .whenNotMatchedInsert(
             condition="source.operation = 'I'",
             values={
                 "order_id": "source.order_id",
                 "customer_id": "source.customer_id",
                 "product_id": "source.product_id",
                 "quantity": "source.quantity",
                 "unit_price": "source.unit_price",
                 "order_date": "source.order_date",
                 "status": "source.status",
                 "_is_deleted": lit(False),
                 "_created_at": "source._cdc_timestamp",
                 "_last_updated": "source._cdc_timestamp"
             }
         )
         .execute())

        print("CDC with Soft Delete completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## SCD Type 2 - Slowly Changing Dimensions

# COMMAND ----------

def apply_scd_type2(source_df, target_path, key_column, track_columns):
    """
    Implementa SCD Type 2 para mantener historial de cambios.
    """

    if DeltaTable.isDeltaTable(spark, target_path):
        target_table = DeltaTable.forPath(spark, target_path)
        target_df = spark.read.format("delta").load(target_path)

        # Crear hash de columnas a trackear
        source_with_hash = source_df.withColumn(
            "_row_hash",
            hash(concat_ws("||", *[col(c) for c in track_columns]))
        )

        # Obtener registros actuales (is_current = true)
        current_records = target_df.filter(col("_is_current") == True)

        # Encontrar cambios
        changes = (source_with_hash.alias("source")
            .join(
                current_records.alias("target"),
                col(f"source.{key_column}") == col(f"target.{key_column}"),
                "left"
            )
            .filter(
                (col("target._row_hash").isNull()) |  # Nuevo registro
                (col("source._row_hash") != col("target._row_hash"))  # Cambio
            )
            .select("source.*"))

        if changes.count() > 0:
            # Cerrar registros actuales que cambiaron
            (target_table.alias("target")
             .merge(
                 changes.alias("source"),
                 f"target.{key_column} = source.{key_column} AND target._is_current = true"
             )
             .whenMatchedUpdate(
                 set={
                     "_is_current": lit(False),
                     "_end_date": current_timestamp()
                 }
             )
             .execute())

            # Insertar nuevas versiones
            new_versions = (changes
                .withColumn("_is_current", lit(True))
                .withColumn("_start_date", current_timestamp())
                .withColumn("_end_date", lit(None).cast("timestamp"))
                .withColumn("_version", lit(1)))  # En producción, calcular versión

            new_versions.write.format("delta").mode("append").save(target_path)

            print(f"SCD Type 2: {changes.count()} records updated")
        else:
            print("SCD Type 2: No changes detected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecución de CDC

# COMMAND ----------

# Generar batch de cambios
cdc_batch = generate_cdc_batch()
print("CDC Batch:")
cdc_batch.show(truncate=False)

# COMMAND ----------

# Aplicar CDC
apply_cdc_merge(
    source_df=cdc_batch,
    target_path=f"{CDC_PATH}/orders_cdc",
    key_column="order_id"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificación de Historial Delta

# COMMAND ----------

# Ver historial de cambios
history = spark.sql(f"DESCRIBE HISTORY delta.`{CDC_PATH}/orders_cdc`")
history.select("version", "timestamp", "operation", "operationMetrics").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Travel

# COMMAND ----------

# Leer versión anterior
# df_v0 = spark.read.format("delta").option("versionAsOf", 0).load(f"{CDC_PATH}/orders_cdc")

# Leer estado en timestamp específico
# df_timestamp = spark.read.format("delta").option("timestampAsOf", "2024-01-28").load(f"{CDC_PATH}/orders_cdc")
