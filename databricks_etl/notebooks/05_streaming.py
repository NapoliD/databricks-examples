# Databricks notebook source
# MAGIC %md
# MAGIC # Spark Structured Streaming
# MAGIC Procesamiento de datos en tiempo real con PySpark.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, current_timestamp,
    sum as spark_sum, count, avg, max as spark_max,
    expr, to_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, IntegerType, TimestampType
)

# COMMAND ----------

STREAM_INPUT = "/mnt/streaming/input/"
STREAM_OUTPUT = "/mnt/streaming/output/"
CHECKPOINT_PATH = "/mnt/streaming/checkpoints/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema de Eventos

# COMMAND ----------

event_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("event_type", StringType(), False),
    StructField("user_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("timestamp", StringType(), True),
    StructField("metadata", StringType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lectura de Stream

# COMMAND ----------

def read_json_stream(input_path, schema):
    """Lee stream de archivos JSON."""

    return (spark.readStream
        .format("json")
        .schema(schema)
        .option("maxFilesPerTrigger", 1)  # Procesar 1 archivo por micro-batch
        .load(input_path))


def read_kafka_stream(bootstrap_servers, topic):
    """Lee stream desde Kafka."""

    return (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
        .select(
            col("key").cast("string"),
            from_json(col("value").cast("string"), event_schema).alias("data"),
            col("timestamp")
        )
        .select("data.*", "timestamp"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformaciones de Stream

# COMMAND ----------

def transform_events(stream_df):
    """Aplica transformaciones al stream de eventos."""

    return (stream_df
        .withColumn("event_timestamp", to_timestamp(col("timestamp")))
        .withColumn("processing_time", current_timestamp())
        .filter(col("event_type").isNotNull())
        .filter(col("amount") > 0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Window Aggregations

# COMMAND ----------

def aggregate_by_window(stream_df, window_duration="5 minutes", slide_duration="1 minute"):
    """
    Agrega eventos por ventana de tiempo.
    - window_duration: tamaño de la ventana
    - slide_duration: cada cuánto se genera una nueva ventana
    """

    return (stream_df
        .withWatermark("event_timestamp", "10 minutes")  # Permite late data hasta 10 min
        .groupBy(
            window(col("event_timestamp"), window_duration, slide_duration),
            col("event_type")
        )
        .agg(
            count("event_id").alias("event_count"),
            spark_sum("amount").alias("total_amount"),
            avg("amount").alias("avg_amount"),
            spark_sum("quantity").alias("total_quantity")
        ))


def aggregate_by_tumbling_window(stream_df, window_duration="1 minute"):
    """Agrega con tumbling window (sin overlap)."""

    return (stream_df
        .withWatermark("event_timestamp", "5 minutes")
        .groupBy(
            window(col("event_timestamp"), window_duration),
            col("product_id")
        )
        .agg(
            count("*").alias("events"),
            spark_sum("amount").alias("revenue"),
            spark_sum("quantity").alias("units_sold")
        ))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escritura de Stream

# COMMAND ----------

def write_stream_to_delta(stream_df, output_path, checkpoint_path, output_mode="append"):
    """Escribe stream a Delta Lake."""

    return (stream_df.writeStream
        .format("delta")
        .outputMode(output_mode)
        .option("checkpointLocation", checkpoint_path)
        .start(output_path))


def write_stream_to_console(stream_df, output_mode="complete"):
    """Escribe stream a consola (para debug)."""

    return (stream_df.writeStream
        .format("console")
        .outputMode(output_mode)
        .option("truncate", False)
        .start())


def write_stream_to_kafka(stream_df, bootstrap_servers, topic, checkpoint_path):
    """Escribe stream a Kafka."""

    return (stream_df
        .selectExpr("CAST(event_id AS STRING) AS key", "to_json(struct(*)) AS value")
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", topic)
        .option("checkpointLocation", checkpoint_path)
        .start())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejemplo: Pipeline Completo

# COMMAND ----------

def run_streaming_pipeline():
    """Ejecuta pipeline de streaming completo."""

    # 1. Leer stream
    raw_stream = read_json_stream(STREAM_INPUT, event_schema)

    # 2. Transformar
    transformed_stream = transform_events(raw_stream)

    # 3. Agregar por ventana
    windowed_stream = aggregate_by_window(
        transformed_stream,
        window_duration="5 minutes",
        slide_duration="1 minute"
    )

    # 4. Escribir a Delta
    query = write_stream_to_delta(
        windowed_stream,
        output_path=f"{STREAM_OUTPUT}/windowed_events",
        checkpoint_path=f"{CHECKPOINT_PATH}/windowed_events",
        output_mode="append"
    )

    return query

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejemplo: Real-time Revenue

# COMMAND ----------

def run_realtime_revenue():
    """Pipeline de revenue en tiempo real."""

    # Leer eventos de compra
    purchases = (read_json_stream(f"{STREAM_INPUT}/purchases/", event_schema)
        .filter(col("event_type") == "purchase"))

    # Agregar revenue por minuto
    revenue_per_minute = (purchases
        .withWatermark("event_timestamp", "2 minutes")
        .groupBy(
            window(col("event_timestamp"), "1 minute")
        )
        .agg(
            spark_sum("amount").alias("revenue"),
            count("event_id").alias("transactions"),
            avg("amount").alias("avg_transaction")
        ))

    # Escribir
    query = (revenue_per_minute.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_PATH}/realtime_revenue")
        .start(f"{STREAM_OUTPUT}/realtime_revenue"))

    return query

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoreo de Queries

# COMMAND ----------

def monitor_streaming_queries():
    """Muestra estado de todas las queries de streaming activas."""

    for query in spark.streams.active:
        print(f"\nQuery: {query.name}")
        print(f"  Status: {query.status}")
        print(f"  Recent Progress:")
        for progress in query.recentProgress[-3:]:
            print(f"    - Batch: {progress.get('batchId')}, "
                  f"Input Rows: {progress.get('numInputRows')}, "
                  f"Processing Time: {progress.get('batchDuration')} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecutar (descomentar para usar)

# COMMAND ----------

# query = run_streaming_pipeline()
# query.awaitTermination()  # Esperar indefinidamente

# Para detener:
# query.stop()
