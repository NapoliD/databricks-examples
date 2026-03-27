# Databricks notebook source
# MAGIC %md
# MAGIC # Performance & Optimization
# MAGIC Técnicas de optimización para PySpark y Delta Lake.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, spark_partition_id, count
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Particionado

# COMMAND ----------

def demo_partitioning():
    """Demuestra impacto del particionado."""

    SILVER_PATH = "/mnt/silver/"
    orders = spark.read.format("delta").load(f"{SILVER_PATH}/orders")

    print("=== SIN PARTICIONADO ===")
    start = time.time()
    result = orders.filter(col("order_date") == "2024-01-20").count()
    print(f"Tiempo: {time.time() - start:.2f}s, Records: {result}")

    print("\n=== CON PARTICIONADO ===")
    # Escribir con partición
    (orders.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("order_date")
     .save("/tmp/orders_partitioned"))

    orders_partitioned = spark.read.format("delta").load("/tmp/orders_partitioned")

    start = time.time()
    result = orders_partitioned.filter(col("order_date") == "2024-01-20").count()
    print(f"Tiempo: {time.time() - start:.2f}s, Records: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Repartition vs Coalesce

# COMMAND ----------

def demo_repartition():
    """Demuestra diferencia entre repartition y coalesce."""

    df = spark.range(1000000)

    print(f"Particiones iniciales: {df.rdd.getNumPartitions()}")

    # Repartition: full shuffle, puede aumentar o disminuir particiones
    df_repartitioned = df.repartition(10)
    print(f"Después de repartition(10): {df_repartitioned.rdd.getNumPartitions()}")

    # Coalesce: no shuffle, solo puede reducir particiones
    df_coalesced = df.coalesce(2)
    print(f"Después de coalesce(2): {df_coalesced.rdd.getNumPartitions()}")

    # Repartition por columna (útil para joins posteriores)
    df_by_col = df.repartition(10, "id")
    print(f"Repartition por columna: {df_by_col.rdd.getNumPartitions()}")

    # Verificar distribución
    print("\nDistribución por partición:")
    (df_repartitioned
     .withColumn("partition_id", spark_partition_id())
     .groupBy("partition_id")
     .count()
     .orderBy("partition_id")
     .show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Broadcast Joins

# COMMAND ----------

def demo_broadcast_join():
    """Demuestra broadcast join vs shuffle join."""

    # Crear DataFrames de ejemplo
    large_df = spark.range(10000000).withColumnRenamed("id", "order_id")
    small_df = spark.range(100).withColumnRenamed("id", "customer_id")

    # Agregar columna para join
    large_df = large_df.withColumn("customer_id", col("order_id") % 100)

    print("=== SHUFFLE JOIN (default) ===")
    start = time.time()
    result = large_df.join(small_df, "customer_id")
    result.count()
    print(f"Tiempo: {time.time() - start:.2f}s")
    result.explain()

    print("\n=== BROADCAST JOIN ===")
    start = time.time()
    result_broadcast = large_df.join(broadcast(small_df), "customer_id")
    result_broadcast.count()
    print(f"Tiempo: {time.time() - start:.2f}s")
    result_broadcast.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Caching

# COMMAND ----------

def demo_caching():
    """Demuestra uso de cache y persist."""

    df = spark.range(10000000)

    # Sin cache
    print("=== SIN CACHE ===")
    start = time.time()
    df.filter(col("id") > 5000000).count()
    df.filter(col("id") < 5000000).count()
    print(f"Tiempo total: {time.time() - start:.2f}s")

    # Con cache
    print("\n=== CON CACHE ===")
    df_cached = df.cache()
    start = time.time()
    df_cached.filter(col("id") > 5000000).count()  # Primera acción: materializa cache
    df_cached.filter(col("id") < 5000000).count()  # Segunda acción: usa cache
    print(f"Tiempo total: {time.time() - start:.2f}s")

    # Liberar cache
    df_cached.unpersist()

    # Storage levels
    from pyspark import StorageLevel

    # MEMORY_ONLY: default
    # MEMORY_AND_DISK: si no entra en memoria, usa disco
    # DISK_ONLY: solo en disco
    # MEMORY_ONLY_SER: serializado (menos memoria, más CPU)

    df.persist(StorageLevel.MEMORY_AND_DISK)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Explain Plans

# COMMAND ----------

def demo_explain():
    """Demuestra análisis de explain plan."""

    df1 = spark.range(1000000).withColumnRenamed("id", "id1")
    df2 = spark.range(1000).withColumnRenamed("id", "id2")

    # Join simple
    result = df1.join(df2, df1.id1 == df2.id2)

    print("=== EXPLAIN SIMPLE ===")
    result.explain()

    print("\n=== EXPLAIN EXTENDED ===")
    result.explain(extended=True)

    print("\n=== EXPLAIN COST ===")
    result.explain(mode="cost")

    print("\n=== EXPLAIN FORMATTED ===")
    result.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Delta Lake Optimizations

# COMMAND ----------

def optimize_delta_table(table_path):
    """Optimizaciones específicas de Delta Lake."""

    # OPTIMIZE: compacta archivos pequeños
    spark.sql(f"OPTIMIZE delta.`{table_path}`")

    # OPTIMIZE con Z-ORDER: co-localiza datos para queries comunes
    spark.sql(f"OPTIMIZE delta.`{table_path}` ZORDER BY (order_date, customer_id)")

    # VACUUM: elimina archivos antiguos (default: 7 días)
    spark.sql(f"VACUUM delta.`{table_path}` RETAIN 168 HOURS")

    # Ver estadísticas
    spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. AQE - Adaptive Query Execution

# COMMAND ----------

def demo_aqe():
    """Demuestra Adaptive Query Execution."""

    # Verificar si AQE está habilitado
    print(f"AQE enabled: {spark.conf.get('spark.sql.adaptive.enabled')}")

    # Habilitar AQE
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

    # AQE ajusta automáticamente:
    # - Número de particiones de shuffle
    # - Manejo de skew en joins
    # - Conversión de sort-merge join a broadcast join

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices

# COMMAND ----------

def best_practices_example():
    """Ejemplos de buenas prácticas."""

    # 1. Filtrar temprano
    df = spark.read.format("delta").load("/mnt/silver/orders")

    # MAL: filtrar después de join
    # df.join(other_df).filter(...)

    # BIEN: filtrar antes de join
    df_filtered = df.filter(col("status") == "completed")
    # df_filtered.join(other_df)

    # 2. Seleccionar solo columnas necesarias
    # MAL: select("*")
    # BIEN: select("order_id", "amount", "date")
    df_selected = df.select("order_id", "total_amount", "order_date")

    # 3. Evitar UDFs cuando sea posible
    # MAL: usar UDF para operaciones que Spark tiene built-in
    # BIEN: usar funciones de pyspark.sql.functions

    # 4. Usar tipos correctos
    df_typed = df.withColumn("quantity", col("quantity").cast("integer"))

    # 5. Particionar por cardinalidad adecuada
    # - No particionar por columnas de alta cardinalidad (ej: order_id)
    # - Particionar por columnas usadas frecuentemente en filtros

    # 6. Monitorear skew
    (df.withColumn("partition_id", spark_partition_id())
     .groupBy("partition_id")
     .count()
     .orderBy(col("count").desc())
     .show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Métricas de Performance

# COMMAND ----------

def analyze_performance():
    """Analiza métricas de performance."""

    # Spark UI metrics
    print("Jobs:", spark.sparkContext.statusTracker().getActiveJobIds())

    # Configuraciones importantes
    configs = [
        "spark.sql.shuffle.partitions",
        "spark.sql.adaptive.enabled",
        "spark.sql.autoBroadcastJoinThreshold",
        "spark.default.parallelism",
        "spark.executor.memory",
        "spark.driver.memory"
    ]

    print("\nConfiguraciones actuales:")
    for config in configs:
        try:
            value = spark.conf.get(config)
            print(f"  {config}: {value}")
        except:
            print(f"  {config}: not set")

# COMMAND ----------

# Ejecutar demos
# demo_partitioning()
# demo_repartition()
# demo_broadcast_join()
# demo_caching()
# demo_explain()
# analyze_performance()
