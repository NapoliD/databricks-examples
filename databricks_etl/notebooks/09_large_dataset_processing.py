# Databricks notebook source
# MAGIC %md
# MAGIC # Procesamiento de Datasets Grandes (+100GB)
# MAGIC Estrategias para procesar eficientemente datasets que no caben en memoria.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, year, month, dayofmonth, current_timestamp,
    sum as spark_sum, count, avg, max as spark_max,
    min as spark_min, lit, hash, abs as spark_abs
)
from delta.tables import DeltaTable
from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuración para Large Datasets

# COMMAND ----------

# Configuraciones recomendadas para datasets grandes
def configure_for_large_dataset(spark, dataset_size_gb):
    """
    Configura Spark para procesar datasets grandes.

    Args:
        spark: SparkSession
        dataset_size_gb: Tamaño estimado del dataset en GB
    """
    # Particiones de shuffle basadas en tamaño
    # Regla: ~128MB por partición
    num_partitions = max(200, int(dataset_size_gb * 1024 / 128))

    spark.conf.set("spark.sql.shuffle.partitions", num_partitions)

    # Tamaño máximo de partición de archivos
    spark.conf.set("spark.sql.files.maxPartitionBytes", "128m")

    # AQE para optimización automática
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

    # Broadcast threshold - reducir para datasets grandes
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10m")

    # Spill to disk cuando sea necesario
    spark.conf.set("spark.memory.fraction", "0.6")
    spark.conf.set("spark.memory.storageFraction", "0.5")

    print(f"Configurado para ~{dataset_size_gb}GB:")
    print(f"  - Shuffle partitions: {num_partitions}")
    print(f"  - Max partition bytes: 128MB")
    print(f"  - AQE: enabled")

# Configurar para 500GB
configure_for_large_dataset(spark, 500)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estrategia 1: Particionado Inteligente
# MAGIC
# MAGIC El particionado correcto puede reducir el escaneo de datos en 90%+.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análisis de Patrones de Query
# MAGIC
# MAGIC Antes de particionar, analizar cómo se consultan los datos.

# COMMAND ----------

def analyze_query_patterns(table_name):
    """
    Analiza patrones de query en los logs de Spark.
    En producción, usar Query History de Databricks.
    """
    # Simular análisis de patrones
    patterns = {
        "filters_by_date": 0.90,      # 90% de queries filtran por fecha
        "filters_by_country": 0.60,   # 60% también filtran por país
        "filters_by_customer": 0.30,  # 30% filtran por customer_id
        "full_scans": 0.05            # Solo 5% son full scans
    }

    print(f"Patrones de query para {table_name}:")
    for pattern, freq in patterns.items():
        print(f"  {pattern}: {freq*100:.0f}%")

    # Recomendación
    print("\nRecomendación de particionado:")
    print("  1. Particionar por: year, month (granularidad mensual)")
    print("  2. Z-ORDER por: country, customer_id")

    return patterns

analyze_query_patterns("gold.transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implementación de Particionado

# COMMAND ----------

def write_with_optimal_partitioning(df, path, partition_cols, zorder_cols=None):
    """
    Escribe DataFrame con particionado óptimo para queries.

    Args:
        df: DataFrame a escribir
        path: Ruta de destino
        partition_cols: Columnas de partición (ej: ["year", "month"])
        zorder_cols: Columnas para Z-ORDER (ej: ["country", "customer_id"])
    """
    # Agregar columnas de partición si no existen
    df_with_parts = df
    if "year" in partition_cols and "year" not in df.columns:
        df_with_parts = df_with_parts.withColumn("year", year("event_date"))
    if "month" in partition_cols and "month" not in df.columns:
        df_with_parts = df_with_parts.withColumn("month", month("event_date"))

    # Escribir con particionado
    (df_with_parts.write
     .format("delta")
     .mode("overwrite")
     .partitionBy(*partition_cols)
     .option("maxRecordsPerFile", 1000000)  # ~100MB por archivo
     .save(path))

    print(f"Datos escritos en {path}")
    print(f"Particionado por: {partition_cols}")

    # Aplicar Z-ORDER si se especifica
    if zorder_cols:
        zorder_statement = f"OPTIMIZE delta.`{path}` ZORDER BY ({', '.join(zorder_cols)})"
        print(f"Aplicando Z-ORDER por: {zorder_cols}")
        spark.sql(zorder_statement)

    return path

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificar Efectividad del Particionado

# COMMAND ----------

def measure_partition_pruning(spark, path, filter_condition):
    """
    Mide cuántas particiones se escanean con un filtro específico.
    """
    df = spark.read.format("delta").load(path)

    # Plan sin filtro
    df_full = df.select("*")
    plan_full = df_full._jdf.queryExecution().executedPlan().toString()

    # Plan con filtro
    df_filtered = df.filter(filter_condition)
    plan_filtered = df_filtered._jdf.queryExecution().executedPlan().toString()

    # Contar particiones en el plan
    print(f"Filtro: {filter_condition}")
    print(f"Verificar 'PartitionFilters' en el plan para confirmar pruning:")
    df_filtered.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estrategia 2: Procesamiento Incremental
# MAGIC
# MAGIC Nunca reprocesar todo cuando puedes procesar solo los cambios.

# COMMAND ----------

class IncrementalProcessor:
    """
    Procesador incremental usando Delta Lake Change Data Feed.
    """

    def __init__(self, source_path, target_path, checkpoint_path):
        self.source_path = source_path
        self.target_path = target_path
        self.checkpoint_path = checkpoint_path

    def get_last_processed_version(self):
        """Obtiene la última versión procesada del checkpoint."""
        try:
            checkpoint = spark.read.json(self.checkpoint_path)
            return checkpoint.select("version").collect()[0][0]
        except:
            return 0

    def save_checkpoint(self, version):
        """Guarda la versión procesada."""
        checkpoint_data = [{"version": version, "timestamp": str(datetime.now())}]
        spark.createDataFrame(checkpoint_data).write.mode("overwrite").json(self.checkpoint_path)

    def get_changes(self, start_version):
        """
        Obtiene cambios desde la última versión procesada.
        Requiere Change Data Feed habilitado en la tabla fuente.
        """
        return (spark.read
                .format("delta")
                .option("readChangeFeed", "true")
                .option("startingVersion", start_version)
                .load(self.source_path))

    def process_incremental(self, transform_func):
        """
        Procesa solo los cambios incrementales.

        Args:
            transform_func: Función que transforma el DataFrame de cambios
        """
        last_version = self.get_last_processed_version()
        current_version = (DeltaTable.forPath(spark, self.source_path)
                          .history(1)
                          .select("version")
                          .collect()[0][0])

        if last_version >= current_version:
            print("No hay cambios nuevos para procesar")
            return

        print(f"Procesando versiones {last_version + 1} a {current_version}")

        # Obtener cambios
        changes = self.get_changes(last_version + 1)

        # Filtrar solo inserts y updates (no deletes para este ejemplo)
        new_data = changes.filter(col("_change_type").isin(["insert", "update_postimage"]))

        # Aplicar transformación
        transformed = transform_func(new_data.drop("_change_type", "_commit_version", "_commit_timestamp"))

        # Merge con tabla destino
        if DeltaTable.isDeltaTable(spark, self.target_path):
            delta_table = DeltaTable.forPath(spark, self.target_path)

            delta_table.alias("target").merge(
                transformed.alias("source"),
                "target.id = source.id"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            transformed.write.format("delta").save(self.target_path)

        # Guardar checkpoint
        self.save_checkpoint(current_version)
        print(f"Procesamiento incremental completado. Nueva versión: {current_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparación: Full Refresh vs Incremental

# COMMAND ----------

def compare_processing_approaches():
    """
    Compara costo de full refresh vs incremental.
    """
    # Métricas simuladas basadas en un dataset de 500GB
    scenarios = {
        "full_refresh": {
            "data_scanned_gb": 500,
            "compute_time_hours": 3.0,
            "cost_per_hour": 50,  # Cluster cost
        },
        "incremental_daily": {
            "data_scanned_gb": 2,  # Solo datos del día
            "compute_time_hours": 0.25,  # 15 minutos
            "cost_per_hour": 50,
        }
    }

    print("=== Comparación de Costos (30 días) ===\n")

    for approach, metrics in scenarios.items():
        daily_cost = metrics["compute_time_hours"] * metrics["cost_per_hour"]
        monthly_cost = daily_cost * 30

        print(f"{approach.upper().replace('_', ' ')}:")
        print(f"  Datos escaneados por ejecución: {metrics['data_scanned_gb']} GB")
        print(f"  Tiempo por ejecución: {metrics['compute_time_hours']} horas")
        print(f"  Costo diario: ${daily_cost:.2f}")
        print(f"  Costo mensual: ${monthly_cost:.2f}")
        print()

    savings = (scenarios["full_refresh"]["compute_time_hours"] * 50 * 30 -
               scenarios["incremental_daily"]["compute_time_hours"] * 50 * 30)
    print(f"AHORRO MENSUAL: ${savings:.2f}")

compare_processing_approaches()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estrategia 3: Evitar Anti-Patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anti-Pattern 1: collect() en datasets grandes

# COMMAND ----------

# MAL - Causa OOM
# all_data = df.collect()  # Trae 500GB al driver

# BIEN - Procesar en Spark
def safe_aggregation(df, group_col, agg_col):
    """Agregación segura que no usa collect()."""
    result = (df
              .groupBy(group_col)
              .agg(
                  spark_sum(agg_col).alias("total"),
                  count("*").alias("count"),
                  avg(agg_col).alias("average")
              ))

    # Solo collect el resultado agregado (pequeño)
    return result.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anti-Pattern 2: UDFs lentas

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# MAL - UDF en Python es lenta
@udf(StringType())
def slow_categorize(amount):
    if amount > 1000:
        return "high"
    elif amount > 100:
        return "medium"
    return "low"

# BIEN - Usar funciones nativas de Spark
from pyspark.sql.functions import when

def fast_categorize(df, amount_col):
    """Categorización usando funciones nativas (100x más rápido)."""
    return df.withColumn(
        "category",
        when(col(amount_col) > 1000, "high")
        .when(col(amount_col) > 100, "medium")
        .otherwise("low")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anti-Pattern 3: Múltiples acciones sobre el mismo DataFrame

# COMMAND ----------

# MAL - Lee los datos 3 veces
# count1 = df.filter(col("status") == "A").count()
# count2 = df.filter(col("status") == "B").count()
# count3 = df.filter(col("status") == "C").count()

# BIEN - Una sola pasada
def efficient_counts(df, status_col):
    """Cuenta múltiples condiciones en una sola pasada."""
    return df.groupBy(status_col).count().collect()

# O usando cache si necesitas múltiples operaciones
def with_caching(df):
    """Usa cache para evitar recomputación."""
    df_cached = df.cache()

    # Múltiples operaciones sobre datos cacheados
    total = df_cached.count()
    stats = df_cached.describe().collect()

    # Liberar cache cuando termine
    df_cached.unpersist()

    return total, stats

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estrategia 4: Monitoreo y Troubleshooting

# COMMAND ----------

def diagnose_slow_job(df):
    """
    Diagnóstico de jobs lentos.
    Ejecutar antes de optimizar para identificar el problema.
    """
    print("=== DIAGNÓSTICO DE PERFORMANCE ===\n")

    # 1. Verificar número de particiones
    num_partitions = df.rdd.getNumPartitions()
    print(f"1. Particiones: {num_partitions}")
    if num_partitions < 100:
        print("   ADVERTENCIA: Pocas particiones, considerar repartition()")
    elif num_partitions > 10000:
        print("   ADVERTENCIA: Demasiadas particiones, considerar coalesce()")

    # 2. Verificar plan de ejecución
    print("\n2. Plan de ejecución:")
    df.explain(mode="simple")

    # 3. Verificar si hay shuffles innecesarios
    plan = df._jdf.queryExecution().executedPlan().toString()
    shuffle_count = plan.lower().count("exchange")
    print(f"\n3. Número de shuffles: {shuffle_count}")
    if shuffle_count > 3:
        print("   ADVERTENCIA: Muchos shuffles, revisar joins y aggregations")

    # 4. Sugerencias
    print("\n4. CHECKLIST:")
    print("   [ ] Verificar Spark UI -> Stages -> Task distribution")
    print("   [ ] Verificar si hay skew (tasks muy largas)")
    print("   [ ] Verificar spill to disk en Executors tab")
    print("   [ ] Verificar GC time en Executors tab")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen de Configuraciones por Tamaño

# COMMAND ----------

def get_recommended_config(dataset_size_gb, cluster_memory_gb):
    """
    Retorna configuración recomendada según tamaño del dataset.
    """
    configs = {
        "small": {  # < 10GB
            "shuffle_partitions": 200,
            "broadcast_threshold": "100m",
            "max_partition_bytes": "128m"
        },
        "medium": {  # 10-100GB
            "shuffle_partitions": 400,
            "broadcast_threshold": "50m",
            "max_partition_bytes": "128m"
        },
        "large": {  # 100-500GB
            "shuffle_partitions": 1000,
            "broadcast_threshold": "10m",
            "max_partition_bytes": "128m"
        },
        "xlarge": {  # 500GB+
            "shuffle_partitions": 2000,
            "broadcast_threshold": "-1",  # Disable auto broadcast
            "max_partition_bytes": "64m"
        }
    }

    if dataset_size_gb < 10:
        size_category = "small"
    elif dataset_size_gb < 100:
        size_category = "medium"
    elif dataset_size_gb < 500:
        size_category = "large"
    else:
        size_category = "xlarge"

    config = configs[size_category]

    print(f"Dataset: ~{dataset_size_gb}GB | Categoría: {size_category}")
    print(f"\nConfiguraciones recomendadas:")
    for key, value in config.items():
        print(f"  spark.sql.{key.replace('_', '.')}: {value}")

    return config

# Ejemplos
get_recommended_config(500, 240)  # 500GB dataset, cluster con 240GB RAM total
