# Databricks notebook source
# MAGIC %md
# MAGIC # Handling Data Skew en Joins
# MAGIC Técnicas para manejar distribución desigual de datos que causa cuellos de botella en procesamiento distribuido.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, rand, concat, broadcast,
    count, sum as spark_sum, floor
)
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## El Problema: Data Skew
# MAGIC
# MAGIC Cuando pocas keys concentran la mayoría de los datos, algunos executors procesan
# MAGIC mucho más que otros, causando:
# MAGIC - Jobs extremadamente lentos
# MAGIC - Errores de Out of Memory (OOM)
# MAGIC - Recursos desperdiciados (executors idle esperando a los sobrecargados)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulación de Skew
# MAGIC Creamos un dataset donde 5 clientes tienen el 70% de las órdenes

# COMMAND ----------

def create_skewed_orders(spark, num_orders=1000000):
    """
    Genera órdenes con distribución skewed.
    5 clientes corporativos tienen 70% de las órdenes.
    """
    # 70% de órdenes para 5 clientes grandes
    large_customers = ["CORP_001", "CORP_002", "CORP_003", "CORP_004", "CORP_005"]
    num_large = int(num_orders * 0.7)

    # 30% distribuido entre 10,000 clientes pequeños
    num_small = num_orders - num_large

    large_orders = spark.range(num_large).select(
        col("id").alias("order_id"),
        (floor(rand() * 5)).cast("int").alias("customer_idx"),
        (rand() * 1000).alias("amount")
    ).withColumn(
        "customer_id",
        concat(lit("CORP_00"), (col("customer_idx") + 1).cast("string"))
    ).drop("customer_idx")

    small_orders = spark.range(num_large, num_orders).select(
        col("id").alias("order_id"),
        concat(lit("CUST_"), (rand() * 10000).cast("int").cast("string")).alias("customer_id"),
        (rand() * 500).alias("amount")
    )

    return large_orders.union(small_orders)

def create_customers(spark):
    """Tabla de clientes con información adicional."""
    large = [(f"CORP_00{i}", f"Corporativo {i}", "Enterprise", 0.15) for i in range(1, 6)]
    small = [(f"CUST_{i}", f"Cliente {i}", "Standard", 0.0) for i in range(10000)]

    schema = ["customer_id", "name", "tier", "discount"]
    return spark.createDataFrame(large + small, schema)

# COMMAND ----------

# Crear datasets
df_orders = create_skewed_orders(spark)
df_customers = create_customers(spark)

# Verificar el skew
print("Distribución de órdenes por cliente (top 10):")
df_orders.groupBy("customer_id").count().orderBy(col("count").desc()).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnóstico: Identificar Skew
# MAGIC
# MAGIC Antes de optimizar, hay que confirmar que el problema es skew.

# COMMAND ----------

def diagnose_skew(df, key_column, threshold_ratio=10):
    """
    Diagnostica si existe skew en una columna.

    Args:
        df: DataFrame a analizar
        key_column: Columna a verificar
        threshold_ratio: Si max/median > threshold, hay skew

    Returns:
        dict con métricas de skew
    """
    stats = df.groupBy(key_column).count()

    count_stats = stats.select(
        spark_sum("count").alias("total"),
        count("*").alias("unique_keys")
    ).collect()[0]

    distribution = stats.orderBy(col("count").desc())
    top_keys = distribution.limit(10).collect()

    counts = [row["count"] for row in stats.collect()]
    counts.sort()
    median = counts[len(counts) // 2]
    max_count = counts[-1]

    skew_ratio = max_count / median if median > 0 else float('inf')

    result = {
        "total_records": count_stats["total"],
        "unique_keys": count_stats["unique_keys"],
        "max_count": max_count,
        "median_count": median,
        "skew_ratio": skew_ratio,
        "has_skew": skew_ratio > threshold_ratio,
        "top_keys": [(row[key_column], row["count"]) for row in top_keys]
    }

    print(f"=== Diagnóstico de Skew en '{key_column}' ===")
    print(f"Total registros: {result['total_records']:,}")
    print(f"Keys únicas: {result['unique_keys']:,}")
    print(f"Max por key: {result['max_count']:,}")
    print(f"Mediana por key: {result['median_count']:,}")
    print(f"Ratio de skew: {result['skew_ratio']:.2f}x")
    print(f"¿Tiene skew?: {'SÍ' if result['has_skew'] else 'NO'}")
    print(f"\nTop 5 keys:")
    for key, cnt in result['top_keys'][:5]:
        pct = (cnt / result['total_records']) * 100
        print(f"  {key}: {cnt:,} ({pct:.1f}%)")

    return result

# COMMAND ----------

skew_info = diagnose_skew(df_orders, "customer_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solución 1: Broadcast Join
# MAGIC
# MAGIC Si la tabla pequeña cabe en memoria (~10MB por executor), usar broadcast.
# MAGIC Evita shuffle completamente.

# COMMAND ----------

def join_with_broadcast(df_large, df_small, join_key):
    """
    Join usando broadcast para la tabla pequeña.
    Ideal cuando df_small < 10MB.
    """
    return df_large.join(
        broadcast(df_small),
        join_key,
        "left"
    )

# COMMAND ----------

# Broadcast join - la tabla de clientes es pequeña
result_broadcast = join_with_broadcast(df_orders, df_customers, "customer_id")
print(f"Registros resultantes: {result_broadcast.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solución 2: Salting
# MAGIC
# MAGIC Cuando broadcast no es posible, distribuimos las hot keys artificialmente.

# COMMAND ----------

def join_with_salting(df_large, df_small, join_key, salt_factor=10, hot_keys=None):
    """
    Join con salting para distribuir hot keys.

    Args:
        df_large: DataFrame grande (el que tiene skew)
        df_small: DataFrame pequeño
        join_key: Columna de join
        salt_factor: Número de buckets para distribuir
        hot_keys: Lista de keys con skew (si None, detecta automáticamente)
    """
    # Detectar hot keys si no se proporcionan
    if hot_keys is None:
        key_counts = df_large.groupBy(join_key).count()
        threshold = key_counts.approxQuantile("count", [0.99], 0.01)[0]
        hot_keys = [row[join_key] for row in
                   key_counts.filter(col("count") > threshold).collect()]

    print(f"Hot keys detectadas: {hot_keys}")

    # Agregar salt a registros con hot keys
    df_large_salted = df_large.withColumn(
        "salt",
        when(col(join_key).isin(hot_keys), (rand() * salt_factor).cast("int"))
        .otherwise(lit(0))
    ).withColumn(
        "join_key_salted",
        concat(col(join_key), lit("_"), col("salt"))
    )

    # Replicar filas de tabla pequeña para hot keys
    from pyspark.sql.functions import explode, array, when

    df_small_replicated = df_small.withColumn(
        "salt",
        when(col(join_key).isin(hot_keys),
             explode(array([lit(i) for i in range(salt_factor)])))
        .otherwise(lit(0))
    ).withColumn(
        "join_key_salted",
        concat(col(join_key), lit("_"), col("salt"))
    )

    # Join con keys salteadas
    result = df_large_salted.join(
        df_small_replicated,
        "join_key_salted",
        "left"
    ).drop("salt", "join_key_salted")

    return result

# COMMAND ----------

# Este import faltaba arriba
from pyspark.sql.functions import when, explode, array

# Aplicar salting
result_salted = join_with_salting(
    df_orders,
    df_customers,
    "customer_id",
    salt_factor=10
)

print(f"Registros resultantes: {result_salted.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solución 3: Adaptive Query Execution (AQE)
# MAGIC
# MAGIC Spark 3.0+ puede manejar skew automáticamente con AQE.

# COMMAND ----------

# Configurar AQE para skew handling automático
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")

# Con AQE, el join normal se optimiza automáticamente
result_aqe = df_orders.join(df_customers, "customer_id", "left")
result_aqe.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solución 4: Reparticionado Previo
# MAGIC
# MAGIC Para casos moderados, reparticionar puede ayudar.

# COMMAND ----------

def join_with_repartition(df_large, df_small, join_key, num_partitions=200):
    """
    Reparticiona ambos DataFrames antes del join.
    Útil para skew moderado.
    """
    df_large_repart = df_large.repartition(num_partitions, join_key)
    df_small_repart = df_small.repartition(num_partitions, join_key)

    return df_large_repart.join(df_small_repart, join_key, "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparación de Técnicas

# COMMAND ----------

# MAGIC %md
# MAGIC | Técnica | Cuándo usar | Ventajas | Desventajas |
# MAGIC |---------|-------------|----------|-------------|
# MAGIC | **Broadcast** | Tabla pequeña (<10MB) | Elimina shuffle | Límite de tamaño |
# MAGIC | **Salting** | Hot keys conocidas | Control granular | Complejidad de código |
# MAGIC | **AQE** | Spark 3.0+ | Automático | Menos control |
# MAGIC | **Repartition** | Skew moderado | Simple | No resuelve skew extremo |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoreo: Verificar en Spark UI
# MAGIC
# MAGIC Después de aplicar la solución, verificar en Spark UI:
# MAGIC 1. **Stages tab**: Tasks deberían tener duración similar
# MAGIC 2. **SQL tab**: No debería haber particiones con >> datos que otras
# MAGIC 3. **Executors tab**: Uso de memoria balanceado

# COMMAND ----------

# Función para verificar balance de particiones
def check_partition_balance(df):
    """Verifica si las particiones están balanceadas."""
    partition_sizes = df.rdd.mapPartitions(
        lambda x: [sum(1 for _ in x)]
    ).collect()

    if not partition_sizes:
        print("DataFrame vacío")
        return

    avg_size = sum(partition_sizes) / len(partition_sizes)
    max_size = max(partition_sizes)
    min_size = min(partition_sizes)

    print(f"Particiones: {len(partition_sizes)}")
    print(f"Promedio por partición: {avg_size:,.0f}")
    print(f"Máximo: {max_size:,} ({max_size/avg_size:.1f}x promedio)")
    print(f"Mínimo: {min_size:,}")
    print(f"Balance: {'BUENO' if max_size/avg_size < 3 else 'REVISAR'}")

# COMMAND ----------

print("Antes de optimización:")
check_partition_balance(df_orders)

print("\nDespués de repartición:")
check_partition_balance(df_orders.repartition(100, "customer_id"))
