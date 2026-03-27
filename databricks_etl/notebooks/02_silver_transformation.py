# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Data Transformation & Cleansing
# MAGIC Limpieza, normalización y enriquecimiento de datos desde Bronze.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, lower, upper, initcap,
    to_date, to_timestamp, current_timestamp,
    when, coalesce, lit, regexp_replace,
    row_number, count, sum as spark_sum
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuración

# COMMAND ----------

BRONZE_PATH = "/mnt/bronze/"
SILVER_PATH = "/mnt/silver/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones de Limpieza

# COMMAND ----------

def clean_string_columns(df, columns):
    """Limpia columnas de texto: trim y normalización."""
    for col_name in columns:
        df = df.withColumn(col_name, trim(col(col_name)))
    return df


def remove_duplicates(df, key_columns, order_column="_ingestion_timestamp"):
    """
    Elimina duplicados manteniendo el registro más reciente.
    """
    window = Window.partitionBy(key_columns).orderBy(col(order_column).desc())

    df_deduped = (df
        .withColumn("_row_num", row_number().over(window))
        .filter(col("_row_num") == 1)
        .drop("_row_num"))

    return df_deduped


def handle_nulls(df, column_defaults):
    """
    Maneja valores nulos con valores por defecto.
    column_defaults: dict con {columna: valor_default}
    """
    for col_name, default_value in column_defaults.items():
        df = df.withColumn(col_name, coalesce(col(col_name), lit(default_value)))
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformación de Orders

# COMMAND ----------

def transform_orders_to_silver():
    """Transforma orders de Bronze a Silver."""

    # Leer desde Bronze
    orders_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/orders")

    # Limpiar strings
    orders_clean = clean_string_columns(orders_bronze, ["order_id", "customer_id", "product_id", "status"])

    # Castear tipos
    orders_typed = (orders_clean
        .withColumn("quantity", col("quantity").cast(IntegerType()))
        .withColumn("unit_price", col("unit_price").cast(DoubleType()))
        .withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))
        .withColumn("status", lower(col("status"))))

    # Calcular total de orden
    orders_enriched = orders_typed.withColumn(
        "total_amount",
        col("quantity") * col("unit_price")
    )

    # Manejar nulls
    orders_with_defaults = handle_nulls(orders_enriched, {
        "quantity": 1,
        "unit_price": 0.0,
        "status": "unknown"
    })

    # Eliminar duplicados
    orders_deduped = remove_duplicates(orders_with_defaults, ["order_id"])

    # Agregar timestamp de procesamiento
    orders_silver = orders_deduped.withColumn("_processed_timestamp", current_timestamp())

    # Escribir a Silver
    (orders_silver.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("order_date")
     .save(f"{SILVER_PATH}/orders"))

    print(f"Orders Silver: {orders_silver.count()} records")
    return orders_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformación de Customers

# COMMAND ----------

def transform_customers_to_silver():
    """Transforma customers de Bronze a Silver."""

    customers_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/customers")

    # Limpiar y normalizar
    customers_clean = (customers_bronze
        .withColumn("customer_id", trim(col("customer_id")))
        .withColumn("name", initcap(trim(col("name"))))
        .withColumn("email", lower(trim(col("email"))))
        .withColumn("country", initcap(trim(col("country"))))
        .withColumn("registration_date", to_date(col("registration_date"), "yyyy-MM-dd")))

    # Validar email (básico)
    customers_validated = customers_clean.withColumn(
        "is_valid_email",
        col("email").rlike("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
    )

    # Eliminar duplicados
    customers_deduped = remove_duplicates(customers_validated, ["customer_id"])

    # Agregar timestamp
    customers_silver = customers_deduped.withColumn("_processed_timestamp", current_timestamp())

    # Escribir a Silver
    (customers_silver.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("country")
     .save(f"{SILVER_PATH}/customers"))

    print(f"Customers Silver: {customers_silver.count()} records")
    return customers_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformación de Payments

# COMMAND ----------

def transform_payments_to_silver():
    """Transforma payments de Bronze a Silver."""

    payments_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/payments")

    payments_clean = (payments_bronze
        .withColumn("payment_id", trim(col("payment_id")))
        .withColumn("order_id", trim(col("order_id")))
        .withColumn("amount", col("amount").cast(DoubleType()))
        .withColumn("payment_method", lower(trim(col("payment_method"))))
        .withColumn("payment_date", to_date(col("payment_date"), "yyyy-MM-dd"))
        .withColumn("status", lower(trim(col("status")))))

    # Categorizar método de pago
    payments_categorized = payments_clean.withColumn(
        "payment_category",
        when(col("payment_method").isin("credit_card", "debit_card"), "card")
        .when(col("payment_method") == "paypal", "digital_wallet")
        .when(col("payment_method") == "bank_transfer", "bank")
        .otherwise("other")
    )

    payments_deduped = remove_duplicates(payments_categorized, ["payment_id"])

    payments_silver = payments_deduped.withColumn("_processed_timestamp", current_timestamp())

    (payments_silver.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("payment_date")
     .save(f"{SILVER_PATH}/payments"))

    print(f"Payments Silver: {payments_silver.count()} records")
    return payments_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabla Enriquecida: Orders + Customers + Payments

# COMMAND ----------

def create_enriched_orders():
    """Crea tabla enriquecida con join de orders, customers y payments."""

    orders = spark.read.format("delta").load(f"{SILVER_PATH}/orders")
    customers = spark.read.format("delta").load(f"{SILVER_PATH}/customers")
    payments = spark.read.format("delta").load(f"{SILVER_PATH}/payments")

    # Join orders con customers
    orders_with_customers = orders.alias("o").join(
        customers.alias("c"),
        col("o.customer_id") == col("c.customer_id"),
        "left"
    ).select(
        col("o.order_id"),
        col("o.customer_id"),
        col("c.name").alias("customer_name"),
        col("c.email").alias("customer_email"),
        col("c.country").alias("customer_country"),
        col("o.product_id"),
        col("o.quantity"),
        col("o.unit_price"),
        col("o.total_amount"),
        col("o.order_date"),
        col("o.status").alias("order_status")
    )

    # Join con payments
    enriched_orders = orders_with_customers.alias("oc").join(
        payments.alias("p"),
        col("oc.order_id") == col("p.order_id"),
        "left"
    ).select(
        col("oc.*"),
        col("p.payment_id"),
        col("p.amount").alias("payment_amount"),
        col("p.payment_method"),
        col("p.payment_category"),
        col("p.payment_date"),
        col("p.status").alias("payment_status")
    ).withColumn("_processed_timestamp", current_timestamp())

    (enriched_orders.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("order_date")
     .save(f"{SILVER_PATH}/enriched_orders"))

    print(f"Enriched Orders: {enriched_orders.count()} records")
    return enriched_orders

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecución

# COMMAND ----------

# Ejecutar transformaciones
orders_silver = transform_orders_to_silver()
customers_silver = transform_customers_to_silver()
payments_silver = transform_payments_to_silver()

# Crear tabla enriquecida
enriched_orders = create_enriched_orders()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validación

# COMMAND ----------

enriched_orders.show(10, truncate=False)
