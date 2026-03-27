"""
Tests para transformaciones del pipeline ETL.
Ejecutar con pytest en Databricks o localmente con PySpark.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType


@pytest.fixture(scope="session")
def spark():
    """Crea sesión de Spark para tests."""
    return (SparkSession.builder
            .master("local[*]")
            .appName("ETL_Tests")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())


@pytest.fixture
def sample_orders(spark):
    """Datos de ejemplo para orders."""
    data = [
        ("ORD001", "CUST001", "PROD001", 2, 29.99, "2024-01-15", "completed"),
        ("ORD002", "CUST002", "PROD002", 1, 149.99, "2024-01-15", "completed"),
        ("ORD003", "CUST001", "PROD003", 3, 9.99, "2024-01-16", "pending"),
        ("ORD004", None, "PROD001", 1, 29.99, "2024-01-16", "completed"),  # null customer
        ("ORD005", "CUST003", "PROD002", -1, 149.99, "2024-01-17", "completed"),  # negative qty
    ]

    schema = StructType([
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("product_id", StringType(), False),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("order_date", StringType(), True),
        StructField("status", StringType(), True)
    ])

    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_customers(spark):
    """Datos de ejemplo para customers."""
    data = [
        ("CUST001", "Juan Garcia", "juan@email.com", "Argentina"),
        ("CUST002", "Maria Lopez", "maria@email.com", "Mexico"),
        ("CUST003", "Carlos R", "invalid-email", "Colombia"),  # email inválido
    ]

    schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("country", StringType(), True)
    ])

    return spark.createDataFrame(data, schema)


class TestDataQuality:
    """Tests de calidad de datos."""

    def test_no_null_order_ids(self, sample_orders):
        """Verifica que order_id no tenga nulls."""
        null_count = sample_orders.filter(col("order_id").isNull()).count()
        assert null_count == 0, f"Found {null_count} null order_ids"

    def test_unique_order_ids(self, sample_orders):
        """Verifica unicidad de order_id."""
        total = sample_orders.count()
        unique = sample_orders.select("order_id").distinct().count()
        assert total == unique, f"Found {total - unique} duplicate order_ids"

    def test_positive_quantities(self, sample_orders):
        """Verifica que quantities sean positivas."""
        negative = sample_orders.filter(col("quantity") <= 0).count()
        # En este test esperamos 1 registro con cantidad negativa
        assert negative == 1, f"Expected 1 negative quantity, found {negative}"

    def test_valid_status_values(self, sample_orders):
        """Verifica valores válidos de status."""
        valid_statuses = ["completed", "pending", "cancelled", "processing"]
        invalid = sample_orders.filter(~col("status").isin(valid_statuses)).count()
        assert invalid == 0, f"Found {invalid} invalid status values"

    def test_email_format(self, sample_customers):
        """Verifica formato de email."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        invalid = sample_customers.filter(~col("email").rlike(email_pattern)).count()
        # Esperamos 1 email inválido
        assert invalid == 1, f"Expected 1 invalid email, found {invalid}"


class TestTransformations:
    """Tests de transformaciones."""

    def test_total_amount_calculation(self, sample_orders):
        """Verifica cálculo de total_amount."""
        df = sample_orders.withColumn(
            "total_amount",
            col("quantity") * col("unit_price")
        )

        # Verificar primer registro
        first = df.filter(col("order_id") == "ORD001").first()
        expected = 2 * 29.99
        assert abs(first["total_amount"] - expected) < 0.01

    def test_filter_completed_orders(self, sample_orders):
        """Verifica filtro de órdenes completadas."""
        completed = sample_orders.filter(col("status") == "completed")
        assert completed.count() == 3

    def test_join_orders_customers(self, sample_orders, sample_customers):
        """Verifica join entre orders y customers."""
        joined = sample_orders.join(
            sample_customers,
            sample_orders.customer_id == sample_customers.customer_id,
            "inner"
        )

        # Solo 3 orders tienen customer_id válido que existe en customers
        assert joined.count() == 4


class TestAggregations:
    """Tests de agregaciones."""

    def test_count_by_status(self, sample_orders):
        """Verifica conteo por status."""
        counts = (sample_orders
                  .groupBy("status")
                  .agg(count("*").alias("count"))
                  .collect())

        status_counts = {row["status"]: row["count"] for row in counts}
        assert status_counts.get("completed", 0) == 3
        assert status_counts.get("pending", 0) == 2

    def test_revenue_by_date(self, sample_orders):
        """Verifica cálculo de revenue por fecha."""
        revenue = (sample_orders
                   .withColumn("total", col("quantity") * col("unit_price"))
                   .groupBy("order_date")
                   .agg({"total": "sum"})
                   .collect())

        assert len(revenue) > 0


class TestSchemaValidation:
    """Tests de validación de schema."""

    def test_orders_schema(self, sample_orders):
        """Verifica schema de orders."""
        expected_columns = ["order_id", "customer_id", "product_id",
                           "quantity", "unit_price", "order_date", "status"]

        for col_name in expected_columns:
            assert col_name in sample_orders.columns, f"Missing column: {col_name}"

    def test_column_types(self, sample_orders):
        """Verifica tipos de columnas."""
        schema = {field.name: field.dataType for field in sample_orders.schema.fields}

        assert isinstance(schema["quantity"], IntegerType)
        assert isinstance(schema["unit_price"], DoubleType)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
