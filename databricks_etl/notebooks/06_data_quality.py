# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality Framework
# MAGIC Validación y monitoreo de calidad de datos.

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, count, sum as spark_sum, when, lit,
    isnan, isnull, length, regexp_extract,
    current_timestamp, to_json, struct
)
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tipos de Validación

# COMMAND ----------

class ValidationLevel(Enum):
    ERROR = "error"      # Falla el pipeline
    WARNING = "warning"  # Log pero continúa
    INFO = "info"        # Solo informativo


@dataclass
class ValidationResult:
    rule_name: str
    column: str
    level: ValidationLevel
    passed: bool
    total_records: int
    failed_records: int
    failure_rate: float
    message: str

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reglas de Validación

# COMMAND ----------

class DataQualityValidator:
    """Framework de validación de calidad de datos."""

    def __init__(self, df: DataFrame):
        self.df = df
        self.results: List[ValidationResult] = []
        self.total_records = df.count()

    def validate_not_null(self, columns: List[str], level: ValidationLevel = ValidationLevel.ERROR):
        """Valida que las columnas no tengan valores nulos."""
        for column in columns:
            null_count = self.df.filter(col(column).isNull() | isnan(column)).count()
            passed = null_count == 0

            self.results.append(ValidationResult(
                rule_name="not_null",
                column=column,
                level=level,
                passed=passed,
                total_records=self.total_records,
                failed_records=null_count,
                failure_rate=round(null_count / self.total_records * 100, 2) if self.total_records > 0 else 0,
                message=f"Column '{column}' has {null_count} null values"
            ))

        return self

    def validate_unique(self, columns: List[str], level: ValidationLevel = ValidationLevel.ERROR):
        """Valida que no haya duplicados en las columnas especificadas."""
        for column in columns:
            total = self.df.count()
            unique = self.df.select(column).distinct().count()
            duplicates = total - unique
            passed = duplicates == 0

            self.results.append(ValidationResult(
                rule_name="unique",
                column=column,
                level=level,
                passed=passed,
                total_records=total,
                failed_records=duplicates,
                failure_rate=round(duplicates / total * 100, 2) if total > 0 else 0,
                message=f"Column '{column}' has {duplicates} duplicate values"
            ))

        return self

    def validate_positive(self, columns: List[str], level: ValidationLevel = ValidationLevel.ERROR):
        """Valida que los valores numéricos sean positivos."""
        for column in columns:
            negative_count = self.df.filter(col(column) <= 0).count()
            passed = negative_count == 0

            self.results.append(ValidationResult(
                rule_name="positive",
                column=column,
                level=level,
                passed=passed,
                total_records=self.total_records,
                failed_records=negative_count,
                failure_rate=round(negative_count / self.total_records * 100, 2) if self.total_records > 0 else 0,
                message=f"Column '{column}' has {negative_count} non-positive values"
            ))

        return self

    def validate_range(self, column: str, min_val: float, max_val: float, level: ValidationLevel = ValidationLevel.ERROR):
        """Valida que los valores estén dentro de un rango."""
        out_of_range = self.df.filter(
            (col(column) < min_val) | (col(column) > max_val)
        ).count()
        passed = out_of_range == 0

        self.results.append(ValidationResult(
            rule_name="range",
            column=column,
            level=level,
            passed=passed,
            total_records=self.total_records,
            failed_records=out_of_range,
            failure_rate=round(out_of_range / self.total_records * 100, 2) if self.total_records > 0 else 0,
            message=f"Column '{column}' has {out_of_range} values outside range [{min_val}, {max_val}]"
        ))

        return self

    def validate_in_set(self, column: str, valid_values: List[Any], level: ValidationLevel = ValidationLevel.ERROR):
        """Valida que los valores estén en un conjunto definido."""
        invalid_count = self.df.filter(~col(column).isin(valid_values)).count()
        passed = invalid_count == 0

        self.results.append(ValidationResult(
            rule_name="in_set",
            column=column,
            level=level,
            passed=passed,
            total_records=self.total_records,
            failed_records=invalid_count,
            failure_rate=round(invalid_count / self.total_records * 100, 2) if self.total_records > 0 else 0,
            message=f"Column '{column}' has {invalid_count} values not in allowed set"
        ))

        return self

    def validate_regex(self, column: str, pattern: str, level: ValidationLevel = ValidationLevel.WARNING):
        """Valida que los valores coincidan con un patrón regex."""
        invalid_count = self.df.filter(
            ~col(column).rlike(pattern)
        ).count()
        passed = invalid_count == 0

        self.results.append(ValidationResult(
            rule_name="regex",
            column=column,
            level=level,
            passed=passed,
            total_records=self.total_records,
            failed_records=invalid_count,
            failure_rate=round(invalid_count / self.total_records * 100, 2) if self.total_records > 0 else 0,
            message=f"Column '{column}' has {invalid_count} values not matching pattern"
        ))

        return self

    def validate_email(self, column: str, level: ValidationLevel = ValidationLevel.WARNING):
        """Valida formato de email."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return self.validate_regex(column, email_pattern, level)

    def validate_length(self, column: str, min_len: int = None, max_len: int = None, level: ValidationLevel = ValidationLevel.WARNING):
        """Valida longitud de strings."""
        conditions = []
        if min_len:
            conditions.append(length(col(column)) < min_len)
        if max_len:
            conditions.append(length(col(column)) > max_len)

        if conditions:
            filter_cond = conditions[0]
            for c in conditions[1:]:
                filter_cond = filter_cond | c

            invalid_count = self.df.filter(filter_cond).count()
            passed = invalid_count == 0

            self.results.append(ValidationResult(
                rule_name="length",
                column=column,
                level=level,
                passed=passed,
                total_records=self.total_records,
                failed_records=invalid_count,
                failure_rate=round(invalid_count / self.total_records * 100, 2) if self.total_records > 0 else 0,
                message=f"Column '{column}' has {invalid_count} values with invalid length"
            ))

        return self

    def validate_referential_integrity(self, column: str, reference_df: DataFrame, ref_column: str, level: ValidationLevel = ValidationLevel.ERROR):
        """Valida integridad referencial."""
        orphan_count = self.df.join(
            reference_df,
            self.df[column] == reference_df[ref_column],
            "left_anti"
        ).count()
        passed = orphan_count == 0

        self.results.append(ValidationResult(
            rule_name="referential_integrity",
            column=column,
            level=level,
            passed=passed,
            total_records=self.total_records,
            failed_records=orphan_count,
            failure_rate=round(orphan_count / self.total_records * 100, 2) if self.total_records > 0 else 0,
            message=f"Column '{column}' has {orphan_count} orphan records"
        ))

        return self

    def get_results(self) -> List[ValidationResult]:
        """Retorna resultados de validación."""
        return self.results

    def get_summary(self) -> Dict:
        """Retorna resumen de validación."""
        total_rules = len(self.results)
        passed_rules = sum(1 for r in self.results if r.passed)
        failed_rules = total_rules - passed_rules

        errors = [r for r in self.results if not r.passed and r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if not r.passed and r.level == ValidationLevel.WARNING]

        return {
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "pass_rate": round(passed_rules / total_rules * 100, 2) if total_rules > 0 else 0,
            "errors": len(errors),
            "warnings": len(warnings),
            "status": "PASSED" if len(errors) == 0 else "FAILED"
        }

    def print_report(self):
        """Imprime reporte de validación."""
        summary = self.get_summary()

        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Total Records: {self.total_records}")
        print(f"Total Rules: {summary['total_rules']}")
        print(f"Passed: {summary['passed_rules']}")
        print(f"Failed: {summary['failed_rules']}")
        print(f"Pass Rate: {summary['pass_rate']}%")
        print(f"Status: {summary['status']}")
        print("-" * 60)

        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"{status} [{result.level.value.upper()}] {result.rule_name} - {result.column}")
            if not result.passed:
                print(f"   {result.message} ({result.failure_rate}%)")

        print("=" * 60)

    def to_dataframe(self) -> DataFrame:
        """Convierte resultados a DataFrame para persistencia."""
        rows = []
        for r in self.results:
            rows.append({
                "rule_name": r.rule_name,
                "column": r.column,
                "level": r.level.value,
                "passed": r.passed,
                "total_records": r.total_records,
                "failed_records": r.failed_records,
                "failure_rate": r.failure_rate,
                "message": r.message,
                "validation_timestamp": current_timestamp()
            })

        return spark.createDataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejemplo de Uso

# COMMAND ----------

# Cargar datos de ejemplo
SILVER_PATH = "/mnt/silver/"
orders = spark.read.format("delta").load(f"{SILVER_PATH}/orders")

# COMMAND ----------

# Ejecutar validaciones
validator = (DataQualityValidator(orders)
    .validate_not_null(["order_id", "customer_id", "product_id"])
    .validate_unique(["order_id"])
    .validate_positive(["quantity", "unit_price", "total_amount"])
    .validate_in_set("status", ["completed", "pending", "cancelled", "processing"])
    .validate_range("quantity", 1, 100)
    .validate_range("unit_price", 0.01, 10000))

# COMMAND ----------

# Imprimir reporte
validator.print_report()

# COMMAND ----------

# Obtener resumen
summary = validator.get_summary()
print(json.dumps(summary, indent=2))

# COMMAND ----------

# Guardar resultados
results_df = validator.to_dataframe()
results_df.write.format("delta").mode("append").save("/mnt/data_quality/validation_history")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validación de Customers

# COMMAND ----------

customers = spark.read.format("delta").load(f"{SILVER_PATH}/customers")

customer_validator = (DataQualityValidator(customers)
    .validate_not_null(["customer_id", "name", "email"])
    .validate_unique(["customer_id", "email"])
    .validate_email("email")
    .validate_length("name", min_len=2, max_len=100))

customer_validator.print_report()
