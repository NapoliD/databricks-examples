# Databricks notebook source
# MAGIC %md
# MAGIC # Optimización de Costos en Databricks
# MAGIC Técnicas para reducir costos de compute y storage manteniendo performance.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Análisis de Costos Actual
# MAGIC
# MAGIC Antes de optimizar, medir el baseline.

# COMMAND ----------

class CostAnalyzer:
    """
    Analiza costos de clusters y jobs.
    En producción, usar Databricks Cost Management APIs.
    """

    # Costos aproximados por tipo de instancia (USD/hora)
    INSTANCE_COSTS = {
        "i3.xlarge": 0.312,
        "i3.2xlarge": 0.624,
        "r5.xlarge": 0.252,
        "r5.2xlarge": 0.504,
        "m5.xlarge": 0.192,
        "m5.2xlarge": 0.384,
    }

    # DBU costs
    DBU_COST_JOBS = 0.15  # Jobs compute
    DBU_COST_ALL_PURPOSE = 0.55  # All-purpose compute

    def calculate_cluster_cost(self, instance_type, num_workers, hours, is_job_cluster=True):
        """
        Calcula costo estimado de un cluster.
        """
        instance_cost = self.INSTANCE_COSTS.get(instance_type, 0.30)
        dbu_cost = self.DBU_COST_JOBS if is_job_cluster else self.DBU_COST_ALL_PURPOSE

        # Costo total = (instancias + DBUs) * horas
        total_instance_cost = instance_cost * (num_workers + 1) * hours  # +1 for driver
        total_dbu_cost = dbu_cost * (num_workers + 1) * hours

        return {
            "instance_cost": total_instance_cost,
            "dbu_cost": total_dbu_cost,
            "total": total_instance_cost + total_dbu_cost,
            "cost_per_hour": (total_instance_cost + total_dbu_cost) / hours
        }

    def analyze_current_usage(self):
        """
        Simula análisis de uso actual.
        En producción, usar APIs de Databricks.
        """
        current_usage = {
            "clusters": [
                {
                    "name": "ETL-Production",
                    "type": "all-purpose",
                    "instance": "i3.2xlarge",
                    "workers": 10,
                    "hours_per_day": 24,
                    "days_per_month": 30,
                    "utilization_pct": 35
                },
                {
                    "name": "Analytics",
                    "type": "all-purpose",
                    "instance": "r5.xlarge",
                    "workers": 5,
                    "hours_per_day": 12,
                    "days_per_month": 22,
                    "utilization_pct": 45
                },
                {
                    "name": "Nightly-Batch",
                    "type": "job",
                    "instance": "i3.xlarge",
                    "workers": 8,
                    "hours_per_day": 3,
                    "days_per_month": 30,
                    "utilization_pct": 85
                }
            ]
        }

        print("=== ANÁLISIS DE USO ACTUAL ===\n")

        total_monthly = 0
        for cluster in current_usage["clusters"]:
            monthly_hours = cluster["hours_per_day"] * cluster["days_per_month"]
            is_job = cluster["type"] == "job"

            cost = self.calculate_cluster_cost(
                cluster["instance"],
                cluster["workers"],
                monthly_hours,
                is_job
            )

            total_monthly += cost["total"]

            print(f"{cluster['name']}:")
            print(f"  Tipo: {cluster['type']}")
            print(f"  Config: {cluster['workers']}x {cluster['instance']}")
            print(f"  Horas/mes: {monthly_hours}")
            print(f"  Utilización: {cluster['utilization_pct']}%")
            print(f"  Costo mensual: ${cost['total']:,.2f}")
            print()

        print(f"TOTAL MENSUAL: ${total_monthly:,.2f}")
        return current_usage, total_monthly

# COMMAND ----------

analyzer = CostAnalyzer()
usage, baseline_cost = analyzer.analyze_current_usage()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Optimización: Auto-scaling y Auto-terminate

# COMMAND ----------

def optimize_cluster_config(cluster_info):
    """
    Genera configuración optimizada de cluster.
    """
    optimizations = []
    new_config = cluster_info.copy()

    # 1. Si utilización < 50%, reducir workers mínimos
    if cluster_info["utilization_pct"] < 50:
        suggested_min = max(2, cluster_info["workers"] // 3)
        optimizations.append({
            "type": "autoscaling",
            "change": f"Cambiar de {cluster_info['workers']} workers fijos a {suggested_min}-{cluster_info['workers']} auto-scaling",
            "reason": f"Utilización actual: {cluster_info['utilization_pct']}%"
        })
        new_config["min_workers"] = suggested_min
        new_config["max_workers"] = cluster_info["workers"]

    # 2. Si es all-purpose y no se usa 24/7, agregar auto-terminate
    if cluster_info["type"] == "all-purpose":
        optimizations.append({
            "type": "auto_terminate",
            "change": "Agregar auto-terminate de 15 minutos",
            "reason": "Evita pagar por tiempo idle"
        })
        new_config["auto_terminate_minutes"] = 15

    # 3. Sugerir job clusters para cargas batch
    if cluster_info["type"] == "all-purpose" and cluster_info["hours_per_day"] < 8:
        optimizations.append({
            "type": "cluster_type",
            "change": "Migrar a Job Cluster",
            "reason": f"Solo {cluster_info['hours_per_day']}h/día de uso. Job clusters son 70% más baratos en DBUs"
        })
        new_config["type"] = "job"

    return new_config, optimizations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ejemplo de Configuración Optimizada

# COMMAND ----------

# Cluster JSON optimizado
optimized_cluster_config = {
    "cluster_name": "ETL-Production-Optimized",
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "i3.xlarge",  # Downsize si es posible
    "autoscale": {
        "min_workers": 2,
        "max_workers": 10
    },
    "autotermination_minutes": 15,
    "aws_attributes": {
        "first_on_demand": 1,  # Solo 1 on-demand para estabilidad
        "availability": "SPOT_WITH_FALLBACK",
        "zone_id": "auto",
        "spot_bid_price_percent": 100
    },
    "spark_conf": {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true"
    },
    "custom_tags": {
        "Environment": "Production",
        "CostCenter": "DataEngineering"
    }
}

print("Configuración de cluster optimizada:")
print(json.dumps(optimized_cluster_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Spot Instances
# MAGIC
# MAGIC Usar Spot puede reducir costos 60-70%.

# COMMAND ----------

class SpotInstanceStrategy:
    """
    Estrategias para uso de Spot instances.
    """

    @staticmethod
    def get_spot_config(workload_type):
        """
        Retorna configuración de Spot según tipo de workload.
        """
        configs = {
            "batch_tolerant": {
                # Workloads que pueden reiniciar sin problema
                "first_on_demand": 0,
                "availability": "SPOT",
                "spot_bid_price_percent": 100,
                "description": "100% Spot - máximo ahorro, acepta interrupciones"
            },
            "batch_important": {
                # Workloads batch que no deben fallar
                "first_on_demand": 1,
                "availability": "SPOT_WITH_FALLBACK",
                "spot_bid_price_percent": 100,
                "description": "Spot con fallback - ahorro con resiliencia"
            },
            "interactive": {
                # Notebooks y queries interactivas
                "first_on_demand": 2,
                "availability": "SPOT_WITH_FALLBACK",
                "spot_bid_price_percent": 100,
                "description": "Driver + 1 on-demand - estabilidad para UX"
            },
            "streaming": {
                # Workloads de streaming 24/7
                "first_on_demand": 3,
                "availability": "SPOT_WITH_FALLBACK",
                "spot_bid_price_percent": 100,
                "description": "Core nodes on-demand - no interrumpir streaming"
            }
        }
        return configs.get(workload_type, configs["batch_important"])

    @staticmethod
    def calculate_spot_savings(on_demand_cost, spot_discount=0.65):
        """
        Calcula ahorro estimado con Spot.
        """
        spot_cost = on_demand_cost * (1 - spot_discount)
        savings = on_demand_cost - spot_cost

        return {
            "on_demand_cost": on_demand_cost,
            "spot_cost": spot_cost,
            "savings": savings,
            "savings_pct": spot_discount * 100
        }

# COMMAND ----------

# Ejemplo de ahorro con Spot
spot = SpotInstanceStrategy()

for workload in ["batch_tolerant", "batch_important", "interactive", "streaming"]:
    config = spot.get_spot_config(workload)
    print(f"\n{workload.upper()}:")
    print(f"  Config: {config['description']}")
    print(f"  first_on_demand: {config['first_on_demand']}")

# Calcular ahorro
savings = spot.calculate_spot_savings(10000)  # $10K baseline
print(f"\nAhorro estimado mensual: ${savings['savings']:,.2f} ({savings['savings_pct']:.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Right-sizing de Instancias

# COMMAND ----------

def analyze_and_rightsize(spark):
    """
    Analiza uso de recursos y sugiere right-sizing.
    En producción, analizar métricas de Ganglia/CloudWatch.
    """
    # Métricas simuladas
    metrics = {
        "avg_cpu_utilization": 0.30,
        "max_cpu_utilization": 0.75,
        "avg_memory_utilization": 0.85,
        "max_memory_utilization": 0.95,
        "spill_to_disk_gb": 50,
        "current_instance": "i3.2xlarge",
        "current_specs": {"vcpu": 8, "memory_gb": 61, "storage_gb": 1900}
    }

    print("=== ANÁLISIS DE RIGHT-SIZING ===\n")
    print(f"Instancia actual: {metrics['current_instance']}")
    print(f"  vCPUs: {metrics['current_specs']['vcpu']}")
    print(f"  Memoria: {metrics['current_specs']['memory_gb']} GB")
    print(f"\nMétricas de uso:")
    print(f"  CPU promedio: {metrics['avg_cpu_utilization']*100:.0f}%")
    print(f"  CPU máximo: {metrics['max_cpu_utilization']*100:.0f}%")
    print(f"  Memoria promedio: {metrics['avg_memory_utilization']*100:.0f}%")
    print(f"  Memoria máxima: {metrics['max_memory_utilization']*100:.0f}%")
    print(f"  Spill to disk: {metrics['spill_to_disk_gb']} GB")

    # Diagnóstico
    print("\nDIAGNÓSTICO:")

    recommendations = []

    if metrics["avg_cpu_utilization"] < 0.50 and metrics["avg_memory_utilization"] > 0.80:
        print("  - CPU subutilizada, memoria alta")
        print("  - Recomendación: Cambiar a instancia memory-optimized (r5)")
        recommendations.append("r5.xlarge")

    if metrics["spill_to_disk_gb"] > 20:
        print("  - Alto spill to disk indica falta de memoria")
        print("  - Considerar más memoria o más workers")

    if metrics["avg_cpu_utilization"] > 0.80:
        print("  - CPU saturada")
        print("  - Considerar instancias compute-optimized (c5)")
        recommendations.append("c5.2xlarge")

    # Tabla de instancias alternativas
    alternatives = {
        "r5.xlarge": {"vcpu": 4, "memory_gb": 32, "cost_hour": 0.252, "use_case": "Memory-bound"},
        "r5.2xlarge": {"vcpu": 8, "memory_gb": 64, "cost_hour": 0.504, "use_case": "Memory-heavy"},
        "m5.xlarge": {"vcpu": 4, "memory_gb": 16, "cost_hour": 0.192, "use_case": "Balanced"},
        "c5.2xlarge": {"vcpu": 8, "memory_gb": 16, "cost_hour": 0.340, "use_case": "CPU-bound"},
    }

    print("\nINSTANCIAS ALTERNATIVAS:")
    for instance, specs in alternatives.items():
        print(f"  {instance}: {specs['vcpu']} vCPU, {specs['memory_gb']}GB RAM, ${specs['cost_hour']}/hr - {specs['use_case']}")

    return recommendations

# COMMAND ----------

analyze_and_rightsize(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Optimización de Storage

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta Lake Storage Optimization

# COMMAND ----------

def optimize_delta_storage(table_path):
    """
    Optimiza storage de tabla Delta.
    """
    print(f"Optimizando storage para: {table_path}\n")

    # 1. OPTIMIZE - Compacta small files
    print("1. Ejecutando OPTIMIZE (compacta small files)...")
    # spark.sql(f"OPTIMIZE delta.`{table_path}`")
    print("   OPTIMIZE delta.`{table_path}`")

    # 2. VACUUM - Elimina archivos antiguos
    print("\n2. Ejecutando VACUUM (elimina versiones antiguas)...")
    # spark.sql(f"VACUUM delta.`{table_path}` RETAIN 168 HOURS")  # 7 días
    print("   VACUUM delta.`{table_path}` RETAIN 168 HOURS")

    # 3. Verificar estado
    print("\n3. Estado actual de la tabla:")
    # detail = spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`").collect()[0]
    # print(f"   Archivos: {detail['numFiles']}")
    # print(f"   Tamaño: {detail['sizeInBytes'] / (1024**3):.2f} GB")

    print("\nNota: Descomentar comandos para ejecutar en Databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lifecycle Policies

# COMMAND ----------

def setup_data_lifecycle():
    """
    Define políticas de ciclo de vida de datos.
    """
    policies = {
        "bronze": {
            "retention_days": 90,
            "vacuum_frequency": "weekly",
            "description": "Datos raw, mantener 90 días para reprocesamiento"
        },
        "silver": {
            "retention_days": 365,
            "vacuum_frequency": "weekly",
            "description": "Datos limpios, mantener 1 año"
        },
        "gold": {
            "retention_days": 730,
            "vacuum_frequency": "monthly",
            "description": "Datos de negocio, mantener 2 años"
        },
        "staging": {
            "retention_days": 7,
            "vacuum_frequency": "daily",
            "description": "Datos temporales, limpiar agresivamente"
        }
    }

    print("=== POLÍTICAS DE CICLO DE VIDA ===\n")
    for layer, policy in policies.items():
        print(f"{layer.upper()}:")
        print(f"  Retención: {policy['retention_days']} días")
        print(f"  VACUUM: {policy['vacuum_frequency']}")
        print(f"  Descripción: {policy['description']}")
        print()

    # Script de mantenimiento
    print("Script de mantenimiento (ejecutar como job programado):")
    print("""
    -- Ejecutar semanalmente
    OPTIMIZE bronze.* WHERE ingestion_date < current_date() - INTERVAL 7 DAYS;
    VACUUM bronze.* RETAIN 2160 HOURS;  -- 90 días

    OPTIMIZE silver.* WHERE process_date < current_date() - INTERVAL 7 DAYS;
    VACUUM silver.* RETAIN 8760 HOURS;  -- 365 días
    """)

    return policies

# COMMAND ----------

setup_data_lifecycle()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Resumen de Optimizaciones

# COMMAND ----------

def generate_optimization_report(baseline_cost):
    """
    Genera reporte de optimizaciones y ahorro estimado.
    """
    optimizations = [
        {
            "category": "Compute",
            "action": "Auto-scaling (2-10 workers en vez de 10 fijos)",
            "savings_pct": 40,
            "complexity": "Baja"
        },
        {
            "category": "Compute",
            "action": "Auto-terminate (15 min idle)",
            "savings_pct": 20,
            "complexity": "Baja"
        },
        {
            "category": "Compute",
            "action": "Spot instances (SPOT_WITH_FALLBACK)",
            "savings_pct": 60,
            "complexity": "Media"
        },
        {
            "category": "Compute",
            "action": "Job clusters en vez de all-purpose para batch",
            "savings_pct": 70,
            "complexity": "Media"
        },
        {
            "category": "Compute",
            "action": "Right-sizing (i3.2xlarge → r5.xlarge)",
            "savings_pct": 35,
            "complexity": "Baja"
        },
        {
            "category": "Processing",
            "action": "Incremental en vez de full refresh",
            "savings_pct": 90,
            "complexity": "Alta"
        },
        {
            "category": "Storage",
            "action": "VACUUM regular + lifecycle policies",
            "savings_pct": 30,
            "complexity": "Baja"
        }
    ]

    print("=" * 60)
    print("REPORTE DE OPTIMIZACIÓN DE COSTOS")
    print("=" * 60)
    print(f"\nCosto baseline mensual: ${baseline_cost:,.2f}")
    print("\n" + "-" * 60)
    print(f"{'Categoría':<12} {'Acción':<45} {'Ahorro':<8} {'Complejidad'}")
    print("-" * 60)

    # Calcular ahorro compuesto (no aditivo simple)
    remaining_pct = 100
    for opt in optimizations:
        applicable_savings = remaining_pct * (opt["savings_pct"] / 100) * 0.3  # Factor conservador
        print(f"{opt['category']:<12} {opt['action'][:44]:<45} {opt['savings_pct']}%{'':>4} {opt['complexity']}")

    # Estimación conservadora de ahorro total
    estimated_savings_pct = 55  # Conservador
    estimated_savings = baseline_cost * (estimated_savings_pct / 100)
    new_cost = baseline_cost - estimated_savings

    print("-" * 60)
    print(f"\nESTIMACIÓN CONSERVADORA:")
    print(f"  Ahorro mensual: ${estimated_savings:,.2f} ({estimated_savings_pct}%)")
    print(f"  Nuevo costo: ${new_cost:,.2f}")
    print(f"  Ahorro anual: ${estimated_savings * 12:,.2f}")

    return estimated_savings

# COMMAND ----------

generate_optimization_report(baseline_cost)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Wins (Implementar Primero)
# MAGIC
# MAGIC 1. **Auto-terminate** en todos los clusters all-purpose (15 min)
# MAGIC 2. **Auto-scaling** con min_workers = 2
# MAGIC 3. **Spot instances** con fallback
# MAGIC 4. **VACUUM** semanal en todas las tablas Delta
# MAGIC
# MAGIC Estos 4 cambios pueden reducir costos 40-50% con mínimo esfuerzo.
