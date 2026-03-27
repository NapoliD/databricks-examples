# Predicción de Churn de Clientes con MLflow

> **Importante:** Todos los datos en este proyecto son **ficticios** y fueron creados únicamente con fines demostrativos y educativos. Este proyecto muestra mis habilidades con MLflow, Databricks y pipelines de ML en producción.

Construí este proyecto para demostrar operaciones de Machine Learning (MLOps) de extremo a extremo usando MLflow en Databricks. El objetivo es predecir el churn (abandono) de clientes en un contexto de e-commerce usando un pipeline completo desde feature engineering hasta inferencia por lotes.

---

## Por Qué Construí Este Proyecto

Como Data Engineer/ML Engineer, quería demostrar que entiendo no solo cómo entrenar modelos, sino cómo operacionalizarlos en producción. Este proyecto demuestra:

- **Feature Engineering a Escala**: Usando PySpark y Databricks Feature Store
- **Tracking de Experimentos**: Logging completo con MLflow
- **Comparación de Modelos**: Evaluando múltiples algoritmos sistemáticamente
- **Model Registry**: Control de versiones y gestión del ciclo de vida de modelos
- **Inferencia por Lotes**: Pipelines de predicción escalables

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE STORE                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Features RFM   │  │   Behavioral    │  │   Temporal      │             │
│  │   - Recencia    │  │   - Tasa cancel │  │   - Antigüedad  │             │
│  │   - Frecuencia  │  │   - Tam. carrito│  │   - Velocidad   │             │
│  │   - Monetario   │  │   - Diversidad  │  │   - Ciclo vida  │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MLFLOW EXPERIMENT TRACKING                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │    Logistic     │  │  Random Forest  │  │    XGBoost      │  ← Mejor    │
│  │   Regression    │  │                 │  │                 │             │
│  │   AUC: ~0.78    │  │   AUC: ~0.85    │  │   AUC: ~0.89    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL REGISTRY                                    │
│                                                                             │
│    v1 (Archived) ──► v2 (Staging) ──► v3 (Production)                      │
│                                                                             │
│    Tags: data_version, training_date, author, validated                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INFERENCIA POR LOTES                                  │
│                                                                             │
│    Cargar Modelo ──► Scoring Clientes ──► Guardar en Delta Lake            │
│                                                                             │
│    Output: churn_probability, churn_risk (ALTO/MEDIO/BAJO)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
databricks_ml/
├── notebooks/
│   ├── 01_feature_engineering.py    # Feature Store + RFM/Behavioral/Temporal
│   ├── 02_model_training.py         # Entrenar y comparar 3 algoritmos
│   ├── 03_mlflow_tracking.py        # Tracking avanzado de experimentos
│   ├── 04_model_registry.py         # Versionado y ciclo de vida
│   └── 05_batch_inference.py        # Pipeline de scoring en producción
│
├── tests/
│   └── test_features.py             # Tests unitarios para features
│
├── config/
│   └── ml_config.yaml               # Configuración centralizada
│
├── README.md                        # Documentación en inglés
└── README_ES.md                     # Documentación en español (este archivo)
```

---

## Descripción de Notebooks

### 01. Feature Engineering

Diseñé un pipeline completo de feature engineering que captura tres dimensiones del comportamiento del cliente:

| Grupo de Features | Features | Significado de Negocio |
|-------------------|----------|------------------------|
| **RFM** | recency_days, frequency, monetary_value | Métricas core de valor del cliente |
| **Behavioral** | cancellation_rate, avg_basket_size, unique_products | Indicadores de satisfacción |
| **Temporal** | days_since_registration, purchase_velocity | Etapa del ciclo de vida |

**Decisiones clave que tomé:**
- Usé Feature Store para versionado y reutilización de features
- Creé la etiqueta de churn basada en 30 días de inactividad
- Manejé valores nulos con defaults lógicos de negocio

### 02. Model Training

Entrené y comparé tres algoritmos para encontrar el mejor:

| Modelo | Por Qué Lo Elegí |
|--------|------------------|
| **Logistic Regression** | Baseline interpretable, bueno para entender importancia de features |
| **Random Forest** | Robusto a outliers, captura relaciones no lineales |
| **XGBoost** | Estado del arte para datos tabulares, maneja bien clases desbalanceadas |

**Técnicas clave:**
- Cross-validation para estimaciones de performance confiables
- Pesos de clase para datos desbalanceados
- Análisis de importancia de features entre modelos

### 03. MLflow Tracking

Demostré capacidades avanzadas de tracking con MLflow:

- **Tags**: Metadata del proyecto para organización (author, data_version, environment)
- **Parámetros**: Todos los hiperparámetros y características de los datos
- **Métricas**: AUC, precision, recall, F1-score
- **Artifacts**: Curvas ROC, matrices de confusión, gráficos de importancia
- **Nested Runs**: Búsqueda de hiperparámetros organizada

### 04. Model Registry

Implementé el ciclo de vida completo del modelo:

```
None → Staging → Production → Archived
```

**Prácticas clave:**
- Siempre validar en Staging antes de Production
- Archivar (no borrar) modelos de producción antiguos
- Agregar descripciones y tags significativos
- Cargar por stage en código de producción para actualizaciones sin cambios

### 05. Batch Inference

Construí un pipeline de scoring listo para producción:

- **Pandas UDFs**: Para scoring distribuido a escala
- **Categorías de Riesgo**: ALTO (>70%), MEDIO (40-70%), BAJO (<40%)
- **Delta Lake**: Almacenamiento particionado para queries eficientes
- **Métricas de Monitoreo**: Seguimiento de distribuciones de predicciones

---

## Tecnologías Utilizadas

| Tecnología | Propósito |
|------------|-----------|
| **Databricks** | Plataforma cloud, notebooks, clusters |
| **MLflow** | Tracking de experimentos, model registry |
| **PySpark** | Procesamiento distribuido de datos |
| **Delta Lake** | Almacenamiento ACID, time travel |
| **Feature Store** | Versionado y serving de features |
| **XGBoost** | Modelo de gradient boosting |
| **scikit-learn** | Algoritmos de ML y utilidades |
| **pytest** | Testing unitario |

---

## Cómo Ejecutar

### En Databricks

1. Clonar o importar este repositorio a tu workspace
2. Configurar paths en `config/ml_config.yaml`
3. Ejecutar notebooks en orden: 01 → 02 → 03 → 04 → 05

### Localmente (para testing)

```bash
# Instalar dependencias
pip install pyspark mlflow xgboost scikit-learn pandas numpy pytest

# Ejecutar tests
pytest tests/ -v
```

---

## Aprendizajes Clave

Construir este proyecto me enseñó varias lecciones importantes:

1. **El Feature Engineering Importa Más Que La Selección del Modelo**: Las features RFM solas proporcionaron la mayor parte del poder predictivo.

2. **El Tracking con MLflow es Esencial**: Sin logging apropiado, reproducir experimentos se vuelve imposible.

3. **El Model Registry Proporciona Seguridad**: El deployment basado en stages previene problemas accidentales en producción.

4. **Batch vs Real-time**: Para predicción de churn, el scoring diario por lotes es suficiente y más costo-efectivo.

5. **Calidad de Datos Primero**: Pasé tiempo significativo asegurando que las features estuvieran correctamente calculadas.

---

## Qué Agregaría en Producción

Si este fuera un sistema de producción real, agregaría:

- **Unity Catalog**: Para governance de datos y lineage
- **Databricks Workflows**: Para orquestación
- **Model Monitoring**: Detección de drift con reentrenamiento automático
- **A/B Testing**: Rollout gradual de nuevos modelos
- **Real-time Serving**: Para clientes de alto valor

---

## Sobre los Datos

> **Todos los datos en este proyecto son completamente ficticios.**

Creé datos sintéticos de clientes, órdenes y pagos para demostrar el pipeline de ML. Los patrones en los datos están diseñados para ser lo suficientemente realistas para entrenar modelos significativos, pero no se utilizó información real de clientes.

---

## Contacto

Este proyecto es parte de mi portfolio demostrando habilidades de MLOps con Databricks y MLflow. No dudes en contactarme si tienes preguntas sobre la implementación.

---

## Licencia

MIT License
