# Churn Decision Analytics — Concesionario Automovilístico

Sistema de predicción de churn y optimización de campañas de retención para un concesionario, con modelo ML, cálculo de CLTV y dashboard interactivo de decisión comercial.

---

## Estructura del proyecto

```
Modelo Churn/
├── sandbox.ipynb                        # Pipeline ML completo (67 celdas)
├── app.py                               # Dashboard Streamlit
├── datos_churn_entrenamiento.csv        # 58,049 clientes históricos (train/test)
├── nuevos_clientes.csv                  # 10,000 clientes activos (sin etiqueta)
├── Costes.csv                           # Márgenes y costes por modelo de vehículo (A–K)
├── predicciones_nuevos_clientes.csv     # Output: predicciones + CLTV + acciones
├── pipeline_winner.pkl                  # Modelo serializado + metadatos
├── img/                                 # Gráficos exportados para la app
│   ├── roc_gains_lift.png
│   ├── roc_gains_lift_dashboard.png
│   ├── feature_importance.png
│   ├── beneficio_vs_umbral.png
│   ├── customer_value_churn_risk.png
│   └── margenes.png
├── .streamlit/                          # Configuración Streamlit
└── venv/                                # Entorno virtual Python 3.14
```

---

## Definición del problema

> **Churn** = cliente sin revisión en los últimos **400 días** (`Churn_400 = Y`).

El objetivo es predecir qué clientes de la cartera activa (`nuevos_clientes.csv`) tienen alta probabilidad de no volver, y diseñar una estrategia de intervención económicamente óptima: contactar solo a clientes cuyo **valor esperado recuperado** supere el coste de la campaña.

---

## Pipeline ML — `sandbox.ipynb`

El notebook implementa el pipeline completo en 10 etapas:

| Etapa | Celdas | Descripción |
|-------|--------|-------------|
| **1. DATOS** | 3 | Carga de `datos_churn_entrenamiento.csv` y `nuevos_clientes.csv` |
| **2. VALIDAR** | 4–8 | Tipos, nulos, distribución del target, columnas exclusivas de train |
| **3. EDA** | 9–13 | Tasa de churn por modelo/zona/combustible, distribuciones, correlaciones |
| **4. FEATURE ENGINEERING** | 14–20 | Ingeniería de variables, split temporal, pipeline de preprocesamiento, Baseline Logístico |
| **5. TRAIN** | 21–24 | Cross-validation de múltiples modelos (Logistic Regression, RF, XGBoost, LightGBM) |
| **6. ITERAR** | 25–38 | Selección del ganador, calibración isotónica, umbral económico óptimo, prior correction, análisis de sensibilidad, restricciones de margen |
| **7. TEST** | 39–44 | Evaluación final: AUC-ROC, gains, lift, feature importance |
| **8. WINNER** | 45–47 | Ranking final, exportación de `pipeline_winner.pkl` |
| **9. PREDICCIÓN** | 48–51 | Aplicación sobre `nuevos_clientes.csv`, cálculo EVR y ROI, exportación |
| **10. CLTV** | 52–66 | Customer Lifetime Value, Expected Value at Risk, matriz de decisión, exportación enriquecida |

### Split temporal

```
Train   2018–2019  →  Calibración  2020  →  Test  2021+
```

### Variables con nulos destacados

| Variable | Nulos |
|----------|-------|
| `QUEJA` | 57% |
| `DAYS_LAST_SERVICE` | 47% |
| `STATUS_SOCIAL` | 22% |
| `GENERO` | ~1.5% |

---

## Modelo ganador — LightGBM

- **Preprocesamiento:** `SimpleImputer(median)` + `StandardScaler` (numéricas) / `SimpleImputer(most_frequent)` + `OneHotEncoder` (categóricas)
- **Calibración:** Isotónica (mejora la fiabilidad de las probabilidades predichas)
- **Prior correction:** ajuste Bayes de la tasa base de churn — el modelo entrena con ~8.8% de churn histórico, pero la prevalencia real estimada puede diferir; las probabilidades se recalibran en tiempo real en la app

---

## Lógica económica

### Coste de revisión compuesto

```
C(n) = BASE × (1 + α)^n
```

- `α = 7%` para modelos A y B
- `α = 10%` para el resto (C–K)
- `n` = número de revisiones acumuladas del cliente

### Coste de marketing

```
CosteMarketing = 0.01 × C(n)   (1% sobre el coste de revisión)
```

### Descuento flota

```
Descuento = €1,000   si n ≥ 5 revisiones   (segmento Alto, Q3/Q4 CLTV)
```

### Expected Value at Risk (EVR)

```
EVR = prob_churn_ajustada × Margen_absoluto(C(n))
```

### CLTV

```
CLTV = Σ_{k=1}^{H} [ C(n+k) × net_margin% × (1 - p)^k ] / (1 + r)^k
```

- `H` = horizonte de revisiones (por defecto 5)
- `r` = tasa de descuento (por defecto 10%)
- `p` = probabilidad de churn ajustada por prior

### Restricciones de margen

- Margen mínimo del concesionario: **30%**
- Comisión de marca: **7%** del ingreso
- Solo se actúa si `ROI = EVR / CosteTotal ≥ ROI_min`

---

## Dashboard — `app.py`

```bash
streamlit run app.py
```

### Sidebar — controles

| Control | Descripción |
|---------|-------------|
| Escenario | Conservador / Base / Agresivo (presets de parámetros) |
| Churn estimado (%) | Prior real de churn en la cartera activa |
| Tasa descuento (%) | Para el cálculo de CLTV |
| Horizonte (revisiones) | Número de revisiones futuras a considerar |
| ROI mínimo | Umbral de rentabilidad para activar intervención |
| Presupuesto máximo | % de la cartera a intervenir |
| Cargar cartera del mes | CSV con nuevos clientes (mismo formato que `predicciones_nuevos_clientes.csv`) |

### Secciones del dashboard

| # | Sección | Contenido |
|---|---------|-----------|
| 01 | **Modelo Predictivo** | ROC, Gains/Lift, métricas de test, feature importance |
| 02 | **Optimización Económica** | Curva beneficio neto vs umbral, breakeven, sensibilidad |
| 03 | **Segmentación Estratégica** | Scatter Customer Value × Churn Risk (Alto/Medio/Bajo) |
| 04 | **Estrategia de Campañas** | Acciones por segmento y cuartil CLTV con recuento de clientes |
| 05 | **Análisis Estratégico** | Comparativa de escenarios, implicaciones para dirección |
| 06 | **Búsqueda Individual** | Ficha por Customer ID: probabilidad, CLTV, EVR, acción recomendada |
| 07 | **Simulador de Campaña** | ¿Qué pasa si contacto N clientes? Proyección de beneficio esperado |

---

## Output principal — `predicciones_nuevos_clientes.csv`

| Columna | Descripción |
|---------|-------------|
| `Customer_ID` | Identificador del cliente |
| `prob_churn` | Probabilidad de churn (modelo calibrado, prior de entrenamiento) |
| `segmento_riesgo` | Alto / Medio / Bajo |
| `cltv` | Customer Lifetime Value (€) |
| `evr` | Expected Value at Risk (€) |
| `roi_intervencion` | ROI esperado de la campaña |
| `accion_recomendada` | Acción comercial asignada |
| `intensidad` | Alta / Media / Baja / Ninguna |

---

## Instalación

```bash
# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias (si no están instaladas)
pip install pandas numpy scikit-learn lightgbm streamlit matplotlib seaborn

# Ejecutar el notebook (pipeline completo)
jupyter notebook sandbox.ipynb

# Lanzar la aplicación
streamlit run app.py
```

> Los archivos de datos se descargan desde Google Cloud Storage con `data.py` (requiere credenciales GCS: `gs://jvelare-public`).

---

## Entregables

1. **`app.py`** — Aplicación interactiva: evaluación del modelo, curvas de probabilidad, segmentación y asignación de campañas
2. **`Presentación.pptx`** — Máximo 5 diapositivas (sin portada ni conclusión): estrategia comercial, evaluación del modelo y justificación económica
3. **`predicciones_nuevos_clientes.csv`** — Predicciones finales con CLTV y plan de acción por cliente
4. **`pipeline_winner.pkl`** — Pipeline serializado listo para staging/producción
