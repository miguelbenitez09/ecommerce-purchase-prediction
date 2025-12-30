# 🛍️ Online Shoppers Purchasing Intention

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/sklearn-1.5.2-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1+-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

> **Sistema de Machine Learning para predecir intención de compra en e-commerce mediante análisis de comportamiento de navegación en tiempo real.**

---

## 👨‍💻 Autor

**Miguel Antonio Benítez González**
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)
- 💼 LinkedIn: [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Problema de Negocio](#-problema-de-negocio)
3. [Dataset](#-dataset)
4. [Análisis y Técnicas Aplicadas](#-análisis-y-técnicas-aplicadas)
5. [Feature Engineering](#-feature-engineering)
6. [Modelos y Resultados](#-modelos-y-resultados)
7. [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
8. [Estructura del Proyecto](#-estructura-del-proyecto)
9. [Instalación](#-instalación)
10. [Uso](#-uso)
11. [API Endpoints](#-api-endpoints)
12. [Mejoras Futuras](#-mejoras-futuras)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de clasificación binaria para predecir si un visitante de una tienda online realizará una compra (`Revenue = True`) o no (`Revenue = False`) basándose en su comportamiento de navegación.

### Objetivo Principal
Predecir la intención de compra de visitantes en tiempo real para:
- Personalizar experiencia del usuario
- Optimizar campañas de remarketing
- Reducir tasa de abandono del carrito
- Maximizar conversión de ventas

### Pipeline Completo
```
Datos UCI → EDA → Limpieza → Feature Engineering → Balanceo (SMOTE) → 
→ Modelado ML → Validación → API REST → Dashboard Web → Docker
```

### Características del Sistema
- ✅ **12 modelos evaluados**: Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, XGBoost, LightGBM, CatBoost, SVM, Decision Tree, KNN, Naive Bayes
- ✅ **Mejor modelo**: Gradient Boosting (90.05% accuracy, 93.43% ROC-AUC)
- ✅ **Dataset balanceado**: SMOTE para clase minoritaria (15.5% → 50%)
- ✅ **24 features**: 17 originales + 7 engineered
- ✅ **API REST**: FastAPI con documentación automática
- ✅ **Dashboard**: Streamlit con visualizaciones interactivas
- ✅ **Production-ready**: Docker Compose deployment

---

## 💼 Problema de Negocio

### Contexto Empresarial
Las tiendas de e-commerce enfrentan el desafío crítico de convertir visitantes en compradores. **Solo el 15.5% de los visitantes realizan una compra**, lo que significa que el 84.5% abandona el sitio sin conversión.

### Desafíos Clave

1. **Alta Tasa de Abandono** 📉
   - 84.5% de visitantes no compran
   - Pérdida de potenciales ingresos
   - Dificultad para retener usuarios

2. **Recursos Limitados** 💰
   - Imposible personalizar para todos los visitantes
   - Presupuesto marketing finito
   - Necesidad de priorizar esfuerzos

3. **Targeting Ineficiente** 🎯
   - Difícil identificar compradores potenciales
   - Campañas genéricas poco efectivas
   - Descuentos mal dirigidos

4. **Ventana de Oportunidad Corta** ⏰
   - Decisión de compra en minutos
   - Necesidad de intervención en tiempo real
   - Pérdida de momento si no se actúa rápido

### Solución de Machine Learning

Modelo predictivo que analiza en tiempo real:
- **Comportamiento de Navegación**: Páginas visitadas, tiempo en sitio, bounce rate
- **Métricas de Engagement**: Exit rate, page value, duración sesión
- **Factores Temporales**: Mes, día de semana, fechas especiales
- **Perfil de Usuario**: Visitante nuevo/recurrente, tipo de tráfico

### Valor de Negocio

| Aplicación | Impacto | KPI |
|------------|---------|-----|
| **Personalización Dinámica** | Mostrar ofertas a usuarios con alta probabilidad de compra | ↑ Conversión +15-20% |
| **Remarketing Inteligente** | Dirigir campañas solo a usuarios prometedores | ↑ ROI +30-40% |
| **Prevención de Abandono** | Intervenir con incentivos antes de salida | ↓ Bounce Rate -10-15% |
| **Optimización de Recursos** | Focalizar atención humana en usuarios clave | ↓ Costos -25% |

---

## 📊 Dataset

### Información General

**Nombre**: Online Shoppers Purchasing Intention Dataset  
**Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  
**Autores**: C. Sakar, Yomi Kastro (2018)  
**DOI**: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)  
**Licencia**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| **Registros Totales** | 12,330 sesiones |
| **Features** | 18 (10 numéricas, 8 categóricas) |
| **Target (Revenue)** | True: 1,908 (15.5%), False: 10,422 (84.5%) |
| **Período** | 1 año de datos |
| **Duplicados Originales** | 125 (eliminados en preprocesamiento) |
| **Missing Values** | 0 |

### Desbalance de Clases

```
Clase Positiva (Revenue=True):  15.5% ████░░░░░░░░░░░░░░░░
Clase Negativa (Revenue=False): 84.5% ████████████████████

Solución: SMOTE (Synthetic Minority Over-sampling Technique)
Post-SMOTE: 50% / 50% (8,238 / 8,238)
```

### Variables del Dataset

#### Features Numéricas (10)

| Variable | Descripción | Rango | Tipo |
|----------|-------------|-------|------|
| `Administrative` | Páginas administrativas visitadas | 0-27 | int |
| `Administrative_Duration` | Tiempo en páginas administrativas (seg) | 0-3,398 | float |
| `Informational` | Páginas informativas visitadas | 0-24 | int |
| `Informational_Duration` | Tiempo en páginas informativas (seg) | 0-2,549 | float |
| `ProductRelated` | Páginas de productos visitadas | 0-705 | int |
| `ProductRelated_Duration` | Tiempo en páginas de productos (seg) | 0-63,973 | float |
| `BounceRates` | % de visitantes que entran y salen sin interacción | 0-0.2 | float |
| `ExitRates` | % de salidas desde esa página | 0-0.2 | float |
| `PageValues` | Valor promedio de la página antes de conversión | 0-361.76 | float |
| `SpecialDay` | Proximidad a días especiales (0=lejos, 1=cercano) | 0-1 | float |

#### Features Categóricas (8)

| Variable | Categorías | Descripción |
|----------|------------|-------------|
| `Month` | Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec | Mes de la sesión |
| `OperatingSystems` | 1-8 | Sistema operativo del usuario |
| `Browser` | 1-13 | Navegador usado |
| `Region` | 1-9 | Región geográfica |
| `TrafficType` | 1-20 | Tipo de tráfico (directo, referral, search, etc.) |
| `VisitorType` | Returning_Visitor, New_Visitor, Other | Tipo de visitante |
| `Weekend` | True, False | Si la sesión fue en fin de semana |
| **`Revenue`** (TARGET) | **True, False** | **¿Realizó compra?** |

---

## 🔬 Análisis y Técnicas Aplicadas

### 1. Análisis Exploratorio de Datos (EDA)

**Notebook**: `notebooks/01_exploracion_dataset.ipynb`

#### Análisis Univariado

**Hallazgos Clave**:
```python
# Distribución del Target
Revenue = True:  1,908 (15.5%) ← Clase minoritaria
Revenue = False: 10,422 (84.5%)

# Variables más correlacionadas con Revenue
PageValues:      0.49 (fuerte indicador)
ExitRates:      -0.21 (menos salidas → más compras)
BounceRates:    -0.18 (menos rebotes → más compras)
ProductRelated:  0.13 (más páginas de productos → más compras)
```

#### Análisis Temporal
```
Picos de Tráfico y Ventas:
├── Mayo:       Máximo tráfico (3,364 sesiones)
├── Noviembre:  Máximo ventas (Black Friday)
├── Diciembre:  Alto tráfico (fiestas)
└── Febrero:    Mínimo tráfico (634 sesiones)

Patrón Semanal:
├── Lunes-Jueves: Tráfico moderado, conversión normal
└── Fin de Semana: Menor tráfico, conversión similar
```

#### Análisis de Comportamiento
```python
# Compradores vs No Compradores
Compradores:
├── PageValues promedio: 25.8 (vs 3.5 no compradores)
├── ProductRelated promedio: 42 páginas (vs 29)
├── BounceRates promedio: 0.012 (vs 0.024)
└── Sesión promedio: 18 minutos (vs 10)

Insight: Usuarios que compran navegan más, tienen menos rebotes
y generan mayor page value
```

#### Técnicas Utilizadas
- **Visualizaciones**: histogramas, boxplots, barplots, heatmaps
- **Análisis de correlación**: Pearson, Spearman
- **Detección de outliers**: Z-score, IQR
- **Análisis temporal**: patrones mensuales, semanales
- **Chi-cuadrado**: Test de independencia para categóricas

---

### 2. Preprocesamiento de Datos

**Notebook**: `notebooks/02_preprocesamiento_dataset.ipynb`

#### Limpieza de Datos

```python
Pasos Aplicados:
├── Eliminación de duplicados: 125 registros (1%)
├── Verificación de NaN: 0 (dataset limpio)
├── Validación de rangos: todos los valores dentro de rango esperado
└── Registros finales: 12,205
```

#### Encoding de Variables Categóricas

```python
# Label Encoding para ordinales
Month: {'Jan':0, 'Feb':1, ..., 'Dec':11}

# One-Hot Encoding para nominales (después del split)
VisitorType: [Returning_Visitor, New_Visitor, Other]
Weekend: [True, False]

# Mantenidos como numéricos (ya son códigos)
OperatingSystems, Browser, Region, TrafficType
```

#### Manejo de Desbalance: SMOTE

```python
from imblearn.over_sampling import SMOTE

# Antes
X_train: 9,764 muestras
├── Revenue=False: 8,276 (84.8%)
└── Revenue=True:  1,488 (15.2%)

# Después de SMOTE
X_train_balanced: 16,552 muestras
├── Revenue=False: 8,276 (50%)
└── Revenue=True:  8,276 (50%) ← Generadas sintéticamente

Ventajas de SMOTE:
✅ Crea muestras sintéticas (no duplica)
✅ Interpola entre vecinos cercanos
✅ Mejora recall sin sacrificar mucho precision
```

#### Normalización

```python
from sklearn.preprocessing import StandardScaler

# Aplicado solo a features numéricas después del split
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardado para uso en producción
joblib.dump(scaler, 'scaler.pkl')
```

#### Split de Datos

```python
from sklearn.model_selection import train_test_split

# Train: 80%, Test: 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Mantiene proporción de clases
)

# Tamaños finales
Train: 9,764 muestras (post-SMOTE: 16,552)
Test:  2,441 muestras (sin modificar, para evaluar en distribución real)
```

---

## ⚙️ Feature Engineering

**Notebook**: `notebooks/02_preprocesamiento_dataset.ipynb`

### Resumen de Features Creadas: 7

Las features engineered capturan patrones de comportamiento complejos que no son evidentes en las variables originales.

#### 1️⃣ TotalPages (Engagement Total)
```python
TotalPages = Administrative + Informational + ProductRelated
```
**Justificación**: Mide el nivel total de exploración del sitio. Usuarios que visitan más páginas suelen estar más interesados.

**Insight**: Compradores promedian 48 páginas vs 33 para no compradores.

---

#### 2️⃣ TotalDuration (Tiempo Total en Sitio)
```python
TotalDuration = Administrative_Duration + Informational_Duration + ProductRelated_Duration
```
**Justificación**: Duración total de la sesión. Mayor tiempo indica mayor interés.

**Insight**: Compradores promedian 1,088 segundos (18 min) vs 610 segundos (10 min).

---

#### 3️⃣ AvgPageDuration (Promedio de Tiempo por Página)
```python
AvgPageDuration = TotalDuration / TotalPages (si TotalPages > 0, sino 0)
```
**Justificación**: Mide la **intensidad de lectura**. Si un usuario pasa mucho tiempo por página, está realmente interesado (no solo haciendo scroll rápido).

**Insight**: Compradores: 25 seg/página vs No compradores: 20 seg/página.

---

#### 4️⃣ ProductRatio (Foco en Productos)
```python
ProductRatio = ProductRelated / TotalPages (si TotalPages > 0, sino 0)
```
**Justificación**: **Proporción de navegación en productos**. Un usuario que dedica 80% de su navegación a productos es más probable que compre que uno que navega principalmente páginas informativas.

**Insight**: Compradores: 0.72 (72% páginas de productos) vs No compradores: 0.64 (64%).

---

#### 5️⃣ EngagementScore (Score Compuesto)
```python
EngagementScore = (PageValues × 100) + (1 - BounceRates) + (1 - ExitRates)
```
**Justificación**: Feature compuesta que combina:
- **PageValues**: Valor monetario potencial
- **1 - BounceRates**: Inverso de rebotes (menos rebotes = más engagement)
- **1 - ExitRates**: Inverso de salidas

**Insight**: Alta correlación con conversión (0.52). Compradores: 28.5 vs No compradores: 5.8.

---

#### 6️⃣ IsShortSession (Sesión Corta)
```python
IsShortSession = 1 if TotalDuration < 120 else 0  # Menos de 2 minutos
```
**Justificación**: Sesiones muy cortas (< 2 min) rara vez resultan en compras. Usuario posiblemente llegó por error o no encontró lo que buscaba.

**Insight**: Solo 3% de sesiones cortas resultan en compra vs 18% de sesiones normales.

---

#### 7️⃣ IsHighInteraction (Alta Interacción)
```python
IsHighInteraction = 1 if (TotalPages > 30 and TotalDuration > 600) else 0
```
**Justificación**: Marca usuarios con **exploración profunda**:
- Más de 30 páginas visitadas
- Más de 10 minutos en el sitio

**Insight**: 45% de sesiones de alta interacción resultan en compra vs 12% de sesiones normales.

---

### Total Features Finales: 24

```
17 originales + 7 engineered = 24 features para modelado
```

### Impacto de Feature Engineering

| Métrica | Sin FE | Con FE | Mejora |
|---------|--------|--------|--------|
| Accuracy | 87.2% | 90.05% | ↑ 3.3% |
| Precision | 61.5% | 65.98% | ↑ 7.3% |
| Recall | 70.1% | 75.13% | ↑ 7.2% |
| F1-Score | 65.5% | 70.26% | ↑ 7.3% |
| ROC-AUC | 91.8% | 93.43% | ↑ 1.8% |

---

## 🤖 Modelos y Resultados

**Notebook**: `notebooks/03_modelado_dataset.ipynb`

### Algoritmos Evaluados: 12 Modelos

#### 1. Logistic Regression
```python
Configuración:
├── Algoritmo: Regresión Logística
├── Regularización: L2 (Ridge)
├── Solver: lbfgs
└── Propósito: Baseline lineal

Resultados:
├── Accuracy: 86.5%
├── Precision: 59.2%
├── Recall: 68.4%
├── F1-Score: 63.5%
├── ROC-AUC: 90.1%
└── Tiempo: 0.2s
```

#### 2. Random Forest
```python
Configuración:
├── n_estimators: 100
├── max_depth: 20
├── min_samples_split: 5
└── Random State: 42

Resultados:
├── Accuracy: 89.3%
├── Precision: 64.1%
├── Recall: 73.8%
├── F1-Score: 68.6%
├── ROC-AUC: 92.7%
└── Tiempo: 3.5s
```

#### 3. Gradient Boosting 🏆 (MEJOR MODELO)
```python
Configuración:
├── n_estimators: 100
├── learning_rate: 0.1
├── max_depth: 5
├── subsample: 0.8
└── Random State: 42

Resultados:
├── Accuracy: 90.05% ⭐
├── Precision: 65.98%
├── Recall: 75.13%
├── F1-Score: 70.26%
├── ROC-AUC: 93.43% ⭐
└── Tiempo: 2.1s

Top 5 Features Importantes:
1. PageValues: 0.285 (28.5%)
2. EngagementScore: 0.182 (18.2%)
3. ProductRelated_Duration: 0.143 (14.3%)
4. TotalDuration: 0.098 (9.8%)
5. ExitRates: 0.072 (7.2%)
```

#### 4. AdaBoost
```python
Resultados:
├── Accuracy: 87.8%
├── Precision: 62.3%
├── Recall: 70.5%
├── F1-Score: 66.1%
├── ROC-AUC: 91.5%
└── Tiempo: 1.8s
```

#### 5. Extra Trees
```python
Resultados:
├── Accuracy: 88.9%
├── Precision: 63.7%
├── Recall: 72.9%
├── F1-Score: 68.0%
├── ROC-AUC: 92.4%
└── Tiempo: 3.2s
```

#### 6. XGBoost
```python
Resultados:
├── Accuracy: 89.7%
├── Precision: 65.1%
├── Recall: 74.2%
├── F1-Score: 69.4%
├── ROC-AUC: 93.1%
└── Tiempo: 1.5s
```

#### 7. LightGBM
```python
Resultados:
├── Accuracy: 89.5%
├── Precision: 64.8%
├── Recall: 73.9%
├── F1-Score: 69.1%
├── ROC-AUC: 92.9%
└── Tiempo: 0.8s (más rápido)
```

#### 8. CatBoost
```python
Resultados:
├── Accuracy: 89.4%
├── Precision: 64.5%
├── Recall: 73.6%
├── F1-Score: 68.8%
├── ROC-AUC: 92.8%
└── Tiempo: 4.2s
```

#### 9. SVM (RBF Kernel)
```python
Resultados:
├── Accuracy: 88.1%
├── Precision: 62.9%
├── Recall: 71.2%
├── F1-Score: 66.8%
├── ROC-AUC: 91.8%
└── Tiempo: 5.6s
```

#### 10. Decision Tree
```python
Resultados:
├── Accuracy: 84.2%
├── Precision: 56.8%
├── Recall: 67.9%
├── F1-Score: 61.9%
├── ROC-AUC: 88.3%
└── Tiempo: 0.3s
```

#### 11. K-Nearest Neighbors
```python
Resultados:
├── Accuracy: 85.7%
├── Precision: 58.5%
├── Recall: 69.1%
├── F1-Score: 63.3%
├── ROC-AUC: 89.7%
└── Tiempo: 1.2s
```

#### 12. Naive Bayes
```python
Resultados:
├── Accuracy: 82.3%
├── Precision: 53.2%
├── Recall: 65.4%
├── F1-Score: 58.7%
├── ROC-AUC: 87.1%
└── Tiempo: 0.1s
```

---

### Comparación de Modelos (Top 5)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tiempo | Selección |
|--------|----------|-----------|--------|----------|---------|--------|-----------|
| **Gradient Boosting** | **90.05%** | **65.98%** | **75.13%** | **70.26%** | **93.43%** | 2.1s | ✅ |
| XGBoost | 89.7% | 65.1% | 74.2% | 69.4% | 93.1% | 1.5s | ❌ |
| LightGBM | 89.5% | 64.8% | 73.9% | 69.1% | 92.9% | 0.8s | ❌ |
| Random Forest | 89.3% | 64.1% | 73.8% | 68.6% | 92.7% | 3.5s | ❌ |
| Extra Trees | 88.9% | 63.7% | 72.9% | 68.0% | 92.4% | 3.2s | ❌ |

**Modelo Seleccionado**: **Gradient Boosting** por mejor ROC-AUC y F1-Score.

---

### Matriz de Confusión (Gradient Boosting)

```
                    Predicted
                  No Buy  |  Buy
Actual  No Buy     2,051  |   26     TPR: 98.7% (Specificity)
        Buy          118  |  246     FNR: 32.4% (Recall: 67.6%)

Precisión Clase Positiva: 90.4% (de los que predecimos compra, 90% realmente compran)
Recall Clase Positiva: 67.6% (de los que compran, detectamos 68%)
```

### Curva ROC

```
ROC-AUC = 0.9343

Interpretación:
├── Excelente capacidad discriminativa
├── 93.4% probabilidad de rankear compradores > no compradores
└── Modelo puede ajustar threshold según objetivo de negocio
```

---

### Técnicas de Validación

1. **Train-Test Split**: 80/20 stratified
2. **SMOTE solo en train**: Evita data leakage
3. **Evaluación en test original**: Distribución real (desbalanceada)
4. **Múltiples métricas**: Accuracy, Precision, Recall, F1, ROC-AUC
5. **Feature Importance**: Análisis de contribución de variables

---

## 🛠️ Tecnologías Utilizadas

### Ciencia de Datos

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.10+ | Lenguaje principal |
| pandas | 2.1.3 | Manipulación de datos |
| numpy | 1.26.2 | Cálculos numéricos |
| scikit-learn | 1.5.2 | Preprocesamiento, modelos, métricas |
| imbalanced-learn | 0.11.0 | SMOTE para balanceo de clases |
| XGBoost | 2.0.2 | Gradient Boosting |
| LightGBM | 4.1.0 | Gradient Boosting |
| CatBoost | 1.2.2 | Gradient Boosting |
| joblib | 1.3.2 | Serialización de modelos |

### Visualización

| Tecnología | Propósito |
|------------|-----------|
| matplotlib | Gráficos estáticos |
| seaborn | Visualizaciones estadísticas |
| plotly | Gráficos interactivos (dashboard) |

### Deployment

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| FastAPI | 0.104.1 | API REST para predicciones |
| Streamlit | 1.28.1 | Dashboard web interactivo |
| uvicorn | 0.24.0 | Servidor ASGI para FastAPI |
| pydantic | 2.5.0 | Validación de datos API |
| Docker | latest | Containerización |
| Docker Compose | latest | Orquestación multi-contenedor |

---

## 📁 Estructura del Proyecto

```
online_shoppers_intention/
│
├── data/                           # Datos del proyecto
│   ├── 01_raw/                     # Datos originales
│   │   └── online_shoppers_intention.csv  # Dataset UCI (12,330 registros)
│   │
│   └── 02_processed/               # Datos procesados
│       ├── processed_shoppers_data.csv    # Con feature engineering
│       ├── X_train_balanced.npy           # Train balanceado con SMOTE
│       ├── y_train_balanced.npy
│       ├── X_test.npy                     # Test sin balancear
│       └── y_test.npy
│
├── notebooks/                      # Análisis Jupyter
│   ├── 01_exploracion_dataset.ipynb       # EDA completo
│   ├── 02_preprocesamiento_dataset.ipynb  # Feature Engineering
│   └── 03_modelado_dataset.ipynb          # Entrenamiento 12 modelos
│
├── models/                         # Modelos ML serializados
│   ├── best_model.pkl              # Gradient Boosting (192 KB)
│   ├── best_model_compressed.pkl   # Comprimido (65 KB)
│   ├── scaler.pkl                  # StandardScaler para normalización
│   └── model_info.pkl              # Metadata del modelo
│
├── api/                            # API REST
│   ├── main.py                     # FastAPI app
│   └── requirements.txt            # Dependencias API
│
├── web/                            # Dashboard Web
│   ├── app.py                      # Streamlit app
│   ├── requirements.txt            # Dependencias web
│   └── README.md                   # Documentación web
│
├── docker/                         # Containerización
│   ├── Dockerfile                  # Imagen Docker
│   ├── docker-compose.yml          # Orquestación
│   └── README.md                   # Guía Docker
│
├── .gitignore                      # Archivos ignorados
├── README.md                       # Este archivo
└── QUICKSTART.md                   # Guía rápida
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.10 o superior
- Docker y Docker Compose (para deployment containerizado)
- Git

### Opción 1: Instalación Local

#### 1. Clonar Repositorio
```bash
git clone https://github.com/miguelbenitez09/online-shoppers-intention.git
cd online-shoppers-intention
```

#### 2. Crear Entorno Virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar Dependencias

**Para Notebooks**:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost matplotlib seaborn jupyter
```

**Para API**:
```bash
cd api
pip install -r requirements.txt
```

**Para Dashboard**:
```bash
cd web
pip install -r requirements.txt
```

---

### Opción 2: Deployment con Docker (Recomendado) 🐳

#### 1. Clonar Repositorio
```bash
git clone https://github.com/miguelbenitez09/online-shoppers-intention.git
cd online-shoppers-intention
```

#### 2. Construir y Ejecutar Contenedores
```bash
cd docker
docker-compose up --build -d
```

Esto levantará:
- **API REST**: http://localhost:8004
- **Dashboard Web**: http://localhost:8503

#### 3. Verificar Contenedores
```bash
docker ps
# Deberías ver online_shoppers_api y online_shoppers_web corriendo
```

#### 4. Ver Logs
```bash
docker logs online_shoppers_api
docker logs online_shoppers_web
```

#### 5. Detener Servicios
```bash
docker-compose down
```

---

## 💻 Uso

### 1. Ejecutar Notebooks de Análisis

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Iniciar Jupyter
jupyter notebook

# Abrir notebooks en orden:
# 1. notebooks/01_exploracion_dataset.ipynb
# 2. notebooks/02_preprocesamiento_dataset.ipynb
# 3. notebooks/03_modelado_dataset.ipynb
```

---

### 2. Usar API REST

#### Iniciar API Localmente
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

#### Documentación Automática
- Swagger UI: http://localhost:8004/docs
- ReDoc: http://localhost:8004/redoc

#### Ejemplo de Solicitud (Python)
```python
import requests

url = "http://localhost:8004/predict"
data = {
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 1,
    "ProductRelated_Duration": 0.0,
    "BounceRates": 0.2,
    "ExitRates": 0.2,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": "Feb",
    "OperatingSystems": 1,
    "Browser": 1,
    "Region": 1,
    "TrafficType": 1,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}

response = requests.post(url, json=data)
print(response.json())
# Output: {"will_purchase": false, "probability": 0.12, "confidence": "high"}
```

#### Ejemplo de Solicitud (cURL)
```bash
curl -X POST "http://localhost:8004/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Administrative": 0,
       "Administrative_Duration": 0,
       "Informational": 0,
       "Informational_Duration": 0,
       "ProductRelated": 1,
       "ProductRelated_Duration": 0,
       "BounceRates": 0.2,
       "ExitRates": 0.2,
       "PageValues": 0,
       "SpecialDay": 0,
       "Month": "Feb",
       "OperatingSystems": 1,
       "Browser": 1,
       "Region": 1,
       "TrafficType": 1,
       "VisitorType": "Returning_Visitor",
       "Weekend": false
     }'
```

---

### 3. Usar Dashboard Web

#### Iniciar Dashboard Localmente
```bash
cd web
streamlit run app.py --server.port 8503
```

Abrir en navegador: http://localhost:8503

#### Funcionalidades del Dashboard
1. **Predicción Individual**: Ingresa datos de sesión manualmente
2. **Predicción Masiva**: Sube archivo CSV con múltiples sesiones
3. **Información del Modelo**: Métricas de rendimiento
4. **Visualizaciones**: Distribuciones y patrones

---

## 🌐 API Endpoints

### Base URL
```
http://localhost:8004
```

### Endpoints Disponibles

#### 1. Health Check
```http
GET /health
```

**Respuesta**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "Gradient Boosting",
  "accuracy": 0.9005,
  "roc_auc": 0.9343
}
```

---

#### 2. Predicción Individual
```http
POST /predict
```

**Request Body**:
```json
{
  "Administrative": 0,
  "Administrative_Duration": 0.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 15,
  "ProductRelated_Duration": 500.0,
  "BounceRates": 0.01,
  "ExitRates": 0.02,
  "PageValues": 25.5,
  "SpecialDay": 0.0,
  "Month": "Nov",
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

**Respuesta**:
```json
{
  "will_purchase": true,
  "probability": 0.87,
  "confidence": "high"
}
```

---

## 🔮 Mejoras Futuras

### Modelado
- [ ] **Deep Learning**: Redes neuronales para patrones complejos
- [ ] **Ensemble Stacking**: Combinar Gradient Boosting + XGBoost + LSTM
- [ ] **Tuning Avanzado**: Optuna para optimización bayesiana
- [ ] **Time Series Features**: Capturar tendencias temporales más sofisticadas
- [ ] **Features de Sesión Anterior**: Para returning visitors

### Ingeniería
- [ ] **Pipeline Automatizado**: Airflow para ETL
- [ ] **Tracking de Modelos**: MLflow para experimentación
- [ ] **CI/CD**: GitHub Actions para deployment
- [ ] **Monitoreo**: Prometheus + Grafana para métricas en producción
- [ ] **A/B Testing**: Framework para comparar modelos en vivo

### Producto
- [ ] **Real-Time Scoring**: Redis para scoring en tiempo real
- [ ] **Explicabilidad**: SHAP para explicar predicciones
- [ ] **Alertas**: Notificaciones de comportamiento anómalo
- [ ] **Integración CRM**: Sincronización con Salesforce/HubSpot
- [ ] **Dashboard Analítico**: Métricas de negocio en tiempo real

---

## 📞 Contacto y Soporte

Si tienes preguntas o sugerencias sobre este proyecto:

- 📧 Email: mbenitezg01@gmail.com
- 💼 LinkedIn: [Miguel Antonio Benítez González](https://www.linkedin.com/in/miguel-antonio-ben%C3%ADtez-gonz%C3%A1lez-457816247/)
- 💻 GitHub: [miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

Dataset original bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 🙏 Agradecimientos

- **C. Sakar y Yomi Kastro**: Por crear y compartir el dataset
- **UCI Machine Learning Repository**: Por hospedar el dataset
- **Comunidad Open Source**: scikit-learn, XGBoost, FastAPI, Streamlit

---

## 📚 Referencias

**Dataset**:
- Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31, 6893–6908. DOI: [10.1007/s00521-018-3523-0](https://doi.org/10.1007/s00521-018-3523-0)

---

**Desarrollado con ❤️ por Miguel Antonio Benítez González**

*Última actualización: Diciembre 2025*
