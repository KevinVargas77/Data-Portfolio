# 🌍 GDP World Economic Analysis - Portfolio Project

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-green)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📋 Descripción del Proyecto

Análisis completo y profesional de datos económicos mundiales (PIB 2020-2025) que incluye:

- **🔍 Análisis Exploratorio de Datos (EDA)** - Estadísticas descriptivas completas
- **📊 Visualizaciones Interactivas** - Gráficos avanzados con Plotly 
- **📈 Series Temporales** - Predicciones con Prophet y ARIMA
- **🤖 Machine Learning** - Clustering, clasificación y predicción económica
- **🚀 Dashboard Interactivo** - Aplicación web completa con Streamlit

## 🎯 Características Principales

### ✅ Análisis Implementados

- [x] **EDA Completo**: Estadísticas descriptivas, análisis de crecimiento, impacto COVID-19
- [x] **Visualizaciones Avanzadas**: 5 tipos de gráficos interactivos (evolución, mapas de calor, análisis regional)
- [x] **Predicciones Temporales**: Modelos Prophet para predicciones económicas 2026-2028
- [x] **Machine Learning**: Clustering de economías (5 grupos), clasificación desarrolladas/emergentes (89.5% precisión)
- [x] **Dashboard Web**: Interfaz interactiva completa con múltiples páginas de análisis
- [x] **Predicciones 2028**: Estimaciones PIB usando modelos de regresión (R²=0.999)

### 🏆 Resultados Destacados

- **📊 89.5% precisión** en clasificación de economías desarrolladas vs emergentes
- **🎯 R²=0.999** en modelo de predicción de PIB 
- **🌍 5 clusters económicos** identificados via K-means (Silhouette Score: 0.465)
- **📈 Predicciones hasta 2028** para las principales economías mundiales
- **🔍 Análisis COVID-19** con identificación de países más/menos afectados

## 🗂️ Estructura del Proyecto

```
gdp-world-economic-analysis/
├── data/
│   └── external/
│       └── gdp_per_country.py          # Script inicial de carga de datos
├── src/
│   ├── analysis/
│   │   ├── eda_analysis.py             # Análisis exploratorio completo
│   │   └── ml_analysis.py              # Machine Learning y clustering
│   ├── visualization/
│   │   └── gdp_visualizations.py       # Visualizaciones interactivas
│   ├── forecasting/
│   │   └── time_series_analysis.py     # Series temporales y predicción
│   └── dashboard/
│       └── streamlit_app.py            # Dashboard web interactivo
├── reports/
│   └── figures/                        # Resultados HTML generados
├── notebooks/                          # Jupyter notebooks de desarrollo
├── tests/                              # Tests unitarios
├── requirements.txt                    # Dependencias del proyecto
├── pyproject.toml                      # Configuración del proyecto
└── README.md                           # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.11+ (recomendado 3.13)
- pip package manager
- Conexión a internet (para descarga de datasets)

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd gdp-world-economic-analysis
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Dependencias principales

```
streamlit>=1.28.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.15.0
prophet>=1.1.4
scikit-learn>=1.3.0
kagglehub>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
```

## 🎮 Uso del Proyecto

### 🖥️ Ejecutar Dashboard Completo

```bash
cd src/dashboard
streamlit run streamlit_app.py
```

### 📊 Ejecutar Análisis Individuales

```bash
# EDA Completo
python src/analysis/eda_analysis.py

# Visualizaciones
python src/visualization/gdp_visualizations.py

# Machine Learning
python src/analysis/ml_analysis.py

# Series Temporales
python src/forecasting/time_series_analysis.py
```

## 📈 Componentes del Análisis

### 1. 🔍 Análisis Exploratorio (EDA)

**Archivo**: `src/analysis/eda_analysis.py`

- Estadísticas descriptivas completas
- Análisis de crecimiento económico por país
- Impacto económico de COVID-19 (2020-2021)
- Análisis regional y por continentes
- Top economías mundiales y su evolución

### 2. 📊 Visualizaciones Interactivas

**Archivo**: `src/visualization/gdp_visualizations.py`

- **Evolución Temporal**: Líneas interactivas de las top economías
- **Mapas de Calor**: Crecimiento anual por país
- **Análisis COVID-19**: Impacto económico de la pandemia
- **Comparación Regional**: PIB por regiones geográficas
- **Rankings Dinámicos**: Evolución de posiciones económicas

### 3. 📈 Series Temporales y Predicciones

**Archivo**: `src/forecasting/time_series_analysis.py`

- **Prophet Models**: Predicciones económicas 2026-2028
- **Análisis de Tendencias**: Estacionalidad y crecimiento
- **Validación de Modelos**: MAE, MAPE, R² scores
- **Predicciones Específicas**: USA, China, India, Alemania, Japón

### 4. 🤖 Machine Learning Avanzado

**Archivo**: `src/analysis/ml_analysis.py`

**Clustering Económico**:
- K-means con 5 clusters económicos
- Análisis de silueta (Score: 0.465)
- Identificación de patrones económicos

**Clasificación Supervisada**:
- Random Forest: Economías desarrolladas vs emergentes
- Precisión: 89.5%
- Features: PIB, crecimiento, estabilidad

**Predicción de PIB 2028**:
- Ridge Regression (R²=0.999)
- Top predicciones: USA $37.8T, China $22.0T, India $5.4T

### 5. 🚀 Dashboard Web Interactivo

**Archivo**: `src/dashboard/streamlit_app.py`

**Páginas del Dashboard**:
- **📊 Resumen Ejecutivo**: Métricas clave y visión general
- **🌍 Principales Economías**: Top países y evolución temporal
- **🦠 Impacto COVID-19**: Análisis específico de la pandemia
- **🗺️ Análisis Regional**: Distribución geográfica del PIB
- **📈 Comparaciones Personalizadas**: Herramientas interactivas
- **🔮 Predicciones y Tendencias**: Proyecciones futuras

## 🎯 Resultados y Insights

### 📊 Principales Hallazgos

1. **Impacto COVID-19**: Identificación de países más resilientes y afectados
2. **Clusters Económicos**: 5 grupos distintos con características únicas
3. **Predicciones 2028**: Crecimiento proyectado para economías emergentes
4. **Tendencias Regionales**: Asia-Pacífico como motor de crecimiento

### 🏆 Métricas de Rendimiento

- **Clasificación ML**: 89.5% precisión en economías desarrolladas/emergentes
- **Predicción PIB**: R²=0.999 en modelos de regresión
- **Clustering**: Silhouette Score 0.465 (excelente separación)
- **Forecasting**: MAPE < 15% en predicciones Prophet

## 🛠️ Tecnologías Utilizadas

### 🐍 Core Python Stack
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación numérica
- **Matplotlib/Seaborn**: Visualización estadística

### 📊 Análisis Avanzado
- **Plotly**: Visualizaciones interactivas
- **Prophet**: Predicción de series temporales
- **Statsmodels**: Modelos estadísticos y ARIMA

### 🤖 Machine Learning
- **Scikit-learn**: ML algorithms y preprocessing
- **KMeans**: Clustering no supervisado
- **Random Forest**: Clasificación supervisada
- **Ridge Regression**: Predicción numérica

### 🚀 Deployment
- **Streamlit**: Framework de dashboard web
- **Kagglehub**: Integración de datasets

## 🔄 Workflow de Desarrollo

### 1. **Data Acquisition** ⬇️
```python
import kagglehub
path = kagglehub.dataset_download("codebynadiia/gdp-per-country-20202025")
```

### 2. **Exploratory Data Analysis** 🔍
```python
analyzer = GDPAnalyzer(df)
analyzer.generate_comprehensive_report()
```

### 3. **Interactive Visualizations** 📊
```python
visualizer = GDPVisualizer(df)
visualizer.create_all_visualizations()
```

### 4. **Time Series Forecasting** 📈
```python
ts_analyzer = TimeSeriesAnalyzer(df)
predictions = ts_analyzer.forecast_with_prophet()
```

### 5. **Machine Learning Analysis** 🤖
```python
ml_analyzer = EconomicMLAnalyzer(df)
ml_analyzer.perform_complete_analysis()
```

### 6. **Interactive Dashboard** 🚀
```python
streamlit run streamlit_app.py
```

## 📝 Próximos Pasos

### 🔮 Mejoras Futuras

- [ ] **Datos en Tiempo Real**: Integración con APIs económicas
- [ ] **Más Indicadores**: Inflación, desempleo, trade balance
- [ ] **Modelos Avanzados**: Deep Learning para predicciones
- [ ] **Análisis de Sentimientos**: Noticias económicas y impacto
- [ ] **Deployment en Cloud**: AWS/Azure/GCP hosting

### 🛠️ Optimizaciones Técnicas

- [ ] **Caching Avanzado**: Redis para datos frecuentes
- [ ] **Paralelización**: Multiprocessing para análisis pesados
- [ ] **Tests Automatizados**: Pytest suite completa
- [ ] **CI/CD Pipeline**: GitHub Actions deployment
- [ ] **Docker Containerization**: Deployment simplificado

## 👤 Autor

**Kevin** - Data Science Portfolio Project
- 📧 Email: [kevin@example.com](mailto:kevin@example.com)
- 🔗 LinkedIn: [linkedin.com/in/kevin](https://linkedin.com/in/kevin)
- 🐱 GitHub: [github.com/kevin](https://github.com/kevin)

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **Dataset**: Kaggle - "GDP per Country 2020-2025" by CodeByNadiia
- **Frameworks**: Streamlit, Prophet, Plotly communities
- **Inspiration**: Economic data analysis best practices

---

<div align="center">

### 🌟 Si te gusta este proyecto, ¡dale una estrella! ⭐

**🎯 Proyecto Portfolio Completo de Ciencia de Datos**

*Análisis Económico Mundial con Python • Machine Learning • Visualización Interactiva*

</div>

