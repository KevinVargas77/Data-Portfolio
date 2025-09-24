# Guía de Despliegue en Streamlit Community Cloud

## 📋 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Preparación del Proyecto](#preparación-del-proyecto)
3. [Configuración de GitHub](#configuración-de-github)
4. [Registro en Streamlit Community Cloud](#registro-en-streamlit-community-cloud)
5. [Despliegue de la Aplicación](#despliegue-de-la-aplicación)
6. [Configuración Avanzada](#configuración-avanzada)
7. [Solución de Problemas](#solución-de-problemas)
8. [Mantenimiento y Actualizaciones](#mantenimiento-y-actualizaciones)

---

## 🚀 Introducción

**Streamlit Community Cloud** es la plataforma gratuita oficial de Streamlit que permite hospedar aplicaciones web de manera sencilla y sin costo. Esta guía te llevará paso a paso para publicar tu dashboard de análisis económico del PIB mundial.

### ✅ Ventajas de Streamlit Community Cloud
- **Completamente gratuito**
- **Despliegue automático** desde GitHub
- **SSL incluido** (HTTPS automático)
- **Escalabilidad automática**
- **Integración perfecta** con repositorios públicos de GitHub

---

## 🛠️ Preparación del Proyecto

### 1. Verificar la Estructura del Proyecto

Tu proyecto debe tener esta estructura:
```
gdp_world_economic_analysis/
├── src/
│   └── dashboard/
│       └── streamlit_app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml (opcional)
```

### 2. Actualizar requirements.txt

Asegúrate de que tu archivo `requirements.txt` contenga todas las dependencias:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
kaggle>=1.5.16
scikit-learn>=1.3.0
prophet>=1.1.4
openpyxl>=3.1.0
```

### 3. Crear un archivo .streamlit/config.toml (Opcional)

```toml
[theme]
primaryColor = "#3A86FF"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501
```

### 4. Optimizar el Código para Producción

**Importante**: Verifica que tu aplicación use caching adecuadamente:
```python
@st.cache_data
def load_gdp_data():
    # Tu código de carga de datos
    pass
```

---

## 🐙 Configuración de GitHub

### 1. Crear un Repositorio en GitHub

1. Ve a **GitHub.com** y haz login
2. Haz clic en **"New repository"**
3. Nombra el repositorio: `gdp-world-economic-analysis`
4. Selecciona **"Public"** (obligatorio para Streamlit Community Cloud gratuito)
5. Haz clic en **"Create repository"**

### 2. Subir tu Proyecto a GitHub

#### Opción A: Desde la línea de comandos
```bash
cd "C:\Users\kevin\OneDrive\Escritorio\Kevin_Learning_Lab\DATA_Studies\datasets_test\gdp_world_economic_analysis"

# Inicializar Git (si no está inicializado)
git init

# Agregar archivos
git add .
git commit -m "Initial commit: GDP World Economic Analysis Dashboard"

# Conectar con GitHub
git remote add origin https://github.com/TU_USUARIO/gdp-world-economic-analysis.git
git branch -M main
git push -u origin main
```

#### Opción B: Desde GitHub Desktop
1. Abre **GitHub Desktop**
2. File → Add Local Repository
3. Selecciona tu carpeta del proyecto
4. Publish repository
5. Asegúrate de que sea **público**

### 3. Verificar que los Archivos Estén Subidos

Confirma que estos archivos estén en tu repositorio de GitHub:
- ✅ `src/dashboard/streamlit_app.py`
- ✅ `requirements.txt`
- ✅ `README.md`

---

## 🌐 Registro en Streamlit Community Cloud

### 1. Acceder a Streamlit Community Cloud

1. Ve a **https://share.streamlit.io/**
2. Haz clic en **"Get started"** o **"Sign up"**

### 2. Autenticación con GitHub

1. Selecciona **"Continue with GitHub"**
2. Autoriza a Streamlit para acceder a tus repositorios públicos
3. Completa tu perfil si es necesario

### 3. Conectar tu Cuenta

Una vez autenticado, verás tu dashboard de Streamlit Community Cloud con opciones para crear nuevas aplicaciones.

---

## 🚀 Despliegue de la Aplicación

### 1. Crear Nueva Aplicación

1. En tu dashboard de Streamlit Community Cloud, haz clic en **"New app"**
2. Selecciona **"From existing repo"**

### 2. Configurar el Despliegue

Completa los campos:

- **Repository**: `TU_USUARIO/gdp-world-economic-analysis`
- **Branch**: `main`
- **Main file path**: `src/dashboard/streamlit_app.py`
- **App URL**: `gdp-analysis-kevin` (o el nombre que prefieras)

### 3. Variables de Entorno (si necesarias)

Si tu app requiere API keys o variables de entorno:
1. Haz clic en **"Advanced settings"**
2. Agrega las variables necesarias:
   ```
   KAGGLE_USERNAME = tu_usuario_kaggle
   KAGGLE_KEY = tu_api_key_kaggle
   ```

### 4. Desplegar

1. Haz clic en **"Deploy!"**
2. Espera a que Streamlit construya y despliegue tu aplicación
3. El proceso puede tomar 2-5 minutos

---

## ⚙️ Configuración Avanzada

### 1. Configuración de Secretos

Para datos sensibles, usa Streamlit Secrets:

1. En tu app desplegada, ve a **Settings** → **Secrets**
2. Agrega tus secretos en formato TOML:
```toml
[kaggle]
username = "tu_usuario"
key = "tu_api_key"

[database]
host = "tu_host"
password = "tu_password"
```

3. En tu código, accede a ellos:
```python
import streamlit as st

kaggle_user = st.secrets["kaggle"]["username"]
kaggle_key = st.secrets["kaggle"]["key"]
```

### 2. Configuración de Memoria y Recursos

Para aplicaciones que requieren más recursos:
```toml
# En .streamlit/config.toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
```

### 3. Configuración de Caching

Optimiza el rendimiento:
```python
# Para datos que cambian poco
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    return pd.read_csv("data.csv")

# Para operaciones costosas
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
```

---

## 🔧 Solución de Problemas

### Errores Comunes y Soluciones

#### 1. Error: "requirements.txt not found"
```bash
# Asegúrate de que requirements.txt esté en la raíz del proyecto
touch requirements.txt
git add requirements.txt
git commit -m "Add requirements.txt"
git push
```

#### 2. Error: ModuleNotFoundError
```txt
# Agrega la dependencia faltante a requirements.txt
numpy>=1.24.0
pandas>=2.0.0
streamlit>=1.28.0
```

#### 3. Error: "File not found"
```python
# Usa rutas relativas correctas
df = pd.read_csv("src/data/gdp_data.csv")  # ✅ Correcto
df = pd.read_csv("C:/Users/.../gdp_data.csv")  # ❌ Incorrecto
```

#### 4. Error de Memoria
```python
# Optimiza el uso de memoria
@st.cache_data
def process_large_data(df):
    # Procesa en chunks
    return df.sample(n=10000)  # Muestra solo una parte
```

#### 5. Error de Timeout
```python
# Reduce el tiempo de carga
@st.cache_data(ttl=600)  # Cache por 10 minutos
def load_data():
    # Carga solo los datos necesarios
    return df.iloc[:1000]  # Limita filas
```

### Logs y Debugging

1. **Ver logs en tiempo real**: En tu app, ve a **Settings** → **Logs**
2. **Debugging local**: Usa `streamlit run --logger.level debug`
3. **Testing local**: Siempre prueba localmente antes de desplegar

---

## 🔄 Mantenimiento y Actualizaciones

### 1. Actualizaciones Automáticas

Streamlit Community Cloud redespliega automáticamente cuando haces push a GitHub:

```bash
# Hacer cambios localmente
git add .
git commit -m "Update dashboard colors"
git push origin main
# La app se actualiza automáticamente en 2-3 minutos
```

### 2. Rollback a Versión Anterior

Si algo sale mal:
1. En GitHub, ve a tu repositorio
2. Ve a **Commits**
3. Haz clic en **"Revert"** en el commit problemático
4. La app se revertirá automáticamente

### 3. Monitoreo de la Aplicación

#### Métricas Disponibles:
- **Visitors**: Número de visitantes únicos
- **Sessions**: Sesiones activas
- **Resource usage**: Uso de CPU y memoria

#### Panel de Control:
1. Ve a **https://share.streamlit.io/**
2. Haz clic en tu aplicación
3. Ve a **Analytics** para ver estadísticas

### 4. Configuración de Dominio Personalizado

Para un dominio personalizado (opcional):
1. Compra un dominio
2. Configura CNAME apuntando a tu app de Streamlit
3. Contacta soporte de Streamlit para configuración SSL

---

## 📝 Checklist Final

### Antes del Despliegue:
- [ ] ✅ Proyecto funcionando localmente
- [ ] ✅ requirements.txt actualizado
- [ ] ✅ Código optimizado con @st.cache_data
- [ ] ✅ Repositorio público en GitHub
- [ ] ✅ README.md descriptivo

### Durante el Despliegue:
- [ ] ✅ Cuenta creada en Streamlit Community Cloud
- [ ] ✅ App configurada correctamente
- [ ] ✅ Variables de entorno configuradas
- [ ] ✅ Despliegue exitoso

### Después del Despliegue:
- [ ] ✅ App funcional y accesible
- [ ] ✅ Todas las funcionalidades working
- [ ] ✅ Performance aceptable
- [ ] ✅ URL compartida y documentada

---

## 🎯 Tu URL Final

Una vez desplegado, tu dashboard estará disponible en:
```
https://gdp-analysis-kevin.streamlit.app/
```

**¡Tu dashboard de análisis económico del PIB mundial estará disponible públicamente y podrás compartirlo con empleadores, colegas y la comunidad!**

---

## 📞 Soporte y Recursos

### Recursos Oficiales:
- **Documentación**: https://docs.streamlit.io/streamlit-community-cloud
- **Foro de la Comunidad**: https://discuss.streamlit.io/
- **GitHub Issues**: https://github.com/streamlit/streamlit/issues

### Contacto:
- **Kevin Vargas** - Autor del Dashboard
- **LinkedIn**: https://www.linkedin.com/in/kevargas/
- **Portfolio**: Tu GDP Analysis Dashboard

---

*Guía creada para el despliegue del proyecto "GDP World Economic Analysis Dashboard" - Septiembre 2025*