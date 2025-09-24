# 🚀 Template para Nuevo Dashboard Streamlit

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Agregar shared_resources al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared_resources'))

# Configuración de la página
st.set_page_config(
    page_title="[NOMBRE_PROYECTO] Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 [NOMBRE_PROYECTO] Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Agregar controles aquí
    date_range = st.date_input("Rango de fechas")
    
    st.markdown("---")
    st.markdown("### 📈 Métricas Principales")
    # Métricas en sidebar

# Layout principal
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Visualización 1")
    # Agregar gráfico 1

with col2:
    st.subheader("📈 Visualización 2")
    # Agregar gráfico 2

# Sección completa
st.markdown("---")
st.subheader("🔍 Análisis Detallado")

# Pestañas para organizar contenido
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Análisis", "📈 Predicciones"])

with tab1:
    st.markdown("### Overview del proyecto")
    # Contenido del overview

with tab2:
    st.markdown("### Análisis detallado")
    # Contenido del análisis

with tab3:
    st.markdown("### Predicciones y tendencias")
    # Contenido de predicciones

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Desarrollado por <strong>Kevin Vargas</strong> | 
        <a href='https://github.com/KevinVargas77'>GitHub</a> | 
        <a href='mailto:kevinvargas00@gmail.com'>Email</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)