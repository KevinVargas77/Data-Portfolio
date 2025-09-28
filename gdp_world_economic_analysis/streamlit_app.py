#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Streamlit Dashboard
==================================================

Dashboard interactivo para análisis económico mundial del PIB.
Incluye visualizaciones, análisis de machine learning y forecasting.

Autor: Kevin Vargas
Fecha: Septiembre 2025
"""

import sys
import os
from pathlib import Path

# Agregar el directorio src al path para importaciones
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Configurar el directorio de trabajo
os.chdir(current_dir)

# Importaciones principales
import streamlit as st
import pandas as pd
import numpy as np

# Verificar si plotly está disponible y manejar error graciosamente
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("📦 **Error de dependencia**: Plotly no está instalado correctamente.")
    st.info("💡 **Solución**: Instalando plotly automáticamente...")
    
    # Intentar instalación automática
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.15.0"])
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
        st.success("✅ Plotly instalado exitosamente. Recarga la página.")
    except Exception as e:
        st.error(f"❌ No se pudo instalar plotly: {e}")
        PLOTLY_AVAILABLE = False

if PLOTLY_AVAILABLE:
    # Importar y ejecutar la aplicación principal
    dashboard_app_path = current_dir / "src" / "dashboard" / "streamlit_app.py"
    
    if dashboard_app_path.exists():
        # Leer y ejecutar el contenido de la aplicación principal
        with open(dashboard_app_path, 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Remover las importaciones que ya hicimos aquí para evitar conflictos
        lines = app_content.split('\n')
        filtered_lines = []
        skip_imports = False
        
        for line in lines:
            if line.strip().startswith('import streamlit as st'):
                continue
            elif line.strip().startswith('import pandas as pd'):
                continue
            elif line.strip().startswith('import numpy as np'):
                continue
            elif line.strip().startswith('import plotly'):
                continue
            elif line.strip().startswith('from plotly'):
                continue
            else:
                filtered_lines.append(line)
        
        # Ejecutar el código filtrado
        exec('\n'.join(filtered_lines))
    else:
        st.error("❌ No se encontró el archivo principal del dashboard.")
        st.info(f"📂 Buscando en: {dashboard_app_path}")
else:
    st.error("❌ No se puede ejecutar el dashboard sin plotly.")
    st.info("🔧 Por favor, contacta al administrador para resolver las dependencias.")