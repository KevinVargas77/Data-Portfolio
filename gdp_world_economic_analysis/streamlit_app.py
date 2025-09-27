#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Streamlit App Entry Point
======================================================

Archivo principal para el deployment en Streamlit Cloud.
Redirige a la aplicación principal en src/dashboard/

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

# Cambiar al directorio del dashboard para rutas relativas
dashboard_path = current_dir / "src" / "dashboard"
os.chdir(dashboard_path)

# Importar y ejecutar la aplicación principal
if __name__ == "__main__":
    # Importar el contenido de la aplicación principal
    exec(open("streamlit_app.py").read())