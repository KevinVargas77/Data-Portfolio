#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Dashboard Launcher
=================================================

Script para ejecutar el dashboard de Streamlit de manera profesional.
Incluye manejo de errores y configuración automática.

Uso:
    python run_dashboard.py

Autor: Kevin
Fecha: Septiembre 2025
"""

import subprocess
import sys
import os
import webbrowser
from pathlib import Path

def main():
    """Función principal para lanzar el dashboard"""
    
    print("Iniciando GDP World Economic Analysis Dashboard")
    print("=" * 60)
    
    # Verificar dependencias
    try:
        import streamlit
        import pandas
        import plotly
        import kagglehub
        print("Todas las dependencias instaladas")
    except ImportError as e:
        print(f"Error: Falta dependencia - {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return 1
    
    # Cambiar al directorio del dashboard
    dashboard_path = Path(__file__).parent / "src" / "dashboard"
    
    if not dashboard_path.exists():
        print(f"Error: No se encuentra el directorio {dashboard_path}")
        return 1

    os.chdir(dashboard_path)
    print(f"Cambiando a directorio: {dashboard_path}")

    # Verificar archivo del dashboard
    app_file = dashboard_path / "streamlit_app.py"
    if not app_file.exists():
        print(f"Error: No se encuentra {app_file}")
        return 1

    print("Archivo del dashboard encontrado")    # Configurar puerto
    port = 8502
    
    try:
        print(f"Iniciando servidor en puerto {port}...")
        print("URL Local: http://localhost:8502")
        print("Presiona Ctrl+C para detener")
        print("=" * 60)
        
        # Ejecutar Streamlit
        result = subprocess.run([
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            "streamlit_app.py", 
            "--server.port", str(port),
            "--server.headless", "false"
        ])
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\nDashboard detenido por el usuario")
        return 0
    except Exception as e:
        print(f"\nError al ejecutar dashboard: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)