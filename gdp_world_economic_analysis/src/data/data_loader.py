# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Data Loader
==========================================

Módulo centralizado para carga y preparación de datos económicos.
Maneja la descarga desde Kaggle y prepara los datos para análisis.

Autor: Kevin
Fecha: Septiembre 2025
"""

import pandas as pd
import kagglehub
import os
from typing import Tuple

def load_gdp_data() -> Tuple[pd.DataFrame, list]:
    """
    Carga los datos de PIB mundial desde Kaggle
    
    Returns:
        tuple: (DataFrame con datos PIB, lista de años disponibles)
    """
    print("🔄 Cargando datos económicos mundiales...")
    
    try:
        # Download dataset from Kaggle
        path = kagglehub.dataset_download("codebynadiia/gdp-per-country-20202025")
        csv_file = os.path.join(path, "2020-2025.csv")
        
        # Load and prepare data
        df = pd.read_csv(csv_file)
        
        # Get year columns
        years = [col for col in df.columns if col.isdigit()]
        
        # Convert GDP values to numeric
        for year in years:
            df[year] = pd.to_numeric(df[year], errors='coerce')
        
        print(f"✅ Datos cargados: {len(df)} países, período {min(years)}-{max(years)}")
        return df, years
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        raise

if __name__ == "__main__":
    df, years = load_gdp_data()
    print(f"📊 Dataset listo: {df.shape[0]} países, {len(years)} años")