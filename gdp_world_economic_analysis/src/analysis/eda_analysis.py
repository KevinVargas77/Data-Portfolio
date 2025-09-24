# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Exploratory Data Analysis (EDA)
==============================================================

Este módulo contiene funciones para realizar análisis exploratorio
completo de los datos de PIB mundial 2020-2025.

Autor: Kevin
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración para visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_gdp_data():
    """Carga los datos de PIB desde el dataset"""
    import kagglehub
    import os
    
    print("🔄 Descargando dataset...")
    # Download dataset
    path = kagglehub.dataset_download("codebynadiia/gdp-per-country-20202025")
    csv_file = os.path.join(path, "2020-2025.csv")
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"✅ Dataset cargado: {df.shape[0]} países, {df.shape[1]} columnas")
    
    return df

def basic_analysis(df):
    """Realiza análisis básico del dataset"""
    print("\n" + "="*60)
    print("📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*60)
    
    years = [col for col in df.columns if col.isdigit()]
    
    # Información general
    print(f"📈 Países analizados: {len(df)}")
    print(f"📅 Período de análisis: {min(years)} - {max(years)}")
    
    # Valores faltantes
    print("\n🔍 VALORES FALTANTES POR AÑO:")
    for year in years:
        missing = df[year].isna().sum()
        missing_pct = (missing / len(df)) * 100
        print(f"  {year}: {missing} países ({missing_pct:.1f}%)")
    
    # Estadísticas descriptivas
    print(f"\n📈 ESTADÍSTICAS BÁSICAS PIB (en millones USD):")
    for year in years:
        if not df[year].isna().all():
            gdp_data = df[year].dropna()
            print(f"  {year}: Media=${gdp_data.mean():.0f}M, Mediana=${gdp_data.median():.0f}M")

def top_economies_analysis(df):
    """Analiza las principales economías"""
    print("\n" + "="*60)
    print("🌍 PRINCIPALES ECONOMÍAS MUNDIALES")
    print("="*60)
    
    years = [col for col in df.columns if col.isdigit()]
    latest_year = max(years)
    
    # Top 15 economías
    top_15 = df.nlargest(15, latest_year)[['Country', latest_year]].dropna()
    
    print(f"\n🏆 TOP 15 ECONOMÍAS ({latest_year}):")
    print("-" * 45)
    
    for i, (_, row) in enumerate(top_15.iterrows(), 1):
        gdp_billions = row[latest_year] / 1000
        print(f"{i:2d}. {row['Country']:<20} ${gdp_billions:>8.1f}B")

def covid_impact_analysis(df):
    """Analiza impacto COVID-19"""
    print("\n" + "="*60)
    print("🦠 ANÁLISIS IMPACTO COVID-19")
    print("="*60)
    
    # Calcular cambio 2020-2021
    if '2020' in df.columns and '2021' in df.columns:
        df_temp = df.copy()
        df_temp['Change_2020_2021'] = ((df_temp['2021'] - df_temp['2020']) / df_temp['2020'] * 100)
        
        # Países más afectados (decrecimiento)
        worst_hit = df_temp.nsmallest(10, 'Change_2020_2021')[['Country', 'Change_2020_2021']].dropna()
        print("\n🔻 Países más afectados (2020-2021):")
        for _, row in worst_hit.iterrows():
            print(f"  {row['Country']:<20} {row['Change_2020_2021']:>6.1f}%")
        
        # Países con mejor recuperación
        best_recovery = df_temp.nlargest(10, 'Change_2020_2021')[['Country', 'Change_2020_2021']].dropna()
        print("\n🔺 Mejor crecimiento (2020-2021):")
        for _, row in best_recovery.iterrows():
            print(f"  {row['Country']:<20} {row['Change_2020_2021']:>6.1f}%")

def growth_analysis(df):
    """Analiza tendencias de crecimiento"""
    print("\n" + "="*60)
    print("📈 ANÁLISIS DE CRECIMIENTO")
    print("="*60)
    
    years = [col for col in df.columns if col.isdigit()]
    
    # Calcular CAGR (Compound Annual Growth Rate)
    growth_data = []
    
    for _, row in df.iterrows():
        country = row['Country']
        first_year_val = None
        last_year_val = None
        
        # Encontrar primer y último valor válido
        for year in years:
            if pd.notna(row[year]):
                if first_year_val is None:
                    first_year_val = row[year]
                    first_year = int(year)
                last_year_val = row[year]
                last_year = int(year)
        
        # Calcular CAGR si hay suficientes datos
        if first_year_val and last_year_val and first_year_val > 0:
            years_diff = last_year - first_year
            if years_diff > 0:
                cagr = ((last_year_val / first_year_val) ** (1/years_diff) - 1) * 100
                growth_data.append({'Country': country, 'CAGR': cagr})
    
    if growth_data:
        growth_df = pd.DataFrame(growth_data)
        
        print(f"\n🚀 TOP 10 PAÍSES CON MAYOR CRECIMIENTO (CAGR):")
        top_growth = growth_df.nlargest(10, 'CAGR')
        for _, row in top_growth.iterrows():
            print(f"  {row['Country']:<20} {row['CAGR']:>6.1f}% anual")

def main():
    """Función principal que ejecuta todo el análisis"""
    print("🚀 INICIANDO ANÁLISIS ECONÓMICO MUNDIAL")
    print("🌍 PIB por país 2020-2025")
    print("="*50)
    
    # Cargar datos
    df = load_gdp_data()
    
    # Ejecutar análisis
    basic_analysis(df)
    top_economies_analysis(df)
    covid_impact_analysis(df)
    growth_analysis(df)
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print("📊 Este es el primer paso del proyecto completo")
    print("🔜 Próximos pasos: Visualizaciones, Series Temporales, ML, Dashboard")

if __name__ == "__main__":
    main()