# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Time Series Forecasting
=====================================================

Este módulo realiza análisis de series temporales y predicciones de PIB 
usando Prophet, ARIMA y análisis estadístico.

Autor: Kevin
Fecha: Septiembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time series libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """Clase para análisis de series temporales económicas"""
    
    def __init__(self, df):
        """
        Inicializa el analizador de series temporales
        
        Args:
            df (pd.DataFrame): DataFrame con datos de PIB
        """
        self.df = df.copy()
        self.years = [col for col in df.columns if col.isdigit()]
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepara los datos para análisis de series temporales"""
        print("🔧 Preparando datos para análisis temporal...")
        
        # Convertir a numeric
        for year in self.years:
            self.df[year] = pd.to_numeric(self.df[year], errors='coerce')
        
        # Crear DataFrame melted con fechas
        self.df_melted = self.df.melt(
            id_vars=['Country'], 
            value_vars=self.years,
            var_name='Year', 
            value_name='GDP'
        )
        self.df_melted['Year'] = pd.to_numeric(self.df_melted['Year'])
        self.df_melted['GDP'] = pd.to_numeric(self.df_melted['GDP'], errors='coerce')
        
        # Crear fecha (usando enero 1 de cada año)
        self.df_melted['Date'] = pd.to_datetime(self.df_melted['Year'], format='%Y')
        
        print(f"✅ Datos preparados para {len(self.df)} países")
    
    def analyze_global_gdp_trend(self):
        """Analiza la tendencia global del PIB"""
        print("🌍 Analizando tendencia global del PIB...")
        
        # Calcular PIB mundial por año
        global_gdp = self.df_melted.groupby('Year')['GDP'].sum().reset_index()
        global_gdp['Date'] = pd.to_datetime(global_gdp['Year'], format='%Y')
        global_gdp['GDP_Trillions'] = global_gdp['GDP'] / 1_000_000  # Convertir a trillones
        
        # Visualización
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=global_gdp['Date'],
            y=global_gdp['GDP_Trillions'],
            mode='lines+markers',
            name='PIB Mundial',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=10, color='#1f77b4'),
            hovertemplate='<b>PIB Mundial</b><br>' +
                        'Año: %{x|%Y}<br>' +
                        'PIB: $%{y:.2f}T<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "🌍 Evolución del PIB Mundial (2020-2025)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title="Año",
            yaxis_title="PIB Mundial (Trillones USD)",
            template='plotly_white',
            height=500,
            width=1000
        )
        
        return global_gdp, fig
    
    def prophet_forecast(self, country_name, forecast_years=3):
        """
        Realiza predicción usando Prophet para un país específico
        
        Args:
            country_name (str): Nombre del país
            forecast_years (int): Años a predecir hacia el futuro
        """
        print(f"🔮 Realizando predicción Prophet para {country_name}...")
        
        # Obtener datos del país
        country_data = self.df_melted[
            (self.df_melted['Country'] == country_name) & 
            (self.df_melted['GDP'].notna())
        ].copy()
        
        if len(country_data) < 3:
            print(f"⚠️ Datos insuficientes para {country_name}")
            return None, None
        
        # Preparar datos para Prophet
        prophet_data = country_data[['Date', 'GDP']].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data = prophet_data.sort_values('ds')
        
        # Crear y entrenar modelo Prophet
        try:
            model = Prophet(
                yearly_seasonality=True,
                daily_seasonality=False,
                weekly_seasonality=False,
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=10.0
            )
            model.fit(prophet_data)
            
            # Crear fechas futuras
            last_date = prophet_data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=forecast_years,
                freq='YS'  # Año inicio
            )
            
            future_df = pd.DataFrame({'ds': list(prophet_data['ds']) + list(future_dates)})
            
            # Realizar predicción
            forecast = model.predict(future_df)
            
            # Crear visualización
            fig = go.Figure()
            
            # Datos históricos
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'],
                y=prophet_data['y'] / 1000,  # Convertir a billones
                mode='lines+markers',
                name='Datos Históricos',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, color='#1f77b4')
            ))
            
            # Predicciones
            forecast_future = forecast[forecast['ds'] > last_date]
            fig.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat'] / 1000,
                mode='lines+markers',
                name='Predicción',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8, color='#ff7f0e')
            ))
            
            # Intervalo de confianza
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
                y=pd.concat([forecast_future['yhat_upper'] / 1000, forecast_future['yhat_lower'][::-1] / 1000]),
                fill='toself',
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo Confianza',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title={
                    'text': f"🔮 Predicción PIB {country_name} usando Prophet",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Año",
                yaxis_title="PIB (Billones USD)",
                template='plotly_white',
                height=600,
                width=1000,
                hovermode='x unified'
            )
            
            return forecast, fig
            
        except Exception as e:
            print(f"❌ Error en Prophet para {country_name}: {str(e)}")
            return None, None
    
    def arima_forecast(self, country_name, forecast_years=3):
        """
        Realiza predicción usando ARIMA para un país específico
        
        Args:
            country_name (str): Nombre del país
            forecast_years (int): Años a predecir
        """
        print(f"📈 Realizando predicción ARIMA para {country_name}...")
        
        # Obtener datos del país
        country_data = self.df_melted[
            (self.df_melted['Country'] == country_name) & 
            (self.df_melted['GDP'].notna())
        ].copy()
        
        if len(country_data) < 4:
            print(f"⚠️ Datos insuficientes para ARIMA en {country_name}")
            return None, None
        
        # Preparar series temporal
        ts_data = country_data.set_index('Date')['GDP'].sort_index()
        
        try:
            # Test de estacionariedad
            adf_result = adfuller(ts_data)
            print(f"ADF Test p-value: {adf_result[1]:.4f}")
            
            # Auto ARIMA simplificado (probando diferentes órdenes)
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Entrenar mejor modelo
            model = ARIMA(ts_data, order=best_order)
            fitted_model = model.fit()
            
            print(f"Mejor orden ARIMA: {best_order}, AIC: {best_aic:.2f}")
            
            # Realizar predicción
            forecast_result = fitted_model.forecast(steps=forecast_years)
            forecast_ci = fitted_model.get_forecast(steps=forecast_years).conf_int()
            
            # Crear fechas para predicción
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=forecast_years,
                freq='YS'
            )
            
            # Crear visualización
            fig = go.Figure()
            
            # Datos históricos
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data / 1000,
                mode='lines+markers',
                name='Datos Históricos',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=8, color='#2ca02c')
            ))
            
            # Predicciones
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_result / 1000,
                mode='lines+markers',
                name='Predicción ARIMA',
                line=dict(color='#d62728', width=3, dash='dash'),
                marker=dict(size=8, color='#d62728')
            ))
            
            # Intervalo de confianza
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_dates, forecast_dates[::-1]]),
                y=pd.concat([forecast_ci.iloc[:, 1] / 1000, forecast_ci.iloc[:, 0][::-1] / 1000]),
                fill='toself',
                fillcolor='rgba(214,39,40,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo Confianza',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                title={
                    'text': f"📈 Predicción PIB {country_name} usando ARIMA {best_order}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Año",
                yaxis_title="PIB (Billones USD)",
                template='plotly_white',
                height=600,
                width=1000,
                hovermode='x unified'
            )
            
            return fitted_model, fig
            
        except Exception as e:
            print(f"❌ Error en ARIMA para {country_name}: {str(e)}")
            return None, None
    
    def forecast_multiple_countries(self, countries=None, method='prophet'):
        """
        Realiza predicciones para múltiples países
        
        Args:
            countries (list): Lista de países a analizar
            method (str): 'prophet' o 'arima'
        """
        if countries is None:
            # Top 10 economías
            latest_year = max(self.years)
            countries = self.df.nlargest(10, latest_year)['Country'].tolist()
        
        print(f"🎯 Realizando predicciones {method.upper()} para {len(countries)} países...")
        
        results = {}
        successful_forecasts = 0
        
        for country in countries:
            print(f"\n📊 Procesando: {country}")
            
            if method.lower() == 'prophet':
                forecast, fig = self.prophet_forecast(country, forecast_years=3)
            else:
                forecast, fig = self.arima_forecast(country, forecast_years=3)
            
            if forecast is not None:
                results[country] = {'forecast': forecast, 'figure': fig}
                successful_forecasts += 1
                print(f"✅ {country}: Predicción completada")
            else:
                print(f"❌ {country}: Falló la predicción")
        
        print(f"\n🎉 Predicciones completadas: {successful_forecasts}/{len(countries)}")
        return results
    
    def generate_forecast_summary(self, forecast_results):
        """Genera resumen de las predicciones"""
        print("\n" + "="*60)
        print("📋 RESUMEN DE PREDICCIONES ECONÓMICAS")
        print("="*60)
        
        for country, results in forecast_results.items():
            print(f"\n🌍 {country}:")
            
            if 'forecast' in results and results['forecast'] is not None:
                forecast_data = results['forecast']
                
                # Para Prophet
                if 'yhat' in forecast_data.columns:
                    future_forecasts = forecast_data[
                        forecast_data['ds'] > pd.to_datetime('2025-01-01')
                    ]
                    
                    if not future_forecasts.empty:
                        avg_forecast = future_forecasts['yhat'].mean() / 1000
                        print(f"   📈 PIB promedio proyectado: ${avg_forecast:.1f}B")
                        
                        growth_rate = ((future_forecasts['yhat'].iloc[-1] / 
                                      forecast_data[forecast_data['ds'] <= pd.to_datetime('2025-01-01')]['yhat'].iloc[-1]) - 1) * 100
                        print(f"   🚀 Crecimiento proyectado: {growth_rate:.1f}%")
        
        print("\n✅ Resumen de predicciones generado")


def load_gdp_data():
    """Carga los datos de PIB"""
    import kagglehub
    import os
    
    path = kagglehub.dataset_download("codebynadiia/gdp-per-country-20202025")
    csv_file = os.path.join(path, "2020-2025.csv")
    return pd.read_csv(csv_file)


def main():
    """Función principal para análisis de series temporales"""
    print("🔮 INICIANDO ANÁLISIS DE SERIES TEMPORALES")
    print("="*60)
    
    # Cargar datos
    df = load_gdp_data()
    print(f"📊 Datos cargados: {df.shape[0]} países")
    
    # Crear analizador
    analyzer = TimeSeriesAnalyzer(df)
    
    # 1. Análisis global
    global_gdp, global_fig = analyzer.analyze_global_gdp_trend()
    
    # 2. Predicciones para países principales
    top_countries = ['United States', 'China', 'Germany', 'Japan', 'India']
    
    print(f"\n🎯 Realizando predicciones para países principales...")
    
    # Prophet forecasts
    prophet_results = analyzer.forecast_multiple_countries(
        countries=top_countries[:3], 
        method='prophet'
    )
    
    # ARIMA forecasts
    arima_results = analyzer.forecast_multiple_countries(
        countries=top_countries[:2], 
        method='arima'
    )
    
    # 3. Generar resúmenes
    if prophet_results:
        analyzer.generate_forecast_summary(prophet_results)
    
    # 4. Guardar resultados
    output_dir = "reports/figures/"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar gráfico global
    global_fig.write_html(f"{output_dir}gdp_global_trend.html")
    print(f"💾 Guardado: {output_dir}gdp_global_trend.html")
    
    # Guardar predicciones
    saved_count = 0
    for country, results in prophet_results.items():
        if results['figure'] is not None:
            filename = f"{output_dir}forecast_prophet_{country.replace(' ', '_').lower()}.html"
            results['figure'].write_html(filename)
            print(f"💾 Guardado: {filename}")
            saved_count += 1
    
    for country, results in arima_results.items():
        if results['figure'] is not None:
            filename = f"{output_dir}forecast_arima_{country.replace(' ', '_').lower()}.html"
            results['figure'].write_html(filename)
            print(f"💾 Guardado: {filename}")
            saved_count += 1
    
    print(f"\n" + "="*60)
    print("✅ ANÁLISIS DE SERIES TEMPORALES COMPLETADO")
    print(f"📊 {saved_count + 1} visualizaciones guardadas")
    print("🔜 Siguiente: Machine Learning y Clustering")

if __name__ == "__main__":
    main()