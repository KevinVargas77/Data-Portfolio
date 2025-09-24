# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Advanced Visualizations
=====================================================

Este módulo crea visualizaciones avanzadas e interactivas para el análisis de PIB mundial.

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
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')

class GDPVisualizer:
    """Clase para crear visualizaciones avanzadas de datos económicos"""
    
    def __init__(self, df):
        """
        Inicializa el visualizador
        
        Args:
            df (pd.DataFrame): DataFrame con datos de PIB
        """
        self.df = df.copy()
        self.years = [col for col in df.columns if col.isdigit()]
        
        # Preparar datos
        self._prepare_data()
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _prepare_data(self):
        """Prepara los datos para visualización"""
        # Convertir a numeric
        for year in self.years:
            self.df[year] = pd.to_numeric(self.df[year], errors='coerce')
        
        # Crear DataFrame melted
        self.df_melted = self.df.melt(
            id_vars=['Country'], 
            value_vars=self.years,
            var_name='Year', 
            value_name='GDP'
        )
        self.df_melted['Year'] = pd.to_numeric(self.df_melted['Year'])
        self.df_melted['GDP'] = pd.to_numeric(self.df_melted['GDP'], errors='coerce')
    
    def create_top_economies_evolution(self):
        """Crea gráfico de evolución de top economías"""
        print("📊 Creando gráfico de evolución de top economías...")
        
        # Obtener top 15 economías del último año
        latest_year = max(self.years)
        top_15_countries = self.df.nlargest(15, latest_year)['Country'].tolist()
        
        # Crear gráfico interactivo
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, country in enumerate(top_15_countries):
            country_data = self.df_melted[self.df_melted['Country'] == country]
            
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['GDP'] / 1000,  # Convertir a billones
                mode='lines+markers',
                name=country,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{country}</b><br>' +
                            'Año: %{x}<br>' +
                            'PIB: $%{y:.1f}B<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': "🌍 Evolución PIB de las Top 15 Economías Mundiales (2020-2025)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title="Año",
            yaxis_title="PIB (Billones USD)",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1200,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def create_gdp_heatmap(self):
        """Crea mapa de calor del PIB por año"""
        print("🔥 Creando mapa de calor del PIB...")
        
        # Preparar datos para heatmap (top 20 países)
        latest_year = max(self.years)
        top_20_countries = self.df.nlargest(20, latest_year)['Country'].tolist()
        
        # Crear matriz para heatmap
        heatmap_data = []
        for country in top_20_countries:
            country_row = self.df[self.df['Country'] == country]
            gdp_values = []
            for year in self.years:
                gdp = country_row[year].iloc[0] if not country_row[year].isna().iloc[0] else 0
                gdp_values.append(gdp / 1000)  # Convertir a billones
            heatmap_data.append(gdp_values)
        
        # Crear heatmap con plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=self.years,
            y=top_20_countries,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b><br>' +
                        'Año: %{x}<br>' +
                        'PIB: $%{z:.1f}B<br>' +
                        '<extra></extra>',
            colorbar=dict(title="PIB (Billones USD)")
        ))
        
        fig.update_layout(
            title={
                'text': "🔥 Mapa de Calor: PIB Top 20 Economías por Año",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Año",
            yaxis_title="País",
            height=800,
            width=1000,
            template='plotly_white'
        )
        
        return fig
    
    def create_covid_impact_chart(self):
        """Crea gráfico de impacto COVID-19"""
        print("🦠 Creando análisis visual del impacto COVID-19...")
        
        if '2020' in self.years and '2021' in self.years:
            # Calcular cambios porcentuales
            df_temp = self.df.copy()
            df_temp['Change_2020_2021'] = ((df_temp['2021'] - df_temp['2020']) / df_temp['2020'] * 100)
            
            # Filtrar países con datos válidos
            df_temp = df_temp.dropna(subset=['Change_2020_2021'])
            
            # Clasificar países
            df_temp['Impact_Category'] = df_temp['Change_2020_2021'].apply(
                lambda x: 'Fuerte Crecimiento' if x > 20 else
                         'Crecimiento Moderado' if x > 5 else
                         'Crecimiento Leve' if x > 0 else
                         'Decrecimiento Leve' if x > -5 else
                         'Fuerte Decrecimiento'
            )
            
            # Crear gráfico
            fig = px.histogram(
                df_temp, 
                x='Change_2020_2021',
                color='Impact_Category',
                title="📊 Distribución del Cambio en PIB (2020-2021) - Impacto COVID-19",
                labels={
                    'Change_2020_2021': 'Cambio en PIB (%)',
                    'count': 'Número de Países'
                },
                color_discrete_map={
                    'Fuerte Decrecimiento': '#ff4444',
                    'Decrecimiento Leve': '#ff8888',
                    'Crecimiento Leve': '#88ff88',
                    'Crecimiento Moderado': '#44ff44',
                    'Fuerte Crecimiento': '#00aa00'
                }
            )
            
            fig.update_layout(
                height=500,
                width=1000,
                template='plotly_white'
            )
            
            return fig
        else:
            print("⚠️ Datos insuficientes para análisis COVID-19")
            return None
    
    def create_growth_comparison(self):
        """Crea comparación de crecimiento entre períodos"""
        print("📈 Creando análisis de crecimiento comparativo...")
        
        # Calcular CAGR para cada país
        growth_data = []
        
        for _, row in self.df.iterrows():
            country = row['Country']
            gdp_values = []
            years_values = []
            
            for year in self.years:
                if pd.notna(row[year]):
                    gdp_values.append(row[year])
                    years_values.append(int(year))
            
            if len(gdp_values) >= 2:
                initial_gdp = gdp_values[0]
                final_gdp = gdp_values[-1]
                num_years = years_values[-1] - years_values[0]
                
                if initial_gdp > 0 and num_years > 0:
                    cagr = ((final_gdp / initial_gdp) ** (1/num_years) - 1) * 100
                    growth_data.append({
                        'Country': country,
                        'CAGR': cagr,
                        'Initial_GDP': initial_gdp / 1000,
                        'Final_GDP': final_gdp / 1000
                    })
        
        growth_df = pd.DataFrame(growth_data)
        
        if not growth_df.empty:
            # Seleccionar top y bottom países
            top_growth = growth_df.nlargest(15, 'CAGR')
            bottom_growth = growth_df.nsmallest(10, 'CAGR')
            
            combined_df = pd.concat([top_growth, bottom_growth])
            
            # Crear gráfico
            fig = px.scatter(
                combined_df,
                x='Initial_GDP',
                y='CAGR',
                size='Final_GDP',
                color='CAGR',
                hover_name='Country',
                title=f"📈 Crecimiento Económico vs PIB Inicial ({min(self.years)}-{max(self.years)})",
                labels={
                    'Initial_GDP': f'PIB Inicial {min(self.years)} (Billones USD)',
                    'CAGR': 'Tasa de Crecimiento Anual Compuesto (%)',
                    'Final_GDP': f'PIB Final {max(self.years)} (Billones USD)'
                },
                color_continuous_scale='RdYlGn',
                size_max=60
            )
            
            fig.update_layout(
                height=600,
                width=1000,
                template='plotly_white'
            )
            
            return fig
        else:
            print("⚠️ No hay suficientes datos para análisis de crecimiento")
            return None
    
    def create_regional_pie_chart(self):
        """Crea gráfico de torta por regiones"""
        print("🗺️ Creando análisis regional...")
        
        # Clasificación regional simplificada
        regions = {
            'América del Norte': ['United States', 'Canada', 'Mexico'],
            'Europa': ['Germany', 'France', 'Italy', 'Spain', 'United Kingdom', 
                      'Russia', 'Netherlands', 'Poland', 'Belgium', 'Sweden',
                      'Norway', 'Denmark', 'Finland', 'Austria', 'Switzerland'],
            'Asia': ['China', 'Japan', 'India', 'South Korea', 'Indonesia', 
                    'Taiwan', 'Thailand', 'Singapore', 'Malaysia', 'Philippines',
                    'Vietnam', 'Bangladesh', 'Pakistan'],
            'América Latina': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru',
                             'Venezuela', 'Ecuador', 'Uruguay', 'Paraguay'],
            'Oriente Medio': ['Saudi Arabia', 'Turkey', 'Israel', 'Iran', 'UAE',
                            'Kuwait', 'Qatar', 'Iraq'],
            'África': ['South Africa', 'Nigeria', 'Egypt', 'Morocco', 'Kenya',
                      'Ethiopia', 'Ghana', 'Algeria'],
            'Oceanía': ['Australia', 'New Zealand']
        }
        
        latest_year = max(self.years)
        regional_gdp = {}
        
        for region, countries in regions.items():
            region_countries = self.df[self.df['Country'].isin(countries)]
            total_gdp = region_countries[latest_year].sum()
            regional_gdp[region] = total_gdp / 1_000_000  # Convertir a trillones
        
        # Filtrar regiones con PIB > 0
        regional_gdp = {k: v for k, v in regional_gdp.items() if v > 0}
        
        # Crear gráfico de torta
        fig = px.pie(
            values=list(regional_gdp.values()),
            names=list(regional_gdp.keys()),
            title=f"🌍 Distribución Regional del PIB Mundial ({latest_year})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                        'PIB: $%{value:.2f}T<br>' +
                        'Porcentaje: %{percent}<br>' +
                        '<extra></extra>'
        )
        
        fig.update_layout(
            height=600,
            width=800,
            template='plotly_white',
            font_size=12
        )
        
        return fig
    
    def generate_all_visualizations(self):
        """Genera todas las visualizaciones y las guarda"""
        print("🎨 Generando todas las visualizaciones...")
        
        figures = {}
        
        # 1. Evolución de top economías
        figures['evolution'] = self.create_top_economies_evolution()
        
        # 2. Mapa de calor
        figures['heatmap'] = self.create_gdp_heatmap()
        
        # 3. Impacto COVID-19
        covid_fig = self.create_covid_impact_chart()
        if covid_fig:
            figures['covid_impact'] = covid_fig
        
        # 4. Comparación de crecimiento
        growth_fig = self.create_growth_comparison()
        if growth_fig:
            figures['growth_comparison'] = growth_fig
        
        # 5. Análisis regional
        figures['regional'] = self.create_regional_pie_chart()
        
        # Guardar visualizaciones como HTML
        output_dir = "reports/figures/"
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            filename = f"{output_dir}gdp_{name}.html"
            fig.write_html(filename)
            print(f"💾 Guardado: {filename}")
        
        print(f"✅ {len(figures)} visualizaciones creadas exitosamente!")
        return figures


def load_gdp_data():
    """Carga los datos de PIB"""
    import kagglehub
    import os
    
    path = kagglehub.dataset_download("codebynadiia/gdp-per-country-20202025")
    csv_file = os.path.join(path, "2020-2025.csv")
    return pd.read_csv(csv_file)


def main():
    """Función principal para generar visualizaciones"""
    print("🎨 INICIANDO CREACIÓN DE VISUALIZACIONES AVANZADAS")
    print("="*60)
    
    # Cargar datos
    df = load_gdp_data()
    print(f"📊 Datos cargados: {df.shape[0]} países")
    
    # Crear visualizador
    visualizer = GDPVisualizer(df)
    
    # Generar todas las visualizaciones
    figures = visualizer.generate_all_visualizations()
    
    # Mostrar una visualización como ejemplo
    if 'evolution' in figures:
        print("\n📈 Mostrando visualización de evolución...")
        figures['evolution'].show()
    
    print("\n" + "="*60)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print("📁 Archivos guardados en: reports/figures/")
    print("🔜 Siguiente: Análisis de Series Temporales")

if __name__ == "__main__":
    main()