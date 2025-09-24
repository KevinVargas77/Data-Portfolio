# -*- coding: utf-8 -*-
"""
GDP World Economic Analysis - Machine Learning Analysis
=======================================================

Este módulo aplica técnicas de Machine Learning para clustering de economías,
clasificación y análisis predictivo.

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

# Machine Learning libraries
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score, adjusted_rand_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')

class EconomicMLAnalyzer:
    """Clase para análisis de Machine Learning en datos económicos"""
    
    def __init__(self, df):
        """
        Inicializa el analizador ML
        
        Args:
            df (pd.DataFrame): DataFrame con datos de PIB
        """
        self.df = df.copy()
        self.years = [col for col in df.columns if col.isdigit()]
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepara características para ML"""
        print("🔧 Preparando características para Machine Learning...")
        
        # Convertir a numeric
        for year in self.years:
            self.df[year] = pd.to_numeric(self.df[year], errors='coerce')
        
        # Crear características económicas
        self.features_df = self.df.copy()
        
        # 1. PIB promedio
        gdp_cols = [col for col in self.years if col in self.df.columns]
        self.features_df['GDP_Mean'] = self.df[gdp_cols].mean(axis=1, skipna=True)
        
        # 2. PIB mediano
        self.features_df['GDP_Median'] = self.df[gdp_cols].median(axis=1, skipna=True)
        
        # 3. Volatilidad (desviación estándar)
        self.features_df['GDP_Std'] = self.df[gdp_cols].std(axis=1, skipna=True)
        
        # 4. Coeficiente de variación
        self.features_df['GDP_CV'] = (self.features_df['GDP_Std'] / self.features_df['GDP_Mean']) * 100
        
        # 5. Tasa de crecimiento total
        if len(gdp_cols) >= 2:
            first_year = gdp_cols[0]
            last_year = gdp_cols[-1]
            
            self.features_df['Total_Growth'] = (
                (self.df[last_year] - self.df[first_year]) / self.df[first_year] * 100
            )
            
            # 6. CAGR (Compound Annual Growth Rate)
            years_diff = int(last_year) - int(first_year)
            self.features_df['CAGR'] = (
                ((self.df[last_year] / self.df[first_year]) ** (1/years_diff) - 1) * 100
            )
        
        # 7. PIB más reciente
        latest_year = max(self.years)
        self.features_df['GDP_Latest'] = self.df[latest_year]
        
        # 8. Tendencia (usando regresión lineal simple)
        trend_slopes = []
        for idx, row in self.df.iterrows():
            gdp_values = []
            years_numeric = []
            
            for year in self.years:
                if pd.notna(row[year]):
                    gdp_values.append(row[year])
                    years_numeric.append(int(year))
            
            if len(gdp_values) >= 3:
                # Regresión lineal simple
                X = np.array(years_numeric).reshape(-1, 1)
                y = np.array(gdp_values)
                
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(X, y)
                trend_slopes.append(lr.coef_[0])
            else:
                trend_slopes.append(np.nan)
        
        self.features_df['Trend_Slope'] = trend_slopes
        
        # Limpiar datos faltantes
        self.features_df = self.features_df.dropna(subset=[
            'GDP_Mean', 'GDP_Std', 'Total_Growth', 'CAGR', 'GDP_Latest'
        ])
        
        print(f"✅ Características preparadas para {len(self.features_df)} países")
        
    def perform_clustering_analysis(self, n_clusters=5):
        """Realiza análisis de clustering de economías"""
        print("🎯 Realizando análisis de clustering...")
        
        # Seleccionar características para clustering
        feature_cols = ['GDP_Mean', 'GDP_Std', 'CAGR', 'GDP_Latest']
        X = self.features_df[feature_cols].copy()
        
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. K-Means Clustering
        print(f"🔍 Aplicando K-Means con {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.features_df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calcular silhouette score
        silhouette_avg = silhouette_score(X_scaled, self.features_df['KMeans_Cluster'])
        print(f"📊 Silhouette Score K-Means: {silhouette_avg:.3f}")
        
        # 2. Clustering Jerárquico
        print("🌳 Aplicando Clustering Jerárquico...")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.features_df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)
        
        # 3. PCA para visualización
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        self.features_df['PCA1'] = X_pca[:, 0]
        self.features_df['PCA2'] = X_pca[:, 1]
        
        # Crear visualizaciones
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('K-Means Clustering', 'Clustering Jerárquico'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        # K-Means plot
        for i in range(n_clusters):
            cluster_data = self.features_df[self.features_df['KMeans_Cluster'] == i]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['PCA1'],
                    y=cluster_data['PCA2'],
                    mode='markers',
                    name=f'K-Means Cluster {i}',
                    text=cluster_data['Country'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'PCA1: %{x:.2f}<br>' +
                                'PCA2: %{y:.2f}<br>' +
                                '<extra></extra>',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Hierarchical plot
        for i in range(n_clusters):
            cluster_data = self.features_df[self.features_df['Hierarchical_Cluster'] == i]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['PCA1'],
                    y=cluster_data['PCA2'],
                    mode='markers',
                    name=f'Jerárquico Cluster {i}',
                    text=cluster_data['Country'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'PCA1: %{x:.2f}<br>' +
                                'PCA2: %{y:.2f}<br>' +
                                '<extra></extra>',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=8,
                        opacity=0.7,
                        symbol='diamond'
                    ),
                    showlegend=True
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title={
                'text': "🎯 Clustering de Economías Mundiales",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=600,
            width=1400,
            template='plotly_white'
        )
        
        # Análisis de clusters
        self._analyze_clusters()
        
        return fig
    
    def _analyze_clusters(self):
        """Analiza las características de cada cluster"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS DE CLUSTERS ECONÓMICOS")
        print("="*60)
        
        for cluster in sorted(self.features_df['KMeans_Cluster'].unique()):
            cluster_data = self.features_df[self.features_df['KMeans_Cluster'] == cluster]
            
            print(f"\n🎯 CLUSTER {cluster} ({len(cluster_data)} países):")
            print("-" * 40)
            
            # Países en el cluster
            countries = cluster_data['Country'].tolist()
            print(f"🌍 Países: {', '.join(countries[:5])}" + 
                  (f" y {len(countries)-5} más" if len(countries) > 5 else ""))
            
            # Características promedio
            print(f"💰 PIB promedio: ${cluster_data['GDP_Mean'].mean()/1000:.1f}B")
            print(f"📈 CAGR promedio: {cluster_data['CAGR'].mean():.1f}%")
            print(f"📊 Volatilidad promedio: {cluster_data['GDP_Std'].mean()/1000:.1f}B")
            
            # Clasificar tipo de economía
            avg_gdp = cluster_data['GDP_Mean'].mean()
            avg_growth = cluster_data['CAGR'].mean()
            
            if avg_gdp > 1000000:  # > 1T
                economy_type = "💎 Economías Grandes"
            elif avg_gdp > 100000:  # > 100B
                economy_type = "🏭 Economías Medianas"
            else:
                economy_type = "🌱 Economías Pequeñas"
            
            if avg_growth > 10:
                growth_type = "🚀 Alto Crecimiento"
            elif avg_growth > 3:
                growth_type = "📈 Crecimiento Moderado"
            else:
                growth_type = "🐌 Crecimiento Lento"
            
            print(f"🏷️  Tipo: {economy_type} - {growth_type}")
    
    def classify_economy_development(self):
        """Clasifica economías como desarrolladas/emergentes usando ML"""
        print("🎓 Clasificando economías: Desarrolladas vs Emergentes...")
        
        # Clasificación manual inicial para entrenamiento
        # (Esta es una simplificación - en la realidad usaríamos datos más robustos)
        developed_countries = [
            'United States', 'Germany', 'Japan', 'United Kingdom', 'France',
            'Italy', 'Canada', 'South Korea', 'Australia', 'Spain',
            'Netherlands', 'Switzerland', 'Belgium', 'Austria', 'Sweden',
            'Norway', 'Denmark', 'Finland', 'Ireland', 'New Zealand'
        ]
        
        # Crear etiquetas
        self.features_df['Development_Level'] = self.features_df['Country'].apply(
            lambda x: 'Desarrollada' if x in developed_countries else 'Emergente'
        )
        
        # Preparar datos para entrenamiento
        feature_cols = ['GDP_Mean', 'GDP_Latest', 'CAGR', 'GDP_Std', 'Total_Growth']
        X = self.features_df[feature_cols]
        y = self.features_df['Development_Level']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar Random Forest
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = rf_classifier.predict(X_test_scaled)
        
        # Evaluación
        accuracy = rf_classifier.score(X_test_scaled, y_test)
        print(f"📊 Precisión del modelo: {accuracy:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5)
        print(f"🎯 Cross-validation score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📈 Importancia de características:")
        for _, row in feature_importance.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.3f}")
        
        # Visualización de importancia
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="📈 Importancia de Características para Clasificación",
            labels={'Importance': 'Importancia', 'Feature': 'Característica'}
        )
        
        fig.update_layout(
            height=400,
            width=800,
            template='plotly_white'
        )
        
        return rf_classifier, scaler, fig
    
    def predict_future_gdp(self, target_year=2028):
        """Predice PIB futuro usando ML"""
        print(f"🔮 Prediciendo PIB para {target_year} usando Machine Learning...")
        
        # Preparar datos para entrenamiento
        # Usar años como características y PIB como target
        feature_cols = ['GDP_Mean', 'CAGR', 'Trend_Slope', 'Total_Growth']
        X = self.features_df[feature_cols].copy()
        
        # Target: PIB del último año disponible
        latest_year = max(self.years)
        y = self.features_df['GDP_Latest'].copy()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar múltiples modelos
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0)
        }
        
        best_model = None
        best_score = -float('inf')
        best_name = ""
        
        print("\n🎯 Evaluando modelos:")
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"   {name}: R² = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        print(f"\n🏆 Mejor modelo: {best_name} (R² = {best_score:.3f})")
        
        # Predicciones para todos los países
        X_all_scaled = scaler.transform(X)
        predictions = best_model.predict(X_all_scaled)
        
        # Aplicar factor de crecimiento para el año objetivo
        years_to_predict = target_year - int(latest_year)
        
        # Usar CAGR para proyectar al futuro
        future_predictions = []
        for i, (_, row) in enumerate(self.features_df.iterrows()):
            current_gdp = predictions[i]
            cagr = row['CAGR'] / 100  # Convertir a decimal
            
            # Aplicar crecimiento compuesto
            future_gdp = current_gdp * ((1 + cagr) ** years_to_predict)
            future_predictions.append(future_gdp)
        
        self.features_df[f'GDP_Predicted_{target_year}'] = future_predictions
        
        # Crear visualización de top predicciones
        top_predictions = self.features_df.nlargest(15, f'GDP_Predicted_{target_year}')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_predictions[f'GDP_Predicted_{target_year}'] / 1000,
            y=top_predictions['Country'],
            orientation='h',
            marker=dict(
                color=top_predictions[f'GDP_Predicted_{target_year}'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="PIB Predicho (Billones)")
            ),
            text=[f'${x/1000:.1f}B' for x in top_predictions[f'GDP_Predicted_{target_year}']],
            textposition='inside'
        ))
        
        fig.update_layout(
            title={
                'text': f"🔮 Predicciones PIB para {target_year} - Top 15 Economías",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="PIB Predicho (Billones USD)",
            yaxis_title="País",
            height=600,
            width=1000,
            template='plotly_white'
        )
        
        print(f"\n🎯 TOP 5 PREDICCIONES PIB {target_year}:")
        for i, (_, row) in enumerate(top_predictions.head().iterrows(), 1):
            gdp_pred = row[f'GDP_Predicted_{target_year}'] / 1000
            print(f"   {i}. {row['Country']}: ${gdp_pred:.1f}B")
        
        return best_model, scaler, fig


def load_gdp_data():
    """Carga los datos de PIB"""
    import kagglehub
    import os
    
    path = kagglehub.dataset_download("codebynadiia/gdp-per-country-20202025")
    csv_file = os.path.join(path, "2020-2025.csv")
    return pd.read_csv(csv_file)


def main():
    """Función principal para análisis ML"""
    print("🤖 INICIANDO ANÁLISIS DE MACHINE LEARNING")
    print("="*60)
    
    # Cargar datos
    df = load_gdp_data()
    print(f"📊 Datos cargados: {df.shape[0]} países")
    
    # Crear analizador ML
    analyzer = EconomicMLAnalyzer(df)
    
    # 1. Clustering Analysis
    print("\n1️⃣ ANÁLISIS DE CLUSTERING")
    clustering_fig = analyzer.perform_clustering_analysis(n_clusters=5)
    
    # 2. Classification Analysis
    print("\n2️⃣ CLASIFICACIÓN DE ECONOMÍAS")
    classifier, clf_scaler, importance_fig = analyzer.classify_economy_development()
    
    # 3. GDP Prediction
    print("\n3️⃣ PREDICCIÓN DE PIB")
    predictor, pred_scaler, prediction_fig = analyzer.predict_future_gdp(target_year=2028)
    
    # Guardar resultados
    output_dir = "reports/figures/"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar visualizaciones
    clustering_fig.write_html(f"{output_dir}ml_clustering_analysis.html")
    print(f"💾 Guardado: {output_dir}ml_clustering_analysis.html")
    
    importance_fig.write_html(f"{output_dir}ml_feature_importance.html")
    print(f"💾 Guardado: {output_dir}ml_feature_importance.html")
    
    prediction_fig.write_html(f"{output_dir}ml_gdp_predictions_2028.html")
    print(f"💾 Guardado: {output_dir}ml_gdp_predictions_2028.html")
    
    print(f"\n" + "="*60)
    print("✅ ANÁLISIS DE MACHINE LEARNING COMPLETADO")
    print("🎯 3 modelos entrenados exitosamente")
    print("📊 3 visualizaciones guardadas")
    print("🔜 Siguiente: Dashboard Interactivo Final")

if __name__ == "__main__":
    main()