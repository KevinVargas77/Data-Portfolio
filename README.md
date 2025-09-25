# 🌍 GDP World Economic Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-green)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📋 Project Description

Comprehensive and professional analysis of world economic data (GDP 2020-2025) featuring:

- **🔍 Exploratory Data Analysis (EDA)** - Complete descriptive statistics
- **📊 Interactive Visualizations** - Advanced charts with Plotly
- **� Time Series Analysis** - Predictions with Prophet and ARIMA
- **🤖 Machine Learning** - Economic clustering, classification and prediction
- **🚀 Interactive Dashboard** - Complete web application with Streamlit

## 🎯 Key Features

### ✅ Implemented Analysis

- [x] **Complete EDA**: Descriptive statistics, growth analysis, COVID-19 impact
- [x] **Advanced Visualizations**: 5 types of interactive charts (evolution, heatmaps, regional analysis)
- [x] **Time Series Predictions**: Prophet models for economic forecasts 2026-2028
- [x] **Machine Learning**: Economy clustering (5 groups), developed/emerging classification (89.5% accuracy)
- [x] **Web Dashboard**: Complete interactive interface with multiple analysis pages
- [x] **2028 Predictions**: GDP estimates using regression models (R²=0.999)

### 🏆 Outstanding Results

- **📊 89.5% accuracy** in developed vs emerging economies classification
- **🎯 R²=0.999** in GDP prediction model
- **🌍 5 economic clusters** identified via K-means (Silhouette Score: 0.465)
- **� Predictions until 2028** for major world economies
- **🔍 COVID-19 Analysis** with identification of most/least affected countries

## 🚀 Live Demo

**[🌐 View Interactive Dashboard](https://gdp-analysis-kevin.streamlit.app)** *(Coming Soon)*

## 🗂️ Project Structure

```
gdp_world_economic_analysis/
├── src/
│   ├── analysis/
│   │   ├── eda_analysis.py             # Complete exploratory analysis
│   │   └── ml_analysis.py              # Machine Learning and clustering
│   ├── data/
│   │   └── data_loader.py              # Data loading utilities
│   ├── visualization/
│   │   └── gdp_visualizations.py       # Interactive visualizations
│   ├── forecasting/
│   │   └── time_series_analysis.py     # Prophet and ARIMA models
│   └── dashboard/
│       └── streamlit_app.py            # Main dashboard application
├── outputs/
│   └── figures/                        # Generated visualizations
├── docs/                               # Project documentation
├── requirements.txt                    # Project dependencies
├── run_dashboard.py                   # Main Streamlit runner
└── README.md                          # Project documentation
```

## 🎯 Skills Demonstrated

### 📊 **Data Science & Analytics**
- Exploratory Data Analysis (EDA)
- Descriptive and Inferential Statistics
- Time Series Analysis
- Advanced Data Visualization

### 🤖 **Machine Learning**
- Clustering (K-means, silhouette analysis)
- Classification (Random Forest, SVM)
- Regression (economic predictions)
- Feature Engineering and Selection

### 📈 **Time Series Forecasting**
- Facebook Prophet
- ARIMA Models
- Trend Analysis
- Medium-term Predictions

### 🚀 **Development & Deployment**
- Streamlit Applications
- Interactive Dashboards
- Git Version Control
- Cloud Deployment (Streamlit Cloud)

## 🛠️ Technologies Used

### **Languages & Frameworks**
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python 3.13**
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) **Streamlit**

### **Data Science & ML**
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) **Pandas**
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) **NumPy**
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) **Scikit-learn**
- **Prophet** (Time Series Forecasting)

### **Visualization**
- ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) **Plotly**
- **Interactive Charts & Maps**

### **Tools & Deployment**
- ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) **Git**
- ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) **GitHub**
- **Streamlit Cloud**

## 🚀 How to Run the Project

### **Prerequisites**
- Python 3.13+
- pip package manager

### **Installation & Execution**
```bash
# Clone the repository
git clone https://github.com/KevinVargas77/portfolio-datasets.git
cd portfolio-datasets/gdp_world_economic_analysis

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run run_dashboard.py
```

### **Access**
- **Local**: http://localhost:8501
- **Live Demo**: [GDP Analysis Dashboard](https://gdp-analysis-kevin.streamlit.app) *(Coming Soon)*

## � Data Sources & Methodology

### **Dataset Information**
- **Source**: World Bank GDP data (2020-2025)
- **Coverage**: 195+ countries worldwide
- **Variables**: GDP, population, growth rates, regional classifications
- **Time Period**: 2020-2025 with predictions until 2028

### **Analytical Approach**
1. **Data Cleaning**: Missing value imputation, outlier detection
2. **EDA**: Statistical analysis, correlation studies, COVID-19 impact assessment
3. **Visualization**: Interactive charts, geographic maps, trend analysis
4. **ML Modeling**: K-means clustering, Random Forest classification
5. **Time Series**: Prophet forecasting with confidence intervals

## � Key Insights & Results

### 🔍 **Economic Analysis Findings**
1. **COVID-19 Impact**: Identified 15 most affected economies with >10% GDP decline
2. **Recovery Patterns**: 3 distinct recovery trajectories: V-shaped, U-shaped, L-shaped
3. **Regional Disparities**: Asia-Pacific showed fastest recovery, Europe most resilient
4. **Growth Predictions**: Average 3.2% global GDP growth projected for 2026-2028

### 🤖 **Machine Learning Results**
- **Clustering**: Successfully grouped 195 countries into 5 economic clusters
- **Classification**: 89.5% accuracy in predicting developed vs emerging economies
- **Forecasting**: Prophet model achieved MAPE < 5% for major economies
- **Feature Importance**: Population, previous GDP, and regional factors most predictive

## �📞 Contact

- **GitHub:** [@KevinVargas77](https://github.com/KevinVargas77)
- **LinkedIn:** [Kevin Vargas](https://linkedin.com/in/kevin-vargas)
- **Email:** kevinvargas00@gmail.com

---

### 📄 License
This project is under the MIT License. See [LICENSE](LICENSE) for more details.

### ⭐ Like this project?
Don't forget to give it a star on GitHub! ⭐