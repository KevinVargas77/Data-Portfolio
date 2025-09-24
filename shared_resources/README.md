# 🛠️ Shared Resources

Esta carpeta contiene recursos compartidos entre todos los proyectos del portfolio.

## 📁 Estructura

```
shared_resources/
├── utils/                  # Utilidades comunes
│   ├── __init__.py
│   ├── data_processing.py  # Funciones de procesamiento de datos
│   ├── visualization.py   # Funciones de visualización comunes
│   └── ml_utils.py        # Utilidades de machine learning
├── templates/             # Templates para nuevos proyectos
│   ├── streamlit_template.py
│   ├── requirements_template.txt
│   └── README_template.md
└── assets/               # Recursos estáticos (imágenes, estilos)
    ├── logos/
    └── styles/
```

## 🎯 Propósito

- **Reutilización de código**: Evitar duplicación entre proyectos
- **Consistencia**: Mantener un estilo visual y funcional uniforme
- **Eficiencia**: Acelerar el desarrollo de nuevos dashboards
- **Mantenimiento**: Centralizar funciones comunes para facilitar actualizaciones

## 📝 Cómo usar

```python
# Importar utilidades compartidas en tus proyectos
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_resources'))

from utils.data_processing import clean_data, normalize_columns
from utils.visualization import create_plotly_chart, apply_theme
```