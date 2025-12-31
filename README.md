# IA Explicable para Clasificación de Radiografías de Tórax

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto final para el curso "Ética y Explicabilidad de la IA" en ICAI - Universidad Pontificia Comillas, Madrid.

## Descripción General

Este proyecto investiga la aplicación de técnicas de **IA Explicable (XAI)** para identificar y diagnosticar posibles problemas de aprendizaje por atajos (shortcut learning) en modelos de deep learning entrenados para clasificación de radiografías de tórax. Utilizando el dataset ChestX-ray14, entrenamos modelos CNN y aplicamos diversos métodos de interpretabilidad (Grad-CAM, análisis de activaciones y sanity checks) para asegurar que los modelos aprenden características clínicamente relevantes en lugar de correlaciones espurias.

### Objetivos Principales

- Entrenar modelos CNN baseline y propensos a atajos para clasificación binaria de radiografías de tórax
- Aplicar técnicas XAI locales y globales para comprender la toma de decisiones del modelo
- Detectar aprendizaje por atajos mediante análisis sistemático de XAI
- Validar la fiabilidad de las interpretaciones mediante sanity checks

## Dataset

El proyecto utiliza el dataset [**chest-xray-pneumonia**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) que contiene 5.863 imágenes de radiografías de tórax frontales. Características del dataset:

- Clasificación binaria (neumonía vs. normal)
- Pacientes únicos no especificados
- Imágenes en formato JPEG con resolución variable

**Nota**: El dataset no está incluido en este repositorio debido a su tamaño. Las instrucciones de descarga se proporcionan en la sección de [Instalación](#instalación).

## Estructura del Proyecto

```
final_project_xai/
├── data/                    # Directorio del dataset (no incluido en el repo)
├── models/                  # Sólo incluimos archivos JSON con configuraciones y resultados. Para replicar los pesos de entrenamiento, ver scripts/train_*.py
├── notebooks/              # Notebooks de Jupyter para análisis
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_results.ipynb
│   ├── 03_corrupted_visualization.ipynb
│   ├── 04_shortcut_comparison.ipynb
│   ├── 05_analysis_overfitting.ipynb
│   ├── 06_local_xai_gradcam.ipynb
│   ├── 07_xai_global_analysis_gradcam.ipynb
│   ├── 08_xai_global_analysis_activation.ipynb
│   └── 09_xai_sanity_checks.ipynb
├── reports/
├── scripts/                # Scripts de entrenamiento
│   ├── train_baseline.py
│   └── train_shortcut.py
├── src/                    # Módulos de código fuente
│   ├── data/              # Carga y preprocesamiento de datos
│   ├── explainability/    # Implementación de XAI
│   ├── models/            # Arquitecturas de modelos
│   └── utils/             # Funciones utilitarias
├── environment.yaml        # Especificación del entorno Conda
├── requirements.txt        # Dependencias de Python
├── final_report.pdf       # Informe completo del proyecto
└── README.md
```

## Metodología

### Entrenamiento de Modelos

1. **Modelo Baseline**: ResNet-18 entrenado en el dataset original chest-xray-pneumonia (preentreno en ImageNet)
2. **Modelo con Atajos**: Modelo intencionalmente expuesto a atajos artificiales (e.g., esquinas coloreadas) para demostrar la detección de shortcuts (sin preentrenamiento para enfatizar el efecto)

### Técnicas XAI Aplicadas

- **Grad-CAM**: Gradient-weighted Class Activation Mapping para explicaciones visuales
- **Análisis de Activaciones**: Investigación de activaciones de capas intermedias (técnica **Activation Maximization**)
- **Sanity Checks**: Tests de aleatorización de modelos y datos para validar métodos de interpretación

### Pipeline de Análisis

1. Exploración y preprocesamiento de datos
2. Entrenamiento de modelos (variantes baseline y con atajos)
3. Evaluación de rendimiento y análisis de overfitting
4. XAI Local: Explicaciones de predicciones individuales
5. XAI Global: Análisis del comportamiento agregado del modelo
6. Sanity checks para fiabilidad de interpretaciones

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- Conda (recomendado) o pip
- GPU compatible con CUDA (opcional, pero recomendada para entrenamiento)

### Pasos de Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone https://github.com/Javirios03/final_project_xai.git
   cd final_project_xai
   ```

2. **Descargar el dataset ChestX-ray14**:

   - Ejecutar el siguiente script para descargar y organizar el dataset:
     ```
     python src/data/download.py
     ```

3. **Crear el entorno** (elegir un método):

   **Opción A: Usando Conda (recomendado)**:

   ```bash
   conda env create -f environment.yaml
   conda activate xai_project
   ```

   **Opción B: Usando pip**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

### Entrenamiento de Modelos

Entrenar el modelo baseline, modificando los hiperparámetros según sea necesario en el propio script:

```bash
python scripts/train_baseline.py
```

Entrenar el modelo con atajos (con corrupciones artificiales), modificando los hiperparámetros según sea necesario en el propio script:

```bash
python scripts/train_shortcut.py
```

### Ejecutar Notebooks de Análisis

Lanza Jupyter y explora los notebooks de análisis:

```bash
jupyter notebook notebooks/
```

**Orden de ejecución recomendado**:

1. `01_data_exploration.ipynb` - Comprender el dataset
2. `02_baseline_results.ipynb` - Evaluar rendimiento baseline
3. `06_local_xai_gradcam.ipynb` - Análisis XAI local
4. `07_xai_global_analysis_gradcam.ipynb` - Patrones XAI globales
5. `09_xai_sanity_checks.ipynb` - Validar interpretaciones

## Resultados Principales

- Implementación exitosa de Grad-CAM para interpretación de radiografías de tórax
- Detección de aprendizaje por atajos mediante análisis sistemático de XAI
- Validación de métodos de interpretación a través de sanity checks comprehensivos
- Identificación de características clínicamente relevantes vs. espurias en predicciones del modelo

Los resultados detallados y visualizaciones están disponibles en el [informe final](final_report.pdf) y en los notebooks de análisis.

## Gestión de Dependencias

### Añadir Nuevos Paquetes

**Mediante Conda**:

```bash
conda activate xai_project
conda install nombre_del_paquete=versión
# Actualizar manualmente environment.yaml
conda env export --no-builds > environment_locked.yml
```

**Mediante pip**:

```bash
pip install nombre_del_paquete==versión
# Actualizar manualmente environment.yaml en la sección pip
pip freeze > requirements_frozen.txt
conda env export --no-builds > environment_locked.yml
```

## Tecnologías Utilizadas

- **Deep Learning**: PyTorch, TorchVision
- **XAI**: Grad-CAM, análisis de activaciones personalizado
- **Procesamiento de Datos**: NumPy, Pandas, Pillow
- **Visualización**: Matplotlib, Seaborn
- **Imagen Médica**: PyDICOM, SimpleITK

## Trabajo Futuro

- Extender a métodos XAI adicionales (SHAP, Integrated Gradients)
- Implementar explicaciones contrafactuales
- Desarrollar dashboard de visualización interactivo
- Aplicar a otras modalidades de imagen médica y a conjuntos de datos más variados

## Autor

**Francisco Javier Ríos Montes**  
Máster en Inteligencia Artificial Avanzada - ICAI, Universidad Pontificia Comillas  
[GitHub](https://github.com/Javirios03) | [LinkedIn](www.linkedin.com/in/francisco-javier-rios)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Citación

Si utilizas este trabajo, por favor cita:

```bibtex
@misc{rios2025xai_chest,
  author = {Ríos Montes, Francisco Javier},
  title = {IA Explicable para Clasificación de Radiografías de Tórax},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Javirios03/final_project_xai}
}
```

---
