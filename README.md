# Modelo de Predicción de Bajo Peso al Nacer en Perú

Este repositorio contiene el código y los recursos para desarrollar un modelo de aprendizaje automático que predice la probabilidad de que un recién nacido en Perú tenga bajo peso (menos de 2500 gramos). El objetivo principal es identificar los factores de riesgo más influyentes a partir de la Encuesta Demográfica y de Salud Familiar (ENDES) y otras fuentes de datos socioeconómicos y geográficos.

## 📝 Descripción del Proyecto

El bajo peso al nacer es un problema de salud pública significativo, ya que está asociado con un mayor riesgo de morbilidad y mortalidad infantil. Este proyecto busca abordar este problema mediante la creación de un modelo predictivo que pueda ayudar a identificar a las madres en riesgo y, potencialmente, guiar las intervenciones de salud pública.

El flujo de trabajo principal del proyecto, como se detalla en el notebook `running_model.ipynb`, incluye:

1.  **Carga y preprocesamiento de datos**: Se utilizan datos de la encuesta ENDES de los años 2023 y 2024, junto con datos de ubigeo que contienen información socioeconómica y geográfica a nivel de provincia.
2.  **Ingeniería de características**: Se crean nuevas variables, como la distancia desde el hogar de la encuestada hasta la capital de su provincia.
3.  **Entrenamiento y evaluación del modelo**: Se entrenan y comparan varios modelos de clasificación (Balanced Random Forest, LightGBM, Regresión Logística y XGBoost) para predecir el bajo peso al nacer.
4.  **Análisis de resultados**: Se evalúan los modelos utilizando diversas métricas y se identifican las características más importantes para la predicción.



# 📊 Datos Utilizados
El modelo se basa en las siguientes fuentes de datos:

- **Encuesta Demográfica y de Salud Familiar (ENDES)**: Se utilizan los módulos de los años 2023 y 2024, que contienen información detallada sobre la salud materna e infantil, características del hogar y datos demográficos.

- **Datos de Ubigeo**: Un archivo CSV (`ubigeo_provincia.csv`) que contiene datos socioeconómicos y geográficos a nivel de provincia, como la densidad de población, el Índice de Desarrollo Humano (IDH), y los porcentajes de pobreza.

# Metodología

## 1. Preprocesamiento de Datos
- Se cargan los datos de los diferentes módulos de la ENDES para los años 2023 y 2024 utilizando la clase `EndesProcessor`.
- Los datos de ambos años se combinan en un único DataFrame.
- Se realiza una limpieza de datos, que incluye la eliminación de duplicados y la fusión con los datos socioeconómicos del archivo de ubigeo.

## 2. Ingeniería de Características
- **Distancia a la Capital de la Provincia**: Se calcula la distancia en kilómetros desde la ubicación del hogar de la encuestada hasta la capital de su provincia utilizando las coordenadas de latitud y longitud. Esta variable sirve como un proxy de la accesibilidad a servicios.

## 3. Entrenamiento del Modelo
- **Preparación de Datos**: Los datos se dividen en conjuntos de entrenamiento y prueba. La variable objetivo es `bajo_peso_nacimiento`, una variable binaria que indica si el peso del bebé al nacer fue inferior a 2500 gramos.
  
- **Modelos Utilizados**: Se entrenan y evalúan los siguientes modelos de clasificación:
  - Balanced Random Forest
  - LightGBM
  - Regresión Logística
  - XGBoost

- **Evaluación y Selección del Mejor Modelo**: Los modelos se evalúan utilizando validación cruzada y una variedad de métricas, incluyendo Precisión, Recall (Sensibilidad), Especificidad, F1-Score, AUC-ROC y AUC-PR. El modelo con el mejor rendimiento general (en este caso, XGBoost) se selecciona como el modelo final.

# 📈 Resultados
Los resultados de la evaluación de los modelos se guardan en el archivo `metricas_resultados.xlsx`. El modelo XGBoost demostró ser el más eficaz, con un AUC-ROC del 70.31% y un buen equilibrio entre precisión y recall en comparación con los otros modelos.

## Factores Clave (Importancia de las Características)
El análisis de la importancia de las características del mejor modelo (XGBoost) reveló los factores más determinantes para predecir el bajo peso al nacer. El gráfico de importancia de características se guarda como `feature_importance.jpg`.

# 💾 Modelo Final y Resultados
- **Mejor Modelo**: El objeto del mejor modelo entrenado se guarda en `data/02_model_output/best_model.pkl`. Este archivo puede ser cargado para realizar predicciones sobre nuevos datos.

- **Métricas y Resultados Detallados**: Una hoja de cálculo de Excel (`data/03_reporting/metricas_resultados.xlsx`) contiene:
  - Las métricas de rendimiento de todos los modelos evaluados.
  - Los datos de prueba con las predicciones de probabilidad del mejor modelo.
  - La lista completa de la importancia de las variables.

# 📂 Estructura del Repositorio
```text
.
├── data
│   ├── 01_raw
│   │   └── inei
│   │       ├── 2023
│   │       │   └── modulos_endes
│   │       ├── 2024
│   │       │   └── modulos_endes
│   │       └── geo
│   │           └── ubigeo_provincia.csv
│   ├── 02_model_output
│   │   └── best_model.pkl
│   └── 03_reporting
│       ├── feature_importance.jpg
│       └── metricas_resultados.xlsx
├── notebooks
│   └── running_model.ipynb
└── src
    └── scripts
        ├── getting_modules.py
        ├── merging_data.py
        └── modeler_helper.py