# Directorio de Datos (`/data`)

Este directorio contiene todos los conjuntos de datos utilizados y generados en el proyecto, desde los datos brutos iniciales hasta los resultados y modelos finales.


---
## Descripción de las Carpetas

### 📂 `/raw`
Contiene los datos originales sin procesar, que sirven como la fuente principal para todo el análisis. Esta carpeta debe ser tratada como de solo lectura.

* **Microdatos de la Encuesta Demográfica y de Salud Familiar (ENDES):** Archivos originales obtenidos del Instituto Nacional de Estadística e Informática (INEI). Contienen las respuestas de los encuestados a nivel nacional.
* **Datos Geográficos:** Archivo con la ubicación geográfica (latitud, longitud) de las capitales de provincia y sus respectivos códigos de UBIGEO.

### 🧠 `/model_output`
Almacena el objeto del modelo final, entrenado y serializado (guardado en un archivo).

* **Contenido:** El mejor modelo (`best_model.pkl`) guardado después del proceso de entrenamiento y validación. Esto permite cargar el modelo para hacer predicciones sin necesidad de reentrenarlo.

### 📊 `/reporting`
Contiene los artefactos generados que resumen el rendimiento y las características del modelo.

* **Métricas Guardadas:** Tablas (`.csv`) con los resultados de la validación cruzada y el rendimiento final del modelo.
* **Visualizaciones:** Gráficos (`.png`) como la importancia de las características o la matriz de confusión, guardados para su fácil consulta.

---
## Fuentes de Datos y Referencias

* **Códigos de Ubicación Geográfica (UBIGEO):**
    * [Plataforma Nacional de Datos Abiertos del Perú](https://datosabiertos.gob.pe/dataset/codigos-equivalentes-de-ubigeo-del-peru)

* **Encuesta Demográfica y de Salud Familiar (ENDES):**
* [Sistema de Transferencia de Resultados del INEI](https://proyectos.inei.gob.pe/iinei/srienaho/Consulta_por_Encuesta.asp)