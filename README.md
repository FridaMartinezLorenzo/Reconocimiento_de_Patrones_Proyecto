# Proyecto de Reconocimiento de Patrones  
## Fase 1: Análisis Exploratorio de Datos (EDA)

Este repositorio corresponde a la **primera fase del proyecto de la materia de Reconocimiento de Patrones**, cuyo objetivo principal es realizar un **análisis exploratorio de datos (Exploratory Data Analysis, EDA)** sobre datasets relacionados con el **estrés en estudiantes y en contextos fisiológicos**, con el fin de comprender la estructura de los datos, identificar patrones iniciales y preparar la información para etapas posteriores de modelado.

---

## 📌 Objetivo de la Fase 1

El objetivo de esta fase es:

- Comprender la naturaleza de los datos
- Analizar variables relevantes y su distribución
- Identificar valores faltantes, outliers y posibles sesgos
- Explorar relaciones entre variables
- Construir un **pipeline de análisis exploratorio reproducible**

Esta fase **no incluye entrenamiento de modelos**, sino que sienta las bases para las siguientes etapas del proyecto.

---

## 📊 Datasets Utilizados

Ambos datasets fueron obtenidos de **Kaggle**.

### a) Student Stress Monitoring Dataset
Dataset enfocado en el monitoreo del estrés en estudiantes, el cual incluye variables relacionadas con:

- Hábitos de estudio
- Sueño
- Actividad física
- Presión académica
- Factores emocionales y de estilo de vida

Este conjunto de datos permite analizar cómo diferentes aspectos de la vida estudiantil se relacionan con los niveles de estrés.

---

### b) WESAD (Wearable Stress and Affect Detection) Dataset
Dataset orientado a la detección de estrés y estados afectivos mediante **sensores fisiológicos** obtenidos de dispositivos wearables, tales como:

- Frecuencia cardíaca
- Conductancia de la piel
- Temperatura
- Señales fisiológicas continuas

Este dataset es más complejo y permite explorar patrones fisiológicos asociados al estrés.

---

## 🔁 Pipeline de Análisis Exploratorio

Para ambos datasets se siguió el mismo **flujo general de análisis**, adaptado según las características de cada conjunto de datos:

1. **Carga de datos**
   - Importación de archivos desde Kaggle
   - Lectura y verificación del formato de los datos

2. **Inspección inicial**
   - Visualización de las primeras filas
   - Revisión de tipos de datos
   - Dimensionalidad del dataset
3. **Visualización**
   - Histogramas
   - Boxplots
   - Gráficas de correlación
   - Visualizaciones comparativas entre variables relevantes

4. **Limpieza de datos**
   - Identificación y manejo de valores faltantes
   - Corrección de tipos de datos
   - Eliminación o análisis de registros inconsistentes

5. **Análisis estadístico descriptivo**
   - Medidas básicas (media, mediana, desviación estándar)
   - Distribución de variables numéricas
   - Análisis de variables categóricas

6. **Análisis de relaciones**
   - Identificación de correlaciones
   - Exploración de posibles variables predictoras
   - Comparación entre niveles de estrés y otras variables



