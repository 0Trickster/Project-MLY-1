# proyecto-ml-spotify

# Machine Learning: Predicción de Popularidad Musical

> **Asignatura:** Machine Learning (MLY1101)
> 
> **Sección:** MLY1101-002D
> 
> **Institución:** Duoc UC
> 
> **Integrantes:** Abel Aravena, Benjamín Aravena, Gabriel Castillo

---

## Descripción del proyecto

Este proyecto corresponde al desarrollo de un modelo de **Machine Learning aplicado a un dataset de canciones de Spotify**, cuyo objetivo es analizar las características musicales de distintas canciones y utilizarlas para determinar su nivel de popularidad.

El proyecto contempla distintas etapas del ciclo de desarrollo de un modelo de Machine Learning, comenzando con el análisis exploratorio de los datos y continuando con procesos de limpieza, tratamiento de valores atípicos, ingeniería de características y preprocesamiento.

Como variable objetivo se utiliza `popularity`, a partir de la cual se construye una variable categórica ordinal denominada `popularity_category`, que permite clasificar las canciones en cinco niveles de popularidad:

| Valor | Categoría |
| ----: | --------- |
|     0 | Muy baja  |
|     1 | Baja      |
|     2 | Media     |
|     3 | Alta      |
|     4 | Muy alta  |

---

## Flujo general del proyecto

El desarrollo del proyecto se estructura en las siguientes etapas:

```text
Dataset original
      │
      ▼
Análisis exploratorio
      │
      ▼
Limpieza de datos
      │
      ▼
Tratamiento de outliers
      │
      ▼
Winsorización
      │
      ▼
Ingeniería de características
      │
      ▼
Preprocesamiento
      │
      ▼
Train / Test Split
      │
      ▼
Entrenamiento del modelo
      │
      ▼
Evaluación
      │
      ▼
Modelo final
```

---

# Estrategia de ramificación

Para el desarrollo del proyecto se utiliza **Trunk-Based Development (TBD)**, debido a que permite trabajar con ramas de corta duración e integrar los cambios constantemente hacia la rama principal.

Esta estrategia resulta conveniente para un proyecto de Machine Learning debido a que:

* Permite integrar cambios pequeños y frecuentes.
* Facilita la detección temprana de errores.
* Reduce los conflictos durante la integración del código.
* Permite mantener la rama `main` en un estado estable.
* Facilita la implementación posterior de procesos de Integración Continua (CI).

Las ramas temporales se utilizan para desarrollar funcionalidades o modificaciones específicas y posteriormente son integradas a `main`.

---

## Commits convencionales

Para mantener una correcta trazabilidad del proyecto se utiliza el estándar **Conventional Commits**.

Los principales tipos utilizados son:

* `feature:` Nuevas funcionalidades o características.
* `fix:` Corrección de errores.
* `hotfix:` Correcciones urgentes.
* `docs:` Cambios relacionados exclusivamente con documentación.
* `ci:` Cambios relacionados con Integración Continua.
* `refactor:` Refactorización del código sin modificar su comportamiento.
* `chore:` Cambios de mantenimiento o configuración.

Ejemplos:

```text
feature: agregar variable de categoria de popularidad
feature: implementar preprocesamiento de variables
fix: corregir codificacion de variables ordinales
docs: actualizar README
ci: agregar pipeline de pruebas
refactor: reorganizar funciones de preprocesamiento
```

En los casos de uso, se preferirá el uso de descripciones **en inglés** para los commits.

---

## Ramas convencionales

Bajo la estrategia **Trunk-Based Development**, se utilizan ramas temporales que posteriormente son integradas a la rama `main`.

Desde la última actualización de este documento se adopta formalmente la convención:

```text
caracteristica/descripcion-corta
```

Por ejemplo:

```text
feature/preprocessing
feature/feature-engineering
feature/model-training
feature/model-evaluation
```

Las ramas utilizan únicamente caracteres en minúscula, además de los símbolos `/` y `-`.

Una vez finalizado e integrado el trabajo correspondiente, la rama temporal puede ser eliminada.

---

# Estructura del proyecto

La organización general del proyecto se encuentra distribuida de la siguiente manera:

```text
proyecto-ml-spotify
│
├── dataset/
│   ├── spotify_crudo.csv
│   ├── spotify_winsorizado.csv
│   ├── spotify_engineering.csv
│   ├── Spotify_train_preprocesado.csv
│   └── Spotify_test_preprocesado.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Limpieza.ipynb
│   ├── 03_Ingenieria_Caracteristicas.ipynb
│   └── 04_Preprocesamiento.ipynb
│
|──reports/
|  |──Informe_ejecutivo.html
|
├── utils/
|   └── c_datos.py
|   └── det_outliers.py
│   ├── eda_utils.py
│   └── vis_descriptiva.py
|   └── rel_variables.py
│
│
└── README.md
```

> La estructura puede variar de acuerdo con la organización definitiva del repositorio.

---

# Dataset

El proyecto utiliza un dataset compuesto por información relacionada con canciones de Spotify.

Entre las principales variables disponibles se encuentran:

### Información de las canciones

```text
artists
album_name
track_name
track_genre
```

### Características musicales

```text
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
```

### Duración

```text
duration_ms
duration_min
duration_category
```

### Variable objetivo

```text
popularity
popularity_category
```

---

# Etapas de procesamiento

## 1. Análisis exploratorio

En la primera etapa se realiza un análisis exploratorio del dataset con el objetivo de comprender:

* La estructura de los datos.
* Los tipos de variables.
* La distribución de las variables numéricas.
* Las variables categóricas.
* Los valores faltantes.
* Los valores duplicados.
* La presencia de valores atípicos.
* La distribución de la variable objetivo.

---

## 2. Limpieza de datos

Durante la etapa de limpieza se realizan diferentes operaciones sobre el dataset original.

Entre ellas:

* Eliminación de columnas que no aportan información relevante.
* Detección de valores faltantes.
* Eliminación de registros con valores faltantes.
* Detección y eliminación de registros duplicados.
* Análisis de valores atípicos.

Entre las columnas eliminadas durante esta etapa se encuentran:

```text
Unnamed: 0
track_id
time_signature
```

Las siguientes variables si bien no se eliminaron en el apartado de limpieza ni en la ingeniería de datos, se eliminaron
de manera implícita a través del `pipeline` de preprocesamiento con el comando `remainder_drop`:

```text
artists
album_name
track_name
explicit
```

---

## 3. Tratamiento de outliers

Para identificar valores atípicos se utiliza principalmente el método del **Rango Intercuartílico (IQR)**.

El método utiliza los siguientes elementos:

```text
IQR = Q3 - Q1

Límite inferior = Q1 - 1.5 × IQR

Límite superior = Q3 + 1.5 × IQR
```

Los valores que se encuentran fuera de estos límites son considerados potenciales valores atípicos.

Posteriormente se aplica **winsorización**, reemplazando los valores extremos por los límites correspondientes en lugar de eliminar los registros.

```text
Valor extremo inferior
          │
          ▼
    Límite inferior
          │
          ▼
   Valor reemplazado


Valor extremo superior
          │
          ▼
    Límite superior
          │
          ▼
   Valor reemplazado
```

El resultado de esta etapa se almacena en:

```text
spotify_winsorizado.csv
```

---

# 4. Ingeniería de características

En esta etapa se crean nuevas variables a partir de las variables existentes.

Una de las principales transformaciones consiste en convertir la duración de las canciones desde milisegundos a minutos:

```text
duration_min = duration_ms / 60000
```

Posteriormente se crea una variable categórica denominada:

```text
duration_category
```

Esta variable permite representar la duración mediante categorías ordinales:

```text
Corta
Media
Larga
Muy larga
```

El orden utilizado es:

```text
Corta < Media < Larga < Muy larga
```

También se transforma `popularity` en una variable categórica ordinal denominada:

```text
popularity_category
```

Las categorías utilizadas son:

```text
Muy baja
Baja
Media
Alta
Muy alta
```

El resultado de esta etapa se almacena en:

```text
spotify_engineering.csv
```

---

# 5. Preprocesamiento

El preprocesamiento permite transformar las variables para que puedan ser utilizadas posteriormente por los modelos de Machine Learning.

Las variables se dividen en tres grupos principales:

```text
                         Dataset
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
     Categóricas        Ordinales       Numéricas
   alta cardinalidad         │               │
            │                │               │
            ▼                ▼               ▼
     BinaryEncoder     OrdinalEncoder   StandardScaler
                             │
                             ▼
                      StandardScaler
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                   Dataset transformado
```

### Variables categóricas

Las variables categóricas de alta cardinalidad son transformadas mediante `BinaryEncoder`.

Actualmente se consideran:

```text
artists
album_name
track_name
track_genre
```

El uso de codificación binaria permite representar las categorías utilizando un número reducido de columnas en comparación con una codificación One-Hot.

---

### Variable ordinal

La variable:

```text
duration_category
```

se transforma utilizando `OrdinalEncoder`.

Se establece explícitamente el siguiente orden:

```text
Corta → Media → Larga → Muy larga
```

Posteriormente se aplica `StandardScaler`.

---

### Variables numéricas

Las variables numéricas reciben una transformación mediante `StandardScaler`.

Entre ellas se encuentran:

```text
danceability
energy
key
loudness
mode
speechiness
acousticness
instrumentalness
liveness
valence
tempo
duration_min
```

---

## ColumnTransformer

El `ColumnTransformer` permite aplicar diferentes transformaciones dependiendo del tipo de variable.

La estructura implementada permite procesar de manera independiente las variables categóricas, ordinales y numéricas, para posteriormente combinar los resultados en un único dataset transformado.

El resultado de esta etapa corresponde a los archivos:

```text
Spotify_train_preprocesado.csv
Spotify_test_preprocesado.csv
```

---

# Separación de entrenamiento y prueba

Antes de entrenar el modelo se divide el dataset en conjuntos de entrenamiento y prueba.

La distribución utilizada es:

```text
80% → Entrenamiento
20% → Prueba
```

La división utiliza:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

El parámetro `stratify=y` permite mantener una proporción similar de las cinco categorías de popularidad tanto en entrenamiento como en prueba.

El valor:

```text
random_state=42
```

permite reproducir la misma división de los datos al ejecutar nuevamente el proceso.

---

# Variable objetivo

La variable objetivo original es:

```text
popularity
```

Sin embargo, para el problema de clasificación se genera:

```text
popularity_category
```

El mapeo utilizado es:

```python
mapeo_popularidad = {
    'Muy baja': 0,
    'Baja': 1,
    'Media': 2,
    'Alta': 3,
    'Muy alta': 4
}
```

Por lo tanto:

| Código | Categoría |
| -----: | --------- |
|      0 | Muy baja  |
|      1 | Baja      |
|      2 | Media     |
|      3 | Alta      |
|      4 | Muy alta  |

La variable `popularity` se elimina de las características utilizadas para entrenar el modelo, debido a que `popularity_category` fue construida directamente a partir de ella.

Esto evita **target leakage**, ya que utilizar `popularity` como predictor permitiría al modelo acceder indirectamente a la información utilizada para construir la variable objetivo.

---

# Prevención de Data Leakage

Durante el preprocesamiento se mantiene la separación entre los datos de entrenamiento y prueba.

El procedimiento utilizado es:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_transformado = pipeline.fit_transform(X_train)

X_test_transformado = pipeline.transform(X_test)
```

El pipeline se ajusta únicamente utilizando el conjunto de entrenamiento mediante:

```text
fit_transform(X_train)
```

Mientras que el conjunto de prueba solamente se transforma mediante:

```text
transform(X_test)
```

De esta manera se evita que la información estadística del conjunto de prueba sea utilizada durante el ajuste del preprocesamiento.

---

# Tecnologías utilizadas

El proyecto utiliza principalmente las siguientes tecnologías:

* Python 3.14.2
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Category Encoders
* Jupyter Notebook
* Git
* GitHub

---

# Dependencias principales

Las principales bibliotecas utilizadas durante el desarrollo son:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
category_encoders
jupyter
```
---

# Ejecución

El proyecto se desarrolla principalmente mediante notebooks de Jupyter.

Para iniciar Jupyter:

```bash
jupyter notebook
```

Posteriormente se deben ejecutar los notebooks respetando el orden definido por el flujo de procesamiento:

```text
01 → Análisis exploratorio

02 → Limpieza y tratamiento de datos

03 → Ingeniería de características

04 → Preprocesamiento
```

Cada etapa genera los archivos necesarios para la siguiente.

---

# Archivos generados

Durante el procesamiento se generan los siguientes datasets:

### Dataset original

```text
spotify_crudo.csv
```

Contiene los datos originales utilizados como punto de partida.

### Dataset winsorizado

```text
spotify_winsorizado.csv
```

Contiene los datos después del tratamiento de valores atípicos mediante winsorización.

### Dataset con ingeniería de características

```text
spotify_engineering.csv
```

Contiene las nuevas variables creadas durante la etapa de Feature Engineering.

### Dataset de entrenamiento

```text
Spotify_train_preprocesado.csv
```

Contiene el 80% de los datos utilizados para el posterior entrenamiento del modelo.

### Dataset de prueba

```text
Spotify_test_preprocesado.csv
```

Contiene el 20% de los datos reservados para la evaluación del modelo.

---

# Estado actual del proyecto

Actualmente el proyecto contempla las siguientes etapas:

| Etapa                             | Estado     |
| --------------------------------- | ---------- |
| Análisis exploratorio             | Completado |
| Limpieza de datos                 | Completado |
| Tratamiento de outliers           | Completado |
| Winsorización                     | Completado |
| Ingeniería de características     | Completado |
| Creación de `popularity_category` | Completado |
| Preprocesamiento                  | Completado |
| División Train/Test               | Completado |
| Entrenamiento del modelo          | Pendiente  |
| Evaluación del modelo             | Pendiente  |
| Selección del modelo final        | Pendiente  |
| Optimización del modelo           | Pendiente  |

---

# Flujo completo del proyecto

```text
┌──────────────────────────┐
│   spotify_crudo.csv      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Análisis exploratorio    │
│          (EDA)           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Limpieza de datos        │
│ - NaN                    │
│ - Duplicados             │
│ - Columnas innecesarias  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Tratamiento de outliers  │
│       Winsorización      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ spotify_winsorizado.csv  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Ingeniería de            │
│ características          │
│ - duration_min           │
│ - duration_category      │
│ - popularity_category    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ spotify_engineering.csv  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Preprocesamiento         │
│                          │
│ BinaryEncoder            │
│ OrdinalEncoder           │
│ StandardScaler           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Train / Test Split       │
│                          │
│ 80% Train                │
│ 20% Test                 │
└────────────┬─────────────┘
             │
             ├──────────────────┐
             ▼                  ▼
┌──────────────────────┐  ┌──────────────────────┐
│ Train preprocesado   │  │ Test preprocesado    │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └────────────┬────────────┘
                        ▼
              ┌──────────────────┐
              │ Entrenamiento    │
              │ del modelo       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Evaluación       │
              │ del modelo       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Modelo final     │
              └──────────────────┘
```

---

# Autores

**Abel Aravena**
**Benjamín Aravena**
**Gabriel Castillo**

**Duoc UC — Machine Learning (MLY1101)**

---
