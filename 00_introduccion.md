# K-Means Clustering — Tutorial

## ¿Qué es K-Means?

K-Means es un algoritmo de **aprendizaje no supervisado** que agrupa observaciones en *K* grupos (clusters) basándose en la similitud entre ellas. El objetivo es que los puntos dentro de un mismo cluster sean lo más parecidos posible entre sí, y lo más diferentes posible de los puntos de otros clusters.

La idea central es sencilla: cada cluster queda representado por su **centroide** (el punto promedio de todas las observaciones que pertenecen a ese grupo). El algoritmo ajusta iterativamente estos centroides hasta que las asignaciones dejan de cambiar.

---

## ¿Cuándo usar K-Means?

K-Means es una buena opción cuando:

- No tienes etiquetas en tus datos y quieres descubrir estructura oculta.
- Los clusters que esperas tienen forma **aproximadamente esférica** y tamaño similar.
- Tienes una idea o hipótesis sobre cuántos grupos podrían existir (aunque hay métodos para encontrar K óptimo).

---

## Estructura del tutorial

Este tutorial está dividido en dos grandes partes:

### Parte 1 — Dataset Sintético (concepto paso a paso)

Antes de trabajar con datos reales, construiremos un dataset artificial para entender **cómo funciona K-Means internamente**, sin ninguna caja negra.

| Paso | Archivo | Descripción |
|------|---------|-------------|
| 1 | [Generación de datos](01_generacion_datos.md) | Crear el dataset sintético con `make_blobs` |
| 2 | [Asignación aleatoria inicial](02_asignacion_aleatoria.md) | Punto de partida: clusters aleatorios y primeros centroides |
| 3 | [Proceso iterativo](03_proceso_iterativo.md) | El corazón del algoritmo: reasignar y recalcular hasta converger |

### Parte 2 — Dataset NCI60 (aplicación real)

Aplicaremos K-Means a datos reales de expresión genética de 64 líneas celulares de cáncer.

| Paso | Archivo | Descripción |
|------|---------|-------------|
| 4 | [Carga del dataset NCI60](04_nci60_carga.md) | Exploración inicial del dataset de alta dimensionalidad |
| 5 | [Reducción dimensional con PCA](05_pca.md) | Reducir 6830 variables a componentes principales para visualizar |
| 6 | [K-Means sobre NCI60](06_kmeans_nci60.md) | Aplicar K-Means y visualizar con PCA |
| 7 | [K óptimo: Elbow & Silhouette](07_k_optimo.md) | Encontrar el número ideal de clusters |
| 8 | [K=7 y Heatmap de genes](08_kmeans_k7_heatmap.md) | Resultado final y análisis de genes diferenciales |

---

## Librerías utilizadas

```python
# Manipulación de datos
import pandas as pd
import numpy as np

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Distancias
from scipy.spatial.distance import cdist

# Dataset NCI60
from ISLP import load_data
```

> **Nota:** Para instalar la librería ISLP (necesaria para el dataset NCI60):
> ```bash
> pip install ISLP
> ```

---

*Siguiente: [Generación del dataset sintético →](01_generacion_datos.md)*
