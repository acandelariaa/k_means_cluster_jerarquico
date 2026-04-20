# Paso 1 — Generación del Dataset Sintético

## ¿Por qué un dataset sintético?

Antes de aplicar K-Means a datos reales, es útil trabajar con un dataset **que nosotros mismos controlamos**. Así sabemos de antemano cuántos clusters existen y podemos verificar que el algoritmo los descubre correctamente.

Para esto usamos `make_blobs` de scikit-learn, que genera puntos distribuidos en "manchas" o grupos bien definidos.

---

## El código

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Parámetros del dataset
n_samples = 1000    # Total de observaciones
n_features = 2      # Número de variables (dimensiones)
n_centers = 3       # Número de clusters reales
cluster_std = 2.0   # Qué tan dispersos son los puntos dentro de cada cluster

X, y = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_centers,
    cluster_std=cluster_std
)

# Convertir a DataFrame para facilitar el manejo
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

print("Dataset generado:")
display(df.head())
```

---

## ¿Qué hace cada parámetro?

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| `n_samples` | 1000 | Generamos 1000 puntos en total |
| `n_features` | 2 | Cada punto tiene 2 coordenadas (x, y), lo que nos permite graficarlo directamente |
| `n_centers` | 3 | Hay 3 grupos "reales" en los datos |
| `cluster_std` | 2.0 | Dispersión de los puntos alrededor de cada centro; mayor valor = grupos más solapados |

> **¿Por qué 2 features?** Con 2 dimensiones podemos hacer un scatterplot directo sin necesidad de reducción dimensional. En la segunda parte del tutorial veremos qué hacer cuando los datos tienen cientos o miles de variables.

---

## Resultado esperado

`make_blobs` regresa dos cosas:

- **`X`**: un array de forma `(1000, 2)` con las coordenadas de cada punto.
- **`y`**: las etiquetas *reales* de cada punto (0, 1 o 2). En K-Means estas etiquetas **no se usan** durante el entrenamiento, pero nos sirven para validar al final.

El DataFrame resultante tiene esta forma:

| | Feature 1 | Feature 2 |
|--|-----------|-----------|
| 0 | -5.23 | 3.71 |
| 1 | 2.14 | -1.05 |
| ... | ... | ... |

---

## Punto importante

En este paso todavía **no hemos aplicado K-Means**. Simplemente tenemos los datos. En el siguiente paso veremos cuál es el estado inicial del algoritmo: una asignación completamente aleatoria de clusters.

---

*← [Introducción](00_introduccion.md) | [Asignación aleatoria inicial →](02_asignacion_aleatoria.md)*
