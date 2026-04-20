# Paso 4 — El Dataset NCI60: Carga y Exploración

## ¿Qué es NCI60?

NCI60 es un dataset clásico de bioinformática que contiene mediciones de **expresión genética** de 64 líneas celulares de cáncer, provenientes de 14 tipos distintos de tejido (leucemia, pulmón, colon, cerebro, etc.).

Cada línea celular tiene mediciones para **6,830 genes**, lo que lo convierte en un ejemplo perfecto de datos de **alta dimensionalidad**: no podemos simplemente graficar los datos para ver qué forma tienen.

Usaremos K-Means para descubrir si existe estructura de agrupamiento en estos datos genéticos.

---

## Instalación e importación

```python
# Instalar la librería ISLP (solo la primera vez)
!pip install ISLP
```

```python
import ISLP
from ISLP import load_data
import pandas as pd

# Cargar el dataset
df_nci60_raw = load_data('NCI60')

# El dataset viene como un diccionario con dos componentes:
# - 'data': la matriz de expresión genética
# - 'labels': las etiquetas del tipo de cáncer
X_nci60 = pd.DataFrame(df_nci60_raw['data'])
y_nci60 = df_nci60_raw['labels']['label']
```

---

## Exploración inicial

```python
print("--- Características del dataset NCI60 ---")
print(f"Forma de X (muestras × genes): {X_nci60.shape}")
print(f"Forma de y (etiquetas):         {y_nci60.shape}")

print("\n--- Conteo de líneas celulares por tipo de cáncer ---")
display(y_nci60.value_counts().sort_index())

print("\n--- Primeras filas de las características ---")
display(X_nci60.head())

print("\n--- Primeras etiquetas ---")
display(y_nci60.head())
```

---

## ¿Qué nos dice la exploración?

Al ejecutar el código anterior, obtenemos información clave:

**Dimensiones:** `X_nci60.shape` nos regresa `(64, 6830)`, es decir:
- **64 filas** → cada una es una línea celular de cáncer.
- **6,830 columnas** → cada una es un gen diferente.

**Tipos de cáncer presentes:**

| Tipo | Líneas celulares |
|------|-----------------|
| CNS | 5 |
| Colon | 7 |
| K562A-repro | 1 |
| Leukemia | 6 |
| Lung | 9 |
| MCF7A-repro | 1 |
| Melanoma | 8 |
| Ovarian | 6 |
| Prostate | 2 |
| Renal | 9 |
| Unknown | 1 |
| ... | ... |

> **Nota importante:** En K-Means estas etiquetas `y_nci60` **no se usan durante el entrenamiento**. Son solo para validar al final si los clusters descubiertos corresponden a los tipos de cáncer conocidos.

---

## El reto de la alta dimensionalidad

Con 6,830 variables, no podemos simplemente hacer un scatter plot como en la parte anterior. Necesitamos una estrategia para visualizar estos datos.

La solución es **PCA (Análisis de Componentes Principales)**: una técnica que comprime la información de miles de variables en unas pocas "super-variables" que capturan la mayor parte de la variabilidad de los datos.

Esto es exactamente lo que haremos en el siguiente paso.

---

*← [Proceso iterativo](03_proceso_iterativo.md) | [Reducción dimensional con PCA →](05_pca.md)*
