# 🔍 Unidad 1. Fundamentos del Aprendizaje No Supervisado

Esta unidad introduce los conceptos fundamentales del **Aprendizaje No Supervisado**, sus diferencias con el aprendizaje supervisado, las bibliotecas Python necesarias, el flujo de trabajo típico, y las metodologías esenciales para preparar datos y evaluar resultados en ausencia de etiquetas.


![Ilustración de unsup overview](../assets/images/unsup_overview.svg)
---

## 1.1. ¿Qué es el Aprendizaje No Supervisado?

El **Aprendizaje No Supervisado** es una rama del Machine Learning donde los algoritmos trabajan con datos **sin etiquetas**. A diferencia del aprendizaje supervisado, no existe una "respuesta correcta" predefinida que guíe el entrenamiento.

### Definiciones Clave

* **Definición formal:** "El aprendizaje no supervisado es el entrenamiento de un modelo usando información que no está clasificada ni etiquetada, permitiendo al algoritmo actuar sobre esa información sin guía."

* **Objetivo principal:** Descubrir **estructuras ocultas**, **patrones** o **agrupaciones** inherentes en los datos que no son evidentes a simple vista.

* **Analogía:** Imagina que te dan una caja con miles de fotografías sin ninguna descripción. El aprendizaje no supervisado sería como organizarlas automáticamente en grupos (paisajes, retratos, animales, etc.) basándose únicamente en las similitudes visuales entre ellas.

### Diferencias con el Aprendizaje Supervisado

| Aspecto | Supervisado | No Supervisado |
| :--- | :--- | :--- |
| **Datos** | Etiquetados (X, y) | Sin etiquetas (solo X) |
| **Objetivo** | Predecir una variable objetivo | Descubrir estructura en los datos |
| **Evaluación** | Métricas claras (accuracy, F1, MSE) | Métricas indirectas (silueta, inercia) |
| **Ejemplos** | Clasificación, Regresión | Clustering, Reducción de dimensionalidad |
| **Feedback** | Conocemos si la predicción es correcta | No hay "respuesta correcta" |

### Tipos de Problemas No Supervisados

El aprendizaje no supervisado abarca principalmente cuatro tipos de problemas:

1. **Clustering (Agrupamiento):**
   * **Objetivo:** Dividir los datos en grupos (clusters) donde los elementos dentro de un grupo son similares entre sí y diferentes a los de otros grupos.
   * **Algoritmos:** K-Means, DBSCAN, Clustering Jerárquico, OPTICS, Mean Shift.
   * **Aplicación:** Segmentación de clientes, agrupación de documentos.

2. **Reducción de Dimensionalidad:**
   * **Objetivo:** Reducir el número de variables (features) manteniendo la mayor cantidad de información posible.
   * **Algoritmos:** PCA, t-SNE, UMAP, LDA, Autoencoders.
   * **Aplicación:** Visualización de datos, compresión, preprocesamiento.

3. **Detección de Anomalías:**
   * **Objetivo:** Identificar puntos de datos que se desvían significativamente del comportamiento normal.
   * **Algoritmos:** Isolation Forest, One-Class SVM, LOF (Local Outlier Factor).
   * **Aplicación:** Detección de fraudes, mantenimiento predictivo.

4. **Reglas de Asociación:**
   * **Objetivo:** Descubrir relaciones interesantes entre variables en grandes conjuntos de datos.
   * **Algoritmos:** Apriori, FP-Growth, Eclat.
   * **Aplicación:** Análisis de cesta de compra, sistemas de recomendación.

---

## 1.2. Flujo de Trabajo del Aprendizaje No Supervisado

El proceso general para aplicar técnicas no supervisadas sigue estos pasos:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Definición  │───▶│  2. Preparación │───▶│  3. Selección   │
│  del Problema   │    │    de Datos     │    │  del Algoritmo  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 6. Aplicación   │◀───│ 5. Validación   │◀───│ 4. Entrenamiento│
│ e Interpretación│    │   y Evaluación  │    │    del Modelo   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1. Definición del Problema
* ¿Qué queremos descubrir? ¿Grupos de clientes? ¿Patrones anómalos?
* ¿Cuántas dimensiones tienen los datos? ¿Son visualizables?
* ¿Hay conocimiento del dominio que pueda guiar la interpretación?

### 2. Preparación de Datos
* **Limpieza:** Manejo de valores faltantes y outliers.
* **Escalado:** Crucial para algoritmos basados en distancias (K-Means, DBSCAN).
* **Selección de características:** Eliminar features irrelevantes o redundantes.

### 3. Selección del Algoritmo
* Depende del tipo de problema y las características de los datos.
* Considerar: tamaño del dataset, número de clusters esperado, forma de los clusters.

### 4. Entrenamiento del Modelo
* No hay etiquetas, por lo que no hay conjunto de "validación" tradicional.
* Se ajustan hiperparámetros mediante técnicas específicas (método del codo, silueta).

### 5. Validación y Evaluación
* Métricas internas: Silueta, Inercia, Davies-Bouldin.
* Validación visual: Gráficos de clusters, dendrogramas.
* Validación externa (si hay etiquetas disponibles): NMI, ARI.

### 6. Aplicación e Interpretación
* Asignar significado a los clusters descubiertos.
* Integrar resultados en procesos de negocio o análisis posteriores.

---

## 1.3. Bibliotecas Python para Aprendizaje No Supervisado

### Instalación de Bibliotecas Esenciales

```bash
# Instalación con pip
pip install numpy pandas matplotlib seaborn scikit-learn

# Bibliotecas adicionales específicas
pip install mlxtend          # Para reglas de asociación (Apriori)
pip install umap-learn       # Para UMAP (reducción de dimensionalidad)
pip install hdbscan          # Para HDBSCAN (clustering avanzado)
pip install yellowbrick      # Para visualización de ML
```

### Bibliotecas Principales y sus Módulos

| Biblioteca | Módulo | Funcionalidad | Algoritmos/Funciones |
| :--- | :--- | :--- | :--- |
| `scikit-learn` | `sklearn.cluster` | Clustering | `KMeans`, `DBSCAN`, `AgglomerativeClustering` |
| `scikit-learn` | `sklearn.decomposition` | Reducción de dimensionalidad | `PCA`, `TruncatedSVD`, `NMF` |
| `scikit-learn` | `sklearn.manifold` | Embedding no lineal | `TSNE`, `MDS`, `Isomap` |
| `scikit-learn` | `sklearn.ensemble` | Detección de anomalías | `IsolationForest` |
| `scikit-learn` | `sklearn.neighbors` | Detección de outliers | `LocalOutlierFactor` |
| `scikit-learn` | `sklearn.metrics` | Métricas de evaluación | `silhouette_score`, `calinski_harabasz_score` |
| `mlxtend` | `mlxtend.frequent_patterns` | Reglas de asociación | `apriori`, `association_rules` |
| `scipy` | `scipy.cluster.hierarchy` | Clustering jerárquico | `linkage`, `dendrogram`, `fcluster` |

### Imports Típicos para Aprendizaje No Supervisado

```python
# Bibliotecas básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Algoritmos de Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Reducción de Dimensionalidad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Detección de Anomalías
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Reglas de Asociación
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Métricas de Evaluación
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Clustering Jerárquico (scipy)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
```

---

## 1.4. Preprocesamiento de Datos para Aprendizaje No Supervisado

El preprocesamiento es **aún más crítico** en el aprendizaje no supervisado que en el supervisado, ya que los algoritmos son muy sensibles a la escala y calidad de los datos.

### 1.4.1. Escalado de Características

**¿Por qué es obligatorio?**

La mayoría de algoritmos no supervisados se basan en **medidas de distancia** (Euclidiana, Manhattan, etc.). Si las variables tienen escalas muy diferentes, las de mayor magnitud dominarán completamente el cálculo.

**Ejemplo del problema:**
```
Cliente A: Edad=25, Salario=50000
Cliente B: Edad=35, Salario=51000
Cliente C: Edad=26, Salario=80000
```
Sin escalado, la diferencia de salario (miles) dominará sobre la edad (decenas).

**Métodos de Escalado:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1. Estandarización (Z-score) - El más común
# Transforma datos para tener media=0 y desviación estándar=1
# Fórmula: z = (x - μ) / σ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Normalización Min-Max
# Escala valores al rango [0, 1]
# Fórmula: x_norm = (x - x_min) / (x_max - x_min)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 3. RobustScaler - Resistente a outliers
# Usa mediana y rango intercuartílico en lugar de media y std
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
```

**¿Cuándo usar cada uno?**

| Método | Cuándo usar |
| :--- | :--- |
| `StandardScaler` | Datos aproximadamente normales, sin muchos outliers |
| `MinMaxScaler` | Cuando necesitas valores acotados [0,1], ej. para redes neuronales |
| `RobustScaler` | Cuando hay outliers significativos en los datos |

### 1.4.2. Manejo de Valores Faltantes

En aprendizaje no supervisado, los valores faltantes son problemáticos porque:
- Muchos algoritmos no los aceptan directamente
- Pueden distorsionar las medidas de distancia

```python
from sklearn.impute import SimpleImputer, KNNImputer

# 1. Imputación simple (media, mediana, moda)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 2. Imputación basada en KNN (más sofisticada)
# Imputa usando los k vecinos más cercanos
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### 1.4.3. Manejo de Datos Categóricos

Los algoritmos de clustering generalmente requieren datos numéricos:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Para variables nominales: One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['categoria'])

# Alternativa: OneHotEncoder de sklearn
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(df[['categoria']])
```

**Nota:** Para clustering con variables categóricas puras, considerar algoritmos especializados como **K-Modes** o **K-Prototypes** (biblioteca `kmodes`).

---

## 1.5. Métricas de Evaluación en Aprendizaje No Supervisado

Evaluar modelos no supervisados es más complejo porque **no hay etiquetas de referencia**. Existen dos tipos de métricas:

### 1.5.1. Métricas Internas (sin etiquetas reales)

Evalúan la calidad del clustering basándose únicamente en los datos y las asignaciones de cluster.

#### Coeficiente de Silueta (Silhouette Score)

Mide qué tan similar es un punto a su propio cluster comparado con otros clusters.

**Fórmula para un punto $i$:**
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Donde:
- $a(i)$ = distancia media de $i$ a los otros puntos de su mismo cluster (cohesión)
- $b(i)$ = distancia media mínima de $i$ a los puntos del cluster más cercano (separación)

**Interpretación:**
- $s(i) \approx 1$: El punto está bien asignado a su cluster
- $s(i) \approx 0$: El punto está en la frontera entre clusters
- $s(i) < 0$: El punto probablemente está mal asignado

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Silueta promedio del clustering
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")

# Silueta por cada muestra (para análisis detallado)
sample_scores = silhouette_samples(X, labels)
```

#### Inercia (Within-Cluster Sum of Squares - WCSS)

Suma de las distancias al cuadrado de cada punto al centroide de su cluster. **Solo para K-Means.**

$$WCSS = \sum_{i=1}^{k}\sum_{x \in C_i} ||x - \mu_i||^2$$

**Interpretación:**
- Menor inercia = clusters más compactos
- Se usa en el **Método del Codo** para encontrar el número óptimo de clusters

```python
# La inercia se obtiene directamente del modelo KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(f"Inercia: {kmeans.inertia_}")
```

#### Índice Calinski-Harabasz (Variance Ratio Criterion)

Ratio entre la dispersión entre clusters y la dispersión dentro de clusters.

$$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}$$

Donde:
- $SS_B$ = dispersión entre clusters
- $SS_W$ = dispersión dentro de clusters
- $k$ = número de clusters
- $n$ = número de muestras

**Interpretación:** Mayor valor = mejor clustering

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {score:.3f}")
```

#### Índice Davies-Bouldin

Mide la similitud promedio entre cada cluster y su cluster más similar.

$$DB = \frac{1}{k}\sum_{i=1}^{k}\max_{j \neq i}\left(\frac{s_i + s_j}{d_{ij}}\right)$$

Donde:
- $s_i$ = dispersión media del cluster $i$
- $d_{ij}$ = distancia entre centroides de clusters $i$ y $j$

**Interpretación:** Menor valor = mejor clustering (clusters más separados y compactos)

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Score: {score:.3f}")
```

### 1.5.2. Métricas Externas (con etiquetas reales)

Cuando disponemos de etiquetas reales (ground truth), podemos comparar los clusters descubiertos con las clases verdaderas.

#### Adjusted Rand Index (ARI)

Mide la similitud entre dos asignaciones de clusters, ajustada por azar.

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels_pred)
# Rango: [-1, 1], donde 1 = asignación perfecta
```

#### Normalized Mutual Information (NMI)

Mide la información mutua entre las asignaciones, normalizada.

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, labels_pred)
# Rango: [0, 1], donde 1 = asignación perfecta
```

### 1.5.3. Tabla Resumen de Métricas

| Métrica | Tipo | Rango | Mejor valor | Uso principal |
| :--- | :--- | :--- | :--- | :--- |
| Silhouette | Interna | [-1, 1] | Cercano a 1 | Evaluar calidad general |
| Inercia (WCSS) | Interna | [0, ∞) | Menor | Método del codo |
| Calinski-Harabasz | Interna | [0, ∞) | Mayor | Comparar configuraciones |
| Davies-Bouldin | Interna | [0, ∞) | Menor | Comparar configuraciones |
| ARI | Externa | [-1, 1] | Cercano a 1 | Validar con ground truth |
| NMI | Externa | [0, 1] | Cercano a 1 | Validar con ground truth |

---

## 1.6. El Método del Codo (Elbow Method)

Es la técnica más popular para determinar el número óptimo de clusters en K-Means.

### Concepto

1. Ejecutar K-Means con diferentes valores de $k$ (número de clusters)
2. Para cada $k$, calcular la inercia (WCSS)
3. Graficar $k$ vs. inercia
4. Buscar el "codo": el punto donde la reducción de inercia se desacelera significativamente

### Implementación Completa

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# Calcular inercia para diferentes valores de k
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)', fontsize=12)
plt.ylabel('Inercia (WCSS)', fontsize=12)
plt.title('Método del Codo para Determinar k Óptimo', fontsize=14)
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.show()
```

### Interpretación Visual

```
Inercia
   │
   │\
   │ \
   │  \
   │   \____ ← "Codo" (k óptimo)
   │        \____
   │             \____
   └─────────────────────── k
```

---

## 1.7. Visualización de Resultados

La visualización es fundamental en aprendizaje no supervisado para interpretar y comunicar resultados.

### 1.7.1. Visualización de Clusters en 2D

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Crear datos
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# Aplicar K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Visualizar
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                      alpha=0.6, edgecolors='w', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
            s=200, edgecolors='black', linewidths=2, label='Centroides')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Resultados del Clustering K-Means')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.show()
```

### 1.7.2. Gráfico de Silueta

```python
from sklearn.metrics import silhouette_samples
import numpy as np

def plot_silhouette(X, labels, n_clusters):
    """Grafica el análisis de silueta por cluster."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Valores de silueta para el cluster i
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        
        color = plt.cm.viridis(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Silueta media: {silhouette_avg:.3f}')
    ax.set_xlabel("Coeficiente de Silueta")
    ax.set_ylabel("Cluster")
    ax.legend()
    plt.title("Análisis de Silueta")
    plt.show()
```

---

## 1.8. Ejemplos Prácticos y Recursos Externos

### Recursos y Tutoriales Recomendados

* **Documentación oficial de scikit-learn - Clustering:**
  [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)

* **Documentación oficial de scikit-learn - Reducción de Dimensionalidad:**
  [https://scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html)

* **Tutorial de K-Means con datos reales (Customer Segmentation):**
  [https://www.kaggle.com/code/kushal1996/customer-segmentation-k-means-analysis](https://www.kaggle.com/code/kushal1996/customer-segmentation-k-means-analysis)

* **Ejemplo completo de PCA para visualización:**
  [https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)

* **Market Basket Analysis con Apriori:**
  [https://www.kaggle.com/code/datatheque/association-rules-mining-market-basket-analysis](https://www.kaggle.com/code/datatheque/association-rules-mining-market-basket-analysis)

* **Detección de anomalías con Isolation Forest:**
  [https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html](https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html)

---

## 1.9. Comparativa de Algoritmos de Clustering

| Algoritmo | Forma clusters | Escalabilidad | Requiere k | Maneja ruido | Complejidad |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means** | Esféricos | Muy alta | Sí | No | O(n·k·i) |
| **DBSCAN** | Arbitraria | Media | No | Sí | O(n²) o O(n log n) |
| **Jerárquico** | Arbitraria | Baja | No* | No | O(n³) |
| **Gaussian Mixture** | Elípticos | Alta | Sí | No | O(n·k·i) |
| **OPTICS** | Arbitraria | Media | No | Sí | O(n²) |

*El clustering jerárquico no requiere k a priori, pero sí para "cortar" el dendrograma.

---

## 1.10. Buenas Prácticas

### ✅ Hacer siempre:

1. **Escalar los datos** antes de aplicar algoritmos basados en distancias
2. **Explorar los datos** visualmente antes del clustering (EDA)
3. **Probar múltiples algoritmos** y comparar resultados
4. **Usar múltiples métricas** para evaluar la calidad
5. **Validar los resultados** con conocimiento del dominio
6. **Documentar las decisiones** (por qué se eligió cierto k, algoritmo, etc.)

### ❌ Evitar:

1. Asumir que existe una estructura de clusters cuando puede no haberla
2. Confiar ciegamente en una sola métrica
3. Ignorar outliers sin investigarlos
4. Aplicar algoritmos sin entender sus supuestos
5. Sobrevalorar el número de clusters (más no siempre es mejor)

---

## 1.11. Ejercicio Integrador: Pipeline Completo

```python
"""
Pipeline completo de clustering con evaluación
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import load_iris

# 1. CARGAR DATOS
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

print("="*50)
print("PIPELINE DE CLUSTERING NO SUPERVISADO")
print("="*50)

# 2. PREPROCESAMIENTO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n[1] Datos escalados: {X_scaled.shape}")

# 3. MÉTODO DEL CODO
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inercia')
axes[0].set_title('Método del Codo')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 'go-', linewidth=2)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silueta vs k')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. MODELO FINAL (k=3 basado en análisis)
k_optimo = 3
kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
labels = kmeans_final.fit_predict(X_scaled)

print(f"\n[2] Clustering con k={k_optimo}")
print(f"    Distribución de clusters: {np.bincount(labels)}")

# 5. EVALUACIÓN
print(f"\n[3] MÉTRICAS DE EVALUACIÓN:")
print(f"    - Silhouette Score:      {silhouette_score(X_scaled, labels):.4f}")
print(f"    - Calinski-Harabasz:     {calinski_harabasz_score(X_scaled, labels):.4f}")
print(f"    - Davies-Bouldin:        {davies_bouldin_score(X_scaled, labels):.4f}")

# 6. ANÁLISIS DE RESULTADOS
print(f"\n[4] ANÁLISIS POR CLUSTER:")
df = pd.DataFrame(X, columns=feature_names)
df['cluster'] = labels
print(df.groupby('cluster').mean().round(2))
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
