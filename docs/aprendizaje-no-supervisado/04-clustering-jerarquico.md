# 🌲 Unidad 4. Clustering Jerárquico

El **Clustering Jerárquico** es una familia de algoritmos que construyen una **jerarquía de clusters** en lugar de una partición plana. Su característica distintiva es que produce un **dendrograma**: una estructura de árbol que muestra cómo se forman o dividen los clusters a diferentes niveles de similitud. Este enfoque permite explorar la estructura de los datos a múltiples escalas sin necesidad de especificar el número de clusters a priori.

![Ilustración de hierarchical](../assets/images/hierarchical.svg)
---

## 4.1. ¿Cómo Funciona el Clustering Jerárquico?

### Dos Enfoques Principales

Existen dos estrategias opuestas para construir la jerarquía:

```
┌─────────────────────────────────────────────────────────────┐
│ TIPOS DE CLUSTERING JERÁRQUICO                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. AGLOMERATIVO (Bottom-Up) - El más común                  │
│    ─────────────────────────────────────                    │
│    • Empieza: cada punto es su propio cluster               │
│    • Proceso: fusiona los dos clusters más cercanos         │
│    • Termina: todos los puntos en un único cluster          │
│                                                             │
│         ○ ○ ○ ○ ○   →   ○○ ○ ○○   →   ○○○ ○○   →   ○○○○○    │
│         5 clusters     4 clusters    2 clusters   1 cluster │
│                                                             │
│ 2. DIVISIVO (Top-Down) - Menos común                        │
│    ──────────────────────────────                           │
│    • Empieza: todos los puntos en un único cluster          │
│    • Proceso: divide el cluster menos coherente             │
│    • Termina: cada punto es su propio cluster               │
│                                                             │
│         ○○○○○   →   ○○○ ○○   →   ○○ ○ ○○   →   ○ ○ ○ ○ ○    │
│         1 cluster   2 clusters   4 clusters   5 clusters    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Algoritmo Aglomerativo Paso a Paso

```
┌─────────────────────────────────────────────────────────────┐
│ ALGORITMO AGLOMERATIVO (AGNES)                              │
├─────────────────────────────────────────────────────────────┤
│ Entrada: Datos X con n puntos, criterio de enlace           │
│ Salida: Dendrograma (árbol de fusiones)                     │
│                                                             │
│ 1. INICIALIZACIÓN:                                          │
│    - Crear n clusters (uno por cada punto)                  │
│    - Calcular matriz de distancias entre todos los pares    │
│                                                             │
│ 2. REPETIR n-1 veces:                                       │
│    a) Encontrar los dos clusters más cercanos               │
│    b) Fusionarlos en un nuevo cluster                       │
│    c) Actualizar la matriz de distancias                    │
│    d) Registrar la fusión y su altura en el dendrograma     │
│                                                             │
│ 3. RESULTADO:                                               │
│    - Dendrograma completo                                   │
│    - Para obtener k clusters: "cortar" el dendrograma       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### El Dendrograma

El dendrograma es la visualización clave del clustering jerárquico:

```
Altura
(distancia)
   │
   │         ┌─────────┐
   │     ┌───┤         │
   │     │   └────┬────┘
   │  ┌──┤        │
   │  │  └───┬────┘
   │──┤      │
   │  └──────┤
   │         │
   └─────────┴──────────────
            A  B  C  D  E  (puntos)
```

- **Eje vertical:** altura de fusión (distancia entre clusters fusionados)
- **Eje horizontal:** puntos o clusters individuales
- **Líneas horizontales:** fusiones entre clusters
- **Cortar horizontalmente:** obtener un número específico de clusters

---

## 4.2. Explicación Matemática

### Matriz de Distancias

El clustering jerárquico comienza calculando una **matriz de distancias** $D$ de tamaño $n \times n$:

$$D_{ij} = d(x_i, x_j)$$

Donde $d$ es una función de distancia (típicamente Euclidiana).

### Criterios de Enlace (Linkage)

La clave del clustering jerárquico es cómo se calcula la distancia entre clusters. Existen varios **criterios de enlace**:

#### 1. Enlace Simple (Single Linkage) - "Vecino más cercano"

Distancia entre los dos puntos más cercanos de cada cluster:

$$d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$$

- ✅ Puede detectar clusters de formas arbitrarias
- ❌ Sensible al "efecto cadena" (clusters elongados)

#### 2. Enlace Completo (Complete Linkage) - "Vecino más lejano"

Distancia entre los dos puntos más lejanos de cada cluster:

$$d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$$

- ✅ Produce clusters compactos de tamaño similar
- ❌ Sensible a outliers

#### 3. Enlace Promedio (Average Linkage - UPGMA)

Promedio de todas las distancias entre pares de puntos:

$$d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$$

- ✅ Balance entre single y complete
- ✅ Menos sensible a outliers

#### 4. Enlace de Ward (Ward's Method)

Minimiza el incremento en la varianza total al fusionar clusters:

$$d(C_i, C_j) = \sqrt{\frac{2|C_i||C_j|}{|C_i|+|C_j|}} ||\mu_i - \mu_j||$$

Donde $\mu_i$ y $\mu_j$ son los centroides de los clusters.

Equivalentemente, Ward minimiza:

$$\Delta = \sum_{x \in C_i \cup C_j} ||x - \mu_{ij}||^2 - \sum_{x \in C_i} ||x - \mu_i||^2 - \sum_{x \in C_j} ||x - \mu_j||^2$$

- ✅ Tiende a producir clusters esféricos y de tamaño similar
- ✅ Similar a K-Means pero jerárquico
- ❌ Asume clusters esféricos

### Visualización de Criterios de Enlace

```
          Cluster A              Cluster B
         
           ●  ●                    ▲  ▲
         ●      ●                ▲      ▲
           ●  ●                    ▲  ▲

Single:   d = distancia mínima (más corta)
Complete: d = distancia máxima (más larga)
Average:  d = promedio de todas las distancias
Ward:     d = incremento mínimo en varianza
```

---

## 4.3. Pros y Contras

| Ventajas | Desventajas |
| :--- | :--- |
| **No requiere k:** El número de clusters se elige después | **Complejidad alta:** $O(n^3)$ tiempo, $O(n^2)$ espacio |
| **Dendrograma informativo:** Permite explorar estructura a múltiples niveles | **No escalable:** Impracticable para datasets grandes (>10K puntos) |
| **Flexibilidad:** Diferentes criterios de enlace para diferentes necesidades | **Irreversible:** Una fusión mala no puede deshacerse |
| **Determinístico:** Mismo resultado cada ejecución | **Sensible a outliers:** Especialmente con complete linkage |
| **Interpretable:** El dendrograma es fácil de entender | **Sensible a la elección de linkage:** Resultados muy diferentes |

---

## 4.4. Ejemplo Básico en Python

Este ejemplo muestra el uso básico del clustering jerárquico con visualización del dendrograma.

```python
# ============================================================
# EJEMPLO BÁSICO: Clustering Jerárquico con Dendrograma
# ============================================================

# Importar bibliotecas necesarias
import numpy as np                          # Operaciones numéricas
import matplotlib.pyplot as plt             # Visualización
from scipy.cluster.hierarchy import (       # Funciones de scipy
    linkage,        # Calcular el enlace jerárquico
    dendrogram,     # Crear el dendrograma
    fcluster        # Obtener clusters del dendrograma
)
from sklearn.datasets import make_blobs     # Datos sintéticos
from sklearn.preprocessing import StandardScaler  # Escalado

# -------------------------------------------------------------
# 1. GENERAR DATOS DE EJEMPLO
# -------------------------------------------------------------
# Crear 4 clusters bien definidos
X, y_true = make_blobs(
    n_samples=50,       # 50 puntos (pequeño para visualización clara)
    centers=4,          # 4 clusters
    cluster_std=0.60,   # Dispersión moderada
    random_state=42
)

print(f"Forma de los datos: {X.shape}")

# -------------------------------------------------------------
# 2. CALCULAR LA MATRIZ DE ENLACE
# -------------------------------------------------------------
# linkage() calcula el clustering jerárquico
# Devuelve una matriz Z de (n-1) x 4:
# - Columnas 0 y 1: índices de clusters fusionados
# - Columna 2: distancia entre ellos (altura del enlace)
# - Columna 3: número de puntos en el nuevo cluster

Z = linkage(
    X,                  # Datos
    method='ward',      # Criterio de enlace
    metric='euclidean'  # Métrica de distancia
)

print(f"\nMatriz de enlace Z (primeras 5 filas):")
print(f"[cluster_1, cluster_2, distancia, n_puntos]")
print(Z[:5])

# -------------------------------------------------------------
# 3. VISUALIZAR EL DENDROGRAMA
# -------------------------------------------------------------
plt.figure(figsize=(14, 8))

# dendrogram() crea la visualización del árbol jerárquico
# truncate_mode='level' limita el número de niveles mostrados
dn = dendrogram(
    Z,
    truncate_mode='lastp',  # Mostrar los últimos p clusters fusionados
    p=20,                    # Número de clusters a mostrar
    leaf_rotation=90,        # Rotar etiquetas de hojas
    leaf_font_size=10,       # Tamaño de fuente
    show_contracted=True     # Mostrar clusters contraídos
)

plt.xlabel('Punto o Cluster', fontsize=12)
plt.ylabel('Distancia (Altura)', fontsize=12)
plt.title('Dendrograma - Clustering Jerárquico (Ward Linkage)', fontsize=14)

# Añadir línea horizontal para "cortar" el dendrograma
# Esta línea indica dónde cortaríamos para obtener cierto número de clusters
corte = 7  # Altura de corte
plt.axhline(y=corte, color='r', linestyle='--', 
            label=f'Corte en altura={corte}')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 4. OBTENER ASIGNACIONES DE CLUSTER
# -------------------------------------------------------------
# fcluster() "corta" el dendrograma para obtener clusters
# t: umbral de corte
# criterion: cómo interpretar t

# Opción 1: Cortar por distancia (altura)
labels_dist = fcluster(Z, t=corte, criterion='distance')
print(f"\nClusters (corte por distancia={corte}): {np.unique(labels_dist)}")
print(f"Distribución: {np.bincount(labels_dist)[1:]}")  # [1:] porque fcluster empieza en 1

# Opción 2: Especificar número de clusters directamente
labels_k = fcluster(Z, t=4, criterion='maxclust')
print(f"\nClusters (k=4): {np.unique(labels_k)}")
print(f"Distribución: {np.bincount(labels_k)[1:]}")

# -------------------------------------------------------------
# 5. VISUALIZAR CLUSTERS RESULTANTES
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Clusters por corte de distancia
scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels_dist,
                           cmap='viridis', edgecolors='w', s=50)
axes[0].set_title(f'Clusters (corte altura={corte})')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Clusters especificando k=4
scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=labels_k,
                           cmap='viridis', edgecolors='w', s=50)
axes[1].set_title('Clusters (k=4 especificado)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.show()

print("\n✅ El dendrograma permite explorar la estructura a diferentes niveles")
print("✅ Cortando a diferentes alturas obtenemos diferentes números de clusters")
```

---

## 4.5. Ejemplo Avanzado: Comparación de Criterios de Enlace

Este ejemplo compara diferentes criterios de enlace y muestra cómo elegir el número óptimo de clusters.

```python
# ============================================================
# EJEMPLO AVANZADO: Análisis Completo de Clustering Jerárquico
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.datasets import load_iris

# -------------------------------------------------------------
# 1. CARGAR Y PREPARAR DATOS
# -------------------------------------------------------------
iris = load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names

print("="*60)
print("ANÁLISIS DE CLUSTERING JERÁRQUICO - DATASET IRIS")
print("="*60)
print(f"\nDimensiones: {X.shape}")
print(f"Features: {feature_names}")

# Estandarizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------
# 2. COMPARAR DIFERENTES CRITERIOS DE ENLACE
# -------------------------------------------------------------
print("\n" + "="*60)
print("COMPARACIÓN DE CRITERIOS DE ENLACE")
print("="*60)

# Métodos de enlace a comparar
methods = ['single', 'complete', 'average', 'ward']
method_names = {
    'single': 'Single (Vecino más cercano)',
    'complete': 'Complete (Vecino más lejano)',
    'average': 'Average (Promedio)',
    'ward': 'Ward (Minimiza varianza)'
}

# Crear figura para dendrogramas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Almacenar resultados
linkage_results = {}

for idx, method in enumerate(methods):
    # Calcular enlace
    Z = linkage(X_scaled, method=method)
    linkage_results[method] = Z
    
    # Calcular correlación cophenetica
    # Mide qué tan bien el dendrograma preserva las distancias originales
    c, _ = cophenet(Z, pdist(X_scaled))
    
    # Dibujar dendrograma
    ax = axes[idx]
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=8,
               show_contracted=True)
    ax.set_title(f'{method_names[method]}\nCorrelación Cophenetica: {c:.3f}')
    ax.set_xlabel('Muestra')
    ax.set_ylabel('Distancia')

plt.tight_layout()
plt.show()

# Mostrar correlaciones copheneticas
print("\nCorrelación Cophenetica por método:")
print("(Mayor = mejor preservación de distancias originales)")
for method in methods:
    c, _ = cophenet(linkage_results[method], pdist(X_scaled))
    print(f"  {method_names[method]}: {c:.4f}")

# -------------------------------------------------------------
# 3. ANÁLISIS DEL DENDROGRAMA (Ward)
# -------------------------------------------------------------
print("\n" + "="*60)
print("ANÁLISIS DETALLADO - MÉTODO WARD")
print("="*60)

Z_ward = linkage_results['ward']

# Analizar las últimas fusiones (más informativas)
print("\nÚltimas 10 fusiones:")
print("Fusión | Cluster1 | Cluster2 | Distancia | Tamaño")
print("-"*55)
for i in range(-10, 0):
    row = Z_ward[i]
    print(f"{len(Z_ward)+i+1:6} | {int(row[0]):8} | {int(row[1]):8} | {row[2]:9.3f} | {int(row[3]):6}")

# -------------------------------------------------------------
# 4. DETERMINAR NÚMERO ÓPTIMO DE CLUSTERS
# -------------------------------------------------------------
print("\n" + "="*60)
print("DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLUSTERS")
print("="*60)

# Método 1: Análisis de las distancias de fusión
# Buscar "saltos" grandes en las distancias
heights = Z_ward[:, 2]
height_diffs = np.diff(heights)

# Las últimas fusiones (las más significativas)
print("\nSaltos de distancia en últimas fusiones:")
for i in range(-5, 0):
    n_clusters = len(Z_ward) - len(Z_ward) - i
    print(f"  {n_clusters} → {n_clusters-1} clusters: salto = {height_diffs[i]:.3f}")

# Método 2: Usar métricas de evaluación
k_range = range(2, 11)
metrics = {'k': [], 'silhouette': [], 'calinski': []}

for k in k_range:
    labels = fcluster(Z_ward, t=k, criterion='maxclust')
    
    metrics['k'].append(k)
    metrics['silhouette'].append(silhouette_score(X_scaled, labels))
    metrics['calinski'].append(calinski_harabasz_score(X_scaled, labels))

df_metrics = pd.DataFrame(metrics)
print("\nMétricas por número de clusters:")
print(df_metrics.round(4).to_string(index=False))

# Visualizar métricas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Dendrograma con líneas de corte
ax1 = axes[0]
dendrogram(Z_ward, ax=ax1, truncate_mode='lastp', p=20)
ax1.axhline(y=8, color='r', linestyle='--', label='k=3')
ax1.axhline(y=5, color='g', linestyle='--', label='k=5')
ax1.set_title('Dendrograma Ward')
ax1.legend()

# Silhouette
ax2 = axes[1]
ax2.plot(df_metrics['k'], df_metrics['silhouette'], 'bo-')
ax2.set_xlabel('Número de clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silueta vs k')
ax2.grid(True, alpha=0.3)

# Calinski-Harabasz
ax3 = axes[2]
ax3.plot(df_metrics['k'], df_metrics['calinski'], 'go-')
ax3.set_xlabel('Número de clusters')
ax3.set_ylabel('Calinski-Harabasz')
ax3.set_title('Calinski-Harabasz vs k')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 5. MODELO FINAL CON SKLEARN
# -------------------------------------------------------------
print("\n" + "="*60)
print("MODELO FINAL CON AgglomerativeClustering")
print("="*60)

k_optimo = 3  # Basado en análisis

# Usando sklearn.cluster.AgglomerativeClustering
# Más flexible y permite especificar k directamente
model = AgglomerativeClustering(
    n_clusters=k_optimo,        # Número de clusters
    metric='euclidean',         # Métrica de distancia
    linkage='ward',             # Criterio de enlace
    # Parámetros adicionales:
    # compute_full_tree: bool, calcular árbol completo aunque n_clusters esté especificado
    # distance_threshold: None o float, distancia para el corte (si se usa, n_clusters debe ser None)
)

# Entrenar y obtener etiquetas
labels = model.fit_predict(X_scaled)

print(f"\nResultados:")
print(f"  Número de clusters: {model.n_clusters_}")
print(f"  Número de hojas: {model.n_leaves_}")
print(f"  Número de componentes conectados: {model.n_connected_components_}")

# Distribución
print(f"\nDistribución de puntos:")
for i in range(k_optimo):
    count = np.sum(labels == i)
    print(f"  Cluster {i}: {count} puntos ({count/len(labels)*100:.1f}%)")

# Métricas
print(f"\nMétricas de evaluación:")
print(f"  Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
print(f"  Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels):.4f}")

# -------------------------------------------------------------
# 6. COMPARACIÓN CON GROUND TRUTH
# -------------------------------------------------------------
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

print(f"\n" + "="*60)
print("COMPARACIÓN CON ETIQUETAS REALES")
print("="*60)

ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)

print(f"\n  Adjusted Rand Index: {ari:.4f}")
print(f"  Normalized Mutual Info: {nmi:.4f}")

# Matriz de contingencia
print("\nMatriz de Contingencia:")
contingency = pd.crosstab(
    pd.Series(labels, name='Cluster'),
    pd.Series(y_true, name='Especie'),
    margins=True
)
contingency.columns = list(iris.target_names) + ['Total']
print(contingency)

# -------------------------------------------------------------
# 7. PERFIL DE CLUSTERS
# -------------------------------------------------------------
print(f"\n" + "="*60)
print("PERFIL DE CADA CLUSTER")
print("="*60)

df_result = pd.DataFrame(X, columns=feature_names)
df_result['cluster'] = labels

print("\nMedia por cluster (valores originales):")
print(df_result.groupby('cluster').mean().round(2))

# -------------------------------------------------------------
# 8. VISUALIZACIÓN FINAL
# -------------------------------------------------------------
from sklearn.decomposition import PCA

# Reducir a 2D para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Clusters jerárquicos
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                           cmap='viridis', edgecolors='w', s=50)
axes[0].set_title('Clustering Jerárquico (Ward)')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter1, ax=axes[0])

# Ground truth
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true,
                           cmap='viridis', edgecolors='w', s=50)
axes[1].set_title('Especies Reales')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter2, ax=axes[1])

# Comparación de métodos de enlace (para k=3)
labels_by_method = {}
for method in methods:
    labels_by_method[method] = fcluster(linkage_results[method], 
                                         t=3, criterion='maxclust')

# Mostrar silueta por método
silhouettes = [silhouette_score(X_scaled, labels_by_method[m]) for m in methods]
bars = axes[2].bar(methods, silhouettes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[2].set_ylabel('Silhouette Score')
axes[2].set_title('Silueta por Criterio de Enlace (k=3)')
axes[2].set_ylim([0, max(silhouettes) * 1.2])

# Añadir valores sobre las barras
for bar, val in zip(bars, silhouettes):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 9. EJEMPLO CON DISTANCE_THRESHOLD
# -------------------------------------------------------------
print("\n" + "="*60)
print("ALTERNATIVA: Clustering por Umbral de Distancia")
print("="*60)

# En lugar de especificar k, especificar distancia de corte
model_dist = AgglomerativeClustering(
    n_clusters=None,            # No especificar k
    distance_threshold=10,      # Umbral de distancia
    metric='euclidean',
    linkage='ward'
)

labels_dist = model_dist.fit_predict(X_scaled)
n_clusters_found = len(np.unique(labels_dist))

print(f"\nCon distance_threshold=10:")
print(f"  Clusters encontrados: {n_clusters_found}")
print(f"  Distribución: {np.bincount(labels_dist)}")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
```

---

## 4.6. Hiperparámetros en scikit-learn

### scipy.cluster.hierarchy.linkage

| Parámetro | Descripción | Valores |
| :--- | :--- | :--- |
| `y` | Datos o matriz de distancias | array (n, d) o condensada |
| `method` | Criterio de enlace | 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward' |
| `metric` | Métrica de distancia | 'euclidean', 'cityblock', 'cosine', etc. |
| `optimal_ordering` | Reordenar hojas para minimizar distancias | True/False |

### sklearn.cluster.AgglomerativeClustering

| Parámetro | Descripción | Valores |
| :--- | :--- | :--- |
| `n_clusters` | Número de clusters (None si se usa distance_threshold) | int o None |
| `distance_threshold` | Umbral de distancia para corte | float o None |
| `metric` | Métrica de distancia | 'euclidean', 'manhattan', 'cosine', etc. |
| `linkage` | Criterio de enlace | 'ward', 'complete', 'average', 'single' |
| `compute_full_tree` | Calcular árbol completo | 'auto', True, False |
| `compute_distances` | Calcular distancias entre clusters | True/False |

---

## 4.7. Criterio de Enlace: ¿Cuál Elegir?

| Criterio | Forma de Clusters | Sensibilidad a Outliers | Cuándo Usar |
| :--- | :--- | :--- | :--- |
| **Single** | Arbitraria, elongada | Baja | Detectar clusters de forma irregular |
| **Complete** | Compacta, esférica | Alta | Clusters de tamaño similar |
| **Average** | Moderada | Media | Balance general |
| **Ward** | Compacta, esférica | Media | Similar a K-Means, varianza mínima |

### Visualización del Efecto del Enlace

```python
# Datos con forma irregular (dos lunas)
from sklearn.datasets import make_moons
X_moons, _ = make_moons(n_samples=200, noise=0.05)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
methods = ['single', 'complete', 'average', 'ward']

for ax, method in zip(axes.flatten(), methods):
    model = AgglomerativeClustering(n_clusters=2, linkage=method)
    labels = model.fit_predict(X_moons)
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis')
    ax.set_title(f'{method.capitalize()} Linkage')

plt.tight_layout()
plt.show()
# Single linkage funcionará mejor con estos datos irregulares
```

---

## 4.8. Aplicaciones Reales

### 1. Análisis Filogenético (Biología)

Construir árboles evolutivos basados en similitud genética.
* **Tutorial:** [Phylogenetic Trees with Hierarchical Clustering](https://www.kaggle.com/code/andradaolteanu/hierarchical-clustering-phylogenetic-trees)

### 2. Segmentación de Documentos

Agrupar documentos similares para organización automática.

### 3. Análisis de Expresión Génica

Identificar grupos de genes con patrones de expresión similares.
* **Ejemplo:** [Gene Expression Clustering](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html)

### 4. Segmentación de Mercado

El dendrograma permite identificar segmentos a diferentes niveles de granularidad.

---

## 4.9. Comparación con Otros Métodos

| Aspecto | Jerárquico | K-Means | DBSCAN |
| :--- | :--- | :--- | :--- |
| Requiere k | No (a priori) | Sí | No |
| Escalabilidad | Baja ($O(n^3)$) | Alta ($O(nk)$) | Media ($O(n^2)$) |
| Formas de cluster | Depende del enlace | Esféricas | Arbitrarias |
| Outliers | No los detecta | No los detecta | Sí los detecta |
| Interpretabilidad | Alta (dendrograma) | Alta (centroides) | Moderada |
| Exploración multinivel | Sí | No | No |

---

## 4.10. Resumen y Mejores Prácticas

### Checklist para Clustering Jerárquico

- [ ] **Escalar los datos** (especialmente para Ward)
- [ ] **Elegir criterio de enlace** apropiado para la forma esperada de clusters
- [ ] **Calcular correlación cophenetica** para validar el dendrograma
- [ ] **Analizar el dendrograma** visualmente antes de cortar
- [ ] **Probar varios puntos de corte** y evaluar con métricas
- [ ] **Comparar con otros métodos** (K-Means, DBSCAN)

### ¿Cuándo Elegir Clustering Jerárquico?

✅ **Usar Jerárquico cuando:**
- Dataset pequeño-mediano (< 10K puntos)
- Quieres explorar la estructura a múltiples niveles
- El dendrograma es informativo para el dominio
- No sabes cuántos clusters hay

❌ **Considerar alternativas cuando:**
- Dataset grande → K-Means o Mini-Batch K-Means
- Necesitas identificar outliers → DBSCAN
- Clusters de formas muy irregulares → DBSCAN + Single linkage
- Eficiencia computacional es crítica

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
