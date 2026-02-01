# 🔍 Unidad 7. Isolation Forest - Detección de Anomalías

**Isolation Forest** es un algoritmo de **detección de anomalías no supervisado** basado en árboles de decisión. A diferencia de otros métodos que intentan modelar los datos normales, Isolation Forest se enfoca en **aislar las anomalías**. La idea clave es que las anomalías son más fáciles de aislar porque son raras y tienen valores atípicos, lo que significa que requieren menos divisiones para separarlas del resto.


![Ilustración de isolation forest](../assets/images/isolation_forest.svg)
---

## 7.1. ¿Cómo Funciona Isolation Forest?

### La Intuición

Imagina que tienes un bosque de puntos de datos. Si eliges un punto al azar y empiezas a hacer divisiones aleatorias:

- **Puntos normales:** Están en regiones densas, rodeados de muchos puntos → necesitan muchas divisiones para ser aislados
- **Anomalías:** Están solos, lejos del resto → se aíslan con pocas divisiones

```
┌─────────────────────────────────────────────────────────────┐
│ INTUICIÓN DE ISOLATION FOREST                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     Datos originales:                                       │
│                                                             │
│     ●●●●●●             ← Cluster denso                     │
│     ●●●●●●             (puntos normales)                   │
│     ●●●●●●                                                 │
│                                                             │
│                                         ⊗ ← Anomalía       │
│                                                             │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│     Una división aleatoria puede aislar la anomalía:       │
│                                                             │
│     ●●●●●● │                                               │
│     ●●●●●● │                                               │
│     ●●●●●● │                   ⊗ ← ¡Aislada con 1 corte!   │
│            │                                               │
│                                                             │
│     Pero un punto normal necesita más cortes:              │
│                                                             │
│     ●●─┬─●● │                                              │
│     ●●─┼─●● │    Necesita 3+ cortes para                   │
│     ●●─┴─●● │    aislar un punto del cluster               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### El Algoritmo

1. **Construir múltiples árboles de aislamiento (iForest)**
2. Para cada árbol:
   - Seleccionar aleatoriamente un subconjunto de datos
   - Seleccionar aleatoriamente una característica
   - Seleccionar aleatoriamente un valor de corte entre min y max
   - Dividir los datos recursivamente hasta que cada punto quede aislado o se alcance un límite
3. **Calcular la "path length"** (longitud del camino) promedio para cada punto
4. **Puntos con path length corta = Anomalías**

---

## 7.2. Explicación Matemática

### Path Length (Longitud del Camino)

La **path length** $h(x)$ de un punto $x$ es el número de divisiones necesarias para aislar ese punto desde la raíz hasta el nodo terminal.

Para un punto $x$, calculamos el **path length promedio** sobre todos los árboles:
$$E[h(x)] = \frac{1}{t} \sum_{i=1}^{t} h_i(x)$$

Donde $t$ es el número de árboles y $h_i(x)$ es la path length en el árbol $i$.

### Path Length Esperada

Para un árbol binario construido con $n$ puntos, la path length promedio esperada para un punto **normal** es aproximadamente:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

Donde $H(i)$ es el número harmónico: $H(i) = \ln(i) + \gamma$ (γ ≈ 0.5772 es la constante de Euler-Mascheroni).

### Anomaly Score

El **anomaly score** normaliza la path length:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

**Interpretación:**
- $s(x, n) \approx 1$: Punto es una anomalía (path length muy corta)
- $s(x, n) \approx 0.5$: Punto normal (path length promedio)
- $s(x, n) < 0.5$: Punto muy normal (path length larga)

```
      Anomaly Score
      
      1.0 ─────────────── ⊗ Anomalías
          
      0.5 ─────────────── ● Puntos normales
          
      0.0 ───────────────
```

---

## 7.3. Pros y Contras

| Ventajas | Desventajas |
| :--- | :--- |
| **No necesita etiquetas:** Completamente no supervisado | **Sensible a contaminación:** El parámetro contamination afecta mucho |
| **Muy eficiente:** Complejidad $O(n \log n)$ | **No ideal para datos de alta dimensión:** Pierde efectividad |
| **Escalable:** Funciona bien con datasets grandes | **Asume anomalías son aislables:** No funciona si las anomalías forman clusters |
| **Robusto:** No asume distribución de los datos | **No probabilístico:** Solo da scores, no probabilidades |
| **Pocos hiperparámetros:** Fácil de configurar | **Puede perder anomalías sutiles:** Si están cerca de datos normales |

---

## 7.4. Ejemplo Básico en Python

Este ejemplo muestra el uso básico de Isolation Forest para detectar anomalías.

```python
# ============================================================
# EJEMPLO BÁSICO: Isolation Forest para detección de anomalías
# ============================================================

# Importar bibliotecas necesarias
import numpy as np                          # Operaciones numéricas
import matplotlib.pyplot as plt             # Visualización
from sklearn.ensemble import IsolationForest  # Algoritmo principal
from sklearn.datasets import make_blobs     # Generar datos sintéticos

# -------------------------------------------------------------
# 1. CREAR DATOS CON ANOMALÍAS
# -------------------------------------------------------------
# Generar datos normales (cluster)
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)

# Añadir anomalías (puntos lejanos del cluster)
n_anomalies = 20
X_anomalies = np.random.uniform(low=-6, high=6, size=(n_anomalies, 2))

# Combinar datos
X = np.vstack([X_normal, X_anomalies])

# Crear etiquetas verdaderas para evaluación
# 1 = normal, -1 = anomalía
y_true = np.array([1] * len(X_normal) + [-1] * n_anomalies)

print("="*50)
print("ISOLATION FOREST - DETECCIÓN DE ANOMALÍAS")
print("="*50)
print(f"\nTotal de puntos: {len(X)}")
print(f"Puntos normales: {len(X_normal)}")
print(f"Anomalías verdaderas: {n_anomalies}")
print(f"Tasa de contaminación real: {n_anomalies/len(X):.2%}")

# -------------------------------------------------------------
# 2. VISUALIZAR DATOS ORIGINALES
# -------------------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_normal[:, 0], X_normal[:, 1], c='blue', alpha=0.6, label='Normal', s=30)
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c='red', marker='x', s=100, 
            label='Anomalías', linewidths=2)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Datos Originales (con etiquetas verdaderas)')
plt.legend()
plt.grid(True, alpha=0.3)

# -------------------------------------------------------------
# 3. APLICAR ISOLATION FOREST
# -------------------------------------------------------------
# Crear y entrenar el modelo
# contamination: proporción esperada de anomalías
iso_forest = IsolationForest(
    n_estimators=100,           # Número de árboles
    contamination=0.1,          # Proporción esperada de anomalías (10%)
    random_state=42,            # Reproducibilidad
    max_samples='auto'          # Muestras por árbol
)

# Entrenar y predecir
# fit_predict devuelve: 1 (normal), -1 (anomalía)
y_pred = iso_forest.fit_predict(X)

# -------------------------------------------------------------
# 4. ANALIZAR RESULTADOS
# -------------------------------------------------------------
# Contar predicciones
n_pred_normal = np.sum(y_pred == 1)
n_pred_anomaly = np.sum(y_pred == -1)

print(f"\n--- Resultados de Isolation Forest ---")
print(f"Predichos como normal: {n_pred_normal}")
print(f"Predichos como anomalía: {n_pred_anomaly}")

# Calcular métricas de rendimiento
from sklearn.metrics import classification_report, confusion_matrix

# Convertir etiquetas para métricas
print(f"\n--- Reporte de Clasificación ---")
print(classification_report(y_true, y_pred, target_names=['Anomalía (-1)', 'Normal (1)']))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print(f"Matriz de Confusión:")
print(f"                 Predicho Anomalía | Predicho Normal")
print(f"Real Anomalía:          {cm[0,0]:4d}      |      {cm[0,1]:4d}")
print(f"Real Normal:            {cm[1,0]:4d}      |      {cm[1,1]:4d}")

# -------------------------------------------------------------
# 5. VISUALIZAR PREDICCIONES
# -------------------------------------------------------------
plt.subplot(1, 2, 2)

# Separar por predicción
mask_pred_normal = y_pred == 1
mask_pred_anomaly = y_pred == -1

plt.scatter(X[mask_pred_normal, 0], X[mask_pred_normal, 1], 
            c='green', alpha=0.6, label='Predicho Normal', s=30)
plt.scatter(X[mask_pred_anomaly, 0], X[mask_pred_anomaly, 1], 
            c='red', marker='x', s=100, label='Predicho Anomalía', linewidths=2)

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Predicciones de Isolation Forest')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6. OBTENER ANOMALY SCORES
# -------------------------------------------------------------
# decision_function devuelve el score de anomalía
# Valores más negativos = más anómalo
scores = iso_forest.decision_function(X)

# score_samples devuelve valores similares (opuesto de la depth)
# Valores más negativos = más anómalo

print(f"\n--- Anomaly Scores ---")
print(f"Score medio (normales): {scores[y_true == 1].mean():.3f}")
print(f"Score medio (anomalías): {scores[y_true == -1].mean():.3f}")
print(f"Rango de scores: [{scores.min():.3f}, {scores.max():.3f}]")

# Visualizar distribución de scores
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(scores[y_true == 1], bins=30, alpha=0.7, label='Normal', color='blue')
plt.hist(scores[y_true == -1], bins=10, alpha=0.7, label='Anomalía', color='red')
plt.xlabel('Anomaly Score (decision_function)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Anomaly Scores')
plt.legend()
plt.axvline(x=0, color='black', linestyle='--', label='Umbral (0)')

plt.subplot(1, 2, 2)
# Colorear puntos por score
scatter = plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlGn', 
                      alpha=0.7, s=30, edgecolors='k', linewidths=0.5)
plt.colorbar(scatter, label='Anomaly Score')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Mapa de Anomaly Scores\n(Verde=Normal, Rojo=Anomalía)')

plt.tight_layout()
plt.show()

print("""
Interpretación de decision_function:
- Valores positivos → punto probablemente normal
- Valores negativos → punto probablemente anómalo
- El umbral por defecto es 0 (controlado por contamination)
""")
```

---

## 7.5. Ejemplo Avanzado: Análisis de Hiperparámetros y Casos de Uso

Este ejemplo explora la configuración óptima y casos de uso reales.

```python
# ============================================================
# EJEMPLO AVANZADO: Isolation Forest - Análisis profundo
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# 1. CREAR DATASET COMPLEJO
# -------------------------------------------------------------
np.random.seed(42)

# Datos normales: dos clusters
from sklearn.datasets import make_blobs
X_normal1, _ = make_blobs(n_samples=400, centers=[[2, 2]], cluster_std=0.8)
X_normal2, _ = make_blobs(n_samples=400, centers=[[-2, -2]], cluster_std=0.8)
X_normal = np.vstack([X_normal1, X_normal2])

# Anomalías de diferentes tipos
anomalies_global = np.random.uniform(-6, 6, (15, 2))  # Aleatorias
anomalies_local = np.array([[0, 0], [0.5, 0.5], [-0.5, -0.5]])  # Entre clusters
anomalies_edge = np.array([[4, 2], [2, 4], [-4, -2]])  # En bordes

X_anomalies = np.vstack([anomalies_global, anomalies_local, anomalies_edge])

X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * len(X_normal) + [-1] * len(X_anomalies))

print("="*60)
print("ISOLATION FOREST - ANÁLISIS AVANZADO")
print("="*60)
print(f"\nDataset: {len(X)} puntos ({len(X_normal)} normales, {len(X_anomalies)} anomalías)")

# -------------------------------------------------------------
# 2. EFECTO DEL NÚMERO DE ESTIMADORES
# -------------------------------------------------------------
print("\n[1] EFECTO DEL NÚMERO DE ESTIMADORES (n_estimators)")
print("-"*50)

n_estimators_list = [10, 50, 100, 200, 500]
results_estimators = []

for n_est in n_estimators_list:
    iso = IsolationForest(n_estimators=n_est, contamination=0.05, random_state=42)
    y_pred = iso.fit_predict(X)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    results_estimators.append(f1)
    print(f"  n_estimators={n_est:3d}: F1={f1:.3f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(n_estimators_list, results_estimators, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Estimadores')
plt.ylabel('F1-Score (Anomalías)')
plt.title('Efecto del Número de Árboles')
plt.grid(True, alpha=0.3)

# -------------------------------------------------------------
# 3. EFECTO DE LA CONTAMINACIÓN
# -------------------------------------------------------------
print("\n[2] EFECTO DE LA CONTAMINACIÓN (contamination)")
print("-"*50)

contamination_list = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
results_contamination = {'precision': [], 'recall': [], 'f1': []}

for cont in contamination_list:
    iso = IsolationForest(n_estimators=100, contamination=cont, random_state=42)
    y_pred = iso.fit_predict(X)
    
    prec = precision_score(y_true, y_pred, pos_label=-1)
    rec = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    
    results_contamination['precision'].append(prec)
    results_contamination['recall'].append(rec)
    results_contamination['f1'].append(f1)
    
    print(f"  contamination={cont:.2f}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

plt.subplot(1, 2, 2)
plt.plot(contamination_list, results_contamination['precision'], 'g^-', label='Precision', linewidth=2)
plt.plot(contamination_list, results_contamination['recall'], 'rs-', label='Recall', linewidth=2)
plt.plot(contamination_list, results_contamination['f1'], 'bo-', label='F1-Score', linewidth=2)
plt.axvline(x=len(X_anomalies)/len(X), color='k', linestyle='--', alpha=0.5, label='Contam. real')
plt.xlabel('Contamination')
plt.ylabel('Score')
plt.title('Efecto del Parámetro Contamination')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nContaminación real en los datos: {len(X_anomalies)/len(X):.3f}")

# -------------------------------------------------------------
# 4. EFECTO DE max_samples
# -------------------------------------------------------------
print("\n[3] EFECTO DE max_samples")
print("-"*50)

max_samples_list = [32, 64, 128, 256, 'auto']
results_samples = []

for max_s in max_samples_list:
    iso = IsolationForest(n_estimators=100, contamination=0.05, 
                          max_samples=max_s, random_state=42)
    y_pred = iso.fit_predict(X)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    results_samples.append(f1)
    print(f"  max_samples={str(max_s):5s}: F1={f1:.3f}")

# -------------------------------------------------------------
# 5. CURVA ROC CON DIFERENTES UMBRALES
# -------------------------------------------------------------
print("\n[4] ANÁLISIS DE UMBRALES Y CURVA ROC")
print("-"*50)

# Entrenar modelo
iso = IsolationForest(n_estimators=100, random_state=42)
iso.fit(X)

# Obtener scores (invertir signo para ROC)
scores = -iso.decision_function(X)  # Más alto = más anómalo

# Calcular ROC
fpr, tpr, thresholds = roc_curve(y_true == -1, scores)
roc_auc = auc(fpr, tpr)

print(f"  AUC-ROC: {roc_auc:.3f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualizar diferentes umbrales
plt.subplot(1, 3, 2)

# Umbral automático (contamination=0.05)
iso_auto = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
y_pred_auto = iso_auto.fit_predict(X)

mask_normal = y_pred_auto == 1
mask_anomaly = y_pred_auto == -1

plt.scatter(X[mask_normal, 0], X[mask_normal, 1], c='blue', alpha=0.5, s=20, label='Normal')
plt.scatter(X[mask_anomaly, 0], X[mask_anomaly, 1], c='red', marker='x', s=80, 
            label='Anomalía', linewidths=2)
plt.title('Contamination = 5%')
plt.legend()

# Umbral más estricto
plt.subplot(1, 3, 3)

iso_strict = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
y_pred_strict = iso_strict.fit_predict(X)

mask_normal = y_pred_strict == 1
mask_anomaly = y_pred_strict == -1

plt.scatter(X[mask_normal, 0], X[mask_normal, 1], c='blue', alpha=0.5, s=20, label='Normal')
plt.scatter(X[mask_anomaly, 0], X[mask_anomaly, 1], c='red', marker='x', s=80, 
            label='Anomalía', linewidths=2)
plt.title('Contamination = 2%')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6. VISUALIZAR REGIONES DE DECISIÓN
# -------------------------------------------------------------
print("\n[5] VISUALIZACIÓN DE REGIONES DE DECISIÓN")
print("-"*50)

# Crear grid para visualización
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
grid = np.c_[xx.ravel(), yy.ravel()]

# Entrenar modelo
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X)

# Obtener scores para el grid
Z = iso.decision_function(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Contour de scores
contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.8)
plt.colorbar(contour, label='Anomaly Score')

# Puntos reales
plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], c='blue', edgecolors='k', 
            s=20, alpha=0.6, label='Normal')
plt.scatter(X[y_true == -1, 0], X[y_true == -1, 1], c='red', marker='x',
            s=100, linewidths=2, label='Anomalía real')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Mapa de Anomaly Scores\n(Verde=Normal, Rojo=Anómalo)')
plt.legend()

plt.subplot(1, 2, 2)
# Frontera de decisión
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.contourf(xx, yy, Z, levels=[Z.min(), 0], colors=['red'], alpha=0.3)
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors=['green'], alpha=0.3)

plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], c='blue', edgecolors='k',
            s=20, alpha=0.6, label='Normal')
plt.scatter(X[y_true == -1, 0], X[y_true == -1, 1], c='red', marker='x',
            s=100, linewidths=2, label='Anomalía real')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Frontera de Decisión\n(Rojo=Región anómala)')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 7. CASO REAL: DETECCIÓN DE FRAUDE (SIMULADO)
# -------------------------------------------------------------
print("\n[6] CASO DE USO: DETECCIÓN DE FRAUDE")
print("-"*50)

# Simular datos de transacciones
np.random.seed(42)
n_normal = 10000
n_fraud = 100  # 1% de fraude

# Transacciones normales: monto bajo-medio, hora comercial
normal_amount = np.abs(np.random.normal(50, 30, n_normal))
normal_hour = np.random.normal(14, 4, n_normal)  # Centrado en 2pm
normal_hour = np.clip(normal_hour, 0, 24)

# Transacciones fraudulentas: montos altos, horas inusuales
fraud_amount = np.abs(np.random.normal(500, 200, n_fraud))
fraud_hour = np.random.uniform(0, 6, n_fraud)  # Madrugada

# Combinar
X_fraud = np.column_stack([
    np.concatenate([normal_amount, fraud_amount]),
    np.concatenate([normal_hour, fraud_hour])
])
y_fraud = np.array([1] * n_normal + [-1] * n_fraud)

# Estandarizar
scaler = StandardScaler()
X_fraud_scaled = scaler.fit_transform(X_fraud)

# Entrenar Isolation Forest
iso_fraud = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
y_pred_fraud = iso_fraud.fit_predict(X_fraud_scaled)

# Resultados
print(f"  Transacciones totales: {len(X_fraud):,}")
print(f"  Fraudes reales: {n_fraud}")
print(f"  Fraudes detectados: {np.sum(y_pred_fraud == -1)}")

# Métricas
prec = precision_score(y_fraud, y_pred_fraud, pos_label=-1)
rec = recall_score(y_fraud, y_pred_fraud, pos_label=-1)
f1 = f1_score(y_fraud, y_pred_fraud, pos_label=-1)

print(f"\n  Precision: {prec:.3f} (De los detectados, qué % son fraude)")
print(f"  Recall: {rec:.3f} (De los fraudes reales, qué % detectamos)")
print(f"  F1-Score: {f1:.3f}")

# Visualizar
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_fraud[y_fraud == 1, 0], X_fraud[y_fraud == 1, 1], 
            c='green', alpha=0.3, s=5, label='Normal')
plt.scatter(X_fraud[y_fraud == -1, 0], X_fraud[y_fraud == -1, 1],
            c='red', marker='x', s=50, label='Fraude real')
plt.xlabel('Monto ($)')
plt.ylabel('Hora del día')
plt.title('Datos Reales')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_fraud[y_pred_fraud == 1, 0], X_fraud[y_pred_fraud == 1, 1],
            c='green', alpha=0.3, s=5, label='Predicho Normal')
plt.scatter(X_fraud[y_pred_fraud == -1, 0], X_fraud[y_pred_fraud == -1, 1],
            c='red', marker='x', s=50, label='Predicho Fraude')
plt.xlabel('Monto ($)')
plt.ylabel('Hora del día')
plt.title('Predicciones de Isolation Forest')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 8. MEJORES PRÁCTICAS
# -------------------------------------------------------------
print("\n" + "="*60)
print("MEJORES PRÁCTICAS PARA ISOLATION FOREST")
print("="*60)

print("""
1. PREPROCESAMIENTO:
   - Estandarizar/normalizar las características
   - Considerar encoding adecuado para categorías
   - Manejar valores faltantes antes

2. HIPERPARÁMETROS:
   - n_estimators: 100 suele ser suficiente
   - contamination: Usar conocimiento del dominio si es posible
   - max_samples: 'auto' o sqrt(n_samples)

3. EVALUACIÓN:
   - Si hay etiquetas: Precision, Recall, F1, AUC-ROC
   - Sin etiquetas: Inspección manual de anomalías detectadas
   - Analizar distribución de scores

4. CONSIDERACIONES:
   - No asume distribución de los datos
   - Funciona mejor con anomalías globales/aisladas
   - Puede fallar con anomalías en clusters
   - Revisar puntos cerca del umbral manualmente
""")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
```

---

## 7.6. Hiperparámetros de Isolation Forest en scikit-learn

| Parámetro | Descripción | Valores | Recomendación |
| :--- | :--- | :--- | :--- |
| `n_estimators` | Número de árboles | int > 0 | 100 (más = más estable) |
| `contamination` | Proporción esperada de anomalías | float (0, 0.5) o 'auto' | Basado en conocimiento del dominio |
| `max_samples` | Muestras por árbol | int, float, 'auto' | 'auto' (min(256, n_samples)) |
| `max_features` | Características por árbol | int o float | 1.0 (todas) |
| `bootstrap` | Muestreo con reemplazo | bool | False |
| `random_state` | Semilla | int o None | Fijar para reproducibilidad |
| `n_jobs` | Procesadores paralelos | int | -1 para usar todos |
| `warm_start` | Añadir árboles incrementalmente | bool | False |

---

## 7.7. Aplicaciones Reales

### 1. Detección de Fraude Financiero
Identificar transacciones sospechosas en tarjetas de crédito.
* [Ejemplo: Credit Card Fraud Detection](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)

### 2. Detección de Intrusiones en Redes
Identificar patrones de tráfico anómalos que puedan indicar ataques.

### 3. Mantenimiento Predictivo
Detectar comportamientos anómalos en sensores de maquinaria industrial.

### 4. Control de Calidad
Identificar productos defectuosos en líneas de producción.

### 5. Salud
Detectar patrones anómalos en datos médicos (ECG, diagnósticos).

---

## 7.8. Comparación con Otros Métodos de Detección de Anomalías

| Método | Tipo | Velocidad | Escalabilidad | Mejor para |
| :--- | :--- | :--- | :--- | :--- |
| **Isolation Forest** | Basado en árboles | Rápido | Excelente | Anomalías globales |
| **One-Class SVM** | Basado en kernel | Lento | Pobre | Datasets pequeños |
| **LOF** | Basado en densidad | Medio | Media | Anomalías locales |
| **DBSCAN** | Clustering | Medio | Buena | Clusters + anomalías |
| **Autoencoder** | Deep Learning | Lento (train) | Buena | Alta dimensionalidad |

---

## 7.9. Resumen y Checklist

### Checklist para usar Isolation Forest

- [ ] **Preprocesar los datos** (estandarizar, manejar NaN)
- [ ] **Estimar contamination** si es posible
- [ ] **Empezar con n_estimators=100**
- [ ] **Visualizar anomaly scores** para entender la distribución
- [ ] **Ajustar contamination** según resultados
- [ ] **Validar con métricas** si hay etiquetas disponibles
- [ ] **Inspeccionar manualmente** las anomalías detectadas

### ¿Cuándo usar Isolation Forest?

✅ **Usar Isolation Forest cuando:**
- Tienes un dataset grande
- Las anomalías son raras y diferentes al resto
- No tienes etiquetas de anomalías
- Necesitas un método rápido y escalable

❌ **Considerar alternativas cuando:**
- Las anomalías forman clusters → LOF o DBSCAN
- Dataset muy pequeño → One-Class SVM
- Datos de alta dimensión complejos → Autoencoders
- Necesitas probabilidades → Gaussian Mixture

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
