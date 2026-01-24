# 🤖 Unidad 3. Modelos de Aprendizaje Supervisado para Predicción Categórica

La **clasificación** es una subcategoría del aprendizaje supervisado donde el objetivo es predecir una etiqueta de clase categórica (discreta) para una instancia de datos dada. A diferencia de la regresión, que predice valores continuos, la clasificación asigna entradas a una de varias categorías predefinidas.

---

### 3.1. Entrenamiento y Testing en Clasificación

El proceso de construcción de un modelo de clasificación sigue el flujo estándar de Machine Learning:

1.  **División de Datos:** Se divide el dataset en un conjunto de **entrenamiento** (para ajustar el modelo) y un conjunto de **prueba** (para evaluar su rendimiento en datos no vistos).
2.  **Entrenamiento:** El algoritmo aprende la frontera de decisión que separa las diferentes clases basándose en las características (features) de los datos de entrenamiento.
3.  **Testing (Predicción):** El modelo asigna etiquetas a los datos de prueba.
4.  **Evaluación:** Se comparan las etiquetas predichas con las etiquetas reales para calcular métricas de rendimiento.

---

### 3.2. Ejemplos Frecuentes de Uso

La clasificación está omnipresente en aplicaciones modernas:

*   **Detección de Spam:** Clasificar correos como "Spam" o "No Spam".
*   **Diagnóstico Médico:** Determinar si un paciente tiene una enfermedad ("Positivo") o no ("Negativo") basándose en síntomas y análisis.
*   **Reconocimiento de Imágenes:** Identificar si una imagen contiene un "Gato", "Perro" o "Coche".
*   **Aprobación de Créditos:** Clasificar a un solicitante como de "Alto Riesgo" o "Bajo Riesgo".
*   **Análisis de Sentimientos:** Clasificar opiniones como "Positivas", "Negativas" o "Neutrales".

---

### 3.3. Algoritmos de Clasificación en Machine Learning

Existen diversos algoritmos para abordar problemas de clasificación:

*   **Regresión Logística:** Simple, interpretable y base para redes neuronales.
*   **K-Nearest Neighbors (KNN):** Basado en similitud y distancia.
*   **Support Vector Machines (SVM):** Busca el hiperplano de separación óptimo.
*   **Árboles de Decisión y Random Forest:** Basados en reglas de decisión jerárquicas.
*   **Naive Bayes:** Basado en probabilidad y el teorema de Bayes.
*   **Redes Neuronales:** Para patrones complejos y datos no estructurados.

---

### 3.4. Regresión Logística

A pesar de su nombre, la **Regresión Logística** es un algoritmo de **clasificación**, no de regresión. Se utiliza para estimar la probabilidad de que una instancia pertenezca a una clase particular (por ejemplo, probabilidad de que un correo sea spam).

#### Conceptos Básicos y Matemáticos

La regresión logística utiliza la **función sigmoide** (o logística) para transformar la salida de una ecuación lineal en un valor de probabilidad entre 0 y 1.

1.  **Función Lineal:** $z = w \cdot x + b$ (donde $w$ son los pesos y $x$ las características).
2.  **Función Sigmoide:** $\sigma(z) = \frac{1}{1 + e^{-z}}$

Si la probabilidad estimada $\hat{p} = \sigma(z)$ es mayor o igual a 0.5, el modelo predice la clase 1; de lo contrario, predice la clase 0.

#### Algoritmo del Gradiente Descendente

Para entrenar el modelo, necesitamos encontrar los pesos $w$ y el sesgo $b$ que minimicen el error. La función de costo utilizada es la **Log Loss** (Pérdida Logarítmica), ya que el error cuadrático medio no es convexo para esta función.

El **Gradiente Descendente** es un algoritmo de optimización iterativo:
1.  Inicializa los pesos aleatoriamente.
2.  Calcula el gradiente de la función de costo (la dirección en la que el error aumenta más rápido).
3.  Actualiza los pesos moviéndose en la dirección opuesta al gradiente para reducir el error.
    $$w_{nuevo} = w_{viejo} - \eta \cdot \nabla Costo$$
    (Donde $\eta$ es la tasa de aprendizaje).

#### Ejemplo en Python

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Cargar datos
data = load_breast_cancer()
X, y = data.data, data.target

# Dividir y Escalar (Importante para Gradiente Descendente)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar modelo
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predecir
y_pred = log_reg.predict(X_test)
```

---

### 3.5. Métricas de Rendimiento

Evaluar un clasificador va más allá de simplemente contar cuántos aciertos tuvo.

#### Matriz de Confusión
Es una tabla que resume el rendimiento del modelo comparando las clases reales con las predichas.

| | Predicho Negativo (0) | Predicho Positivo (1) |
| :--- | :---: | :---: |
| **Real Negativo (0)** | **TN** (True Negative) | **FP** (False Positive) |
| **Real Positivo (1)** | **FN** (False Negative) | **TP** (True Positive) |

*   **TP:** Enfermos detectados correctamente.
*   **TN:** Sanos detectados correctamente.
*   **FP (Error Tipo I):** Sanos detectados erróneamente como enfermos ("Falsa Alarma").
*   **FN (Error Tipo II):** Enfermos no detectados ("Peligroso").

#### Métricas Derivadas

1.  **Accuracy (Exactitud):** Proporción total de predicciones correctas.
    $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

2.  **Error Rate (Tasa de Error):** Proporción de predicciones incorrectas.
    $$Error Rate = 1 - Accuracy = \frac{FP + FN}{Total}$$

3.  **Sensitivity / Recall / TPR (Tasa de Verdaderos Positivos):** Capacidad para detectar la clase positiva.
    $$Sensitivity = \frac{TP}{TP + FN}$$

4.  **Specificity / TNR (Tasa de Verdaderos Negativos):** Capacidad para detectar la clase negativa.
    $$Specificity = \frac{TN}{TN + FP}$$

5.  **False Positive Rate (FPR):**
    $$FPR = 1 - Specificity = \frac{FP}{TN + FP}$$

6.  **Precision (Precisión):** De los que predije positivos, ¿cuántos lo son realmente?
    $$Precision = \frac{TP}{TP + FP}$$

7.  **F1-Score (F-Measure):** Media armónica de Precision y Recall. Útil cuando las clases están desbalanceadas.
    $$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

8.  **Kappa Statistic (Cohen's Kappa):** Mide la concordancia entre la predicción y la realidad, ajustada por el azar. Un valor de 1 es concordancia perfecta, 0 es igual al azar.

#### Ejemplo en Python

```python
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
print(f"Kappa Score: {cohen_kappa_score(y_test, y_pred):.4f}")
```

---

### 3.6. Curva ROC y AUC

La **Curva ROC** (Receiver Operating Characteristic) es un gráfico que ilustra el rendimiento de un clasificador binario a medida que varía el umbral de discriminación.
*   **Eje X:** False Positive Rate (1 - Specificity).
*   **Eje Y:** True Positive Rate (Sensitivity).

Un modelo ideal se acerca a la esquina superior izquierda (TPR=1, FPR=0). La línea diagonal representa un clasificador aleatorio.

**AUC (Area Under Curve):** Es el área bajo la curva ROC. Resume el rendimiento en un solo número.
*   AUC = 0.5: Aleatorio.
*   AUC = 1.0: Perfecto.

---

### 3.7. Sensibilidad, Especificidad y el Teorema de Bayes

Estos conceptos están íntimamente ligados al Teorema de Bayes cuando queremos calcular la probabilidad real de tener una condición dado un resultado positivo en un test (Probabilidad a Posteriori).

Supongamos un test médico para una enfermedad rara:
*   $P(E)$: Probabilidad a priori de tener la enfermedad (Prevalencia).
*   $P(+|E)$: Sensibilidad del test.
*   $P(-|No E)$: Especificidad del test.

Si un paciente da positivo, ¿cuál es la probabilidad de que realmente tenga la enfermedad $P(E|+)$?

$$P(E|+) = \frac{P(+|E) \cdot P(E)}{P(+|E) \cdot P(E) + P(+|No E) \cdot P(No E)}$$

Donde $P(+|No E)$ es el False Positive Rate ($1 - Especificidad$).
Este cálculo demuestra que si la prevalencia de la enfermedad es muy baja, incluso un test con alta sensibilidad y especificidad puede generar muchos falsos positivos, haciendo que la probabilidad real de estar enfermo sea baja a pesar del resultado positivo.

---

📅 **Fecha de creación:** 19/11/2025
✍️ **Autor:** Fran García
