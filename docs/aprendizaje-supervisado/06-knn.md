# 🤖 Unidad 6. Algoritmo K-Nearest Neighbors (KNN)

El algoritmo **K-Nearest Neighbors** (K-Vecinos Más Cercanos) es uno de los métodos más simples y efectivos en Machine Learning supervisado. Se utiliza tanto para problemas de **clasificación** como de **regresión**. Es un algoritmo **no paramétrico** (no hace suposiciones sobre la distribución de los datos subyacentes) y de **aprendizaje perezoso** (lazy learning), lo que significa que no "aprende" un modelo discriminativo durante la fase de entrenamiento, sino que memoriza los datos de entrenamiento para realizar predicciones en el momento necesario.

---

### 6.1. ¿Cómo funciona el algoritmo?

La intuición detrás de KNN es sencilla y se basa en la proximidad: "Dime con quién andas y te diré quién eres". Para clasificar un nuevo punto de datos, el algoritmo busca en todo el conjunto de datos de entrenamiento los **'k'** puntos más cercanos (vecinos) a ese nuevo punto.

1.  **Calcular distancias:** Se calcula la distancia matemática entre el punto nuevo que queremos predecir y todos los puntos existentes en el dataset.
2.  **Buscar vecinos:** Se seleccionan los $k$ puntos con las distancias más cortas.
3.  **Votación (Clasificación):** La clase del nuevo punto se determina por mayoría de votos de sus vecinos. La clase más frecuente entre los $k$ vecinos se asigna al nuevo punto.
4.  **Promedio (Regresión):** El valor del nuevo punto es el promedio (o media ponderada) de los valores numéricos de sus vecinos.

---

### 6.2. Explicación Matemática

El núcleo de KNN es la medición de la distancia para determinar la similitud. La métrica más común es la **Distancia Euclidiana**, aunque existen otras dependiendo del tipo de datos y el problema.

Dados dos puntos $P$ y $Q$ en un espacio n-dimensional (donde $n$ es el número de características):
$P = (p_1, p_2, ..., p_n)$
$Q = (q_1, q_2, ..., q_n)$

*   **Distancia Euclidiana (L2):** Es la distancia en línea recta "a vuelo de pájaro". Es la más utilizada por defecto.
    $$d(P, Q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

*   **Distancia Manhattan (L1):** Suma de las diferencias absolutas. Es útil en sistemas tipo cuadrícula (como manzanas de una ciudad).
    $$d(P, Q) = \sum_{i=1}^{n} |q_i - p_i|$$

*   **Distancia Minkowski:** Una generalización matemática de las anteriores.
    $$d(P, Q) = (\sum_{i=1}^{n} |q_i - p_i|^p)^{1/p}$$
    (Si $p=1$ es Manhattan, si $p=2$ es Euclidiana).

---

### 6.3. Pros y Contras

| Ventajas | Desventajas |
| :--- | :--- |
| **Simplicidad:** Es extremadamente fácil de entender, explicar e implementar. | **Costo Computacional:** Es lento en la fase de predicción con grandes datasets, ya que debe calcular distancias con *todos* los puntos cada vez. |
| **Sin Entrenamiento:** La fase de entrenamiento es casi instantánea (solo almacena datos), ideal si los datos cambian constantemente. | **Sensible a Outliers:** Los valores atípicos o ruido pueden afectar significativamente la predicción si $k$ es pequeño. |
| **Versatilidad:** Sirve tanto para tareas de clasificación como de regresión. | **Sensible a la Escala:** Requiere estrictamente que los datos estén normalizados o estandarizados. |
| **No Lineal:** Se adapta bien a fronteras de decisión irregulares y complejas. | **Maldición de la Dimensionalidad:** Su rendimiento decae drásticamente cuando hay muchas dimensiones (features) irrelevantes. |

---

### 6.4. Ejemplo en Python con `scikit-learn`

A continuación, un ejemplo básico de clasificación usando el dataset Iris y la clase `KNeighborsClassifier`.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# 1. Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Escalar datos (CRUCIAL para KNN)
# Como KNN usa distancias, las variables con rangos grandes dominarán a las pequeñas.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Instanciar el modelo (k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# 5. Entrenar (En KNN esto es solo almacenar los datos)
knn.fit(X_train, y_train)

# 6. Predecir y Evaluar
y_pred = knn.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```

---

### 6.5. Ejemplos Comunes de Uso

*   **Sistemas de Recomendación:** Sugerir productos, películas o música basándose en las preferencias de usuarios "vecinos" con gustos similares (Filtrado Colaborativo).
*   **Reconocimiento de Patrones:** Reconocimiento de caracteres escritos a mano (OCR) o clasificación de imágenes simples basándose en la similitud de píxeles.
*   **Detección de Anomalías:** Identificar fraudes bancarios o intrusiones en redes detectando eventos que están "lejos" de los grupos de vecinos normales.
*   **Imputación de Datos Faltantes:** Rellenar valores nulos en un dataset basándose en los valores de los vecinos más cercanos (`KNNImputer`).
*   **Medicina:** Clasificación de pacientes con perfiles similares para predecir riesgos de enfermedades.

### 6.6. Aplicaciones Reales de KNN

Aunque es un algoritmo simple, KNN se utiliza en sistemas donde la interpretabilidad y la simplicidad son clave:

*   **Sistemas de Recomendación (Retail):** Empresas como Amazon o Netflix utilizan variantes de algoritmos basados en vecindad para recomendar productos ("Los usuarios que compraron X también compraron Y").
    *   [Sistemas de recomendación con KNN](https://github.com/topics/recommendation-system)
*   **Reconocimiento de Escritura a Mano:** El servicio postal de EE.UU. (USPS) utilizó métodos basados en vecindad para reconocer dígitos escritos a mano en códigos postales.
    *   [Dataset MNIST y KNN](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
*   **Detección de Intrusiones (Ciberseguridad):** Clasificar actividades de red como normales o sospechosas basándose en su similitud con patrones de ataques conocidos.
*   **Bioinformática:** Clasificación de muestras de genes o proteínas basándose en su similitud con perfiles conocidos para el diagnóstico de enfermedades.

---

### 6.7. Consideraciones Finales

1.  **Elección de 'k' (Hiperparámetro clave):**
    *   Un $k$ muy pequeño (ej. $k=1$) hace que el modelo sea muy sensible al ruido (**Overfitting**).
    *   Un $k$ muy grande suaviza demasiado la frontera de decisión y puede incluir vecinos de otras clases lejanas (**Underfitting**).
    *   Se suele elegir un $k$ **impar** para evitar empates en clasificación binaria.
    *   El valor óptimo se encuentra usualmente mediante validación cruzada (técnica del codo o *Elbow Method*).

2.  **Escalado de Características:**
    *   Dado que KNN se basa puramente en distancias, es **obligatorio** escalar las variables (usando `StandardScaler` o `MinMaxScaler`). Si una variable tiene una magnitud mucho mayor que otra (ej. Salario [1000-5000] vs Edad [20-60]), la variable de mayor magnitud dominará completamente el cálculo de la distancia, haciendo que la otra sea irrelevante.

---

📅 **Fecha de creación:** 19/11/2025
✍️ **Autor:** Fran García
