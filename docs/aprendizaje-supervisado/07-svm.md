# 🤖 Unidad 7. Máquinas de Vectores de Soporte (SVM)

Las **Máquinas de Vectores de Soporte** (Support Vector Machines o SVM) son un conjunto de algoritmos de aprendizaje supervisado potentes y versátiles, utilizados tanto para **clasificación** (SVC) como para **regresión** (SVR). Su objetivo principal es encontrar el hiperplano óptimo que mejor separe las clases en el espacio de características.

---

### 7.1. ¿Cómo funciona el algoritmo?

La idea central de SVM es encontrar una línea (en 2D), un plano (en 3D) o un **hiperplano** (en más dimensiones) que divida los datos en clases distintas. Pero no cualquier separación sirve; SVM busca la separación que tenga el **mayor margen** posible.

1.  **Hiperplano:** Es la frontera de decisión que separa las clases.
2.  **Vectores de Soporte:** Son los puntos de datos más cercanos al hiperplano. Estos puntos son los más "difíciles" de clasificar y son los únicos que importan para definir la posición del hiperplano.
3.  **Margen:** Es la distancia entre el hiperplano y los vectores de soporte más cercanos de cada clase. SVM intenta **maximizar** este margen para mejorar la generalización del modelo.

---

### 7.2. Explicación Matemática y el "Kernel Trick"

Matemáticamente, para un problema linealmente separable, buscamos los parámetros $w$ (vector de pesos) y $b$ (sesgo) tal que el hiperplano se defina como:
$$w \cdot x + b = 0$$

El objetivo es minimizar $||w||$ (lo que equivale a maximizar el margen) sujeto a que todas las muestras estén correctamente clasificadas fuera del margen.

#### El Truco del Kernel (Kernel Trick)
Cuando los datos no son separables linealmente (ej. un círculo dentro de otro), SVM utiliza una técnica llamada **Kernel Trick**.
Esta técnica proyecta los datos originales a un espacio de **mayor dimensión** donde sí son linealmente separables, sin necesidad de calcular explícitamente las coordenadas en ese espacio complejo (lo cual sería computacionalmente costoso).

Kernels comunes:
*   **Lineal:** Para datos linealmente separables.
*   **Polinómico:** Mapea a espacios de dimensiones polinómicas.
*   **RBF (Radial Basis Function):** El más popular. Mapea a un espacio de dimensión infinita. Es muy efectivo para fronteras de decisión complejas y curvas.

---

### 7.3. Pros y Contras

| Ventajas | Desventajas |
| :--- | :--- |
| **Alta Dimensionalidad:** Es muy efectivo en espacios con muchas dimensiones (incluso si hay más dimensiones que muestras). | **Grandes Datasets:** No escala bien con datasets muy grandes (el tiempo de entrenamiento crece cúbicamente). |
| **Eficiencia de Memoria:** Solo usa un subconjunto de puntos de entrenamiento (los vectores de soporte) para definir el modelo. | **Ruido:** Es sensible al ruido y a clases que se solapan mucho (si no se ajustan bien los parámetros). |
| **Versatilidad:** Gracias a los Kernels, puede modelar relaciones lineales y no lineales complejas. | **Probabilidades:** No proporciona estimaciones de probabilidad directas (se calculan mediante validación cruzada costosa). |
| **Robustez:** Maximizar el margen ayuda a reducir el riesgo de overfitting. | **Ajuste de Parámetros:** Requiere un ajuste cuidadoso de hiperparámetros clave ($C$, $\gamma$, Kernel). |

---

### 7.4. Ejemplo en Python con `scikit-learn`

Ejemplo de clasificación usando `SVC` (Support Vector Classification) con un kernel RBF.

```python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# 1. Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Escalar datos (IMPORTANTE para SVM)
# SVM es sensible a la escala porque intenta maximizar la distancia (margen).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Instanciar el modelo
# kernel='rbf' es el valor por defecto. C es el parámetro de regularización.
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

# 5. Entrenar
clf.fit(X_train, y_train)

# 6. Predecir y Evaluar
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```

---

### 7.5. Ejemplos Comunes de Uso

*   **Clasificación de Texto:** Categorización de noticias, detección de spam y análisis de sentimientos. SVM maneja muy bien la alta dimensionalidad de los vectores de texto (Bag of Words).
*   **Reconocimiento de Imágenes:** Clasificación de imágenes, reconocimiento facial y reconocimiento de escritura a mano (OCR).
*   **Bioinformática:** Clasificación de proteínas y genes, donde los datos suelen tener muchas características y pocas muestras.
*   **Detección de Intrusos:** Identificar actividad maliciosa en redes basándose en patrones de tráfico.

### 7.6. Aplicaciones Reales de SVM

SVM ha sido uno de los algoritmos más exitosos antes del auge del Deep Learning y sigue siendo muy relevante:

*   **Clasificación de Imágenes (Histórico):** Antes de las redes neuronales convolucionales (CNN), SVM era el estándar para clasificación de imágenes y detección de objetos (ej. detección de peatones).
*   **Bioinformática (Clasificación de Proteínas):** Se utiliza para clasificar proteínas en familias funcionales y predecir la estructura secundaria de las proteínas, dado que maneja muy bien la alta dimensionalidad de los datos genómicos.
    *   [SVM en Bioinformática](https://github.com/topics/bioinformatics-machine-learning)
*   **Reconocimiento de Escritura:** SVM ha demostrado ser muy eficaz en el reconocimiento de caracteres manuscritos (OCR), compitiendo con redes neuronales en datasets como MNIST.
*   **Geología y Minería:** Clasificación de tipos de suelo y rocas a partir de datos sísmicos o imágenes satelitales.

---

### 7.7. Consideraciones Finales

1.  **Parámetro C (Regularización):**
    *   Controla el equilibrio entre tener un margen amplio y clasificar correctamente los puntos de entrenamiento.
    *   **C alto:** Intenta clasificar todo correctamente (riesgo de Overfitting, margen estrecho).
    *   **C bajo:** Permite algunos errores para obtener un margen más amplio (mejor generalización, margen suave).

2.  **Parámetro Gamma ($\gamma$) (Solo para kernels RBF/Poly):**
    *   Define qué tan lejos llega la influencia de un solo ejemplo de entrenamiento.
    *   **Gamma alto:** Solo los puntos muy cercanos influyen. Puede llevar a fronteras de decisión muy ajustadas e irregulares (Overfitting).
    *   **Gamma bajo:** La influencia llega lejos. La frontera de decisión es más suave (Underfitting si es muy bajo).

3.  **Escalado:** Al igual que KNN, SVM se basa en distancias. Es **crítico** estandarizar los datos antes de entrenar.

---

📅 **Fecha de creación:** 19/11/2025
✍️ **Autor:** Fran García
