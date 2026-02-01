# 🤖 Unidad 7. Máquinas de Vectores de Soporte (SVM)

Las **Máquinas de Vectores de Soporte** (Support Vector Machines o SVM) son un conjunto de algoritmos de aprendizaje supervisado potentes y versátiles, utilizados tanto para **clasificación** (SVC) como para **regresión** (SVR). Su objetivo principal es encontrar el hiperplano óptimo que mejor separe las clases en el espacio de características.

---

### 7.1. ¿Cómo funciona el algoritmo?

La idea central de SVM es encontrar una línea (en 2D), un plano (en 3D) o un **hiperplano** (en más dimensiones) que divida los datos en clases distintas. Pero no cualquier separación sirve; SVM busca la separación que tenga el **mayor margen** posible.

1. **Hiperplano:** Es la frontera de decisión que separa las clases.
2. **Vectores de Soporte:** Son los puntos de datos más cercanos al hiperplano. Estos puntos son los más "difíciles" de clasificar y son los únicos que importan para definir la posición del hiperplano.
3. **Margen:** Es la distancia entre el hiperplano y los vectores de soporte más cercanos de cada clase. SVM intenta **maximizar** este margen para mejorar la generalización del modelo.

---

### 7.2. Explicación Matemática y el "Kernel Trick"

Matemáticamente, para un problema linealmente separable, buscamos los parámetros $w$ (vector de pesos) y $b$ (sesgo) tal que el hiperplano se defina como:
$$w \cdot x + b = 0$$

El objetivo es minimizar $||w||$ (lo que equivale a maximizar el margen) sujeto a que todas las muestras estén correctamente clasificadas fuera del margen.

#### El Truco del Kernel (Kernel Trick)

Cuando los datos no son separables linealmente (ej. un círculo dentro de otro), SVM utiliza una técnica llamada **Kernel Trick**.
Esta técnica proyecta los datos originales a un espacio de **mayor dimensión** donde sí son linealmente separables, sin necesidad de calcular explícitamente las coordenadas en ese espacio complejo (lo cual sería computacionalmente costoso).

Kernels comunes:

* **Lineal:** Para datos linealmente separables.
* **Polinómico:** Mapea a espacios de dimensiones polinómicas.
* **RBF (Radial Basis Function):** El más popular. Mapea a un espacio de dimensión infinita. Es muy efectivo para fronteras de decisión complejas y curvas.

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
print("\nReporte de Clasificación:")
print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))

# Comparar diferentes kernels
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
C_values = [0.1, 1, 10, 100]
results = {}

print("\n" + "="*60)
print("COMPARACIÓN DE KERNELS")
print("="*60)

for kernel in kernels:
    scores = []
    for C in C_values:
        svm_model = svm.SVC(kernel=kernel, C=C, gamma='scale')
        cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
        scores.append(cv_scores.mean())
    results[kernel] = scores
    print(f"\n{kernel.upper()} Kernel:")
    for C, score in zip(C_values, scores):
        print(f"  C={C:5.1f}: Accuracy={score:.4f}")

# Visualizar comparación
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico 1: Comparación de kernels
for kernel, scores in results.items():
    axes[0].plot(C_values, scores, marker='o', label=kernel, linewidth=2)
axes[0].set_xlabel('Valor de C (Regularización)')
axes[0].set_ylabel('Accuracy (Cross-Validation)')
axes[0].set_title('Comparación de Kernels SVM')
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Fronteras de decisión (2D)
from matplotlib.colors import ListedColormap

# Usar solo 2 características para visualización
X_2d = iris.data[:, [2, 3]]  # petal length y width
y_2d = iris.target
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42)

# Escalar
scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

# Entrenar SVM RBF
svm_rbf = svm.SVC(kernel='rbf', C=10, gamma='scale')
svm_rbf.fit(X_train_2d_scaled, y_train_2d)

# Crear malla
h = 0.02
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
axes[1].contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
axes[1].scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], c=y_train_2d,
                cmap=cmap_bold, edgecolor='k', s=50, alpha=0.7)

# Marcar vectores de soporte
support_vectors = svm_rbf.support_vectors_
axes[1].scatter(support_vectors[:, 0], support_vectors[:, 1], 
                s=200, linewidth=2, facecolors='none', edgecolors='black',
                label='Vectores de Soporte')

axes[1].set_xlabel('Petal Length (scaled)')
axes[1].set_ylabel('Petal Width (scaled)')
axes[1].set_title('SVM con Kernel RBF - Fronteras de Decisión')
axes[1].legend()
plt.tight_layout()
plt.show()

print(f"\nNúmero de vectores de soporte: {len(svm_rbf.support_vectors_)}")
print(f"Accuracy en test (2D): {svm_rbf.score(X_test_2d_scaled, y_test_2d):.4f}")
```

---

### 7.5. Ejemplos Comunes de Uso

* **Clasificación de Texto:** Categorización de noticias, detección de spam y análisis de sentimientos. SVM maneja muy bien la alta dimensionalidad de los vectores de texto (Bag of Words).
* **Reconocimiento de Imágenes:** Clasificación de imágenes, reconocimiento facial y reconocimiento de escritura a mano (OCR).
* **Bioinformática:** Clasificación de proteínas y genes, donde los datos suelen tener muchas características y pocas muestras.
* **Detección de Intrusos:** Identificar actividad maliciosa en redes basándose en patrones de tráfico.

### 7.6. Aplicaciones Reales de SVM

SVM ha sido uno de los algoritmos más exitosos antes del auge del Deep Learning y sigue siendo muy relevante:

* **Clasificación de Imágenes (Histórico):** Antes de las redes neuronales convolucionales (CNN), SVM era el estándar para clasificación de imágenes y detección de objetos (ej. detección de peatones).
* **Bioinformática (Clasificación de Proteínas):** Se utiliza para clasificar proteínas en familias funcionales y predecir la estructura secundaria de las proteínas, dado que maneja muy bien la alta dimensionalidad de los datos genómicos.
  * [SVM en Bioinformática](https://github.com/topics/bioinformatics-machine-learning)
* **Reconocimiento de Escritura:** SVM ha demostrado ser muy eficaz en el reconocimiento de caracteres manuscritos (OCR), compitiendo con redes neuronales en datasets como MNIST.
* **Geología y Minería:** Clasificación de tipos de suelo y rocas a partir de datos sísmicos o imágenes satelitales.

---

### 7.7. Consideraciones Finales

1. **Parámetro C (Regularización):**
    * Controla el equilibrio entre tener un margen amplio y clasificar correctamente los puntos de entrenamiento.
    * **C alto:** Intenta clasificar todo correctamente (riesgo de Overfitting, margen estrecho).
    * **C bajo:** Permite algunos errores para obtener un margen más amplio (mejor generalización, margen suave).

2. **Parámetro Gamma ($\gamma$) (Solo para kernels RBF/Poly):**
    * Define qué tan lejos llega la influencia de un solo ejemplo de entrenamiento.
    * **Gamma alto:** Solo los puntos muy cercanos influyen. Puede llevar a fronteras de decisión muy ajustadas e irregulares (Overfitting).
    * **Gamma bajo:** La influencia llega lejos. La frontera de decisión es más suave (Underfitting si es muy bajo).

3. **Escalado:** Al igual que KNN, SVM se basa en distancias. Es **crítico** estandarizar los datos antes de entrenar.

---

📅 **Fecha de creación:** 19/11/2025
✍️ **Autor:** Fran García
