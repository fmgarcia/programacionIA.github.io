
# 🤖 Unidad 1. Machine Learning Basado en el Análisis de Datos

Esta unidad introduce los conceptos fundamentales del Machine Learning (ML), su flujo de trabajo, las herramientas clave de la biblioteca `scikit-learn` del lenguaje Python, y las metodologías esenciales para la preparación, división y preprocesamiento de datos.

---

### 1.1. ¿Qué es el Machine Learning?

El Machine Learning (ML) se define como un campo de estudio que utiliza modelos estadísticos para aprender de los datos. Un aspecto clave es que modelos relativamente simples pueden realizar predicciones complejas.

#### Definiciones Clave

* **Definición temprana (Samuel, 1959):** "Programar computadoras para que aprendan de la experiencia debería eliminar la necesidad de gran parte de este esfuerzo de programación detallado".
* **Definición moderna (Mitchell, 1997):** "Se dice que un programa de computadora aprende de la **experiencia E** con respecto a alguna clase de **tareas T** y una medida de **rendimiento P**, si su rendimiento en las tareas T, medido por P, mejora con la experiencia E".
* **Definición matemática (Ej. Regresión Lineal):** Un modelo matemático que intenta encontrar la relación óptima entre variables. Por ejemplo, predecir ventas (Target, $y$) basándose en gastos de publicidad (Feature, $x$). El modelo $y = wx + b$ aprende los **parámetros** $w$ (peso) y $b$ iterando desde valores arbitrarios ($f_1$) hasta un valor óptimo ($f_3$) que minimiza el error.

#### ML y Otros Campos
El Machine Learning está profundamente interconectado con otros campos:
* Es un subcampo de la **Inteligencia Artificial**.
* **Deep Learning** es un subcampo del Machine Learning.
* Se solapa significativamente con **Estadística**, **Minería de Datos** y **Reconocimiento de Patrones**.

#### Tipos de Machine Learning
Según el método de supervisión, el ML se divide en:
1.  **Supervisado:** Se proporciona un patrón objetivo (datos etiquetados). El capítulo se centra en este tipo, que incluye algoritmos como Regresión Lineal, Regresión Logística, Árboles de Decisión, KNN, SVM y Redes Neuronales.
2.  **No Supervisado:** El patrón objetivo debe ser descubierto (datos no etiquetados). Incluye Clustering, PCA y Análisis de Asociación.
3.  **Refuerzo:** Se aprende mediante la optimización de políticas (recompensas y castigos).

#### Flujo de Trabajo del Machine Learning
El proceso general para construir un modelo de ML es:
1.  **Definición del Problema:** Comprender el objetivo de negocio.
2.  **Preparación de Datos:** Recolección de datos brutos (Raw Data) y preprocesamiento.
3.  **Machine Learning (Modelado):** Se divide la data en conjuntos de **Train** (Entrenamiento), **Validate** (Validación) y **Test** (Prueba).
4.  **Entrenamiento y Evaluación:** Esta fase incluye **Ingeniería de características** (Feature engineering), **Modelado y optimización** (entrenar el modelo con los datos), y **Evaluación de rendimiento** (Performance metrics).
5.  **Aplicación:** Aplicar el modelo en la vida real.

#### Parámetros vs. Hiperparámetros
* **Parámetros:** Se aprenden *desde los datos* durante el entrenamiento. Contienen el patrón de los datos (ej. $w$ y $b$ en regresión lineal, pesos de una red neuronal).
* **Hiperparámetros:** Se configuran *manualmente* por el practicante *antes* del entrenamiento. Se "afinan" (tunan) para optimizar el rendimiento (ej. el valor $k$ en KNN, la tasa de aprendizaje, la profundidad máxima de un árbol).

---

### 1.2. Biblioteca Python scikit-learn

`scikit-learn` es la biblioteca de ML más representativa de Python.

#### Características
* Proporciona una interfaz de biblioteca integrada y unificada.
* Incluye una amplia variedad de algoritmos de ML, funciones de preprocesamiento y selección de modelos.
* Es simple, eficiente y está construida sobre **NumPy, SciPy y matplotlib**.
* Es de código abierto y puede usarse comercialmente.
* **No soporta GPU**.

#### Mecanismo de `scikit-learn`
El flujo de trabajo de la API de `scikit-learn` es intuitivo y sigue tres pasos:
1.  **Instance:** Crear una instancia del objeto del modelo (Estimator).
2.  **Fit:** Entrenar el modelo con los datos.
3.  **Predict / transform:** Usar el modelo entrenado para hacer predicciones o transformar datos.



#### Estimator, Classifier y Regressor
* **`Estimator`:** El objeto base. Aprende de los datos usando el método `.fit()` y puede hacer predicciones usando `.predict()`.
* **`Classifier`:** Un estimador para tareas de clasificación (ej. `DecisionTreeClassifier`, `KNeighborsClassifier`).
* **`Regressor`:** Un estimador para tareas de regresión (predicción numérica) (ej. `LinearRegression`, `KNeighborsRegressor`).

#### Sintaxis Básica de `scikit-learn`
* **Importar un estimador:**
    `from sklearn.linear_model import LinearRegression`
* **Importar un preprocesador:**
    `from sklearn.preprocessing import StandardScaler`
* **Importar división de datos:**
    `from sklearn.model_selection import train_test_split`
* **Importar métricas:**
    `from sklearn import metrics`

* **Instanciar (con hiperparámetros):**
    `myModel = KNeighborsClassifier(n_neighbors=10)`
* **Dividir los datos (Hold-out):**
    `X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)`
* **Entrenar el modelo (Supervisado):**
    `myModel.fit(X_train, Y_train)`
* **Hacer predicciones:**
    `Y_pred = myModel.predict(X_test)`
* **Evaluar el rendimiento:**
    `metrics.accuracy_score(Y_test, Y_pred)`
* **Afinar hiperparámetros (con Cross-Validation):**
    `myGridCV = GridSearchCV(estimator, parameter_grid, cv=5)`

#### Ejemplo Práctico: Estandarización
El preprocesamiento, como la **estandarización**, es crucial para mejorar el rendimiento. La estandarización (o *z-transformation*) convierte los datos para que sigan una distribución normal estándar, usando la fórmula $z = \frac{x - m}{\sigma}$ (donde $m$ es la media y $\sigma$ la desviación estándar).

En `scikit-learn`, se usa `StandardScaler`:
1.  **Importar:** `from sklearn.preprocessing import StandardScaler`
2.  **Instanciar:** `scaler = StandardScaler()`
3.  **Ajustar (Fit):** Se aprende la media ($m$) y la desviación ($\sigma$) **solo de los datos de entrenamiento**:
    `scaler.fit(X_train)`
4.  **Transformar:** Se aplica la transformación a los datos de entrenamiento y prueba:
    `X_train = scaler.transform(X_train)`
    `X_test = scaler.transform(X_test)`
5.  **`fit_transform`:** Se pueden combinar los pasos 3 y 4 (solo para `X_train`):
    `X_train = scaler.fit_transform(X_train)`

Antes de la estandarización, las columnas pueden tener rangos de valores muy diferentes. Después, todos los valores están centrados alrededor de 0, lo que ayuda a muchos algoritmos a converger mejor.

#### Módulos Principales de `scikit-learn`

| Módulo | Función Principal | Ejemplos |
| :--- | :--- | :--- |
| `sklearn.datasets` | Cargar datasets de ejemplo. | `load_iris()`, `load_breast_cancer()` |
| `sklearn.preprocessing` | Preprocesamiento de datos (escalado, codificación). | `StandardScaler`, `LabelEncoder`, `OneHotEncoder` |
| `sklearn.model_selection` | División de datos, validación y afinado de hiperparámetros. | `train_test_split`, `GridSearchCV`, `KFold` |
| `sklearn.metrics` | Evaluación de rendimiento del modelo. | `accuracy_score`, `precision_score`, `recall_score`, `roc_auc_score` |
| `sklearn.linear_model` | Algoritmos lineales. | `LinearRegression`, `LogisticRegression` |
| `sklearn.tree` | Algoritmos de Árboles de Decisión. | `DecisionTreeClassifier` |
| `sklearn.neighbors` | Algoritmos de vecinos cercanos. | `KNeighborsClassifier` (K-NN) |
| `sklearn.svm` | Support Vector Machine (Máquinas de Vectores de Soporte). | `SVC` |
| `sklearn.ensemble` | Algoritmos de Ensamblado (Ensemble). | `RandomForestClassifier`, `AdaBoostClassifier` |
| `sklearn.cluster` | Algoritmos de clustering (No supervisado). | `KMeans`, `DBSCAN` |
| `sklearn.pipeline` | Herramienta para encadenar pasos de preprocesamiento y modelado. | `Pipeline` |

---

### 1.3. Preparación y División del Dataset

La división de datos es fundamental para evaluar un modelo de ML. El conjunto de datos general se divide en un conjunto de **entrenamiento** y uno de **evaluación (prueba)**.

#### Overfitting (Sobreajuste) y Generalización
* **Generalización:** Es la capacidad del modelo para predecir con precisión datos nuevos que no ha visto antes.
* **Overfitting:** Ocurre cuando un modelo se ajusta *demasiado* a los datos de entrenamiento, aprendiendo incluso el ruido.
* **Underfitting (Subajuste):** Ocurre cuando un modelo es *demasiado simple* (baja capacidad) y no puede capturar el patrón subyacente de los datos.



> **El Dilema:** A medida que aumenta la complejidad (flexibilidad) del modelo:
> * El error en el **conjunto de entrenamiento** (Training set) siempre disminuye.
> * El error en el **conjunto de prueba** (Test set) disminuye al principio, pero luego comienza a *aumentar*. El punto donde el error de prueba empieza a subir es donde comienza el overfitting.

El conjunto de prueba es **esencial** para detectar el overfitting y seleccionar un modelo que generalice bien.

#### Cross-Validation (Validación Cruzada)
El conjunto de prueba (Test set) debe usarse **¡solo una vez!** al final, para la evaluación final.

Para evaluar el modelo *durante* el entrenamiento (por ejemplo, para afinar hiperparámetros), necesitamos una forma de simular un "conjunto de prueba" sin tocar el real. Para esto, dividimos el **conjunto de entrenamiento** (Training Data) en dos partes más pequeñas: un nuevo conjunto de `Train` y un conjunto de `Cross Validate` (Validación).

**Método: k-Fold Cross-Validation**
Es el método más común:
1.  Se subdivide el conjunto de entrenamiento (original) en *k* partes iguales (folds). (Usualmente k=10).
2.  Se itera *k* veces (rondas).
3.  En cada ronda, se usa 1 fold como conjunto de **validación** y los *k-1* folds restantes como conjunto de **entrenamiento**.
4.  Se calcula la métrica de rendimiento (ej. accuracy) en cada ronda.
5.  El rendimiento final del modelo es el **promedio** de las métricas de las *k* rondas.



**Método: Leave One Out (LOO)**
Es un caso extremo de k-Fold donde $k$ es igual al número total de muestras. Se entrena con todos los datos menos uno, y se valida con ese único dato. Es computacionalmente muy costoso.

---

### 1.4. Preprocesamiento de Datos

Preparar los datos es vital para un buen modelo. Esto incluye la limpieza (manejo de valores atípicos y faltantes) y la transformación (escalado y codificación).

#### Manejo de Valores Faltantes (Missing Values)
Los valores faltantes (identificados en Python como `np.nan` o `NaN`) deben ser tratados.

**1. Identificación:**
Se pueden contar usando `df.isnull().sum()`.

**2. Eliminación (con Pandas `dropna()`):**
* `df.dropna()`: Elimina cualquier **fila** que contenga al menos un `NaN` (eje por defecto 0).
* `df.dropna(axis=1)`: Elimina cualquier **columna** que contenga un `NaN`.
* `df.dropna(how='all')`: Elimina filas/columnas donde **todos** los valores son `NaN`.
* `df.dropna(thresh=N)`: Mantiene las filas que tienen al menos `N` valores no-`NaN`.
* `df.dropna(subset=['col_name'])`: Elimina filas que tienen `NaN` específicamente en la columna 'col_name'.

**3. Imputación (Relleno):**
Se usa cuando eliminar datos resultaría en una pérdida significativa de información.
* **Métodos simples:** Rellenar con un valor (ej. 'unknown'), la media, la mediana o el valor más frecuente (moda) de la columna.
* **Con `scikit-learn` (`SimpleImputer`):** Es el método preferido.
    * `from sklearn.impute import SimpleImputer`
    * `impt = SimpleImputer(strategy='mean')` (Estrategias: 'mean', 'median', 'most_frequent').
    * `impt.fit(X_train)`: Aprende la media (o mediana/moda) del set de entrenamiento.
    * `X_train_imputed = impt.transform(X_train)`: Aplica la imputación.
    * `X_test_imputed = impt.transform(X_test)`: Aplica la *misma* imputación (con la media de train) al set de prueba.

#### Manejo de Datos Categóricos
Los algoritmos de ML requieren entradas numéricas. Los datos categóricos deben ser convertidos.

**1. Datos Ordinales (con orden):**
Ej. Tallas: 'M' < 'L' < 'XL'.
Se deben mapear a enteros que respeten ese orden.
* `size_mapping = {'M': 1, 'L': 2, 'XL': 3}`
* `df['size'] = df['size'].map(size_mapping)`

**2. Datos Nominales (sin orden) y Etiquetas de Clase:**
Ej. Colores: 'red', 'green', 'blue' o Etiquetas: 'setosa', 'versicolor'.

* **Codificación de Etiquetas (Label Encoding):**
    Convierte cada etiqueta única en un entero (ej. 'class1': 0, 'class2': 1). Se usa `LabelEncoder` de `scikit-learn`.
    * `from sklearn.preprocessing import LabelEncoder`
    * `enc = LabelEncoder()`
    * `y_encoded = enc.fit_transform(df['classlabel'])`

* **One-Hot Encoding (para características nominales):**
    Usar `LabelEncoder` para características (X) es incorrecto, ya que crea un orden artificial. Se debe usar **One-Hot Encoding**.
    Crea nuevas columnas "dummy" (0 o 1) para cada categoría, indicando presencia (1) o ausencia (0).
    * **Método Pandas:** `pd.get_dummies(df['species'])`.
    * **Método `scikit-learn`:** `OneHotEncoder`. Este método es preferido en pipelines y a menudo devuelve una **matriz dispersa** (sparse matrix) para ahorrar memoria, ya que la mayoría de los valores serán 0.

#### División de Datos Estratificada (Stratify)
Al usar `train_test_split`, si el dataset está desbalanceado (ej. 90% clase A, 10% clase B), una división aleatoria simple podría resultar en un set de prueba sin muestras de la clase B.
* **Solución:** Usar el parámetro `stratify=y`.
* Esto asegura que la proporción de las clases (ej. 90/10) se mantenga *idéntica* tanto en el conjunto de entrenamiento como en el de prueba, reflejando el dataset original.

#### Tópicos Avanzados: Tradeoff de Sesgo-Varianza y Regularización
* **Tradeoff de Sesgo-Varianza:**
    * **Sesgo (Bias):** Error por suposiciones incorrectas (Underfitting).
    * **Varianza (Variance):** Error por sensibilidad excesiva a los datos de entrenamiento (Overfitting).
    * **Error Total $\approx$ Sesgo² + Varianza**. El objetivo es encontrar la complejidad óptima que minimice este error total.
* **Regularización:** Técnica para prevenir el overfitting en modelos lineales penalizando coeficientes (pesos) grandes.
    * **Ridge (L2):** Añade una penalización $\lambda\sum{w_j^2}$. Encoge los pesos, pero no los hace cero.
    * **Lasso (L1):** Añade una penalización $\lambda\sum{|w_j|}$. Puede forzar que algunos pesos sean exactamente cero, realizando una selección de características automática.
    * **ElasticNet:** Combina penalizaciones L1 y L2.

---

### 1.5. Práctica: Solución de Problemas con scikit-learn (Ej. Iris)

Esta sección aplica todos los conceptos anteriores en un caso práctico completo usando el dataset "Iris".

#### 1. Entendimiento del Problema y Datos (EDA)
* **Objetivo:** Clasificar la especie de una flor Iris (Target).
* **Clases (Target):** 3 especies (Setosa, Versicolor, Virginica).
* **Características (Features):** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
* **Análisis de Datos:**
    * Se cargan los datos y se convierten a un DataFrame de Pandas.
    * **Valores Faltantes:** Se comprueba con `iris.isnull().sum()`. No se encontraron.
    * **Distribución de Clases:** Se comprueba con `iris.groupby('target').size()`. Hay 50 muestras de cada clase (33.3% cada una). Es un **dataset balanceado**.
    * **Estadísticas y Correlación:** `iris.describe()` y `iris.corr()`. Se observa que `petal_length` y `petal_width` están altamente correlacionados (0.96), sugiriendo un problema de multicolinealidad.
    * **Visualización:** Se usan `pairplot` y `heatmap` para confirmar visualmente las relaciones y la alta correlación.

#### 2. División y Preparación de Datos
* **Separación X/y:** Se separan las características (X) del objetivo (y).
* **División Train/Test:** Se usa `train_test_split` (ej. 80% train, 20% test).
* **Validación Cruzada:**
    * Se muestra cómo usar `KFold` (CV estándar) y `StratifiedKFold` (CV estratificada).
    * `StratifiedKFold` es preferible porque mantiene la distribución 33/33/33 de las clases en cada fold, asegurando que la validación sea representativa.

#### 3. Selección y Evaluación del Modelo
* **Curva de Aprendizaje (`Learning Curve`):**
    Se usa para diagnosticar bias vs. variance. Muestra el rendimiento del modelo a medida que ve más datos de entrenamiento.
* **Afinado de Hiperparámetros (`GridSearchCV`):**
    Se utiliza para encontrar la mejor combinación de hiperparámetros (ej. `criterion`, `max_depth` para un `DecisionTreeClassifier`) probando todas las combinaciones posibles mediante validación cruzada.

#### 4. Métricas de Evaluación (Clasificación)
Una vez que el modelo (`GridSearchCV`) está entrenado y se hacen predicciones sobre el `X_test`, se evalúa el rendimiento.

* **Matriz de Confusión (`Confusion Matrix`):**
    Es la base para todas las métricas. Compara los valores reales (True label) con los predichos (Predicted label).
    * **TP (True Positive):** Real = 1, Predicho = 1.
    * **FN (False Negative):** Real = 1, Predicho = 0.
    * **FP (False Positive):** Real = 0, Predicho = 1.
    * **TN (True Negative):** Real = 0, Predicho = 0.

* **Métricas Clave:**
    * **Accuracy (Exactitud):** $\frac{TP + TN}{Total}$. Proporción de predicciones correctas. (Usar con cuidado en datasets desbalanceados).
    * **Precision (Precisión):** $\frac{TP}{TP + FP}$. De los que *dijimos* que eran positivos, ¿cuántos acertamos?.
    * **Recall (Sensibilidad o TPR):** $\frac{TP}{TP + FN}$. De *todos los positivos reales*, ¿cuántos encontramos?.
    * **F1-Score:** La media armónica de Precision y Recall. Es una métrica excelente para datasets desbalanceados. $F_1 = 2 \frac{Precision \times Recall}{Precision + Recall}$.
    * **FPR (Tasa de Falsos Positivos):** $\frac{FP}{FP + TN}$. Proporción de negativos reales que clasificamos incorrectamente como positivos.

* **Curva ROC y AUC:**
    * **Curva ROC:** Gráfica que muestra el rendimiento de un clasificador en todos los umbrales de clasificación. Muestra **TPR** (Eje Y) vs. **FPR** (Eje X).
    * **AUC (Area Under the Curve):** El área bajo la curva ROC. Es una métrica única que resume el rendimiento del modelo.
        * AUC = 1.0: Clasificador perfecto.
        * AUC = 0.5: Clasificador inútil (aleatorio).
        * Un AUC de 0.85 o más se considera bueno.

#### 5. Predicción Final
* Se carga el modelo final (el mejor `estimator_` encontrado por `GridSearchCV`).
* Se realizan las predicciones finales sobre el conjunto de prueba (`X_test`).
* Los resultados se guardan, por ejemplo, en un archivo CSV.

---

📅 **Fecha de creación:** 27/10/2025  
✍️ **Autor:** Fran García

