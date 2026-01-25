# 🤖 Scikit-learn - Introducción al Machine Learning

**Scikit-learn** es la librería más popular de Python para Machine Learning. Proporciona herramientas simples y eficientes para análisis predictivo y modelado estadístico.

---

## 1. Instalación e Importación

```python
# Instalación
# pip install scikit-learn

# Importación (convención estándar)
import sklearn

# Importaciones específicas comunes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
```

---

## 2. Conceptos Fundamentales

### Tipos de Aprendizaje

| Tipo | Descripción | Ejemplos |
| :--- | :--- | :--- |
| **Supervisado** | Datos etiquetados (con respuesta) | Clasificación, Regresión |
| **No supervisado** | Sin etiquetas | Clustering, Reducción dimensionalidad |

### Términos Importantes

```python
# X = Features (características/variables independientes)
# y = Target (objetivo/variable dependiente)
# Modelo = Algoritmo que aprende patrones de los datos
# Entrenamiento = Proceso de ajuste del modelo
# Predicción = Uso del modelo para nuevos datos
```

---

## 3. Flujo de Trabajo Básico

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. CARGAR DATOS
# X = características, y = objetivo
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# 2. DIVIDIR DATOS (entrenamiento y prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. PREPROCESAR (opcional pero recomendado)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. CREAR Y ENTRENAR MODELO
modelo = LogisticRegression()
modelo.fit(X_train_scaled, y_train)

# 5. HACER PREDICCIONES
y_pred = modelo.predict(X_test_scaled)

# 6. EVALUAR
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy:.2%}")  # Precisión: 100.00%
```

---

## 4. Datasets de Ejemplo

```python
from sklearn import datasets

# Clasificación
iris = datasets.load_iris()           # 3 clases de flores
digits = datasets.load_digits()       # Dígitos escritos
wine = datasets.load_wine()           # Tipos de vino
breast_cancer = datasets.load_breast_cancer()  # Cáncer de mama

# Regresión
boston = datasets.load_boston()       # Precios de casas (deprecated)
diabetes = datasets.load_diabetes()   # Progresión diabetes

# Información del dataset
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 
#            'DESCR', 'feature_names', 'filename'])

print(iris.DESCR[:500])  # Descripción
print(iris.feature_names)  # Nombres de características
print(iris.target_names)   # Nombres de clases

# Crear DataFrame para explorar
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```

### Generar Datos Sintéticos

```python
from sklearn.datasets import make_classification, make_regression, make_blobs

# Para clasificación
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    random_state=42
)

# Para regresión
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=10,
    random_state=42
)

# Para clustering
X, y = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)
```

---

## 5. División de Datos

### train_test_split

```python
from sklearn.model_selection import train_test_split

X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 0, 1, 1, 1]

# División básica (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Entrenamiento: {len(X_train)} muestras")
print(f"Prueba: {len(X_test)} muestras")

# Parámetros importantes:
# test_size: proporción para test (0.2 = 20%)
# random_state: semilla para reproducibilidad
# stratify: mantener proporción de clases
# shuffle: mezclar datos antes de dividir

# Con estratificación (importante para clasificación desbalanceada)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### División en 3 partes

```python
from sklearn.model_selection import train_test_split

# Primero separar test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Luego separar validación
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 de 0.8 = 0.2
)

# Resultado: 60% train, 20% val, 20% test
```

---

## 6. Preprocesamiento de Datos

### Escalado de Features

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

X = np.array([[1, 10, 100],
              [2, 20, 200],
              [3, 30, 300]])

# StandardScaler: media=0, desviación=1 (más común)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("StandardScaler:")
print(X_scaled)

# MinMaxScaler: valores entre 0 y 1
scaler_mm = MinMaxScaler()
X_minmax = scaler_mm.fit_transform(X)
print("\nMinMaxScaler:")
print(X_minmax)

# RobustScaler: resistente a outliers
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
```

### ⚠️ Importante: fit vs transform

```python
from sklearn.preprocessing import StandardScaler

# CORRECTO: fit solo en train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled = scaler.transform(X_test)        # solo transform

# INCORRECTO: NO hacer fit en test
# X_test_scaled = scaler.fit_transform(X_test)  # ❌ Fuga de datos
```

### Codificación de Variables Categóricas

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# LabelEncoder: convierte categorías a números
le = LabelEncoder()
colores = ["rojo", "verde", "azul", "rojo", "azul"]
colores_encoded = le.fit_transform(colores)
print(colores_encoded)  # [2 1 0 2 0]

# Inverso
print(le.inverse_transform([0, 1, 2]))  # ['azul' 'verde' 'rojo']

# OneHotEncoder: crea columnas binarias
ohe = OneHotEncoder(sparse_output=False)
colores_array = np.array(colores).reshape(-1, 1)
colores_onehot = ohe.fit_transform(colores_array)
print(colores_onehot)
# [[0. 0. 1.]   rojo
#  [0. 1. 0.]   verde
#  [1. 0. 0.]   azul
#  [0. 0. 1.]   rojo
#  [1. 0. 0.]]  azul
```

### Manejo de Valores Faltantes

```python
from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[1, 2, np.nan],
              [3, np.nan, 6],
              [7, 8, 9],
              [np.nan, 5, 3]])

# Imputar con la media
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
print(X_imputed)

# Estrategias disponibles:
# "mean": media (numéricos)
# "median": mediana (numéricos)
# "most_frequent": moda (categóricos)
# "constant": valor fijo (fill_value=0)

# Con valor constante
imputer_const = SimpleImputer(strategy="constant", fill_value=0)
```

---

## 7. Modelos de Clasificación

### Regresión Logística

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cargar datos
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Entrenar modelo
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)

# Predecir
y_pred = modelo.predict(X_test)

# Probabilidades
y_proba = modelo.predict_proba(X_test)
print("Probabilidades:", y_proba[0])

# Evaluar
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

# Crear modelo
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar
knn.fit(X_train, y_train)

# Predecir
y_pred = knn.predict(X_test)
print(f"Accuracy KNN: {accuracy_score(y_test, y_pred):.2%}")

# Probar diferentes valores de k
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print(f"k={k}: {score:.2%}")
```

### Árbol de Decisión

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Crear modelo
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Predecir
y_pred = tree.predict(X_test)
print(f"Accuracy Tree: {accuracy_score(y_test, y_pred):.2%}")

# Visualizar árbol
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True)
plt.show()

# Importancia de features
importances = tree.feature_importances_
for name, importance in zip(iris.feature_names, importances):
    print(f"{name}: {importance:.4f}")
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Crear modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predecir
y_pred = rf.predict(X_test)
print(f"Accuracy RF: {accuracy_score(y_test, y_pred):.2%}")

# Importancia de features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), 
           [iris.feature_names[i] for i in indices], rotation=45)
plt.title("Importancia de Features")
plt.show()
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM requiere escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear modelo
svm = SVC(kernel="rbf", C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

# Predecir
y_pred = svm.predict(X_test_scaled)
print(f"Accuracy SVM: {accuracy_score(y_test, y_pred):.2%}")

# Con probabilidades
svm_proba = SVC(kernel="rbf", probability=True)
svm_proba.fit(X_train_scaled, y_train)
y_proba = svm_proba.predict_proba(X_test_scaled)
```

---

## 8. Modelos de Regresión

### Regresión Lineal

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Cargar datos
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Entrenar modelo
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predecir
y_pred = lr.predict(X_test)

# Evaluar
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Coeficientes
print("\nCoeficientes:")
for name, coef in zip(diabetes.feature_names, lr.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercepto: {lr.intercept_:.4f}")
```

### Ridge y Lasso (Regularización)

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge (regularización L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"R² Ridge: {ridge.score(X_test, y_test):.4f}")

# Lasso (regularización L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"R² Lasso: {lasso.score(X_test, y_test):.4f}")

# Lasso hace selección de features (algunos coeficientes = 0)
print(f"Features usadas: {np.sum(lasso.coef_ != 0)}")
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

y_pred = rfr.predict(X_test)
print(f"R² RF: {rfr.score(X_test, y_test):.4f}")
```

---

## 9. Métricas de Evaluación

### Clasificación

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Datos de ejemplo
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]

# Métricas básicas
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
print(f"Precision: {precision_score(y_true, y_pred):.2%}")
print(f"Recall: {recall_score(y_true, y_pred):.2%}")
print(f"F1-Score: {f1_score(y_true, y_pred):.2%}")

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusión:")
print(cm)
#          Pred 0  Pred 1
# Real 0    [TN      FP]
# Real 1    [FN      TP]

# Reporte completo
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
```

### Visualizar Matriz de Confusión

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

# Opción 1: Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# Opción 2: sklearn
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                        display_labels=iris.target_names)
plt.show()
```

### Regresión

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

y_true = [100, 150, 200, 250, 300]
y_pred = [110, 140, 190, 260, 310]

# Métricas
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2%}")
```

### Curva ROC y AUC

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# Necesitas probabilidades
y_proba = modelo.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva

# Calcular ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Graficar
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend()
plt.show()
```

---

## 10. Validación Cruzada

```python
from sklearn.model_selection import cross_val_score, KFold

# Validación cruzada simple
modelo = LogisticRegression(max_iter=200)
scores = cross_val_score(modelo, X, y, cv=5)  # 5 folds

print(f"Scores: {scores}")
print(f"Media: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# KFold personalizado
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=kfold)

# Stratified para clasificación (mantiene proporciones)
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=skfold)
```

---

## 11. Búsqueda de Hiperparámetros

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Definir parámetros a buscar
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Crear búsqueda
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # Usar todos los cores
    verbose=1
)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

# Mejores parámetros
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score: {grid_search.best_score_:.4f}")

# Usar mejor modelo
mejor_modelo = grid_search.best_estimator_
y_pred = mejor_modelo.predict(X_test)
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Distribuciones de parámetros
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Búsqueda aleatoria
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Número de combinaciones a probar
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Mejores parámetros: {random_search.best_params_}")
```

---

## 12. Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Crear pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Usar como modelo normal
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# Grid Search con Pipeline
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'linear']
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

### Pipeline con múltiples pasos

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Pipeline para diferentes tipos de columnas
numeric_features = [0, 1, 2, 3]  # índices de columnas numéricas
categorical_features = [4, 5]     # índices de columnas categóricas

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline completo
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Usar
clf.fit(X_train, y_train)
```

---

## 13. Guardar y Cargar Modelos

```python
import joblib
import pickle

# Opción 1: joblib (recomendado para sklearn)
# Guardar
joblib.dump(modelo, 'modelo.joblib')

# Cargar
modelo_cargado = joblib.load('modelo.joblib')
y_pred = modelo_cargado.predict(X_test)

# Opción 2: pickle
with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)

with open('modelo.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

# Guardar pipeline completo
joblib.dump(pipe, 'pipeline_completo.joblib')
```

---

## 14. Ejemplo Completo: Clasificación

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. CARGAR DATOS
data = load_breast_cancer()
X = data.data
y = data.target
print(f"Forma de X: {X.shape}")
print(f"Clases: {data.target_names}")

# 2. EXPLORAR DATOS
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
print(df.describe())

# 3. DIVIDIR DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. ESCALAR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. COMPARAR MODELOS
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

resultados = {}
for nombre, modelo in modelos.items():
    # Validación cruzada
    scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)
    
    # Entrenar y evaluar
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    resultados[nombre] = {
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'test_acc': acc
    }
    
    print(f"\n{nombre}:")
    print(f"  CV: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    print(f"  Test: {acc:.4f}")

# 6. EVALUAR MEJOR MODELO
mejor = max(resultados, key=lambda x: resultados[x]['test_acc'])
print(f"\nMejor modelo: {mejor}")

# Reentrenar
mejor_modelo = modelos[mejor]
y_pred = mejor_modelo.predict(X_test_scaled)

# Reporte
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title(f'Matriz de Confusión - {mejor}')
plt.show()

# 7. GUARDAR MODELO
import joblib
joblib.dump(mejor_modelo, 'mejor_modelo_cancer.joblib')
joblib.dump(scaler, 'scaler_cancer.joblib')
print("\nModelo guardado!")
```

---

## 15. Resumen de Módulos Principales

| Módulo | Descripción |
| :--- | :--- |
| `sklearn.model_selection` | División de datos, validación cruzada, búsqueda de hiperparámetros |
| `sklearn.preprocessing` | Escalado, codificación, transformaciones |
| `sklearn.impute` | Manejo de valores faltantes |
| `sklearn.linear_model` | Modelos lineales (regresión, logística) |
| `sklearn.tree` | Árboles de decisión |
| `sklearn.ensemble` | Random Forest, Gradient Boosting |
| `sklearn.svm` | Support Vector Machines |
| `sklearn.neighbors` | K-Nearest Neighbors |
| `sklearn.cluster` | Algoritmos de clustering |
| `sklearn.metrics` | Métricas de evaluación |
| `sklearn.pipeline` | Pipelines de preprocesamiento + modelo |
| `sklearn.datasets` | Datasets de ejemplo |

---

## 16. Buenas Prácticas

```python
# ✅ HACER:
# 1. Siempre dividir datos ANTES de cualquier preprocesamiento
# 2. fit() solo en datos de entrenamiento
# 3. transform() en train y test con el mismo objeto
# 4. Usar pipelines para evitar fugas de datos
# 5. Establecer random_state para reproducibilidad
# 6. Usar validación cruzada para evaluar
# 7. Escalar datos para algoritmos sensibles (SVM, KNN, redes)
# 8. Estratificar en clasificación desbalanceada

# ❌ NO HACER:
# 1. Escalar todo el dataset antes de dividir
# 2. Seleccionar features basándose en todo el dataset
# 3. Ignorar el balanceo de clases
# 4. Evaluar solo en training
# 5. Olvidar guardar el scaler junto con el modelo
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
