# 🤖 Unidad 8. Algoritmos de Ensamblado (Ensemble Learning)

Los **Algoritmos de Ensamblado** (Ensemble Methods) son una técnica de Machine Learning que combina las predicciones de múltiples modelos base (conocidos como *weak learners* o aprendices débiles) para construir un modelo final más robusto y preciso (*strong learner*).

La intuición detrás de esto es la **"Sabiduría de las Masas"**: así como la opinión colectiva de un grupo de expertos suele ser mejor que la de un solo experto, un grupo de modelos predictivos suele superar el rendimiento de un modelo individual.

---

### 8.1. Conceptos Clave y Categorías

El objetivo principal es reducir el **sesgo** (bias) o la **varianza** (variance), o ambos. Los métodos de ensamblado se dividen principalmente en tres categorías según cómo combinan los modelos:

1. **Voting (Votación):** Se entrenan varios modelos diferentes (ej. KNN, SVM, Árbol) y se "vota" para decidir la clase final.
2. **Bagging (Bootstrap Aggregating):** Se entrena el *mismo* algoritmo muchas veces en paralelo, pero con diferentes subconjuntos aleatorios de los datos de entrenamiento. Su objetivo es reducir la **varianza** (evitar overfitting). El ejemplo clásico es **Random Forest**.
3. **Boosting:** Se entrena el *mismo* algoritmo de forma **secuencial**. Cada nuevo modelo intenta corregir los errores cometidos por el modelo anterior. Su objetivo es reducir el **sesgo** (evitar underfitting). Ejemplos: AdaBoost, XGBoost.

---

### 8.2. Voting Classifiers (Votación)

Es la forma más simple de ensamblado. Consiste en agregar las predicciones de clasificadores totalmente diferentes.

#### Tipos de Votación

* **Hard Voting (Votación Dura):** Cada clasificador vota por una clase. La clase con la mayoría de votos gana (moda).
* **Soft Voting (Votación Suave):** Si los clasificadores pueden estimar probabilidades (tienen método `predict_proba`), se promedian las probabilidades de cada clase. La clase con el promedio de probabilidad más alto gana. El *Soft Voting* suele funcionar mejor porque da más peso a los votos con "alta confianza".

#### Ejemplo en Python (Voting)

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Datos de ejemplo
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Modelos individuales
log_clf = LogisticRegression(random_state=42)
rnd_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42) # probability=True necesario para Soft Voting

# Ensamblado por Votación (Soft)
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft'
)

# Entrenamiento y Comparación
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf.__class__.__name__}: {accuracy_score(y_test, y_pred):.4f}")
```

---

### 8.3. Bagging y Random Forest

**Bagging** (Bootstrap Aggregating) implica entrenar el mismo algoritmo en diferentes subconjuntos aleatorios del dataset de entrenamiento.

* **Bootstrap:** El muestreo se hace *con reemplazo* (una misma muestra puede aparecer varias veces en el mismo subconjunto).
* **Pasting:** El muestreo se hace *sin reemplazo*.

Una vez entrenados, los modelos agregan sus predicciones (moda para clasificación, promedio para regresión).

#### Random Forest (Bosques Aleatorios)

Es una implementación específica y optimizada de Bagging usando **Árboles de Decisión**.
Introduce aleatoriedad extra: al dividir un nodo en el árbol, no busca la mejor característica de *todas* las disponibles, sino la mejor característica dentro de un **subconjunto aleatorio de características**. Esto hace que los árboles sean más diversos (descorrelacionados), lo que reduce drásticamente la varianza.

**Hiperparámetros Clave:**

* `n_estimators`: Número de árboles (más es mejor, pero más lento).
* `max_features`: Número máximo de características a considerar en cada división.
* `bootstrap`: Si usar muestreo con reemplazo (True por defecto).
* `n_jobs`: Número de núcleos de CPU a usar (-1 para todos).

#### Ejemplo en Python (Random Forest)

```python
from sklearn.ensemble import RandomForestClassifier

# Instanciar Random Forest
# 500 árboles, usando todos los núcleos de CPU
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Importancia de Características
# Random Forest permite ver qué variables son más útiles
for name, score in zip(["Feature 1", "Feature 2"], rnd_clf.feature_importances_):
    print(f"{name}: {score}")
```

---

### 8.4. Boosting (Impulso)

El Boosting entrena predictores secuencialmente, cada uno intentando corregir a su predecesor.

#### 8.4.1. AdaBoost (Adaptive Boosting)

El algoritmo presta más atención a las instancias de entrenamiento que el predecesor clasificó incorrectamente.

1. Entrena un clasificador base.
2. Aumenta el **peso relativo** de las instancias mal clasificadas.
3. Entrena un segundo clasificador con los pesos actualizados.
4. Repite el proceso.

**Hiperparámetros Clave:**

* `n_estimators`: Número de iteraciones.
* `learning_rate`: Cuánto contribuye cada modelo. Un valor bajo requiere más estimadores.

**Ejemplo Python (AdaBoost):**

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost usando Árboles de Decisión muy simples (stumps)
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42
)
ada_clf.fit(X_train, y_train)
```

#### 8.4.2. Gradient Boosting Machine (GBM)

En lugar de ajustar los pesos de las instancias, GBM intenta ajustar el nuevo predictor a los **errores residuales** (la diferencia entre el valor real y el predicho) del predictor anterior.

**Ejemplo Python (GradientBoosting de sklearn):**

```python
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X_train, y_train)
```

#### 8.4.3. XGBoost (Extreme Gradient Boosting)

Es una versión optimizada de Gradient Boosting diseñada para ser altamente eficiente, flexible y portátil. Es el algoritmo dominante en competiciones de Machine Learning (Kaggle).

* **Regularización:** Incluye regularización L1 y L2 para evitar overfitting.
* **Paralelización:** Construcción de árboles en paralelo.
* **Manejo de nulos:** Aprende automáticamente la mejor dirección para valores faltantes.

**Hiperparámetros Clave:**

* `eta` (learning_rate): Paso de reducción de pesos para prevenir overfitting.
* `max_depth`: Profundidad máxima del árbol.
* `subsample`: Ratio de muestras de entrenamiento usadas.
* `colsample_bytree`: Ratio de columnas usadas por árbol.

**Ejemplo Python (XGBoost):**

```python
import xgboost as xgb

# XGBoost tiene su propia estructura de datos optimizada (DMatrix), pero es compatible con sklearn
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

xgb_clf.fit(X_train, y_train)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_clf.predict(X_test)):.4f}")

# Comparación exhaustiva de todos los métodos de ensamblado
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, VotingClassifier)
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*70)
print("COMPARACIÓN COMPLETA DE MÉTODOS DE ENSAMBLADO")
print("="*70)

# Modelos a comparar
modelos = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
}

# Entrenar y evaluar cada modelo
resultados = {}
for nombre, modelo in modelos.items():
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)
    
    # Entrenar y predecir
    modelo.fit(X_train, y_train)
    y_pred_modelo = modelo.predict(X_test)
    
    # Guardar resultados
    resultados[nombre] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred_modelo)
    }
    
    print(f"\n{nombre}:")
    print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_modelo):.4f}")

# Visualización comparativa
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Comparación de accuracies
modelo_names = list(resultados.keys())
cv_means = [resultados[m]['cv_mean'] for m in modelo_names]
test_accs = [resultados[m]['test_accuracy'] for m in modelo_names]

x_pos = np.arange(len(modelo_names))
width = 0.35

axes[0].bar(x_pos - width/2, cv_means, width, label='CV Score', alpha=0.8)
axes[0].bar(x_pos + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
axes[0].set_xlabel('Modelo')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Comparación de Métodos de Ensamblado')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(modelo_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0.5, 1.0])

# Gráfico 2: Importancia de características (Random Forest vs XGBoost)
feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]

rf_importance = modelos['Random Forest'].feature_importances_
xgb_importance = modelos['XGBoost'].feature_importances_

x_pos_feat = np.arange(len(feature_names))
axes[1].barh(x_pos_feat - width/2, rf_importance, width, label='Random Forest', alpha=0.8)
axes[1].barh(x_pos_feat + width/2, xgb_importance, width, label='XGBoost', alpha=0.8)
axes[1].set_yticks(x_pos_feat)
axes[1].set_yticklabels(feature_names)
axes[1].set_xlabel('Importancia')
axes[1].set_title('Importancia de Características')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Voting Classifier combinando los mejores modelos
print("\n" + "="*70)
print("VOTING CLASSIFIER (Combinación de Modelos)")
print("="*70)

voting_clf = VotingClassifier(
    estimators=[
        ('rf', modelos['Random Forest']),
        ('gb', modelos['Gradient Boosting']),
        ('xgb', modelos['XGBoost'])
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print(f"\nVoting Classifier Accuracy: {accuracy_score(y_test, y_pred_voting):.4f}")
print("\nComparación Final:")
print(f"  Mejor modelo individual: {max(resultados.items(), key=lambda x: x[1]['test_accuracy'])[0]}")
print(f"  Accuracy máxima individual: {max(r['test_accuracy'] for r in resultados.values()):.4f}")
print(f"  Voting Classifier: {accuracy_score(y_test, y_pred_voting):.4f}")
```

#### 8.4.4. LightGBM (Light Gradient Boosting Machine)

Desarrollado por Microsoft. A diferencia de otros que crecen el árbol por niveles (level-wise), LightGBM crece por hojas (**leaf-wise**). Elige la hoja con mayor pérdida para crecer.

* **Ventajas:** Mucho más rápido que XGBoost en grandes datasets y consume menos memoria.
* **Desventajas:** Puede hacer overfitting fácilmente en datasets pequeños (< 10,000 filas).

**Hiperparámetros Clave:**

* `num_leaves`: Parámetro principal para controlar la complejidad (en lugar de max_depth).
* `min_data_in_leaf`: Importante para evitar overfitting.

**Ejemplo Python (LightGBM):**

```python
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)

lgb_clf.fit(X_train, y_train)
print(f"LightGBM Accuracy: {accuracy_score(y_test, lgb_clf.predict(X_test)):.4f}")
```

---

### 8.5. Resumen Comparativo

| Técnica | Algoritmo Principal | Estrategia | Objetivo | Paralelizable |
| :--- | :--- | :--- | :--- | :--- |
| **Voting** | VotingClassifier | Promedio de modelos distintos | Robustez general | Sí |
| **Bagging** | Random Forest | Modelos iguales, datos aleatorios (independientes) | Reducir Varianza | Sí (Muy rápido) |
| **Boosting** | AdaBoost, XGBoost | Modelos iguales, secuenciales (dependientes) | Reducir Sesgo | No (Secuencial)* |

*\* Nota: XGBoost y LightGBM paralelizan la construcción dentro del árbol, pero los árboles se crean secuencialmente.*

### 8.6. Aplicaciones Reales de Algoritmos de Ensamblado

Los métodos de ensamblado dominan actualmente las competiciones de ciencia de datos y las aplicaciones industriales en datos estructurados:

* **Detección de Fraude (Banca):** Algoritmos como XGBoost y Random Forest son el estándar en la industria financiera para detectar transacciones fraudulentas en tiempo real debido a su alta precisión y velocidad.
  * [Detección de fraude con XGBoost](https://github.com/topics/fraud-detection)
* **Diagnóstico Médico:** Random Forest se utiliza para diagnosticar enfermedades (como la retinopatía diabética) analizando múltiples variables de pacientes, ya que proporciona una medida de qué síntomas son más relevantes.
* **Ranking de Búsqueda (Search Engines):** Motores de búsqueda utilizan Gradient Boosting para ordenar los resultados de búsqueda (Learning to Rank), optimizando la relevancia para el usuario.
  * [Learning to Rank con LightGBM](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank)
* **Predicción de Demanda (Retail):** Cadenas de suministro usan estos modelos para predecir la demanda futura de productos, optimizando el inventario y reduciendo desperdicios.

---

### 8.7. Consideraciones Finales

1. **Random Forest** es una excelente "primera opción". Es robusto, requiere poco ajuste de hiperparámetros y nos da la importancia de las características.
2. **XGBoost / LightGBM** suelen ofrecer el **mejor rendimiento** (Accuracy) en datos tabulares estructurados, pero requieren más ajuste de hiperparámetros y cuidado con el overfitting.
3. **Escalado:** Los algoritmos basados en árboles (Random Forest, Boosting) **NO requieren escalado** de características (StandardScaler), lo cual es una gran ventaja práctica.
4. **Interpretabilidad:** Los modelos de ensamblado son "Cajas Negras". Perdemos la interpretabilidad simple de un solo Árbol de Decisión o una Regresión Lineal, aunque podemos usar la "Importancia de Características" para entender qué variables pesan más.

---

📅 **Fecha de creación:** 19/11/2025
✍️ **Autor:** Fran García
