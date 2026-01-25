# 🧠 Unidad 1. Fundamentos del Deep Learning

El **Deep Learning** (Aprendizaje Profundo) es un subcampo del Machine Learning que utiliza redes neuronales artificiales con múltiples capas para aprender representaciones jerárquicas de los datos.

---

## 1.1. ¿Qué es el Deep Learning?

El Deep Learning se diferencia del ML tradicional en su capacidad para aprender **automáticamente** las características relevantes de los datos, sin necesidad de ingeniería manual de features.

### Machine Learning vs Deep Learning

| Aspecto | ML Tradicional | Deep Learning |
| :--- | :--- | :--- |
| **Features** | Ingeniería manual | Aprendizaje automático |
| **Datos** | Funciona con menos datos | Requiere muchos datos |
| **Hardware** | CPU suficiente | GPU/TPU preferible |
| **Interpretabilidad** | Alta | Baja (caja negra) |
| **Rendimiento (big data)** | Se estanca | Mejora con más datos |

### ¿Por qué "Profundo"?

El término "profundo" se refiere a la **cantidad de capas** en la red neuronal. Mientras que las redes neuronales tradicionales tenían 1-2 capas ocultas, las redes profundas pueden tener cientos o miles de capas.

```
Red Neuronal Tradicional:    Red Neuronal Profunda:
    Input                        Input
      |                            |
    [Capa]                      [Capa]
      |                            |
    Output                      [Capa]
                                   |
                                [Capa]
                                   |
                                  ...
                                   |
                                [Capa]
                                   |
                                Output
```

---

## 1.2. La Neurona Artificial (Perceptrón)

La unidad básica de una red neuronal es la **neurona artificial**, inspirada (de forma simplificada) en las neuronas biológicas.

### Funcionamiento

1.  **Entradas:** Recibe valores $x_1, x_2, ..., x_n$
2.  **Pesos:** Cada entrada tiene un peso asociado $w_1, w_2, ..., w_n$
3.  **Suma ponderada:** $z = \sum_{i=1}^{n} w_i \cdot x_i + b$ (donde $b$ es el bias)
4.  **Activación:** $a = f(z)$ (función de activación)
5.  **Salida:** El valor $a$ se pasa a la siguiente capa

### Representación Matemática

$$a = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(W^T X + b)$$

```python
import numpy as np

def neurona(X, W, b, activacion='sigmoid'):
    """
    Simula una neurona artificial.
    
    X: vector de entradas
    W: vector de pesos
    b: bias
    """
    # Suma ponderada
    z = np.dot(W, X) + b
    
    # Función de activación
    if activacion == 'sigmoid':
        a = 1 / (1 + np.exp(-z))
    elif activacion == 'relu':
        a = np.maximum(0, z)
    elif activacion == 'tanh':
        a = np.tanh(z)
    
    return a

# Ejemplo
X = np.array([0.5, 0.3, 0.2])
W = np.array([0.4, 0.6, 0.8])
b = 0.1

salida = neurona(X, W, b)
print(f"Salida de la neurona: {salida:.4f}")
```

---

## 1.3. Funciones de Activación

Las funciones de activación introducen **no-linealidad** en la red, permitiendo aprender relaciones complejas.

### Funciones Comunes

| Función | Fórmula | Rango | Uso |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | (0, 1) | Clasificación binaria (salida) |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | Capas ocultas (alternativa a sigmoid) |
| **ReLU** | $f(z) = \max(0, z)$ | [0, ∞) | Capas ocultas (más usada) |
| **Leaky ReLU** | $f(z) = \max(0.01z, z)$ | (-∞, ∞) | Evita "neuronas muertas" |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | (0, 1) | Clasificación multiclase (salida) |

### Visualización

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 100)

# Funciones de activación
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
leaky_relu = np.where(z > 0, z, 0.01 * z)

# Graficar
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(z, sigmoid)
axes[0, 0].set_title('Sigmoid')
axes[0, 0].grid(True)

axes[0, 1].plot(z, tanh)
axes[0, 1].set_title('Tanh')
axes[0, 1].grid(True)

axes[1, 0].plot(z, relu)
axes[1, 0].set_title('ReLU')
axes[1, 0].grid(True)

axes[1, 1].plot(z, leaky_relu)
axes[1, 1].set_title('Leaky ReLU')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

### ¿Por qué ReLU es Popular?

*   **Evita el problema del gradiente desvaneciente:** La derivada es 1 para valores positivos.
*   **Computacionalmente eficiente:** Solo comparación y selección.
*   **Promueve dispersión:** Muchas neuronas producen 0, creando representaciones dispersas.

---

## 1.4. Arquitectura de una Red Neuronal

### Capas

1.  **Capa de Entrada:** Recibe los datos crudos. Número de neuronas = número de features.
2.  **Capas Ocultas:** Procesan la información. El número y tamaño define la capacidad del modelo.
3.  **Capa de Salida:** Produce la predicción final. Depende del tipo de problema.

### Ejemplo de Arquitectura

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Red neuronal simple
modelo = Sequential([
    Input(shape=(10,)),              # 10 features de entrada
    Dense(64, activation='relu'),    # Primera capa oculta: 64 neuronas
    Dense(32, activation='relu'),    # Segunda capa oculta: 32 neuronas
    Dense(16, activation='relu'),    # Tercera capa oculta: 16 neuronas
    Dense(1, activation='sigmoid')   # Salida: clasificación binaria
])

modelo.summary()
```

### Tipos de Problemas y Salidas

| Problema | Capa de Salida | Función de Activación | Loss Function |
| :--- | :--- | :--- | :--- |
| Regresión | 1 neurona | Lineal (ninguna) | MSE |
| Clasificación Binaria | 1 neurona | Sigmoid | Binary Crossentropy |
| Clasificación Multiclase | N neuronas (N clases) | Softmax | Categorical Crossentropy |

---

## 1.5. Forward Propagation

Es el proceso de pasar los datos a través de la red para obtener una predicción.

```python
import numpy as np

def forward_propagation(X, pesos, biases, activaciones):
    """
    Realiza forward propagation a través de la red.
    """
    activacion = X
    activaciones_cache = [X]
    
    for i, (W, b, func_act) in enumerate(zip(pesos, biases, activaciones)):
        z = np.dot(activacion, W) + b
        
        if func_act == 'relu':
            activacion = np.maximum(0, z)
        elif func_act == 'sigmoid':
            activacion = 1 / (1 + np.exp(-z))
        elif func_act == 'softmax':
            exp_z = np.exp(z - np.max(z))
            activacion = exp_z / exp_z.sum()
        
        activaciones_cache.append(activacion)
    
    return activacion, activaciones_cache
```

---

## 1.6. Función de Pérdida (Loss Function)

La función de pérdida mide qué tan lejos están las predicciones de los valores reales.

### Funciones Comunes

**Mean Squared Error (MSE)** - Para regresión:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Binary Cross-Entropy** - Para clasificación binaria:

$$BCE = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Categorical Cross-Entropy** - Para clasificación multiclase:

$$CCE = -\sum_{i=1}^{n}\sum_{c=1}^{C}y_{i,c}\log(\hat{y}_{i,c})$$

```python
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15  # Evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## 1.7. Backpropagation

El **Backpropagation** es el algoritmo que calcula los gradientes de la función de pérdida con respecto a cada peso, usando la regla de la cadena.

### Proceso

1.  **Forward pass:** Calcular predicciones.
2.  **Calcular pérdida:** Comparar predicción con valor real.
3.  **Backward pass:** Calcular gradientes desde la salida hacia la entrada.
4.  **Actualizar pesos:** Usar los gradientes para ajustar los pesos.

### Regla de la Cadena

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Donde:
- $L$ es la pérdida
- $a$ es la activación
- $z$ es la suma ponderada
- $w$ es el peso

---

## 1.8. Optimizadores

Los optimizadores actualizan los pesos basándose en los gradientes calculados.

### Gradient Descent

$$w_{nuevo} = w_{actual} - \eta \cdot \nabla L$$

Donde $\eta$ es el **learning rate** (tasa de aprendizaje).

### Variantes Populares

| Optimizador | Descripción | Cuándo usarlo |
| :--- | :--- | :--- |
| **SGD** | Actualiza con mini-batches | Baseline, simple |
| **SGD + Momentum** | Añade "inercia" a las actualizaciones | Acelera convergencia |
| **RMSprop** | Adapta learning rate por parámetro | Datos no estacionarios |
| **Adam** | Combina Momentum + RMSprop | Default recomendado |
| **AdamW** | Adam con weight decay correcto | Estado del arte |

### Ejemplo en Keras

```python
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Diferentes optimizadores
optimizadores = {
    'adam': Adam(learning_rate=0.001),
    'sgd': SGD(learning_rate=0.01, momentum=0.9),
    'rmsprop': RMSprop(learning_rate=0.001)
}

# Compilar modelo con Adam
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 1.9. Ejemplo Completo: Clasificación con Keras

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Crear dataset sintético
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_classes=2,
    random_state=42
)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar (importante para redes neuronales)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir modelo
modelo = Sequential([
    Input(shape=(20,)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenar
historia = modelo.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluar
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f"\nAccuracy en test: {accuracy:.4f}")

# Visualizar entrenamiento
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historia.history['loss'], label='Train')
plt.plot(historia.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historia.history['accuracy'], label='Train')
plt.plot(historia.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 1.10. Frameworks de Deep Learning

### TensorFlow / Keras

*   Desarrollado por Google.
*   Keras es la API de alto nivel.
*   Excelente para producción y despliegue.

### PyTorch

*   Desarrollado por Meta (Facebook).
*   Muy popular en investigación.
*   Definición dinámica del grafo computacional.

```python
# Mismo modelo en PyTorch
import torch
import torch.nn as nn

class RedNeuronal(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

modelo_pytorch = RedNeuronal()
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
