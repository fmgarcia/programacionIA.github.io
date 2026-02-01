# 🧠 Unidad 3. Redes Neuronales Recurrentes (RNN, LSTM, GRU)

Las **Redes Neuronales Recurrentes** (RNN) son arquitecturas diseñadas para procesar datos secuenciales, donde el orden de los elementos importa. Son fundamentales para tareas como procesamiento de texto, series temporales y audio.


![Ilustración de rnn](../assets/images/rnn.svg)
---

## 3.1. ¿Por qué Redes Recurrentes?

### Limitaciones de las Redes Feed-Forward

Las redes neuronales tradicionales:

*   Procesan entradas de tamaño fijo.
*   No tienen "memoria" de entradas anteriores.
*   Tratan cada entrada de forma independiente.

### Datos Secuenciales

Muchos problemas involucran secuencias donde el contexto importa:

*   **Texto:** "No me gusta" vs "Me gusta mucho" - el significado cambia por contexto.
*   **Series Temporales:** El precio de hoy depende de precios anteriores.
*   **Audio:** Los fonemas se interpretan según los anteriores.
*   **Video:** Los frames tienen continuidad temporal.

---

## 3.2. Arquitectura de una RNN

La idea clave es que la red tiene un **estado oculto** que se actualiza en cada paso temporal y captura información de pasos anteriores.

### Estructura

```
        ┌──────────────────────────────────┐
        │                                  │
        ▼                                  │
    ┌───────┐   ┌───────┐   ┌───────┐     │
    │ h(t-1)│──▶│  h(t) │──▶│ h(t+1)│─────┘
    └───────┘   └───────┘   └───────┘
        ▲           ▲           ▲
        │           │           │
    ┌───────┐   ┌───────┐   ┌───────┐
    │ x(t-1)│   │  x(t) │   │ x(t+1)│
    └───────┘   └───────┘   └───────┘
    
    Entrada     Entrada     Entrada
    paso t-1    paso t      paso t+1
```

### Ecuaciones

En cada paso temporal $t$:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

$$y_t = W_{hy} \cdot h_t + b_y$$

Donde:

*   $h_t$ = Estado oculto en tiempo $t$
*   $x_t$ = Entrada en tiempo $t$
*   $y_t$ = Salida en tiempo $t$
*   $W_{hh}, W_{xh}, W_{hy}$ = Matrices de pesos (compartidas en todos los pasos)

### Implementación Básica

```python
import numpy as np

class RNNSimple:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar pesos
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, h_prev):
        """
        Forward pass a través de la secuencia.
        inputs: lista de vectores de entrada
        h_prev: estado oculto inicial
        """
        h = h_prev
        outputs = []
        hidden_states = [h]
        
        for x in inputs:
            # Actualizar estado oculto
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            # Calcular salida
            y = np.dot(self.Why, h) + self.by
            
            outputs.append(y)
            hidden_states.append(h)
        
        return outputs, hidden_states

# Ejemplo
rnn = RNNSimple(input_size=10, hidden_size=20, output_size=5)
secuencia = [np.random.randn(10, 1) for _ in range(5)]
h0 = np.zeros((20, 1))

outputs, states = rnn.forward(secuencia, h0)
print(f"Número de salidas: {len(outputs)}")
print(f"Forma de cada salida: {outputs[0].shape}")
```

---

## 3.3. El Problema del Gradiente Desvaneciente

### El Problema

Durante backpropagation a través del tiempo (BPTT), los gradientes se multiplican repetidamente. Esto causa:

*   **Gradiente Desvaneciente:** Los gradientes se hacen muy pequeños, y la red no puede aprender dependencias a largo plazo.
*   **Gradiente Explosivo:** Los gradientes crecen exponencialmente (menos común, se mitiga con gradient clipping).

### Consecuencia

Las RNN simples tienen dificultad para aprender relaciones entre elementos distantes en la secuencia.

```
"El gato, que estaba en el jardín persiguiendo a un pájaro, ___"
                    ↑                                         ↑
              (muchas palabras entre "gato" y el verbo final)
```

---

## 3.4. LSTM (Long Short-Term Memory)

Las **LSTM** (Hochreiter & Schmidhuber, 1997) resuelven el problema del gradiente desvaneciente mediante **puertas** (gates) que controlan el flujo de información.

### Componentes

1.  **Cell State ($C_t$):** La "memoria a largo plazo" que fluye a través de la red.
2.  **Forget Gate ($f_t$):** Decide qué información descartar de la memoria.
3.  **Input Gate ($i_t$):** Decide qué información nueva añadir.
4.  **Output Gate ($o_t$):** Decide qué parte de la memoria usar para la salida.

### Ecuaciones

**Forget Gate:**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate:**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Actualización de Celda:**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate:**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

### Diagrama

```
                    ┌─────────────────────────────────────────┐
                    │              Cell State                  │
    C(t-1) ────────▶│───────[×]─────────[+]──────────────────▶│───▶ C(t)
                    │        ↑           ↑                     │
                    │     forget      input                    │
                    │      gate       gate                     │
                    │        │           │                     │
                    │   ┌────┴────┐ ┌────┴────┐               │
                    │   │    σ   │ │   σ × tanh│              │
                    │   └────────┘ └──────────┘               │
                    │        ↑           ↑                     │
    h(t-1) ─────────│────────┼───────────┼───────────[×]─────▶│───▶ h(t)
                    │                              ↑           │
                    │                           output         │
                    │                            gate          │
                    │                         ┌───┴───┐        │
                    │                         │   σ   │        │
                    │                         └───────┘        │
                    └─────────────────────────────────────────┘
                                      ↑
    x(t) ─────────────────────────────┘
```

### Implementación en Keras

```python
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras import Sequential

# LSTM simple
modelo = Sequential([
    Input(shape=(100, 50)),  # Secuencia de 100 pasos, 50 features cada uno
    LSTM(128),               # 128 unidades LSTM
    Dense(1, activation='sigmoid')
])

# LSTM apiladas (stacked)
modelo_profundo = Sequential([
    Input(shape=(100, 50)),
    LSTM(128, return_sequences=True),  # return_sequences=True para apilar
    LSTM(64, return_sequences=True),
    LSTM(32),                          # Última capa: return_sequences=False
    Dense(1)
])
```

---

## 3.5. GRU (Gated Recurrent Unit)

Las **GRU** (Cho et al., 2014) son una simplificación de las LSTM con menos parámetros.

### Diferencias con LSTM

*   Combina forget e input gates en un solo **update gate**.
*   No tiene cell state separado.
*   Menos parámetros → más rápido de entrenar.
*   Rendimiento similar en muchas tareas.

### Ecuaciones

**Update Gate:**
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**Reset Gate:**
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**Candidato:**
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

**Estado Oculto:**
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### Implementación

```python
from tensorflow.keras.layers import GRU

modelo_gru = Sequential([
    Input(shape=(100, 50)),
    GRU(128),
    Dense(1)
])
```

### Comparación LSTM vs GRU

| Característica | LSTM | GRU |
| :--- | :--- | :--- |
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parámetros | Más | Menos |
| Memoria | Cell state separado | Solo hidden state |
| Velocidad | Más lento | Más rápido |
| Rendimiento | Similar | Similar |

---

## 3.6. RNN Bidireccionales

Procesan la secuencia en ambas direcciones, capturando contexto pasado y futuro.

```
Forward:   x1 → x2 → x3 → x4 → x5
Backward:  x1 ← x2 ← x3 ← x4 ← x5

Salida en t: [forward_h(t), backward_h(t)]
```

```python
from tensorflow.keras.layers import Bidirectional

# LSTM Bidireccional
modelo_bi = Sequential([
    Input(shape=(100, 50)),
    Bidirectional(LSTM(64)),  # Salida: 128 (64 forward + 64 backward)
    Dense(1)
])
```

---

## 3.7. Tipos de Arquitecturas RNN

### Según Entrada/Salida

| Tipo | Entrada | Salida | Uso |
| :--- | :--- | :--- | :--- |
| **Many-to-One** | Secuencia | Un valor | Clasificación de texto |
| **One-to-Many** | Un valor | Secuencia | Generación de texto |
| **Many-to-Many (igual)** | Secuencia | Secuencia (misma longitud) | POS Tagging, NER |
| **Many-to-Many (diferente)** | Secuencia | Secuencia (otra longitud) | Traducción (Seq2Seq) |

```python
# Many-to-One: Clasificación de sentimiento
modelo_m2o = Sequential([
    Input(shape=(100, 50)),
    LSTM(64),
    Dense(2, activation='softmax')  # Positivo/Negativo
])

# Many-to-Many: Etiquetado de secuencia
modelo_m2m = Sequential([
    Input(shape=(100, 50)),
    LSTM(64, return_sequences=True),
    Dense(10, activation='softmax')  # Una etiqueta por paso
])
```

---

## 3.8. Ejemplo Completo: Predicción de Series Temporales

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generar datos sintéticos (serie temporal sinusoidal con ruido)
np.random.seed(42)
t = np.linspace(0, 100, 1000)
serie = np.sin(t) + 0.1 * np.random.randn(1000)

# Normalizar
scaler = MinMaxScaler()
serie_normalizada = scaler.fit_transform(serie.reshape(-1, 1))

# Crear secuencias para entrenamiento
def crear_secuencias(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 50
X, y = crear_secuencias(serie_normalizada, window_size)

# División train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train shape: {X_train.shape}")  # (N, 50, 1)
print(f"y_train shape: {y_train.shape}")  # (N, 1)

# Crear modelo LSTM
modelo = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(25, activation='relu'),
    Dense(1)
])

modelo.compile(optimizer='adam', loss='mse')

# Entrenar
historia = modelo.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predecir
predicciones = modelo.predict(X_test)

# Visualizar
plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Real', alpha=0.7)
plt.plot(predicciones, label='Predicción', alpha=0.7)
plt.title('Predicción de Serie Temporal con LSTM')
plt.legend()
plt.show()
```

---

## 3.9. Ejemplo: Clasificación de Texto con LSTM

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar dataset IMDB
max_features = 10000  # Vocabulario
maxlen = 200          # Longitud máxima de secuencia

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Padding para longitud uniforme
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Modelo
from tensorflow.keras.layers import Embedding

modelo = Sequential([
    Input(shape=(maxlen,)),
    Embedding(max_features, 128),  # Embedding de palabras
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
])

modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenar
historia = modelo.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluar
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f"Accuracy en test: {accuracy:.4f}")
```

---

## 3.10. Técnicas de Regularización para RNN

### Dropout

```python
from tensorflow.keras.layers import Dropout

modelo = Sequential([
    Input(shape=(100, 50)),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # Dropout en LSTM
    Dropout(0.5),  # Dropout después de LSTM
    Dense(1)
])
```

### Batch Normalization

```python
from tensorflow.keras.layers import BatchNormalization

modelo = Sequential([
    Input(shape=(100, 50)),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(32),
    Dense(1)
])
```

### Gradient Clipping

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
modelo.compile(optimizer=optimizer, loss='mse')
```

---

## 3.11. Consideraciones Prácticas

### Cuándo usar RNN/LSTM/GRU

*   **Secuencias cortas a medianas:** RNN simple puede funcionar.
*   **Secuencias largas:** LSTM o GRU (mejor manejo de dependencias largas).
*   **Recursos limitados:** GRU (menos parámetros).
*   **Contexto bidireccional necesario:** BiLSTM/BiGRU.

### Alternativas Modernas

Para muchas tareas de NLP, los **Transformers** han superado a las RNN:

*   Paralelización más eficiente.
*   Mejor captura de dependencias largas.
*   Modelos preentrenados disponibles (BERT, GPT).

Sin embargo, las RNN siguen siendo útiles para:

*   Series temporales.
*   Streaming de datos (procesamiento en tiempo real).
*   Recursos computacionales limitados.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
