# 🛠️ Unidad 8. Frameworks y Práctica con Deep Learning

Esta unidad cubre los principales frameworks de Deep Learning, herramientas de desarrollo, y mejores prácticas para proyectos de producción.

---

## 8.1. Comparación de Frameworks

### TensorFlow vs PyTorch

| Característica | TensorFlow | PyTorch |
| :--- | :--- | :--- |
| **Desarrollador** | Google | Facebook (Meta) |
| **Paradigma** | Grafo estático → Eager execution | Eager execution (dinámico) |
| **API de Alto Nivel** | Keras (integrado) | torch.nn |
| **Debugging** | TensorBoard, tf.debugging | PyDB, hooks |
| **Producción** | TF Serving, TF Lite, TF.js | TorchServe, ONNX |
| **Comunidad** | Industria, producción | Investigación, academia |
| **Curva de Aprendizaje** | Moderada | Más intuitivo |

### Recomendación

*   **TensorFlow/Keras:** Producción, despliegue móvil, principiantes.
*   **PyTorch:** Investigación, prototipado rápido, flexibilidad.

---

## 8.2. TensorFlow y Keras

### Arquitectura de TensorFlow 2.x

```
┌────────────────────────────────────────────┐
│              Tu Código Python              │
├────────────────────────────────────────────┤
│                 Keras API                  │
├────────────────────────────────────────────┤
│     TensorFlow Core (Operaciones)          │
├────────────────────────────────────────────┤
│  Hardware: CPU / GPU (CUDA) / TPU / Edge   │
└────────────────────────────────────────────┘
```

### Formas de Crear Modelos en Keras

#### Sequential API (más simple)

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

#### Functional API (más flexible)

```python
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

# Múltiples entradas
input_a = Input(shape=(32,), name='input_a')
input_b = Input(shape=(64,), name='input_b')

x_a = Dense(16, activation='relu')(input_a)
x_b = Dense(32, activation='relu')(input_b)

merged = Concatenate()([x_a, x_b])
x = Dense(64, activation='relu')(merged)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=[input_a, input_b], outputs=output)
```

#### Subclassing (máxima flexibilidad)

```python
class MiModelo(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = tf.nn.dropout(x, rate=0.3)
        x = self.dense2(x)
        return self.dense3(x)

model = MiModelo()
```

### Entrenamiento Personalizado

```python
# Training loop personalizado
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function  # Compilar para mayor velocidad
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_acc_metric.update_state(y, predictions)
    return loss

# Loop de entrenamiento
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
    
    train_acc = train_acc_metric.result()
    print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {train_acc:.4f}")
    train_acc_metric.reset_states()
```

---

## 8.3. PyTorch

### Estructura Básica

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Definir modelo
class MiRed(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# Instanciar
model = MiRed(784, 256, 10)

# Mover a GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Entrenamiento en PyTorch

```python
# Configurar
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Crear DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Entrenar
model.train()  # Modo entrenamiento
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Actualizar pesos
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {running_loss/len(train_loader):.4f}")
```

### Evaluación

```python
model.eval()  # Modo evaluación
correct = 0
total = 0

with torch.no_grad():  # No calcular gradientes
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
```

---

## 8.4. Callbacks y Monitorización

### Callbacks en Keras

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

callbacks = [
    # Guardar el mejor modelo
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Reducir learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    ),
    
    # TensorBoard
    TensorBoard(log_dir='./logs'),
    
    # Log a CSV
    CSVLogger('training_log.csv')
]

model.fit(X_train, y_train, callbacks=callbacks)
```

### TensorBoard

```bash
# En terminal
tensorboard --logdir=./logs
# Abrir navegador en http://localhost:6006
```

```python
# Logging personalizado
import tensorflow as tf

log_dir = "logs/custom"
summary_writer = tf.summary.create_file_writer(log_dir)

with summary_writer.as_default():
    tf.summary.scalar('custom_metric', value, step=epoch)
    tf.summary.image('generated', images, step=epoch)
    tf.summary.histogram('weights', model.layers[0].weights[0], step=epoch)
```

---

## 8.5. GPU y Aceleración

### Verificar GPU

```python
# TensorFlow
import tensorflow as tf
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

# PyTorch
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
```

### Configurar Memoria GPU (TensorFlow)

```python
# Crecimiento dinámico de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# O limitar memoria máxima
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
)
```

### Entrenamiento Multi-GPU

```python
# TensorFlow
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(...)

model.fit(...)

# PyTorch
model = nn.DataParallel(model)  # Simple
# O usar DistributedDataParallel para mejor rendimiento
```

---

## 8.6. Optimización del Pipeline de Datos

### tf.data API

```python
import tensorflow as tf

# Crear dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Pipeline optimizado
dataset = (dataset
    .cache()                    # Cachear en memoria
    .shuffle(buffer_size=10000) # Mezclar
    .batch(32)                  # Crear batches
    .prefetch(tf.data.AUTOTUNE) # Cargar siguiente batch mientras se entrena
)

# Para imágenes desde directorio
dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(224, 224),
    batch_size=32
)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### DataLoader en PyTorch

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Carga paralela
    pin_memory=True,    # Acelera transferencia a GPU
    persistent_workers=True
)
```

---

## 8.7. Guardado y Carga de Modelos

### TensorFlow/Keras

```python
# Guardar modelo completo
model.save('modelo_completo.keras')

# Cargar
model = tf.keras.models.load_model('modelo_completo.keras')

# Solo pesos
model.save_weights('pesos.weights.h5')
model.load_weights('pesos.weights.h5')

# SavedModel (para producción)
model.save('modelo_savedmodel', save_format='tf')
```

### PyTorch

```python
# Solo pesos (recomendado)
torch.save(model.state_dict(), 'modelo_pesos.pth')

# Cargar
model = MiRed(784, 256, 10)
model.load_state_dict(torch.load('modelo_pesos.pth'))
model.eval()

# Checkpoint completo (para continuar entrenamiento)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Cargar checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 8.8. Debugging y Profiling

### Debugging en TensorFlow

```python
# Eager execution por defecto permite debugging normal
tf.config.run_functions_eagerly(True)

# Verificar valores
@tf.function
def mi_funcion(x):
    tf.debugging.assert_non_negative(x, message="x debe ser positivo")
    tf.print("Valor de x:", x)
    return x ** 2
```

### Profiling

```python
# TensorFlow Profiler
import tensorflow as tf

# En el callback de TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    profile_batch=(10, 20)  # Perfilar batches 10-20
)

# PyTorch Profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
) as prof:
    for step, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss.backward()
        prof.step()
```

---

## 8.9. Despliegue a Producción

### TensorFlow Serving

```python
# Exportar modelo
model.save('modelo_produccion/1')

# Docker
# docker pull tensorflow/serving
# docker run -p 8501:8501 --mount type=bind,source=/path/modelo_produccion,target=/models/mi_modelo -e MODEL_NAME=mi_modelo -t tensorflow/serving

# Hacer predicción vía REST
import requests
import json

data = json.dumps({"instances": X_test[:5].tolist()})
response = requests.post(
    'http://localhost:8501/v1/models/mi_modelo:predict',
    data=data,
    headers={"content-type": "application/json"}
)
predictions = response.json()['predictions']
```

### TensorFlow Lite (Móvil)

```python
# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Cuantización
tflite_model = converter.convert()

# Guardar
with open('modelo.tflite', 'wb') as f:
    f.write(tflite_model)
```

### ONNX (Interoperabilidad)

```python
# PyTorch a ONNX
import torch.onnx

dummy_input = torch.randn(1, 784)
torch.onnx.export(
    model,
    dummy_input,
    "modelo.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# TensorFlow a ONNX
import tf2onnx
model_proto, _ = tf2onnx.convert.from_keras(model)
```

---

## 8.10. Mejores Prácticas

### Estructura de Proyecto

```
mi_proyecto/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── checkpoints/
│   └── saved/
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── architectures.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train.py
│   │   └── callbacks.py
│   └── utils/
│       └── helpers.py
├── notebooks/
│   └── exploration.ipynb
├── configs/
│   └── config.yaml
├── tests/
├── requirements.txt
└── README.md
```

### Reproducibilidad

```python
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # PyTorch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

set_seeds(42)
```

### Configuración con YAML

```yaml
# config.yaml
model:
  architecture: resnet50
  num_classes: 10
  dropout: 0.3

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  
data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: [224, 224]
```

```python
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

batch_size = config['training']['batch_size']
```

### Logging con Weights & Biases

```python
import wandb

wandb.init(project="mi-proyecto", config=config)

# Durante entrenamiento
wandb.log({
    "loss": loss,
    "accuracy": accuracy,
    "epoch": epoch
})

# Al final
wandb.finish()
```

---

## 8.11. Recursos Adicionales

### Documentación Oficial

*   [TensorFlow](https://www.tensorflow.org/learn)
*   [PyTorch](https://pytorch.org/tutorials/)
*   [Keras](https://keras.io/guides/)
*   [Hugging Face](https://huggingface.co/docs)

### Cursos Recomendados

*   Deep Learning Specialization (Coursera/Andrew Ng).
*   Fast.ai (Practical Deep Learning).
*   Stanford CS231n (Computer Vision).
*   Stanford CS224n (NLP).

### Libros

*   "Deep Learning" - Goodfellow, Bengio, Courville.
*   "Hands-On Machine Learning" - Aurélien Géron.
*   "Deep Learning with Python" - François Chollet.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
