# 💬 Unidad 6. Clasificación de Texto

La **Clasificación de Texto** es una de las tareas más comunes y útiles del NLP. Consiste en asignar automáticamente una o más categorías predefinidas a un documento de texto.


![Ilustración de text class](../assets/images/text_class.svg)
---

## 6.1. ¿Qué es la Clasificación de Texto?

Es el proceso de categorizar texto en grupos organizados, utilizando técnicas de NLP y Machine Learning.

### Tipos de Clasificación

| Tipo | Descripción | Ejemplo |
| :--- | :--- | :--- |
| **Binaria** | Dos clases | Spam / No Spam |
| **Multiclase** | Múltiples clases, una por documento | Categoría de noticia (Deportes, Política, Tecnología) |
| **Multi-etiqueta** | Múltiples etiquetas por documento | Tags de un artículo (puede ser "Python" Y "Machine Learning") |
| **Jerárquica** | Categorías con estructura de árbol | Taxonomías de productos |

### Aplicaciones Comunes

*   **Filtrado de Spam:** Clasificar emails como spam o legítimos.
*   **Análisis de Sentimientos:** Positivo / Negativo / Neutral.
*   **Categorización de Noticias:** Deportes, Política, Economía, etc.
*   **Detección de Idioma:** Identificar el idioma de un texto.
*   **Clasificación de Tickets de Soporte:** Routing automático.
*   **Detección de Contenido Tóxico:** Moderación de comentarios.
*   **Clasificación de Intents:** Para chatbots y asistentes virtuales.

---

## 6.2. Pipeline de Clasificación de Texto

```
1. Recolección de Datos
        ↓
2. Preprocesamiento
        ↓
3. Representación (Vectorización)
        ↓
4. División Train/Test
        ↓
5. Entrenamiento del Modelo
        ↓
6. Evaluación
        ↓
7. Optimización
        ↓
8. Despliegue
```

---

## 6.3. Clasificación con Machine Learning Tradicional

### Ejemplo Completo: Clasificación de Noticias

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset de ejemplo (en práctica usar datasets reales)
datos = {
    'texto': [
        "El equipo ganó el campeonato de fútbol este domingo",
        "El partido de baloncesto terminó con una victoria aplastante",
        "El tenista español avanzó a la final del torneo",
        "El gobierno aprobó nuevas medidas económicas",
        "El presidente firmó el acuerdo internacional",
        "Las elecciones se celebrarán el próximo mes",
        "Apple lanzó su nuevo iPhone con mejoras en cámara",
        "Microsoft presentó actualizaciones de Windows",
        "Google desarrolla nueva inteligencia artificial",
        "Las acciones subieron tras el anuncio del banco central",
        "La bolsa cerró con ganancias récord",
        "El PIB creció un 3% el último trimestre"
    ],
    'categoria': [
        'deportes', 'deportes', 'deportes',
        'politica', 'politica', 'politica',
        'tecnologia', 'tecnologia', 'tecnologia',
        'economia', 'economia', 'economia'
    ]
}

df = pd.DataFrame(datos)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    df['texto'], df['categoria'], 
    test_size=0.25, 
    random_state=42,
    stratify=df['categoria']  # Mantener proporción de clases
)

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigramas y bigramas
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar y comparar múltiples modelos
modelos = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

resultados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train_tfidf, y_train)
    y_pred = modelo.predict(X_test_tfidf)
    
    # Calcular accuracy
    accuracy = (y_pred == y_test).mean()
    resultados[nombre] = accuracy
    
    print(f"\n{'='*50}")
    print(f"{nombre}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
```

### Selección del Mejor Modelo

```python
# Comparación visual de modelos
plt.figure(figsize=(10, 6))
plt.bar(resultados.keys(), resultados.values())
plt.title('Comparación de Modelos')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, (nombre, acc) in enumerate(resultados.items()):
    plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
plt.tight_layout()
plt.show()
```

---

## 6.4. Clasificación con Deep Learning

### Red Neuronal Simple con Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Preprocesamiento
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding para longitud uniforme
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)

# Modelo
model = Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(
    X_train_pad, y_train_encoded,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluación
loss, accuracy = model.evaluate(X_test_pad, y_test_encoded)
print(f"\nAccuracy en test: {accuracy:.4f}")
```

---

## 6.5. Clasificación con Transformers

### Usando Hugging Face

```python
from transformers import pipeline

# Pipeline de clasificación zero-shot
classifier = pipeline("zero-shot-classification")

texto = "El nuevo smartphone tiene una cámara de 200 megapíxeles"
categorias = ["tecnología", "deportes", "política", "economía"]

resultado = classifier(texto, categorias)
print(f"Texto: {texto}")
print(f"Categoría: {resultado['labels'][0]} ({resultado['scores'][0]:.3f})")
```

### Fine-tuning de BERT para Clasificación

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np

# Preparar datos
train_data = Dataset.from_dict({
    'text': list(X_train),
    'label': list(y_train_encoded)
})

test_data = Dataset.from_dict({
    'text': list(X_test),
    'label': list(y_test_encoded)
})

# Cargar tokenizer y modelo
model_name = "bert-base-multilingual-cased"  # Para español
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True,
        max_length=128
    )

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Cargar modelo
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes
)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Métrica
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics
)

# Entrenar
trainer.train()

# Evaluar
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
```

---

## 6.6. Clasificación Multi-etiqueta

Cuando un documento puede pertenecer a múltiples categorías simultáneamente.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Datos multi-etiqueta
textos = [
    "Python es excelente para machine learning y análisis de datos",
    "JavaScript permite crear aplicaciones web interactivas",
    "El framework Django usa Python para desarrollo web",
    "React y Node.js son populares en desarrollo web"
]

# Cada texto puede tener múltiples etiquetas
etiquetas = [
    ['python', 'machine-learning', 'data-science'],
    ['javascript', 'web'],
    ['python', 'web'],
    ['javascript', 'web']
]

# Binarizar etiquetas
mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform(etiquetas)

print("Clases:", mlb.classes_)
print("Etiquetas binarizadas:\n", y_multi)

# Vectorizar textos
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(textos)

# Modelo multi-etiqueta
modelo_multi = MultiOutputClassifier(LogisticRegression())
modelo_multi.fit(X, y_multi)

# Predicción
nuevo_texto = ["Aprende Python para ciencia de datos y web"]
nuevo_X = vectorizer.transform(nuevo_texto)
prediccion = modelo_multi.predict(nuevo_X)

# Decodificar
etiquetas_pred = mlb.inverse_transform(prediccion)
print(f"Predicción: {etiquetas_pred}")
```

---

## 6.7. Detección de Idioma

Un caso especial de clasificación de texto.

```python
from langdetect import detect, detect_langs

textos = [
    "Hello, how are you today?",
    "Hola, ¿cómo estás hoy?",
    "Bonjour, comment allez-vous?",
    "Guten Tag, wie geht es Ihnen?",
    "Ciao, come stai oggi?"
]

for texto in textos:
    idioma = detect(texto)
    probabilidades = detect_langs(texto)
    print(f"'{texto[:30]}...'")
    print(f"  Idioma: {idioma}")
    print(f"  Probabilidades: {probabilidades}")
    print()
```

---

## 6.8. Clasificación de Intents (Chatbots)

Para sistemas conversacionales, clasificamos la intención del usuario.

```python
# Datos de ejemplo para chatbot
intents_data = {
    'texto': [
        "hola", "buenos días", "qué tal",
        "adiós", "hasta luego", "nos vemos",
        "cuál es el precio", "cuánto cuesta", "precio del producto",
        "quiero comprar", "añadir al carrito", "realizar pedido",
        "dónde está mi pedido", "estado del envío", "cuándo llega"
    ],
    'intent': [
        'saludo', 'saludo', 'saludo',
        'despedida', 'despedida', 'despedida',
        'consulta_precio', 'consulta_precio', 'consulta_precio',
        'compra', 'compra', 'compra',
        'seguimiento', 'seguimiento', 'seguimiento'
    ]
}

df_intents = pd.DataFrame(intents_data)

# Entrenar clasificador de intents
vectorizer_intent = TfidfVectorizer(ngram_range=(1, 2))
X_intent = vectorizer_intent.fit_transform(df_intents['texto'])

clf_intent = LogisticRegression()
clf_intent.fit(X_intent, df_intents['intent'])

# Función para clasificar mensaje de usuario
def clasificar_intent(mensaje, vectorizer, modelo):
    X = vectorizer.transform([mensaje])
    intent = modelo.predict(X)[0]
    proba = modelo.predict_proba(X).max()
    return intent, proba

# Probar
mensajes_test = [
    "hola, buenas tardes",
    "cuánto vale este artículo",
    "quiero hacer un pedido",
    "dónde está mi paquete"
]

print("Clasificación de Intents:")
print("-" * 50)
for msg in mensajes_test:
    intent, confianza = clasificar_intent(msg, vectorizer_intent, clf_intent)
    print(f"'{msg}'")
    print(f"  → Intent: {intent} (confianza: {confianza:.2f})")
    print()
```

---

## 6.9. Métricas de Evaluación

### Para Clasificación Multi-clase

```python
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1-Score (macro):", f1_score(y_test, y_pred, average='macro'))

# Reporte detallado
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.show()
```

### Promedios para Multi-clase

| Promedio | Descripción |
| :--- | :--- |
| **micro** | Calcula métricas globalmente contando TP, FP, FN totales |
| **macro** | Promedio simple de métricas por clase (no pondera por tamaño) |
| **weighted** | Promedio ponderado por el número de muestras por clase |

---

## 6.10. Consideraciones Prácticas

### Desbalance de Clases

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calcular pesos de clase
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Usar en modelo
clf = LogisticRegression(class_weight='balanced')
```

### Validación Cruzada

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='f1_macro')
print(f"F1-Score (CV): {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### Optimización de Hiperparámetros

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [100, 500, 1000]
}

grid_search = GridSearchCV(
    LogisticRegression(), 
    param_grid, 
    cv=5, 
    scoring='f1_macro'
)
grid_search.fit(X_train_tfidf, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score: {grid_search.best_score_:.4f}")
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
