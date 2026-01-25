# 💬 Unidad 7. Transformers y Modelos de Lenguaje

Los **Transformers** han revolucionado el NLP desde su introducción en 2017. Esta arquitectura es la base de modelos como BERT, GPT, T5 y los modernos Large Language Models (LLMs).

---

## 7.1. Introducción a los Transformers

### El Problema con las RNN

Antes de los Transformers, las Redes Neuronales Recurrentes (RNN) y sus variantes (LSTM, GRU) eran el estándar para procesamiento de secuencias. Sin embargo, tenían limitaciones:

*   **Procesamiento secuencial:** No pueden paralelizarse eficientemente.
*   **Memoria a largo plazo:** Dificultad para capturar dependencias lejanas.
*   **Entrenamiento lento:** Debido a la naturaleza secuencial.

### La Solución: Atención

El paper "Attention Is All You Need" (Vaswani et al., 2017) introdujo el mecanismo de **Self-Attention** que permite:

*   **Paralelización completa:** Procesa toda la secuencia a la vez.
*   **Contexto global:** Cada token puede atender a cualquier otro token.
*   **Escalabilidad:** Permite entrenar modelos muy grandes.

---

## 7.2. Arquitectura del Transformer

### Componentes Principales

```
┌─────────────────────────────────────────────┐
│                  OUTPUT                      │
└─────────────────────────────────────────────┘
                      ↑
┌─────────────────────────────────────────────┐
│              DECODER (x N)                   │
│  ┌─────────────────────────────────────┐    │
│  │    Feed Forward Neural Network      │    │
│  └─────────────────────────────────────┘    │
│                    ↑                         │
│  ┌─────────────────────────────────────┐    │
│  │      Cross-Attention                │    │
│  └─────────────────────────────────────┘    │
│                    ↑                         │
│  ┌─────────────────────────────────────┐    │
│  │  Masked Self-Attention              │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                      ↑
┌─────────────────────────────────────────────┐
│              ENCODER (x N)                   │
│  ┌─────────────────────────────────────┐    │
│  │    Feed Forward Neural Network      │    │
│  └─────────────────────────────────────┘    │
│                    ↑                         │
│  ┌─────────────────────────────────────┐    │
│  │         Self-Attention              │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                      ↑
┌─────────────────────────────────────────────┐
│  Input Embedding + Positional Encoding      │
└─────────────────────────────────────────────┘
                      ↑
┌─────────────────────────────────────────────┐
│                  INPUT                       │
└─────────────────────────────────────────────┘
```

### Self-Attention

El mecanismo de **Self-Attention** permite que cada token "atienda" a todos los demás tokens de la secuencia.

Para cada token, se calculan tres vectores:

*   **Query (Q):** Lo que este token está "buscando"
*   **Key (K):** Lo que este token "ofrece"
*   **Value (V):** La información que contiene

La fórmula de atención:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde $d_k$ es la dimensión de las keys.

### Multi-Head Attention

En lugar de una sola atención, se usan múltiples "cabezas" de atención que aprenden diferentes tipos de relaciones:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Donde cada cabeza:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Positional Encoding

Como los Transformers no tienen noción inherente de posición, se añade información posicional mediante:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

---

## 7.3. Tipos de Modelos Transformer

### Encoder-only (BERT)

*   Procesa toda la secuencia bidireccionalmente.
*   Ideal para: Clasificación, NER, Question Answering extractivo.
*   Ejemplos: BERT, RoBERTa, ALBERT, DistilBERT.

### Decoder-only (GPT)

*   Genera texto de izquierda a derecha (autoregresivo).
*   Ideal para: Generación de texto, completado.
*   Ejemplos: GPT-2, GPT-3, GPT-4, LLaMA, Mistral.

### Encoder-Decoder (T5)

*   Arquitectura completa original.
*   Ideal para: Traducción, resumen, seq2seq.
*   Ejemplos: T5, BART, mBART.

---

## 7.4. BERT (Bidirectional Encoder Representations from Transformers)

BERT fue un hito en NLP al introducir el preentrenamiento bidireccional.

### Preentrenamiento

BERT se preentrena con dos tareas:

1.  **Masked Language Model (MLM):** Predecir tokens enmascarados.
    *   Entrada: "El [MASK] come pescado"
    *   Predicción: "gato"

2.  **Next Sentence Prediction (NSP):** Predecir si una oración sigue a otra.

### Usando BERT con Hugging Face

```python
from transformers import BertTokenizer, BertModel
import torch

# Cargar modelo y tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Tokenizar texto
texto = "El procesamiento de lenguaje natural es fascinante."
inputs = tokenizer(texto, return_tensors='pt', padding=True, truncation=True)

# Obtener embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Embedding del token [CLS] (representa toda la oración)
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"Dimensión del embedding: {cls_embedding.shape}")  # [1, 768]

# Embedding de cada token
all_embeddings = outputs.last_hidden_state
print(f"Embeddings por token: {all_embeddings.shape}")  # [1, num_tokens, 768]
```

### Clasificación con BERT

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Datos de ejemplo
train_texts = ["me encanta", "lo odio", "está bien", "increíble", "terrible"]
train_labels = [1, 0, 1, 1, 0]

# Crear dataset
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

# Tokenizar
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)

# Cargar modelo para clasificación
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=2
)

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Entrenar
trainer.train()
```

---

## 7.5. GPT y Modelos Generativos

GPT (Generative Pre-trained Transformer) es un modelo autoregresivo que genera texto token por token.

### Generación de Texto

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar modelo y tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Texto inicial (prompt)
prompt = "The future of artificial intelligence is"

# Tokenizar
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generar texto
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,  # Creatividad (0=determinístico, 1=más aleatorio)
    top_k=50,         # Considerar top-k tokens
    top_p=0.95,       # Nucleus sampling
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decodificar
texto_generado = tokenizer.decode(output[0], skip_special_tokens=True)
print(texto_generado)
```

### Parámetros de Generación

| Parámetro | Descripción |
| :--- | :--- |
| `max_length` | Longitud máxima del texto generado |
| `temperature` | Controla aleatoriedad (0=determinístico, >1=más creativo) |
| `top_k` | Muestrear de los k tokens más probables |
| `top_p` | Nucleus sampling - muestrear del núcleo de probabilidad p |
| `num_beams` | Beam search para mejor calidad |
| `repetition_penalty` | Penalizar repeticiones |

---

## 7.6. Modelos en Español

### BETO (BERT en Español)

```python
from transformers import AutoTokenizer, AutoModel

# BETO - BERT entrenado en español
tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

texto = "El procesamiento de lenguaje natural es muy útil."
inputs = tokenizer(texto, return_tensors='pt')
outputs = model(**inputs)
```

### RoBERTuito (Tweets en Español)

```python
from transformers import pipeline

# Modelo entrenado en tweets en español
sentiment = pipeline(
    'sentiment-analysis',
    model='pysentimiento/robertuito-sentiment-analysis'
)

textos = [
    "Me encanta este producto, es genial!",
    "Qué mal servicio, no lo recomiendo",
    "El paquete llegó bien"
]

for texto in textos:
    result = sentiment(texto)[0]
    print(f"'{texto}' → {result['label']} ({result['score']:.3f})")
```

---

## 7.7. Sentence Transformers

Para obtener embeddings de oraciones completas que capturen el significado semántico.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo de sentence embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Oraciones a comparar
oraciones = [
    "El gato está sentado en el sofá",
    "Un felino descansa en el mueble",
    "El perro corre por el parque",
    "Machine learning es inteligencia artificial"
]

# Obtener embeddings
embeddings = model.encode(oraciones)

# Calcular similitudes
similitudes = cosine_similarity(embeddings)

print("Matriz de Similitud:")
for i, s1 in enumerate(oraciones):
    for j, s2 in enumerate(oraciones):
        if i < j:
            print(f"'{s1[:30]}...' vs '{s2[:30]}...'")
            print(f"  Similitud: {similitudes[i][j]:.3f}")
            print()
```

### Búsqueda Semántica

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Base de conocimiento
documentos = [
    "Python es un lenguaje de programación versátil",
    "Machine learning permite a las computadoras aprender de datos",
    "El deep learning usa redes neuronales profundas",
    "Los gatos son animales domésticos populares",
    "El fútbol es el deporte más popular del mundo"
]

# Codificar documentos
doc_embeddings = model.encode(documentos, convert_to_tensor=True)

# Query de búsqueda
query = "¿Qué es la inteligencia artificial?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Buscar documentos similares
scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_results = scores.argsort(descending=True)[:3]

print(f"Query: '{query}'")
print("\nResultados:")
for idx in top_results:
    print(f"  [{scores[idx]:.3f}] {documentos[idx]}")
```

---

## 7.8. Large Language Models (LLMs)

Los LLMs son modelos con miles de millones de parámetros entrenados en enormes cantidades de texto.

### Modelos Populares

| Modelo | Organización | Parámetros | Características |
| :--- | :--- | :--- | :--- |
| GPT-4 | OpenAI | ~1.8T | Multimodal, mejor razonamiento |
| Claude | Anthropic | ~100B+ | Conversacional, seguro |
| LLaMA 2 | Meta | 7B-70B | Open source |
| Mistral | Mistral AI | 7B | Eficiente, open source |
| Gemini | Google | ? | Multimodal |

### Usando LLMs con la API de OpenAI

```python
from openai import OpenAI

client = OpenAI(api_key="tu-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Eres un experto en NLP."},
        {"role": "user", "content": "Explica qué es el mecanismo de atención."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Usando Modelos Open Source (Ollama)

```python
import ollama

response = ollama.chat(model='llama2', messages=[
    {'role': 'user', 'content': '¿Qué es el procesamiento de lenguaje natural?'}
])

print(response['message']['content'])
```

---

## 7.9. Fine-tuning y Técnicas Avanzadas

### Parameter-Efficient Fine-Tuning (PEFT)

Técnicas para adaptar modelos grandes sin entrenar todos los parámetros:

*   **LoRA (Low-Rank Adaptation):** Añade matrices de bajo rango a las capas.
*   **Prefix Tuning:** Añade tokens aprendibles al inicio.
*   **Prompt Tuning:** Aprende embeddings de prompt.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Configuración LoRA
lora_config = LoraConfig(
    r=16,              # Rango de las matrices
    lora_alpha=32,     # Factor de escalado
    target_modules=["q_proj", "v_proj"],  # Módulos a adaptar
    lora_dropout=0.1,
    bias="none"
)

# Cargar modelo y aplicar LoRA
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
peft_model = get_peft_model(model, lora_config)

# Ver parámetros entrenables
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.0622
```

---

## 7.10. Consideraciones Prácticas

### Recursos y Costos

| Modelo | RAM GPU | Tiempo Inferencia |
| :--- | :--- | :--- |
| BERT-base | ~1GB | ~ms |
| GPT-2 | ~2GB | ~ms |
| LLaMA-7B | ~14GB | ~segundos |
| LLaMA-70B | ~140GB | ~segundos |

### Mejores Prácticas

1.  **Empezar simple:** Usar modelos más pequeños primero.
2.  **Cuantización:** Reducir precisión para menor uso de memoria (int8, int4).
3.  **Batching:** Procesar múltiples entradas juntas.
4.  **Caching:** Cachear embeddings de documentos.
5.  **Distillation:** Crear modelos más pequeños que imiten a los grandes.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
