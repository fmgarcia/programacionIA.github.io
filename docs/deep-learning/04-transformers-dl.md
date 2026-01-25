# 🤖 Unidad 4. Arquitectura Transformer

Los **Transformers** (Vaswani et al., 2017) revolucionaron el campo del Deep Learning al introducir el mecanismo de **atención** como único componente para modelar secuencias, eliminando la necesidad de recurrencia.

---

## 4.1. Motivación: Limitaciones de las RNN

| Limitación RNN | Solución Transformer |
| :--- | :--- |
| Procesamiento secuencial (no paralelizable) | Procesamiento paralelo completo |
| Dificultad con dependencias largas | Atención directa entre cualquier par de posiciones |
| Gradientes que se desvanecen | Conexiones directas mediante atención |

---

## 4.2. Mecanismo de Atención

### ¿Qué es la Atención?

La atención permite que cada elemento de una secuencia "preste atención" a todos los demás elementos, ponderando su importancia.

**Intuición:**

```
Frase: "El gato negro que vive en mi casa está durmiendo en el sofá"
                 ↑                              ↑
              "está" presta más atención a "gato" que a "casa"
```

### Scaled Dot-Product Attention

La atención se calcula usando tres vectores para cada elemento:

*   **Query (Q):** Lo que estamos buscando.
*   **Key (K):** Etiqueta de cada elemento.
*   **Value (V):** El contenido real.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde $d_k$ es la dimensión de las keys (para estabilidad numérica).

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calcula la atención escalada por producto punto.
    
    Q: queries (batch, seq_len, d_k)
    K: keys (batch, seq_len, d_k)
    V: values (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Calcular scores
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Aplicar máscara si existe
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Softmax para obtener pesos de atención
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Multiplicar por valores
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
```

### Multi-Head Attention

En lugar de una sola atención, usamos **múltiples "cabezas"** que aprenden diferentes tipos de relaciones:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Donde cada cabeza es:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Divide la última dimensión en (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        scaled_attention = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = scaled_attention / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)
```

---

## 4.3. Arquitectura Completa del Transformer

### Diagrama General

```
         ┌─────────────────────────────────────┐
         │           DECODER                    │
         │  ┌───────────────────────────────┐  │
         │  │   Linear + Softmax → Output   │  │
         │  └───────────────────────────────┘  │
         │              ↑                       │
         │  ┌───────────────────────────────┐  │
         │  │   Multi-Head Attention        │  │ ← Cross-attention
         │  │   (queries del decoder,       │  │    (conecta encoder
         │  │    keys/values del encoder)   │  │     y decoder)
         │  └───────────────────────────────┘  │
         │              ↑                       │
         │  ┌───────────────────────────────┐  │
         │  │   Masked Multi-Head           │  │
         │  │   Self-Attention              │  │
         │  └───────────────────────────────┘  │
         │              ↑                       │
         │  ┌───────────────────────────────┐  │
         │  │   Output Embedding +          │  │
         │  │   Positional Encoding         │  │
         │  └───────────────────────────────┘  │
         └─────────────────────────────────────┘
                        ↑
         ┌─────────────────────────────────────┐
         │           ENCODER                    │
         │  ┌───────────────────────────────┐  │
         │  │   Feed Forward Neural Net     │  │
         │  └───────────────────────────────┘  │
         │              ↑                       │
         │  ┌───────────────────────────────┐  │
         │  │   Multi-Head Self-Attention   │  │
         │  └───────────────────────────────┘  │
         │              ↑                       │
         │  ┌───────────────────────────────┐  │
         │  │   Input Embedding +           │  │
         │  │   Positional Encoding         │  │
         │  └───────────────────────────────┘  │
         └─────────────────────────────────────┘
                        ↑
                   Input Tokens
```

---

## 4.4. Componentes del Transformer

### Positional Encoding

Como no hay recurrencia, necesitamos inyectar información de posición:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

```python
def positional_encoding(position, d_model):
    """Genera positional encodings."""
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    # Seno a índices pares
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Coseno a índices impares
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates
```

### Feed-Forward Network

Después de la atención, cada posición pasa por una red feed-forward:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

```python
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
```

### Layer Normalization y Residual Connections

Cada sublayer tiene:

1.  **Residual Connection:** $\text{output} = x + \text{Sublayer}(x)$
2.  **Layer Normalization:** Normaliza las activaciones.

```python
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        # Multi-head attention + residual + norm
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward + residual + norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
```

---

## 4.5. Self-Attention vs Cross-Attention

### Self-Attention

Q, K, V vienen de la **misma** secuencia:

```python
# En el encoder
output = self_attention(x, x, x)  # Q=K=V=x
```

### Cross-Attention

Q viene del decoder, K y V del encoder:

```python
# En el decoder (después del self-attention)
output = cross_attention(
    q=decoder_output,
    k=encoder_output,
    v=encoder_output
)
```

### Masked Self-Attention

En el decoder, cada posición solo puede atender a posiciones anteriores (para generación autoregresiva):

```python
def create_look_ahead_mask(size):
    """Máscara triangular para evitar mirar el futuro."""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)
```

---

## 4.6. Implementación Completa de un Transformer

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, LayerNormalization

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 input_vocab_size, target_vocab_size, 
                 pe_input, pe_target, rate=0.1):
        super().__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        
        self.final_layer = Dense(target_vocab_size)
    
    def call(self, inputs, training):
        inp, tar = inputs
        
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(tar, enc_output, training, 
                                   combined_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        return final_output

class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        self.dropout = Dropout(rate)
    
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, mask)
        
        return x

class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 vocab_size, maximum_position_encoding, rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        self.dropout = Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)
        
        return x
```

---

## 4.7. Variantes de Transformers

### Encoder-Only (BERT)

Solo usa el encoder. Ideal para:

*   Clasificación de texto.
*   NER.
*   Question Answering extractivo.

```python
# Usando Hugging Face
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model(**inputs)
# outputs.last_hidden_state: (batch, seq_len, hidden_size)
```

### Decoder-Only (GPT)

Solo usa el decoder con masked self-attention. Ideal para:

*   Generación de texto.
*   Completar oraciones.
*   Modelos de lenguaje.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode("Once upon a time", return_tensors='pt')
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))
```

### Encoder-Decoder (T5, BART)

Arquitectura completa. Ideal para:

*   Traducción.
*   Resumen.
*   Question Answering generativo.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Traducción
input_text = "translate English to German: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4.8. Ejemplo Práctico: Fine-tuning de BERT

```python
from transformers import (
    BertTokenizer, 
    TFBertForSequenceClassification,
    create_optimizer
)
import tensorflow as tf

# Cargar modelo y tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Datos de ejemplo
textos = [
    "I love this movie!",
    "This film was terrible.",
    "Great acting and plot.",
    "Waste of time."
]
etiquetas = [1, 0, 1, 0]  # 1=positivo, 0=negativo

# Tokenizar
encodings = tokenizer(
    textos,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='tf'
)

# Crear dataset
dataset = tf.data.Dataset.from_tensor_slices((
    dict(encodings),
    etiquetas
)).batch(2)

# Configurar optimizador
batch_size = 2
num_train_steps = len(textos) // batch_size * 3  # 3 épocas
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=0
)

# Compilar
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Entrenar
model.fit(dataset, epochs=3)

# Predecir
nuevo_texto = "Amazing film, highly recommended!"
inputs = tokenizer(nuevo_texto, return_tensors='tf', truncation=True, padding=True)
outputs = model(inputs)
prediccion = tf.nn.softmax(outputs.logits, axis=-1)
print(f"Probabilidades: {prediccion.numpy()}")
```

---

## 4.9. Comparación de Arquitecturas

| Característica | RNN/LSTM | Transformer |
| :--- | :--- | :--- |
| Paralelización | Secuencial | Total |
| Dependencias largas | Difícil | Fácil (atención directa) |
| Complejidad computacional | $O(n)$ | $O(n^2)$ |
| Memoria | $O(1)$ por paso | $O(n^2)$ para atención |
| Preentrenamiento | Limitado | Muy efectivo |

---

## 4.10. Tendencias Actuales

### Efficient Transformers

Para reducir la complejidad $O(n^2)$:

*   **Sparse Attention:** Solo atiende a un subconjunto de posiciones.
*   **Linear Attention:** Aproximaciones lineales.
*   **Longformer, BigBird:** Para secuencias muy largas.

### Large Language Models (LLMs)

*   **GPT-4, Claude, Llama:** Miles de millones de parámetros.
*   **Instruction Tuning:** Afinados para seguir instrucciones.
*   **RLHF:** Entrenamiento con feedback humano.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
