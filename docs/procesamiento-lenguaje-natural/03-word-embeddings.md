# 💬 Unidad 3. Word Embeddings: Word2Vec y GloVe

Los **Word Embeddings** (incrustaciones de palabras) son representaciones vectoriales densas que capturan el significado semántico de las palabras. A diferencia de BoW/TF-IDF, los embeddings colocan palabras con significados similares cerca en el espacio vectorial.

---

## 3.1. Limitaciones de BoW/TF-IDF

Los métodos clásicos tienen problemas fundamentales:

*   **Vectores dispersos:** Un vocabulario de 50,000 palabras genera vectores de 50,000 dimensiones, la mayoría con valor 0.
*   **Sin relación semántica:** "feliz" y "contento" tienen representaciones completamente diferentes.
*   **Alta dimensionalidad:** Costoso en memoria y computación.
*   **Sin generalización:** El modelo no puede inferir similitudes entre palabras no vistas.

### La Idea de los Embeddings

**"Una palabra se conoce por la compañía que tiene"** - J.R. Firth, 1957

Los embeddings aprenden representaciones **densas** (típicamente 50-300 dimensiones) donde:

*   Palabras similares tienen vectores similares.
*   Las relaciones semánticas se capturan como operaciones vectoriales.
*   El famoso ejemplo: `vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")`

---

## 3.2. Word2Vec

**Word2Vec** es un modelo desarrollado por Google en 2013 que aprende embeddings de palabras usando redes neuronales superficiales. Existen dos arquitecturas principales:

### 3.2.1. Skip-gram

Dado una palabra del centro (target), predice las palabras del contexto (alrededor).

```
Oración: "El gato come pescado fresco"
Ventana de contexto = 2

Si target = "come":
- Predice: "gato", "pescado" (contexto)
```

**Intuición:** Si dos palabras aparecen frecuentemente en contextos similares, tendrán embeddings similares.

### 3.2.2. CBOW (Continuous Bag of Words)

Lo opuesto a Skip-gram: dado el contexto (palabras alrededor), predice la palabra del centro.

```
Contexto: ["el", "gato", "pescado", "fresco"]
Predice: "come"
```

### Comparación

| Característica | Skip-gram | CBOW |
| :--- | :--- | :--- |
| **Predice** | Contexto dado palabra | Palabra dado contexto |
| **Rendimiento** | Mejor con palabras raras | Mejor con palabras frecuentes |
| **Velocidad** | Más lento | Más rápido |
| **Datos pequeños** | Mejor | Peor |

### Implementación con Gensim

```python
from gensim.models import Word2Vec

# Corpus de ejemplo (lista de listas de tokens)
corpus = [
    ["el", "gato", "come", "pescado"],
    ["el", "perro", "come", "carne"],
    ["el", "gato", "duerme", "mucho"],
    ["el", "perro", "corre", "rápido"],
    ["el", "pájaro", "vuela", "alto"]
]

# Entrenar modelo Word2Vec
model = Word2Vec(
    sentences=corpus,
    vector_size=100,    # Dimensiones del embedding
    window=5,           # Tamaño de ventana de contexto
    min_count=1,        # Frecuencia mínima de palabra
    workers=4,          # Threads para entrenamiento
    sg=1                # 1 = Skip-gram, 0 = CBOW
)

# Obtener el vector de una palabra
vector_gato = model.wv['gato']
print(f"Dimensiones: {vector_gato.shape}")  # (100,)
print(f"Vector 'gato': {vector_gato[:5]}...")  # Primeros 5 valores

# Palabras más similares
similares = model.wv.most_similar('gato', topn=3)
print(f"\nPalabras similares a 'gato': {similares}")

# Similitud entre dos palabras
similitud = model.wv.similarity('gato', 'perro')
print(f"Similitud gato-perro: {similitud:.4f}")
```

### Hiperparámetros Importantes

| Parámetro | Descripción | Valor típico |
| :--- | :--- | :--- |
| `vector_size` | Dimensiones del embedding | 100-300 |
| `window` | Palabras de contexto a considerar | 5-10 |
| `min_count` | Frecuencia mínima para incluir palabra | 5 |
| `sg` | 0=CBOW, 1=Skip-gram | 1 para datos pequeños |
| `negative` | Muestras negativas (optimización) | 5-20 |
| `epochs` | Iteraciones sobre el corpus | 5-15 |

---

## 3.3. GloVe (Global Vectors)

**GloVe** (Stanford, 2014) es otro método popular para crear embeddings. A diferencia de Word2Vec que aprende de ventanas locales, GloVe utiliza estadísticas globales de co-ocurrencia del corpus.

### Diferencia con Word2Vec

*   **Word2Vec:** Aprende de predicción local (ventana de contexto).
*   **GloVe:** Construye una matriz de co-ocurrencia global y factoriza esa matriz.

### Usando GloVe Preentrenado

GloVe proporciona embeddings preentrenados en grandes corpus (Wikipedia, Twitter, Common Crawl).

```python
import numpy as np

# Cargar embeddings GloVe preentrenados (descargar primero)
def cargar_glove(path, dim=100):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Ejemplo de uso (requiere descargar glove.6B.100d.txt)
# glove = cargar_glove('glove.6B.100d.txt')
# print(glove['king'].shape)  # (100,)
```

### Comparación Word2Vec vs GloVe

| Característica | Word2Vec | GloVe |
| :--- | :--- | :--- |
| **Método** | Predictivo (red neuronal) | Factorización de matriz |
| **Información** | Local (ventana) | Global (co-ocurrencia) |
| **Entrenamiento** | Incremental posible | Requiere todo el corpus |
| **Rendimiento** | Competitivo | Competitivo |

---

## 3.4. Operaciones con Embeddings

Una propiedad fascinante de los embeddings es que las relaciones semánticas se capturan como operaciones vectoriales.

### Analogías

```python
# rey - hombre + mujer ≈ reina
result = model.wv.most_similar(
    positive=['rey', 'mujer'],
    negative=['hombre'],
    topn=1
)
# [('reina', 0.85)]

# París - Francia + España ≈ Madrid
result = model.wv.most_similar(
    positive=['paris', 'españa'],
    negative=['francia'],
    topn=1
)
# [('madrid', 0.82)]
```

### Similitud Coseno

La similitud entre embeddings se calcula típicamente con **similitud coseno**:

$$\text{similitud}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$$

Valores cercanos a 1 indican alta similitud, cercanos a 0 indican poca relación.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Calcular similitud entre dos vectores
vec1 = model.wv['gato']
vec2 = model.wv['perro']

similitud = cosine_similarity([vec1], [vec2])[0][0]
print(f"Similitud coseno: {similitud:.4f}")
```

---

## 3.5. Embeddings Preentrenados

En la práctica, es común usar embeddings preentrenados en grandes corpus:

### Fuentes Populares

| Nombre | Corpus de Entrenamiento | Dimensiones |
| :--- | :--- | :--- |
| **Word2Vec (Google)** | Google News (100B palabras) | 300 |
| **GloVe (Stanford)** | Wikipedia + Gigaword | 50, 100, 200, 300 |
| **FastText (Facebook)** | Wikipedia + Common Crawl | 300 |

### Usando FastText (mejor para palabras OOV)

**FastText** extiende Word2Vec al considerar sub-palabras (n-gramas de caracteres). Esto permite generar embeddings para palabras no vistas (Out-of-Vocabulary).

```python
import fasttext.util

# Descargar modelo preentrenado en español
fasttext.util.download_model('es', if_exists='ignore')
ft = fasttext.load_model('cc.es.300.bin')

# Obtener embedding
vector = ft.get_word_vector('gato')

# Funciona con palabras no vistas (OOV)
vector_typo = ft.get_word_vector('gatito')  # Funciona!
```

---

## 3.6. Embeddings para Documentos

Los embeddings de palabras pueden extenderse a documentos completos:

### Promedio de Embeddings

La forma más simple: promediar los embeddings de todas las palabras.

```python
import numpy as np

def documento_a_vector(documento, model, dim=100):
    """Convierte un documento a vector promediando embeddings de palabras."""
    tokens = documento.lower().split()
    vectores = []
    
    for token in tokens:
        if token in model.wv:
            vectores.append(model.wv[token])
    
    if vectores:
        return np.mean(vectores, axis=0)
    else:
        return np.zeros(dim)

# Uso
doc = "el gato come pescado"
vec_doc = documento_a_vector(doc, model)
```

### Doc2Vec

**Doc2Vec** extiende Word2Vec para aprender embeddings de documentos directamente.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Preparar documentos etiquetados
documentos = [
    TaggedDocument(words=['el', 'gato', 'come', 'pescado'], tags=['doc1']),
    TaggedDocument(words=['el', 'perro', 'come', 'carne'], tags=['doc2']),
]

# Entrenar modelo
model_doc2vec = Doc2Vec(documentos, vector_size=50, window=2, min_count=1, epochs=100)

# Obtener vector de un documento
vector_doc1 = model_doc2vec.dv['doc1']

# Inferir vector para documento nuevo
nuevo_doc = ['el', 'gato', 'duerme']
vector_nuevo = model_doc2vec.infer_vector(nuevo_doc)
```

---

## 3.7. Ejemplo Práctico: Búsqueda Semántica

```python
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Corpus de documentos
documentos = [
    "el gato duerme en el sofá",
    "el perro corre por el parque",
    "el pájaro vuela sobre los árboles",
    "el pez nada en el acuario",
    "el gato caza ratones por la noche"
]

# Preparar corpus tokenizado
corpus_tokens = [doc.split() for doc in documentos]

# Entrenar Word2Vec
model = Word2Vec(corpus_tokens, vector_size=50, window=3, min_count=1, epochs=50)

def busqueda_semantica(query, documentos, model, topn=3):
    """Busca documentos similares a una query."""
    # Vector de la query
    query_tokens = query.lower().split()
    query_vecs = [model.wv[t] for t in query_tokens if t in model.wv]
    
    if not query_vecs:
        return []
    
    query_vec = np.mean(query_vecs, axis=0).reshape(1, -1)
    
    # Vectores de documentos
    doc_vecs = []
    for doc in documentos:
        tokens = doc.split()
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            doc_vecs.append(np.mean(vecs, axis=0))
        else:
            doc_vecs.append(np.zeros(50))
    
    doc_vecs = np.array(doc_vecs)
    
    # Calcular similitudes
    similitudes = cosine_similarity(query_vec, doc_vecs)[0]
    
    # Ordenar por similitud
    indices = similitudes.argsort()[::-1][:topn]
    
    return [(documentos[i], similitudes[i]) for i in indices]

# Buscar
query = "felino descansa"
resultados = busqueda_semantica(query, documentos, model)

print(f"Query: '{query}'")
print("\nResultados:")
for doc, score in resultados:
    print(f"  [{score:.3f}] {doc}")
```

---

## 3.8. Consideraciones y Limitaciones

### Ventajas de Word Embeddings

*   **Representación densa:** Vectores pequeños (100-300 dim) vs miles en BoW.
*   **Captura semántica:** Palabras similares están cerca en el espacio vectorial.
*   **Transferencia:** Embeddings preentrenados pueden usarse en múltiples tareas.

### Limitaciones

*   **Una representación por palabra:** "banco" (asiento) y "banco" (institución) tienen el mismo vector.
*   **Estáticos:** No cambian según el contexto de la oración.
*   **Requieren mucho texto:** Para entrenar buenos embeddings propios.
*   **Sesgos:** Pueden capturar sesgos presentes en los datos de entrenamiento.

Los modelos contextuales como **BERT** y **GPT** resuelven la limitación de representaciones estáticas generando embeddings diferentes para la misma palabra según su contexto.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
