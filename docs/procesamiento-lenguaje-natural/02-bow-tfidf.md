# 💬 Unidad 2. Representación de Texto: BoW y TF-IDF

Para que los algoritmos de Machine Learning puedan procesar texto, necesitamos convertir las palabras en **vectores numéricos**. Esta unidad cubre las técnicas clásicas de representación de texto: **Bag of Words (BoW)** y **TF-IDF**.

---

## 2.1. El Problema de la Representación

Los algoritmos de ML trabajan con números, pero el texto es:

*   **Discreto:** Las palabras son símbolos, no números.
*   **Variable:** Los documentos tienen diferentes longitudes.
*   **Disperso:** El vocabulario puede ser muy grande.

La pregunta clave es: **¿Cómo convertimos texto en vectores numéricos que capturen su significado?**

---

## 2.2. Bag of Words (BoW)

El modelo **Bag of Words** (Bolsa de Palabras) es la representación más simple. Convierte un documento en un vector donde cada dimensión representa una palabra del vocabulario, y el valor es la **frecuencia** de esa palabra en el documento.

### Características

*   **Ignora el orden:** "El perro muerde al hombre" y "El hombre muerde al perro" tendrían la misma representación.
*   **Ignora la gramática:** Solo cuenta palabras.
*   **Vector disperso:** La mayoría de los valores son 0 (la mayoría de las palabras del vocabulario no aparecen en un documento dado).

### Proceso de Construcción

1.  **Crear el vocabulario:** Lista de todas las palabras únicas en el corpus.
2.  **Vectorizar:** Para cada documento, contar la frecuencia de cada palabra del vocabulario.

### Ejemplo Manual

```
Corpus:
- Doc1: "el gato come pescado"
- Doc2: "el perro come carne"
- Doc3: "el gato y el perro juegan"

Vocabulario: [el, gato, come, pescado, perro, carne, y, juegan]

Matriz BoW:
         el  gato  come  pescado  perro  carne  y  juegan
Doc1:     1    1     1      1       0      0    0    0
Doc2:     1    0     1      0       1      1    0    0
Doc3:     2    1     0      0       1      0    1    1
```

### Implementación con scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato y el perro juegan"
]

# Crear vectorizador
vectorizer = CountVectorizer()

# Ajustar y transformar
X = vectorizer.fit_transform(corpus)

# Ver vocabulario
print("Vocabulario:", vectorizer.get_feature_names_out())
# ['carne' 'come' 'el' 'gato' 'juegan' 'perro' 'pescado']

# Ver matriz (convertida a array denso para visualización)
print("\nMatriz BoW:\n", X.toarray())
# [[0 1 1 1 0 0 1]
#  [1 1 1 0 0 1 0]
#  [0 0 2 1 1 1 0]]
```

### Parámetros Importantes de CountVectorizer

| Parámetro | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `max_features` | Limita el vocabulario a las N palabras más frecuentes | `max_features=1000` |
| `stop_words` | Elimina stop words | `stop_words='english'` o lista personalizada |
| `ngram_range` | Incluye n-gramas (secuencias de N palabras) | `ngram_range=(1, 2)` para unigramas y bigramas |
| `min_df` | Ignora palabras que aparecen en menos de N documentos | `min_df=2` |
| `max_df` | Ignora palabras que aparecen en más del X% de documentos | `max_df=0.95` |
| `binary` | Solo indica presencia (1) o ausencia (0), no frecuencia | `binary=True` |

### N-gramas

Los **n-gramas** son secuencias de N palabras consecutivas. Permiten capturar algo de contexto y orden.

*   **Unigrama (n=1):** "machine", "learning"
*   **Bigrama (n=2):** "machine learning"
*   **Trigrama (n=3):** "natural language processing"

```python
# Incluir bigramas además de unigramas
vectorizer_ngram = CountVectorizer(ngram_range=(1, 2))
X_ngram = vectorizer_ngram.fit_transform(corpus)
print("Vocabulario con bigramas:", vectorizer_ngram.get_feature_names_out())
# ['carne', 'come', 'come carne', 'come pescado', 'el', 'el gato', ...]
```

---

## 2.3. Limitaciones de BoW

El modelo BoW tiene varias limitaciones importantes:

1.  **Todas las palabras tienen el mismo peso:** "el", "de", "que" pesan igual que palabras más informativas.
2.  **Alta dimensionalidad:** El vocabulario puede ser muy grande (miles o millones de palabras).
3.  **Vectores muy dispersos:** La mayoría de los elementos son 0.
4.  **Ignora el significado:** Palabras sinónimas tienen representaciones completamente diferentes.
5.  **Ignora el orden:** Pierde información contextual crucial.

---

## 2.4. TF-IDF (Term Frequency - Inverse Document Frequency)

**TF-IDF** mejora BoW al dar más peso a las palabras que son:

*   **Frecuentes en el documento** (Term Frequency - TF)
*   **Pero raras en el corpus general** (Inverse Document Frequency - IDF)

Esto penaliza palabras comunes como "el", "de", "que" y da más importancia a palabras distintivas.

### Fórmulas Matemáticas

**Term Frequency (TF):** Frecuencia de un término en un documento.

$$TF(t, d) = \frac{\text{Número de veces que } t \text{ aparece en } d}{\text{Total de términos en } d}$$

O simplemente el conteo crudo: $TF(t, d) = f_{t,d}$

**Inverse Document Frequency (IDF):** Mide la rareza de un término en el corpus.

$$IDF(t) = \log\left(\frac{N}{df_t}\right)$$

Donde:

*   $N$ = Número total de documentos
*   $df_t$ = Número de documentos que contienen el término $t$

**TF-IDF:** El producto de ambos.

$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

### Ejemplo Manual

```
Corpus (3 documentos):
- Doc1: "machine learning is fun"
- Doc2: "machine learning and deep learning"
- Doc3: "deep learning is powerful"

Palabra: "learning"
- TF en Doc1: 1/4 = 0.25
- TF en Doc2: 2/5 = 0.40  (aparece 2 veces)
- TF en Doc3: 1/4 = 0.25
- IDF: log(3/3) = log(1) = 0  (aparece en TODOS los docs → poco informativa)
- TF-IDF: 0 en todos los documentos

Palabra: "fun"
- TF en Doc1: 1/4 = 0.25
- IDF: log(3/1) = log(3) ≈ 1.1  (aparece solo en 1 doc → muy informativa)
- TF-IDF en Doc1: 0.25 × 1.1 ≈ 0.275
```

### Implementación con scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato y el perro juegan"
]

# Crear vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Ajustar y transformar
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Ver vocabulario
print("Vocabulario:", tfidf_vectorizer.get_feature_names_out())

# Ver matriz TF-IDF
import pandas as pd
df_tfidf = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)
print("\nMatriz TF-IDF:")
print(df_tfidf.round(3))
```

**Resultado:**

| | carne | come | el | gato | juegan | perro | pescado |
|---|---|---|---|---|---|---|---|
| Doc1 | 0.000 | 0.450 | 0.340 | 0.534 | 0.000 | 0.000 | 0.630 |
| Doc2 | 0.630 | 0.450 | 0.340 | 0.000 | 0.000 | 0.534 | 0.000 |
| Doc3 | 0.000 | 0.000 | 0.485 | 0.380 | 0.565 | 0.380 | 0.000 |

Observa cómo:

*   "pescado" y "carne" tienen valores altos (palabras distintivas de cada documento).
*   "el" tiene valores relativamente bajos (muy común).
*   "come" tiene valores moderados (aparece en 2 de 3 docs).

### Parámetros de TfidfVectorizer

| Parámetro | Descripción |
| :--- | :--- |
| `norm` | Normalización del vector ('l1', 'l2' o None) |
| `use_idf` | Si usar IDF (True por defecto) |
| `smooth_idf` | Suaviza IDF para evitar división por cero |
| `sublinear_tf` | Aplica logaritmo a TF (1 + log(TF)) |

---

## 2.5. Comparación BoW vs TF-IDF

| Característica | BoW | TF-IDF |
| :--- | :--- | :--- |
| **Valores** | Frecuencias crudas (enteros) | Pesos ponderados (decimales) |
| **Palabras comunes** | Alto peso (por frecuencia) | Bajo peso (penalizadas por IDF) |
| **Palabras distintivas** | Peso proporcional a frecuencia | Alto peso (frecuentes localmente, raras globalmente) |
| **Uso típico** | Baseline simple, conteo rápido | Clasificación de texto, búsqueda de documentos |

---

## 2.6. Ejemplo Práctico: Clasificación de Sentimientos

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Dataset de ejemplo
textos = [
    "me encanta esta película, es genial",
    "película increíble, la recomiendo mucho",
    "excelente actuación y gran historia",
    "la mejor película que he visto",
    "qué película tan mala y aburrida",
    "no me gustó nada, muy decepcionante",
    "terrible, la peor película del año",
    "muy aburrida, no la recomiendo"
]
etiquetas = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=positivo, 0=negativo

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    textos, etiquetas, test_size=0.25, random_state=42
)

# Vectorizar con TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar clasificador Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predecir y evaluar
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

# Probar con texto nuevo
nuevo_texto = ["esta película es fantástica"]
nuevo_tfidf = vectorizer.transform(nuevo_texto)
prediccion = clf.predict(nuevo_tfidf)
print(f"\nPredicción para '{nuevo_texto[0]}': {'Positivo' if prediccion[0] else 'Negativo'}")
```

---

## 2.7. Aplicaciones Reales

*   **Motores de Búsqueda:** TF-IDF es fundamental para ranking de documentos relevantes (aunque modernos usan técnicas más avanzadas).
*   **Sistemas de Recomendación de Contenido:** Encontrar artículos o noticias similares.
*   **Clasificación de Documentos:** Categorizar emails, tickets de soporte, documentos legales.
*   **Detección de Plagio:** Comparar similitud entre documentos.

---

## 2.8. Limitaciones y Próximos Pasos

Aunque BoW y TF-IDF son útiles, tienen limitaciones importantes:

*   **No capturan semántica:** "bueno" y "excelente" son vectores completamente diferentes.
*   **No capturan contexto:** El significado de una palabra puede cambiar según el contexto.
*   **Alta dimensionalidad:** Vocabularios grandes generan vectores enormes.

Las técnicas modernas como **Word Embeddings** (Word2Vec, GloVe) y **Transformers** (BERT, GPT) resuelven muchas de estas limitaciones al aprender representaciones densas y semánticas.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
