# 💬 Unidad 1. Fundamentos del Procesamiento de Lenguaje Natural

El **Procesamiento de Lenguaje Natural (NLP)** es una disciplina en la intersección de la lingüística, la informática y la inteligencia artificial. Su objetivo es permitir que las máquinas comprendan, interpreten y generen lenguaje humano de forma útil.


![Ilustración de nlp basics](../assets/images/nlp_basics.svg)
---

## 1.1. ¿Qué es el NLP?

El NLP abarca un amplio rango de tareas computacionales que involucran el lenguaje humano:

*   **Comprensión:** Extraer significado del texto (clasificación, extracción de entidades, análisis de sentimientos).
*   **Generación:** Crear texto coherente (chatbots, resúmenes, traducción).
*   **Transformación:** Convertir texto de una forma a otra (corrección ortográfica, parafraseo).

### ¿Por qué es difícil?

El lenguaje humano presenta desafíos únicos para las máquinas:

*   **Ambigüedad:** Una palabra puede tener múltiples significados ("banco" puede ser una institución financiera o un asiento).
*   **Contexto:** El significado cambia según el contexto ("Hace frío" vs "Es un tipo frío").
*   **Variabilidad:** Sinónimos, jerga, errores ortográficos, diferentes idiomas.
*   **Conocimiento implícito:** Los humanos entienden ironía, sarcasmo y referencias culturales.

---

## 1.2. Pipeline Típico de NLP

Un proyecto de NLP generalmente sigue estos pasos:

```
Texto Crudo → Preprocesamiento → Representación Vectorial → Modelo ML/DL → Salida
```

1.  **Adquisición de Datos:** Obtener el corpus de texto (scraping, APIs, datasets públicos).
2.  **Preprocesamiento:** Limpiar y normalizar el texto.
3.  **Representación:** Convertir texto a números (vectores).
4.  **Modelado:** Aplicar algoritmos de ML o Deep Learning.
5.  **Evaluación:** Medir el rendimiento con métricas apropiadas.
6.  **Despliegue:** Poner el modelo en producción.

---

## 1.3. Preprocesamiento de Texto

El preprocesamiento es crucial para reducir el ruido y normalizar el texto.

### Tokenización

Dividir el texto en unidades más pequeñas llamadas **tokens** (palabras, subpalabras o caracteres).

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

texto = "El NLP es fascinante. Permite crear chatbots increíbles."

# Tokenización por oraciones
oraciones = sent_tokenize(texto, language='spanish')
print(oraciones)
# ['El NLP es fascinante.', 'Permite crear chatbots increíbles.']

# Tokenización por palabras
palabras = word_tokenize(texto, language='spanish')
print(palabras)
# ['El', 'NLP', 'es', 'fascinante', '.', 'Permite', 'crear', 'chatbots', 'increíbles', '.']
```

### Normalización de Texto

*   **Lowercasing:** Convertir todo a minúsculas para que "Casa" y "casa" sean iguales.
*   **Eliminación de puntuación y caracteres especiales.**
*   **Eliminación de números** (si no son relevantes).
*   **Eliminación de espacios extra.**

```python
import re

def normalizar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    # Eliminar caracteres especiales (mantener letras, números y espacios)
    texto = re.sub(r'[^a-záéíóúüñ0-9\s]', '', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

print(normalizar_texto("¡Hola! ¿Cómo estás???  Muy   bien."))
# 'hola cómo estás muy bien'
```

### Stop Words

Las **stop words** son palabras muy comunes que generalmente no aportan significado semántico (artículos, preposiciones, conjunciones).

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))
print(list(stop_words)[:10])
# ['al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra', 'cual']

palabras_filtradas = [w for w in palabras if w.lower() not in stop_words]
print(palabras_filtradas)
```

### Stemming vs Lematización

Ambas técnicas reducen las palabras a su forma base, pero de forma diferente:

*   **Stemming:** Corta la palabra de forma heurística para obtener su raíz. Es rápido pero puede producir raíces que no son palabras reales ("correr", "corriendo" → "corr").

```python
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
palabras = ["corriendo", "corrí", "correr", "corredores"]
stems = [stemmer.stem(p) for p in palabras]
print(stems)
# ['corr', 'corr', 'corr', 'corred']
```

*   **Lematización:** Utiliza un diccionario para encontrar el **lema** (forma canónica) de la palabra. Es más preciso pero más lento ("corriendo" → "correr").

```python
import spacy

nlp = spacy.load('es_core_news_sm')
doc = nlp("Los perros estaban corriendo por el parque")
lemmas = [token.lemma_ for token in doc]
print(lemmas)
# ['el', 'perro', 'estar', 'correr', 'por', 'el', 'parque']
```

---

## 1.4. Bibliotecas Fundamentales de NLP

| Biblioteca | Descripción | Fortalezas |
| :--- | :--- | :--- |
| **NLTK** | Natural Language Toolkit. La biblioteca clásica para aprender NLP. | Educativa, muchos recursos, corpus incluidos. |
| **spaCy** | Biblioteca industrial de NLP. | Rápida, pipelines preentrenados, NER, POS. |
| **Transformers (Hugging Face)** | Modelos de lenguaje preentrenados (BERT, GPT, etc.). | Estado del arte, fácil uso, muchos modelos. |
| **Gensim** | Especializada en modelado de tópicos y embeddings. | Word2Vec, Doc2Vec, LDA. |
| **TextBlob** | Interfaz simple para tareas comunes de NLP. | Fácil de usar, análisis de sentimientos. |

### Ejemplo Completo con spaCy

```python
import spacy

# Cargar modelo en español
nlp = spacy.load('es_core_news_sm')

texto = "Apple está buscando comprar una startup del Reino Unido por 1.000 millones de dólares."
doc = nlp(texto)

# Tokenización automática
print("Tokens:", [token.text for token in doc])

# Part-of-Speech Tagging (Etiquetado gramatical)
print("\nPOS Tagging:")
for token in doc:
    print(f"  {token.text}: {token.pos_} ({token.dep_})")

# Named Entity Recognition (NER)
print("\nEntidades:")
for ent in doc.ents:
    print(f"  {ent.text}: {ent.label_}")
# Apple: ORG, Reino Unido: LOC, 1.000 millones de dólares: MONEY
```

---

## 1.5. Tareas Comunes de NLP

El campo del NLP abarca muchas tareas específicas:

### Clasificación de Texto

Asignar una categoría a un documento.

*   **Análisis de Sentimientos:** Positivo / Negativo / Neutral.
*   **Detección de Spam:** Spam / No Spam.
*   **Clasificación de Noticias:** Deportes / Política / Tecnología.

### Extracción de Información

*   **Named Entity Recognition (NER):** Identificar personas, lugares, organizaciones, fechas.
*   **Extracción de Relaciones:** Encontrar relaciones entre entidades ("Apple" - adquirió - "Startup").

### Generación de Texto

*   **Resumen Automático:** Crear versiones cortas de textos largos.
*   **Traducción Automática:** Convertir texto entre idiomas.
*   **Chatbots:** Generar respuestas conversacionales.
*   **Completado de Texto:** GPT, modelos de lenguaje.

### Similitud y Búsqueda

*   **Búsqueda Semántica:** Encontrar documentos similares por significado.
*   **Question Answering (QA):** Responder preguntas basándose en un contexto.

---

## 1.6. Datasets Populares en NLP

| Dataset | Descripción | Tarea |
| :--- | :--- | :--- |
| **IMDB Reviews** | 50,000 reseñas de películas | Análisis de sentimientos |
| **AG News** | 120,000 artículos de noticias | Clasificación de texto |
| **SQuAD** | Preguntas sobre artículos de Wikipedia | Question Answering |
| **GLUE / SuperGLUE** | Benchmark de múltiples tareas | Evaluación de modelos |
| **CoNLL-2003** | Corpus anotado con entidades | NER |

---

## 1.7. Consideraciones Éticas en NLP

El NLP plantea importantes cuestiones éticas:

*   **Sesgo en los Datos:** Los modelos pueden perpetuar sesgos de género, raza o cultura presentes en los datos de entrenamiento.
*   **Privacidad:** Los modelos entrenados pueden memorizar información sensible.
*   **Desinformación:** La generación de texto puede usarse para crear fake news.
*   **Interpretabilidad:** Los modelos de Deep Learning son "cajas negras".

Es responsabilidad del profesional ser consciente de estos riesgos y tomar medidas para mitigarlos.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
