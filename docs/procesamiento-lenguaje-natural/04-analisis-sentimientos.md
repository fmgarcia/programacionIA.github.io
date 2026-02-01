# 💬 Unidad 4. Análisis de Sentimientos

El **Análisis de Sentimientos** (Sentiment Analysis), también conocido como **minería de opiniones**, es una de las aplicaciones más populares del NLP. Consiste en determinar la actitud, emoción u opinión expresada en un texto.


![Ilustración de sentiment](../assets/images/sentiment.svg)
---

## 4.1. ¿Qué es el Análisis de Sentimientos?

Es el proceso computacional de identificar y categorizar opiniones expresadas en texto para determinar si la actitud del escritor hacia un tema es:

*   **Positiva:** "Me encanta este producto, es increíble"
*   **Negativa:** "Terrible servicio, muy decepcionado"
*   **Neutral:** "El producto llegó ayer"

### Niveles de Análisis

| Nivel | Descripción | Ejemplo |
| :--- | :--- | :--- |
| **Documento** | Sentimiento global del texto | Reseña completa de un producto |
| **Oración** | Sentimiento de cada oración | Cada frase de una reseña |
| **Aspecto** | Sentimiento sobre aspectos específicos | "La batería es excelente, pero la cámara es mala" |
| **Entidad** | Sentimiento hacia entidades específicas | Opiniones sobre diferentes marcas en un texto |

### Tipos de Salida

*   **Binaria:** Positivo / Negativo
*   **Ternaria:** Positivo / Neutral / Negativo
*   **Escala:** Rating numérico (1-5 estrellas)
*   **Emociones:** Alegría, Tristeza, Ira, Miedo, Sorpresa, Disgusto

---

## 4.2. Enfoques para Análisis de Sentimientos

### Enfoque Basado en Léxico

Utiliza diccionarios de palabras con polaridad predefinida.

**Ventajas:**

*   No requiere datos de entrenamiento.
*   Interpretable y transparente.
*   Funciona bien en dominios generales.

**Desventajas:**

*   No captura contexto ni negaciones bien.
*   Depende de la calidad del léxico.
*   Dificultad con sarcasmo e ironía.

```python
# Ejemplo simple de enfoque léxico
lexico_positivo = {'bueno', 'excelente', 'genial', 'increíble', 'mejor', 'feliz', 'encanta'}
lexico_negativo = {'malo', 'terrible', 'horrible', 'peor', 'triste', 'odio', 'decepcionante'}

def analisis_lexico(texto):
    tokens = texto.lower().split()
    score = 0
    
    for token in tokens:
        if token in lexico_positivo:
            score += 1
        elif token in lexico_negativo:
            score -= 1
    
    if score > 0:
        return "Positivo"
    elif score < 0:
        return "Negativo"
    else:
        return "Neutral"

# Ejemplos
print(analisis_lexico("Este producto es excelente y genial"))  # Positivo
print(analisis_lexico("Terrible servicio, muy malo"))  # Negativo
print(analisis_lexico("El paquete llegó ayer"))  # Neutral
```

### Enfoque Basado en Machine Learning

Entrena clasificadores supervisados con datos etiquetados.

**Proceso típico:**

1.  Recolectar datos etiquetados (reseñas con ratings).
2.  Preprocesar texto (tokenización, limpieza).
3.  Vectorizar (BoW, TF-IDF, embeddings).
4.  Entrenar clasificador (Naive Bayes, SVM, Logistic Regression).
5.  Evaluar y ajustar.

**Ventajas:**

*   Captura patrones complejos.
*   Adaptable a dominios específicos.
*   Mejor rendimiento general.

**Desventajas:**

*   Requiere datos etiquetados.
*   Puede no generalizar a otros dominios.

---

## 4.3. Implementación con Machine Learning

### Dataset: IMDB Movie Reviews

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Cargar datos (ejemplo simplificado)
# En la práctica, usar datasets como IMDB de Keras o Hugging Face
reseñas = [
    ("me encanta esta película, actuaciones increíbles", 1),
    ("película brillante, la recomiendo totalmente", 1),
    ("una historia hermosa y conmovedora", 1),
    ("excelente dirección y guión", 1),
    ("muy entretenida, la mejor del año", 1),
    ("qué película tan aburrida y larga", 0),
    ("terrible actuación, muy decepcionante", 0),
    ("perdí mi tiempo viendo esto", 0),
    ("la peor película que he visto", 0),
    ("no la recomiendo para nada, muy mala", 0),
]

textos = [r[0] for r in reseñas]
etiquetas = [r[1] for r in reseñas]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    textos, etiquetas, test_size=0.3, random_state=42
)

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entrenar múltiples modelos
modelos = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_tfidf, y_train)
    y_pred = modelo.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"{nombre}: Accuracy = {acc:.4f}")
```

---

## 4.4. Análisis de Sentimientos con Bibliotecas

### TextBlob (Rápido y Simple)

```python
from textblob import TextBlob

def analizar_sentimiento_textblob(texto):
    blob = TextBlob(texto)
    # Polaridad: -1 (negativo) a 1 (positivo)
    # Subjetividad: 0 (objetivo) a 1 (subjetivo)
    return {
        'texto': texto,
        'polaridad': blob.sentiment.polarity,
        'subjetividad': blob.sentiment.subjectivity,
        'sentimiento': 'Positivo' if blob.sentiment.polarity > 0 
                       else 'Negativo' if blob.sentiment.polarity < 0 
                       else 'Neutral'
    }

# Ejemplos
textos = [
    "I love this product, it's amazing!",
    "Terrible experience, very disappointed",
    "The package arrived yesterday"
]

for texto in textos:
    resultado = analizar_sentimiento_textblob(texto)
    print(f"{resultado['texto']}")
    print(f"  → {resultado['sentimiento']} (pol: {resultado['polaridad']:.2f})")
    print()
```

### VADER (Especializado en Redes Sociales)

**VADER** (Valence Aware Dictionary for Sentiment Reasoning) está específicamente diseñado para texto de redes sociales, manejando emojis, mayúsculas, puntuación, etc.

```python
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

textos = [
    "I LOVE this!!! 😍",
    "This is okay, nothing special",
    "Worst product EVER!!! 😡😡😡",
    "The movie was good, but the ending was bad"
]

for texto in textos:
    scores = sia.polarity_scores(texto)
    print(f"'{texto}'")
    print(f"  Scores: {scores}")
    print(f"  Compound: {scores['compound']:.3f}")
    print()
```

**Output de VADER:**

*   `neg`: Proporción negativa
*   `neu`: Proporción neutral
*   `pos`: Proporción positiva
*   `compound`: Score normalizado entre -1 y 1

---

## 4.5. Análisis de Sentimientos con Transformers

Los modelos basados en Transformers (BERT, RoBERTa) ofrecen el mejor rendimiento actual.

### Usando Hugging Face

```python
from transformers import pipeline

# Cargar pipeline de análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis")

# Análisis simple
textos = [
    "I love this movie, it's fantastic!",
    "This is the worst product I've ever bought",
    "The weather is nice today"
]

for texto in textos:
    resultado = sentiment_pipeline(texto)[0]
    print(f"'{texto}'")
    print(f"  → {resultado['label']} (score: {resultado['score']:.4f})")
    print()

# Para español, usar modelo específico
sentiment_es = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

texto_es = "Esta película es absolutamente maravillosa"
resultado_es = sentiment_es(texto_es)
print(f"Español: '{texto_es}' → {resultado_es}")
```

### Modelo Específico para Español

```python
from transformers import pipeline

# Modelo entrenado específicamente para español
classifier = pipeline(
    "text-classification",
    model="pysentimiento/robertuito-sentiment-analysis"
)

textos_es = [
    "Me encanta este restaurante, la comida es deliciosa",
    "El servicio fue terrible y tardaron mucho",
    "El pedido llegó a tiempo"
]

for texto in textos_es:
    resultado = classifier(texto)[0]
    print(f"'{texto}'")
    print(f"  → {resultado['label']} ({resultado['score']:.3f})")
```

---

## 4.6. Análisis de Sentimientos por Aspectos

El **Aspect-Based Sentiment Analysis (ABSA)** identifica sentimientos hacia aspectos específicos de un producto o servicio.

```
"La batería del teléfono dura mucho, pero la cámara es terrible"

Aspectos:
- batería → Positivo
- cámara → Negativo
```

### Implementación Simple

```python
import spacy

nlp = spacy.load('es_core_news_sm')

aspectos_positivos = {
    'batería': ['dura', 'excelente', 'increíble', 'buena'],
    'cámara': ['nítida', 'clara', 'buena', 'increíble'],
    'pantalla': ['brillante', 'clara', 'grande', 'hermosa'],
    'diseño': ['elegante', 'bonito', 'moderno', 'hermoso']
}

aspectos_negativos = {
    'batería': ['corta', 'mala', 'terrible', 'poco'],
    'cámara': ['borrosa', 'mala', 'terrible', 'pésima'],
    'pantalla': ['pequeña', 'oscura', 'mala'],
    'diseño': ['feo', 'malo', 'anticuado']
}

def absa_simple(texto):
    """Análisis de sentimiento por aspectos simple."""
    texto_lower = texto.lower()
    resultados = {}
    
    # Buscar aspectos y palabras de sentimiento cercanas
    for aspecto in aspectos_positivos.keys():
        if aspecto in texto_lower:
            sentimiento = "Neutral"
            
            # Buscar palabras positivas
            for palabra in aspectos_positivos[aspecto]:
                if palabra in texto_lower:
                    sentimiento = "Positivo"
                    break
            
            # Buscar palabras negativas
            for palabra in aspectos_negativos.get(aspecto, []):
                if palabra in texto_lower:
                    sentimiento = "Negativo"
                    break
            
            resultados[aspecto] = sentimiento
    
    return resultados

# Ejemplo
texto = "La batería es excelente y dura todo el día, pero la cámara es terrible y borrosa"
print(absa_simple(texto))
# {'batería': 'Positivo', 'cámara': 'Negativo'}
```

---

## 4.7. Aplicaciones Reales

### Monitoreo de Redes Sociales

```python
# Ejemplo: Analizar sentimiento de menciones de una marca
menciones = [
    "@MarcaX Gran servicio al cliente, muy satisfecho!",
    "@MarcaX El producto llegó dañado, muy decepcionado",
    "@MarcaX ¿Cuál es el horario de atención?",
    "@MarcaX Increíble calidad, lo recomiendo totalmente",
    "@MarcaX El peor servicio que he recibido, jamás volveré"
]

from collections import Counter

def monitorear_marca(menciones, classifier):
    """Analiza el sentimiento de menciones de una marca."""
    resultados = []
    
    for mencion in menciones:
        analisis = classifier(mencion)[0]
        resultados.append(analisis['label'])
    
    # Resumen
    conteo = Counter(resultados)
    total = len(resultados)
    
    print("=== Resumen de Sentimiento ===")
    for label, count in conteo.most_common():
        porcentaje = (count / total) * 100
        print(f"{label}: {count} ({porcentaje:.1f}%)")
    
    return resultados

# monitorear_marca(menciones, sentiment_pipeline)
```

### Análisis de Reseñas de Productos

```python
def analizar_reseñas_producto(reseñas):
    """
    Analiza reseñas de un producto y genera un resumen.
    """
    positivas = []
    negativas = []
    
    for reseña in reseñas:
        # Aquí iría el clasificador
        # resultado = classifier(reseña)
        # Por simplicidad, usamos longitud como proxy
        if len(reseña) > 50 and "excelente" in reseña.lower():
            positivas.append(reseña)
        elif "malo" in reseña.lower() or "terrible" in reseña.lower():
            negativas.append(reseña)
    
    return {
        'total': len(reseñas),
        'positivas': len(positivas),
        'negativas': len(negativas),
        'score': len(positivas) / max(len(reseñas), 1)
    }
```

---

## 4.8. Desafíos y Consideraciones

### Desafíos Comunes

1.  **Sarcasmo e Ironía:** "Oh genial, otro producto que no funciona" → Detectado erróneamente como positivo.

2.  **Negaciones:** "No me gustó nada" → Palabras positivas ("gustó") en contexto negativo.

3.  **Comparaciones:** "Mejor que la competencia pero aún malo" → Mezcla de sentimientos.

4.  **Contexto del Dominio:** "La película es una bomba" → Puede ser muy buena (éxito) o muy mala (fracaso).

5.  **Multilingüe:** Modelos entrenados en inglés no funcionan bien en español sin adaptación.

### Métricas de Evaluación

| Métrica | Uso |
| :--- | :--- |
| **Accuracy** | Proporción de predicciones correctas |
| **Precision** | De los predichos positivos, cuántos son correctos |
| **Recall** | De los positivos reales, cuántos detectamos |
| **F1-Score** | Media armónica de Precision y Recall |
| **Macro F1** | Promedio de F1 por clase (útil con clases desbalanceadas) |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
