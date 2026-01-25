# 💬 Unidad 8. Aplicaciones Avanzadas de NLP

Esta unidad cubre aplicaciones avanzadas y casos de uso prácticos del NLP en el mundo real, incluyendo chatbots, sistemas de preguntas y respuestas, resumen automático y traducción automática.

---

## 8.1. Sistemas de Preguntas y Respuestas (QA)

Los sistemas de **Question Answering (QA)** responden preguntas en lenguaje natural basándose en un contexto o base de conocimiento.

### Tipos de QA

| Tipo | Descripción | Ejemplo |
| :--- | :--- | :--- |
| **Extractivo** | Extrae la respuesta directamente del texto | "¿Quién fundó Apple?" → "Steve Jobs" (del contexto) |
| **Generativo** | Genera una respuesta en lenguaje natural | Respuesta elaborada basada en múltiples fuentes |
| **Open-domain** | Responde sobre cualquier tema | Wikipedia QA |
| **Closed-domain** | Responde sobre un dominio específico | QA médico, legal |

### QA Extractivo con Transformers

```python
from transformers import pipeline

# Pipeline de QA
qa_pipeline = pipeline("question-answering")

# Contexto
contexto = """
Apple Inc. es una empresa tecnológica estadounidense fundada en 1976 por Steve Jobs, 
Steve Wozniak y Ronald Wayne. La compañía tiene su sede en Cupertino, California.
Apple es conocida por productos como el iPhone, iPad, Mac y Apple Watch.
En 2023, Apple alcanzó una valoración de mercado de 3 billones de dólares.
"""

# Preguntas
preguntas = [
    "¿Quién fundó Apple?",
    "¿Dónde está la sede de Apple?",
    "¿Cuál es la valoración de Apple?",
    "¿Cuándo se fundó Apple?"
]

print("Sistema de Preguntas y Respuestas")
print("=" * 50)

for pregunta in preguntas:
    resultado = qa_pipeline(question=pregunta, context=contexto)
    print(f"\nP: {pregunta}")
    print(f"R: {resultado['answer']} (confianza: {resultado['score']:.3f})")
```

### QA con Retrieval (RAG - Retrieval Augmented Generation)

```python
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# Base de conocimiento
documentos = [
    "Python es un lenguaje de programación de alto nivel creado por Guido van Rossum en 1991.",
    "JavaScript fue creado por Brendan Eich en 1995 para Netscape.",
    "Java fue desarrollado por James Gosling en Sun Microsystems en 1995.",
    "C++ fue creado por Bjarne Stroustrup comenzando en 1979.",
    "Rust es un lenguaje de programación de sistemas desarrollado por Mozilla desde 2010."
]

# Modelo de embeddings para retrieval
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
doc_embeddings = embedder.encode(documentos)

# Pipeline de QA
qa_pipeline = pipeline("question-answering")

def responder_pregunta_rag(pregunta, documentos, doc_embeddings, embedder, qa_pipeline, top_k=2):
    """
    Sistema RAG: Retrieve + Generate
    """
    # 1. Retrieve: Encontrar documentos relevantes
    query_embedding = embedder.encode(pregunta)
    similarities = np.dot(doc_embeddings, query_embedding)
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Contexto relevante
    contexto = " ".join([documentos[i] for i in top_indices])
    
    # 2. Generate: Responder usando el contexto
    resultado = qa_pipeline(question=pregunta, context=contexto)
    
    return {
        'pregunta': pregunta,
        'respuesta': resultado['answer'],
        'confianza': resultado['score'],
        'documentos_usados': [documentos[i] for i in top_indices]
    }

# Probar
pregunta = "¿Quién creó Python?"
resultado = responder_pregunta_rag(pregunta, documentos, doc_embeddings, embedder, qa_pipeline)

print(f"Pregunta: {resultado['pregunta']}")
print(f"Respuesta: {resultado['respuesta']}")
print(f"Confianza: {resultado['confianza']:.3f}")
print(f"Documentos consultados: {resultado['documentos_usados']}")
```

---

## 8.2. Chatbots y Asistentes Conversacionales

### Arquitectura de un Chatbot

```
┌─────────────────────────────────────────────────────┐
│                    Usuario                           │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              NLU (Natural Language Understanding)    │
│  - Intent Classification                             │
│  - Entity Extraction                                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Dialog Manager                          │
│  - State Tracking                                    │
│  - Policy Selection                                  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              NLG (Natural Language Generation)       │
│  - Response Generation                               │
│  - Template Filling                                  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                    Respuesta                         │
└─────────────────────────────────────────────────────┘
```

### Chatbot Simple con Reglas

```python
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ChatbotSimple:
    def __init__(self):
        self.intents = {
            'saludo': {
                'patterns': ['hola', 'buenos días', 'buenas tardes', 'qué tal'],
                'responses': ['¡Hola! ¿En qué puedo ayudarte?', '¡Buenos días! ¿Cómo puedo asistirte?']
            },
            'despedida': {
                'patterns': ['adiós', 'hasta luego', 'chao', 'nos vemos'],
                'responses': ['¡Hasta luego!', '¡Que tengas un buen día!']
            },
            'precio': {
                'patterns': ['cuánto cuesta', 'precio', 'valor', 'cuánto vale'],
                'responses': ['Nuestros precios varían. ¿Qué producto te interesa?']
            },
            'ayuda': {
                'patterns': ['ayuda', 'necesito ayuda', 'cómo funciona', 'help'],
                'responses': ['Puedo ayudarte con información de productos, precios y pedidos.']
            },
            'default': {
                'patterns': [],
                'responses': ['No entiendo tu pregunta. ¿Podrías reformularla?']
            }
        }
        
        self._entrenar_clasificador()
    
    def _entrenar_clasificador(self):
        textos = []
        labels = []
        
        for intent, data in self.intents.items():
            if intent != 'default':
                for pattern in data['patterns']:
                    textos.append(pattern)
                    labels.append(intent)
        
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(textos)
        
        self.clf = LogisticRegression()
        self.clf.fit(X, labels)
    
    def responder(self, mensaje):
        import random
        
        # Preprocesar
        mensaje_limpio = mensaje.lower().strip()
        
        # Clasificar intent
        X = self.vectorizer.transform([mensaje_limpio])
        intent = self.clf.predict(X)[0]
        confianza = self.clf.predict_proba(X).max()
        
        # Si confianza baja, usar default
        if confianza < 0.3:
            intent = 'default'
        
        # Seleccionar respuesta
        respuestas = self.intents[intent]['responses']
        respuesta = random.choice(respuestas)
        
        return {
            'intent': intent,
            'confianza': confianza,
            'respuesta': respuesta
        }

# Usar chatbot
bot = ChatbotSimple()

mensajes = [
    "Hola, buenos días",
    "¿Cuánto cuesta el producto?",
    "Necesito ayuda por favor",
    "Adiós, gracias",
    "¿Cuál es el clima hoy?"  # Fuera de dominio
]

print("Chatbot Demo")
print("=" * 50)
for msg in mensajes:
    resultado = bot.responder(msg)
    print(f"\nUsuario: {msg}")
    print(f"Bot: {resultado['respuesta']}")
    print(f"  (intent: {resultado['intent']}, conf: {resultado['confianza']:.2f})")
```

### Chatbot con LLM

```python
from openai import OpenAI

class ChatbotLLM:
    def __init__(self, api_key, system_prompt=None):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt or """
        Eres un asistente virtual amable y útil. 
        Respondes de forma concisa y clara.
        Si no sabes algo, lo admites honestamente.
        """
        self.historial = []
    
    def responder(self, mensaje_usuario):
        # Añadir mensaje al historial
        self.historial.append({
            "role": "user",
            "content": mensaje_usuario
        })
        
        # Llamar a la API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt}
            ] + self.historial,
            temperature=0.7,
            max_tokens=500
        )
        
        respuesta = response.choices[0].message.content
        
        # Añadir respuesta al historial
        self.historial.append({
            "role": "assistant",
            "content": respuesta
        })
        
        return respuesta
    
    def reiniciar_conversacion(self):
        self.historial = []

# Uso
# bot_llm = ChatbotLLM(api_key="tu-api-key")
# print(bot_llm.responder("¿Qué es el machine learning?"))
```

---

## 8.3. Resumen Automático de Texto

### Tipos de Resumen

| Tipo | Descripción | Método |
| :--- | :--- | :--- |
| **Extractivo** | Selecciona oraciones importantes del texto original | TextRank, BERT extractivo |
| **Abstractivo** | Genera un nuevo texto que resume el contenido | T5, BART, GPT |

### Resumen Extractivo (TextRank)

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def resumir_extractivo(texto, num_oraciones=3):
    """
    Resumen extractivo usando TextRank.
    """
    parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
    summarizer = TextRankSummarizer()
    
    resumen = summarizer(parser.document, num_oraciones)
    
    return " ".join([str(oracion) for oracion in resumen])

# Texto largo de ejemplo
texto_largo = """
La inteligencia artificial ha experimentado un crecimiento exponencial en los últimos años.
Los avances en deep learning han permitido crear sistemas capaces de reconocer imágenes,
traducir idiomas y mantener conversaciones naturales con humanos.

Empresas como Google, Microsoft y OpenAI lideran la investigación en este campo.
Los modelos de lenguaje como GPT-4 pueden generar texto indistinguible del escrito por humanos.
Esto ha generado debates sobre el futuro del trabajo y la ética en la IA.

A pesar de los avances, los expertos advierten que la IA actual no posee verdadera comprensión
o consciencia. Los sistemas actuales son herramientas sofisticadas de reconocimiento de patrones.
El camino hacia la inteligencia artificial general (AGI) aún es largo e incierto.
"""

resumen = resumir_extractivo(texto_largo, num_oraciones=2)
print("Resumen Extractivo:")
print(resumen)
```

### Resumen Abstractivo con Transformers

```python
from transformers import pipeline

# Pipeline de resumen
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Texto en inglés (modelos en inglés funcionan mejor)
texto_en = """
Artificial intelligence has experienced exponential growth in recent years.
Advances in deep learning have enabled systems capable of recognizing images,
translating languages, and having natural conversations with humans.
Companies like Google, Microsoft, and OpenAI lead research in this field.
Language models like GPT-4 can generate text indistinguishable from human writing.
This has sparked debates about the future of work and AI ethics.
"""

resumen = summarizer(texto_en, max_length=50, min_length=20, do_sample=False)
print("Resumen Abstractivo:")
print(resumen[0]['summary_text'])
```

### Resumen en Español con mT5

```python
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Cargar modelo multilingüe
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

def resumir_mt5(texto, max_length=100):
    """
    Resumen con mT5 (multilingüe).
    """
    # Prefijo para tarea de resumen
    input_text = f"summarize: {texto}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=20,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    resumen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resumen

# Ejemplo
# resumen_es = resumir_mt5(texto_largo)
# print(resumen_es)
```

---

## 8.4. Traducción Automática

### Traducción con MarianMT

```python
from transformers import MarianMTModel, MarianTokenizer

def traducir(texto, src_lang="es", tgt_lang="en"):
    """
    Traduce texto entre idiomas usando MarianMT.
    """
    # Cargar modelo para el par de idiomas
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Tokenizar
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    
    # Traducir
    translated = model.generate(**inputs)
    
    # Decodificar
    resultado = tokenizer.decode(translated[0], skip_special_tokens=True)
    return resultado

# Ejemplos
texto_es = "El procesamiento de lenguaje natural permite a las máquinas entender el lenguaje humano."
texto_en = traducir(texto_es, "es", "en")
print(f"Original (ES): {texto_es}")
print(f"Traducido (EN): {texto_en}")

# Traducción inversa
texto_back = traducir(texto_en, "en", "es")
print(f"Back-translation: {texto_back}")
```

### Traducción con Pipeline

```python
from transformers import pipeline

# Pipeline de traducción
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

textos_es = [
    "Buenos días, ¿cómo estás?",
    "Me gusta programar en Python.",
    "El aprendizaje automático es fascinante."
]

print("Traducciones ES → EN:")
for texto in textos_es:
    resultado = translator(texto)[0]['translation_text']
    print(f"  {texto}")
    print(f"  → {resultado}")
    print()
```

---

## 8.5. Generación de Texto

### Completado de Texto

```python
from transformers import pipeline

# Pipeline de generación
generator = pipeline("text-generation", model="gpt2")

# Prompt
prompt = "The future of artificial intelligence will"

# Generar
resultado = generator(
    prompt,
    max_length=100,
    num_return_sequences=3,
    temperature=0.8,
    top_p=0.9
)

print("Generaciones:")
for i, gen in enumerate(resultado):
    print(f"\n{i+1}. {gen['generated_text']}")
```

### Generación Controlada

```python
from transformers import pipeline

# Generador con control de estilo
generator = pipeline("text-generation", model="gpt2-medium")

prompts_con_estilo = [
    "Write a formal email: Dear Mr. Smith,",
    "Write a casual text message: Hey!",
    "Write a news headline: BREAKING:"
]

for prompt in prompts_con_estilo:
    resultado = generator(prompt, max_length=50, num_return_sequences=1)
    print(f"Prompt: {prompt}")
    print(f"Output: {resultado[0]['generated_text']}")
    print("-" * 50)
```

---

## 8.6. Detección de Texto Generado por IA

Con el auge de los LLMs, detectar texto generado por IA es cada vez más importante.

```python
from transformers import pipeline

# Detector de texto IA (ejemplo conceptual)
# Nota: Los detectores actuales no son 100% precisos

def analizar_texto_ia(texto):
    """
    Heurísticas simples para detectar texto generado por IA.
    """
    señales = {
        'longitud_oraciones': [],
        'palabras_repetidas': {},
        'conectores_comunes': 0
    }
    
    # Analizar longitud de oraciones
    oraciones = texto.split('.')
    señales['longitud_oraciones'] = [len(o.split()) for o in oraciones if o.strip()]
    
    # Variabilidad de longitud (IA tiende a ser más uniforme)
    if señales['longitud_oraciones']:
        variabilidad = np.std(señales['longitud_oraciones'])
    else:
        variabilidad = 0
    
    # Conectores comunes en texto de IA
    conectores_ia = ['además', 'sin embargo', 'por lo tanto', 'en conclusión', 
                     'furthermore', 'however', 'therefore', 'in conclusion']
    
    texto_lower = texto.lower()
    for conector in conectores_ia:
        if conector in texto_lower:
            señales['conectores_comunes'] += 1
    
    # Score simple
    score_ia = 0.5  # Base
    
    if variabilidad < 5:  # Poca variabilidad
        score_ia += 0.2
    
    if señales['conectores_comunes'] >= 3:
        score_ia += 0.2
    
    return {
        'score_ia': min(score_ia, 1.0),
        'señales': señales,
        'prediccion': 'Probablemente IA' if score_ia > 0.6 else 'Probablemente humano'
    }

# Ejemplo
import numpy as np

texto_sospechoso = """
La inteligencia artificial ha revolucionado muchos campos. Sin embargo, también 
presenta desafíos. Por lo tanto, es importante considerarlos. Además, debemos 
ser conscientes de las implicaciones éticas. En conclusión, la IA es una 
herramienta poderosa que debe usarse responsablemente.
"""

resultado = analizar_texto_ia(texto_sospechoso)
print(f"Score IA: {resultado['score_ia']:.2f}")
print(f"Predicción: {resultado['prediccion']}")
```

---

## 8.7. Aplicaciones Empresariales

### Extracción de Información de Documentos

```python
import spacy
from collections import defaultdict

nlp = spacy.load('es_core_news_lg')

def extraer_informacion_contrato(texto):
    """
    Extrae información clave de un contrato.
    """
    doc = nlp(texto)
    
    info = {
        'partes': [],
        'fechas': [],
        'montos': [],
        'ubicaciones': []
    }
    
    for ent in doc.ents:
        if ent.label_ in ['PER', 'ORG']:
            info['partes'].append(ent.text)
        elif ent.label_ == 'DATE':
            info['fechas'].append(ent.text)
        elif ent.label_ == 'MONEY':
            info['montos'].append(ent.text)
        elif ent.label_ in ['LOC', 'GPE']:
            info['ubicaciones'].append(ent.text)
    
    # Eliminar duplicados
    for key in info:
        info[key] = list(set(info[key]))
    
    return info

# Ejemplo
contrato = """
El presente contrato se celebra entre Empresa ABC S.A. y Juan Pérez García,
con fecha 15 de enero de 2024 en Madrid, España.
El monto total del contrato es de 50.000 euros, pagaderos en 12 meses.
"""

info_contrato = extraer_informacion_contrato(contrato)
print("Información extraída del contrato:")
for key, values in info_contrato.items():
    if values:
        print(f"  {key}: {', '.join(values)}")
```

### Análisis de Feedback de Clientes

```python
from collections import Counter
import pandas as pd

def analizar_feedback(comentarios, sentiment_pipeline):
    """
    Analiza feedback de clientes para obtener insights.
    """
    resultados = []
    
    for comentario in comentarios:
        # Análisis de sentimiento
        sent = sentiment_pipeline(comentario)[0]
        
        resultados.append({
            'comentario': comentario[:50] + '...' if len(comentario) > 50 else comentario,
            'sentimiento': sent['label'],
            'score': sent['score']
        })
    
    df = pd.DataFrame(resultados)
    
    # Resumen
    resumen = {
        'total_comentarios': len(comentarios),
        'distribucion_sentimiento': df['sentimiento'].value_counts().to_dict(),
        'score_promedio': df['score'].mean(),
        'ejemplos_negativos': df[df['sentimiento'] == 'NEGATIVE'].head(3)['comentario'].tolist()
    }
    
    return resumen

# Ejemplo conceptual
# comentarios = ["Excelente producto!", "Muy mal servicio", "Normal, nada especial"]
# resumen = analizar_feedback(comentarios, sentiment_pipeline)
```

---

## 8.8. Consideraciones Éticas y Mejores Prácticas

### Sesgos en NLP

*   Los modelos heredan sesgos de los datos de entrenamiento.
*   Evaluar y mitigar sesgos de género, raza, etc.
*   Usar datasets diversos y representativos.

### Privacidad

*   Los modelos pueden memorizar información sensible.
*   Implementar técnicas de anonimización.
*   Cumplir con regulaciones (GDPR, etc.).

### Transparencia

*   Documentar limitaciones de los modelos.
*   Proporcionar explicaciones cuando sea posible.
*   Informar a usuarios cuando interactúan con IA.

### Uso Responsable

*   No generar desinformación o contenido dañino.
*   Implementar filtros y salvaguardas.
*   Considerar el impacto social de las aplicaciones.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
