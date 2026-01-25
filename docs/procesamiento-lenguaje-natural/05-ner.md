# 💬 Unidad 5. Reconocimiento de Entidades Nombradas (NER)

El **Reconocimiento de Entidades Nombradas** (Named Entity Recognition - NER) es una tarea fundamental del NLP que consiste en identificar y clasificar entidades mencionadas en texto en categorías predefinidas como personas, organizaciones, lugares, fechas, etc.

---

## 5.1. ¿Qué es NER?

NER extrae información estructurada del texto no estructurado, identificando:

*   **Quién:** Personas, organizaciones
*   **Dónde:** Lugares, direcciones
*   **Cuándo:** Fechas, horas
*   **Qué:** Productos, eventos
*   **Cuánto:** Cantidades, porcentajes, dinero

### Ejemplo

```
Texto: "Apple anunció que Tim Cook visitará Madrid el 15 de enero para 
        presentar el nuevo iPhone 15 con un precio de 999 euros."

Entidades:
- Apple → ORGANIZACIÓN
- Tim Cook → PERSONA
- Madrid → LUGAR
- 15 de enero → FECHA
- iPhone 15 → PRODUCTO
- 999 euros → DINERO
```

---

## 5.2. Categorías Comunes de Entidades

### Etiquetas Estándar (CoNLL)

| Etiqueta | Descripción | Ejemplos |
| :--- | :--- | :--- |
| **PER** | Persona | Tim Cook, María García |
| **ORG** | Organización | Apple, Microsoft, ONU |
| **LOC** | Ubicación | Madrid, Río Amazonas |
| **MISC** | Misceláneo | iPhone, COVID-19 |

### Etiquetas Extendidas

| Etiqueta | Descripción | Ejemplos |
| :--- | :--- | :--- |
| **DATE** | Fecha | 15 de enero, 2024 |
| **TIME** | Hora | 3:00 PM, mediodía |
| **MONEY** | Dinero | $100, 999 euros |
| **PERCENT** | Porcentaje | 15%, 0.5% |
| **QUANTITY** | Cantidad | 100 km, 5 kg |
| **EVENT** | Evento | Copa Mundial, Premios Oscar |
| **PRODUCT** | Producto | iPhone, Windows 11 |
| **LAW** | Ley/Regulación | GDPR, Constitución |

---

## 5.3. Formato de Anotación: BIO

El esquema **BIO** (Beginning-Inside-Outside) es el formato estándar para etiquetar secuencias:

*   **B-XXX:** Beginning - Primera palabra de la entidad XXX
*   **I-XXX:** Inside - Palabra intermedia/final de la entidad XXX
*   **O:** Outside - No es parte de ninguna entidad

### Ejemplo

```
Texto:     Tim    Cook   visitará   Madrid   el   próximo   lunes
Etiquetas: B-PER  I-PER  O          B-LOC    O    O         B-DATE
```

### Variantes

*   **IOB1:** B- solo cuando hay entidades consecutivas del mismo tipo.
*   **IOB2 (BIO):** B- siempre al inicio de una entidad.
*   **BIOES:** Añade S (Single) para entidades de una sola palabra y E (End).

---

## 5.4. NER con spaCy

**spaCy** ofrece modelos preentrenados con excelente soporte para NER.

```python
import spacy

# Cargar modelo en español
nlp = spacy.load('es_core_news_lg')  # Modelo grande para mejor precisión

texto = """
Apple Inc. anunció que Tim Cook visitará la sede de Madrid el 15 de enero de 2024.
La compañía presentará el iPhone 15 Pro con un precio de 1.199 euros.
"""

doc = nlp(texto)

# Extraer entidades
print("Entidades encontradas:")
print("-" * 50)
for ent in doc.ents:
    print(f"{ent.text:25} → {ent.label_:10} ({ent.start_char}-{ent.end_char})")
```

**Salida esperada:**

```
Entidades encontradas:
--------------------------------------------------
Apple Inc.                → ORG        (1-11)
Tim Cook                  → PER        (26-34)
Madrid                    → LOC        (54-60)
15 de enero de 2024       → DATE       (64-83)
iPhone 15 Pro             → MISC       (111-124)
1.199 euros               → MONEY      (143-154)
```

### Visualización de Entidades

```python
from spacy import displacy

# Visualización en notebook o HTML
displacy.render(doc, style="ent", jupyter=True)

# O guardar como HTML
html = displacy.render(doc, style="ent", page=True)
with open("entidades.html", "w", encoding="utf-8") as f:
    f.write(html)
```

---

## 5.5. NER con Transformers (Hugging Face)

Los modelos basados en BERT ofrecen el mejor rendimiento actual.

```python
from transformers import pipeline

# Pipeline de NER
ner_pipeline = pipeline("ner", grouped_entities=True)

texto = "Apple CEO Tim Cook announced the new iPhone 15 in California on September 12th."

entidades = ner_pipeline(texto)

print("Entidades (BERT):")
for ent in entidades:
    print(f"{ent['word']:20} → {ent['entity_group']:10} (score: {ent['score']:.3f})")
```

### Modelo Específico para Español

```python
from transformers import pipeline

# Modelo NER entrenado en español
ner_es = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    grouped_entities=True
)

texto_es = "El presidente Pedro Sánchez visitó Barcelona el martes pasado"
entidades_es = ner_es(texto_es)

for ent in entidades_es:
    print(f"{ent['word']:20} → {ent['entity_group']}")
```

---

## 5.6. Entrenamiento de Modelo NER Personalizado

### Con spaCy

```python
import spacy
from spacy.training import Example
import random

# Datos de entrenamiento en formato spaCy
TRAIN_DATA = [
    ("Apple lanzará el iPhone 16 en septiembre", {
        "entities": [(0, 5, "ORG"), (18, 27, "PRODUCT"), (31, 41, "DATE")]
    }),
    ("Microsoft anunció Windows 12 ayer", {
        "entities": [(0, 9, "ORG"), (18, 28, "PRODUCT"), (29, 33, "DATE")]
    }),
    ("El CEO de Google, Sundar Pichai, visitó Madrid", {
        "entities": [(10, 16, "ORG"), (18, 31, "PER"), (41, 47, "LOC")]
    }),
]

# Crear modelo en blanco
nlp = spacy.blank("es")

# Añadir componente NER
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Añadir etiquetas
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Entrenamiento
optimizer = nlp.begin_training()

for epoch in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}
    
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {losses['ner']:.4f}")

# Guardar modelo
nlp.to_disk("modelo_ner_custom")

# Probar
doc = nlp("Samsung presentará el Galaxy S25 en enero")
for ent in doc.ents:
    print(f"{ent.text} → {ent.label_}")
```

---

## 5.7. Métricas de Evaluación para NER

### Métricas Principales

| Métrica | Fórmula | Descripción |
| :--- | :--- | :--- |
| **Precision** | TP / (TP + FP) | De las entidades predichas, cuántas son correctas |
| **Recall** | TP / (TP + FN) | De las entidades reales, cuántas encontramos |
| **F1-Score** | 2×(P×R)/(P+R) | Media armónica de Precision y Recall |

### Tipos de Match

*   **Exact Match:** La entidad predicha coincide exactamente en texto y tipo.
*   **Partial Match:** El texto coincide parcialmente.
*   **Type Match:** El tipo es correcto aunque los límites no sean exactos.

```python
from seqeval.metrics import classification_report, f1_score

# Etiquetas reales (formato BIO)
y_true = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'O']]
# Etiquetas predichas
y_pred = [['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'O']]

print(classification_report(y_true, y_pred))
print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
```

---

## 5.8. Aplicaciones Reales de NER

### Extracción de Información de Noticias

```python
import spacy

nlp = spacy.load('es_core_news_lg')

noticia = """
El presidente de Estados Unidos, Joe Biden, se reunió con 
el canciller alemán Olaf Scholz en Berlín el pasado miércoles.
Discutieron sobre la situación en Ucrania y el acuerdo comercial 
de 500 millones de dólares entre ambos países.
"""

doc = nlp(noticia)

# Extraer información estructurada
info = {
    'personas': [],
    'lugares': [],
    'organizaciones': [],
    'fechas': [],
    'dinero': []
}

for ent in doc.ents:
    if ent.label_ == 'PER':
        info['personas'].append(ent.text)
    elif ent.label_ in ['LOC', 'GPE']:
        info['lugares'].append(ent.text)
    elif ent.label_ == 'ORG':
        info['organizaciones'].append(ent.text)
    elif ent.label_ == 'DATE':
        info['fechas'].append(ent.text)
    elif ent.label_ == 'MONEY':
        info['dinero'].append(ent.text)

print("Información Extraída:")
for key, values in info.items():
    if values:
        print(f"  {key.upper()}: {', '.join(set(values))}")
```

### Anonimización de Datos (GDPR)

```python
def anonimizar_texto(texto, nlp):
    """
    Anonimiza información personal en el texto.
    """
    doc = nlp(texto)
    
    # Crear texto anonimizado
    texto_anon = texto
    
    # Reemplazar de atrás hacia adelante para mantener índices
    for ent in reversed(doc.ents):
        if ent.label_ in ['PER', 'PERSON']:
            reemplazo = '[NOMBRE]'
        elif ent.label_ in ['LOC', 'GPE']:
            reemplazo = '[UBICACIÓN]'
        elif ent.label_ == 'ORG':
            reemplazo = '[ORGANIZACIÓN]'
        elif ent.label_ == 'EMAIL':
            reemplazo = '[EMAIL]'
        elif ent.label_ in ['PHONE', 'CARDINAL'] and len(ent.text) > 6:
            reemplazo = '[TELÉFONO]'
        else:
            continue
        
        texto_anon = texto_anon[:ent.start_char] + reemplazo + texto_anon[ent.end_char:]
    
    return texto_anon

# Ejemplo
texto = "Juan García de Madrid llamó al 612345678 para hablar con Apple"
texto_anonimizado = anonimizar_texto(texto, nlp)
print(f"Original: {texto}")
print(f"Anonimizado: {texto_anonimizado}")
```

### Extracción de CVs

```python
def extraer_info_cv(cv_texto, nlp):
    """
    Extrae información estructurada de un CV.
    """
    doc = nlp(cv_texto)
    
    info_cv = {
        'nombre': None,
        'ubicacion': None,
        'empresas': [],
        'fechas': [],
        'habilidades': []  # Requeriría un modelo personalizado
    }
    
    for ent in doc.ents:
        if ent.label_ == 'PER' and info_cv['nombre'] is None:
            info_cv['nombre'] = ent.text
        elif ent.label_ in ['LOC', 'GPE']:
            info_cv['ubicacion'] = ent.text
        elif ent.label_ == 'ORG':
            info_cv['empresas'].append(ent.text)
        elif ent.label_ == 'DATE':
            info_cv['fechas'].append(ent.text)
    
    return info_cv

cv = """
María López García
Desarrolladora Senior | Madrid, España

Experiencia:
- Google (2020-2023): Ingeniera de Software
- Microsoft (2018-2020): Desarrolladora Junior

Educación:
- Universidad Complutense de Madrid (2014-2018)
"""

print(extraer_info_cv(cv, nlp))
```

---

## 5.9. Desafíos y Consideraciones

### Desafíos Comunes

1.  **Entidades anidadas:** "Banco de España" → ORG que contiene LOC.
2.  **Ambigüedad:** "Apple" puede ser la empresa o la fruta.
3.  **Entidades discontinuas:** "Microsoft y Google Inc." → Dos ORGs.
4.  **Variación de nombres:** "EEUU", "Estados Unidos", "USA" → Mismo lugar.
5.  **Nuevas entidades:** Nombres de productos, personas, empresas nuevas.
6.  **Dominio específico:** Entidades médicas, legales, financieras requieren modelos especializados.

### Mejores Prácticas

*   Usar modelos grandes preentrenados como base.
*   Fine-tuning con datos del dominio específico.
*   Combinar reglas y modelos ML para mejor precisión.
*   Validar y corregir errores manualmente para mejorar el modelo.
*   Considerar Entity Linking para resolver ambigüedades.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
