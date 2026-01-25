# 📄 Trabajando con JSON en Python

**JSON** (JavaScript Object Notation) es un formato ligero de intercambio de datos, fácil de leer y escribir para humanos, y fácil de parsear y generar para máquinas. Es el formato estándar para APIs web y configuraciones.

---

## 1. ¿Qué es JSON?

JSON es un formato de texto que representa datos estructurados. Su sintaxis es muy similar a los diccionarios y listas de Python.

### Estructura de JSON

```json
{
    "nombre": "Fran",
    "edad": 25,
    "activo": true,
    "cursos": ["Python", "Machine Learning", "Deep Learning"],
    "direccion": {
        "ciudad": "Alicante",
        "pais": "España"
    }
}
```

### Tipos de datos en JSON

| JSON | Python |
| :--- | :--- |
| `object` `{}` | `dict` |
| `array` `[]` | `list` |
| `string` `"texto"` | `str` |
| `number` `123` o `1.5` | `int` o `float` |
| `true` / `false` | `True` / `False` |
| `null` | `None` |

---

## 2. El Módulo `json`

Python incluye el módulo `json` en su biblioteca estándar:

```python
import json
```

### Funciones Principales

| Función | Descripción |
| :--- | :--- |
| `json.loads()` | Convierte string JSON → diccionario Python |
| `json.dumps()` | Convierte diccionario Python → string JSON |
| `json.load()` | Lee archivo JSON → diccionario Python |
| `json.dump()` | Escribe diccionario Python → archivo JSON |

---

## 3. Convertir JSON String a Diccionario

### json.loads() - Parse de String

```python
import json

# JSON como string (nota las comillas triples)
profesor_string = """
{
    "nombre": "Fran",
    "apellido": "García",
    "edad": 25,
    "domicilio": {
        "direccion": "Lillo Juan, 128",
        "ciudad": "San Vicente del Raspeig",
        "comunidad": "Valenciana",
        "codigoPostal": "03690"
    },
    "numerosTelefonos": [
        {"tipo": "casa", "numero": "666 666 666"},
        {"tipo": "movil", "numero": "777 777 777"}
    ]
}
"""

# Convertir JSON string a diccionario Python
profesor_dict = json.loads(profesor_string)

# Verificar el tipo
print(type(profesor_dict))  # <class 'dict'>

# Acceder a los datos
print(profesor_dict['nombre'])      # Fran
print(profesor_dict['edad'])        # 25

# Acceder a datos anidados
print(profesor_dict['domicilio']['ciudad'])  # San Vicente del Raspeig

# Acceder a listas
print(profesor_dict['numerosTelefonos'][0]['numero'])  # 666 666 666
```

---

## 4. Convertir Diccionario a JSON String

### json.dumps() - Serialización

```python
import json

# Diccionario Python
usuario = {
    "nombre": "Ana",
    "edad": 30,
    "activo": True,
    "cursos": ["Python", "Data Science"],
    "puntuacion": None
}

# Convertir a JSON string
json_string = json.dumps(usuario)
print(json_string)
# {"nombre": "Ana", "edad": 30, "activo": true, "cursos": ["Python", "Data Science"], "puntuacion": null}

# Con formato legible (indentación)
json_bonito = json.dumps(usuario, indent=4)
print(json_bonito)
# {
#     "nombre": "Ana",
#     "edad": 30,
#     "activo": true,
#     "cursos": [
#         "Python",
#         "Data Science"
#     ],
#     "puntuacion": null
# }

# Con caracteres especiales (español)
datos = {"ciudad": "San José", "país": "España"}
print(json.dumps(datos))  # {"ciudad": "San Jos\u00e9", "pa\u00eds": "Espa\u00f1a"}
print(json.dumps(datos, ensure_ascii=False))  # {"ciudad": "San José", "país": "España"}

# Ordenar claves alfabéticamente
print(json.dumps(usuario, sort_keys=True, indent=2))
```

---

## 5. Leer JSON desde Archivo

### json.load() - Lectura de Archivo

```python
import json

# Método 1: Leer y parsear directamente
with open('profesor.json', 'r', encoding='utf-8') as archivo:
    datos = json.load(archivo)

print(datos['nombre'])  # Fran
print(datos['domicilio']['ciudad'])  # San Vicente del Raspeig

# Método 2: Leer como string y luego parsear
with open('profesor.json', 'r', encoding='utf-8') as archivo:
    contenido = archivo.read()  # String con el JSON
    datos = json.loads(contenido)  # Parsear el string
```

### Ejemplo con archivo luke.json

```python
import json

with open('luke.json', 'r', encoding='utf-8') as f:
    luke = json.load(f)

print(luke['name'])    # Luke Skywalker
print(luke['height'])  # 172
print(luke['films'])   # Lista de URLs de películas
```

---

## 6. Escribir JSON en Archivo

### json.dump() - Escritura en Archivo

```python
import json

# Datos a guardar
usuario = {
    "nombre": "Fran",
    "apellido": "García",
    "edad": 48,
    "ciudad": "Alicante",
    "habilidades": ["Python", "Machine Learning", "SQL"]
}

# Guardar en archivo JSON
with open('usuario.json', 'w', encoding='utf-8') as archivo:
    json.dump(usuario, archivo)

# Con formato legible
with open('usuario_bonito.json', 'w', encoding='utf-8') as archivo:
    json.dump(usuario, archivo, indent=4, ensure_ascii=False)
```

### Archivo resultante (usuario_bonito.json):

```json
{
    "nombre": "Fran",
    "apellido": "García",
    "edad": 48,
    "ciudad": "Alicante",
    "habilidades": [
        "Python",
        "Machine Learning",
        "SQL"
    ]
}
```

---

## 7. Leer JSON desde Internet (APIs)

### Usando la librería `requests`

```python
# Primero instalar: pip install requests
import requests
import json

# Hacer petición GET a una API
url = "https://jsonplaceholder.typicode.com/todos/1"
response = requests.get(url)

# Método 1: Usar .json() de requests (más simple)
datos = response.json()
print(datos)
# {'userId': 1, 'id': 1, 'title': 'delectus aut autem', 'completed': False}

# Método 2: Parsear el texto de la respuesta
datos = json.loads(response.text)

# Acceder a los datos
print(datos['userId'])     # 1
print(datos['title'])      # delectus aut autem
print(datos['completed'])  # False
```

### Obtener Lista de Datos

```python
import requests

url = "https://jsonplaceholder.typicode.com/todos"
response = requests.get(url)
tareas = response.json()  # Lista de diccionarios

print(type(tareas))  # <class 'list'>
print(len(tareas))   # 200

# Filtrar con list comprehension
pendientes = [t for t in tareas if not t['completed']]
completadas = [t for t in tareas if t['completed']]

print(f"Pendientes: {len(pendientes)}")   # 110
print(f"Completadas: {len(completadas)}") # 90

# Mostrar las primeras 3 tareas pendientes
for tarea in pendientes[:3]:
    print(f"ID: {tarea['id']} - {tarea['title']}")
```

### API con Autenticación (Headers)

```python
import requests
import json

# API que requiere token de autenticación
token = "XXXX"  # Reemplazar con tu token real

url = "https://api.football-data.org/v4/teams/86/matches?status=SCHEDULED"

# Configurar cabeceras
cabeceras = {
    "X-Auth-Token": token,
    "User-Agent": "PostmanRuntime/7.26.8"
}

response = requests.get(url, headers=cabeceras)

if response.status_code == 200:
    datos = response.json()
    print(datos)
else:
    print(f"Error: {response.status_code}")
```

### Ejemplo: API de Star Wars

```python
import requests
import json

codigo_personaje = input("Introduce el código del personaje: ")

url = f"https://swapi.dev/api/people/{codigo_personaje}/?format=json"
response = requests.get(url)

if response.status_code == 200:
    personaje = response.json()
    
    print(f"Nombre: {personaje['name']}")
    print(f"Altura: {personaje['height']} cm")
    print(f"Peso: {personaje['mass']} kg")
    print(f"Color de ojos: {personaje['eye_color']}")
    
    # Obtener información de las películas
    print("\nPelículas:")
    for url_pelicula in personaje['films']:
        resp_pelicula = requests.get(url_pelicula)
        pelicula = resp_pelicula.json()
        print(f"  - {pelicula['title']} ({pelicula['release_date']})")
else:
    print("Personaje no encontrado")
```

---

## 8. Manejo de Errores en JSON

```python
import json

# JSON mal formado
json_invalido = '{"nombre": "Ana", "edad": }'

try:
    datos = json.loads(json_invalido)
except json.JSONDecodeError as e:
    print(f"Error al parsear JSON: {e}")
    # Error al parsear JSON: Expecting value: line 1 column 27 (char 26)

# Archivo que no existe
try:
    with open('no_existe.json', 'r') as f:
        datos = json.load(f)
except FileNotFoundError:
    print("El archivo no existe")
except json.JSONDecodeError:
    print("El archivo no contiene JSON válido")
```

---

## 9. JSON y Clases Python (Serialización Avanzada)

### Problema: Las clases no son serializables directamente

```python
import json

class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

persona = Persona("Ana", 25)

# Esto genera error
# json.dumps(persona)  # TypeError: Object of type Persona is not JSON serializable
```

### Solución 1: Convertir a diccionario manualmente

```python
import json

class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
    
    def to_dict(self):
        return {
            "nombre": self.nombre,
            "edad": self.edad
        }

persona = Persona("Ana", 25)
json_string = json.dumps(persona.to_dict())
print(json_string)  # {"nombre": "Ana", "edad": 25}
```

### Solución 2: Usar `__dict__`

```python
import json

class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

persona = Persona("Ana", 25)
json_string = json.dumps(persona.__dict__)
print(json_string)  # {"nombre": "Ana", "edad": 25}
```

### Solución 3: Encoder Personalizado

```python
import json

class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

class PersonaEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Persona):
            return {
                "nombre": obj.nombre,
                "edad": obj.edad
            }
        return super().default(obj)

persona = Persona("Ana", 25)
json_string = json.dumps(persona, cls=PersonaEncoder)
print(json_string)  # {"nombre": "Ana", "edad": 25}
```

### Decoder Personalizado (JSON → Objeto)

```python
import json

class Tarea:
    def __init__(self, user_id, id, title, completed):
        self.user_id = user_id
        self.id = id
        self.title = title
        self.completed = completed
    
    def __repr__(self):
        return f"Tarea({self.id}: {self.title})"

class TareaDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.object_hook)
    
    def object_hook(self, json_dict):
        # Solo convertir si tiene las claves esperadas
        if 'userId' in json_dict and 'title' in json_dict:
            return Tarea(
                json_dict.get('userId'),
                json_dict.get('id'),
                json_dict.get('title'),
                json_dict.get('completed')
            )
        return json_dict

# Usar el decoder
import requests

url = "https://jsonplaceholder.typicode.com/todos/1"
response = requests.get(url)

tarea = json.loads(response.text, cls=TareaDecoder)
print(type(tarea))       # <class 'Tarea'>
print(tarea.id)          # 1
print(tarea.title)       # delectus aut autem
print(tarea.completed)   # False
```

---

## 10. Ejemplo Práctico: Consumir API y Crear Dataset

```python
import requests
import json
import csv
from datetime import datetime

# Configuración
API_KEY = "XXXX"  # Reemplazar con tu API key
BASE_URL = "https://api.rawg.io/api/games"

# Clase para videojuegos
class Videojuego:
    def __init__(self, nombre, anyo, imagen, valoracion):
        self.nombre = nombre
        self.anyo = anyo
        self.imagen = imagen
        self.valoracion = valoracion
    
    def __str__(self):
        return f"{self.nombre},{self.anyo},{self.imagen},{self.valoracion}"

# Obtener datos de la API
videojuegos = []

for pagina in range(1, 3):  # 2 páginas
    url = f"{BASE_URL}?key={API_KEY}&page={pagina}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        datos = response.json()
        
        for juego in datos['results']:
            # Crear objeto Videojuego
            vj = Videojuego(
                nombre=juego['name'],
                anyo=juego['released'][:4] if juego['released'] else 'N/A',
                imagen=juego['background_image'] or 'Sin imagen',
                valoracion=juego['rating']
            )
            videojuegos.append(vj)

print(f"Total videojuegos obtenidos: {len(videojuegos)}")

# Filtrar los mejor valorados
mejores = [vj for vj in videojuegos if vj.valoracion > 4.0]
mejores.sort(key=lambda x: x.valoracion, reverse=True)

# Guardar en CSV
with open('videojuegos.csv', 'w', newline='', encoding='utf-8') as f:
    f.write("Nombre,Año,Imagen,Rating\n")
    for vj in mejores:
        f.write(f"{vj}\n")

print(f"Guardados {len(mejores)} videojuegos con rating > 4.0")
```

---

## 11. Trabajar con JSON Anidados Complejos

```python
import json

# JSON complejo con múltiples niveles
empresa_json = """
{
    "nombre": "TechCorp",
    "fundacion": 2010,
    "departamentos": [
        {
            "nombre": "Desarrollo",
            "empleados": [
                {"nombre": "Ana", "cargo": "Senior Dev", "salario": 45000},
                {"nombre": "Luis", "cargo": "Junior Dev", "salario": 28000}
            ]
        },
        {
            "nombre": "Marketing",
            "empleados": [
                {"nombre": "María", "cargo": "Manager", "salario": 52000}
            ]
        }
    ],
    "activa": true
}
"""

empresa = json.loads(empresa_json)

# Navegar por la estructura
print(empresa['nombre'])  # TechCorp

# Iterar departamentos
for depto in empresa['departamentos']:
    print(f"\nDepartamento: {depto['nombre']}")
    for emp in depto['empleados']:
        print(f"  - {emp['nombre']} ({emp['cargo']}): {emp['salario']}€")

# Calcular salario total
salario_total = sum(
    emp['salario'] 
    for depto in empresa['departamentos'] 
    for emp in depto['empleados']
)
print(f"\nSalario total empresa: {salario_total}€")

# Encontrar empleado mejor pagado
todos_empleados = [
    emp 
    for depto in empresa['departamentos'] 
    for emp in depto['empleados']
]
mejor_pagado = max(todos_empleados, key=lambda x: x['salario'])
print(f"Mejor pagado: {mejor_pagado['nombre']} - {mejor_pagado['salario']}€")
```

---

## 12. Resumen de Funciones

| Función | Entrada | Salida | Uso |
| :--- | :--- | :--- | :--- |
| `json.loads(s)` | String JSON | Dict/List Python | Parsear string |
| `json.dumps(obj)` | Dict/List Python | String JSON | Serializar a string |
| `json.load(f)` | Archivo JSON | Dict/List Python | Leer archivo |
| `json.dump(obj, f)` | Dict/List + Archivo | - | Escribir archivo |

### Parámetros Útiles de `dumps()` y `dump()`

| Parámetro | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `indent` | Indentación para formato legible | `indent=4` |
| `sort_keys` | Ordenar claves alfabéticamente | `sort_keys=True` |
| `ensure_ascii` | Permitir caracteres no ASCII | `ensure_ascii=False` |
| `cls` | Encoder personalizado | `cls=MiEncoder` |

---

## 13. Buenas Prácticas

```python
# ✅ HACER:
# 1. Usar encoding='utf-8' al abrir archivos
# 2. Manejar excepciones (JSONDecodeError, FileNotFoundError)
# 3. Usar ensure_ascii=False para caracteres especiales
# 4. Validar la estructura del JSON antes de acceder a claves
# 5. Usar .get() para acceso seguro a claves

datos = {"nombre": "Ana"}
edad = datos.get('edad', 0)  # Devuelve 0 si 'edad' no existe

# ❌ NO HACER:
# 1. Confiar ciegamente en la estructura del JSON externo
# 2. Olvidar cerrar archivos (usar 'with')
# 3. Ignorar los códigos de estado en peticiones HTTP
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
