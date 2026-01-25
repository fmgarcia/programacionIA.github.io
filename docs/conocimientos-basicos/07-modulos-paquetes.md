# 📚 Unidad 7. Módulos y Paquetes

Los **módulos** y **paquetes** permiten organizar y reutilizar código en Python. Un módulo es un archivo `.py` con código, y un paquete es una carpeta con módulos.

---

## 7.1. Importar Módulos

### Import Básico

```python
# Importar módulo completo
import math

# Usar funciones del módulo
print(math.sqrt(16))    # 4.0
print(math.pi)          # 3.141592653589793
print(math.ceil(4.2))   # 5
print(math.floor(4.8))  # 4
```

### Import con Alias

```python
import math as m

print(m.sqrt(25))  # 5.0
print(m.pi)        # 3.14159...
```

### Importar Elementos Específicos

```python
from math import sqrt, pi, cos

print(sqrt(36))  # 6.0
print(pi)        # 3.14159...
print(cos(0))    # 1.0
```

### Importar con Alias Específico

```python
from math import sqrt as raiz_cuadrada

print(raiz_cuadrada(49))  # 7.0
```

### Importar Todo (No Recomendado)

```python
from math import *

print(sqrt(64))  # 8.0
print(sin(0))    # 0.0

# No se recomienda porque puede causar conflictos de nombres
```

---

## 7.2. Módulos de la Biblioteca Estándar

Python incluye muchos módulos útiles.

### Módulo `math` - Matemáticas

```python
import math

# Constantes
print(math.pi)   # 3.141592653589793
print(math.e)    # 2.718281828459045
print(math.tau)  # 6.283185307179586 (2π)

# Funciones básicas
print(math.sqrt(16))     # 4.0 (raíz cuadrada)
print(math.pow(2, 8))    # 256.0 (potencia)
print(math.factorial(5)) # 120 (5!)

# Redondeo
print(math.ceil(4.1))    # 5 (hacia arriba)
print(math.floor(4.9))   # 4 (hacia abajo)
print(math.trunc(4.7))   # 4 (truncar)

# Trigonometría (en radianes)
print(math.sin(math.pi/2))  # 1.0
print(math.cos(0))          # 1.0
print(math.degrees(math.pi))  # 180.0 (radianes a grados)
print(math.radians(180))      # 3.14159... (grados a radianes)

# Logaritmos
print(math.log(10))      # 2.302... (ln)
print(math.log10(100))   # 2.0
print(math.log2(8))      # 3.0
```

### Módulo `random` - Números Aleatorios

```python
import random

# Número aleatorio
print(random.random())        # Float entre 0 y 1
print(random.uniform(1, 10))  # Float entre 1 y 10
print(random.randint(1, 100)) # Entero entre 1 y 100 (incluidos)

# Elegir de una secuencia
colores = ["rojo", "verde", "azul"]
print(random.choice(colores))  # Un elemento al azar

# Múltiples elecciones
print(random.choices(colores, k=5))  # 5 elementos (con repetición)
print(random.sample(range(1, 50), k=6))  # 6 elementos (sin repetición)

# Mezclar lista
numeros = [1, 2, 3, 4, 5]
random.shuffle(numeros)  # Mezcla in-place
print(numeros)

# Semilla para reproducibilidad
random.seed(42)
print(random.randint(1, 100))  # Siempre da el mismo resultado
```

### Módulo `datetime` - Fechas y Horas

```python
from datetime import datetime, date, time, timedelta

# Fecha y hora actual
ahora = datetime.now()
print(ahora)  # 2026-01-15 10:30:45.123456

# Solo fecha
hoy = date.today()
print(hoy)  # 2026-01-15

# Crear fecha específica
mi_fecha = date(2026, 12, 25)
print(mi_fecha)  # 2026-12-25

# Crear datetime específico
mi_datetime = datetime(2026, 12, 25, 18, 30)
print(mi_datetime)  # 2026-12-25 18:30:00

# Acceder a componentes
print(ahora.year)   # 2026
print(ahora.month)  # 1
print(ahora.day)    # 15
print(ahora.hour)   # 10
print(ahora.minute) # 30

# Formatear fechas
print(ahora.strftime("%d/%m/%Y"))        # 15/01/2026
print(ahora.strftime("%H:%M:%S"))        # 10:30:45
print(ahora.strftime("%A, %d de %B"))    # Wednesday, 15 de January

# Parsear strings a fechas
fecha_str = "25/12/2026"
fecha = datetime.strptime(fecha_str, "%d/%m/%Y")
print(fecha)  # 2026-12-25 00:00:00

# Operaciones con fechas
manana = hoy + timedelta(days=1)
hace_una_semana = hoy - timedelta(weeks=1)
en_dos_horas = ahora + timedelta(hours=2)

# Diferencia entre fechas
navidad = date(2026, 12, 25)
dias_restantes = navidad - hoy
print(f"Faltan {dias_restantes.days} días para Navidad")
```

### Módulo `os` - Sistema Operativo

```python
import os

# Directorio actual
print(os.getcwd())  # C:\Users\...

# Cambiar directorio
# os.chdir("/ruta/nueva")

# Listar archivos
archivos = os.listdir(".")
print(archivos)

# Verificar existencia
print(os.path.exists("archivo.txt"))
print(os.path.isfile("archivo.txt"))
print(os.path.isdir("carpeta"))

# Crear directorio
# os.mkdir("nueva_carpeta")      # Una carpeta
# os.makedirs("ruta/nueva/completa")  # Varias carpetas

# Eliminar
# os.remove("archivo.txt")       # Archivo
# os.rmdir("carpeta_vacia")      # Carpeta vacía

# Variables de entorno
print(os.environ.get("PATH"))
print(os.environ.get("HOME"))

# Información del sistema
print(os.name)       # 'nt' (Windows) o 'posix' (Linux/Mac)
print(os.sep)        # '\' (Windows) o '/' (Linux/Mac)

# Rutas
ruta = os.path.join("carpeta", "subcarpeta", "archivo.txt")
print(ruta)  # carpeta\subcarpeta\archivo.txt

print(os.path.basename("/ruta/al/archivo.txt"))  # archivo.txt
print(os.path.dirname("/ruta/al/archivo.txt"))   # /ruta/al
print(os.path.splitext("documento.pdf"))         # ('documento', '.pdf')
```

### Módulo `sys` - Sistema

```python
import sys

# Versión de Python
print(sys.version)         # 3.12.0 ...
print(sys.version_info)    # (3, 12, 0, 'final', 0)

# Argumentos de línea de comandos
# python script.py arg1 arg2
print(sys.argv)  # ['script.py', 'arg1', 'arg2']

# Rutas de búsqueda de módulos
print(sys.path)

# Salir del programa
# sys.exit(0)  # 0 = éxito, otro número = error
```

### Módulo `json` - JSON

```python
import json

# Diccionario a JSON (serializar)
datos = {
    "nombre": "Ana",
    "edad": 30,
    "ciudades": ["Madrid", "Barcelona"]
}

json_str = json.dumps(datos)
print(json_str)  # {"nombre": "Ana", "edad": 30, ...}

# Con formato legible
json_bonito = json.dumps(datos, indent=4, ensure_ascii=False)
print(json_bonito)

# JSON a diccionario (deserializar)
json_texto = '{"nombre": "Luis", "edad": 25}'
datos = json.loads(json_texto)
print(datos["nombre"])  # Luis

# Guardar JSON en archivo
with open("datos.json", "w") as f:
    json.dump(datos, f, indent=4)

# Leer JSON de archivo
with open("datos.json", "r") as f:
    datos_leidos = json.load(f)
```

### Módulo `re` - Expresiones Regulares

```python
import re

texto = "Mi email es usuario@ejemplo.com y mi teléfono 612345678"

# Buscar patrón
resultado = re.search(r"\d+", texto)  # Buscar dígitos
if resultado:
    print(resultado.group())  # 612345678

# Encontrar todos
emails = "contacto@ejemplo.com, info@empresa.es"
patron_email = r"\b[\w.-]+@[\w.-]+\.\w+\b"
encontrados = re.findall(patron_email, emails)
print(encontrados)  # ['contacto@ejemplo.com', 'info@empresa.es']

# Reemplazar
texto = "Hola 123 Mundo 456"
limpio = re.sub(r"\d+", "XXX", texto)
print(limpio)  # Hola XXX Mundo XXX

# Validar formato
def validar_email(email):
    patron = r"^[\w.-]+@[\w.-]+\.\w{2,}$"
    return bool(re.match(patron, email))

print(validar_email("test@ejemplo.com"))  # True
print(validar_email("invalido"))          # False
```

### Módulo `collections` - Colecciones Especiales

```python
from collections import Counter, defaultdict, namedtuple, deque

# Counter - Contar elementos
palabras = ["a", "b", "a", "c", "a", "b"]
contador = Counter(palabras)
print(contador)  # Counter({'a': 3, 'b': 2, 'c': 1})
print(contador.most_common(2))  # [('a', 3), ('b', 2)]

# defaultdict - Diccionario con valor por defecto
dd = defaultdict(list)
dd["frutas"].append("manzana")
dd["frutas"].append("naranja")
dd["verduras"].append("zanahoria")
print(dict(dd))  # {'frutas': ['manzana', 'naranja'], 'verduras': ['zanahoria']}

# namedtuple - Tupla con nombres
Punto = namedtuple("Punto", ["x", "y"])
p = Punto(10, 20)
print(p.x, p.y)  # 10 20

# deque - Cola de doble extremo (eficiente)
cola = deque([1, 2, 3])
cola.append(4)      # Añadir al final
cola.appendleft(0)  # Añadir al inicio
print(cola)  # deque([0, 1, 2, 3, 4])
```

---

## 7.3. Crear Módulos Propios

### Módulo Simple

Crea un archivo `mi_modulo.py`:

```python
# mi_modulo.py
"""Mi módulo personalizado con funciones útiles."""

PI = 3.14159

def saludar(nombre):
    """Saluda a una persona."""
    return f"¡Hola, {nombre}!"

def area_circulo(radio):
    """Calcula el área de un círculo."""
    return PI * radio ** 2

def es_par(numero):
    """Verifica si un número es par."""
    return numero % 2 == 0
```

Usa el módulo en otro archivo:

```python
# main.py
import mi_modulo

print(mi_modulo.saludar("Ana"))
print(mi_modulo.area_circulo(5))
print(mi_modulo.PI)

# O importar específico
from mi_modulo import saludar, es_par
print(saludar("Luis"))
print(es_par(7))  # False
```

### Bloque `__name__`

```python
# calculadora.py
def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b

# Código que solo se ejecuta si se ejecuta este archivo directamente
if __name__ == "__main__":
    # Pruebas
    print("Probando calculadora...")
    print(f"5 + 3 = {sumar(5, 3)}")
    print(f"10 - 4 = {restar(10, 4)}")
```

Si ejecutas `python calculadora.py`, se ejecutan las pruebas.
Si haces `import calculadora`, las pruebas NO se ejecutan.

---

## 7.4. Crear Paquetes

Un **paquete** es una carpeta con un archivo `__init__.py`.

### Estructura de Paquete

```
mi_paquete/
    __init__.py
    matematicas.py
    texto.py
    utilidades/
        __init__.py
        archivos.py
```

### Contenido de los Archivos

`mi_paquete/__init__.py`:
```python
"""Mi paquete de utilidades."""
from .matematicas import sumar, restar
from .texto import mayusculas
```

`mi_paquete/matematicas.py`:
```python
def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b

def multiplicar(a, b):
    return a * b
```

`mi_paquete/texto.py`:
```python
def mayusculas(texto):
    return texto.upper()

def minusculas(texto):
    return texto.lower()

def invertir(texto):
    return texto[::-1]
```

### Usar el Paquete

```python
# Importar desde el paquete
import mi_paquete
print(mi_paquete.sumar(5, 3))

# Importar módulo específico
from mi_paquete import matematicas
print(matematicas.multiplicar(4, 5))

# Importar función específica
from mi_paquete.texto import invertir
print(invertir("Python"))  # nohtyP

# Importar subpaquete
from mi_paquete.utilidades import archivos
```

---

## 7.5. Gestión de Paquetes con pip

**pip** es el gestor de paquetes de Python.

### Comandos Básicos

```bash
# Ver versión de pip
pip --version

# Instalar paquete
pip install nombre_paquete
pip install numpy
pip install pandas==1.5.0  # Versión específica

# Desinstalar paquete
pip uninstall nombre_paquete

# Actualizar paquete
pip install --upgrade nombre_paquete

# Ver paquetes instalados
pip list

# Información de un paquete
pip show numpy

# Buscar paquetes (en PyPI)
pip search nombre  # Nota: puede estar deshabilitado
```

### Archivo requirements.txt

```bash
# Crear archivo con dependencias actuales
pip freeze > requirements.txt

# Instalar desde requirements.txt
pip install -r requirements.txt
```

Ejemplo de `requirements.txt`:

```
numpy==1.24.0
pandas>=2.0.0
matplotlib
scikit-learn~=1.2.0
```

---

## 7.6. Entornos Virtuales

Los **entornos virtuales** aíslan las dependencias de cada proyecto.

### Crear Entorno Virtual

```bash
# Windows
python -m venv mi_entorno

# Linux/Mac
python3 -m venv mi_entorno
```

### Activar Entorno

```bash
# Windows
mi_entorno\Scripts\activate

# Linux/Mac
source mi_entorno/bin/activate
```

### Desactivar Entorno

```bash
deactivate
```

### Flujo de Trabajo Típico

```bash
# 1. Crear entorno
python -m venv venv

# 2. Activar
venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install numpy pandas matplotlib

# 4. Guardar dependencias
pip freeze > requirements.txt

# 5. Trabajar en el proyecto...

# 6. Desactivar cuando termines
deactivate
```

### Estructura de Proyecto Recomendada

```
mi_proyecto/
    venv/                 # Entorno virtual (no subir a Git)
    src/                  # Código fuente
        __init__.py
        main.py
        utils.py
    tests/                # Pruebas
        test_main.py
    requirements.txt      # Dependencias
    README.md
    .gitignore
```

`.gitignore` típico:
```
venv/
__pycache__/
*.pyc
.env
```

---

## 7.7. Ejercicios Prácticos

### Ejercicio 1: Generador de Contraseñas

```python
import random
import string

def generar_contrasena(longitud=12, incluir_mayusculas=True, 
                       incluir_numeros=True, incluir_simbolos=True):
    """
    Genera una contraseña aleatoria.
    
    Args:
        longitud: Longitud de la contraseña
        incluir_mayusculas: Si incluir letras mayúsculas
        incluir_numeros: Si incluir dígitos
        incluir_simbolos: Si incluir símbolos especiales
    
    Returns:
        str: Contraseña generada
    """
    caracteres = string.ascii_lowercase  # a-z
    
    if incluir_mayusculas:
        caracteres += string.ascii_uppercase  # A-Z
    if incluir_numeros:
        caracteres += string.digits  # 0-9
    if incluir_simbolos:
        caracteres += string.punctuation  # !@#$%...
    
    contrasena = ''.join(random.choice(caracteres) for _ in range(longitud))
    return contrasena

# Generar varias contraseñas
print("Contraseñas generadas:")
for i in range(5):
    print(f"  {i+1}. {generar_contrasena()}")

# Contraseña solo con letras y números
print(f"\nSin símbolos: {generar_contrasena(16, incluir_simbolos=False)}")
```

### Ejercicio 2: Análisis de Fechas

```python
from datetime import datetime, timedelta

def dias_entre_fechas(fecha1_str, fecha2_str, formato="%d/%m/%Y"):
    """Calcula días entre dos fechas."""
    fecha1 = datetime.strptime(fecha1_str, formato)
    fecha2 = datetime.strptime(fecha2_str, formato)
    diferencia = abs((fecha2 - fecha1).days)
    return diferencia

def edad(fecha_nacimiento_str, formato="%d/%m/%Y"):
    """Calcula la edad en años."""
    nacimiento = datetime.strptime(fecha_nacimiento_str, formato)
    hoy = datetime.now()
    edad = hoy.year - nacimiento.year
    # Ajustar si no ha llegado el cumpleaños este año
    if (hoy.month, hoy.day) < (nacimiento.month, nacimiento.day):
        edad -= 1
    return edad

def siguiente_cumpleanos(fecha_nacimiento_str, formato="%d/%m/%Y"):
    """Calcula días hasta el próximo cumpleaños."""
    nacimiento = datetime.strptime(fecha_nacimiento_str, formato)
    hoy = datetime.now()
    
    # Cumpleaños este año
    cumple_este_ano = nacimiento.replace(year=hoy.year)
    
    # Si ya pasó, calcular para el próximo año
    if cumple_este_ano.date() < hoy.date():
        cumple_este_ano = nacimiento.replace(year=hoy.year + 1)
    
    dias = (cumple_este_ano.date() - hoy.date()).days
    return dias

# Pruebas
print(f"Días entre fechas: {dias_entre_fechas('01/01/2020', '31/12/2020')}")
print(f"Mi edad: {edad('15/06/1990')}")
print(f"Días hasta mi cumpleaños: {siguiente_cumpleanos('15/06/1990')}")
```

### Ejercicio 3: Módulo de Estadísticas

Crea `estadisticas.py`:

```python
"""Módulo de estadísticas básicas."""

def media(datos):
    """Calcula la media aritmética."""
    if not datos:
        return None
    return sum(datos) / len(datos)

def mediana(datos):
    """Calcula la mediana."""
    if not datos:
        return None
    ordenados = sorted(datos)
    n = len(ordenados)
    medio = n // 2
    if n % 2 == 0:
        return (ordenados[medio - 1] + ordenados[medio]) / 2
    return ordenados[medio]

def moda(datos):
    """Calcula la moda."""
    if not datos:
        return None
    from collections import Counter
    contador = Counter(datos)
    max_freq = max(contador.values())
    modas = [k for k, v in contador.items() if v == max_freq]
    return modas[0] if len(modas) == 1 else modas

def varianza(datos):
    """Calcula la varianza."""
    if not datos:
        return None
    m = media(datos)
    return sum((x - m) ** 2 for x in datos) / len(datos)

def desviacion_estandar(datos):
    """Calcula la desviación estándar."""
    var = varianza(datos)
    return var ** 0.5 if var is not None else None

def resumen(datos):
    """Genera un resumen estadístico."""
    return {
        "n": len(datos),
        "min": min(datos),
        "max": max(datos),
        "media": media(datos),
        "mediana": mediana(datos),
        "moda": moda(datos),
        "desv_std": desviacion_estandar(datos)
    }

if __name__ == "__main__":
    # Pruebas
    datos = [4, 7, 2, 9, 4, 1, 4, 8, 3, 4]
    print("Datos:", datos)
    for clave, valor in resumen(datos).items():
        print(f"  {clave}: {valor}")
```

---

## 7.8. Resumen

| Concepto | Ejemplo |
| :--- | :--- |
| Import módulo | `import math` |
| Import con alias | `import numpy as np` |
| Import específico | `from math import sqrt` |
| Crear módulo | Archivo `.py` con funciones |
| Crear paquete | Carpeta con `__init__.py` |
| Instalar paquete | `pip install nombre` |
| Entorno virtual | `python -m venv venv` |
| Activar entorno | `venv\Scripts\activate` |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
