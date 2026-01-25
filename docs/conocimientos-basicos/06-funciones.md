# 📚 Unidad 6. Funciones

Las **funciones** son bloques de código reutilizables que realizan una tarea específica. Permiten organizar el código, evitar repeticiones y facilitar el mantenimiento.

---

## 6.1. Definición de Funciones

### Sintaxis Básica

```python
def nombre_funcion():
    # Código de la función
    pass
```

### Primera Función

```python
def saludar():
    print("¡Hola, mundo!")

# Llamar a la función
saludar()  # ¡Hola, mundo!
saludar()  # ¡Hola, mundo!
```

### Funciones con Parámetros

```python
def saludar(nombre):
    print(f"¡Hola, {nombre}!")

saludar("Ana")    # ¡Hola, Ana!
saludar("Luis")   # ¡Hola, Luis!

# Varios parámetros
def sumar(a, b):
    print(f"{a} + {b} = {a + b}")

sumar(5, 3)   # 5 + 3 = 8
sumar(10, 20) # 10 + 20 = 30
```

---

## 6.2. Retorno de Valores

Las funciones pueden devolver valores usando `return`.

```python
def sumar(a, b):
    return a + b

resultado = sumar(5, 3)
print(resultado)  # 8

# Usar directamente en expresiones
print(sumar(10, 20))  # 30
total = sumar(1, 2) + sumar(3, 4)  # 3 + 7 = 10
```

### Return Detiene la Función

```python
def dividir(a, b):
    if b == 0:
        return "Error: división por cero"
    return a / b

print(dividir(10, 2))  # 5.0
print(dividir(10, 0))  # Error: división por cero
```

### Retornar Múltiples Valores

```python
def operaciones(a, b):
    suma = a + b
    resta = a - b
    multiplicacion = a * b
    return suma, resta, multiplicacion  # Retorna tupla

resultado = operaciones(10, 3)
print(resultado)  # (13, 7, 30)

# Desempaquetando
s, r, m = operaciones(10, 3)
print(f"Suma: {s}, Resta: {r}, Mult: {m}")
```

### Funciones sin Return

```python
def mostrar_info(nombre, edad):
    print(f"Nombre: {nombre}")
    print(f"Edad: {edad}")

resultado = mostrar_info("Ana", 25)
print(resultado)  # None (no hay return explícito)
```

---

## 6.3. Tipos de Parámetros

### Parámetros Posicionales

```python
def presentar(nombre, edad, ciudad):
    print(f"{nombre}, {edad} años, de {ciudad}")

# El orden importa
presentar("Ana", 25, "Madrid")
# presentar(25, "Ana", "Madrid")  # Incorrecto
```

### Parámetros con Nombre (Keyword Arguments)

```python
def presentar(nombre, edad, ciudad):
    print(f"{nombre}, {edad} años, de {ciudad}")

# Se pueden pasar en cualquier orden
presentar(edad=25, ciudad="Madrid", nombre="Ana")
presentar("Ana", ciudad="Madrid", edad=25)
```

### Parámetros con Valores por Defecto

```python
def saludar(nombre, saludo="Hola"):
    print(f"{saludo}, {nombre}!")

saludar("Ana")           # Hola, Ana!
saludar("Luis", "Buenos días")  # Buenos días, Luis!

# Otro ejemplo
def potencia(base, exponente=2):
    return base ** exponente

print(potencia(5))      # 25 (5²)
print(potencia(5, 3))   # 125 (5³)
```

⚠️ **Importante:** Los parámetros con valor por defecto deben ir al final.

```python
# Correcto
def funcion(a, b, c=10):
    pass

# Incorrecto
# def funcion(a, b=10, c):  # SyntaxError
```

---

## 6.4. Parámetros Arbitrarios

### *args (Argumentos Posicionales Variables)

```python
def sumar_todos(*numeros):
    print(f"Tipo: {type(numeros)}")  # <class 'tuple'>
    return sum(numeros)

print(sumar_todos(1, 2))           # 3
print(sumar_todos(1, 2, 3, 4, 5))  # 15
print(sumar_todos())               # 0
```

### **kwargs (Argumentos con Nombre Variables)

```python
def mostrar_datos(**datos):
    print(f"Tipo: {type(datos)}")  # <class 'dict'>
    for clave, valor in datos.items():
        print(f"  {clave}: {valor}")

mostrar_datos(nombre="Ana", edad=25)
# nombre: Ana
# edad: 25

mostrar_datos(ciudad="Madrid", pais="España", codigo=28001)
```

### Combinando Todos los Tipos

```python
def funcion_completa(a, b, *args, opcion=True, **kwargs):
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"args = {args}")
    print(f"opcion = {opcion}")
    print(f"kwargs = {kwargs}")

funcion_completa(1, 2, 3, 4, 5, opcion=False, x=10, y=20)
# a = 1
# b = 2
# args = (3, 4, 5)
# opcion = False
# kwargs = {'x': 10, 'y': 20}
```

### Desempaquetado de Argumentos

```python
def sumar(a, b, c):
    return a + b + c

# Desempaquetar lista/tupla con *
numeros = [1, 2, 3]
print(sumar(*numeros))  # 6

# Desempaquetar diccionario con **
datos = {"a": 10, "b": 20, "c": 30}
print(sumar(**datos))  # 60
```

---

## 6.5. Ámbito de Variables (Scope)

### Variables Locales y Globales

```python
# Variable global
mensaje = "Soy global"

def funcion():
    # Variable local
    mensaje_local = "Soy local"
    print(mensaje)        # Accede a global
    print(mensaje_local)  # Accede a local

funcion()
print(mensaje)  # "Soy global"
# print(mensaje_local)  # Error: no existe fuera de la función
```

### Modificar Variables Globales

```python
contador = 0

def incrementar():
    global contador  # Indicamos que usamos la global
    contador += 1

print(contador)  # 0
incrementar()
print(contador)  # 1
incrementar()
print(contador)  # 2
```

### La Regla LEGB

Python busca variables en este orden:

1. **L**ocal - Dentro de la función actual
2. **E**nclosing - Funciones que contienen a la actual
3. **G**lobal - Módulo actual
4. **B**uilt-in - Funciones integradas de Python

```python
x = "global"

def externa():
    x = "enclosing"
    
    def interna():
        x = "local"
        print(x)  # local
    
    interna()
    print(x)  # enclosing

externa()
print(x)  # global
```

---

## 6.6. Funciones Lambda

Las **funciones lambda** son funciones anónimas de una sola expresión.

### Sintaxis

```python
lambda argumentos: expresión
```

### Ejemplos Básicos

```python
# Función normal
def cuadrado(x):
    return x ** 2

# Equivalente con lambda
cuadrado_lambda = lambda x: x ** 2

print(cuadrado(5))        # 25
print(cuadrado_lambda(5)) # 25

# Lambda con varios argumentos
sumar = lambda a, b: a + b
print(sumar(3, 4))  # 7

# Lambda sin argumentos
saludar = lambda: "¡Hola!"
print(saludar())  # ¡Hola!
```

### Lambda con Funciones de Orden Superior

```python
numeros = [1, 2, 3, 4, 5]

# map() - Aplicar función a cada elemento
cuadrados = list(map(lambda x: x**2, numeros))
print(cuadrados)  # [1, 4, 9, 16, 25]

# filter() - Filtrar elementos
pares = list(filter(lambda x: x % 2 == 0, numeros))
print(pares)  # [2, 4]

# sorted() con key
palabras = ["Python", "es", "genial"]
ordenadas = sorted(palabras, key=lambda x: len(x))
print(ordenadas)  # ['es', 'Python', 'genial']

# Ordenar diccionarios
estudiantes = [
    {"nombre": "Ana", "nota": 8},
    {"nombre": "Luis", "nota": 9},
    {"nombre": "María", "nota": 7}
]
por_nota = sorted(estudiantes, key=lambda x: x["nota"], reverse=True)
print(por_nota)  # Ordenados por nota descendente
```

---

## 6.7. Funciones de Orden Superior

Son funciones que reciben otras funciones como argumentos o las devuelven.

### map()

```python
# Aplicar función a cada elemento
numeros = [1, 2, 3, 4, 5]

def duplicar(x):
    return x * 2

duplicados = list(map(duplicar, numeros))
print(duplicados)  # [2, 4, 6, 8, 10]

# Con lambda
triplicados = list(map(lambda x: x * 3, numeros))
print(triplicados)  # [3, 6, 9, 12, 15]

# Con múltiples iterables
lista1 = [1, 2, 3]
lista2 = [10, 20, 30]
sumas = list(map(lambda a, b: a + b, lista1, lista2))
print(sumas)  # [11, 22, 33]
```

### filter()

```python
# Filtrar elementos que cumplen condición
numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def es_par(x):
    return x % 2 == 0

pares = list(filter(es_par, numeros))
print(pares)  # [2, 4, 6, 8, 10]

# Con lambda
impares = list(filter(lambda x: x % 2 != 0, numeros))
print(impares)  # [1, 3, 5, 7, 9]

# Filtrar strings
palabras = ["hola", "", "mundo", "", "python"]
no_vacias = list(filter(None, palabras))  # None filtra valores falsy
print(no_vacias)  # ['hola', 'mundo', 'python']
```

### reduce()

```python
from functools import reduce

# Acumular valores
numeros = [1, 2, 3, 4, 5]

# Suma acumulada
suma = reduce(lambda a, b: a + b, numeros)
print(suma)  # 15

# Producto
producto = reduce(lambda a, b: a * b, numeros)
print(producto)  # 120

# Encontrar máximo (manualmente)
maximo = reduce(lambda a, b: a if a > b else b, numeros)
print(maximo)  # 5
```

### zip()

```python
nombres = ["Ana", "Luis", "María"]
edades = [25, 30, 28]
ciudades = ["Madrid", "Barcelona", "Valencia"]

# Combinar listas
combinados = list(zip(nombres, edades, ciudades))
print(combinados)
# [('Ana', 25, 'Madrid'), ('Luis', 30, 'Barcelona'), ('María', 28, 'Valencia')]

# Iterar
for nombre, edad, ciudad in zip(nombres, edades, ciudades):
    print(f"{nombre}, {edad} años, de {ciudad}")

# Crear diccionario
datos = dict(zip(nombres, edades))
print(datos)  # {'Ana': 25, 'Luis': 30, 'María': 28}
```

---

## 6.8. Funciones Recursivas

Una función **recursiva** es aquella que se llama a sí misma.

### Factorial

```python
# n! = n × (n-1) × (n-2) × ... × 1
def factorial(n):
    if n == 0 or n == 1:  # Caso base
        return 1
    return n * factorial(n - 1)  # Caso recursivo

print(factorial(5))  # 120 (5×4×3×2×1)
print(factorial(0))  # 1
```

### Fibonacci

```python
# 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(10):
    print(fibonacci(i), end=" ")
# 0 1 1 2 3 5 8 13 21 34
```

### Suma de Lista

```python
def suma_recursiva(lista):
    if len(lista) == 0:  # Caso base
        return 0
    return lista[0] + suma_recursiva(lista[1:])

print(suma_recursiva([1, 2, 3, 4, 5]))  # 15
```

### Búsqueda Binaria

```python
def busqueda_binaria(lista, objetivo, inicio=0, fin=None):
    if fin is None:
        fin = len(lista) - 1
    
    if inicio > fin:
        return -1  # No encontrado
    
    medio = (inicio + fin) // 2
    
    if lista[medio] == objetivo:
        return medio
    elif lista[medio] < objetivo:
        return busqueda_binaria(lista, objetivo, medio + 1, fin)
    else:
        return busqueda_binaria(lista, objetivo, inicio, medio - 1)

numeros = [1, 3, 5, 7, 9, 11, 13, 15]
print(busqueda_binaria(numeros, 7))   # 3 (índice)
print(busqueda_binaria(numeros, 6))   # -1 (no está)
```

---

## 6.9. Documentación de Funciones

### Docstrings

```python
def calcular_area_rectangulo(base, altura):
    """
    Calcula el área de un rectángulo.
    
    Args:
        base (float): La base del rectángulo.
        altura (float): La altura del rectángulo.
    
    Returns:
        float: El área del rectángulo.
    
    Example:
        >>> calcular_area_rectangulo(5, 3)
        15
    """
    return base * altura

# Acceder a la documentación
print(calcular_area_rectangulo.__doc__)
help(calcular_area_rectangulo)
```

### Type Hints (Anotaciones de Tipo)

```python
def saludar(nombre: str) -> str:
    return f"Hola, {nombre}"

def sumar(a: int, b: int) -> int:
    return a + b

def procesar_datos(
    datos: list[dict],
    filtro: str = None
) -> list[dict]:
    """Procesa una lista de diccionarios."""
    if filtro:
        return [d for d in datos if filtro in str(d)]
    return datos

# Los type hints son solo indicaciones, no obligan
resultado = sumar(3.5, 2.5)  # Funciona aunque sean floats
print(resultado)  # 6.0
```

---

## 6.10. Decoradores (Introducción)

Los **decoradores** modifican el comportamiento de funciones.

```python
# Decorador simple
def mi_decorador(funcion):
    def wrapper():
        print("Antes de la función")
        funcion()
        print("Después de la función")
    return wrapper

@mi_decorador
def saludar():
    print("¡Hola!")

saludar()
# Antes de la función
# ¡Hola!
# Después de la función
```

### Decorador con Argumentos

```python
def mi_decorador(funcion):
    def wrapper(*args, **kwargs):
        print("Inicio")
        resultado = funcion(*args, **kwargs)
        print("Fin")
        return resultado
    return wrapper

@mi_decorador
def sumar(a, b):
    return a + b

resultado = sumar(3, 4)
# Inicio
# Fin
print(resultado)  # 7
```

### Decorador para Medir Tiempo

```python
import time

def medir_tiempo(funcion):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        fin = time.time()
        print(f"{funcion.__name__} tardó {fin - inicio:.4f} segundos")
        return resultado
    return wrapper

@medir_tiempo
def operacion_lenta():
    time.sleep(1)
    return "Completado"

resultado = operacion_lenta()
# operacion_lenta tardó 1.0012 segundos
```

---

## 6.11. Ejercicios Prácticos

### Ejercicio 1: Calculadora con Funciones

```python
def sumar(a, b):
    return a + b

def restar(a, b):
    return a - b

def multiplicar(a, b):
    return a * b

def dividir(a, b):
    if b == 0:
        return "Error: División por cero"
    return a / b

def potencia(a, b):
    return a ** b

def calculadora():
    while True:
        print("\n=== CALCULADORA ===")
        print("1. Sumar")
        print("2. Restar")
        print("3. Multiplicar")
        print("4. Dividir")
        print("5. Potencia")
        print("6. Salir")
        
        opcion = input("Opción: ")
        
        if opcion == "6":
            print("¡Adiós!")
            break
        
        if opcion not in "12345":
            print("Opción no válida")
            continue
        
        try:
            a = float(input("Primer número: "))
            b = float(input("Segundo número: "))
        except ValueError:
            print("Error: Introduce números válidos")
            continue
        
        operaciones = {
            "1": ("Suma", sumar),
            "2": ("Resta", restar),
            "3": ("Multiplicación", multiplicar),
            "4": ("División", dividir),
            "5": ("Potencia", potencia)
        }
        
        nombre, funcion = operaciones[opcion]
        resultado = funcion(a, b)
        print(f"{nombre}: {resultado}")

calculadora()
```

### Ejercicio 2: Validador de Datos

```python
def validar_email(email):
    """Valida formato básico de email."""
    if "@" not in email:
        return False, "Falta @"
    partes = email.split("@")
    if len(partes) != 2:
        return False, "Formato incorrecto"
    if "." not in partes[1]:
        return False, "Dominio inválido"
    return True, "Email válido"

def validar_telefono(telefono):
    """Valida teléfono español (9 dígitos)."""
    numeros = telefono.replace(" ", "").replace("-", "")
    if not numeros.isdigit():
        return False, "Solo dígitos permitidos"
    if len(numeros) != 9:
        return False, "Debe tener 9 dígitos"
    return True, "Teléfono válido"

def validar_edad(edad):
    """Valida que la edad sea razonable."""
    try:
        edad = int(edad)
        if edad < 0:
            return False, "La edad no puede ser negativa"
        if edad > 150:
            return False, "Edad no realista"
        return True, "Edad válida"
    except ValueError:
        return False, "Debe ser un número"

# Pruebas
print(validar_email("usuario@ejemplo.com"))  # (True, 'Email válido')
print(validar_email("sin_arroba.com"))       # (False, 'Falta @')
print(validar_telefono("612 345 678"))       # (True, 'Teléfono válido')
print(validar_edad("25"))                    # (True, 'Edad válida')
```

### Ejercicio 3: Funciones Estadísticas

```python
def media(numeros):
    """Calcula la media aritmética."""
    if not numeros:
        return None
    return sum(numeros) / len(numeros)

def mediana(numeros):
    """Calcula la mediana."""
    if not numeros:
        return None
    ordenados = sorted(numeros)
    n = len(ordenados)
    medio = n // 2
    if n % 2 == 0:
        return (ordenados[medio - 1] + ordenados[medio]) / 2
    return ordenados[medio]

def moda(numeros):
    """Calcula la moda (valor más frecuente)."""
    if not numeros:
        return None
    frecuencias = {}
    for n in numeros:
        frecuencias[n] = frecuencias.get(n, 0) + 1
    max_freq = max(frecuencias.values())
    modas = [k for k, v in frecuencias.items() if v == max_freq]
    return modas[0] if len(modas) == 1 else modas

def desviacion_estandar(numeros):
    """Calcula la desviación estándar."""
    if not numeros:
        return None
    m = media(numeros)
    varianza = sum((x - m) ** 2 for x in numeros) / len(numeros)
    return varianza ** 0.5

# Pruebas
datos = [4, 7, 2, 9, 4, 1, 4, 8, 3, 4]
print(f"Media: {media(datos):.2f}")            # 4.60
print(f"Mediana: {mediana(datos)}")            # 4.0
print(f"Moda: {moda(datos)}")                  # 4
print(f"Desv. Estándar: {desviacion_estandar(datos):.2f}")  # 2.33
```

---

## 6.12. Resumen

| Concepto | Descripción |
| :--- | :--- |
| `def funcion():` | Definir función |
| `return valor` | Devolver valor |
| `param=valor` | Parámetro con valor por defecto |
| `*args` | Argumentos posicionales variables |
| `**kwargs` | Argumentos con nombre variables |
| `lambda x: x*2` | Función anónima |
| `map()` | Aplicar función a iterables |
| `filter()` | Filtrar elementos |
| `reduce()` | Reducir a un valor |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
