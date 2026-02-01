# 🐍 Unidad 1. Introducción a Python

## 1.1. ¿Qué es Python?


![Ilustración de python intro](../assets/images/python_intro.svg)
**Python** es un lenguaje de programación de alto nivel, interpretado y de propósito general. Fue creado por **Guido van Rossum** y lanzado por primera vez en 1991.

### Características Principales

* **Fácil de aprender:** Su sintaxis es clara y legible, similar al inglés.
* **Versátil:** Se usa en web, ciencia de datos, IA, automatización, juegos...
* **Interpretado:** No necesita compilación, se ejecuta línea por línea.
* **Gran comunidad:** Miles de librerías y recursos disponibles.
* **Gratuito y Open Source:** Puedes usarlo libremente.

### ¿Por qué Python para IA?

Python es el lenguaje dominante en Inteligencia Artificial y Machine Learning por:

1. Librerías especializadas (NumPy, Pandas, TensorFlow, PyTorch).
2. Sintaxis sencilla que permite enfocarse en los algoritmos.
3. Gran comunidad científica y abundante documentación.
4. Integración con otras herramientas y lenguajes.

---

## 1.2. Instalación de Python

### Windows

1. Ve a [python.org/downloads](https://www.python.org/downloads/).
2. Descarga la última versión de Python 3.x.
3. Ejecuta el instalador.
4. **¡IMPORTANTE!** Marca la casilla **"Add Python to PATH"**.
5. Haz clic en "Install Now".

### macOS

Python suele venir preinstalado, pero es recomendable instalar la última versión:

```bash
# Usando Homebrew
brew install python3
```

O descarga el instalador desde [python.org](https://www.python.org/downloads/).

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Verificar la Instalación

Abre una terminal (o CMD en Windows) y escribe:

```bash
python --version
```

O en algunos sistemas:

```bash
python3 --version
```

Deberías ver algo como:

```
Python 3.12.0
```

---

## 1.3. Entornos de Desarrollo (IDEs)

Un **IDE** (Integrated Development Environment) es un programa que facilita escribir código. Aquí tienes las opciones más populares:

### Visual Studio Code (Recomendado)

* **Gratuito** y muy popular.
* Extensiones para Python excelentes.
* Descarga: [code.visualstudio.com](https://code.visualstudio.com/)

Después de instalarlo, instala la extensión "Python" de Microsoft.

### PyCharm

* IDE específico para Python.
* Versión Community gratuita.
* Descarga: [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)

### Jupyter Notebook

* Ideal para ciencia de datos y aprendizaje.
* Permite combinar código con explicaciones.
* Se instala con: `pip install jupyter`

### IDLE

* Viene incluido con Python.
* Básico pero funcional para empezar.

---

## 1.4. Tu Primer Programa

### El Clásico "Hola Mundo"

Crea un archivo llamado `hola.py` y escribe:

```python
print("¡Hola, Mundo!")
```

Para ejecutarlo, abre una terminal en la carpeta donde guardaste el archivo y escribe:

```bash
python hola.py
```

Salida:

```
¡Hola, Mundo!
```

**¡Felicidades!** Acabas de escribir y ejecutar tu primer programa en Python.

### Explicación

* `print()` es una **función** incorporada de Python.
* Muestra en pantalla lo que pongas entre los paréntesis.
* El texto entre comillas se llama **cadena de texto** o **string**.

---

## 1.5. La Función print()

La función `print()` es fundamental. Veamos más ejemplos:

### Imprimir Texto

```python
print("Bienvenido al curso de Python")
print('También puedes usar comillas simples')
```

Salida:

```
Bienvenido al curso de Python
También puedes usar comillas simples
```

### Imprimir Números

```python
print(42)
print(3.14159)
print(-100)
```

Salida:

```
42
3.14159
-100
```

### Imprimir Varios Elementos

```python
print("Mi edad es", 25, "años")
print("Python", "es", "genial")
```

Salida:

```
Mi edad es 25 años
Python es genial
```

### Cambiar el Separador

```python
print("uno", "dos", "tres", sep="-")
print("a", "b", "c", sep=" | ")
```

Salida:

```
uno-dos-tres
a | b | c
```

### Cambiar el Final de Línea

Por defecto, `print()` añade un salto de línea al final. Puedes cambiarlo:

```python
print("Hola", end=" ")
print("Mundo")
```

Salida:

```
Hola Mundo
```

### Imprimir Líneas Vacías

```python
print("Primera línea")
print()  # Línea vacía
print("Segunda línea")
```

Salida:

```
Primera línea

Segunda línea
```

---

## 1.6. Comentarios

Los **comentarios** son texto que Python ignora. Sirven para explicar el código.

### Comentarios de Una Línea

Usa el símbolo `#`:

```python
# Esto es un comentario
print("Hola")  # Esto también es un comentario

# Los comentarios no se ejecutan
# print("Esto no se mostrará")
```

### Comentarios de Varias Líneas

Usa triple comilla:

```python
"""
Este es un comentario
de varias líneas.
Muy útil para explicaciones largas.
"""

'''
También puedes usar
comillas simples triples.
'''
```

### Buenas Prácticas con Comentarios

```python
# MAL: Comentario obvio
x = 5  # Asigna 5 a x

# BIEN: Comentario útil
# Velocidad máxima permitida en zona urbana (km/h)
velocidad_maxima = 50
```

---

## 1.7. La Función input()

La función `input()` permite al usuario introducir datos:

```python
nombre = input("¿Cómo te llamas? ")
print("¡Hola,", nombre + "!")
```

Ejecución:

```
¿Cómo te llamas? Ana
¡Hola, Ana!
```

### Ejemplo: Programa Interactivo

```python
# Programa de saludo personalizado
print("=== Programa de Bienvenida ===")
print()

nombre = input("Introduce tu nombre: ")
ciudad = input("¿De qué ciudad eres? ")

print()
print("¡Bienvenido/a,", nombre + "!")
print("¡Qué bonita es", ciudad + "!")
```

Ejecución:

```
=== Programa de Bienvenida ===

Introduce tu nombre: Carlos
¿De qué ciudad eres? Madrid

¡Bienvenido/a, Carlos!
¡Qué bonita es Madrid!
```

### Nota Importante sobre input()

`input()` **siempre devuelve texto** (string), incluso si el usuario escribe números:

```python
edad = input("¿Cuántos años tienes? ")
print(type(edad))  # <class 'str'>
```

Más adelante veremos cómo convertir ese texto a números.

---

## 1.8. El Modo Interactivo

Puedes usar Python de forma interactiva sin crear archivos. Es útil para pruebas rápidas.

### Iniciar el Modo Interactivo

En la terminal, escribe simplemente:

```bash
python
```

Verás algo como:

```
Python 3.12.0 (main, Oct  2 2024, 00:00:00)
>>> 
```

El `>>>` es el **prompt** donde puedes escribir código:

```python
>>> print("Hola")
Hola
>>> 2 + 2
4
>>> nombre = "Python"
>>> print(nombre)
Python
>>> exit()  # Para salir
```

### Ventajas del Modo Interactivo

* Pruebas rápidas de código.
* Explorar funciones y librerías.
* Calculadora avanzada.

---

## 1.9. Estructura de un Programa Python

Un programa Python típico tiene esta estructura:

```python
# 1. Comentario inicial (descripción del programa)
"""
Programa: Calculadora de edad
Autor: Tu Nombre
Fecha: Enero 2026
"""

# 2. Importaciones (librerías que necesitamos)
# import math  # Ejemplo, no lo usamos aquí

# 3. Definiciones (funciones, clases) - Lo veremos más adelante

# 4. Código principal
print("=== Calculadora de Edad ===")
print()

nombre = input("Tu nombre: ")
anio_nacimiento = input("Año de nacimiento: ")

# Convertimos el año a número (lo veremos en detalle)
anio_actual = 2026
edad = anio_actual - int(anio_nacimiento)

print()
print(f"Hola {nombre}, tienes aproximadamente {edad} años.")
```

---

## 1.10. Errores Comunes de Principiante

### Error de Sintaxis (SyntaxError)

```python
# MAL: Falta cerrar el paréntesis
print("Hola"

# MAL: Falta cerrar las comillas
print("Hola)

# BIEN:
print("Hola")
```

### Error de Nombre (NameError)

```python
# MAL: La variable no existe
print(mensaje)

# BIEN: Primero definimos la variable
mensaje = "Hola"
print(mensaje)
```

### Error de Indentación (IndentationError)

Python usa la indentación (espacios al inicio) para estructurar el código:

```python
# MAL: Indentación incorrecta
print("Hola")
   print("Mundo")  # Error: espacio innecesario

# BIEN:
print("Hola")
print("Mundo")
```

### Consejo: Lee los Mensajes de Error

Python te dice qué está mal y en qué línea:

```
  File "programa.py", line 3
    print("Hola"
               ^
SyntaxError: '(' was never closed
```

Esto te indica:

* El archivo: `programa.py`
* La línea: 3
* El problema: Un paréntesis sin cerrar

---

## 1.11. Ejercicios Prácticos

### Ejercicio 1: Presentación

Crea un programa que muestre tu presentación:

```python
print("========================")
print("    MI PRESENTACIÓN")
print("========================")
print()
print("Nombre: [Tu nombre]")
print("Edad: [Tu edad]")
print("Ciudad: [Tu ciudad]")
print("Hobby: [Tu hobby]")
print()
print("========================")
```

### Ejercicio 2: Datos del Usuario

Crea un programa que pida datos al usuario y los muestre:

```python
print("=== Formulario de Registro ===")
print()

nombre = input("Nombre: ")
apellido = input("Apellido: ")
email = input("Email: ")
telefono = input("Teléfono: ")

print()
print("=== Datos Registrados ===")
print("Nombre completo:", nombre, apellido)
print("Email:", email)
print("Teléfono:", telefono)
```

### Ejercicio 3: Calculadora Simple (Texto)

```python
print("=== Mini Calculadora ===")
print()

numero1 = input("Primer número: ")
numero2 = input("Segundo número: ")

# Por ahora solo mostramos los números
# En la siguiente unidad aprenderemos a hacer operaciones
print()
print("Has introducido:", numero1, "y", numero2)
```

### Ejercicio 4: Generador de Frases

```python
print("=== Generador de Frases ===")
print()

sustantivo = input("Escribe un sustantivo: ")
adjetivo = input("Escribe un adjetivo: ")
verbo = input("Escribe un verbo: ")
lugar = input("Escribe un lugar: ")

print()
print("Tu frase es:")
print(f"El {sustantivo} {adjetivo} {verbo} en {lugar}.")
```

---

## 1.12. Resumen

En esta unidad has aprendido:

| Concepto | Descripción |
| :--- | :--- |
| Python | Lenguaje de programación versátil y fácil de aprender |
| Instalación | Descargar de python.org y añadir al PATH |
| IDE | Programa para escribir código (VS Code recomendado) |
| `print()` | Función para mostrar información en pantalla |
| `input()` | Función para pedir datos al usuario |
| Comentarios | Texto explicativo que Python ignora (`#`) |
| Modo interactivo | Usar Python directamente en la terminal |

---

## 1.13. Próximo Paso

En la siguiente unidad aprenderemos sobre **Variables y Tipos de Datos**, donde descubrirás cómo almacenar información y trabajar con números, texto y otros tipos de datos.

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
