# 📦 Unidad 2. Variables y Tipos de Datos

## 2.1. ¿Qué es una Variable?

Una **variable** es un espacio en la memoria del ordenador donde guardamos información. En Python, técnicamente son **etiquetas** que hacen referencia a objetos en memoria.

![Modelo de Memoria de Variables](../assets/images/python_variables.svg)

```python
# Creamos una variable llamada "edad" y le asignamos el valor 25
edad = 25

# Ahora podemos usar esa variable
print(edad)  # Muestra: 25
```

### Asignación de Variables

El símbolo `=` se usa para **asignar** un valor a una variable:

```python
nombre = "Ana"        # Texto
edad = 30             # Número entero
altura = 1.75         # Número decimal
es_estudiante = True  # Valor booleano (verdadero/falso)
```

### Reglas para Nombrar Variables

| Regla | Ejemplo Válido | Ejemplo Inválido |
| :--- | :--- | :--- |
| Debe empezar con letra o `_` | `nombre`, `_edad` | `1nombre` |
| Solo letras, números y `_` | `mi_variable`, `edad2` | `mi-variable`, `edad@` |
| No puede ser palabra reservada | `mi_class` | `class`, `if`, `for` |
| Distingue mayúsculas/minúsculas | `Nombre` ≠ `nombre` | - |

### Convenciones de Nombres (PEP 8)

```python
# BIEN: snake_case (recomendado en Python)
nombre_completo = "Juan García"
numero_de_telefono = "123456789"
es_mayor_de_edad = True

# MAL: Otros estilos (válidos pero no recomendados en Python)
NombreCompleto = "Juan García"     # PascalCase (se usa para clases)
numeroTelefono = "123456789"       # camelCase (se usa en otros lenguajes)
```

### Ejemplos de Asignación

```python
# Asignación simple
x = 10
mensaje = "Hola Mundo"

# Asignación múltiple
a, b, c = 1, 2, 3
print(a)  # 1
print(b)  # 2
print(c)  # 3

# Mismo valor a varias variables
x = y = z = 0
print(x, y, z)  # 0 0 0

# Intercambiar valores (muy útil en Python)
a = 5
b = 10
a, b = b, a  # Intercambio
print(a)  # 10
print(b)  # 5
```

---

## 2.2. Tipos de Datos Básicos

Python tiene varios tipos de datos fundamentales:

![Jerarquía de Tipos Python](../assets/images/python_types.svg)

| Tipo | Nombre en Python | Ejemplo |
| :--- | :--- | :--- |
| Entero | `int` | `42`, `-7`, `0` |
| Decimal | `float` | `3.14`, `-0.5`, `2.0` |
| Texto | `str` | `"Hola"`, `'Python'` |
| Booleano | `bool` | `True`, `False` |
| Nulo | `NoneType` | `None` |

### Función type()

Puedes saber el tipo de una variable con `type()`:

```python
edad = 25
print(type(edad))  # <class 'int'>

precio = 19.99
print(type(precio))  # <class 'float'>

nombre = "Python"
print(type(nombre))  # <class 'str'>

activo = True
print(type(activo))  # <class 'bool'>

nada = None
print(type(nada))  # <class 'NoneType'>
```

---

## 2.3. Números Enteros (int)

Los **enteros** son números sin decimales, positivos o negativos.

```python
# Enteros positivos
edad = 25
poblacion = 47000000
año = 2026

# Enteros negativos
temperatura = -5
deuda = -1000

# Cero
contador = 0

# Números grandes (puedes usar _ para legibilidad)
millones = 1_000_000
print(millones)  # 1000000
```

### Operaciones con Enteros

```python
a = 10
b = 3

# Operaciones básicas
print(a + b)   # Suma: 13
print(a - b)   # Resta: 7
print(a * b)   # Multiplicación: 30
print(a / b)   # División: 3.3333... (devuelve float)
print(a // b)  # División entera: 3
print(a % b)   # Módulo (resto): 1
print(a ** b)  # Potencia: 1000
```

### Ejemplos Prácticos

```python
# Ejemplo 1: Cálculo de edad
año_actual = 2026
año_nacimiento = 1990
edad = año_actual - año_nacimiento
print(f"Tienes {edad} años")  # Tienes 36 años

# Ejemplo 2: Conversión de minutos a horas
minutos_totales = 150
horas = minutos_totales // 60
minutos_restantes = minutos_totales % 60
print(f"{minutos_totales} minutos = {horas} horas y {minutos_restantes} minutos")
# 150 minutos = 2 horas y 30 minutos

# Ejemplo 3: Cálculo de precio total
precio_unitario = 15
cantidad = 7
total = precio_unitario * cantidad
print(f"Total: {total}€")  # Total: 105€

# Ejemplo 4: Potencias
base = 2
exponente = 10
resultado = base ** exponente
print(f"{base}^{exponente} = {resultado}")  # 2^10 = 1024
```

---

## 2.4. Números Decimales (float)

Los **float** son números con parte decimal.

```python
# Declaración de floats
precio = 19.99
pi = 3.14159
temperatura = -2.5
porcentaje = 0.75

# También puedes escribirlos así
x = 1.0      # Es float aunque sea un número "entero"
y = .5       # Equivale a 0.5
z = 2.       # Equivale a 2.0
```

### Notación Científica

```python
# Para números muy grandes o muy pequeños
distancia_sol = 1.496e11  # 1.496 × 10¹¹ metros
masa_electron = 9.109e-31  # 9.109 × 10⁻³¹ kg

print(distancia_sol)   # 149600000000.0
print(masa_electron)   # 9.109e-31
```

### Operaciones con Decimales

```python
a = 10.5
b = 3.2

print(a + b)   # 13.7
print(a - b)   # 7.3
print(a * b)   # 33.6
print(a / b)   # 3.28125
print(a // b)  # 3.0 (división entera, pero sigue siendo float)
print(a % b)   # 0.8999999... (resto)
print(a ** b)  # 1613.16...
```

### Precisión de los Float

**¡Cuidado!** Los float tienen limitaciones de precisión:

```python
# Esto puede dar resultados inesperados
print(0.1 + 0.2)  # 0.30000000000000004 (no exactamente 0.3)

# Para comparaciones, usa redondeo o tolerancia
resultado = 0.1 + 0.2
print(round(resultado, 1) == 0.3)  # True
```

### Ejemplos Prácticos

```python
# Ejemplo 1: Cálculo de IVA
precio_base = 100.00
iva = 0.21
precio_con_iva = precio_base * (1 + iva)
print(f"Precio con IVA: {precio_con_iva}€")  # 121.0€

# Ejemplo 2: Conversión de temperatura
celsius = 25.0
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")  # 25.0°C = 77.0°F

# Ejemplo 3: Calcular promedio
nota1 = 7.5
nota2 = 8.0
nota3 = 6.5
promedio = (nota1 + nota2 + nota3) / 3
print(f"Promedio: {promedio:.2f}")  # Promedio: 7.33

# Ejemplo 4: Área de un círculo
radio = 5.0
pi = 3.14159
area = pi * (radio ** 2)
print(f"Área del círculo: {area:.2f} unidades²")  # 78.54 unidades²
```

---

## 2.5. Cadenas de Texto (str)

Las **strings** o cadenas son secuencias de caracteres (texto).

### Creación de Strings

```python
# Con comillas dobles
saludo = "Hola, mundo"

# Con comillas simples
nombre = 'Python'

# Ambas son equivalentes, elige una y sé consistente
mensaje1 = "Esto es un string"
mensaje2 = 'Esto también es un string'

# Strings vacíos
vacio = ""
tambien_vacio = ''
```

### Strings Multilínea

```python
# Con triple comilla
poema = """Caminante, no hay camino,
se hace camino al andar.
- Antonio Machado"""

print(poema)

# También con comillas simples
texto = '''Primera línea
Segunda línea
Tercera línea'''
```

### Comillas Dentro de Strings

```python
# Usar comillas diferentes
frase1 = "Él dijo: 'Hola'"
frase2 = 'Ella respondió: "Buenos días"'

# O usar escape con \
frase3 = "Él dijo: \"Hola\""
frase4 = 'No puedo ir a María\'s house'

print(frase1)  # Él dijo: 'Hola'
print(frase3)  # Él dijo: "Hola"
```

### Caracteres Especiales (Escape)

| Secuencia | Significado |
| :--- | :--- |
| `\n` | Nueva línea |
| `\t` | Tabulación |
| `\\` | Barra invertida |
| `\'` | Comilla simple |
| `\"` | Comilla doble |

```python
# Ejemplos de escape
print("Primera línea\nSegunda línea")
# Primera línea
# Segunda línea

print("Columna1\tColumna2\tColumna3")
# Columna1    Columna2    Columna3

print("Ruta: C:\\Users\\Documentos")
# Ruta: C:\Users\Documentos
```

### Raw Strings

Para evitar que Python interprete los escapes:

```python
# String normal
ruta1 = "C:\nuevo\texto"  # \n y \t se interpretan como escape

# Raw string (r al inicio)
ruta2 = r"C:\nuevo\texto"  # Se mantiene literal
print(ruta2)  # C:\nuevo\texto
```

---

## 2.6. Operaciones con Strings

### Concatenación (Unir Strings)

```python
nombre = "Juan"
apellido = "García"

# Con el operador +
nombre_completo = nombre + " " + apellido
print(nombre_completo)  # Juan García

# Con coma en print (añade espacio automático)
print(nombre, apellido)  # Juan García
```

### Repetición

```python
linea = "-" * 30
print(linea)  # ------------------------------

eco = "Hola " * 3
print(eco)  # Hola Hola Hola
```

### Longitud de un String

```python
mensaje = "Python es genial"
print(len(mensaje))  # 16 (incluye espacios)

password = "secreto123"
print(f"Tu contraseña tiene {len(password)} caracteres")
```

### Acceso a Caracteres (Indexación)

Los strings son secuencias, cada carácter tiene una posición (índice):

```
 P  y  t  h  o  n
 0  1  2  3  4  5   (índices positivos)
-6 -5 -4 -3 -2 -1   (índices negativos)
```

```python
texto = "Python"

# Índices positivos (desde el inicio)
print(texto[0])   # P (primer carácter)
print(texto[1])   # y
print(texto[5])   # n (último carácter)

# Índices negativos (desde el final)
print(texto[-1])  # n (último)
print(texto[-2])  # o (penúltimo)
print(texto[-6])  # P (primero)
```

### Slicing (Cortar Strings)

Sintaxis: `string[inicio:fin:paso]`

```python
texto = "Python es genial"

# Desde índice 0 hasta 6 (sin incluir el 6)
print(texto[0:6])    # Python

# Desde índice 7 hasta el final
print(texto[7:])     # es genial

# Desde el inicio hasta índice 6
print(texto[:6])     # Python

# Desde índice -6 hasta el final
print(texto[-6:])    # genial

# Con paso
print(texto[::2])    # Pto sgna (cada 2 caracteres)

# Invertir string
print(texto[::-1])   # laineg se nohtyP
```

### Ejemplos de Slicing

```python
fecha = "25-01-2026"

dia = fecha[0:2]      # "25"
mes = fecha[3:5]      # "01"
año = fecha[6:]       # "2026"

print(f"Día: {dia}, Mes: {mes}, Año: {año}")

# Obtener extensión de archivo
archivo = "documento.pdf"
extension = archivo[-3:]  # "pdf"
nombre = archivo[:-4]     # "documento"
print(f"Nombre: {nombre}, Extensión: {extension}")
```

---

## 2.7. Métodos de Strings

Los strings tienen muchos métodos (funciones) incorporados:

### Cambio de Mayúsculas/Minúsculas

```python
texto = "Hola Mundo"

print(texto.upper())       # HOLA MUNDO
print(texto.lower())       # hola mundo
print(texto.capitalize())  # Hola mundo (solo primera letra)
print(texto.title())       # Hola Mundo (cada palabra)
print(texto.swapcase())    # hOLA mUNDO (invierte)
```

### Búsqueda

```python
frase = "Python es un lenguaje de programación"

# Encontrar posición
print(frase.find("es"))       # 7 (índice donde empieza "es")
print(frase.find("Java"))     # -1 (no encontrado)

# Contar ocurrencias
print(frase.count("a"))       # 3

# Verificar inicio/fin
print(frase.startswith("Python"))  # True
print(frase.endswith("ción"))      # True

# Verificar contenido
print("Python" in frase)      # True
print("Java" in frase)        # False
```

### Modificación

```python
# Reemplazar
texto = "Hola Mundo"
print(texto.replace("Mundo", "Python"))  # Hola Python

# Eliminar espacios
mensaje = "   Hola   "
print(mensaje.strip())   # "Hola" (elimina espacios inicio/fin)
print(mensaje.lstrip())  # "Hola   " (solo izquierda)
print(mensaje.rstrip())  # "   Hola" (solo derecha)

# También elimina otros caracteres
url = "...www.ejemplo.com..."
print(url.strip("."))  # www.ejemplo.com
```

### Dividir y Unir

```python
# split() - Dividir string en lista
frase = "Python es genial"
palabras = frase.split()  # Divide por espacios
print(palabras)  # ['Python', 'es', 'genial']

fecha = "25-01-2026"
partes = fecha.split("-")
print(partes)  # ['25', '01', '2026']

# join() - Unir lista en string
palabras = ['Python', 'es', 'genial']
frase = " ".join(palabras)
print(frase)  # Python es genial

numeros = ['1', '2', '3', '4']
print("-".join(numeros))  # 1-2-3-4
```

### Verificación

```python
# Verificar tipo de contenido
print("12345".isdigit())     # True (solo dígitos)
print("Python".isalpha())    # True (solo letras)
print("Python3".isalnum())   # True (letras y números)
print("   ".isspace())       # True (solo espacios)
print("HOLA".isupper())      # True (todo mayúsculas)
print("hola".islower())      # True (todo minúsculas)
```

---

## 2.8. Formateo de Strings

### f-strings (Recomendado - Python 3.6+)

La forma más moderna y legible:

```python
nombre = "Ana"
edad = 25

# Interpolación básica
print(f"Me llamo {nombre} y tengo {edad} años")
# Me llamo Ana y tengo 25 años

# Expresiones dentro de las llaves
print(f"En 5 años tendré {edad + 5} años")
# En 5 años tendré 30 años

# Llamar métodos
print(f"Mi nombre en mayúsculas: {nombre.upper()}")
# Mi nombre en mayúsculas: ANA
```

### Formateo de Números

```python
precio = 1234.5678
porcentaje = 0.756

# Decimales
print(f"Precio: {precio:.2f}€")  # Precio: 1234.57€

# Separador de miles
print(f"Precio: {precio:,.2f}€")  # Precio: 1,234.57€

# Porcentaje
print(f"Porcentaje: {porcentaje:.1%}")  # Porcentaje: 75.6%

# Alineación y relleno
numero = 42
print(f"{numero:05d}")   # 00042 (rellena con ceros)
print(f"{numero:>10}")   # "        42" (alinea derecha)
print(f"{numero:<10}")   # "42        " (alinea izquierda)
print(f"{numero:^10}")   # "    42    " (centrado)
```

### Método format()

```python
# Alternativa a f-strings
nombre = "Carlos"
edad = 30

print("Me llamo {} y tengo {} años".format(nombre, edad))
print("Me llamo {0} y tengo {1} años".format(nombre, edad))
print("Me llamo {n} y tengo {e} años".format(n=nombre, e=edad))
```

### Operador % (Estilo Antiguo)

```python
# Menos recomendado, pero aún se ve en código antiguo
nombre = "María"
edad = 28
print("Me llamo %s y tengo %d años" % (nombre, edad))
```

---

## 2.9. Booleanos (bool)

Los **booleanos** solo tienen dos valores: `True` (verdadero) o `False` (falso).

```python
es_mayor = True
tiene_permiso = False

print(type(es_mayor))  # <class 'bool'>
```

### Valores Booleanos en Contexto

En Python, muchos valores se pueden evaluar como booleanos:

```python
# Valores "falsos" (Falsy)
print(bool(0))        # False
print(bool(0.0))      # False
print(bool(""))       # False (string vacío)
print(bool([]))       # False (lista vacía)
print(bool(None))     # False

# Valores "verdaderos" (Truthy)
print(bool(1))        # True
print(bool(-5))       # True (cualquier número no cero)
print(bool("Hola"))   # True (string no vacío)
print(bool([1, 2]))   # True (lista no vacía)
```

### Operadores de Comparación

Devuelven valores booleanos:

```python
x = 10
y = 5

print(x == y)   # False (igual a)
print(x != y)   # True (diferente de)
print(x > y)    # True (mayor que)
print(x < y)    # False (menor que)
print(x >= y)   # True (mayor o igual)
print(x <= y)   # False (menor o igual)
```

### Operadores Lógicos

```python
a = True
b = False

print(a and b)  # False (Y lógico: ambos deben ser True)
print(a or b)   # True (O lógico: al menos uno True)
print(not a)    # False (negación)
```

---

## 2.10. None (Valor Nulo)

`None` representa la ausencia de valor:

```python
resultado = None

print(resultado)        # None
print(type(resultado))  # <class 'NoneType'>
print(resultado is None)  # True

# Uso común: valor inicial antes de asignar
usuario = None

# Después de algún proceso
usuario = "admin"

# Verificar si tiene valor
if usuario is not None:
    print(f"Usuario: {usuario}")
```

---

## 2.11. Conversión de Tipos

A veces necesitas convertir entre tipos de datos.

### int() - Convertir a Entero

```python
# Desde string
edad_texto = "25"
edad_numero = int(edad_texto)
print(edad_numero + 5)  # 30

# Desde float (trunca decimales)
precio = 19.99
precio_entero = int(precio)
print(precio_entero)  # 19

# Errores comunes
# int("Hola")  # Error: no se puede convertir texto
# int("19.99")  # Error: tiene decimal
```

### float() - Convertir a Decimal

```python
# Desde string
temperatura = "36.5"
temp_numero = float(temperatura)
print(temp_numero + 0.5)  # 37.0

# Desde entero
numero = 42
decimal = float(numero)
print(decimal)  # 42.0
```

### str() - Convertir a Texto

```python
edad = 25
edad_texto = str(edad)
print("Tengo " + edad_texto + " años")

precio = 19.99
print("Precio: " + str(precio) + "€")

# Útil para concatenar con +
numero = 100
# print("Valor: " + numero)  # Error!
print("Valor: " + str(numero))  # Correcto
```

### bool() - Convertir a Booleano

```python
print(bool(1))      # True
print(bool(0))      # False
print(bool("Sí"))   # True
print(bool(""))     # False
```

### Ejemplo Práctico: Calculadora

```python
print("=== CALCULADORA ===")
print()

# input() siempre devuelve string
num1_texto = input("Primer número: ")
num2_texto = input("Segundo número: ")

# Convertir a números
num1 = float(num1_texto)
num2 = float(num2_texto)

# Ahora podemos hacer operaciones
suma = num1 + num2
resta = num1 - num2
multiplicacion = num1 * num2
division = num1 / num2

print()
print(f"{num1} + {num2} = {suma}")
print(f"{num1} - {num2} = {resta}")
print(f"{num1} × {num2} = {multiplicacion}")
print(f"{num1} ÷ {num2} = {division:.2f}")
```

---

## 2.12. Ejercicios Prácticos

### Ejercicio 1: Intercambio de Variables

```python
# Intercambia los valores de a y b
a = 10
b = 20

print(f"Antes: a={a}, b={b}")

# Tu código aquí
a, b = b, a

print(f"Después: a={a}, b={b}")
# Debe mostrar: Después: a=20, b=10
```

### Ejercicio 2: Calculadora de Propina

```python
print("=== Calculadora de Propina ===")

cuenta = float(input("Importe de la cuenta: "))
porcentaje_propina = float(input("Porcentaje de propina: "))

propina = cuenta * (porcentaje_propina / 100)
total = cuenta + propina

print()
print(f"Cuenta: {cuenta:.2f}€")
print(f"Propina ({porcentaje_propina}%): {propina:.2f}€")
print(f"Total a pagar: {total:.2f}€")
```

### Ejercicio 3: Extractor de Información

```python
email = "usuario@dominio.com"

# Extraer usuario y dominio
arroba_pos = email.find("@")
usuario = email[:arroba_pos]
dominio = email[arroba_pos + 1:]

print(f"Email: {email}")
print(f"Usuario: {usuario}")
print(f"Dominio: {dominio}")
```

### Ejercicio 4: Formateo de Datos

```python
nombre = "ana garcía"
edad = 28
salario = 32500.50

# Formatear correctamente
nombre_formateado = nombre.title()
print(f"Nombre: {nombre_formateado}")
print(f"Edad: {edad} años")
print(f"Salario: {salario:,.2f}€")
```

### Ejercicio 5: Validador de Contraseña

```python
password = input("Introduce una contraseña: ")

longitud = len(password)
tiene_numeros = any(c.isdigit() for c in password)
tiene_letras = any(c.isalpha() for c in password)

print()
print(f"Longitud: {longitud} caracteres")
print(f"Contiene números: {tiene_numeros}")
print(f"Contiene letras: {tiene_letras}")

if longitud >= 8 and tiene_numeros and tiene_letras:
    print("✓ Contraseña válida")
else:
    print("✗ Contraseña no válida")
```

---

## 2.13. Resumen

| Tipo | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `int` | Números enteros | `42`, `-7` |
| `float` | Números decimales | `3.14`, `-0.5` |
| `str` | Texto | `"Hola"`, `'Python'` |
| `bool` | Verdadero/Falso | `True`, `False` |
| `None` | Valor nulo | `None` |

| Conversión | Función |
| :--- | :--- |
| A entero | `int()` |
| A decimal | `float()` |
| A texto | `str()` |
| A booleano | `bool()` |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
