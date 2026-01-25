# ➕ Unidad 3. Operadores

Los **operadores** son símbolos que realizan operaciones sobre valores y variables. Python tiene varios tipos de operadores que veremos en detalle.

---

## 3.1. Operadores Aritméticos

Realizan operaciones matemáticas básicas:

| Operador | Nombre | Ejemplo | Resultado |
| :--- | :--- | :--- | :--- |
| `+` | Suma | `5 + 3` | `8` |
| `-` | Resta | `5 - 3` | `2` |
| `*` | Multiplicación | `5 * 3` | `15` |
| `/` | División | `5 / 3` | `1.666...` |
| `//` | División entera | `5 // 3` | `1` |
| `%` | Módulo (resto) | `5 % 3` | `2` |
| `**` | Potencia | `5 ** 3` | `125` |

### Ejemplos Detallados

```python
# Suma
print(10 + 5)      # 15
print(3.5 + 2.5)   # 6.0
print(-5 + 10)     # 5

# Resta
print(10 - 5)      # 5
print(3 - 8)       # -5

# Multiplicación
print(4 * 7)       # 28
print(2.5 * 4)     # 10.0
print(-3 * 5)      # -15

# División (siempre devuelve float)
print(10 / 2)      # 5.0
print(10 / 3)      # 3.3333...
print(7 / 2)       # 3.5

# División entera (descarta decimales)
print(10 // 3)     # 3
print(7 // 2)      # 3
print(-7 // 2)     # -4 (redondea hacia abajo)

# Módulo (resto de la división)
print(10 % 3)      # 1 (10 = 3*3 + 1)
print(15 % 5)      # 0 (división exacta)
print(7 % 2)       # 1 (impar)
print(8 % 2)       # 0 (par)

# Potencia
print(2 ** 3)      # 8 (2³)
print(5 ** 2)      # 25 (5²)
print(4 ** 0.5)    # 2.0 (raíz cuadrada)
print(27 ** (1/3)) # 3.0 (raíz cúbica)
```

### Usos Prácticos del Módulo

```python
# Verificar si un número es par o impar
numero = 17
if numero % 2 == 0:
    print(f"{numero} es par")
else:
    print(f"{numero} es impar")
# 17 es impar

# Obtener el último dígito de un número
numero = 12345
ultimo_digito = numero % 10
print(f"Último dígito de {numero}: {ultimo_digito}")  # 5

# Ciclos (volver al inicio después de N)
# Útil para índices circulares
posicion = 0
total_elementos = 5

for i in range(10):
    indice = i % total_elementos
    print(f"Paso {i}: índice {indice}")
# 0,1,2,3,4,0,1,2,3,4

# Convertir segundos a horas:minutos:segundos
segundos_totales = 3725

horas = segundos_totales // 3600
minutos = (segundos_totales % 3600) // 60
segundos = segundos_totales % 60

print(f"{segundos_totales} segundos = {horas}h {minutos}m {segundos}s")
# 3725 segundos = 1h 2m 5s
```

### Precedencia de Operadores

Los operadores siguen un orden de evaluación (de mayor a menor precedencia):

1.  `**` (potencia)
2.  `*`, `/`, `//`, `%`
3.  `+`, `-`

```python
# Ejemplos de precedencia
print(2 + 3 * 4)       # 14 (no 20) - multiplicación primero
print((2 + 3) * 4)     # 20 - paréntesis primero
print(2 ** 3 ** 2)     # 512 (2^9) - potencia es de derecha a izquierda
print(10 - 3 - 2)      # 5 - de izquierda a derecha

# Usar paréntesis para claridad
resultado = ((5 + 3) * 2) / 4
print(resultado)  # 4.0
```

---

## 3.2. Operadores de Comparación

Comparan dos valores y devuelven `True` o `False`:

| Operador | Significado | Ejemplo | Resultado |
| :--- | :--- | :--- | :--- |
| `==` | Igual a | `5 == 5` | `True` |
| `!=` | Diferente de | `5 != 3` | `True` |
| `>` | Mayor que | `5 > 3` | `True` |
| `<` | Menor que | `5 < 3` | `False` |
| `>=` | Mayor o igual | `5 >= 5` | `True` |
| `<=` | Menor o igual | `5 <= 3` | `False` |

### Ejemplos con Números

```python
a = 10
b = 5

print(a == b)    # False (¿son iguales?)
print(a != b)    # True (¿son diferentes?)
print(a > b)     # True (¿a es mayor que b?)
print(a < b)     # False (¿a es menor que b?)
print(a >= b)    # True (¿a es mayor o igual que b?)
print(a <= b)    # False (¿a es menor o igual que b?)

# Comparaciones encadenadas
x = 15
print(10 < x < 20)     # True (x está entre 10 y 20)
print(0 <= x <= 100)   # True (x está entre 0 y 100)
```

### Comparación de Strings

```python
# Comparación alfabética (orden lexicográfico)
print("abc" == "abc")    # True
print("abc" == "ABC")    # False (mayúsculas importan)
print("abc" < "abd")     # True (c < d)
print("apple" < "banana")  # True (a < b)

# Comparar ignorando mayúsculas
nombre1 = "Ana"
nombre2 = "ana"
print(nombre1.lower() == nombre2.lower())  # True
```

### Comparación de Otros Tipos

```python
# Listas
print([1, 2] == [1, 2])    # True
print([1, 2] == [2, 1])    # False (orden importa)

# None
x = None
print(x == None)    # True (funciona, pero...)
print(x is None)    # True (recomendado)
```

### Errores Comunes

```python
# MAL: usar = en lugar de ==
# if x = 5:  # Error de sintaxis

# BIEN: usar ==
x = 5
if x == 5:
    print("x es 5")

# Cuidado con los floats
print(0.1 + 0.2 == 0.3)  # False (problema de precisión)
print(abs((0.1 + 0.2) - 0.3) < 0.0001)  # True (comparar con tolerancia)
```

---

## 3.3. Operadores Lógicos

Combinan expresiones booleanas:

| Operador | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `and` | Verdadero si ambos son verdaderos | `True and True` → `True` |
| `or` | Verdadero si al menos uno es verdadero | `True or False` → `True` |
| `not` | Invierte el valor | `not True` → `False` |

### Tablas de Verdad

**AND (Y lógico):**

| A | B | A and B |
| :--- | :--- | :--- |
| True | True | True |
| True | False | False |
| False | True | False |
| False | False | False |

**OR (O lógico):**

| A | B | A or B |
| :--- | :--- | :--- |
| True | True | True |
| True | False | True |
| False | True | True |
| False | False | False |

**NOT (Negación):**

| A | not A |
| :--- | :--- |
| True | False |
| False | True |

### Ejemplos Prácticos

```python
edad = 25
tiene_dni = True
tiene_dinero = True

# AND: todas las condiciones deben cumplirse
puede_comprar_alcohol = edad >= 18 and tiene_dni
print(puede_comprar_alcohol)  # True

# OR: al menos una condición debe cumplirse
es_fin_de_semana = False
es_festivo = True
dia_libre = es_fin_de_semana or es_festivo
print(dia_libre)  # True

# NOT: invierte la condición
esta_cerrado = False
esta_abierto = not esta_cerrado
print(esta_abierto)  # True

# Combinaciones
puede_entrar = edad >= 18 and tiene_dni and not esta_cerrado
print(puede_entrar)  # True
```

### Validación de Rangos

```python
# Verificar si un número está en un rango
nota = 7.5

es_aprobado = nota >= 5
es_notable = nota >= 7 and nota < 9
es_sobresaliente = nota >= 9

print(f"Nota: {nota}")
print(f"Aprobado: {es_aprobado}")        # True
print(f"Notable: {es_notable}")          # True
print(f"Sobresaliente: {es_sobresaliente}")  # False

# Forma más elegante con comparaciones encadenadas
es_notable_v2 = 7 <= nota < 9
print(f"Notable (v2): {es_notable_v2}")  # True
```

### Validación de Formularios

```python
nombre = "Juan"
email = "juan@email.com"
edad = 25
acepta_terminos = True

# Verificar que todos los campos están completos
nombre_valido = len(nombre) > 0
email_valido = "@" in email and "." in email
edad_valida = edad >= 18

# Formulario completo
formulario_valido = nombre_valido and email_valido and edad_valida and acepta_terminos

print(f"Nombre válido: {nombre_valido}")        # True
print(f"Email válido: {email_valido}")          # True
print(f"Edad válida: {edad_valida}")            # True
print(f"Acepta términos: {acepta_terminos}")    # True
print(f"Formulario válido: {formulario_valido}")  # True
```

### Evaluación de Cortocircuito

Python evalúa expresiones de forma "perezosa":

```python
# Con AND: si el primero es False, no evalúa el segundo
# Con OR: si el primero es True, no evalúa el segundo

def funcion_costosa():
    print("Ejecutando función costosa...")
    return True

# No ejecuta funcion_costosa() porque False and X siempre es False
resultado = False and funcion_costosa()
print(resultado)  # False (no imprime el mensaje)

# Sí ejecuta porque necesita evaluar el segundo
resultado = True and funcion_costosa()
print(resultado)  # Imprime mensaje, luego True
```

### Valores Truthy y Falsy

```python
# Valores que Python considera False (falsy)
print(bool(0))        # False
print(bool(0.0))      # False
print(bool(""))       # False
print(bool([]))       # False
print(bool(None))     # False

# Uso práctico
nombre = ""
if nombre:
    print(f"Hola, {nombre}")
else:
    print("No has introducido nombre")
# No has introducido nombre

# Valor por defecto con or
nombre = "" or "Anónimo"
print(nombre)  # Anónimo

usuario = None
nombre_mostrar = usuario or "Invitado"
print(nombre_mostrar)  # Invitado
```

---

## 3.4. Operadores de Asignación

Asignan valores a variables, con posibles operaciones:

| Operador | Equivalente a | Ejemplo |
| :--- | :--- | :--- |
| `=` | Asignación simple | `x = 5` |
| `+=` | `x = x + valor` | `x += 3` |
| `-=` | `x = x - valor` | `x -= 3` |
| `*=` | `x = x * valor` | `x *= 3` |
| `/=` | `x = x / valor` | `x /= 3` |
| `//=` | `x = x // valor` | `x //= 3` |
| `%=` | `x = x % valor` | `x %= 3` |
| `**=` | `x = x ** valor` | `x **= 3` |

### Ejemplos

```python
# Asignación simple
x = 10
print(f"x = {x}")  # x = 10

# Suma y asignación
x += 5  # equivale a x = x + 5
print(f"x += 5 → {x}")  # x = 15

# Resta y asignación
x -= 3  # equivale a x = x - 3
print(f"x -= 3 → {x}")  # x = 12

# Multiplicación y asignación
x *= 2  # equivale a x = x * 2
print(f"x *= 2 → {x}")  # x = 24

# División y asignación
x /= 4  # equivale a x = x / 4
print(f"x /= 4 → {x}")  # x = 6.0

# División entera y asignación
x = 17
x //= 3  # equivale a x = x // 3
print(f"x //= 3 → {x}")  # x = 5

# Módulo y asignación
x = 17
x %= 5  # equivale a x = x % 5
print(f"x %= 5 → {x}")  # x = 2

# Potencia y asignación
x = 2
x **= 4  # equivale a x = x ** 4
print(f"x **= 4 → {x}")  # x = 16
```

### Ejemplo: Contador

```python
contador = 0

contador += 1  # incrementar
print(contador)  # 1

contador += 1
print(contador)  # 2

contador += 1
print(contador)  # 3
```

### Ejemplo: Acumulador

```python
# Calcular suma de números
suma = 0

suma += 10
suma += 20
suma += 30
suma += 40

print(f"Suma total: {suma}")  # 100
```

### Con Strings

```python
mensaje = "Hola"
mensaje += " "
mensaje += "Mundo"
print(mensaje)  # Hola Mundo

# Construir una frase
frase = ""
frase += "Python "
frase += "es "
frase += "genial"
print(frase)  # Python es genial
```

---

## 3.5. Operadores de Identidad

Verifican si dos variables apuntan al **mismo objeto** en memoria:

| Operador | Descripción |
| :--- | :--- |
| `is` | True si son el mismo objeto |
| `is not` | True si son objetos diferentes |

```python
# Comparar con None (uso más común)
x = None
print(x is None)      # True
print(x is not None)  # False

# Diferencia entre == y is
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)    # True (mismo contenido)
print(a is b)    # False (objetos diferentes)
print(a is c)    # True (mismo objeto)

# Verificar el id (dirección en memoria)
print(id(a))  # Por ejemplo: 140234567890
print(id(b))  # Diferente: 140234567891
print(id(c))  # Igual a id(a): 140234567890
```

### Cuándo Usar is vs ==

```python
# Usar 'is' para comparar con None, True, False
valor = None
if valor is None:
    print("No tiene valor")

# Usar '==' para comparar contenido
lista1 = [1, 2, 3]
lista2 = [1, 2, 3]
if lista1 == lista2:
    print("Las listas tienen el mismo contenido")
```

### Nota sobre Enteros Pequeños

Python optimiza los enteros pequeños (-5 a 256):

```python
a = 100
b = 100
print(a is b)  # True (Python reutiliza el objeto)

c = 1000
d = 1000
print(c is d)  # Puede ser False (objetos diferentes)

# SIEMPRE usa == para comparar valores
print(c == d)  # True
```

---

## 3.6. Operadores de Pertenencia

Verifican si un elemento está **dentro** de una secuencia:

| Operador | Descripción |
| :--- | :--- |
| `in` | True si el elemento está en la secuencia |
| `not in` | True si el elemento NO está en la secuencia |

### En Strings

```python
texto = "Hola Mundo"

print("Hola" in texto)      # True
print("hola" in texto)      # False (mayúsculas importan)
print("Python" in texto)    # False
print("X" not in texto)     # True

# Verificar vocales
letra = "e"
vocales = "aeiouAEIOU"
es_vocal = letra in vocales
print(f"'{letra}' es vocal: {es_vocal}")  # True
```

### En Listas

```python
frutas = ["manzana", "naranja", "plátano"]

print("manzana" in frutas)     # True
print("uva" in frutas)         # False
print("pera" not in frutas)    # True

# Verificar si el usuario eligió una opción válida
opciones_validas = [1, 2, 3, 4]
opcion_usuario = 2
if opcion_usuario in opciones_validas:
    print("Opción válida")
else:
    print("Opción no válida")
```

### En Diccionarios

```python
# 'in' verifica las CLAVES, no los valores
persona = {"nombre": "Ana", "edad": 30, "ciudad": "Madrid"}

print("nombre" in persona)    # True (es una clave)
print("Ana" in persona)       # False (es un valor, no una clave)
print("apellido" in persona)  # False

# Para verificar valores
print("Ana" in persona.values())  # True
```

### Ejemplo: Validación de Input

```python
respuestas_validas = ["s", "si", "sí", "n", "no"]

respuesta = input("¿Desea continuar? (s/n): ").lower()

if respuesta in respuestas_validas:
    print("Respuesta válida")
else:
    print("Por favor, responda 's' o 'n'")
```

---

## 3.7. Operadores a Nivel de Bits (Bitwise)

Operan directamente sobre los bits de los números. Útiles en programación de bajo nivel:

| Operador | Nombre | Descripción |
| :--- | :--- | :--- |
| `&` | AND | 1 si ambos bits son 1 |
| `\|` | OR | 1 si al menos un bit es 1 |
| `^` | XOR | 1 si los bits son diferentes |
| `~` | NOT | Invierte todos los bits |
| `<<` | Shift izquierda | Desplaza bits a la izquierda |
| `>>` | Shift derecha | Desplaza bits a la derecha |

```python
a = 5   # En binario: 0101
b = 3   # En binario: 0011

print(f"a = {a} ({bin(a)})")  # a = 5 (0b101)
print(f"b = {b} ({bin(b)})")  # b = 3 (0b11)

# AND: 0101 & 0011 = 0001
print(f"a & b = {a & b} ({bin(a & b)})")  # 1 (0b1)

# OR: 0101 | 0011 = 0111
print(f"a | b = {a | b} ({bin(a | b)})")  # 7 (0b111)

# XOR: 0101 ^ 0011 = 0110
print(f"a ^ b = {a ^ b} ({bin(a ^ b)})")  # 6 (0b110)

# Shift izquierda (multiplica por 2^n)
print(f"a << 1 = {a << 1}")  # 10 (5 * 2)
print(f"a << 2 = {a << 2}")  # 20 (5 * 4)

# Shift derecha (divide por 2^n)
print(f"a >> 1 = {a >> 1}")  # 2 (5 // 2)
```

**Nota:** Los operadores bitwise no son comunes en IA/ML, pero pueden aparecer en optimizaciones de código.

---

## 3.8. Resumen de Precedencia de Operadores

De mayor a menor precedencia:

| Precedencia | Operadores |
| :--- | :--- |
| 1 (más alta) | `()` Paréntesis |
| 2 | `**` Potencia |
| 3 | `+x`, `-x`, `~x` Unarios |
| 4 | `*`, `/`, `//`, `%` |
| 5 | `+`, `-` |
| 6 | `<<`, `>>` |
| 7 | `&` |
| 8 | `^` |
| 9 | `\|` |
| 10 | `==`, `!=`, `<`, `<=`, `>`, `>=`, `is`, `in` |
| 11 | `not` |
| 12 | `and` |
| 13 (más baja) | `or` |

**Consejo:** Usa paréntesis para hacer tu código más legible:

```python
# Difícil de leer
resultado = 2 + 3 * 4 ** 2 / 8 - 1

# Más claro
resultado = 2 + ((3 * (4 ** 2)) / 8) - 1
```

---

## 3.9. Ejercicios Prácticos

### Ejercicio 1: Calculadora Completa

```python
print("=== CALCULADORA ===")
print()

num1 = float(input("Primer número: "))
num2 = float(input("Segundo número: "))

print()
print(f"{num1} + {num2} = {num1 + num2}")
print(f"{num1} - {num2} = {num1 - num2}")
print(f"{num1} × {num2} = {num1 * num2}")
print(f"{num1} ÷ {num2} = {num1 / num2:.4f}")
print(f"{num1} // {num2} = {num1 // num2}")
print(f"{num1} % {num2} = {num1 % num2}")
print(f"{num1} ^ {num2} = {num1 ** num2}")
```

### Ejercicio 2: Verificador de Año Bisiesto

```python
año = int(input("Introduce un año: "))

# Un año es bisiesto si:
# - Es divisible por 4 Y no es divisible por 100
# - O es divisible por 400
es_bisiesto = (año % 4 == 0 and año % 100 != 0) or (año % 400 == 0)

if es_bisiesto:
    print(f"{año} es bisiesto")
else:
    print(f"{año} no es bisiesto")
```

### Ejercicio 3: Clasificador de Edad

```python
edad = int(input("Introduce tu edad: "))

es_bebe = edad < 2
es_niño = 2 <= edad < 12
es_adolescente = 12 <= edad < 18
es_adulto = 18 <= edad < 65
es_senior = edad >= 65

print()
if es_bebe:
    print("Eres un bebé")
elif es_niño:
    print("Eres un niño/a")
elif es_adolescente:
    print("Eres adolescente")
elif es_adulto:
    print("Eres adulto/a")
else:
    print("Eres senior")
```

### Ejercicio 4: Verificador de Triángulo

```python
print("=== Verificador de Triángulo ===")
print()

a = float(input("Lado a: "))
b = float(input("Lado b: "))
c = float(input("Lado c: "))

# Un triángulo es válido si la suma de dos lados
# es mayor que el tercero (para todos los pares)
es_valido = (a + b > c) and (a + c > b) and (b + c > a)

print()
if es_valido:
    print("Los lados forman un triángulo válido")
    
    # Clasificar el triángulo
    if a == b == c:
        print("Es un triángulo equilátero")
    elif a == b or b == c or a == c:
        print("Es un triángulo isósceles")
    else:
        print("Es un triángulo escaleno")
else:
    print("Los lados NO forman un triángulo válido")
```

### Ejercicio 5: Calculadora de Descuentos

```python
print("=== Calculadora de Descuentos ===")
print()

precio = float(input("Precio original: "))
es_cliente_vip = input("¿Es cliente VIP? (s/n): ").lower() == "s"
cantidad = int(input("Cantidad de productos: "))

# Calcular descuentos
descuento_vip = 0.10 if es_cliente_vip else 0
descuento_cantidad = 0.05 if cantidad >= 3 else 0
descuento_total = descuento_vip + descuento_cantidad

subtotal = precio * cantidad
ahorro = subtotal * descuento_total
total = subtotal - ahorro

print()
print(f"Subtotal: {subtotal:.2f}€")
print(f"Descuento VIP: {descuento_vip*100:.0f}%")
print(f"Descuento cantidad: {descuento_cantidad*100:.0f}%")
print(f"Ahorro: {ahorro:.2f}€")
print(f"Total: {total:.2f}€")
```

---

## 3.10. Resumen

| Tipo | Operadores |
| :--- | :--- |
| Aritméticos | `+`, `-`, `*`, `/`, `//`, `%`, `**` |
| Comparación | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| Lógicos | `and`, `or`, `not` |
| Asignación | `=`, `+=`, `-=`, `*=`, `/=`, etc. |
| Identidad | `is`, `is not` |
| Pertenencia | `in`, `not in` |
| Bitwise | `&`, `\|`, `^`, `~`, `<<`, `>>` |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
