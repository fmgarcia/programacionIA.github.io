# 🔀 Unidad 4. Estructuras de Control

Las **estructuras de control** permiten modificar el flujo de ejecución de un programa. Incluyen condicionales (decisiones) y bucles (repeticiones).

---

## 4.1. Condicionales: if, elif, else

Los condicionales permiten ejecutar código solo si se cumple una condición.

### Estructura if Simple

```python
edad = 18

if edad >= 18:
    print("Eres mayor de edad")
    print("Puedes votar")
```

**Importante:** El código dentro del `if` debe estar **indentado** (con 4 espacios o 1 tabulación).

### Estructura if-else

```python
edad = 15

if edad >= 18:
    print("Eres mayor de edad")
else:
    print("Eres menor de edad")
```

### Estructura if-elif-else

Para múltiples condiciones:

```python
nota = 7.5

if nota >= 9:
    print("Sobresaliente")
elif nota >= 7:
    print("Notable")
elif nota >= 5:
    print("Aprobado")
else:
    print("Suspenso")
```

### Ejemplos Prácticos

```python
# Ejemplo 1: Verificar número positivo, negativo o cero
numero = float(input("Introduce un número: "))

if numero > 0:
    print("El número es positivo")
elif numero < 0:
    print("El número es negativo")
else:
    print("El número es cero")
```

```python
# Ejemplo 2: Calculadora de descuentos
precio = float(input("Precio del producto: "))
cantidad = int(input("Cantidad: "))

total = precio * cantidad

# Aplicar descuento según el total
if total >= 100:
    descuento = 0.15  # 15%
    print("¡Descuento del 15%!")
elif total >= 50:
    descuento = 0.10  # 10%
    print("¡Descuento del 10%!")
elif total >= 25:
    descuento = 0.05  # 5%
    print("¡Descuento del 5%!")
else:
    descuento = 0
    print("Sin descuento")

total_final = total * (1 - descuento)
print(f"Total a pagar: {total_final:.2f}€")
```

```python
# Ejemplo 3: Verificar día de la semana
dia = int(input("Introduce el número del día (1-7): "))

if dia == 1:
    print("Lunes")
elif dia == 2:
    print("Martes")
elif dia == 3:
    print("Miércoles")
elif dia == 4:
    print("Jueves")
elif dia == 5:
    print("Viernes")
elif dia == 6:
    print("Sábado")
elif dia == 7:
    print("Domingo")
else:
    print("Día no válido")
```

### Condicionales Anidados

Puedes poner un `if` dentro de otro:

```python
edad = 25
tiene_carnet = True

if edad >= 18:
    print("Eres mayor de edad")
    if tiene_carnet:
        print("Puedes conducir")
    else:
        print("Necesitas sacarte el carnet")
else:
    print("Eres menor de edad")
    print("No puedes conducir")
```

### Operador Ternario

Una forma compacta de escribir if-else en una línea:

```python
# Sintaxis: valor_si_verdadero if condicion else valor_si_falso

edad = 20
estado = "mayor" if edad >= 18 else "menor"
print(f"Eres {estado} de edad")

# Equivalente a:
if edad >= 18:
    estado = "mayor"
else:
    estado = "menor"

# Más ejemplos
numero = 7
paridad = "par" if numero % 2 == 0 else "impar"
print(f"{numero} es {paridad}")

# Con valores numéricos
x = 10
y = 20
maximo = x if x > y else y
print(f"El mayor es: {maximo}")
```

### Múltiples Condiciones

```python
# Usando and, or, not
edad = 25
tiene_dni = True
esta_sobrio = True

if edad >= 18 and tiene_dni and esta_sobrio:
    print("Puede entrar al club")
else:
    print("No puede entrar")

# Verificar rango
temperatura = 25
if 20 <= temperatura <= 30:
    print("Temperatura agradable")

# Con or
dia = "sábado"
if dia == "sábado" or dia == "domingo":
    print("¡Es fin de semana!")

# Con not
lloviendo = False
if not lloviendo:
    print("Podemos salir a pasear")
```

---

## 4.2. Condicionales: match, case

La sentencia `match` (disponible desde Python 3.10) permite comparar una expresión con varios patrones de forma clara y concisa. Es similar al `switch` de otros lenguajes, pero mucho más poderosa.

### Sintaxis Básica

```python
match variable:
    case valor1:
        # código si coincide con valor1
    case valor2:
        # código si coincide con valor2
    case _:
        # caso por defecto (si ninguno coincide)
```

**Importante:** El caso `_` actúa como el `else`, se ejecuta si ningún patrón anterior coincide.

### Ejemplo: Días de la Semana

```python
dia = input("¿En qué día de la semana estamos? ")

match dia:
    case "lunes":
        print("Hoy es lunes, ¡comienza la semana!")
    case "martes":
        print("Hoy es martes, ¡a seguir adelante!")
    case "miércoles":
        print("Hoy es miércoles, ¡ya estamos a mitad de semana!")
    case "jueves":
        print("Hoy es jueves, ¡casi es fin de semana!")
    case "viernes":
        print("Hoy es viernes, ¡por fin es fin de semana!")
    case "sábado":
        print("Hoy es sábado, ¡disfruta tu día de descanso!")
    case "domingo":
        print("Hoy es domingo, ¡prepárate para la semana que viene!")
    case _:
        print("¡Día no reconocido! Por favor, introduce un día válido.")
```

### Múltiples Valores con `|`

Puedes agrupar varios valores en un mismo caso usando el operador `|`:

```python
dia = input("¿En qué día de la semana estamos? ")

match dia:
    case "lunes" | "martes" | "miércoles" | "jueves" | "viernes":
        print("Hoy es un día laboral, ¡a trabajar!")
    case "sábado" | "domingo":
        print("¡Es fin de semana, disfruta tu día de descanso!")
    case _:
        print("¡Día no reconocido! Por favor, introduce un día válido.")
```

### Patrones con Tuplas y Guardas

`match` también puede comparar estructuras como tuplas. Además, se pueden añadir condiciones extra con `if` (llamadas **guardas**):

```python
coordenadas = (3, 4)

match coordenadas:
    case (0, 0):
        print("El punto está en el origen")
    case (x, 0):
        print(f"El punto está sobre el eje X en x={x}")
    case (0, y):
        print(f"El punto está sobre el eje Y en y={y}")
    case (x, y) if x > 0 and y > 0:
        print(f"Primer cuadrante: ({x}, {y})")
    case (x, y) if x < 0 and y > 0:
        print(f"Segundo cuadrante: ({x}, {y})")
    case (x, y) if x < 0 and y < 0:
        print(f"Tercer cuadrante: ({x}, {y})")
    case (x, y) if x > 0 and y < 0:
        print(f"Cuarto cuadrante: ({x}, {y})")
```

---

## 4.3. Bucle while

El bucle `while` repite un bloque de código **mientras** una condición sea verdadera.

### Sintaxis Básica

```python
contador = 1

while contador <= 5:
    print(f"Contador: {contador}")
    contador += 1

print("Fin del bucle")
```

Salida:

```
Contador: 1
Contador: 2
Contador: 3
Contador: 4
Contador: 5
Fin del bucle
```

### Ejemplos Prácticos

```python
# Ejemplo 1: Cuenta regresiva
print("Cuenta regresiva:")
n = 10
while n > 0:
    print(n)
    n -= 1
print("¡Despegue!")
```

```python
# Ejemplo 2: Suma de números hasta que el usuario introduzca 0
print("Suma de números (introduce 0 para terminar)")
suma = 0

numero = int(input("Introduce un número: "))
while numero != 0:
    suma += numero
    numero = int(input("Introduce un número: "))

print(f"La suma total es: {suma}")
```

```python
# Ejemplo 3: Adivinar un número
import random

numero_secreto = random.randint(1, 100)
intentos = 0
adivinado = False

print("Adivina el número (entre 1 y 100)")

while not adivinado:
    intento = int(input("Tu intento: "))
    intentos += 1
    
    if intento < numero_secreto:
        print("Demasiado bajo")
    elif intento > numero_secreto:
        print("Demasiado alto")
    else:
        adivinado = True
        print(f"¡Correcto! Lo adivinaste en {intentos} intentos")
```

```python
# Ejemplo 4: Validar entrada del usuario
respuesta = ""

while respuesta not in ["s", "n"]:
    respuesta = input("¿Desea continuar? (s/n): ").lower()
    if respuesta not in ["s", "n"]:
        print("Por favor, introduce 's' o 'n'")

if respuesta == "s":
    print("Continuando...")
else:
    print("Saliendo...")
```

```python
# Ejemplo 5: Menú interactivo
opcion = 0

while opcion != 4:
    print("\n=== MENÚ ===")
    print("1. Opción A")
    print("2. Opción B")
    print("3. Opción C")
    print("4. Salir")
    
    opcion = int(input("Elige una opción: "))
    
    if opcion == 1:
        print("Has elegido la opción A")
    elif opcion == 2:
        print("Has elegido la opción B")
    elif opcion == 3:
        print("Has elegido la opción C")
    elif opcion == 4:
        print("¡Hasta luego!")
    else:
        print("Opción no válida")
```

### Bucle Infinito

**¡Cuidado!** Si la condición nunca se hace falsa, el bucle nunca termina:

```python
# ¡BUCLE INFINITO! (No ejecutes esto)
# while True:
#     print("Esto se repite para siempre")

# Para salir de un bucle infinito, usa Ctrl+C
```

---

## 4.4. Bucle for

El bucle `for` itera sobre una secuencia (lista, string, range, etc.).

### Sintaxis Básica

```python
# Iterar sobre una lista
frutas = ["manzana", "naranja", "plátano"]

for fruta in frutas:
    print(fruta)
```

Salida:

```
manzana
naranja
plátano
```

### Iterar sobre Strings

```python
palabra = "Python"

for letra in palabra:
    print(letra)
```

Salida:

```
P
y
t
h
o
n
```

### La Función range()

`range()` genera una secuencia de números:

```python
# range(fin) - desde 0 hasta fin-1
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(inicio, fin) - desde inicio hasta fin-1
for i in range(1, 6):
    print(i)  # 1, 2, 3, 4, 5

# range(inicio, fin, paso)
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8 (de 2 en 2)

# Cuenta regresiva (paso negativo)
for i in range(10, 0, -1):
    print(i)  # 10, 9, 8, ..., 1
```

### Ejemplos Prácticos con for

```python
# Ejemplo 1: Tabla de multiplicar
numero = int(input("¿De qué número quieres la tabla? "))

print(f"\nTabla del {numero}:")
for i in range(1, 11):
    resultado = numero * i
    print(f"{numero} x {i} = {resultado}")
```

```python
# Ejemplo 2: Suma de los primeros N números
n = int(input("¿Hasta qué número sumar? "))

suma = 0
for i in range(1, n + 1):
    suma += i

print(f"La suma de 1 a {n} es: {suma}")

# También se puede calcular con la fórmula: n * (n+1) / 2
```

```python
# Ejemplo 3: Factorial
n = int(input("Calcular factorial de: "))

factorial = 1
for i in range(1, n + 1):
    factorial *= i

print(f"{n}! = {factorial}")
```

```python
# Ejemplo 4: Encontrar números pares
print("Números pares del 1 al 20:")
for num in range(1, 21):
    if num % 2 == 0:
        print(num, end=" ")
print()  # Nueva línea al final
# Salida: 2 4 6 8 10 12 14 16 18 20
```

```python
# Ejemplo 5: Dibujar un patrón
n = 5
for i in range(1, n + 1):
    print("*" * i)

# Salida:
# *
# **
# ***
# ****
# *****
```

```python
# Ejemplo 6: Iterar con índice usando enumerate
frutas = ["manzana", "naranja", "plátano"]

for indice, fruta in enumerate(frutas):
    print(f"{indice + 1}. {fruta}")

# Salida:
# 1. manzana
# 2. naranja
# 3. plátano
```

```python
# Ejemplo 7: Iterar sobre diccionarios
persona = {"nombre": "Ana", "edad": 30, "ciudad": "Madrid"}

# Solo claves
for clave in persona:
    print(clave)

# Claves y valores
for clave, valor in persona.items():
    print(f"{clave}: {valor}")
```

---

## 4.5. Control de Bucles: break, continue, else

### break - Salir del Bucle

```python
# Buscar un número en una lista
numeros = [1, 5, 8, 12, 15, 20]
buscar = 12

for num in numeros:
    print(f"Revisando {num}...")
    if num == buscar:
        print(f"¡Encontrado: {buscar}!")
        break  # Sale del bucle

print("Fin de la búsqueda")
```

```python
# Salir de un bucle while
while True:
    respuesta = input("Escribe 'salir' para terminar: ")
    if respuesta == "salir":
        break
    print(f"Escribiste: {respuesta}")

print("Has salido del bucle")
```

### continue - Saltar a la Siguiente Iteración

```python
# Imprimir solo números impares
for i in range(1, 11):
    if i % 2 == 0:  # Si es par
        continue    # Salta a la siguiente iteración
    print(i)

# Salida: 1, 3, 5, 7, 9
```

```python
# Procesar solo elementos válidos
datos = [10, -5, 20, 0, 15, -3]

suma = 0
for num in datos:
    if num <= 0:
        print(f"Ignorando {num} (no válido)")
        continue
    suma += num
    print(f"Sumando {num}, total: {suma}")

print(f"Suma de positivos: {suma}")
```

### else en Bucles

El `else` en un bucle se ejecuta si el bucle termina **sin** un `break`:

```python
# Buscar un número primo
numero = 17

for i in range(2, numero):
    if numero % i == 0:
        print(f"{numero} no es primo (divisible por {i})")
        break
else:
    # Se ejecuta si no hubo break
    print(f"{numero} es primo")
```

```python
# Verificar si un elemento está en una lista
lista = [1, 2, 3, 4, 5]
buscar = 10

for item in lista:
    if item == buscar:
        print(f"¡Encontrado {buscar}!")
        break
else:
    print(f"{buscar} no está en la lista")
```

---

## 4.6. Bucles Anidados

Un bucle dentro de otro:

```python
# Tabla de multiplicar completa
for i in range(1, 6):
    print(f"\n--- Tabla del {i} ---")
    for j in range(1, 11):
        print(f"{i} x {j} = {i * j}")
```

```python
# Matriz de asteriscos
filas = 4
columnas = 6

for i in range(filas):
    for j in range(columnas):
        print("*", end=" ")
    print()  # Nueva línea al final de cada fila

# Salida:
# * * * * * *
# * * * * * *
# * * * * * *
# * * * * * *
```

```python
# Triángulo rectángulo
n = 5
for i in range(1, n + 1):
    for j in range(i):
        print("*", end="")
    print()

# Salida:
# *
# **
# ***
# ****
# *****
```

```python
# Pirámide centrada
n = 5
for i in range(1, n + 1):
    espacios = " " * (n - i)
    asteriscos = "*" * (2 * i - 1)
    print(espacios + asteriscos)

# Salida:
#     *
#    ***
#   *****
#  *******
# *********
```

```python
# Buscar en una matriz
matriz = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

buscar = 5
encontrado = False

for i in range(len(matriz)):
    for j in range(len(matriz[i])):
        if matriz[i][j] == buscar:
            print(f"Encontrado {buscar} en posición [{i}][{j}]")
            encontrado = True
            break
    if encontrado:
        break

if not encontrado:
    print(f"{buscar} no encontrado")
```

---

## 4.7. Comprensión de Listas (List Comprehension)

Una forma concisa de crear listas con bucles:

```python
# Forma tradicional
cuadrados = []
for x in range(10):
    cuadrados.append(x ** 2)
print(cuadrados)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Con list comprehension
cuadrados = [x ** 2 for x in range(10)]
print(cuadrados)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### Sintaxis

```python
[expresion for elemento in iterable]
[expresion for elemento in iterable if condicion]
```

### Ejemplos

```python
# Números pares
pares = [x for x in range(20) if x % 2 == 0]
print(pares)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Convertir a mayúsculas
palabras = ["hola", "mundo", "python"]
mayusculas = [p.upper() for p in palabras]
print(mayusculas)  # ['HOLA', 'MUNDO', 'PYTHON']

# Filtrar y transformar
numeros = [1, -2, 3, -4, 5, -6]
positivos_dobles = [x * 2 for x in numeros if x > 0]
print(positivos_dobles)  # [2, 6, 10]

# Con if-else
numeros = [1, 2, 3, 4, 5]
resultado = ["par" if x % 2 == 0 else "impar" for x in numeros]
print(resultado)  # ['impar', 'par', 'impar', 'par', 'impar']

# Aplanar una lista de listas
matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
plana = [num for fila in matriz for num in fila]
print(plana)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 4.8. Ejercicios Prácticos

### Ejercicio 1: Calculadora con Menú

```python
while True:
    print("\n=== CALCULADORA ===")
    print("1. Sumar")
    print("2. Restar")
    print("3. Multiplicar")
    print("4. Dividir")
    print("5. Salir")
    
    opcion = input("Elige una opción: ")
    
    if opcion == "5":
        print("¡Hasta luego!")
        break
    
    if opcion in ["1", "2", "3", "4"]:
        num1 = float(input("Primer número: "))
        num2 = float(input("Segundo número: "))
        
        if opcion == "1":
            resultado = num1 + num2
            print(f"{num1} + {num2} = {resultado}")
        elif opcion == "2":
            resultado = num1 - num2
            print(f"{num1} - {num2} = {resultado}")
        elif opcion == "3":
            resultado = num1 * num2
            print(f"{num1} × {num2} = {resultado}")
        elif opcion == "4":
            if num2 != 0:
                resultado = num1 / num2
                print(f"{num1} ÷ {num2} = {resultado}")
            else:
                print("Error: No se puede dividir por cero")
    else:
        print("Opción no válida")
```

### Ejercicio 2: Números Primos

```python
n = int(input("Mostrar primos hasta: "))

print(f"Números primos hasta {n}:")

for num in range(2, n + 1):
    es_primo = True
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            es_primo = False
            break
    if es_primo:
        print(num, end=" ")

print()
```

### Ejercicio 3: FizzBuzz

```python
# Clásico ejercicio de programación
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

### Ejercicio 4: Validación de Contraseña

```python
while True:
    password = input("Crea una contraseña: ")
    
    es_valida = True
    errores = []
    
    if len(password) < 8:
        es_valida = False
        errores.append("- Debe tener al menos 8 caracteres")
    
    tiene_mayuscula = False
    tiene_minuscula = False
    tiene_numero = False
    
    for caracter in password:
        if caracter.isupper():
            tiene_mayuscula = True
        elif caracter.islower():
            tiene_minuscula = True
        elif caracter.isdigit():
            tiene_numero = True
    
    if not tiene_mayuscula:
        es_valida = False
        errores.append("- Debe tener al menos una mayúscula")
    if not tiene_minuscula:
        es_valida = False
        errores.append("- Debe tener al menos una minúscula")
    if not tiene_numero:
        es_valida = False
        errores.append("- Debe tener al menos un número")
    
    if es_valida:
        print("✓ Contraseña válida")
        break
    else:
        print("✗ Contraseña no válida:")
        for error in errores:
            print(error)
```

### Ejercicio 5: Piedra, Papel, Tijeras

```python
import random

opciones = ["piedra", "papel", "tijeras"]
puntos_jugador = 0
puntos_computadora = 0

print("=== PIEDRA, PAPEL, TIJERAS ===")
print("(Escribe 'salir' para terminar)")

while True:
    jugador = input("\nTu elección (piedra/papel/tijeras): ").lower()
    
    if jugador == "salir":
        break
    
    if jugador not in opciones:
        print("Opción no válida")
        continue
    
    computadora = random.choice(opciones)
    print(f"Computadora eligió: {computadora}")
    
    if jugador == computadora:
        print("¡Empate!")
    elif (jugador == "piedra" and computadora == "tijeras") or \
         (jugador == "papel" and computadora == "piedra") or \
         (jugador == "tijeras" and computadora == "papel"):
        print("¡Ganaste!")
        puntos_jugador += 1
    else:
        print("Perdiste...")
        puntos_computadora += 1
    
    print(f"Marcador: Tú {puntos_jugador} - {puntos_computadora} Computadora")

print(f"\nResultado final: Tú {puntos_jugador} - {puntos_computadora} Computadora")
```

---

## 4.9. Resumen

| Estructura | Uso |
| :--- | :--- |
| `if` | Ejecutar código si se cumple una condición |
| `elif` | Condición alternativa |
| `else` | Si ninguna condición anterior se cumple |
| `match` / `case` | Comparar una expresión con múltiples patrones |
| `_` | Caso por defecto en `match` |
| `while` | Repetir mientras una condición sea verdadera |
| `for` | Iterar sobre una secuencia |
| `break` | Salir del bucle |
| `continue` | Saltar a la siguiente iteración |
| `range()` | Generar secuencia de números |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
