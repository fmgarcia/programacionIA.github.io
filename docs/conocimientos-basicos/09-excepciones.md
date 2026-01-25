# 📚 Unidad 9. Manejo de Excepciones

Las **excepciones** son errores que ocurren durante la ejecución del programa. El manejo de excepciones permite controlar estos errores de forma elegante.

---

## 9.1. ¿Qué son las Excepciones?

Cuando ocurre un error en Python, se genera una excepción que detiene el programa.

```python
# Error de división por cero
resultado = 10 / 0  # ZeroDivisionError

# Error de índice
lista = [1, 2, 3]
print(lista[10])  # IndexError

# Error de tipo
numero = "texto" + 5  # TypeError

# Error de clave
diccionario = {"a": 1}
print(diccionario["b"])  # KeyError

# Error de archivo
archivo = open("no_existe.txt")  # FileNotFoundError
```

---

## 9.2. Try / Except

### Sintaxis Básica

```python
try:
    # Código que puede causar error
    resultado = 10 / 0
except:
    # Se ejecuta si hay error
    print("¡Ocurrió un error!")
```

### Capturar Excepciones Específicas

```python
try:
    numero = int(input("Introduce un número: "))
    resultado = 100 / numero
    print(f"Resultado: {resultado}")
except ValueError:
    print("Error: Debes introducir un número válido")
except ZeroDivisionError:
    print("Error: No se puede dividir por cero")
```

### Capturar Múltiples Excepciones

```python
try:
    lista = [1, 2, 3]
    indice = int(input("Índice: "))
    valor = lista[indice]
    resultado = 10 / valor
except (ValueError, IndexError) as e:
    print(f"Error de entrada: {e}")
except ZeroDivisionError:
    print("Error: División por cero")
```

### Obtener Información del Error

```python
try:
    resultado = 10 / 0
except ZeroDivisionError as e:
    print(f"Tipo de error: {type(e).__name__}")
    print(f"Mensaje: {e}")
```

---

## 9.3. Else y Finally

### Bloque `else`

Se ejecuta si NO hubo excepciones.

```python
try:
    numero = int(input("Número: "))
    resultado = 100 / numero
except ValueError:
    print("Error: No es un número válido")
except ZeroDivisionError:
    print("Error: No se puede dividir por cero")
else:
    # Se ejecuta solo si no hubo errores
    print(f"El resultado es: {resultado}")
```

### Bloque `finally`

Se ejecuta SIEMPRE, haya o no excepciones.

```python
try:
    archivo = open("datos.txt", "r")
    contenido = archivo.read()
except FileNotFoundError:
    print("Error: Archivo no encontrado")
finally:
    # Siempre se ejecuta
    print("Operación finalizada")
    # Aquí se suelen cerrar recursos
```

### Ejemplo Completo

```python
def dividir(a, b):
    try:
        resultado = a / b
    except ZeroDivisionError:
        print("Error: División por cero")
        return None
    except TypeError:
        print("Error: Los valores deben ser numéricos")
        return None
    else:
        print("División exitosa")
        return resultado
    finally:
        print("Función dividir() finalizada")

print(dividir(10, 2))   # División exitosa, 5.0
print(dividir(10, 0))   # Error, None
print(dividir("a", 2))  # Error, None
```

---

## 9.4. Tipos Comunes de Excepciones

| Excepción | Descripción |
| :--- | :--- |
| `ValueError` | Valor incorrecto |
| `TypeError` | Tipo de dato incorrecto |
| `KeyError` | Clave no existe en diccionario |
| `IndexError` | Índice fuera de rango |
| `ZeroDivisionError` | División por cero |
| `FileNotFoundError` | Archivo no encontrado |
| `AttributeError` | Atributo/método no existe |
| `ImportError` | Error al importar módulo |
| `NameError` | Variable no definida |
| `PermissionError` | Sin permisos |
| `ConnectionError` | Error de conexión |
| `TimeoutError` | Tiempo de espera agotado |

### Ejemplos de Cada Tipo

```python
# ValueError
int("texto")  # No se puede convertir

# TypeError
"texto" + 5  # No se puede sumar string con int
len(123)     # int no tiene longitud

# KeyError
d = {"a": 1}
d["b"]  # Clave no existe

# IndexError
lista = [1, 2, 3]
lista[10]  # Índice fuera de rango

# AttributeError
numero = 5
numero.append(6)  # int no tiene método append

# NameError
print(variable_no_definida)

# FileNotFoundError
open("archivo_inexistente.txt")
```

---

## 9.5. Raise - Lanzar Excepciones

Podemos lanzar excepciones manualmente con `raise`.

```python
def validar_edad(edad):
    if edad < 0:
        raise ValueError("La edad no puede ser negativa")
    if edad > 150:
        raise ValueError("La edad no es realista")
    return True

try:
    validar_edad(-5)
except ValueError as e:
    print(f"Error de validación: {e}")
```

### Ejemplos Prácticos

```python
def dividir(a, b):
    if b == 0:
        raise ZeroDivisionError("El divisor no puede ser cero")
    return a / b

def procesar_lista(lista):
    if not isinstance(lista, list):
        raise TypeError("Se esperaba una lista")
    if len(lista) == 0:
        raise ValueError("La lista no puede estar vacía")
    return sum(lista) / len(lista)

# Uso
try:
    resultado = dividir(10, 0)
except ZeroDivisionError as e:
    print(e)

try:
    media = procesar_lista([])
except ValueError as e:
    print(e)
```

### Re-lanzar Excepciones

```python
def procesar_datos(datos):
    try:
        resultado = datos[0] / datos[1]
        return resultado
    except Exception as e:
        print(f"Error capturado: {e}")
        raise  # Re-lanza la misma excepción

try:
    procesar_datos([10, 0])
except ZeroDivisionError:
    print("Error manejado en el nivel superior")
```

---

## 9.6. Excepciones Personalizadas

Podemos crear nuestras propias excepciones.

```python
class MiError(Exception):
    """Excepción personalizada básica."""
    pass

class EdadInvalidaError(Exception):
    """Error cuando la edad no es válida."""
    def __init__(self, edad, mensaje="Edad no válida"):
        self.edad = edad
        self.mensaje = mensaje
        super().__init__(self.mensaje)
    
    def __str__(self):
        return f"{self.mensaje}: {self.edad}"

# Uso
def verificar_edad(edad):
    if edad < 0 or edad > 150:
        raise EdadInvalidaError(edad)
    if edad < 18:
        raise EdadInvalidaError(edad, "Debe ser mayor de edad")
    return True

try:
    verificar_edad(-5)
except EdadInvalidaError as e:
    print(e)  # Edad no válida: -5

try:
    verificar_edad(15)
except EdadInvalidaError as e:
    print(e)  # Debe ser mayor de edad: 15
```

### Jerarquía de Excepciones

```python
class ErrorAplicacion(Exception):
    """Clase base para errores de la aplicación."""
    pass

class ErrorValidacion(ErrorAplicacion):
    """Error en validación de datos."""
    pass

class ErrorBaseDatos(ErrorAplicacion):
    """Error en operaciones de base de datos."""
    pass

class ErrorConexion(ErrorBaseDatos):
    """Error de conexión a la base de datos."""
    pass

# Uso
def conectar_db():
    raise ErrorConexion("No se pudo conectar al servidor")

try:
    conectar_db()
except ErrorBaseDatos as e:
    print(f"Error de BD: {e}")
except ErrorAplicacion as e:
    print(f"Error de aplicación: {e}")
```

---

## 9.7. Context Managers

Los **context managers** garantizan que los recursos se liberen correctamente.

### El Statement `with`

```python
# Sin context manager (puede haber problemas)
archivo = open("datos.txt", "r")
contenido = archivo.read()
archivo.close()  # Puede no ejecutarse si hay error

# Con context manager (recomendado)
with open("datos.txt", "r") as archivo:
    contenido = archivo.read()
# El archivo se cierra automáticamente
```

### Múltiples Context Managers

```python
with open("entrada.txt", "r") as entrada, open("salida.txt", "w") as salida:
    for linea in entrada:
        salida.write(linea.upper())
```

### Crear Context Manager con `contextlib`

```python
from contextlib import contextmanager

@contextmanager
def temporizador():
    import time
    inicio = time.time()
    print("Iniciando...")
    yield  # Aquí se ejecuta el código del bloque with
    fin = time.time()
    print(f"Tiempo transcurrido: {fin - inicio:.2f} segundos")

# Uso
with temporizador():
    suma = sum(range(1000000))
    print(f"Suma: {suma}")
```

---

## 9.8. Buenas Prácticas

### 1. Ser Específico con las Excepciones

```python
# MAL - Captura todo
try:
    # código
    pass
except:
    print("Error")

# BIEN - Captura específica
try:
    # código
    pass
except ValueError as e:
    print(f"Error de valor: {e}")
except TypeError as e:
    print(f"Error de tipo: {e}")
```

### 2. No Silenciar Excepciones

```python
# MAL - Ignora el error completamente
try:
    resultado = 10 / 0
except:
    pass

# BIEN - Al menos registrar el error
try:
    resultado = 10 / 0
except ZeroDivisionError as e:
    print(f"Advertencia: {e}")
    resultado = 0
```

### 3. Usar Finally para Limpieza

```python
conexion = None
try:
    conexion = abrir_conexion()
    # usar conexión
except Exception as e:
    print(f"Error: {e}")
finally:
    if conexion:
        conexion.cerrar()
```

### 4. Validar Antes cuando sea Posible

```python
# En lugar de capturar excepción
def obtener_elemento_v1(lista, indice):
    try:
        return lista[indice]
    except IndexError:
        return None

# Validar antes (EAFP vs LBYL)
def obtener_elemento_v2(lista, indice):
    if 0 <= indice < len(lista):
        return lista[indice]
    return None
```

---

## 9.9. Ejercicios Prácticos

### Ejercicio 1: Calculadora Segura

```python
class CalculadoraError(Exception):
    """Error de la calculadora."""
    pass

class DivisionPorCeroError(CalculadoraError):
    """Error de división por cero."""
    pass

class OperacionInvalidaError(CalculadoraError):
    """Operación no reconocida."""
    pass

def calculadora_segura():
    operaciones = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b != 0 else None,
        "**": lambda a, b: a ** b
    }
    
    while True:
        print("\n=== CALCULADORA ===")
        entrada = input("Operación (ej: 5 + 3) o 'salir': ").strip()
        
        if entrada.lower() == "salir":
            print("¡Hasta luego!")
            break
        
        try:
            partes = entrada.split()
            if len(partes) != 3:
                raise OperacionInvalidaError("Formato: número operador número")
            
            num1, op, num2 = partes
            num1 = float(num1)
            num2 = float(num2)
            
            if op not in operaciones:
                raise OperacionInvalidaError(f"Operador '{op}' no válido")
            
            if op == "/" and num2 == 0:
                raise DivisionPorCeroError("No se puede dividir por cero")
            
            resultado = operaciones[op](num1, num2)
            print(f"Resultado: {resultado}")
            
        except ValueError:
            print("Error: Introduce números válidos")
        except CalculadoraError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")

calculadora_segura()
```

### Ejercicio 2: Gestor de Archivos Seguro

```python
import os
import json

class ArchivoError(Exception):
    """Error relacionado con archivos."""
    pass

class ArchivoNoEncontradoError(ArchivoError):
    """El archivo no existe."""
    pass

class FormatoInvalidoError(ArchivoError):
    """El formato del archivo no es válido."""
    pass

def leer_json_seguro(ruta):
    """Lee un archivo JSON de forma segura."""
    if not os.path.exists(ruta):
        raise ArchivoNoEncontradoError(f"No existe: {ruta}")
    
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FormatoInvalidoError(f"JSON inválido: {e}")
    except PermissionError:
        raise ArchivoError(f"Sin permisos para leer: {ruta}")

def escribir_json_seguro(datos, ruta, crear_backup=True):
    """Escribe datos en un archivo JSON de forma segura."""
    # Crear backup si existe
    if crear_backup and os.path.exists(ruta):
        backup = ruta + ".backup"
        try:
            import shutil
            shutil.copy(ruta, backup)
        except Exception as e:
            print(f"Advertencia: No se pudo crear backup: {e}")
    
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        return True
    except PermissionError:
        raise ArchivoError(f"Sin permisos para escribir: {ruta}")
    except Exception as e:
        raise ArchivoError(f"Error al escribir: {e}")

# Uso
try:
    datos = leer_json_seguro("config.json")
    print("Datos cargados:", datos)
except ArchivoNoEncontradoError:
    print("Archivo no encontrado, creando uno nuevo...")
    datos = {"configuracion": "default"}
    escribir_json_seguro(datos, "config.json")
except FormatoInvalidoError as e:
    print(f"Error de formato: {e}")
except ArchivoError as e:
    print(f"Error de archivo: {e}")
```

### Ejercicio 3: Validador de Formulario

```python
class ValidacionError(Exception):
    """Error de validación."""
    def __init__(self, campo, mensaje):
        self.campo = campo
        self.mensaje = mensaje
        super().__init__(f"{campo}: {mensaje}")

class Validador:
    @staticmethod
    def validar_email(email):
        if not email:
            raise ValidacionError("email", "El email es obligatorio")
        if "@" not in email or "." not in email:
            raise ValidacionError("email", "Formato de email inválido")
        return True
    
    @staticmethod
    def validar_edad(edad):
        try:
            edad = int(edad)
        except (ValueError, TypeError):
            raise ValidacionError("edad", "Debe ser un número")
        
        if edad < 0:
            raise ValidacionError("edad", "No puede ser negativa")
        if edad < 18:
            raise ValidacionError("edad", "Debe ser mayor de 18 años")
        if edad > 120:
            raise ValidacionError("edad", "Edad no realista")
        return True
    
    @staticmethod
    def validar_telefono(telefono):
        numeros = telefono.replace(" ", "").replace("-", "")
        if not numeros.isdigit():
            raise ValidacionError("teléfono", "Solo se permiten dígitos")
        if len(numeros) != 9:
            raise ValidacionError("teléfono", "Debe tener 9 dígitos")
        return True
    
    @staticmethod
    def validar_password(password):
        if len(password) < 8:
            raise ValidacionError("contraseña", "Mínimo 8 caracteres")
        if not any(c.isupper() for c in password):
            raise ValidacionError("contraseña", "Debe contener mayúsculas")
        if not any(c.isdigit() for c in password):
            raise ValidacionError("contraseña", "Debe contener números")
        return True

def procesar_formulario(datos):
    """Procesa y valida un formulario."""
    errores = []
    
    # Validar cada campo
    validaciones = [
        (Validador.validar_email, datos.get("email", "")),
        (Validador.validar_edad, datos.get("edad", "")),
        (Validador.validar_telefono, datos.get("telefono", "")),
        (Validador.validar_password, datos.get("password", ""))
    ]
    
    for validar, valor in validaciones:
        try:
            validar(valor)
        except ValidacionError as e:
            errores.append(str(e))
    
    if errores:
        print("Errores de validación:")
        for error in errores:
            print(f"  - {error}")
        return False
    
    print("Formulario válido ✓")
    return True

# Probar
formulario = {
    "email": "usuario@ejemplo.com",
    "edad": "25",
    "telefono": "612 345 678",
    "password": "MiPassword123"
}
procesar_formulario(formulario)

formulario_invalido = {
    "email": "invalido",
    "edad": "quince",
    "telefono": "123",
    "password": "corta"
}
procesar_formulario(formulario_invalido)
```

---

## 9.10. Resumen

| Concepto | Descripción |
| :--- | :--- |
| `try` | Bloque que puede causar excepciones |
| `except` | Captura y maneja excepciones |
| `else` | Se ejecuta si no hay excepciones |
| `finally` | Se ejecuta siempre |
| `raise` | Lanza una excepción |
| `Exception` | Clase base para excepciones |
| `as e` | Captura la excepción en variable |

```python
try:
    # Código que puede fallar
    resultado = operacion_riesgosa()
except TipoError as e:
    # Manejar error específico
    print(f"Error: {e}")
except Exception as e:
    # Manejar cualquier otro error
    print(f"Error inesperado: {e}")
else:
    # Se ejecuta si no hubo errores
    print("Éxito")
finally:
    # Limpieza, se ejecuta siempre
    liberar_recursos()
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
