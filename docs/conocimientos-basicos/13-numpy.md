# 📚 NumPy - Computación Numérica

**NumPy** (Numerical Python) es la librería fundamental para computación numérica en Python. Proporciona soporte para arrays multidimensionales y operaciones matemáticas de alto rendimiento.

---

## 1. Instalación e Importación

```python
# Instalación
# pip install numpy

# Importación (convención estándar)
import numpy as np
```

---

## 2. Arrays de NumPy

### Crear Arrays

```python
import numpy as np

# Desde lista
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]

# Array 2D (matriz)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# [[1 2 3]
#  [4 5 6]]

# Array 3D
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3d.shape)  # (2, 2, 2)
```

### Funciones para Crear Arrays

```python
# Array de ceros
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# Array de unos
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Array vacío (valores aleatorios)
empty = np.empty((2, 2))

# Array con valor específico
full = np.full((3, 3), 7)
print(full)
# [[7 7 7]
#  [7 7 7]
#  [7 7 7]]

# Matriz identidad
identidad = np.eye(3)
print(identidad)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Rango de valores
rango = np.arange(0, 10, 2)  # inicio, fin, paso
print(rango)  # [0 2 4 6 8]

# Valores espaciados uniformemente
linspace = np.linspace(0, 1, 5)  # inicio, fin, cantidad
print(linspace)  # [0.   0.25 0.5  0.75 1.  ]

# Valores aleatorios
aleatorio = np.random.rand(3, 3)  # Uniforme [0, 1)
normal = np.random.randn(3, 3)    # Normal (0, 1)
enteros = np.random.randint(0, 100, (3, 3))  # Enteros aleatorios
```

---

## 3. Propiedades de Arrays

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)   # (2, 3) - dimensiones
print(arr.ndim)    # 2 - número de dimensiones
print(arr.size)    # 6 - total de elementos
print(arr.dtype)   # int64 - tipo de datos
print(arr.itemsize)  # 8 - bytes por elemento
print(arr.nbytes)  # 48 - total de bytes
```

### Tipos de Datos

```python
# Especificar tipo al crear
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1, 2, 3], dtype=np.float64)
arr_bool = np.array([1, 0, 1], dtype=np.bool_)

# Convertir tipo
arr = np.array([1.5, 2.7, 3.9])
arr_int = arr.astype(np.int32)
print(arr_int)  # [1 2 3]

# Tipos comunes
# np.int8, np.int16, np.int32, np.int64
# np.float16, np.float32, np.float64
# np.bool_, np.complex64
```

---

## 4. Indexación y Slicing

### Arrays 1D

```python
arr = np.array([10, 20, 30, 40, 50])

# Indexación
print(arr[0])    # 10
print(arr[-1])   # 50

# Slicing
print(arr[1:4])    # [20 30 40]
print(arr[:3])     # [10 20 30]
print(arr[2:])     # [30 40 50]
print(arr[::2])    # [10 30 50] (cada 2)
print(arr[::-1])   # [50 40 30 20 10] (invertido)
```

### Arrays 2D

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Acceder a elemento
print(arr[0, 0])   # 1
print(arr[1, 2])   # 7
print(arr[2, -1])  # 12

# Fila completa
print(arr[0])      # [1 2 3 4]
print(arr[1, :])   # [5 6 7 8]

# Columna completa
print(arr[:, 0])   # [1 5 9]
print(arr[:, -1])  # [4 8 12]

# Submatriz
print(arr[0:2, 1:3])
# [[2 3]
#  [6 7]]

# Filas y columnas específicas
print(arr[[0, 2], :])  # Filas 0 y 2
print(arr[:, [0, 3]])  # Columnas 0 y 3
```

### Indexación Booleana

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Crear máscara booleana
mascara = arr > 5
print(mascara)  # [False False False False False  True  True  True  True  True]

# Filtrar elementos
mayores = arr[arr > 5]
print(mayores)  # [ 6  7  8  9 10]

# Condiciones múltiples
pares_mayores = arr[(arr > 3) & (arr % 2 == 0)]
print(pares_mayores)  # [ 4  6  8 10]

# Modificar con condición
arr[arr < 5] = 0
print(arr)  # [ 0  0  0  0  5  6  7  8  9 10]
```

### Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Índices específicos
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# Para matrices
matriz = np.arange(1, 10).reshape(3, 3)
print(matriz)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

filas = [0, 1, 2]
cols = [2, 1, 0]
print(matriz[filas, cols])  # [3 5 7] (diagonal inversa)
```

---

## 5. Manipulación de Arrays

### Reshape (Cambiar Forma)

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# Reshape a 3x4
matriz = arr.reshape(3, 4)
print(matriz)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Reshape a 2x2x3
tensor = arr.reshape(2, 2, 3)

# -1 calcula automáticamente
auto = arr.reshape(4, -1)  # 4 filas, columnas automáticas
print(auto.shape)  # (4, 3)

# Aplanar
plano = matriz.flatten()  # Copia
plano = matriz.ravel()    # Vista
```

### Concatenar y Dividir

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Concatenar
vertical = np.vstack([a, b])  # Vertical
print(vertical)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

horizontal = np.hstack([a, b])  # Horizontal
print(horizontal)
# [[1 2 5 6]
#  [3 4 7 8]]

# concatenate con axis
concat_v = np.concatenate([a, b], axis=0)  # Vertical
concat_h = np.concatenate([a, b], axis=1)  # Horizontal

# Dividir
arr = np.arange(16).reshape(4, 4)
partes = np.split(arr, 2)  # Dividir en 2
v1, v2 = np.vsplit(arr, 2)  # Vertical
h1, h2 = np.hsplit(arr, 2)  # Horizontal
```

### Transponer

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)

transpuesta = arr.T
print(transpuesta.shape)  # (3, 2)
print(transpuesta)
# [[1 4]
#  [2 5]
#  [3 6]]

# También: np.transpose(arr)
```

### Añadir Dimensiones

```python
arr = np.array([1, 2, 3])
print(arr.shape)  # (3,)

# Añadir dimensión
arr_fila = arr[np.newaxis, :]  # (1, 3)
arr_col = arr[:, np.newaxis]   # (3, 1)

# expand_dims
arr_exp = np.expand_dims(arr, axis=0)  # (1, 3)
arr_exp = np.expand_dims(arr, axis=1)  # (3, 1)

# squeeze (eliminar dimensiones de tamaño 1)
arr = np.array([[[1, 2, 3]]])  # (1, 1, 3)
arr_squeeze = arr.squeeze()  # (3,)
```

---

## 6. Operaciones Matemáticas

### Operaciones Elemento a Elemento

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Aritméticas
print(a + b)   # [11 22 33 44]
print(a - b)   # [ -9 -18 -27 -36]
print(a * b)   # [ 10  40  90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]
print(a ** 2)  # [ 1  4  9 16]
print(a % 2)   # [1 0 1 0]

# Con escalares
print(a + 10)  # [11 12 13 14]
print(a * 2)   # [2 4 6 8]
```

### Funciones Matemáticas

```python
arr = np.array([1, 4, 9, 16, 25])

# Raíz cuadrada
print(np.sqrt(arr))  # [1. 2. 3. 4. 5.]

# Exponencial y logaritmo
print(np.exp(arr))   # Exponencial
print(np.log(arr))   # Logaritmo natural
print(np.log10(arr)) # Logaritmo base 10

# Trigonométricas
angulos = np.array([0, np.pi/2, np.pi])
print(np.sin(angulos))  # [0.0000000e+00 1.0000000e+00 1.2246468e-16]
print(np.cos(angulos))  # [ 1.000000e+00  6.123234e-17 -1.000000e+00]

# Redondeo
arr = np.array([1.2, 2.5, 3.7, 4.1])
print(np.round(arr))   # [1. 2. 4. 4.]
print(np.floor(arr))   # [1. 2. 3. 4.]
print(np.ceil(arr))    # [2. 3. 4. 5.]
print(np.trunc(arr))   # [1. 2. 3. 4.]

# Valor absoluto
arr = np.array([-1, -2, 3, -4])
print(np.abs(arr))  # [1 2 3 4]
```

### Funciones de Agregación

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr))       # 21
print(np.mean(arr))      # 3.5
print(np.std(arr))       # Desviación estándar
print(np.var(arr))       # Varianza
print(np.min(arr))       # 1
print(np.max(arr))       # 6
print(np.prod(arr))      # Producto (720)

# Por eje
print(np.sum(arr, axis=0))  # Por columnas: [5 7 9]
print(np.sum(arr, axis=1))  # Por filas: [ 6 15]

print(np.mean(arr, axis=0))  # Media por columnas
print(np.mean(arr, axis=1))  # Media por filas

# Índices de min/max
print(np.argmin(arr))  # 0 (índice del mínimo)
print(np.argmax(arr))  # 5 (índice del máximo)

# Acumulativos
print(np.cumsum(arr))    # [ 1  3  6 10 15 21]
print(np.cumprod(arr))   # [  1   2   6  24 120 720]
```

---

## 7. Álgebra Lineal

```python
# Producto punto
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32

# Multiplicación de matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))  # Producto matricial
print(A @ B)         # Igual que np.dot para matrices
# [[19 22]
#  [43 50]]

# Multiplicación elemento a elemento (NO matricial)
print(A * B)
# [[ 5 12]
#  [21 32]]

# Transpuesta
print(A.T)

# Determinante
print(np.linalg.det(A))  # -2.0

# Matriz inversa
print(np.linalg.inv(A))

# Autovalores y autovectores
valores, vectores = np.linalg.eig(A)
print("Autovalores:", valores)
print("Autovectores:", vectores)

# Resolver sistema de ecuaciones Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 5])
x = np.linalg.solve(A, b)
print("Solución:", x)  # [2. 1.]

# Norma
v = np.array([3, 4])
print(np.linalg.norm(v))  # 5.0 (norma euclidiana)

# Rango de matriz
print(np.linalg.matrix_rank(A))
```

---

## 8. Broadcasting

Broadcasting permite operar arrays de diferentes formas.

```python
# Escalar con array
arr = np.array([1, 2, 3])
print(arr * 2)  # [2 4 6]

# Array 1D con 2D
matriz = np.array([[1, 2, 3], [4, 5, 6]])
fila = np.array([10, 20, 30])

print(matriz + fila)
# [[11 22 33]
#  [14 25 36]]

# Columna con matriz
columna = np.array([[100], [200]])
print(matriz + columna)
# [[101 102 103]
#  [204 205 206]]

# Crear tabla de multiplicar con broadcasting
filas = np.arange(1, 11).reshape(10, 1)
cols = np.arange(1, 11)
tabla = filas * cols
print(tabla)
```

### Reglas de Broadcasting

1. Si los arrays tienen diferente número de dimensiones, se añaden 1s a la izquierda.
2. Arrays con tamaño 1 en una dimensión se estiran para coincidir.
3. Si los tamaños no coinciden y ninguno es 1, error.

```python
# Ejemplo: (3,) y (3, 3)
a = np.array([1, 2, 3])       # (3,)
b = np.ones((3, 3))           # (3, 3)
# a se convierte en (1, 3) -> broadcasting a (3, 3)
print(a + b)
```

---

## 9. Funciones de Comparación

```python
arr = np.array([1, 2, 3, 4, 5])

# Comparaciones (devuelven arrays booleanos)
print(arr > 3)   # [False False False  True  True]
print(arr == 3)  # [False False  True False False]
print(arr != 3)  # [ True  True False  True  True]

# np.where (como if-else vectorizado)
resultado = np.where(arr > 3, "Grande", "Pequeño")
print(resultado)  # ['Pequeño' 'Pequeño' 'Pequeño' 'Grande' 'Grande']

# np.where para encontrar índices
indices = np.where(arr > 3)
print(indices)  # (array([3, 4]),)

# any y all
print(np.any(arr > 3))   # True (al menos uno)
print(np.all(arr > 0))   # True (todos)
print(np.any(arr > 10))  # False

# Comparar arrays
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])
print(np.array_equal(a, b))  # False
print(np.allclose(a, b, atol=1))  # True (tolerancia)
```

---

## 10. Números Aleatorios

```python
# Establecer semilla para reproducibilidad
np.random.seed(42)

# Distribución uniforme [0, 1)
uniform = np.random.rand(3, 3)

# Distribución normal (media=0, std=1)
normal = np.random.randn(3, 3)

# Enteros aleatorios
enteros = np.random.randint(0, 100, size=(5, 5))

# Elegir de un array
arr = np.array([10, 20, 30, 40, 50])
eleccion = np.random.choice(arr, size=3, replace=False)

# Mezclar array
np.random.shuffle(arr)

# Permutación (devuelve nuevo array)
permutado = np.random.permutation(arr)

# Otras distribuciones
binomial = np.random.binomial(n=10, p=0.5, size=100)
poisson = np.random.poisson(lam=5, size=100)
exponencial = np.random.exponential(scale=1.0, size=100)

# Nueva API (recomendada)
rng = np.random.default_rng(seed=42)
valores = rng.random((3, 3))
enteros = rng.integers(0, 100, size=(3, 3))
```

---

## 11. Copiar Arrays

```python
# Vista (comparte memoria)
arr = np.array([1, 2, 3, 4, 5])
vista = arr[1:4]
vista[0] = 100
print(arr)  # [  1 100   3   4   5] - arr también cambió

# Copia (independiente)
arr = np.array([1, 2, 3, 4, 5])
copia = arr[1:4].copy()
copia[0] = 100
print(arr)  # [1 2 3 4 5] - arr no cambió

# Verificar si comparten memoria
print(np.shares_memory(arr, vista))  # True
print(np.shares_memory(arr, copia))  # False
```

---

## 12. Guardar y Cargar Arrays

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Formato binario NumPy (.npy)
np.save("array.npy", arr)
cargado = np.load("array.npy")

# Múltiples arrays (.npz)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.savez("arrays.npz", arr_a=a, arr_b=b)

datos = np.load("arrays.npz")
print(datos["arr_a"])
print(datos["arr_b"])

# Formato texto
np.savetxt("array.txt", arr, delimiter=",")
cargado = np.loadtxt("array.txt", delimiter=",")

# CSV con encabezado
np.savetxt("datos.csv", arr, delimiter=",", 
           header="col1,col2,col3", comments="")
```

---

## 13. Ejemplo Práctico: Análisis de Datos

```python
import numpy as np

# Simular datos de ventas (100 días, 5 productos)
np.random.seed(42)
ventas = np.random.randint(10, 100, size=(100, 5))

# Estadísticas básicas
print("Total de ventas:", np.sum(ventas))
print("Media diaria por producto:", np.mean(ventas, axis=0))
print("Mejor día (ventas totales):", np.argmax(np.sum(ventas, axis=1)))
print("Producto más vendido:", np.argmax(np.sum(ventas, axis=0)))

# Días con ventas superiores al promedio
promedio_diario = np.mean(np.sum(ventas, axis=1))
dias_buenos = np.where(np.sum(ventas, axis=1) > promedio_diario)[0]
print(f"Días sobre el promedio: {len(dias_buenos)}")

# Normalizar datos
ventas_norm = (ventas - np.min(ventas)) / (np.max(ventas) - np.min(ventas))

# Correlación entre productos
correlacion = np.corrcoef(ventas.T)
print("Correlación producto 0 y 1:", correlacion[0, 1])
```

---

## 14. Resumen de Funciones

| Función | Descripción |
| :--- | :--- |
| `np.array()` | Crear array |
| `np.zeros()`, `np.ones()` | Arrays de ceros/unos |
| `np.arange()`, `np.linspace()` | Rangos de valores |
| `reshape()`, `flatten()` | Cambiar forma |
| `np.vstack()`, `np.hstack()` | Concatenar |
| `np.sum()`, `np.mean()` | Agregaciones |
| `np.dot()`, `@` | Producto matricial |
| `np.where()` | Condición vectorizada |
| `np.save()`, `np.load()` | Guardar/cargar |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
