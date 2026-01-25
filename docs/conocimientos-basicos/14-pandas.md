# 📚 Pandas - Análisis de Datos

**Pandas** es la librería más popular para análisis y manipulación de datos en Python. Proporciona estructuras de datos flexibles (Series y DataFrame) y herramientas para trabajar con datos tabulares.

---

## 1. Instalación e Importación

```python
# Instalación
# pip install pandas

# Importación (convención estándar)
import pandas as pd
import numpy as np
```

---

## 2. Estructuras de Datos

### Series

Una **Series** es un array unidimensional etiquetado.

```python
# Crear Series desde lista
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# Con índices personalizados
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s)
# a    10
# b    20
# c    30

# Desde diccionario
datos = {"manzanas": 10, "naranjas": 20, "plátanos": 15}
s = pd.Series(datos)
print(s)

# Acceso a elementos
print(s["manzanas"])  # 10
print(s[0])           # 10 (por posición)
print(s.values)       # array NumPy
print(s.index)        # Index(['manzanas', 'naranjas', 'plátanos'])
```

### DataFrame

Un **DataFrame** es una tabla 2D con filas y columnas etiquetadas.

```python
# Desde diccionario de listas
datos = {
    "nombre": ["Ana", "Luis", "María", "Pedro"],
    "edad": [25, 30, 28, 35],
    "ciudad": ["Madrid", "Barcelona", "Valencia", "Sevilla"]
}
df = pd.DataFrame(datos)
print(df)
#   nombre  edad     ciudad
# 0    Ana    25     Madrid
# 1   Luis    30  Barcelona
# 2  María    28   Valencia
# 3  Pedro    35    Sevilla

# Con índice personalizado
df = pd.DataFrame(datos, index=["a", "b", "c", "d"])

# Desde lista de diccionarios
registros = [
    {"nombre": "Ana", "edad": 25},
    {"nombre": "Luis", "edad": 30}
]
df = pd.DataFrame(registros)

# Desde array NumPy
arr = np.random.randint(0, 100, (4, 3))
df = pd.DataFrame(arr, columns=["A", "B", "C"])
```

---

## 3. Leer y Escribir Datos

### CSV

```python
# Leer CSV
df = pd.read_csv("datos.csv")

# Con opciones
df = pd.read_csv("datos.csv",
                 sep=";",              # Separador
                 header=0,             # Fila de encabezados
                 index_col=0,          # Columna como índice
                 usecols=["a", "b"],   # Solo estas columnas
                 nrows=100,            # Primeras N filas
                 encoding="utf-8",     # Codificación
                 na_values=["N/A"])    # Valores nulos

# Escribir CSV
df.to_csv("salida.csv", index=False)
df.to_csv("salida.csv", sep=";", encoding="utf-8")
```

### Excel

```python
# Leer Excel (requiere openpyxl o xlrd)
df = pd.read_excel("datos.xlsx", sheet_name="Hoja1")

# Leer múltiples hojas
hojas = pd.read_excel("datos.xlsx", sheet_name=None)  # Diccionario

# Escribir Excel
df.to_excel("salida.xlsx", index=False, sheet_name="Datos")

# Múltiples hojas
with pd.ExcelWriter("salida.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Hoja1")
    df2.to_excel(writer, sheet_name="Hoja2")
```

### Otros Formatos

```python
# JSON
df = pd.read_json("datos.json")
df.to_json("salida.json", orient="records")

# SQL
import sqlite3
conn = sqlite3.connect("base_datos.db")
df = pd.read_sql("SELECT * FROM tabla", conn)
df.to_sql("tabla_nueva", conn, if_exists="replace")

# HTML
tablas = pd.read_html("https://ejemplo.com/tabla")  # Lista de DataFrames
df.to_html("tabla.html")

# Pickle (binario Python)
df.to_pickle("datos.pkl")
df = pd.read_pickle("datos.pkl")

# Parquet (eficiente para big data)
df.to_parquet("datos.parquet")
df = pd.read_parquet("datos.parquet")
```

---

## 4. Explorar Datos

```python
# Crear DataFrame de ejemplo
df = pd.DataFrame({
    "nombre": ["Ana", "Luis", "María", "Pedro", "Carmen"],
    "edad": [25, 30, 28, 35, 22],
    "salario": [35000, 45000, 40000, 55000, 32000],
    "departamento": ["IT", "Ventas", "IT", "RRHH", "Ventas"]
})

# Primeras/últimas filas
print(df.head())      # Primeras 5
print(df.head(3))     # Primeras 3
print(df.tail())      # Últimas 5

# Información general
print(df.info())      # Tipos, nulos, memoria
print(df.shape)       # (5, 4)
print(df.columns)     # Nombres de columnas
print(df.index)       # Índice
print(df.dtypes)      # Tipos por columna

# Estadísticas descriptivas
print(df.describe())         # Numéricas
print(df.describe(include="all"))  # Todas
print(df.describe(include=["object"]))  # Solo texto

# Valores únicos
print(df["departamento"].unique())      # Array de valores únicos
print(df["departamento"].nunique())     # Cantidad de únicos
print(df["departamento"].value_counts())  # Frecuencia
```

---

## 5. Selección de Datos

### Seleccionar Columnas

```python
# Una columna (Series)
print(df["nombre"])
print(df.nombre)  # Equivalente si no tiene espacios

# Varias columnas (DataFrame)
print(df[["nombre", "edad"]])
```

### Seleccionar Filas

```python
# Por índice posicional (iloc)
print(df.iloc[0])       # Primera fila (Series)
print(df.iloc[0:3])     # Primeras 3 filas
print(df.iloc[[0, 2, 4]])  # Filas específicas

# Por etiqueta (loc)
print(df.loc[0])        # Fila con índice 0
print(df.loc[0:2])      # Filas 0 a 2 (incluido)

# Filas y columnas
print(df.iloc[0:3, 0:2])      # Primeras 3 filas, primeras 2 columnas
print(df.loc[0:2, ["nombre", "edad"]])  # Por etiquetas
```

### Filtrar con Condiciones

```python
# Condición simple
mayores_30 = df[df["edad"] > 30]
print(mayores_30)

# Condiciones múltiples
filtro = df[(df["edad"] > 25) & (df["salario"] > 35000)]
print(filtro)

# OR
filtro = df[(df["departamento"] == "IT") | (df["departamento"] == "Ventas")]

# isin() para múltiples valores
filtro = df[df["departamento"].isin(["IT", "RRHH"])]

# Texto con str
filtro = df[df["nombre"].str.startswith("M")]
filtro = df[df["nombre"].str.contains("ar")]

# query() - sintaxis más legible
filtro = df.query("edad > 25 and salario > 35000")
filtro = df.query("departamento == 'IT'")
```

---

## 6. Modificar Datos

### Añadir/Modificar Columnas

```python
# Nueva columna
df["bonus"] = df["salario"] * 0.1
df["email"] = df["nombre"].str.lower() + "@empresa.com"

# Columna condicional
df["senior"] = df["edad"] > 30
df["categoria"] = np.where(df["salario"] > 40000, "Alto", "Normal")

# apply() para funciones personalizadas
def categorizar_edad(edad):
    if edad < 25:
        return "Joven"
    elif edad < 35:
        return "Adulto"
    return "Senior"

df["grupo_edad"] = df["edad"].apply(categorizar_edad)

# map() para reemplazar valores
mapa_dept = {"IT": "Tecnología", "RRHH": "Recursos Humanos", "Ventas": "Comercial"}
df["dept_largo"] = df["departamento"].map(mapa_dept)
```

### Modificar Valores

```python
# Modificar valor específico
df.loc[0, "salario"] = 36000
df.iloc[0, 2] = 36000

# Modificar con condición
df.loc[df["departamento"] == "IT", "salario"] *= 1.1  # Aumento 10%

# replace()
df["departamento"] = df["departamento"].replace("IT", "Tech")
df = df.replace({"IT": "Tech", "RRHH": "HR"})
```

### Renombrar

```python
# Renombrar columnas
df = df.rename(columns={"nombre": "empleado", "edad": "años"})

# Renombrar todas
df.columns = ["col1", "col2", "col3", "col4"]

# Renombrar índice
df = df.rename(index={0: "primero", 1: "segundo"})
```

### Eliminar

```python
# Eliminar columnas
df = df.drop(columns=["bonus", "email"])
df = df.drop("bonus", axis=1)

# Eliminar filas
df = df.drop([0, 1])  # Por índice
df = df.drop(df[df["edad"] < 25].index)  # Por condición
```

---

## 7. Valores Nulos

```python
# Crear DataFrame con nulos
df = pd.DataFrame({
    "A": [1, 2, None, 4],
    "B": [None, 2, 3, 4],
    "C": [1, None, None, 4]
})

# Detectar nulos
print(df.isnull())      # DataFrame booleano
print(df.isna())        # Igual
print(df.isnull().sum())  # Cantidad por columna
print(df.isnull().sum().sum())  # Total

# Eliminar nulos
df_limpio = df.dropna()           # Filas con algún nulo
df_limpio = df.dropna(how="all")  # Filas completamente nulas
df_limpio = df.dropna(subset=["A", "B"])  # Solo si nulo en A o B

# Rellenar nulos
df_relleno = df.fillna(0)              # Con valor
df_relleno = df.fillna(df.mean())      # Con media
df_relleno = df.fillna(method="ffill") # Forward fill
df_relleno = df.fillna(method="bfill") # Backward fill

# Interpolación
df_interp = df.interpolate()
```

---

## 8. Ordenar Datos

```python
df = pd.DataFrame({
    "nombre": ["Ana", "Luis", "María", "Pedro"],
    "edad": [25, 30, 28, 35],
    "salario": [35000, 45000, 40000, 55000]
})

# Ordenar por columna
df_ordenado = df.sort_values("edad")
df_ordenado = df.sort_values("edad", ascending=False)

# Por múltiples columnas
df_ordenado = df.sort_values(["edad", "salario"], ascending=[True, False])

# Por índice
df_ordenado = df.sort_index()

# nlargest / nsmallest
top3 = df.nlargest(3, "salario")
bottom3 = df.nsmallest(3, "edad")

# Ranking
df["rank_salario"] = df["salario"].rank(ascending=False)
```

---

## 9. Agrupar Datos (GroupBy)

```python
df = pd.DataFrame({
    "departamento": ["IT", "Ventas", "IT", "Ventas", "RRHH"],
    "empleado": ["Ana", "Luis", "María", "Pedro", "Carmen"],
    "salario": [35000, 45000, 40000, 55000, 38000],
    "años_empresa": [2, 5, 3, 8, 4]
})

# Agrupar por una columna
grupo = df.groupby("departamento")

# Agregaciones
print(grupo["salario"].mean())    # Media por departamento
print(grupo["salario"].sum())     # Suma
print(grupo.size())               # Tamaño de cada grupo

# Múltiples agregaciones
print(grupo["salario"].agg(["mean", "sum", "count", "min", "max"]))

# Agregaciones diferentes por columna
resultado = grupo.agg({
    "salario": ["mean", "sum"],
    "años_empresa": "mean"
})

# agg() con funciones personalizadas
def rango(x):
    return x.max() - x.min()

print(grupo["salario"].agg(rango))

# Agrupar por múltiples columnas
df["senior"] = df["años_empresa"] > 3
grupo2 = df.groupby(["departamento", "senior"])
print(grupo2["salario"].mean())

# transform() - mantiene forma original
df["salario_medio_dept"] = grupo["salario"].transform("mean")
df["salario_vs_media"] = df["salario"] - df["salario_medio_dept"]
```

---

## 10. Combinar DataFrames

### Concatenar

```python
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

# Vertical (filas)
concat_v = pd.concat([df1, df2])
concat_v = pd.concat([df1, df2], ignore_index=True)

# Horizontal (columnas)
concat_h = pd.concat([df1, df2], axis=1)
```

### Merge (JOIN)

```python
empleados = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "nombre": ["Ana", "Luis", "María", "Pedro"],
    "dept_id": [10, 20, 10, 30]
})

departamentos = pd.DataFrame({
    "dept_id": [10, 20, 30],
    "departamento": ["IT", "Ventas", "RRHH"]
})

# Inner join (por defecto)
resultado = pd.merge(empleados, departamentos, on="dept_id")

# Left join
resultado = pd.merge(empleados, departamentos, on="dept_id", how="left")

# Right join
resultado = pd.merge(empleados, departamentos, on="dept_id", how="right")

# Outer join
resultado = pd.merge(empleados, departamentos, on="dept_id", how="outer")

# Columnas con nombres diferentes
df1 = pd.DataFrame({"id_emp": [1, 2], "nombre": ["A", "B"]})
df2 = pd.DataFrame({"emp_id": [1, 2], "salario": [1000, 2000]})
resultado = pd.merge(df1, df2, left_on="id_emp", right_on="emp_id")
```

### Join

```python
df1 = pd.DataFrame({"A": [1, 2]}, index=["a", "b"])
df2 = pd.DataFrame({"B": [3, 4]}, index=["a", "b"])

# Join por índice
resultado = df1.join(df2)
```

---

## 11. Pivot Tables y Reshape

### Pivot Table

```python
df = pd.DataFrame({
    "fecha": ["2024-01", "2024-01", "2024-02", "2024-02"],
    "producto": ["A", "B", "A", "B"],
    "ventas": [100, 150, 120, 180],
    "cantidad": [10, 15, 12, 18]
})

# Pivot table básico
pivot = df.pivot_table(
    values="ventas",
    index="fecha",
    columns="producto",
    aggfunc="sum"
)

# Con múltiples agregaciones
pivot = df.pivot_table(
    values="ventas",
    index="fecha",
    columns="producto",
    aggfunc=["sum", "mean"]
)

# Totales
pivot = df.pivot_table(
    values="ventas",
    index="fecha",
    columns="producto",
    aggfunc="sum",
    margins=True,
    margins_name="Total"
)
```

### Melt (de ancho a largo)

```python
df_ancho = pd.DataFrame({
    "id": [1, 2],
    "nombre": ["Ana", "Luis"],
    "enero": [100, 150],
    "febrero": [110, 160]
})

df_largo = pd.melt(
    df_ancho,
    id_vars=["id", "nombre"],
    value_vars=["enero", "febrero"],
    var_name="mes",
    value_name="ventas"
)
print(df_largo)
#    id nombre      mes  ventas
# 0   1    Ana    enero     100
# 1   2   Luis    enero     150
# 2   1    Ana  febrero     110
# 3   2   Luis  febrero     160
```

### Stack/Unstack

```python
# Stack: de columnas a filas (MultiIndex)
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["x", "y"])
stacked = df.stack()

# Unstack: de filas a columnas
unstacked = stacked.unstack()
```

---

## 12. Trabajar con Fechas

```python
# Crear columna datetime
df = pd.DataFrame({
    "fecha_str": ["2024-01-15", "2024-02-20", "2024-03-25"]
})

df["fecha"] = pd.to_datetime(df["fecha_str"])

# Crear rango de fechas
fechas = pd.date_range("2024-01-01", periods=10, freq="D")
fechas = pd.date_range("2024-01-01", "2024-12-31", freq="M")

# Extraer componentes
df["año"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["dia"] = df["fecha"].dt.day
df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=Lunes
df["nombre_dia"] = df["fecha"].dt.day_name()
df["trimestre"] = df["fecha"].dt.quarter

# Filtrar por fechas
df = df[df["fecha"] > "2024-02-01"]
df = df[df["fecha"].between("2024-01-01", "2024-06-30")]

# Índice temporal
df = df.set_index("fecha")
df_2024 = df.loc["2024"]
df_enero = df.loc["2024-01"]

# Resample (agrupar por tiempo)
df_mensual = df.resample("M").sum()
df_semanal = df.resample("W").mean()
```

---

## 13. Operaciones con Strings

```python
df = pd.DataFrame({
    "nombre": ["  Ana García  ", "LUIS PÉREZ", "maría lópez"],
    "email": ["ana@email.com", "luis@empresa.es", "maria@otro.net"]
})

# Acceso a métodos string con .str
df["nombre_limpio"] = df["nombre"].str.strip()
df["nombre_upper"] = df["nombre"].str.upper()
df["nombre_lower"] = df["nombre"].str.lower()
df["nombre_title"] = df["nombre"].str.title()

# Dividir
df["dominio"] = df["email"].str.split("@").str[1]

# Reemplazar
df["email_nuevo"] = df["email"].str.replace(".com", ".org")

# Contiene
df["es_empresa"] = df["email"].str.contains("empresa")

# Longitud
df["len_nombre"] = df["nombre"].str.len()

# Extraer con regex
df["usuario"] = df["email"].str.extract(r"(.+)@")
```

---

## 14. Ejemplo Completo: Análisis de Ventas

```python
import pandas as pd
import numpy as np

# Crear datos de ejemplo
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "fecha": pd.date_range("2024-01-01", periods=n, freq="H"),
    "producto": np.random.choice(["Laptop", "Mouse", "Teclado", "Monitor"], n),
    "cantidad": np.random.randint(1, 10, n),
    "precio_unitario": np.random.choice([999, 29, 79, 299], n),
    "region": np.random.choice(["Norte", "Sur", "Este", "Oeste"], n)
})

df["total"] = df["cantidad"] * df["precio_unitario"]

# Análisis
print("=== ANÁLISIS DE VENTAS ===\n")

# Resumen general
print("Resumen estadístico:")
print(df[["cantidad", "total"]].describe())

# Ventas por producto
print("\nVentas por producto:")
print(df.groupby("producto")["total"].agg(["sum", "mean", "count"]))

# Ventas por región
print("\nVentas por región:")
print(df.groupby("region")["total"].sum().sort_values(ascending=False))

# Ventas mensuales
df["mes"] = df["fecha"].dt.to_period("M")
print("\nVentas mensuales:")
print(df.groupby("mes")["total"].sum())

# Top productos por región
print("\nProducto más vendido por región:")
print(df.groupby(["region", "producto"])["total"].sum().unstack().idxmax(axis=1))

# Pivot: productos vs regiones
print("\nTabla resumen (productos x regiones):")
pivot = df.pivot_table(
    values="total",
    index="producto",
    columns="region",
    aggfunc="sum",
    margins=True
)
print(pivot)
```

---

## 15. Resumen de Funciones

| Función | Descripción |
| :--- | :--- |
| `pd.read_csv()` | Leer CSV |
| `df.to_csv()` | Escribir CSV |
| `df.head()`, `df.tail()` | Ver filas |
| `df.info()`, `df.describe()` | Información |
| `df.loc[]`, `df.iloc[]` | Seleccionar |
| `df.query()` | Filtrar |
| `df.groupby()` | Agrupar |
| `pd.merge()` | Combinar |
| `pd.concat()` | Concatenar |
| `df.pivot_table()` | Tabla pivote |
| `df.fillna()`, `df.dropna()` | Manejar nulos |
| `df.sort_values()` | Ordenar |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
