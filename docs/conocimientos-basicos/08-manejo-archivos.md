# 📚 Unidad 8. Manejo de Archivos

El manejo de archivos es fundamental para leer datos, guardar resultados y trabajar con información persistente.

---

## 8.1. Abrir y Cerrar Archivos

### Función `open()`

```python
# Sintaxis básica
archivo = open("nombre_archivo.txt", "modo")
# ... operaciones ...
archivo.close()  # Importante cerrar
```

### Modos de Apertura

| Modo | Descripción |
| :--- | :--- |
| `"r"` | Lectura (por defecto). Error si no existe |
| `"w"` | Escritura. Crea archivo o sobrescribe |
| `"a"` | Añadir. Escribe al final |
| `"x"` | Creación exclusiva. Error si existe |
| `"r+"` | Lectura y escritura |
| `"w+"` | Escritura y lectura (sobrescribe) |
| `"a+"` | Añadir y lectura |
| `"b"` | Modo binario (añadir a otros: `"rb"`, `"wb"`) |

### Uso con `with` (Recomendado)

```python
# El archivo se cierra automáticamente al salir del bloque
with open("archivo.txt", "r") as archivo:
    contenido = archivo.read()
    print(contenido)
# Aquí el archivo ya está cerrado
```

---

## 8.2. Lectura de Archivos

### Leer Todo el Contenido

```python
# Leer todo como string
with open("archivo.txt", "r") as f:
    contenido = f.read()
    print(contenido)

# Leer con codificación específica
with open("archivo.txt", "r", encoding="utf-8") as f:
    contenido = f.read()
```

### Leer Línea por Línea

```python
# Leer una línea
with open("archivo.txt", "r") as f:
    primera_linea = f.readline()
    segunda_linea = f.readline()
    print(primera_linea)
    print(segunda_linea)

# Leer todas las líneas como lista
with open("archivo.txt", "r") as f:
    lineas = f.readlines()
    print(lineas)  # ['línea1\n', 'línea2\n', ...]

# Iterar sobre líneas (más eficiente para archivos grandes)
with open("archivo.txt", "r") as f:
    for linea in f:
        print(linea.strip())  # strip() quita \n
```

### Leer Cantidad Específica

```python
with open("archivo.txt", "r") as f:
    # Leer primeros 100 caracteres
    primeros_100 = f.read(100)
    
    # Leer siguientes 50
    siguientes_50 = f.read(50)
```

### Ejemplos Prácticos de Lectura

```python
# Contar líneas de un archivo
with open("datos.txt", "r") as f:
    num_lineas = sum(1 for _ in f)
print(f"El archivo tiene {num_lineas} líneas")

# Buscar palabra en archivo
def buscar_palabra(archivo, palabra):
    with open(archivo, "r", encoding="utf-8") as f:
        for num_linea, linea in enumerate(f, 1):
            if palabra.lower() in linea.lower():
                print(f"Línea {num_linea}: {linea.strip()}")

buscar_palabra("texto.txt", "python")

# Leer archivo y procesar datos
with open("numeros.txt", "r") as f:
    numeros = [int(linea.strip()) for linea in f if linea.strip()]
    print(f"Suma: {sum(numeros)}")
    print(f"Media: {sum(numeros)/len(numeros)}")
```

---

## 8.3. Escritura de Archivos

### Escribir Texto

```python
# Sobrescribir archivo (o crear si no existe)
with open("salida.txt", "w", encoding="utf-8") as f:
    f.write("Primera línea\n")
    f.write("Segunda línea\n")

# Añadir al final
with open("salida.txt", "a", encoding="utf-8") as f:
    f.write("Línea añadida\n")
```

### Escribir Múltiples Líneas

```python
lineas = ["Línea 1", "Línea 2", "Línea 3"]

# Con writelines (no añade \n automáticamente)
with open("archivo.txt", "w") as f:
    f.writelines(linea + "\n" for linea in lineas)

# Con join
with open("archivo.txt", "w") as f:
    f.write("\n".join(lineas))
```

### Ejemplos Prácticos de Escritura

```python
# Crear informe
def crear_informe(datos, archivo_salida):
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("        INFORME DE DATOS\n")
        f.write("=" * 40 + "\n\n")
        
        for i, dato in enumerate(datos, 1):
            f.write(f"{i}. {dato}\n")
        
        f.write(f"\nTotal: {len(datos)} elementos\n")

datos = ["Manzana", "Naranja", "Plátano"]
crear_informe(datos, "informe.txt")

# Registro de log
from datetime import datetime

def log(mensaje, archivo="app.log"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(archivo, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {mensaje}\n")

log("Aplicación iniciada")
log("Usuario conectado")
log("Error: archivo no encontrado")
```

---

## 8.4. Archivos CSV

CSV (Comma-Separated Values) es un formato común para datos tabulares.

### Módulo `csv`

```python
import csv
```

### Leer CSV

```python
import csv

# Leer como lista de listas
with open("datos.csv", "r", encoding="utf-8") as f:
    lector = csv.reader(f)
    for fila in lector:
        print(fila)  # ['valor1', 'valor2', ...]

# Leer como diccionarios (con encabezados)
with open("datos.csv", "r", encoding="utf-8") as f:
    lector = csv.DictReader(f)
    for fila in lector:
        print(fila)  # {'columna1': 'valor1', ...}
        print(fila["nombre"])  # Acceder por nombre de columna
```

### Escribir CSV

```python
import csv

# Escribir lista de listas
datos = [
    ["Nombre", "Edad", "Ciudad"],
    ["Ana", 25, "Madrid"],
    ["Luis", 30, "Barcelona"],
    ["María", 28, "Valencia"]
]

with open("personas.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerows(datos)  # Escribir todas las filas

# Escribir fila por fila
with open("personas.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerow(["Nombre", "Edad", "Ciudad"])  # Encabezado
    escritor.writerow(["Ana", 25, "Madrid"])
    escritor.writerow(["Luis", 30, "Barcelona"])

# Escribir desde diccionarios
personas = [
    {"nombre": "Ana", "edad": 25, "ciudad": "Madrid"},
    {"nombre": "Luis", "edad": 30, "ciudad": "Barcelona"}
]

with open("personas.csv", "w", newline="", encoding="utf-8") as f:
    campos = ["nombre", "edad", "ciudad"]
    escritor = csv.DictWriter(f, fieldnames=campos)
    escritor.writeheader()  # Escribir encabezados
    escritor.writerows(personas)
```

### Delimitadores Personalizados

```python
# Usar punto y coma como delimitador
with open("datos.csv", "r", encoding="utf-8") as f:
    lector = csv.reader(f, delimiter=";")
    for fila in lector:
        print(fila)

# Usar tabulador
with open("datos.tsv", "w", newline="") as f:
    escritor = csv.writer(f, delimiter="\t")
    escritor.writerow(["Col1", "Col2", "Col3"])
```

### Ejemplo Completo: Sistema de Inventario

```python
import csv
import os

ARCHIVO = "inventario.csv"

def inicializar():
    """Crea el archivo si no existe."""
    if not os.path.exists(ARCHIVO):
        with open(ARCHIVO, "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerow(["ID", "Producto", "Cantidad", "Precio"])

def leer_inventario():
    """Lee y muestra el inventario."""
    with open(ARCHIVO, "r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        productos = list(lector)
        
    if not productos:
        print("Inventario vacío")
        return
    
    print("\n" + "=" * 50)
    print(f"{'ID':<5} {'Producto':<20} {'Cantidad':<10} {'Precio':<10}")
    print("-" * 50)
    for p in productos:
        print(f"{p['ID']:<5} {p['Producto']:<20} {p['Cantidad']:<10} {p['Precio']:<10}")
    print("=" * 50)

def agregar_producto(id, nombre, cantidad, precio):
    """Añade un producto al inventario."""
    with open(ARCHIVO, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow([id, nombre, cantidad, precio])
    print(f"Producto '{nombre}' añadido")

# Uso
inicializar()
agregar_producto("001", "Laptop", 10, 999.99)
agregar_producto("002", "Mouse", 50, 29.99)
agregar_producto("003", "Teclado", 30, 79.99)
leer_inventario()
```

---

## 8.5. Archivos JSON

JSON (JavaScript Object Notation) es ideal para datos estructurados.

```python
import json
```

### Leer JSON

```python
import json

# Desde archivo
with open("datos.json", "r", encoding="utf-8") as f:
    datos = json.load(f)
    print(datos)

# Desde string
json_str = '{"nombre": "Ana", "edad": 25}'
datos = json.loads(json_str)
print(datos["nombre"])  # Ana
```

### Escribir JSON

```python
import json

datos = {
    "nombre": "Ana",
    "edad": 25,
    "ciudades": ["Madrid", "Barcelona"],
    "activo": True
}

# A archivo
with open("datos.json", "w", encoding="utf-8") as f:
    json.dump(datos, f, indent=4, ensure_ascii=False)

# A string
json_str = json.dumps(datos, indent=4, ensure_ascii=False)
print(json_str)
```

### Ejemplo: Configuración de Aplicación

```python
import json
import os

ARCHIVO_CONFIG = "config.json"

CONFIG_DEFAULT = {
    "idioma": "es",
    "tema": "claro",
    "notificaciones": True,
    "max_resultados": 50
}

def cargar_config():
    """Carga configuración o crea archivo por defecto."""
    if os.path.exists(ARCHIVO_CONFIG):
        with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        guardar_config(CONFIG_DEFAULT)
        return CONFIG_DEFAULT.copy()

def guardar_config(config):
    """Guarda configuración en archivo."""
    with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def actualizar_config(clave, valor):
    """Actualiza una opción de configuración."""
    config = cargar_config()
    config[clave] = valor
    guardar_config(config)
    print(f"Configuración actualizada: {clave} = {valor}")

# Uso
config = cargar_config()
print(f"Idioma actual: {config['idioma']}")

actualizar_config("tema", "oscuro")
actualizar_config("max_resultados", 100)
```

### Ejemplo: Base de Datos Simple

```python
import json
import os

class BaseDatosJSON:
    def __init__(self, archivo):
        self.archivo = archivo
        self._inicializar()
    
    def _inicializar(self):
        if not os.path.exists(self.archivo):
            self._guardar([])
    
    def _cargar(self):
        with open(self.archivo, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _guardar(self, datos):
        with open(self.archivo, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
    
    def insertar(self, registro):
        datos = self._cargar()
        registro["id"] = len(datos) + 1
        datos.append(registro)
        self._guardar(datos)
        return registro["id"]
    
    def buscar(self, **criterios):
        datos = self._cargar()
        resultados = []
        for registro in datos:
            if all(registro.get(k) == v for k, v in criterios.items()):
                resultados.append(registro)
        return resultados
    
    def actualizar(self, id, **campos):
        datos = self._cargar()
        for registro in datos:
            if registro.get("id") == id:
                registro.update(campos)
                self._guardar(datos)
                return True
        return False
    
    def eliminar(self, id):
        datos = self._cargar()
        datos = [r for r in datos if r.get("id") != id]
        self._guardar(datos)
    
    def todos(self):
        return self._cargar()

# Uso
db = BaseDatosJSON("usuarios.json")

# Insertar
db.insertar({"nombre": "Ana", "email": "ana@email.com", "edad": 25})
db.insertar({"nombre": "Luis", "email": "luis@email.com", "edad": 30})
db.insertar({"nombre": "María", "email": "maria@email.com", "edad": 25})

# Buscar
print("Usuarios de 25 años:", db.buscar(edad=25))

# Actualizar
db.actualizar(1, edad=26)

# Ver todos
print("Todos los usuarios:", db.todos())
```

---

## 8.6. Manejo de Rutas

### Módulo `os.path`

```python
import os

# Obtener directorio actual
print(os.getcwd())

# Construir rutas
ruta = os.path.join("carpeta", "subcarpeta", "archivo.txt")
print(ruta)  # carpeta\subcarpeta\archivo.txt (Windows)

# Obtener partes de la ruta
ruta = "/home/usuario/documentos/archivo.txt"
print(os.path.basename(ruta))  # archivo.txt
print(os.path.dirname(ruta))   # /home/usuario/documentos
print(os.path.splitext(ruta))  # ('/home/.../archivo', '.txt')

# Verificar existencia
print(os.path.exists("archivo.txt"))
print(os.path.isfile("archivo.txt"))
print(os.path.isdir("carpeta"))

# Ruta absoluta
print(os.path.abspath("archivo.txt"))
```

### Módulo `pathlib` (Moderno)

```python
from pathlib import Path

# Crear objeto Path
ruta = Path("documentos/archivo.txt")

# Propiedades
print(ruta.name)        # archivo.txt
print(ruta.stem)        # archivo
print(ruta.suffix)      # .txt
print(ruta.parent)      # documentos
print(ruta.parts)       # ('documentos', 'archivo.txt')

# Verificar
print(ruta.exists())
print(ruta.is_file())
print(ruta.is_dir())

# Construir rutas
nueva_ruta = Path("carpeta") / "subcarpeta" / "archivo.txt"
print(nueva_ruta)

# Listar archivos
carpeta = Path(".")
for archivo in carpeta.iterdir():
    print(archivo)

# Buscar archivos por patrón
for py_file in Path(".").glob("*.py"):
    print(py_file)

# Buscar recursivamente
for txt_file in Path(".").rglob("*.txt"):
    print(txt_file)

# Crear directorios
Path("nueva_carpeta/subcarpeta").mkdir(parents=True, exist_ok=True)

# Leer/escribir directamente
archivo = Path("archivo.txt")
archivo.write_text("Hola mundo", encoding="utf-8")
contenido = archivo.read_text(encoding="utf-8")
```

---

## 8.7. Operaciones con Archivos

### Copiar, Mover, Eliminar

```python
import shutil
import os

# Copiar archivo
shutil.copy("origen.txt", "destino.txt")
shutil.copy2("origen.txt", "destino.txt")  # Preserva metadata

# Copiar directorio
shutil.copytree("carpeta_origen", "carpeta_destino")

# Mover archivo/directorio
shutil.move("archivo.txt", "nueva_ubicacion/archivo.txt")

# Eliminar archivo
os.remove("archivo.txt")

# Eliminar directorio vacío
os.rmdir("carpeta_vacia")

# Eliminar directorio con contenido
shutil.rmtree("carpeta_con_archivos")
```

### Renombrar

```python
import os

# Renombrar archivo
os.rename("nombre_viejo.txt", "nombre_nuevo.txt")

# Renombrar múltiples archivos
for archivo in os.listdir("."):
    if archivo.endswith(".txt"):
        nuevo_nombre = archivo.replace(".txt", "_backup.txt")
        os.rename(archivo, nuevo_nombre)
```

---

## 8.8. Archivos Binarios

Para imágenes, audio, ejecutables, etc.

```python
# Leer archivo binario
with open("imagen.png", "rb") as f:
    datos = f.read()
    print(f"Tamaño: {len(datos)} bytes")

# Escribir archivo binario
with open("copia.png", "wb") as f:
    f.write(datos)

# Copiar archivo binario
with open("original.pdf", "rb") as origen:
    with open("copia.pdf", "wb") as destino:
        destino.write(origen.read())
```

---

## 8.9. Ejercicios Prácticos

### Ejercicio 1: Análisis de Texto

```python
def analizar_archivo(ruta):
    """Analiza un archivo de texto."""
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
    except FileNotFoundError:
        return "Error: Archivo no encontrado"
    except Exception as e:
        return f"Error: {e}"
    
    lineas = contenido.split("\n")
    palabras = contenido.split()
    caracteres = len(contenido)
    caracteres_sin_espacios = len(contenido.replace(" ", "").replace("\n", ""))
    
    # Palabra más frecuente
    frecuencias = {}
    for palabra in palabras:
        palabra = palabra.lower().strip(".,;:!?\"'")
        if palabra:
            frecuencias[palabra] = frecuencias.get(palabra, 0) + 1
    
    palabra_mas_comun = max(frecuencias.items(), key=lambda x: x[1]) if frecuencias else None
    
    return {
        "lineas": len(lineas),
        "palabras": len(palabras),
        "caracteres": caracteres,
        "caracteres_sin_espacios": caracteres_sin_espacios,
        "palabra_mas_comun": palabra_mas_comun
    }

# Uso
resultado = analizar_archivo("texto.txt")
for clave, valor in resultado.items():
    print(f"{clave}: {valor}")
```

### Ejercicio 2: Procesador de Notas CSV

```python
import csv

def procesar_notas(archivo_entrada, archivo_salida):
    """Procesa notas de estudiantes y genera informe."""
    estudiantes = []
    
    # Leer datos
    with open(archivo_entrada, "r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            nombre = fila["nombre"]
            notas = [float(fila[k]) for k in fila if k.startswith("nota")]
            media = sum(notas) / len(notas)
            aprobado = media >= 5
            estudiantes.append({
                "nombre": nombre,
                "notas": notas,
                "media": round(media, 2),
                "aprobado": aprobado
            })
    
    # Escribir informe
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write("INFORME DE CALIFICACIONES\n")
        f.write("=" * 40 + "\n\n")
        
        for e in estudiantes:
            estado = "APROBADO" if e["aprobado"] else "SUSPENSO"
            f.write(f"Estudiante: {e['nombre']}\n")
            f.write(f"  Notas: {e['notas']}\n")
            f.write(f"  Media: {e['media']}\n")
            f.write(f"  Estado: {estado}\n\n")
        
        # Estadísticas
        medias = [e["media"] for e in estudiantes]
        aprobados = sum(1 for e in estudiantes if e["aprobado"])
        
        f.write("-" * 40 + "\n")
        f.write("ESTADÍSTICAS\n")
        f.write(f"  Total estudiantes: {len(estudiantes)}\n")
        f.write(f"  Aprobados: {aprobados}\n")
        f.write(f"  Suspensos: {len(estudiantes) - aprobados}\n")
        f.write(f"  Media general: {sum(medias)/len(medias):.2f}\n")
    
    print(f"Informe generado en {archivo_salida}")

# Crear CSV de ejemplo
with open("notas.csv", "w", newline="", encoding="utf-8") as f:
    escritor = csv.writer(f)
    escritor.writerow(["nombre", "nota1", "nota2", "nota3"])
    escritor.writerow(["Ana", 8, 7, 9])
    escritor.writerow(["Luis", 5, 4, 6])
    escritor.writerow(["María", 9, 10, 8])
    escritor.writerow(["Pedro", 3, 4, 2])

procesar_notas("notas.csv", "informe_notas.txt")
```

### Ejercicio 3: Organizador de Archivos

```python
import os
import shutil
from pathlib import Path

def organizar_por_extension(directorio):
    """Organiza archivos en carpetas por extensión."""
    
    carpetas = {
        "Imágenes": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documentos": [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".pptx"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mkv", ".mov"],
        "Código": [".py", ".js", ".html", ".css", ".java", ".cpp"],
        "Comprimidos": [".zip", ".rar", ".7z", ".tar", ".gz"]
    }
    
    path = Path(directorio)
    
    for archivo in path.iterdir():
        if archivo.is_file():
            extension = archivo.suffix.lower()
            
            # Encontrar carpeta destino
            destino = "Otros"
            for carpeta, extensiones in carpetas.items():
                if extension in extensiones:
                    destino = carpeta
                    break
            
            # Crear carpeta si no existe
            carpeta_destino = path / destino
            carpeta_destino.mkdir(exist_ok=True)
            
            # Mover archivo
            try:
                shutil.move(str(archivo), str(carpeta_destino / archivo.name))
                print(f"Movido: {archivo.name} -> {destino}/")
            except Exception as e:
                print(f"Error moviendo {archivo.name}: {e}")

# Uso (¡cuidado con el directorio que uses!)
# organizar_por_extension("./descargas")
```

---

## 8.10. Resumen

| Operación | Código |
| :--- | :--- |
| Abrir archivo | `with open("archivo.txt", "r") as f:` |
| Leer todo | `contenido = f.read()` |
| Leer líneas | `lineas = f.readlines()` |
| Escribir | `f.write("texto")` |
| Modo escritura | `"w"` (sobrescribe), `"a"` (añade) |
| CSV lectura | `csv.reader(f)` o `csv.DictReader(f)` |
| CSV escritura | `csv.writer(f)` o `csv.DictWriter(f)` |
| JSON lectura | `json.load(f)` |
| JSON escritura | `json.dump(datos, f)` |
| Rutas modernas | `from pathlib import Path` |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
