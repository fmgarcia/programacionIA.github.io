# 📚 Matplotlib - Visualización de Datos

**Matplotlib** es la librería más utilizada para crear gráficos y visualizaciones en Python. Es altamente personalizable y sirve como base para otras librerías de visualización.

---

## 1. Instalación e Importación

```python
# Instalación
# pip install matplotlib

# Importación (convención estándar)
import matplotlib.pyplot as plt
import numpy as np
```

---

## 2. Gráfico Básico

```python
import matplotlib.pyplot as plt

# Datos
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Crear gráfico
plt.plot(x, y)

# Mostrar
plt.show()
```

### Con Títulos y Etiquetas

```python
plt.plot(x, y)
plt.title("Mi Primer Gráfico")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()
```

### Guardar Figura

```python
plt.plot(x, y)
plt.title("Gráfico")
plt.savefig("grafico.png", dpi=300, bbox_inches="tight")
plt.savefig("grafico.pdf")  # También PDF, SVG, etc.
plt.show()
```

---

## 3. Tipos de Gráficos

### Gráfico de Líneas

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin(x)")
plt.plot(x, y2, label="cos(x)")
plt.title("Funciones Trigonométricas")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
```

### Personalizar Líneas

```python
x = np.arange(0, 10, 1)
y = x ** 2

# Estilos de línea
plt.plot(x, y, color="red", linewidth=2, linestyle="--", marker="o")

# Formato corto: 'color-marcador-línea'
plt.plot(x, y, "b-o")   # Azul, línea sólida, círculos
plt.plot(x, y, "r--s")  # Rojo, línea discontinua, cuadrados
plt.plot(x, y, "g:^")   # Verde, línea punteada, triángulos

plt.show()
```

Colores: `b`(blue), `g`(green), `r`(red), `c`(cyan), `m`(magenta), `y`(yellow), `k`(black), `w`(white)

Marcadores: `o`(círculo), `s`(cuadrado), `^`(triángulo), `*`(estrella), `+`, `x`, `.`

Líneas: `-`(sólida), `--`(discontinua), `:`(punteada), `-.`(punto-raya)

### Gráfico de Dispersión (Scatter)

```python
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
colores = np.random.rand(100)
tamaños = np.random.rand(100) * 500

plt.scatter(x, y, c=colores, s=tamaños, alpha=0.6, cmap="viridis")
plt.colorbar(label="Valor")
plt.title("Gráfico de Dispersión")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### Gráfico de Barras

```python
categorias = ["A", "B", "C", "D", "E"]
valores = [23, 45, 56, 78, 32]

# Barras verticales
plt.bar(categorias, valores, color="steelblue", edgecolor="black")
plt.title("Gráfico de Barras")
plt.xlabel("Categoría")
plt.ylabel("Valor")
plt.show()

# Barras horizontales
plt.barh(categorias, valores, color="coral")
plt.title("Barras Horizontales")
plt.show()
```

### Barras Agrupadas

```python
categorias = ["Q1", "Q2", "Q3", "Q4"]
producto_a = [20, 35, 30, 35]
producto_b = [25, 32, 34, 20]

x = np.arange(len(categorias))
ancho = 0.35

plt.bar(x - ancho/2, producto_a, ancho, label="Producto A", color="steelblue")
plt.bar(x + ancho/2, producto_b, ancho, label="Producto B", color="coral")

plt.xlabel("Trimestre")
plt.ylabel("Ventas")
plt.title("Ventas por Producto")
plt.xticks(x, categorias)
plt.legend()
plt.show()
```

### Barras Apiladas

```python
categorias = ["A", "B", "C", "D"]
valores1 = [20, 35, 30, 25]
valores2 = [25, 32, 34, 20]

plt.bar(categorias, valores1, label="Serie 1")
plt.bar(categorias, valores2, bottom=valores1, label="Serie 2")
plt.legend()
plt.title("Barras Apiladas")
plt.show()
```

### Histograma

```python
datos = np.random.randn(1000)

plt.hist(datos, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
plt.title("Histograma")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()

# Múltiples histogramas
datos1 = np.random.randn(1000)
datos2 = np.random.randn(1000) + 2

plt.hist(datos1, bins=30, alpha=0.5, label="Grupo 1")
plt.hist(datos2, bins=30, alpha=0.5, label="Grupo 2")
plt.legend()
plt.show()
```

### Gráfico de Pastel (Pie)

```python
etiquetas = ["Python", "JavaScript", "Java", "C++", "Otros"]
tamaños = [35, 25, 20, 10, 10]
colores = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
explode = (0.1, 0, 0, 0, 0)  # Destacar Python

plt.pie(tamaños, explode=explode, labels=etiquetas, colors=colores,
        autopct="%1.1f%%", shadow=True, startangle=90)
plt.title("Lenguajes de Programación")
plt.axis("equal")  # Círculo perfecto
plt.show()
```

### Diagrama de Caja (Box Plot)

```python
datos = [np.random.randn(100) + i for i in range(4)]

plt.boxplot(datos, labels=["A", "B", "C", "D"])
plt.title("Diagrama de Caja")
plt.ylabel("Valor")
plt.show()

# Personalizado
plt.boxplot(datos, notch=True, patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.show()
```

### Gráfico de Violín

```python
datos = [np.random.randn(100) * (i+1) for i in range(4)]

plt.violinplot(datos, showmeans=True, showmedians=True)
plt.xticks([1, 2, 3, 4], ["A", "B", "C", "D"])
plt.title("Gráfico de Violín")
plt.show()
```

### Mapa de Calor (Heatmap)

```python
datos = np.random.rand(10, 10)

plt.imshow(datos, cmap="hot", aspect="auto")
plt.colorbar(label="Valor")
plt.title("Mapa de Calor")
plt.show()

# Con anotaciones
fig, ax = plt.subplots()
im = ax.imshow(datos[:5, :5], cmap="YlOrRd")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{datos[i, j]:.2f}", ha="center", va="center")
plt.colorbar(im)
plt.show()
```

### Gráfico de Área

```python
x = np.arange(10)
y1 = np.random.randint(1, 10, 10)
y2 = np.random.randint(1, 10, 10)
y3 = np.random.randint(1, 10, 10)

plt.stackplot(x, y1, y2, y3, labels=["A", "B", "C"], alpha=0.7)
plt.legend(loc="upper left")
plt.title("Gráfico de Área Apilada")
plt.show()
```

---

## 4. Subplots (Múltiples Gráficos)

### Básico

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Gráfico 1 (arriba izquierda)
axes[0, 0].plot([1, 2, 3], [1, 4, 9])
axes[0, 0].set_title("Gráfico 1")

# Gráfico 2 (arriba derecha)
axes[0, 1].bar(["A", "B", "C"], [3, 7, 5])
axes[0, 1].set_title("Gráfico 2")

# Gráfico 3 (abajo izquierda)
axes[1, 0].scatter(np.random.rand(50), np.random.rand(50))
axes[1, 0].set_title("Gráfico 3")

# Gráfico 4 (abajo derecha)
axes[1, 1].hist(np.random.randn(100), bins=20)
axes[1, 1].set_title("Gráfico 4")

plt.tight_layout()  # Ajusta espaciado
plt.show()
```

### Una Fila o Columna

```python
# Una fila
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot([1, 2, 3])
axes[1].bar(["A", "B"], [1, 2])
axes[2].scatter([1, 2], [1, 2])
plt.tight_layout()
plt.show()

# Una columna
fig, axes = plt.subplots(3, 1, figsize=(6, 10))
```

### Subplots con Tamaños Diferentes

```python
fig = plt.figure(figsize=(12, 6))

# Gráfico grande a la izquierda
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(np.sin(np.linspace(0, 10, 100)))
ax1.set_title("Gráfico Principal")

# Dos gráficos pequeños a la derecha
ax2 = fig.add_subplot(2, 2, 2)
ax2.bar(["A", "B", "C"], [1, 2, 3])
ax2.set_title("Gráfico 2")

ax3 = fig.add_subplot(2, 2, 4)
ax3.scatter(np.random.rand(20), np.random.rand(20))
ax3.set_title("Gráfico 3")

plt.tight_layout()
plt.show()
```

---

## 5. Personalización Avanzada

### Configurar Figura

```python
# Tamaño y resolución
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Fondo
fig.patch.set_facecolor("lightgray")
ax.set_facecolor("white")
```

### Ejes y Ticks

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# Límites de ejes
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# Ticks personalizados
ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
ax.set_xticklabels(["0", "π", "2π", "3π"])

# Rotar etiquetas
plt.xticks(rotation=45)

# Grid
ax.grid(True, linestyle="--", alpha=0.7)
ax.grid(True, which="minor", linestyle=":", alpha=0.5)
ax.minorticks_on()

plt.show()
```

### Leyenda

```python
x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x), label="sin(x)")
plt.plot(x, np.cos(x), label="cos(x)")

# Ubicación
plt.legend(loc="upper right")
# Opciones: 'best', 'upper left', 'lower right', 'center', etc.

# Fuera del gráfico
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Personalizada
plt.legend(title="Funciones", fontsize=10, framealpha=0.8,
           facecolor="white", edgecolor="black")

plt.show()
```

### Anotaciones

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# Texto simple
ax.text(5, 0.5, "Texto aquí", fontsize=12, color="red")

# Anotación con flecha
ax.annotate("Máximo", xy=(np.pi/2, 1), xytext=(3, 1.2),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10)

# Línea vertical/horizontal
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=np.pi, color="red", linestyle=":", alpha=0.7)

# Región sombreada
ax.axvspan(2, 4, alpha=0.2, color="yellow")
ax.axhspan(-0.5, 0.5, alpha=0.1, color="green")

plt.show()
```

### Estilos Predefinidos

```python
# Ver estilos disponibles
print(plt.style.available)

# Usar estilo
plt.style.use("seaborn-v0_8")  # o 'ggplot', 'dark_background', etc.

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.title("Con estilo Seaborn")
plt.show()

# Contexto temporal
with plt.style.context("dark_background"):
    plt.plot(x, np.sin(x))
    plt.show()
```

### Colormaps

```python
# Ver colormaps disponibles
# Sequential: 'viridis', 'plasma', 'magma', 'cividis', 'Blues', 'Reds'
# Diverging: 'coolwarm', 'RdYlBu', 'seismic'
# Categorical: 'Set1', 'Set2', 'Pastel1'

datos = np.random.rand(10, 10)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

cmaps = ["viridis", "plasma", "coolwarm", "RdYlBu"]
for ax, cmap in zip(axes.flat, cmaps):
    im = ax.imshow(datos, cmap=cmap)
    ax.set_title(cmap)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

---

## 6. Gráficos 3D

```python
from mpl_toolkits.mplot3d import Axes3D

# Superficie 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Superficie 3D")
plt.show()

# Scatter 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
c = np.random.rand(100)

ax.scatter(x, y, z, c=c, cmap="plasma", s=50)
plt.show()

# Línea 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

t = np.linspace(0, 10*np.pi, 1000)
x = np.sin(t)
y = np.cos(t)
z = t

ax.plot(x, y, z)
ax.set_title("Hélice 3D")
plt.show()
```

---

## 7. Interfaz Orientada a Objetos vs pyplot

### Estilo pyplot (rápido)

```python
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Título")
plt.xlabel("X")
plt.show()
```

### Estilo OO (más control)

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Título")
ax.set_xlabel("X")
plt.show()
```

Recomendación: Usar OO para gráficos complejos y múltiples subplots.

---

## 8. Ejemplo Completo: Dashboard

```python
import matplotlib.pyplot as plt
import numpy as np

# Datos
np.random.seed(42)
meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
ventas = [120, 135, 145, 160, 180, 175]
gastos = [80, 85, 90, 95, 100, 105]
beneficio = np.array(ventas) - np.array(gastos)
productos = ["A", "B", "C", "D"]
ventas_prod = [35, 25, 22, 18]

# Crear figura
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Dashboard de Ventas 2024", fontsize=16, fontweight="bold")

# 1. Gráfico de líneas - Evolución
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(meses, ventas, "b-o", label="Ventas", linewidth=2)
ax1.plot(meses, gastos, "r--s", label="Gastos", linewidth=2)
ax1.fill_between(meses, ventas, gastos, alpha=0.2, color="green")
ax1.set_title("Evolución Ventas vs Gastos")
ax1.set_ylabel("Miles €")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Gráfico de barras - Beneficio
ax2 = fig.add_subplot(2, 2, 2)
colores = ["green" if b > 0 else "red" for b in beneficio]
ax2.bar(meses, beneficio, color=colores, edgecolor="black")
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_title("Beneficio Mensual")
ax2.set_ylabel("Miles €")
for i, v in enumerate(beneficio):
    ax2.text(i, v + 1, f"{v}", ha="center", fontsize=9)

# 3. Gráfico de pastel - Productos
ax3 = fig.add_subplot(2, 2, 3)
explode = (0.05, 0, 0, 0)
ax3.pie(ventas_prod, labels=productos, autopct="%1.1f%%",
        explode=explode, colors=plt.cm.Pastel1.colors[:4],
        shadow=True, startangle=90)
ax3.set_title("Distribución por Producto")

# 4. Histograma - Distribución de transacciones
ax4 = fig.add_subplot(2, 2, 4)
transacciones = np.random.exponential(50, 500)
ax4.hist(transacciones, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
ax4.axvline(np.mean(transacciones), color="red", linestyle="--",
            label=f"Media: {np.mean(transacciones):.1f}€")
ax4.set_title("Distribución de Transacciones")
ax4.set_xlabel("Valor (€)")
ax4.set_ylabel("Frecuencia")
ax4.legend()

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## 9. Animaciones (Introducción)

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10))
    return line,

ani = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.show()

# Guardar como gif (requiere pillow)
# ani.save("animacion.gif", writer="pillow", fps=30)
```

---

## 10. Resumen de Funciones

| Función | Descripción |
| :--- | :--- |
| `plt.plot()` | Gráfico de líneas |
| `plt.scatter()` | Dispersión |
| `plt.bar()`, `plt.barh()` | Barras |
| `plt.hist()` | Histograma |
| `plt.pie()` | Pastel |
| `plt.boxplot()` | Diagrama de caja |
| `plt.imshow()` | Mapa de calor |
| `plt.subplots()` | Múltiples gráficos |
| `plt.title()`, `plt.xlabel()` | Títulos |
| `plt.legend()` | Leyenda |
| `plt.savefig()` | Guardar |
| `plt.show()` | Mostrar |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
