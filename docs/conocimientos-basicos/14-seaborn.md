# 📚 Seaborn - Visualización Estadística

**Seaborn** es una librería de visualización basada en Matplotlib que proporciona una interfaz de alto nivel para crear gráficos estadísticos atractivos y informativos.

---

## 1. Instalación e Importación

```python
# Instalación
# pip install seaborn

# Importación (convención estándar)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

---

## 2. Configuración Inicial

```python
# Establecer estilo
sns.set_theme()  # Estilo por defecto de Seaborn

# Estilos disponibles
sns.set_style("whitegrid")   # Fondo blanco con grid
sns.set_style("darkgrid")    # Fondo gris con grid
sns.set_style("white")       # Fondo blanco sin grid
sns.set_style("dark")        # Fondo gris sin grid
sns.set_style("ticks")       # Ejes con marcas

# Contexto (escala de elementos)
sns.set_context("paper")     # Más pequeño
sns.set_context("notebook")  # Por defecto
sns.set_context("talk")      # Para presentaciones
sns.set_context("poster")    # Más grande

# Paleta de colores
sns.set_palette("deep")      # Por defecto
sns.set_palette("pastel")
sns.set_palette("Set2")
sns.set_palette("husl")

# Configuración completa
sns.set_theme(style="whitegrid", palette="deep", context="notebook")
```

---

## 3. Datasets de Ejemplo

Seaborn incluye datasets para practicar:

```python
# Ver datasets disponibles
print(sns.get_dataset_names())

# Cargar datasets
tips = sns.load_dataset("tips")       # Propinas en restaurante
iris = sns.load_dataset("iris")       # Flores iris
titanic = sns.load_dataset("titanic") # Pasajeros del Titanic
penguins = sns.load_dataset("penguins")  # Pingüinos
flights = sns.load_dataset("flights")    # Vuelos mensuales

print(tips.head())
```

---

## 4. Gráficos de Distribución

### Histograma (histplot)

```python
tips = sns.load_dataset("tips")

# Histograma básico
sns.histplot(data=tips, x="total_bill")
plt.title("Distribución de Cuentas")
plt.show()

# Con KDE (densidad)
sns.histplot(data=tips, x="total_bill", kde=True)
plt.show()

# Por categoría
sns.histplot(data=tips, x="total_bill", hue="time", kde=True)
plt.show()

# Múltiples histogramas
sns.histplot(data=tips, x="total_bill", hue="day", multiple="stack")
plt.show()
```

### KDE Plot (Densidad)

```python
# Densidad simple
sns.kdeplot(data=tips, x="total_bill")
plt.show()

# Por grupos
sns.kdeplot(data=tips, x="total_bill", hue="time", fill=True, alpha=0.5)
plt.show()

# 2D
sns.kdeplot(data=tips, x="total_bill", y="tip", fill=True, cmap="Blues")
plt.show()
```

### Rug Plot

```python
# Muestra puntos individuales en el eje
sns.kdeplot(data=tips, x="total_bill")
sns.rugplot(data=tips, x="total_bill", height=0.05)
plt.show()
```

### ECDF Plot (Función de Distribución Acumulada)

```python
sns.ecdfplot(data=tips, x="total_bill", hue="time")
plt.title("ECDF de Cuentas")
plt.show()
```

### Displot (Figura completa)

```python
# Crea figura con facetas
sns.displot(data=tips, x="total_bill", col="time", row="smoker", kde=True)
plt.show()
```

---

## 5. Gráficos Categóricos

### Strip Plot y Swarm Plot

```python
tips = sns.load_dataset("tips")

# Strip plot (puntos dispersos)
sns.stripplot(data=tips, x="day", y="total_bill")
plt.title("Strip Plot")
plt.show()

# Swarm plot (puntos sin solapamiento)
sns.swarmplot(data=tips, x="day", y="total_bill", hue="sex")
plt.title("Swarm Plot")
plt.show()
```

### Box Plot

```python
# Básico
sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("Box Plot")
plt.show()

# Con hue
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")
plt.show()

# Personalizado
sns.boxplot(data=tips, x="day", y="total_bill",
            palette="Set3", linewidth=1.5, fliersize=3)
plt.show()
```

### Violin Plot

```python
# Básico (muestra distribución)
sns.violinplot(data=tips, x="day", y="total_bill")
plt.title("Violin Plot")
plt.show()

# Split por categoría
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True)
plt.show()

# Con puntos internos
sns.violinplot(data=tips, x="day", y="total_bill", inner="points")
plt.show()
```

### Bar Plot (con intervalos de confianza)

```python
# Muestra media con intervalo de confianza
sns.barplot(data=tips, x="day", y="total_bill")
plt.title("Media de Cuenta por Día")
plt.show()

# Por grupos
sns.barplot(data=tips, x="day", y="total_bill", hue="sex")
plt.show()

# Sin intervalos
sns.barplot(data=tips, x="day", y="total_bill", errorbar=None)
plt.show()

# Con otra estadística
sns.barplot(data=tips, x="day", y="total_bill", estimator=np.median)
plt.show()
```

### Count Plot

```python
# Cuenta frecuencias
sns.countplot(data=tips, x="day")
plt.title("Frecuencia por Día")
plt.show()

# Por grupos
sns.countplot(data=tips, x="day", hue="sex")
plt.show()

# Horizontal
sns.countplot(data=tips, y="day", hue="smoker")
plt.show()
```

### Point Plot

```python
# Muestra medias con líneas conectadas
sns.pointplot(data=tips, x="day", y="total_bill", hue="sex")
plt.title("Point Plot")
plt.show()
```

### Catplot (Figura completa categórica)

```python
# Grid de gráficos categóricos
sns.catplot(data=tips, x="day", y="total_bill", col="time",
            kind="box", height=4, aspect=1)
plt.show()

# kind puede ser: 'strip', 'swarm', 'box', 'violin', 'bar', 'count', 'point'
```

---

## 6. Gráficos de Relación

### Scatter Plot

```python
tips = sns.load_dataset("tips")

# Básico
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Cuenta vs Propina")
plt.show()

# Con hue (color por categoría)
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.show()

# Con style (forma por categoría)
sns.scatterplot(data=tips, x="total_bill", y="tip",
                hue="day", style="time", size="size")
plt.show()
```

### Line Plot

```python
flights = sns.load_dataset("flights")

# Línea con intervalo de confianza
sns.lineplot(data=flights, x="year", y="passengers")
plt.title("Pasajeros por Año")
plt.show()

# Por grupos
sns.lineplot(data=flights, x="year", y="passengers", hue="month")
plt.show()
```

### Relplot (Figura completa de relaciones)

```python
sns.relplot(data=tips, x="total_bill", y="tip",
            col="time", hue="smoker", style="smoker",
            kind="scatter", height=4, aspect=1)
plt.show()
```

---

## 7. Gráficos de Regresión

### Regplot

```python
tips = sns.load_dataset("tips")

# Regresión lineal
sns.regplot(data=tips, x="total_bill", y="tip")
plt.title("Regresión Lineal")
plt.show()

# Sin intervalo de confianza
sns.regplot(data=tips, x="total_bill", y="tip", ci=None)
plt.show()

# Regresión polinómica
sns.regplot(data=tips, x="total_bill", y="tip", order=2)
plt.show()

# Regresión lowess (no paramétrica)
sns.regplot(data=tips, x="total_bill", y="tip", lowess=True)
plt.show()
```

### Lmplot (Figura con facetas)

```python
# Grid de regresiones
sns.lmplot(data=tips, x="total_bill", y="tip", hue="smoker")
plt.show()

# Con facetas
sns.lmplot(data=tips, x="total_bill", y="tip", col="time", row="smoker")
plt.show()
```

### Residplot (Residuos)

```python
# Muestra residuos de la regresión
sns.residplot(data=tips, x="total_bill", y="tip")
plt.title("Residuos")
plt.show()
```

---

## 8. Gráficos Matriciales

### Heatmap (Mapa de Calor)

```python
# Matriz de correlación
tips = sns.load_dataset("tips")
corr = tips[["total_bill", "tip", "size"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Matriz de Correlación")
plt.show()

# Personalizado
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
plt.show()

# Para datos de vuelos
flights = sns.load_dataset("flights")
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")

sns.heatmap(flights_pivot, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Pasajeros por Mes y Año")
plt.show()
```

### Clustermap

```python
# Mapa de calor con clustering jerárquico
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")

sns.clustermap(flights_pivot, cmap="viridis", standard_scale=1)
plt.show()
```

---

## 9. Pair Plot (Matriz de Dispersión)

```python
iris = sns.load_dataset("iris")

# Básico
sns.pairplot(iris)
plt.show()

# Por especies
sns.pairplot(iris, hue="species")
plt.show()

# Personalizado
sns.pairplot(iris, hue="species",
             diag_kind="kde",      # En diagonal: 'hist' o 'kde'
             markers=["o", "s", "D"],
             palette="Set2",
             corner=True)          # Solo triángulo inferior
plt.show()

# Seleccionar variables
sns.pairplot(iris, vars=["sepal_length", "sepal_width", "petal_length"],
             hue="species")
plt.show()
```

---

## 10. Joint Plot

```python
tips = sns.load_dataset("tips")

# Básico (scatter + histogramas)
sns.jointplot(data=tips, x="total_bill", y="tip")
plt.show()

# Con regresión
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")
plt.show()

# Con densidad
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde", fill=True)
plt.show()

# Hexágonos
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex")
plt.show()

# Con hue
sns.jointplot(data=tips, x="total_bill", y="tip", hue="time")
plt.show()
```

---

## 11. FacetGrid (Grillas de Gráficos)

```python
tips = sns.load_dataset("tips")

# Crear grid
g = sns.FacetGrid(tips, col="time", row="smoker", height=4)
g.map(sns.scatterplot, "total_bill", "tip")
g.add_legend()
plt.show()

# Con más personalización
g = sns.FacetGrid(tips, col="day", col_wrap=2, height=4)
g.map(sns.histplot, "total_bill", kde=True)
g.set_titles("{col_name}")
plt.show()

# Con hue
g = sns.FacetGrid(tips, col="time", hue="smoker")
g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
g.add_legend()
plt.show()
```

---

## 12. Paletas de Colores

```python
# Ver paletas disponibles
# Cualitativas: deep, muted, pastel, bright, dark, colorblind
# Secuenciales: Blues, Greens, Reds, viridis, rocket, mako
# Divergentes: coolwarm, RdBu, seismic

# Mostrar paleta
sns.palplot(sns.color_palette("deep"))
plt.show()

# Crear paleta personalizada
custom = sns.color_palette(["#ff0000", "#00ff00", "#0000ff"])

# Usar paleta
sns.barplot(data=tips, x="day", y="total_bill", palette="Set2")
plt.show()

# Paleta para datos cuantitativos
sns.scatterplot(data=tips, x="total_bill", y="tip",
                hue="size", palette="viridis")
plt.show()

# Color continuo
sns.kdeplot(data=tips, x="total_bill", y="tip",
            fill=True, cmap="YlOrRd")
plt.show()
```

---

## 13. Personalización con Matplotlib

```python
tips = sns.load_dataset("tips")

# Seaborn devuelve objetos de Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)

# Personalizar con Matplotlib
ax.set_title("Distribución de Cuentas por Día", fontsize=14, fontweight="bold")
ax.set_xlabel("Día de la Semana", fontsize=12)
ax.set_ylabel("Total de la Cuenta ($)", fontsize=12)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
plt.savefig("grafico_personalizado.png", dpi=150)
plt.show()
```

---

## 14. Ejemplo Completo: Análisis Exploratorio

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos
tips = sns.load_dataset("tips")

# Configurar estilo
sns.set_theme(style="whitegrid", palette="deep")

# Crear figura con múltiples gráficos
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Análisis Exploratorio - Dataset Tips", fontsize=16, fontweight="bold")

# 1. Distribución de cuentas
ax1 = fig.add_subplot(2, 3, 1)
sns.histplot(data=tips, x="total_bill", kde=True, ax=ax1)
ax1.set_title("Distribución de Cuentas")

# 2. Relación cuenta-propina
ax2 = fig.add_subplot(2, 3, 2)
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", ax=ax2)
ax2.set_title("Cuenta vs Propina")

# 3. Box plot por día
ax3 = fig.add_subplot(2, 3, 3)
sns.boxplot(data=tips, x="day", y="total_bill", palette="Set2", ax=ax3)
ax3.set_title("Cuenta por Día")

# 4. Conteo por día y momento
ax4 = fig.add_subplot(2, 3, 4)
sns.countplot(data=tips, x="day", hue="time", ax=ax4)
ax4.set_title("Frecuencia por Día y Momento")
ax4.legend(title="Momento")

# 5. Violin plot fumadores
ax5 = fig.add_subplot(2, 3, 5)
sns.violinplot(data=tips, x="smoker", y="tip", hue="sex", split=True, ax=ax5)
ax5.set_title("Propinas: Fumadores vs No Fumadores")

# 6. Regresión
ax6 = fig.add_subplot(2, 3, 6)
sns.regplot(data=tips, x="total_bill", y="tip", ax=ax6, scatter_kws={"alpha": 0.5})
ax6.set_title("Regresión Lineal")

plt.tight_layout()
plt.savefig("analisis_tips.png", dpi=150, bbox_inches="tight")
plt.show()

# Pair plot separado
sns.pairplot(tips, hue="time", diag_kind="kde", corner=True)
plt.savefig("pairplot_tips.png", dpi=150)
plt.show()

# Heatmap de correlación
plt.figure(figsize=(8, 6))
numeric_tips = tips.select_dtypes(include=[np.number])
sns.heatmap(numeric_tips.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig("correlacion_tips.png", dpi=150)
plt.show()
```

---

## 15. Resumen de Funciones

| Función | Tipo | Descripción |
| :--- | :--- | :--- |
| `sns.histplot()` | Distribución | Histograma |
| `sns.kdeplot()` | Distribución | Densidad |
| `sns.boxplot()` | Categórico | Diagrama de caja |
| `sns.violinplot()` | Categórico | Diagrama de violín |
| `sns.barplot()` | Categórico | Barras con media |
| `sns.countplot()` | Categórico | Conteo |
| `sns.scatterplot()` | Relación | Dispersión |
| `sns.lineplot()` | Relación | Líneas |
| `sns.regplot()` | Regresión | Con línea de regresión |
| `sns.heatmap()` | Matricial | Mapa de calor |
| `sns.pairplot()` | Multivariable | Matriz de dispersión |
| `sns.jointplot()` | Bivariable | Scatter + distribuciones |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
