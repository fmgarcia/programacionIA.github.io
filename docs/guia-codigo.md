# 📋 Guía de Uso del Código Python

Esta documentación incluye ejemplos de código Python interactivos con funcionalidades mejoradas para facilitar su uso.

## ✨ Funcionalidades de los Bloques de Código

### 1. Botón de Copiar al Portapapeles

Todos los bloques de código Python incluyen un **botón de copiar** en la esquina superior derecha. Al hacer clic:

- ✅ El código se copia automáticamente al portapapeles
- 📋 Puedes pegarlo directamente en tu editor o terminal
- ⚡ No necesitas seleccionar manualmente el texto

### 2. Numeración de Líneas

Los bloques de código más largos incluyen **números de línea** para facilitar la referencia y depuración.

### 3. Resaltado de Sintaxis

El código está **coloreado sintácticamente** para mejorar la legibilidad:

- 🔵 Palabras clave de Python en azul
- 🟢 Strings en verde
- 🟡 Comentarios en gris
- 🔴 Números y valores especiales resaltados

## 💡 Cómo Usar los Ejemplos

### Opción 1: Copiar y Pegar Directamente

1. Haz clic en el botón de copiar (📋) en cualquier bloque de código
2. Abre tu editor favorito (VS Code, PyCharm, Jupyter, etc.)
3. Pega el código con `Ctrl+V` (o `Cmd+V` en Mac)
4. Ejecuta el código

### Opción 2: Guardar como Archivo Python

Para guardar un ejemplo como archivo `.py`:

```python
# Ejemplo de código copiado
import numpy as np
from sklearn.datasets import load_iris

# Tu código aquí...
```

1. Copia el código usando el botón de copiar
2. Crea un nuevo archivo: `ejemplo.py`
3. Pega el contenido
4. Ejecuta: `python ejemplo.py`

### Opción 3: Usar en Jupyter Notebooks

Los ejemplos están diseñados para funcionar directamente en Jupyter:

1. Copia el código
2. Crea una nueva celda en tu notebook
3. Pega y ejecuta con `Shift+Enter`

## 📦 Dependencias Necesarias

La mayoría de ejemplos requieren las siguientes bibliotecas:

```python
# Instalar todas las dependencias necesarias
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Para ejemplos específicos de NLP:

```python
pip install nltk spacy transformers
```

## 🔧 Configuración Recomendada

### Para Mejor Experiencia

1. **Editor de Código**: VS Code con extensión de Python
2. **Entorno Virtual**: 
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Jupyter Lab** (opcional):
   ```bash
   pip install jupyterlab
   jupyter lab
   ```

## 📝 Notas Importantes

- ⚠️ Algunos ejemplos requieren descargar datasets grandes
- 💾 Los ejemplos con visualizaciones pueden requerir entorno gráfico
- 🐍 Se recomienda Python 3.8 o superior
- 📊 Para gráficos en servidores remotos, usa `matplotlib.use('Agg')`

## 🆘 Solución de Problemas Comunes

### Error: ModuleNotFoundError

```python
# Solución: Instalar el módulo faltante
pip install nombre_del_modulo
```

### Error: No module named 'sklearn'

```python
# scikit-learn se instala como sklearn
pip install scikit-learn
```

### Gráficos no se muestran

```python
# Agregar al inicio del código
import matplotlib.pyplot as plt
plt.ion()  # Modo interactivo
```

## 🎯 Consejos para Aprendizaje Efectivo

1. **No solo copies**: Lee y entiende cada línea
2. **Modifica parámetros**: Experimenta cambiando valores
3. **Añade prints**: Imprime variables intermedias para entender el flujo
4. **Usa debugger**: Aprende a usar breakpoints en tu IDE
5. **Documenta**: Añade comentarios explicando lo que hace cada parte

## 📚 Recursos Adicionales

- [Documentación oficial de scikit-learn](https://scikit-learn.org/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Kaggle Learn](https://www.kaggle.com/learn)

---

📅 **Última actualización:** 27/01/2026
✍️ **Autor:** Fran García
