# 🤖 Unidad 5. Algoritmo de Bayes y Naive Bayes en Inteligencia Artificial

El algoritmo de **Bayes**, también conocido como **Teorema de Bayes**, es un enfoque probabilístico utilizado para la clasificación y el análisis en inteligencia artificial y aprendizaje automático. Este algoritmo se basa en la probabilidad condicional, lo cual permite actualizar la probabilidad de un evento en función de nueva evidencia.

El algoritmo **Naive Bayes** simplifica el Teorema de Bayes haciendo una suposición fundamental: que todas las características (o atributos) son independientes entre sí. Esta simplificación permite construir modelos de clasificación rápidos y eficientes, especialmente útiles en aplicaciones de clasificación de texto, como la clasificación de correos electrónicos o el análisis de sentimientos.

A continuación, veremos en detalle cómo se deriva el modelo Naive Bayes a partir del Teorema de Bayes y cómo funciona, junto con ejemplos y las fórmulas matemáticas correspondientes.

#### 1. **Teorema de Bayes**

El **Teorema de Bayes** describe la probabilidad de que ocurra un evento \( A \) dado que ya ha ocurrido otro evento \( B \). La fórmula se expresa de la siguiente manera:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$


Donde:
- \( P(A|B) \): Probabilidad de que ocurra el evento \( A \) dado que \( B \) ha ocurrido (probabilidad posterior).
- \( P(B|A) \): Probabilidad de que ocurra el evento \( B \) dado que \( A \) ha ocurrido (verosimilitud).
- \( P(A) \): Probabilidad a priori del evento \( A \).
- \( P(B) \): Probabilidad del evento \( B \).

El Teorema de Bayes permite actualizar la probabilidad a priori de un evento a partir de nueva información (evidencia).

#### 2. **Naive Bayes**

El clasificador **Naive Bayes** se deriva del Teorema de Bayes con la suposición de independencia entre características. En lugar de considerar todas las relaciones posibles entre los atributos, se asume que cada característica es independiente de las demás, dado el resultado. Esto simplifica el cálculo de la probabilidad conjunta.

La probabilidad de que un ejemplo 
$$ 
 x = (x_1, x_2, \dots, x_n)
$$ 
pertenezca a una clase \( C_k \) se puede calcular como:

$$
P(C_k | x) = \frac{P(C_k) \prod_{i=1}^n P(x_i | C_k)}{P(x)}
$$

Dado que \( P(x) \) es constante para todas las clases, podemos simplificar la fórmula a:

$$
P(C_k | x) \propto P(C_k) \prod_{i=1}^n P(x_i | C_k)
$$

Donde:
- P(C_k | x): Probabilidad posterior de que el ejemplo pertenezca a la clase C_k .
- P(C_k): Probabilidad a priori de la clase C_k.
- P(x_i | C_k): Probabilidad condicional de la característica x_i dada la clase C_k.

El clasificador Naive Bayes elige la clase que maximiza esta probabilidad posterior.

#### 3. **Tipos de Clasificadores Naive Bayes**

Existen diferentes tipos de clasificadores Naive Bayes, dependiendo del tipo de datos y de cómo se calcula la probabilidad condicional:

- **Naive Bayes Gaussiano**: Se utiliza cuando las características tienen una distribución continua que se puede aproximar a una distribución normal (gaussiana).
- **Naive Bayes Multinomial**: Es adecuado para datos discretos, como el conteo de palabras en un documento. Es ampliamente utilizado en clasificación de texto.
- **Naive Bayes Bernoulli**: Se utiliza para características binarias. Es útil cuando cada característica es booleana (por ejemplo, si una palabra aparece o no en un documento).

#### 4. **Ejemplo Completo de Naive Bayes**

Vamos a clasificar correos electrónicos como **"spam"** o **"no spam"** usando el algoritmo de Naive Bayes. Para este ejemplo, supongamos que tenemos los siguientes datos de entrenamiento, con algunas palabras clave y la clase correspondiente ("spam" o "no spam"):

| Correo ID | Contenido                     | Clase   |
|-----------|-------------------------------|---------|
| 1         | Oferta barata, gana dinero    | Spam    |
| 2         | Proyecto pendiente de trabajo | No Spam |
| 3         | Oferta especial gratis        | Spam    |
| 4         | Reunión de equipo mañana      | No Spam |
| 5         | Gana premios y dinero ahora   | Spam    |
| 6         | Informe mensual adjunto       | No Spam |

Vamos a suponer que queremos clasificar un nuevo correo con el contenido: **"Oferta gratis y premios"**. Para esto, usaremos el clasificador de Naive Bayes, asumiendo que todas las palabras son independientes (el supuesto "naive").

### Paso 1: Calcular las Probabilidades Previas

Primero calculamos la probabilidad previa de cada clase.

- **Probabilidad de Spam** (π(spam)):

  $$
  P(Spam) = \frac{N_{Spam}}{N_{Total}} = \frac{3}{6} = 0.5
  $$

- **Probabilidad de No Spam** (π(no\_spam)):

  $$
  P(No\,Spam) = \frac{N_{No\,Spam}}{N_{Total}} = \frac{3}{6} = 0.5
  $$

### Paso 2: Calcular la Probabilidad de Cada Palabra

Ahora, necesitamos calcular la probabilidad de cada palabra en el contexto de cada clase (es decir, "spam" y "no spam"). Las palabras únicas en nuestro conjunto de entrenamiento son:

- "oferta", "barata", "gana", "dinero", "proyecto", "pendiente", "trabajo", "especial", "gratis", "reunión", "equipo", "mañana", "premios", "ahora", "informe", "mensual", "adjunto".

Vamos a usar suavizado de Laplace (adicionando 1 a cada recuento) para evitar probabilidades de cero.

Por ejemplo, calculamos la probabilidad de cada palabra para **spam**:

- **P(oferta | Spam)**:

  La palabra "oferta" aparece en 2 de los 3 correos spam.

  $$
  P(oferta \mid Spam) = \frac{2 + 1}{N_{Spam} + V} = \frac{2 + 1}{3 + 17} = \frac{3}{20} = 0.15
  $$

- **P(gratis | Spam)** y **P(premios | Spam)** también se calculan de manera similar.

Donde:
- $N_{Spam}$ : Número de correos spam.
- V : Número de palabras únicas en el vocabulario.

### Paso 3: Clasificar el Nuevo Correo

El nuevo correo es: **"Oferta gratis y premios"**. Queremos calcular la probabilidad de que sea spam o no spam.

Para calcular esto, usamos la fórmula de Naive Bayes:

$$
P(Clase \mid X) \propto P(X \mid Clase) \cdot P(Clase)
$$

Primero calculamos la probabilidad de que el correo sea **spam**:

$$
P(Spam \mid X) \propto P(oferta \mid Spam) \cdot P(gratis \mid Spam) \cdot P(premios \mid Spam) \cdot P(Spam)
$$

Sustituimos los valores y multiplicamos:

$$
P(Spam \mid X) \propto 0.15 \times 0.15 \times 0.1 \times 0.5
$$

Hacemos el mismo cálculo para **No Spam** y comparamos ambas probabilidades.

### Paso 4: Decisión

Finalmente, elegimos la clase que tiene la probabilidad mayor. Si $ P(Spam \mid X) $ es mayor que $ P(No\,Spam \mid X) $, clasificamos el correo como **spam**; de lo contrario, como **no spam**.

#### 5. **Aplicaciones Reales de Naive Bayes**

Debido a su eficiencia y capacidad para manejar grandes volúmenes de datos, Naive Bayes se utiliza en una amplia variedad de aplicaciones del mundo real:

*   **Filtrado de Spam:** Es el uso más clásico. Servicios como Gmail o Outlook utilizan variantes de Naive Bayes para clasificar correos entrantes como deseados o no deseados basándose en la frecuencia de ciertas palabras.
    *   [Ejemplo de implementación de filtro Spam](https://github.com/topics/spam-filter)
*   **Análisis de Sentimientos:** Determinar si una opinión en redes sociales (Twitter, reseñas de productos) es positiva, negativa o neutral. Es muy usado en marketing para monitorear la reputación de marca.
    *   [Análisis de sentimientos en Twitter](https://github.com/topics/sentiment-analysis)
*   **Clasificación de Documentos:** Organizar noticias en categorías (Deportes, Política, Tecnología) o clasificar documentos legales y médicos.
*   **Sistemas de Recomendación:** Filtrado colaborativo para predecir si a un usuario le gustará un recurso dado.

#### 6. **Ventajas y Limitaciones de Naive Bayes**

- **Ventajas**:
  - **Simplicidad y rapidez**: Naive Bayes es simple de implementar y muy rápido para entrenar y hacer predicciones, incluso con grandes volúmenes de datos.
  - **Escalabilidad**: Funciona bien con datos de alta dimensionalidad, como texto.
  - **Robustez frente al ruido**: A pesar de la suposición de independencia, suele funcionar sorprendentemente bien en muchos problemas reales.

- **Limitaciones**:
  - **Suposición de independencia**: La suposición de independencia rara vez se cumple en problemas reales. Esto puede llevar a predicciones menos precisas cuando las características están altamente correlacionadas.
  - **Problemas con datos cero**: Si una característica no se presenta en los datos de entrenamiento, la probabilidad condicional se convierte en cero, lo que hace que la probabilidad posterior también sea cero. Esto se suele solucionar con la suavización de Laplace.

El clasificador **Naive Bayes** sigue siendo una herramienta poderosa para muchas aplicaciones de inteligencia artificial y aprendizaje automático, especialmente en la clasificación de texto y otros problemas donde la independencia de las características no afecta significativamente el rendimiento del modelo.