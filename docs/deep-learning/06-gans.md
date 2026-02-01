# 🎭 Unidad 6. Redes Generativas Adversarias (GANs)

Las **Generative Adversarial Networks** (GANs) son un paradigma de aprendizaje donde dos redes neuronales compiten entre sí, permitiendo generar datos sintéticos de alta calidad como imágenes, audio y texto.

---

## 6.1. Concepto de GANs

Introducidas por Ian Goodfellow en 2014, las GANs consisten en dos redes que se entrenan simultáneamente:

![Ciclo GAN](../assets/images/gan_loop.svg)

### Componentes

*   **Generador (G):** Genera datos falsos a partir de ruido aleatorio.
*   **Discriminador (D):** Distingue entre datos reales y falsos.

### Analogía

```
Generador (Falsificador)           Discriminador (Detective)
        │                                    │
        │ Crea billetes falsos               │ Detecta billetes falsos
        │                                    │
        ▼                                    ▼
   ┌─────────┐                         ┌─────────┐
   │ Ruido z │──▶ G(z) ──▶ Imagen ──▶  │   D     │──▶ Real/Falso
   └─────────┘     falsa               └─────────┘
                                             ▲
                                             │
                                       Imagen real
```

### Objetivo

*   **G quiere:** Engañar a D (que D piense que sus imágenes son reales).
*   **D quiere:** Distinguir correctamente entre real y falso.

Este "juego" lleva a que G genere datos cada vez más realistas.

---

## 6.2. Formulación Matemática

### Función de Pérdida

El entrenamiento de GANs es un juego minimax:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

Donde:

*   $x$ = datos reales
*   $z$ = ruido aleatorio (vector latente)
*   $G(z)$ = dato generado
*   $D(x)$ = probabilidad de que $x$ sea real

### Interpretación

*   **D maximiza:** $\log D(x)$ (clasificar real como real) + $\log(1-D(G(z)))$ (clasificar falso como falso).
*   **G minimiza:** $\log(1-D(G(z)))$ equivale a maximizar $\log D(G(z))$ (engañar a D).

---

## 6.3. Implementación Básica de una GAN

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Hiperparámetros
latent_dim = 100  # Dimensión del ruido de entrada
img_shape = (28, 28, 1)
img_dim = 28 * 28

# =====================
# GENERADOR
# =====================
def build_generator():
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        
        Dense(img_dim, activation='tanh'),
        Reshape(img_shape)
    ], name='generator')
    return model

# =====================
# DISCRIMINADOR
# =====================
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=img_shape),
        
        Dense(512),
        LeakyReLU(alpha=0.2),
        
        Dense(256),
        LeakyReLU(alpha=0.2),
        
        Dense(1, activation='sigmoid')  # Probabilidad de ser real
    ], name='discriminator')
    return model

# Crear modelos
generator = build_generator()
discriminator = build_discriminator()

# Compilar discriminador
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
    metrics=['accuracy']
)

generator.summary()
discriminator.summary()
```

### Crear el Modelo GAN Combinado

```python
# Para entrenar el generador, congelamos el discriminador
discriminator.trainable = False

# Modelo combinado: Generador + Discriminador
gan_input = tf.keras.Input(shape=(latent_dim,))
generated_img = generator(gan_input)
validity = discriminator(generated_img)

gan = tf.keras.Model(gan_input, validity, name='gan')
gan.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
)
```

---

## 6.4. Entrenamiento de la GAN

```python
def train_gan(epochs, batch_size=128, sample_interval=1000):
    # Cargar datos
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # Preprocesar: normalizar a [-1, 1] (para tanh)
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
    
    # Etiquetas
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        # =====================
        # Entrenar Discriminador
        # =====================
        
        # Seleccionar batch aleatorio de imágenes reales
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        
        # Generar imágenes falsas
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)
        
        # Entrenar discriminador
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # =====================
        # Entrenar Generador
        # =====================
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Entrenar generador (queremos que D clasifique las falsas como reales)
        g_loss = gan.train_on_batch(noise, real)
        
        # Guardar pérdidas
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        # Imprimir progreso
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.1f}% - G loss: {g_loss:.4f}")
        
        # Guardar imágenes de muestra
        if epoch % sample_interval == 0:
            sample_images(epoch)
    
    return d_losses, g_losses

def sample_images(epoch, n=5):
    """Genera y guarda imágenes de muestra."""
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)
    
    # Reescalar a [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    cnt = 0
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axes[i, j].axis('off')
            cnt += 1
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(f'gan_images_epoch_{epoch}.png')
    plt.close()

# Entrenar
d_losses, g_losses = train_gan(epochs=30000, batch_size=64, sample_interval=2000)
```

---

## 6.5. Deep Convolutional GAN (DCGAN)

Las **DCGAN** usan capas convolucionales para generar imágenes de mayor calidad.

### Principios de Arquitectura DCGAN

1.  Usar convoluciones transpuestas en el generador.
2.  Usar BatchNormalization en ambas redes.
3.  Usar LeakyReLU en el discriminador.
4.  Usar ReLU en el generador (excepto salida con tanh).

```python
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout

def build_dcgan_generator(latent_dim):
    model = Sequential([
        # Entrada: vector de ruido
        Dense(7 * 7 * 256, input_dim=latent_dim),
        Reshape((7, 7, 256)),
        
        # Upsample a 14x14
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        
        # Upsample a 28x28
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        
        # Salida: imagen 28x28x1
        Conv2D(1, (7, 7), padding='same', activation='tanh')
    ], name='dcgan_generator')
    return model

def build_dcgan_discriminator():
    model = Sequential([
        # Entrada: imagen 28x28x1
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        BatchNormalization(),
        
        Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        BatchNormalization(),
        
        Flatten(),
        Dense(1, activation='sigmoid')
    ], name='dcgan_discriminator')
    return model
```

---

## 6.6. Problemas Comunes y Soluciones

### Mode Collapse

El generador produce solo unos pocos tipos de salidas.

**Soluciones:**

*   Mini-batch discrimination.
*   Unrolled GANs.
*   Wasserstein GAN (WGAN).

### Entrenamiento Inestable

El discriminador o generador dominan.

**Soluciones:**

*   Balancear learning rates.
*   Label smoothing.
*   Spectral normalization.

### Vanishing Gradients

El discriminador se vuelve muy bueno y el generador no recibe gradientes útiles.

**Soluciones:**

*   Usar pérdida de Wasserstein.
*   Feature matching.

---

## 6.7. Wasserstein GAN (WGAN)

WGAN usa la distancia de Wasserstein para una métrica de entrenamiento más estable.

### Cambios Principales

1.  **Pérdida:** Distancia de Wasserstein en lugar de binary crossentropy.
2.  **Discriminador → Crítico:** No produce probabilidad, sino un score.
3.  **Clipping de pesos:** Los pesos del crítico se limitan a [-c, c].

```python
from tensorflow.keras import backend as K

def wasserstein_loss(y_true, y_pred):
    """Pérdida de Wasserstein."""
    return K.mean(y_true * y_pred)

def build_critic():
    """El crítico de WGAN (no usa sigmoid en la salida)."""
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1)  # Sin activación sigmoid
    ])
    return model

# Compilar con pérdida Wasserstein
critic = build_critic()
critic.compile(
    loss=wasserstein_loss,
    optimizer=tf.keras.optimizers.RMSprop(lr=0.00005)
)

def train_wgan(epochs, batch_size=64, n_critic=5, clip_value=0.01):
    """
    Entrenar WGAN.
    n_critic: número de veces que se entrena el crítico por cada vez del generador.
    """
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)
    
    real = np.ones((batch_size, 1))
    fake = -np.ones((batch_size, 1))  # -1 para falsas en WGAN
    
    for epoch in range(epochs):
        # Entrenar crítico n_critic veces
        for _ in range(n_critic):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict(noise, verbose=0)
            
            critic.train_on_batch(real_imgs, real)
            critic.train_on_batch(gen_imgs, fake)
            
            # Clipping de pesos
            for layer in critic.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(weights)
        
        # Entrenar generador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gan.train_on_batch(noise, real)
```

---

## 6.8. Conditional GAN (cGAN)

Las **cGAN** permiten condicionar la generación en una etiqueta o clase.

```
    ┌───────┐     ┌───────────┐
    │ Ruido │────▶│           │
    │   z   │     │ Generador │──▶ Imagen de dígito "7"
    │       │     │     G     │
    └───────┘     │           │
    ┌───────┐     │           │
    │ Label │────▶│           │
    │  "7"  │     └───────────┘
    └───────┘
```

### Implementación

```python
from tensorflow.keras.layers import Embedding, Concatenate, Input
from tensorflow.keras import Model

num_classes = 10  # Dígitos 0-9

def build_cgan_generator(latent_dim, num_classes):
    # Entrada de ruido
    noise_input = Input(shape=(latent_dim,))
    
    # Entrada de etiqueta (embedding)
    label_input = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, latent_dim)(label_input)
    label_embedding = Flatten()(label_embedding)
    
    # Concatenar ruido y etiqueta
    merged = Concatenate()([noise_input, label_embedding])
    
    # Generador
    x = Dense(256)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(784, activation='tanh')(x)
    output = Reshape((28, 28, 1))(x)
    
    return Model([noise_input, label_input], output, name='cgan_generator')

def build_cgan_discriminator(num_classes):
    # Entrada de imagen
    img_input = Input(shape=(28, 28, 1))
    img_flat = Flatten()(img_input)
    
    # Entrada de etiqueta
    label_input = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, 784)(label_input)
    label_embedding = Flatten()(label_embedding)
    
    # Concatenar imagen y etiqueta
    merged = Concatenate()([img_flat, label_embedding])
    
    x = Dense(512)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model([img_input, label_input], output, name='cgan_discriminator')

# Crear modelos
cgan_generator = build_cgan_generator(latent_dim, num_classes)
cgan_discriminator = build_cgan_discriminator(num_classes)

# Generar dígitos específicos
def generate_digit(generator, digit, n=10):
    """Genera n imágenes del dígito especificado."""
    noise = np.random.normal(0, 1, (n, latent_dim))
    labels = np.full((n, 1), digit)
    
    gen_imgs = generator.predict([noise, labels], verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axes = plt.subplots(1, n, figsize=(20, 2))
    for i in range(n):
        axes[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Dígito generado: {digit}')
    plt.show()

# Ejemplo: generar varios "7"
generate_digit(cgan_generator, digit=7)
```

---

## 6.9. Otras Variantes de GANs

| Variante | Característica | Aplicación |
| :--- | :--- | :--- |
| **DCGAN** | Usa convolutions | Imágenes de mayor calidad |
| **WGAN** | Pérdida de Wasserstein | Entrenamiento más estable |
| **cGAN** | Condicionado en etiquetas | Generación controlada |
| **Pix2Pix** | Image-to-image translation | Convertir bocetos a imágenes |
| **CycleGAN** | Traducción sin pares | Convertir fotos a estilo artístico |
| **StyleGAN** | Control de estilo por capas | Caras realistas de alta resolución |
| **ProGAN** | Entrenamiento progresivo | Imágenes de muy alta resolución |

---

## 6.10. Aplicaciones de GANs

### Generación de Imágenes Realistas

```python
# StyleGAN2 con TensorFlow Hub
import tensorflow_hub as hub

# Cargar modelo preentrenado
stylegan = hub.load('https://tfhub.dev/google/progan-128/1')

# Generar imágenes
latent = tf.random.normal([1, 512])
images = stylegan(latent)['default']
```

### Super-Resolución (SRGAN)

```python
# Aumentar resolución de imágenes
# Low-res (64x64) → High-res (256x256)
```

### Transferencia de Estilo

```python
# CycleGAN: Foto → Pintura de Monet
# Sin necesidad de pares de entrenamiento
```

### Data Augmentation

```python
# Generar datos sintéticos para entrenar otros modelos
# Útil cuando hay pocos datos reales
```

---

## 6.11. Métricas de Evaluación

### Inception Score (IS)

Mide calidad y diversidad:

$$IS = \exp(\mathbb{E}_x[D_{KL}(p(y|x) || p(y))])$$

```python
# Calcular Inception Score (simplificado)
from tensorflow.keras.applications.inception_v3 import InceptionV3

def inception_score(images, n_split=10, eps=1e-16):
    inception = InceptionV3(include_top=False, pooling='avg')
    # ... cálculo completo
    pass
```

### Fréchet Inception Distance (FID)

Compara distribución de características:

$$FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$

**Menor FID = mejor calidad y diversidad.**

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
