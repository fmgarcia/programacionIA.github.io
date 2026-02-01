# 🔄 Unidad 5. Autoencoders y Representación Latente

Los **Autoencoders** son redes neuronales que aprenden representaciones comprimidas de los datos de forma no supervisada. Son fundamentales para reducción de dimensionalidad, detección de anomalías y generación de datos.

---

## 5.1. ¿Qué es un Autoencoder?

Un autoencoder es una red que aprende a **codificar** datos en una representación comprimida (espacio latente) y luego **decodificarlos** para reconstruir la entrada original.

![Arquitectura de Autoencoder](../assets/images/autoencoder_arch.svg)

### Arquitectura

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Entrada   │      │   Espacio   │      │   Salida    │
│    (x)      │─────▶│   Latente   │─────▶│    (x')     │
│  (784 dim)  │      │   (32 dim)  │      │  (784 dim)  │
└─────────────┘      └─────────────┘      └─────────────┘
                     
│─── ENCODER ───│    │── LATENTE ──│     │─── DECODER ───│
```

### Objetivo

Minimizar la diferencia entre entrada y salida:

$$\mathcal{L} = ||x - x'||^2 = ||x - \text{Decoder}(\text{Encoder}(x))||^2$$

### Implementación Básica

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# Dimensiones
input_dim = 784  # Por ejemplo, imágenes 28x28
encoding_dim = 32  # Dimensión del espacio latente

# Encoder
encoder = Sequential([
    Input(shape=(input_dim,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(encoding_dim, activation='relu')  # Espacio latente
], name='encoder')

# Decoder
decoder = Sequential([
    Input(shape=(encoding_dim,)),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(input_dim, activation='sigmoid')  # Reconstrucción
], name='decoder')

# Autoencoder completo
autoencoder_input = Input(shape=(input_dim,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(autoencoder_input, decoded, name='autoencoder')

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
```

---

## 5.2. Entrenamiento de Autoencoder

```python
from tensorflow.keras.datasets import mnist
import numpy as np

# Cargar datos
(x_train, _), (x_test, _) = mnist.load_data()

# Preprocesar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Entrenar
# La entrada Y la salida son los mismos datos
historia = autoencoder.fit(
    x_train, x_train,  # Mismo dato como entrada y objetivo
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)
```

### Visualizar Resultados

```python
import matplotlib.pyplot as plt

# Reconstruir imágenes de test
decoded_imgs = autoencoder.predict(x_test)

# Visualizar originales vs reconstruidas
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Reconstruida
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstruida")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5.3. Aplicaciones de Autoencoders

### Reducción de Dimensionalidad

Similar a PCA, pero no lineal:

```python
# Extraer representaciones latentes
encoded_imgs = encoder.predict(x_test)
print(f"Dimensión original: {x_test.shape[1]}")
print(f"Dimensión latente: {encoded_imgs.shape[1]}")

# Visualizar en 2D (si encoding_dim=2)
from sklearn.manifold import TSNE

# Si encoding_dim > 2, usar t-SNE
tsne = TSNE(n_components=2, random_state=42)
encoded_2d = tsne.fit_transform(encoded_imgs[:1000])

plt.figure(figsize=(10, 8))
plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=y_test[:1000], cmap='tab10')
plt.colorbar()
plt.title('Espacio Latente del Autoencoder')
plt.show()
```

### Detección de Anomalías

Las anomalías tienen mayor error de reconstrucción:

```python
def detectar_anomalias(autoencoder, datos, umbral=None):
    """
    Detecta anomalías basándose en el error de reconstrucción.
    """
    reconstrucciones = autoencoder.predict(datos)
    errores = np.mean(np.square(datos - reconstrucciones), axis=1)
    
    if umbral is None:
        # Calcular umbral automáticamente
        umbral = np.mean(errores) + 2 * np.std(errores)
    
    anomalias = errores > umbral
    return anomalias, errores

# Ejemplo con datos normales y anomalías
datos_normales = x_test[:100]
# Crear "anomalías" artificiales (ruido aleatorio)
datos_anomalos = np.random.random((100, 784)).astype('float32')

# Detectar
todos_datos = np.vstack([datos_normales, datos_anomalos])
es_anomalia, errores = detectar_anomalias(autoencoder, todos_datos)

print(f"Anomalías detectadas en datos normales: {es_anomalia[:100].sum()}")
print(f"Anomalías detectadas en datos anómalos: {es_anomalia[100:].sum()}")
```

### Eliminación de Ruido (Denoising)

```python
# Añadir ruido a las imágenes
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Entrenar para reconstruir imágenes limpias desde ruidosas
denoising_autoencoder = autoencoder
denoising_autoencoder.fit(
    x_train_noisy, x_train,  # Entrada ruidosa, objetivo limpio
    epochs=50,
    batch_size=256,
    validation_data=(x_test_noisy, x_test)
)

# Visualizar denoising
denoised_imgs = denoising_autoencoder.predict(x_test_noisy)
```

---

## 5.4. Autoencoder Convolucional

Para imágenes, usar capas convolucionales es más efectivo:

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

# Encoder Convolucional
encoder_conv = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same')
], name='encoder_conv')

# Decoder Convolucional
decoder_conv = Sequential([
    Input(shape=(4, 4, 8)),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same')
], name='decoder_conv')

# Autoencoder completo
input_img = Input(shape=(28, 28, 1))
encoded = encoder_conv(input_img)
decoded = decoder_conv(encoded)
conv_autoencoder = Model(input_img, decoded)

conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

---

## 5.5. Variational Autoencoder (VAE)

Los **VAE** son autoencoders generativos que aprenden una distribución probabilística en el espacio latente.

### Diferencias con Autoencoder Normal

| Autoencoder | VAE |
| :--- | :--- |
| Espacio latente: vectores fijos | Espacio latente: distribución probabilística |
| No puede generar nuevos datos | Puede generar datos muestreando del espacio latente |
| Minimiza error de reconstrucción | Minimiza reconstrucción + KL divergence |

### Espacio Latente del VAE

En lugar de codificar a un vector $z$, codificamos a:

*   $\mu$ (media)
*   $\sigma$ (desviación estándar)

Y muestreamos: $z = \mu + \sigma \cdot \epsilon$, donde $\epsilon \sim \mathcal{N}(0, 1)$

### Función de Pérdida

$$\mathcal{L} = \mathcal{L}_{reconstrucción} + D_{KL}(\mathcal{N}(\mu, \sigma) || \mathcal{N}(0, 1))$$

El término KL fuerza que la distribución latente se parezca a una normal estándar.

### Implementación de VAE

```python
import tensorflow as tf
from tensorflow.keras.layers import Lambda

class Sampling(tf.keras.layers.Layer):
    """Capa de muestreo usando el truco de reparametrización."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Dimensiones
original_dim = 784
intermediate_dim = 256
latent_dim = 2  # Usamos 2 para visualización

# Encoder
encoder_inputs = Input(shape=(original_dim,), name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(encoder_inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
decoder_outputs = Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, decoder_outputs, name='decoder')

# VAE completo
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Pérdida de reconstrucción
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=-1
                )
            )
            
            # KL Divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=-1
                )
            )
            
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Crear y entrenar VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(x_train, epochs=30, batch_size=128)
```

### Visualizar Espacio Latente del VAE

```python
# Codificar datos de test
z_mean, _, _ = vae.encoder.predict(x_test)

# Visualizar espacio latente 2D
plt.figure(figsize=(12, 10))
scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.title('Espacio Latente del VAE')
plt.show()
```

### Generar Nuevos Datos

```python
def generar_digitos(decoder, n=15, digit_size=28, figsize=15):
    """Genera una cuadrícula de dígitos muestreando del espacio latente."""
    # Cuadrícula en el espacio latente
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    figure = np.zeros((digit_size * n, digit_size * n))
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size
            ] = digit
    
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='Greys')
    plt.axis('off')
    plt.title('Dígitos Generados desde el Espacio Latente')
    plt.show()

generar_digitos(vae.decoder)
```

---

## 5.6. Tipos de Autoencoders

| Tipo | Característica | Uso Principal |
| :--- | :--- | :--- |
| **Undercomplete** | Dimensión latente < input | Compresión, reducción dimensionalidad |
| **Overcomplete** | Dimensión latente > input | Necesita regularización extra |
| **Sparse** | Penaliza activaciones densas | Features dispersas |
| **Denoising** | Entrena con ruido | Eliminación de ruido |
| **Contractive** | Penaliza sensibilidad a perturbaciones | Representaciones robustas |
| **Variational (VAE)** | Espacio latente probabilístico | Generación de datos |

### Sparse Autoencoder

```python
from tensorflow.keras import regularizers

# Autoencoder con regularización L1 (sparsity)
sparse_encoder = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu', 
          activity_regularizer=regularizers.l1(1e-5)),  # Sparsity
    Dense(32, activation='relu')
])
```

### Contractive Autoencoder

```python
class ContractiveAutoencoder(Model):
    def __init__(self, encoder, decoder, lambda_contractive=1e-4):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_contractive = lambda_contractive
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            tape.watch(data)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(data)
                encoded = self.encoder(data)
            
            # Jacobiano del encoder
            jacobian = inner_tape.batch_jacobian(encoded, data)
            
            decoded = self.decoder(encoded)
            
            # Pérdida de reconstrucción
            reconstruction_loss = tf.reduce_mean(tf.square(data - decoded))
            
            # Penalización contractiva (norma Frobenius del Jacobiano)
            contractive_loss = tf.reduce_mean(tf.reduce_sum(tf.square(jacobian), axis=[1, 2]))
            
            total_loss = reconstruction_loss + self.lambda_contractive * contractive_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}
```

---

## 5.7. VAE para Generación de Imágenes

```python
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape

# VAE Convolucional para imágenes
latent_dim = 2

# Encoder
encoder_inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = Flatten()(x)
x = Dense(16, activation="relu")(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = Input(shape=(latent_dim,))
x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
```

---

## 5.8. Aplicaciones Avanzadas

### Interpolación en el Espacio Latente

```python
def interpolar(decoder, z1, z2, n_pasos=10):
    """Interpola entre dos puntos en el espacio latente."""
    ratios = np.linspace(0, 1, n_pasos)
    
    plt.figure(figsize=(20, 2))
    for i, ratio in enumerate(ratios):
        z_interp = z1 * (1 - ratio) + z2 * ratio
        z_interp = z_interp.reshape(1, -1)
        img = decoder.predict(z_interp, verbose=0)
        
        plt.subplot(1, n_pasos, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')
    
    plt.suptitle('Interpolación en el Espacio Latente')
    plt.show()

# Ejemplo: interpolar entre dos dígitos
z_mean_test, _, _ = vae.encoder.predict(x_test[:2])
interpolar(vae.decoder, z_mean_test[0], z_mean_test[1])
```

### Aritmética en el Espacio Latente

```python
# Concepto: z(sonrisa) = z(cara_sonriendo) - z(cara_neutral) + z(otra_cara_neutral)
# Resultado: otra_cara debería sonreír

def aritmetica_latente(encoder, decoder, img_a, img_b, img_c):
    """
    Calcula: resultado = z_c + (z_a - z_b)
    Ejemplo: cara_c + (cara_sonriente - cara_neutral)
    """
    z_a, _, _ = encoder.predict(img_a.reshape(1, -1))
    z_b, _, _ = encoder.predict(img_b.reshape(1, -1))
    z_c, _, _ = encoder.predict(img_c.reshape(1, -1))
    
    z_resultado = z_c + (z_a - z_b)
    resultado = decoder.predict(z_resultado)
    
    return resultado
```

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
