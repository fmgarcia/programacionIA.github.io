# 🎭 Faker - Generación de Datos Ficticios

**Faker** es una librería de Python que permite generar datos falsos de manera realista. Es extremadamente útil para pruebas, desarrollo, poblar bases de datos y crear datasets de ejemplo.

---

## 1. Instalación e Importación

```python
# Instalación
# pip install faker

# Importación
from faker import Faker

# Crear instancia (por defecto en inglés)
fake = Faker()

# Crear instancia en español
fake = Faker('es_ES')

# Múltiples idiomas
fake = Faker(['es_ES', 'en_US', 'fr_FR', 'ja_JP'])
```

---

## 2. Datos Personales Básicos

### Nombres

```python
from faker import Faker

fake = Faker('es_ES')

# Nombre completo
print(fake.name())              # María García López

# Nombre de pila
print(fake.first_name())        # Carlos

# Apellido
print(fake.last_name())         # Fernández

# Nombre masculino
print(fake.first_name_male())   # Antonio

# Nombre femenino
print(fake.first_name_female()) # Laura

# Prefijo (Sr., Sra., etc.)
print(fake.prefix())            # Sra.

# Sufijo
print(fake.suffix())            # Jr.
```

### Generar Múltiples Nombres

```python
from faker import Faker

fake = Faker('es_ES')

# Generar 5 nombres
for _ in range(5):
    print(fake.name())

# Salida:
# José Martín Ruiz
# Ana María López García
# Pedro Sánchez Torres
# Carmen Rodríguez Pérez
# Francisco Gómez Díaz
```

---

## 3. Direcciones

```python
from faker import Faker

fake = Faker('es_ES')

# Dirección completa
print(fake.address())
# Ronda de Ana María López 45
# Cuenca, 04267

# Componentes individuales
print(fake.street_name())       # Calle de San Pedro
print(fake.street_address())    # Paseo de García 78
print(fake.city())              # Sevilla
print(fake.state())             # Madrid
print(fake.postcode())          # 28001
print(fake.country())           # España

# Coordenadas geográficas
print(fake.latitude())          # 40.4168
print(fake.longitude())         # -3.7038
print(fake.coordinate())        # (40.4168, -3.7038)
```

---

## 4. Contacto

### Email y Teléfono

```python
from faker import Faker

fake = Faker('es_ES')

# Email
print(fake.email())                    # juan.garcia@example.com
print(fake.free_email())               # maria_lopez@gmail.com
print(fake.company_email())            # ana.martinez@empresa.es
print(fake.safe_email())               # pedro@example.org

# Email personalizado
print(fake.ascii_email())              # carlos.sanchez@example.net

# Teléfono
print(fake.phone_number())             # +34 612 345 678
print(fake.msisdn())                   # 34612345678
```

### Internet

```python
from faker import Faker

fake = Faker()

# Usuario
print(fake.user_name())         # juan_garcia
print(fake.password())          # aB3$kL9mN

# Password personalizado
print(fake.password(
    length=12,
    special_chars=True,
    digits=True,
    upper_case=True,
    lower_case=True
))  # Xk9$mN2pL@qR

# URLs y dominios
print(fake.url())               # https://www.example.com/
print(fake.domain_name())       # garcia.com
print(fake.domain_word())       # martinez
print(fake.tld())               # es

# IP
print(fake.ipv4())              # 192.168.1.100
print(fake.ipv6())              # 2001:0db8:85a3:0000:0000:8a2e:0370:7334
print(fake.mac_address())       # 00:1A:2B:3C:4D:5E

# User Agent
print(fake.user_agent())        # Mozilla/5.0 (Windows NT 10.0; Win64; x64)...
```

---

## 5. Fechas y Tiempos

```python
from faker import Faker

fake = Faker('es_ES')

# Fecha
print(fake.date())                      # 2023-05-15
print(fake.date_this_year())            # Fecha de este año
print(fake.date_this_month())           # Fecha de este mes
print(fake.date_this_decade())          # Fecha de esta década

# Fecha en rango
print(fake.date_between(
    start_date='-30y',
    end_date='today'
))  # Fecha entre hace 30 años y hoy

# Fecha de nacimiento
print(fake.date_of_birth(
    minimum_age=18,
    maximum_age=65
))  # 1975-03-22

# Hora
print(fake.time())                      # 14:35:22

# Fecha y hora completa
print(fake.date_time())                 # 2023-05-15 14:35:22
print(fake.date_time_this_year())

# Timestamp Unix
print(fake.unix_time())                 # 1684159522

# Día y mes
print(fake.day_of_week())               # Miércoles
print(fake.month_name())                # Mayo
print(fake.year())                      # 2023

# Zona horaria
print(fake.timezone())                  # Europe/Madrid
```

---

## 6. Texto

```python
from faker import Faker

fake = Faker('es_ES')

# Palabra
print(fake.word())                      # casa

# Palabras
print(fake.words(nb=5))                 # ['casa', 'perro', 'mesa', 'libro', 'agua']

# Frase
print(fake.sentence())                  # El rápido zorro marrón salta.
print(fake.sentence(nb_words=10))       # Frase de ~10 palabras

# Párrafo
print(fake.paragraph())                 # Párrafo con varias frases
print(fake.paragraph(nb_sentences=5))   # Párrafo con 5 frases

# Múltiples párrafos
print(fake.paragraphs(nb=3))            # Lista de 3 párrafos

# Texto largo
print(fake.text())                      # Texto de ~200 caracteres
print(fake.text(max_nb_chars=500))      # Texto de hasta 500 caracteres

# Lorem Ipsum
print(fake.catch_phrase())              # Frase promocional
print(fake.bs())                        # Jerga empresarial
```

---

## 7. Números y Datos Financieros

### Números

```python
from faker import Faker

fake = Faker()

# Enteros
print(fake.random_int(min=1, max=100))      # 42
print(fake.random_digit())                   # 7
print(fake.random_number(digits=5))          # 45678

# Decimales
print(fake.pyfloat(min_value=0, max_value=100, right_digits=2))  # 45.67
print(fake.pydecimal(left_digits=3, right_digits=2))             # 123.45

# Booleano
print(fake.boolean())                        # True
print(fake.boolean(chance_of_getting_true=75))  # 75% probabilidad de True
```

### Datos Financieros

```python
from faker import Faker

fake = Faker('es_ES')

# Tarjeta de crédito
print(fake.credit_card_number())        # 4532015112830366
print(fake.credit_card_provider())      # Visa
print(fake.credit_card_expire())        # 03/25
print(fake.credit_card_security_code()) # 123
print(fake.credit_card_full())          # Información completa

# IBAN
print(fake.iban())                      # ES9121000418450200051332

# Moneda
print(fake.currency())                  # ('EUR', 'Euro')
print(fake.currency_code())             # EUR
print(fake.currency_name())             # Euro
print(fake.pricetag())                  # 45,99 €
```

---

## 8. Empresas y Trabajo

```python
from faker import Faker

fake = Faker('es_ES')

# Empresa
print(fake.company())                   # García e Hijos S.L.
print(fake.company_suffix())            # S.A.
print(fake.catch_phrase())              # Soluciones innovadoras para el futuro

# Trabajo
print(fake.job())                       # Ingeniero de Software

# NIF/CIF (España)
print(fake.nif())                       # 12345678Z
print(fake.nie())                       # X1234567L
```

---

## 9. Perfiles Completos

```python
from faker import Faker

fake = Faker('es_ES')

# Perfil simple
perfil = fake.simple_profile()
print(perfil)
# {
#     'username': 'maria_garcia',
#     'name': 'María García López',
#     'sex': 'F',
#     'address': 'Calle Mayor 45, Madrid 28001',
#     'mail': 'maria.garcia@example.com',
#     'birthdate': datetime.date(1985, 3, 15)
# }

# Acceder a campos individuales
print(perfil['name'])
print(perfil['mail'])
print(perfil['birthdate'].year)

# Perfil completo
perfil_completo = fake.profile()
print(perfil_completo)
# Incluye: job, company, ssn, residence, blood_group, website...
```

---

## 10. Colores

```python
from faker import Faker

fake = Faker()

print(fake.color_name())        # Azul
print(fake.hex_color())         # #3498db
print(fake.rgb_color())         # 52,152,219
print(fake.rgb_css_color())     # rgb(52,152,219)
print(fake.safe_color_name())   # blue
print(fake.safe_hex_color())    # #0000ff
```

---

## 11. Archivos y Rutas

```python
from faker import Faker

fake = Faker()

# Nombres de archivo
print(fake.file_name())                 # documento.pdf
print(fake.file_name(extension='xlsx')) # datos.xlsx
print(fake.file_extension())            # pdf

# Tipos MIME
print(fake.mime_type())                 # application/pdf

# Rutas
print(fake.file_path())                 # /home/user/docs/file.txt
print(fake.file_path(depth=3))          # Ruta con 3 niveles

# UUID
print(fake.uuid4())                     # a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

## 12. Datos Científicos

```python
from faker import Faker

fake = Faker()

# Python
print(fake.pylist(nb_elements=5))       # Lista de 5 elementos aleatorios
print(fake.pydict(nb_elements=3))       # Diccionario de 3 elementos
print(fake.pytuple(nb_elements=4))      # Tupla de 4 elementos
print(fake.pyset(nb_elements=5))        # Set de 5 elementos

# Estructuras específicas
print(fake.pylist(
    nb_elements=3,
    variable_nb_elements=False,
    value_types=[int]
))  # [42, 17, 89]

print(fake.pydict(
    nb_elements=2,
    value_types=[str]
))  # {'key1': 'valor1', 'key2': 'valor2'}
```

---

## 13. Reproducibilidad con Semilla

```python
from faker import Faker

# Establecer semilla para resultados reproducibles
Faker.seed(12345)
fake = Faker('es_ES')

# Siempre generará los mismos datos
print(fake.name())  # Siempre el mismo nombre
print(fake.email()) # Siempre el mismo email

# También puedes usar seed por instancia
fake1 = Faker('es_ES')
fake1.seed_instance(42)

fake2 = Faker('es_ES')
fake2.seed_instance(42)

print(fake1.name() == fake2.name())  # True
```

---

## 14. Múltiples Idiomas

```python
from faker import Faker

# Instancia multiidioma
fake = Faker(['es_ES', 'en_US', 'fr_FR', 'de_DE', 'ja_JP'])

# Genera datos aleatorios de cualquier idioma
for _ in range(5):
    print(fake.name())

# Salida:
# María García (español)
# John Smith (inglés)
# Pierre Dubois (francés)
# Hans Müller (alemán)
# 田中太郎 (japonés)

# Acceder a un idioma específico
fake_es = Faker('es_ES')
fake_en = Faker('en_US')

print(fake_es.name())  # Nombre español
print(fake_en.name())  # Nombre inglés
```

### Idiomas Disponibles (algunos ejemplos)

| Código | Idioma |
| :--- | :--- |
| `es_ES` | Español (España) |
| `es_MX` | Español (México) |
| `en_US` | Inglés (EEUU) |
| `en_GB` | Inglés (Reino Unido) |
| `fr_FR` | Francés |
| `de_DE` | Alemán |
| `it_IT` | Italiano |
| `pt_BR` | Portugués (Brasil) |
| `ja_JP` | Japonés |
| `zh_CN` | Chino |
| `ru_RU` | Ruso |

---

## 15. Generar CSV con Faker

### Ejemplo: Usuarios de Red Social

```python
from faker import Faker
import csv

# Configuración
faker = Faker(['es_ES', 'en_US', 'fr_FR'])
cantidad_usuarios = 1000
ruta_fichero = 'usuarios_facebook.csv'
campos = ["id", "nombre", "apellidos", "correo", "password", 
          "dia", "mes", "anyo", "genero"]

# Generar CSV
with open(ruta_fichero, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=campos)
    writer.writeheader()
    
    for i in range(1, cantidad_usuarios + 1):
        perfil = faker.simple_profile()
        nombre_completo = perfil['name'].split()
        
        writer.writerow({
            "id": i,
            "nombre": nombre_completo[0],
            "apellidos": ' '.join(nombre_completo[1:]) if len(nombre_completo) > 1 else faker.last_name(),
            "correo": perfil['mail'],
            "password": faker.password(length=10, special_chars=True, digits=True),
            "dia": perfil['birthdate'].day,
            "mes": perfil['birthdate'].month,
            "anyo": perfil['birthdate'].year,
            "genero": perfil['sex']
        })

print(f"Archivo '{ruta_fichero}' creado con {cantidad_usuarios} usuarios.")
```

### Ejemplo: Dataset de Empleados

```python
from faker import Faker
import csv

fake = Faker('es_ES')

campos = ['id', 'nombre', 'apellidos', 'email', 'departamento', 
          'cargo', 'salario', 'fecha_contratacion', 'telefono']

departamentos = ['Ventas', 'Marketing', 'IT', 'RRHH', 'Finanzas', 'Operaciones']

with open('empleados.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    
    for i in range(1, 501):
        writer.writerow({
            'id': i,
            'nombre': fake.first_name(),
            'apellidos': fake.last_name(),
            'email': fake.company_email(),
            'departamento': fake.random_element(departamentos),
            'cargo': fake.job(),
            'salario': fake.random_int(min=20000, max=80000),
            'fecha_contratacion': fake.date_between(start_date='-10y', end_date='today'),
            'telefono': fake.phone_number()
        })

print("Archivo 'empleados.csv' creado con 500 empleados.")
```

---

## 16. Generar JSON con Faker

```python
from faker import Faker
import json

fake = Faker('es_ES')

# Generar lista de usuarios
usuarios = []
for i in range(100):
    usuario = {
        "id": i + 1,
        "nombre": fake.name(),
        "email": fake.email(),
        "telefono": fake.phone_number(),
        "direccion": {
            "calle": fake.street_address(),
            "ciudad": fake.city(),
            "codigo_postal": fake.postcode(),
            "pais": fake.country()
        },
        "fecha_registro": str(fake.date_this_year()),
        "activo": fake.boolean(chance_of_getting_true=80)
    }
    usuarios.append(usuario)

# Guardar en archivo JSON
with open('usuarios.json', 'w', encoding='utf-8') as f:
    json.dump(usuarios, f, indent=4, ensure_ascii=False)

print(f"Archivo 'usuarios.json' creado con {len(usuarios)} usuarios.")
```

---

## 17. Uso con Pandas

```python
from faker import Faker
import pandas as pd

fake = Faker('es_ES')

# Crear DataFrame directamente
n = 1000

df = pd.DataFrame({
    'id': range(1, n + 1),
    'nombre': [fake.name() for _ in range(n)],
    'email': [fake.email() for _ in range(n)],
    'ciudad': [fake.city() for _ in range(n)],
    'fecha_nacimiento': [fake.date_of_birth(minimum_age=18, maximum_age=70) for _ in range(n)],
    'salario': [fake.random_int(min=18000, max=100000) for _ in range(n)],
    'departamento': [fake.random_element(['IT', 'Ventas', 'Marketing', 'RRHH']) for _ in range(n)]
})

print(df.head())
print(f"\nEstadísticas de salario:")
print(df['salario'].describe())

# Guardar en diferentes formatos
df.to_csv('datos_faker.csv', index=False)
df.to_excel('datos_faker.xlsx', index=False)
df.to_json('datos_faker.json', orient='records', indent=2)
```

---

## 18. Proveedores Personalizados

```python
from faker import Faker
from faker.providers import BaseProvider

# Crear proveedor personalizado
class VideoGameProvider(BaseProvider):
    def video_game_genre(self):
        genres = ['RPG', 'FPS', 'Aventura', 'Estrategia', 'Deportes', 
                  'Simulación', 'Puzzle', 'Plataformas']
        return self.random_element(genres)
    
    def video_game_platform(self):
        platforms = ['PC', 'PlayStation 5', 'Xbox Series X', 'Nintendo Switch', 
                     'Steam Deck', 'Mobile']
        return self.random_element(platforms)
    
    def video_game_name(self):
        adjectives = ['Dark', 'Epic', 'Final', 'Eternal', 'Lost', 'Hidden']
        nouns = ['Kingdom', 'Quest', 'Legacy', 'Warriors', 'Legends', 'Chronicles']
        return f"{self.random_element(adjectives)} {self.random_element(nouns)}"

# Usar el proveedor
fake = Faker('es_ES')
fake.add_provider(VideoGameProvider)

for _ in range(5):
    print(f"{fake.video_game_name()} - {fake.video_game_genre()} ({fake.video_game_platform()})")

# Salida:
# Epic Quest - RPG (PlayStation 5)
# Dark Chronicles - FPS (PC)
# Final Warriors - Aventura (Nintendo Switch)
```

---

## 19. Unique (Valores Únicos)

```python
from faker import Faker

fake = Faker('es_ES')

# Generar emails únicos (sin repetición)
emails_unicos = [fake.unique.email() for _ in range(10)]
print(emails_unicos)

# Resetear el registro de únicos
fake.unique.clear()

# Generar nombres únicos
try:
    nombres = [fake.unique.first_name() for _ in range(1000)]
except Exception as e:
    print(f"Error: Se agotaron los nombres únicos disponibles")
```

---

## 20. Ejemplos Prácticos Completos

### Dataset de Tienda Online

```python
from faker import Faker
import csv
import random

fake = Faker('es_ES')

# Productos
categorias = ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Libros']

with open('productos.csv', 'w', newline='', encoding='utf-8') as f:
    campos = ['id', 'nombre', 'categoria', 'precio', 'stock', 'descripcion']
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    
    for i in range(200):
        writer.writerow({
            'id': i + 1,
            'nombre': fake.catch_phrase(),
            'categoria': fake.random_element(categorias),
            'precio': round(random.uniform(9.99, 999.99), 2),
            'stock': fake.random_int(min=0, max=500),
            'descripcion': fake.text(max_nb_chars=200)
        })

# Clientes
with open('clientes.csv', 'w', newline='', encoding='utf-8') as f:
    campos = ['id', 'nombre', 'email', 'telefono', 'direccion', 'ciudad', 'cp']
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    
    for i in range(500):
        writer.writerow({
            'id': i + 1,
            'nombre': fake.name(),
            'email': fake.unique.email(),
            'telefono': fake.phone_number(),
            'direccion': fake.street_address(),
            'ciudad': fake.city(),
            'cp': fake.postcode()
        })

# Pedidos
with open('pedidos.csv', 'w', newline='', encoding='utf-8') as f:
    campos = ['id', 'cliente_id', 'producto_id', 'cantidad', 'fecha', 'estado']
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    
    estados = ['Pendiente', 'Enviado', 'Entregado', 'Cancelado']
    
    for i in range(1000):
        writer.writerow({
            'id': i + 1,
            'cliente_id': fake.random_int(min=1, max=500),
            'producto_id': fake.random_int(min=1, max=200),
            'cantidad': fake.random_int(min=1, max=5),
            'fecha': fake.date_between(start_date='-1y', end_date='today'),
            'estado': fake.random_element(estados)
        })

print("Datasets de tienda online creados:")
print("- productos.csv (200 productos)")
print("- clientes.csv (500 clientes)")
print("- pedidos.csv (1000 pedidos)")
```

---

## 21. Resumen de Métodos Principales

| Categoría | Métodos |
| :--- | :--- |
| **Nombres** | `name()`, `first_name()`, `last_name()` |
| **Direcciones** | `address()`, `city()`, `street_address()`, `postcode()` |
| **Contacto** | `email()`, `phone_number()`, `url()` |
| **Fechas** | `date()`, `date_of_birth()`, `date_between()` |
| **Texto** | `text()`, `sentence()`, `paragraph()`, `word()` |
| **Números** | `random_int()`, `pyfloat()`, `boolean()` |
| **Finanzas** | `credit_card_number()`, `iban()`, `pricetag()` |
| **Empresa** | `company()`, `job()`, `catch_phrase()` |
| **Perfil** | `simple_profile()`, `profile()` |
| **Internet** | `user_name()`, `password()`, `ipv4()` |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
