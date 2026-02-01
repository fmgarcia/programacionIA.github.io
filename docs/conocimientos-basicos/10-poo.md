# 📚 Unidad 10. Programación Orientada a Objetos

La **Programación Orientada a Objetos (POO)** es un paradigma que organiza el código en torno a objetos que combinan datos (atributos) y comportamiento (métodos).

---

## 10.1. Conceptos Fundamentales

### ¿Qué es un Objeto?

Un **objeto** es una entidad que tiene:

*   **Atributos**: Características o datos (variables).
*   **Métodos**: Comportamientos o acciones (funciones).

![Clase vs Objeto](../assets/images/class_vs_object.svg)

**Ejemplo del mundo real:**

Un coche tiene:

*   Atributos: color, marca, velocidad, combustible
*   Métodos: arrancar(), acelerar(), frenar(), apagar()

### ¿Qué es una Clase?

Una **clase** es un molde o plantilla para crear objetos. Define qué atributos y métodos tendrán los objetos.

```
Clase: Coche (plantilla)
    ↓
Objetos: mi_coche, tu_coche, coche_rojo (instancias)
```

---

## 10.2. Crear Clases y Objetos

### Sintaxis Básica

```python
class Coche:
    """Clase que representa un coche."""
    pass  # Clase vacía

# Crear objetos (instancias)
mi_coche = Coche()
tu_coche = Coche()

print(type(mi_coche))  # <class '__main__.Coche'>
```

### El Método `__init__`

El método `__init__` es el **constructor**. Se ejecuta automáticamente al crear un objeto.

```python
class Coche:
    def __init__(self, marca, modelo, año):
        self.marca = marca     # Atributo de instancia
        self.modelo = modelo
        self.año = año

# Crear objetos
mi_coche = Coche("Toyota", "Corolla", 2020)
tu_coche = Coche("Honda", "Civic", 2022)

# Acceder a atributos
print(mi_coche.marca)   # Toyota
print(tu_coche.modelo)  # Civic
```

### ¿Qué es `self`?

`self` es una referencia al objeto actual. Permite acceder a sus atributos y métodos.

```python
class Persona:
    def __init__(self, nombre):
        self.nombre = nombre  # self.nombre es el atributo del objeto
    
    def saludar(self):
        print(f"Hola, soy {self.nombre}")  # Accede al atributo

ana = Persona("Ana")
ana.saludar()  # Hola, soy Ana
```

---

## 10.3. Atributos

### Atributos de Instancia

Pertenecen a cada objeto individual.

```python
class Estudiante:
    def __init__(self, nombre, edad):
        self.nombre = nombre  # Cada estudiante tiene su nombre
        self.edad = edad
        self.notas = []       # Lista vacía para cada estudiante

ana = Estudiante("Ana", 20)
luis = Estudiante("Luis", 22)

ana.notas.append(9)
print(ana.notas)   # [9]
print(luis.notas)  # [] (independiente)
```

### Atributos de Clase

Compartidos por todos los objetos de la clase.

```python
class Estudiante:
    # Atributo de clase
    escuela = "IES Python"
    total_estudiantes = 0
    
    def __init__(self, nombre):
        self.nombre = nombre  # Atributo de instancia
        Estudiante.total_estudiantes += 1

ana = Estudiante("Ana")
luis = Estudiante("Luis")

print(ana.escuela)              # IES Python
print(luis.escuela)             # IES Python
print(Estudiante.total_estudiantes)  # 2

# Modificar atributo de clase
Estudiante.escuela = "IES Nuevo"
print(ana.escuela)   # IES Nuevo
print(luis.escuela)  # IES Nuevo
```

---

## 10.4. Métodos

### Métodos de Instancia

Operan sobre un objeto específico usando `self`.

```python
class CuentaBancaria:
    def __init__(self, titular, saldo=0):
        self.titular = titular
        self.saldo = saldo
    
    def depositar(self, cantidad):
        if cantidad > 0:
            self.saldo += cantidad
            print(f"Depositados {cantidad}€. Saldo: {self.saldo}€")
    
    def retirar(self, cantidad):
        if cantidad > self.saldo:
            print("Saldo insuficiente")
        else:
            self.saldo -= cantidad
            print(f"Retirados {cantidad}€. Saldo: {self.saldo}€")
    
    def obtener_saldo(self):
        return self.saldo

# Uso
cuenta = CuentaBancaria("Ana", 1000)
cuenta.depositar(500)   # Depositados 500€. Saldo: 1500€
cuenta.retirar(200)     # Retirados 200€. Saldo: 1300€
print(cuenta.obtener_saldo())  # 1300
```

### Métodos de Clase

Operan sobre la clase, no sobre instancias. Usan `@classmethod` y `cls`.

```python
class Empleado:
    aumento_anual = 1.05  # 5%
    
    def __init__(self, nombre, salario):
        self.nombre = nombre
        self.salario = salario
    
    @classmethod
    def establecer_aumento(cls, porcentaje):
        cls.aumento_anual = 1 + porcentaje / 100
    
    @classmethod
    def desde_string(cls, cadena):
        """Constructor alternativo."""
        nombre, salario = cadena.split("-")
        return cls(nombre, float(salario))
    
    def aplicar_aumento(self):
        self.salario *= self.aumento_anual

# Uso
Empleado.establecer_aumento(10)  # 10% de aumento

emp1 = Empleado("Ana", 30000)
emp2 = Empleado.desde_string("Luis-35000")  # Constructor alternativo

emp1.aplicar_aumento()
print(emp1.salario)  # 33000
```

### Métodos Estáticos

No reciben `self` ni `cls`. Son funciones relacionadas con la clase.

```python
class Matematicas:
    @staticmethod
    def es_par(numero):
        return numero % 2 == 0
    
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * Matematicas.factorial(n - 1)
    
    @staticmethod
    def es_primo(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

# No necesitan instancia
print(Matematicas.es_par(4))      # True
print(Matematicas.factorial(5))   # 120
print(Matematicas.es_primo(17))   # True
```

---

## 10.5. Encapsulamiento

El **encapsulamiento** oculta los detalles internos y protege los datos.

### Convenciones de Nombres

```python
class Ejemplo:
    def __init__(self):
        self.publico = "Accesible"      # Público
        self._protegido = "Convención"  # Protegido (convención)
        self.__privado = "Oculto"       # Privado (name mangling)

obj = Ejemplo()
print(obj.publico)       # Accesible
print(obj._protegido)    # Accesible (por convención no debería)
# print(obj.__privado)   # AttributeError
print(obj._Ejemplo__privado)  # Accesible (name mangling)
```

### Propiedades (Getters y Setters)

```python
class Temperatura:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter para celsius."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, valor):
        """Setter para celsius."""
        if valor < -273.15:
            raise ValueError("Temperatura por debajo del cero absoluto")
        self._celsius = valor
    
    @property
    def fahrenheit(self):
        """Propiedad calculada (solo lectura)."""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, valor):
        self._celsius = (valor - 32) * 5/9

# Uso
temp = Temperatura(25)
print(temp.celsius)     # 25
print(temp.fahrenheit)  # 77.0

temp.celsius = 30
print(temp.fahrenheit)  # 86.0

temp.fahrenheit = 100
print(temp.celsius)     # 37.77...

# temp.celsius = -300  # ValueError
```

### Ejemplo Completo de Encapsulamiento

```python
class CuentaBancaria:
    def __init__(self, titular, saldo_inicial=0):
        self._titular = titular
        self.__saldo = saldo_inicial
        self.__historial = []
    
    @property
    def titular(self):
        return self._titular
    
    @property
    def saldo(self):
        return self.__saldo
    
    @property
    def historial(self):
        return self.__historial.copy()  # Devuelve copia
    
    def depositar(self, cantidad):
        if cantidad <= 0:
            raise ValueError("La cantidad debe ser positiva")
        self.__saldo += cantidad
        self.__historial.append(f"Depósito: +{cantidad}€")
    
    def retirar(self, cantidad):
        if cantidad <= 0:
            raise ValueError("La cantidad debe ser positiva")
        if cantidad > self.__saldo:
            raise ValueError("Saldo insuficiente")
        self.__saldo -= cantidad
        self.__historial.append(f"Retiro: -{cantidad}€")

# Uso
cuenta = CuentaBancaria("Ana", 1000)
cuenta.depositar(500)
cuenta.retirar(200)

print(cuenta.saldo)      # 1300
print(cuenta.historial)  # ['Depósito: +500€', 'Retiro: -200€']

# No se puede modificar directamente
# cuenta.saldo = 999999  # AttributeError
# cuenta.__saldo = 0     # No afecta al atributo real
```

---

## 10.6. Herencia

La **herencia** permite crear nuevas clases basadas en clases existentes.

![Herencia visualizada](../assets/images/inheritance_hierarchy.svg)

### Sintaxis Básica

```python
class Animal:
    def __init__(self, nombre):
        self.nombre = nombre
    
    def hablar(self):
        print("El animal hace un sonido")

class Perro(Animal):  # Perro hereda de Animal
    def hablar(self):
        print(f"{self.nombre} dice: ¡Guau!")

class Gato(Animal):
    def hablar(self):
        print(f"{self.nombre} dice: ¡Miau!")

# Uso
perro = Perro("Max")
gato = Gato("Luna")

perro.hablar()  # Max dice: ¡Guau!
gato.hablar()   # Luna dice: ¡Miau!

# Verificar herencia
print(isinstance(perro, Perro))   # True
print(isinstance(perro, Animal))  # True
print(issubclass(Perro, Animal))  # True
```

### Método `super()`

`super()` permite acceder a la clase padre.

```python
class Vehiculo:
    def __init__(self, marca, modelo):
        self.marca = marca
        self.modelo = modelo
    
    def info(self):
        return f"{self.marca} {self.modelo}"

class Coche(Vehiculo):
    def __init__(self, marca, modelo, puertas):
        super().__init__(marca, modelo)  # Llama al __init__ del padre
        self.puertas = puertas
    
    def info(self):
        return f"{super().info()} - {self.puertas} puertas"

coche = Coche("Toyota", "Corolla", 4)
print(coche.info())  # Toyota Corolla - 4 puertas
```

### Herencia Múltiple

Python permite heredar de múltiples clases.

```python
class Volador:
    def volar(self):
        print("Estoy volando")

class Nadador:
    def nadar(self):
        print("Estoy nadando")

class Pato(Volador, Nadador):
    def graznar(self):
        print("¡Cuac!")

pato = Pato()
pato.volar()    # Estoy volando
pato.nadar()    # Estoy nadando
pato.graznar()  # ¡Cuac!
```

### MRO (Method Resolution Order)

```python
class A:
    def metodo(self):
        print("A")

class B(A):
    def metodo(self):
        print("B")

class C(A):
    def metodo(self):
        print("C")

class D(B, C):
    pass

d = D()
d.metodo()  # B (busca en B antes que en C)

# Ver orden de resolución
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

---

## 10.7. Polimorfismo

El **polimorfismo** permite que objetos de diferentes clases respondan al mismo método de forma diferente.

![Polimorfismo Visual](../assets/images/polymorphism_shapes.svg)

```python
class Forma:
    def area(self):
        raise NotImplementedError("Subclases deben implementar este método")

class Rectangulo(Forma):
    def __init__(self, base, altura):
        self.base = base
        self.altura = altura
    
    def area(self):
        return self.base * self.altura

class Circulo(Forma):
    def __init__(self, radio):
        self.radio = radio
    
    def area(self):
        import math
        return math.pi * self.radio ** 2

class Triangulo(Forma):
    def __init__(self, base, altura):
        self.base = base
        self.altura = altura
    
    def area(self):
        return self.base * self.altura / 2

# Polimorfismo en acción
formas = [
    Rectangulo(4, 5),
    Circulo(3),
    Triangulo(4, 6)
]

for forma in formas:
    print(f"{type(forma).__name__}: área = {forma.area():.2f}")

# Rectangulo: área = 20.00
# Circulo: área = 28.27
# Triangulo: área = 12.00
```

### Duck Typing

"Si camina como un pato y hace cuac como un pato, es un pato."

```python
class Archivo:
    def leer(self):
        return "Contenido del archivo"

class BaseDatos:
    def leer(self):
        return "Datos de la base de datos"

class API:
    def leer(self):
        return "Respuesta de la API"

def procesar_datos(fuente):
    """No importa el tipo, solo que tenga método leer()."""
    datos = fuente.leer()
    print(f"Procesando: {datos}")

# Funciona con cualquier objeto que tenga leer()
procesar_datos(Archivo())
procesar_datos(BaseDatos())
procesar_datos(API())
```

---

## 10.8. Clases Abstractas

Las clases abstractas definen una interfaz que las subclases deben implementar.

```python
from abc import ABC, abstractmethod

class FiguraGeometrica(ABC):
    """Clase abstracta para figuras geométricas."""
    
    @abstractmethod
    def area(self):
        """Calcula el área de la figura."""
        pass
    
    @abstractmethod
    def perimetro(self):
        """Calcula el perímetro de la figura."""
        pass
    
    def descripcion(self):
        """Método concreto (no abstracto)."""
        return f"Soy una {self.__class__.__name__}"

class Cuadrado(FiguraGeometrica):
    def __init__(self, lado):
        self.lado = lado
    
    def area(self):
        return self.lado ** 2
    
    def perimetro(self):
        return 4 * self.lado

class Circulo(FiguraGeometrica):
    def __init__(self, radio):
        self.radio = radio
    
    def area(self):
        import math
        return math.pi * self.radio ** 2
    
    def perimetro(self):
        import math
        return 2 * math.pi * self.radio

# No se puede instanciar clase abstracta
# figura = FiguraGeometrica()  # TypeError

# Pero sí las subclases concretas
cuadrado = Cuadrado(5)
circulo = Circulo(3)

print(cuadrado.descripcion())  # Soy una Cuadrado
print(f"Área: {cuadrado.area()}")  # 25
print(f"Perímetro: {cuadrado.perimetro()}")  # 20
```

---

## 10.9. Métodos Mágicos (Dunder Methods)

Los métodos mágicos (double underscore) permiten personalizar el comportamiento de los objetos.

### Representación

```python
class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """Para print() y str()"""
        return f"Punto({self.x}, {self.y})"
    
    def __repr__(self):
        """Para representación técnica"""
        return f"Punto(x={self.x}, y={self.y})"

punto = Punto(3, 4)
print(punto)        # Punto(3, 4)
print(repr(punto))  # Punto(x=3, y=4)
```

### Operadores Aritméticos

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, otro):
        """Para + """
        return Vector(self.x + otro.x, self.y + otro.y)
    
    def __sub__(self, otro):
        """Para - """
        return Vector(self.x - otro.x, self.y - otro.y)
    
    def __mul__(self, escalar):
        """Para * (con escalar)"""
        return Vector(self.x * escalar, self.y * escalar)
    
    def __rmul__(self, escalar):
        """Para escalar * vector"""
        return self.__mul__(escalar)
    
    def __abs__(self):
        """Para abs()"""
        return (self.x**2 + self.y**2) ** 0.5
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4, 6)
print(v1 - v2)      # Vector(2, 2)
print(v1 * 2)       # Vector(6, 8)
print(3 * v1)       # Vector(9, 12)
print(abs(v1))      # 5.0
```

### Comparación

```python
class Estudiante:
    def __init__(self, nombre, nota):
        self.nombre = nombre
        self.nota = nota
    
    def __eq__(self, otro):
        """Para == """
        return self.nota == otro.nota
    
    def __lt__(self, otro):
        """Para < """
        return self.nota < otro.nota
    
    def __le__(self, otro):
        """Para <= """
        return self.nota <= otro.nota
    
    def __gt__(self, otro):
        """Para > """
        return self.nota > otro.nota
    
    def __str__(self):
        return f"{self.nombre}: {self.nota}"

estudiantes = [
    Estudiante("Ana", 8),
    Estudiante("Luis", 9),
    Estudiante("María", 7)
]

# sorted() usa __lt__
ordenados = sorted(estudiantes)
for e in ordenados:
    print(e)
# María: 7
# Ana: 8
# Luis: 9
```

### Contenedores

```python
class MiLista:
    def __init__(self, elementos=None):
        self._datos = elementos if elementos else []
    
    def __len__(self):
        """Para len()"""
        return len(self._datos)
    
    def __getitem__(self, indice):
        """Para [] lectura"""
        return self._datos[indice]
    
    def __setitem__(self, indice, valor):
        """Para [] escritura"""
        self._datos[indice] = valor
    
    def __contains__(self, item):
        """Para 'in'"""
        return item in self._datos
    
    def __iter__(self):
        """Para iteración"""
        return iter(self._datos)
    
    def append(self, item):
        self._datos.append(item)

lista = MiLista([1, 2, 3])
print(len(lista))      # 3
print(lista[0])        # 1
print(2 in lista)      # True

for item in lista:
    print(item)

lista[0] = 10
print(lista[0])  # 10
```

### Contexto (with)

```python
class GestorArchivo:
    def __init__(self, nombre, modo):
        self.nombre = nombre
        self.modo = modo
        self.archivo = None
    
    def __enter__(self):
        """Al entrar en 'with'"""
        print(f"Abriendo {self.nombre}")
        self.archivo = open(self.nombre, self.modo)
        return self.archivo
    
    def __exit__(self, tipo_exc, valor_exc, tb):
        """Al salir de 'with'"""
        print(f"Cerrando {self.nombre}")
        if self.archivo:
            self.archivo.close()
        return False  # No suprimir excepciones

# Uso
with GestorArchivo("test.txt", "w") as f:
    f.write("Hola mundo")
# Abriendo test.txt
# Cerrando test.txt
```

---

## 10.10. Composición vs Herencia

### Herencia ("es un")

```python
class Vehiculo:
    def mover(self):
        print("El vehículo se mueve")

class Coche(Vehiculo):  # Un coche ES UN vehículo
    pass
```

### Composición ("tiene un")

```python
class Motor:
    def arrancar(self):
        print("Motor arrancado")
    
    def detener(self):
        print("Motor detenido")

class Rueda:
    def rodar(self):
        print("La rueda rueda")

class Coche:  # Un coche TIENE UN motor y TIENE ruedas
    def __init__(self):
        self.motor = Motor()
        self.ruedas = [Rueda() for _ in range(4)]
    
    def arrancar(self):
        self.motor.arrancar()
    
    def conducir(self):
        for rueda in self.ruedas:
            rueda.rodar()

coche = Coche()
coche.arrancar()
coche.conducir()
```

### Cuándo Usar Cada Una

*   **Herencia**: Cuando hay una relación clara "es un".
*   **Composición**: Cuando hay una relación "tiene un" o para mayor flexibilidad.

---

## 10.11. Ejercicio Completo: Sistema de Gestión

```python
from abc import ABC, abstractmethod
from datetime import datetime

class Persona(ABC):
    """Clase abstracta base para personas."""
    
    def __init__(self, nombre, email):
        self.nombre = nombre
        self.email = email
        self._fecha_registro = datetime.now()
    
    @abstractmethod
    def descripcion(self):
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.nombre}"

class Empleado(Persona):
    """Empleado de la empresa."""
    
    _contador = 0
    
    def __init__(self, nombre, email, puesto, salario):
        super().__init__(nombre, email)
        Empleado._contador += 1
        self.id = f"EMP{Empleado._contador:04d}"
        self.puesto = puesto
        self._salario = salario
        self.proyectos = []
    
    @property
    def salario(self):
        return self._salario
    
    @salario.setter
    def salario(self, valor):
        if valor < 0:
            raise ValueError("El salario no puede ser negativo")
        self._salario = valor
    
    def descripcion(self):
        return f"{self.nombre} - {self.puesto} ({self.id})"
    
    def asignar_proyecto(self, proyecto):
        self.proyectos.append(proyecto)
        proyecto.miembros.append(self)

class Cliente(Persona):
    """Cliente de la empresa."""
    
    _contador = 0
    
    def __init__(self, nombre, email, empresa):
        super().__init__(nombre, email)
        Cliente._contador += 1
        self.id = f"CLI{Cliente._contador:04d}"
        self.empresa = empresa
        self.pedidos = []
    
    def descripcion(self):
        return f"{self.nombre} de {self.empresa} ({self.id})"

class Proyecto:
    """Proyecto de la empresa."""
    
    def __init__(self, nombre, descripcion, presupuesto):
        self.nombre = nombre
        self.descripcion = descripcion
        self.presupuesto = presupuesto
        self.miembros = []
        self.tareas = []
        self._completado = False
    
    def agregar_tarea(self, tarea):
        self.tareas.append(tarea)
    
    @property
    def progreso(self):
        if not self.tareas:
            return 0
        completadas = sum(1 for t in self.tareas if t.completada)
        return (completadas / len(self.tareas)) * 100
    
    def __str__(self):
        return f"Proyecto: {self.nombre} ({self.progreso:.0f}% completado)"

class Tarea:
    """Tarea de un proyecto."""
    
    def __init__(self, titulo, descripcion):
        self.titulo = titulo
        self.descripcion = descripcion
        self.completada = False
    
    def completar(self):
        self.completada = True
    
    def __str__(self):
        estado = "✓" if self.completada else "○"
        return f"{estado} {self.titulo}"

# --- Uso del sistema ---

# Crear empleados
emp1 = Empleado("Ana García", "ana@empresa.com", "Desarrolladora Senior", 45000)
emp2 = Empleado("Luis Martínez", "luis@empresa.com", "Diseñador UX", 38000)

# Crear proyecto
proyecto = Proyecto("App Móvil", "Desarrollo de app para clientes", 50000)

# Asignar empleados
emp1.asignar_proyecto(proyecto)
emp2.asignar_proyecto(proyecto)

# Agregar tareas
proyecto.agregar_tarea(Tarea("Diseño de interfaz", "Crear mockups"))
proyecto.agregar_tarea(Tarea("Backend API", "Desarrollar endpoints"))
proyecto.agregar_tarea(Tarea("Testing", "Pruebas unitarias"))

# Completar tareas
proyecto.tareas[0].completar()
proyecto.tareas[1].completar()

# Mostrar información
print(emp1.descripcion())
print(proyecto)
for tarea in proyecto.tareas:
    print(f"  {tarea}")

# Crear cliente
cliente = Cliente("María López", "maria@cliente.com", "TechCorp")
print(cliente.descripcion())
```

---

## 10.12. Resumen de POO

| Concepto | Descripción |
| :--- | :--- |
| **Clase** | Plantilla para crear objetos |
| **Objeto** | Instancia de una clase |
| **Atributo** | Variable de un objeto |
| **Método** | Función de un objeto |
| **`__init__`** | Constructor |
| **`self`** | Referencia al objeto actual |
| **Encapsulamiento** | Ocultar datos internos |
| **Herencia** | Crear clases basadas en otras |
| **Polimorfismo** | Mismo método, diferente comportamiento |
| **Abstracción** | Definir interfaces |

---

📅 **Fecha de creación:** Enero 2026  
✍️ **Autor:** Fran García
