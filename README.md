# 🧠 Código TFM — Quantum Data Encoding & Hybrid Autoencoders

## 🚨 PRIMERO DE TODO: METERSE EN LA RAMA `develop`

Sí, lo digo antes que nada porque luego vienen los disgustos.

---

## 📌 ¿Qué es esto?

Este repositorio contiene el código utilizado durante mi TFM:

> **Comparación de diferentes técnicas de data encoding para compresión y reconstrucción de imágenes mediante autoencoders híbridos.**

La idea general del proyecto era comparar diferentes técnicas de **quantum data encoding** dentro de un autoencoder híbrido, manteniendo el resto del experimento lo más controlado posible.

Las técnicas utilizadas fueron:

- **Amplitude Encoding**
- **Angle Encoding**
- **Dense Angle Encoding**
- **Basis Encoding**
- **Single Qubit Encoding**

Y sí, todo esto acabó convertido en una cantidad bastante considerable de archivos `.py`.

---

## ⚠️ Aviso importante antes de entrar

Espero sinceramente que **nadie tenga que trabajar nunca sobre este proyecto** XD.

No porque no funcione, sino porque la distribución del código es actualmente "complicada".

El código está bastante caótico, hay archivos de pruebas por todas partes y, para añadir más emoción, **prácticamente no hay comentarios**.  
Me hubiera gustado dejarlo bastante más limpio y organizado, pero sinceramente **no me ha dado la vida**. XD

Así que este README existe principalmente para:

1. Evitar que alguien tarde 3 días en encontrar el archivo que necesita.
2. Explicar, más o menos, qué hace cada cosa.
3. Dejar una pequeña guía para mi yo del futuro.
4. Permitir que alguien pueda coger "inspiración" de lo que hice en su momento.
5. Evitar que alguien abra `Circuito.py` y abandone inmediatamente la informática.

---

# 📂 ¿Dónde está lo importante?

La carpeta que realmente importa es:

```text
Terceras pruebas/
```

Sí, el nombre podría ser muchísimo mejor.

No lo es.

Dentro de esta carpeta hay una **barbaridad de archivos `.py`**. A continuación intento explicar qué hace cada uno.

Digo *intento* porque algunos de estos archivos los escribí hace bastante tiempo y ahora mismo mi memoria sobre ellos es aproximadamente:

> "Sé que esto era importante para algo."

---

# 🗂️ Archivos

## `Archivador.py`

Creo que no se utiliza actualmente.

Lo utilicé en su momento para gestionar carpetas y organizar archivos/resultados.

Probablemente se puede ignorar sin miedo.

---

## `Autoencoder.py`

Tampoco se utiliza en los experimentos finales.

Fue una de las primeras pruebas que hice para entender cómo funcionaba un autoencoder.

Básicamente:

> **Yo intentando entender qué estaba haciendo.**

No es necesario para reproducir los experimentos finales.

---

## `Circuito.py` ⭐

### Probablemente el archivo más importante de todo el proyecto.

Aquí está centralizada prácticamente toda la parte relacionada con los **circuitos cuánticos** y con las partes más puramente cuánticas del autoencoder y del encoding.

Al principio del archivo están las implementaciones de:

- **Dense Angle Encoding**
- **Single Qubit Encoding**

Estas dos técnicas no están disponibles directamente como técnicas de encoding predeterminadas en PennyLane, por lo que tuve que implementarlas.

Después aparecen otras funciones auxiliares. Algunas no se utilizan directamente en los flujos principales, pero pueden resultar interesantes. Por ejemplo:

- `SWAP_Test`
- otras funciones relacionadas con estados y circuitos cuánticos.

### ¿Cómo funciona el flujo principal?

Las funciones principales siguen, de forma general, este esquema:

```text
Imagen / bloque de píxeles
        ↓
Quantum Data Encoding
        ↓
Encoder
        ↓
Latent / Trash
        ↓
Decoder
        ↓
Medición
        ↓
Reconstrucción
```

Al comienzo de las funciones principales se indica mediante un parámetro qué **ansatz** se utiliza.

Actualmente el principal es:

```text
StronglyEntanglingLayers
```

Después, dependiendo de la función utilizada, se aplica la técnica de encoding correspondiente.

A continuación se ejecuta:

1. El **encoder**.
2. El **decoder**.
3. La medición correspondiente.
4. La reconstrucción del bloque.

Una de las cosas importantes de este archivo es tener claro **cómo evoluciona la información a través de los wires**.

Especialmente:

- qué qubits contienen la información útil,
- cuáles corresponden a la parte **latent**,
- cuáles corresponden a la parte **trash**,
- cómo se aplica el encoder,
- cómo se reconstruye la información mediante el decoder,
- y cómo se realiza finalmente la medición.

Esta parte es bastante compleja de explicar únicamente en un README.

Si alguien necesita entenderla en profundidad:

> **ChatGPT probablemente pueda ayudar con el código. XD**

---

# `Funciones_estado.py` ⭐

Otro de los archivos **muy importantes**.

Este archivo contiene todas las funciones relacionadas con la manipulación de los **estados cuánticos** durante el flujo del autoencoder, además de bastantes funciones auxiliares.

Podría considerarse una especie de **puente entre los scripts principales de ejecución y `Circuito.py`**.

Aquí aparecen funciones relacionadas con:

- Normalización de datos.
- Tratamiento de los bloques de píxeles.
- Procesamiento de los bloques antes del autoencoder.
- Procesamiento de los bloques después del autoencoder.
- Guardado de resultados.
- Generación de gráficas.
- Inicialización de los autoencoders.
- Funciones puente para ejecutar los autoencoders.
- Aplicación de las funciones de pérdida.
- Manipulación de estados cuánticos.
- Otras funciones auxiliares necesarias durante el entrenamiento.

Hay **muchísimas funciones** y, a diferencia de otros archivos, la mayoría sí tienen alguna función importante dentro del flujo.

Por eso:

> ⚠️ **Si vas a modificar algo del funcionamiento interno, este archivo merece ser analizado con calma.**

No recomiendo abrirlo a las 3 de la mañana intentando entenderlo.

---

# `Generador_datos.py`

Se utiliza para generar y preparar los datasets necesarios antes de comenzar las ejecuciones.

Entre otras cosas, realiza las divisiones y organización de los datos necesarias para poder utilizarlos posteriormente en los circuitos.

No es especialmente necesario analizarlo para entender el funcionamiento del autoencoder.

Eso sí:

> Para utilizar correctamente un dataset dentro de los circuitos cuánticos, este archivo forma parte del flujo de preparación de datos.

---

# `Graficador.py`

Lo utilicé en su momento para generar algún gráfico.

Actualmente es **descartable**.

Si alguien lo necesita, ahí está.

Si no:

> 🗑️ *Que descanse.*

---

# `Grafico.py`

Este sí tiene una función concreta.

Se utiliza para generar el **gráfico de puntos que aparece en el artículo con los mejores resultados**.

Si estás buscando cómo se generó esa figura concreta del paper, probablemente este sea el archivo que quieres.

---

# `Inicializa_params.py`

Aquí se inicializan los diferentes parámetros utilizados durante los experimentos.

Principalmente cosas como:

- Número de capas.
- Número de qubits.
- Learning rates.
- Tamaño de los bloques.
- Parámetros de las técnicas de encoding.
- Parámetros de los optimizadores.
- Etc.

En general, este archivo sirve para darle **forma y estructura al proceso de optimización** y evitar tener todos estos valores desperdigados por los scripts.

Si quieres cambiar hiperparámetros de los experimentos:

> **Mira aquí primero.**

---

# 📓 El notebook

El notebook es...

> **un trabajo del máster.**

No hay mucho más que decir. JAJAJA.

Está ahí por motivos históricos y porque en algún momento pensé:

> "Voy a hacer esto en un notebook."

Y efectivamente lo hice.

---

# 🧪 `Main.py` y `main_prueba.py`

Aquí está la **sala de experimentos**.

Estos archivos contienen la ejecución de los diferentes autoencoders y permiten realizar los experimentos modificando los hiperparámetros.

Aquí se realizan las ejecuciones de las diferentes técnicas de encoding.

Para ejecutar las cinco técnicas hay que ejecutar las funciones correspondientes, modificando algunas variables según el experimento.

No es probablemente la parte más elegante del proyecto, pero es donde ocurre la magia.

O algo parecido a la magia.

---

# ⚛️ `mainAmplitude_DenseAutoencoder.py`

Este archivo contiene el flujo completo de ejecución del autoencoder para:

- **Amplitude Encoding**
- **Dense Angle Encoding**

Ambos están juntos porque utilizan el mismo número de qubits.

El flujo general es:

```text
Carga de datos
      ↓
Inicialización de parámetros
      ↓
Preparación de bloques 2×2
      ↓
Inicialización del circuito
      ↓
Entrenamiento
      ↓
Dev
      ↓
Test
      ↓
Guardado de resultados
```

Al principio se cargan los datos y se inicializan los parámetros relacionados con:

- los bloques de píxeles,
- el circuito cuántico,
- el entrenamiento,
- etc.

Después comienza el entrenamiento, procesando los diferentes bloques de imágenes.

Más adelante se realiza el proceso correspondiente al **dev set**.

Finalmente:

> **se ejecuta el test y se guardan los resultados necesarios.**

### ¿Cómo se diferencia Amplitude de Dense Angle?

Mediante un parámetro de entrada.

Dependiendo de su valor, se ejecuta una técnica u otra.

Sí, está implementado como un `string`.

No sé por qué.

---

# 📐 `mainAngle_BasisAutoencoder.py`

Exactamente la misma filosofía que el archivo anterior, pero esta vez para:

- **Angle Encoding**
- **Basis Encoding**

El flujo general es equivalente:

```text
Datos
 ↓
Parámetros
 ↓
Bloques
 ↓
Encoding
 ↓
Encoder
 ↓
Decoder
 ↓
Dev
 ↓
Test
 ↓
Resultados
```

---

# 1️⃣ `mainSQOriginal.py`

Lo mismo que los anteriores, pero exclusivamente para:

> **Single Qubit Encoding**

Es decir, un script separado porque aquí el número de qubits y el flujo correspondiente son diferentes.

---

# 🧵 `Paralelizador.py`

Este archivo fue un intento de ejecutar **dos autoencoders en paralelo**.

La idea original fue de **Hodei**.

Que conste.

La culpa fue suya. XD

No funcionaba del todo bien, pero el código puede servir como una **base interesante para intentar paralelizar las ejecuciones** en el futuro.

Si alguien quiere intentar mejorar el tiempo de ejecución:

> Igual merece la pena echarle un ojo.

---

# 🌊 `phase-field-compr` y `read_sim_data_for_training.py`

Estos archivos están relacionados con los datos de **phase field** de Eider.

Los utilicé para hacer una pequeña prueba con ese dataset.

La prueba funciona, por lo que estos archivos pueden servir como punto de partida si en algún momento se quieren utilizar esos datos de nuevo.

---

# 🧹 ¿Y el resto de archivos?

Aquí viene la parte fácil.

Si un archivo **no aparece explicado en este README**, probablemente sea:

- una prueba inicial,
- un experimento que hice y luego abandoné,
- una versión antigua de algo,
- un script auxiliar que ya no utilizo,
- o simplemente resultados que guardé en local.

En principio:

> **Se puede descartar todo eso.**

No debería ser necesario para reproducir los experimentos principales.

---

# 🧭 Resumen rápido para no perderse

Si acabas de entrar al proyecto y no sabes por dónde empezar:

```text
                 ┌──────────────────┐
                 │ Generador_datos  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │Inicializa_params │
                 └────────┬─────────┘
                          ↓
             ┌──────────────────────────┐
             │       MAIN / MAINs       │
             │   Sala de experimentos   │
             └────────────┬─────────────┘
                          ↓
              ┌────────────────────────┐
              │   Funciones_estado.py  │
              │    Puente principal    │
              └───────────┬────────────┘
                          ↓
                ┌──────────────────┐
                │    Circuito.py   │
                │  Parte cuántica  │
                └────────┬─────────┘
                         ↓
                  ┌─────────────┐
                  │ Autoencoder │
                  └─────────────┘
```

Y si solo quieres saber **qué archivos mirar**, sería aproximadamente:

### ⭐ MUY IMPORTANTES

- `Circuito.py`
- `Funciones_estado.py`
- `Inicializa_params.py`
- `Generador_datos.py`
- `mainAmplitude_DenseAutoencoder.py`
- `mainAngle_BasisAutoencoder.py`
- `mainSQOriginal.py`

### 🟡 ÚTILES

- `Main.py`
- `main_prueba.py`
- `Grafico.py`
- `Paralelizador.py`

### 🗑️ PROBABLEMENTE DESCARTABLES

- `Archivador.py`
- `Autoencoder.py`
- `Graficador.py`
- El resto de scripts de pruebas/resultados antiguos.

---

# ❤️ Últimas palabras

Si has llegado hasta aquí:

**enhorabuena.**

Ya sabes aproximadamente dónde está cada cosa y, con suerte, te he ahorrado bastante tiempo buscando archivos.

El código no está precisamente en su estado más bonito, pero **es el código que se utilizó para sacar adelante el TFM y realizar los experimentos del artículo**.

Así que si alguien quiere reutilizarlo, modificarlo o simplemente cotillearlo:

> **Buena suerte.**

Y si encuentras algo que no tiene ningún sentido:

> **probablemente tampoco lo tenía cuando lo escribí.**

Pero si consigues sacar algo útil de él, entonces todo este caos habrá servido para algo. 😎

**PD:** Si vas a tocar algo importante, hazlo en `develop`.

En serio.

# PRIMERO DE TODO: METERSE EN `develop`.
