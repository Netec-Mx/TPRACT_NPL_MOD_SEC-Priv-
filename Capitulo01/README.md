# Construcción de un sistema para analizar y detectar solicitudes bancarias duplicadas

## Metadatos


| Campo           | Valor                                           |
| --------------- | ----------------------------------------------- |
| **Audiencia**   | Usuario de negocio                              |
| **Duración**    | 105 minutos                                     |
| **Complejidad** | Guiada                                          |
| **Software**    | Ya instalado. En este taller no se instala nada |


## Qué vas a hacer

Vas a construir un sistema que compara dos solicitudes bancarias y dice si piden lo mismo. Tú preparas los ejemplos, limpias el texto, separas los datos, entrenas el modelo y lo pruebas con frases nuevas.

El software ya está en la máquina. Tú no instalas programas. Copias cada bloque de Python en el notebook, en orden, y lo ejecutas con `Shift+Enter`.

Una pareja es duplicada cuando las dos frases salen del mismo grupo: la misma solicitud, escrita de otra forma. Dos temas parecidos de grupos distintos cuentan como no duplicados.

## Prerrequisitos

Esto ya lo dejó listo el facilitador. Los pasos de preparación están en [Prerequisitos.md](Prerequisitos.md). Si falta alguno, avísale. No lo instales durante el taller.


| Requisito               | Qué debes encontrar                                                                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Equipo                  | Windows Server con escritorio                                                                                                                                  |
| Editor                  | Visual Studio Code, con las extensiones Python y Jupyter                                                                                                       |
| Carpeta                 | `C:\Users\<tu_usuario>\nlp_banking_lab`                                                                                                                        |
| Kernel del notebook     | **Python (nlp_banking_lab)**                                                                                                                                   |
| Python                  | 3.10, 64 bits, dentro de la carpeta `nlp_env`                                                                                                                  |
| Librerías ya instaladas | TensorFlow 2.15.0, Gast 0.5.4, NumPy 1.26.4, pandas 2.2.1, scikit-learn 1.4.1.post1, NLTK 3.8.1, Gensim 4.3.2, SciPy 1.12.0, Matplotlib 3.8.3, Unidecode 1.3.8 |
| Internet                | Solo la primera vez, para bajar la lista de palabras vacías del español                                                                                        |


Conocimientos que ayudan: leer una tabla y cambiar un texto entre comillas. No hace falta haber entrenado una red neuronal antes.

## Abrir el notebook

1. Abre Visual Studio Code.
2. **Archivo > Abrir carpeta** y elige `C:\Users\<tu_usuario>\nlp_banking_lab`.
3. Confía en la carpeta si Visual Studio Code lo pregunta.
4. Entra a `notebooks`. Crea un notebook llamado `siamese_banking_duplicates.ipynb` (**Archivo > Nuevo archivo**, elige notebook).
5. Arriba a la derecha elige el kernel **Python (nlp_banking_lab)**.
6. Copia cada bloque gris de esta guía en una celda nueva, en el orden en que aparece, y ejecútalo.

Si el kernel no aparece, avisa al facilitador. No cambies de intérprete.

Cuando una celda termine, sigue con la siguiente. Si una celda falla, no sigas: vuelve a ejecutar desde la celda de arranque.

---

## Celda de arranque

Enciende las herramientas y fija la carpeta de trabajo. No cambies esta celda.

```python
import json
import os
import random
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from unidecode import unidecode

tf.get_logger().setLevel("ERROR")

SEMILLA = 42
os.environ["PYTHONHASHSEED"] = str(SEMILLA)
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

CARPETA = os.path.join(os.path.expanduser("~"), "nlp_banking_lab")
CARPETA_DATOS = os.path.join(CARPETA, "data", "raw")
CARPETA_PROCESADOS = os.path.join(CARPETA, "data", "processed")
CARPETA_MODELOS = os.path.join(CARPETA, "models")
CARPETA_VECTORES = os.path.join(CARPETA, "embeddings")
CARPETA_NOTEBOOKS = os.path.join(CARPETA, "notebooks")
CARPETA_NLTK = os.path.join(CARPETA, "nltk_data")

for ruta in (CARPETA_DATOS, CARPETA_PROCESADOS, CARPETA_MODELOS, CARPETA_VECTORES, CARPETA_NOTEBOOKS, CARPETA_NLTK):
    os.makedirs(ruta, exist_ok=True)

print("Python listo")
print("TensorFlow", tf.__version__)
print("Carpeta:", CARPETA)
if not tf.__version__.startswith("2.15"):
    print("Avisa al facilitador: este kernel no es el del laboratorio.")
```

Al importar TensorFlow puede salir un aviso de deprecación. Puedes seguir.

---

## Paso 1. Crear las solicitudes de ejemplo

### Qué haces

Armas ejemplos de solicitudes. Cada grupo es una misma petición escrita varias veces: el texto original y dos, tres o cuatro variantes.

### Código

```python
plantillas = {
    "credito": [
        "Solicito información sobre créditos de libre inversión disponibles",
        "Necesito un préstamo personal para libre inversión",
        "Quiero conocer las tasas de interés para créditos personales",
        "Requiero información sobre líneas de crédito disponibles",
        "Me interesa solicitar un crédito de consumo",
        "Deseo aplicar a un préstamo para gastos personales",
        "Quisiera saber los requisitos para un crédito hipotecario",
        "Necesito financiamiento para compra de vivienda",
        "Solicito aprobación de crédito para adquisición de inmueble",
        "Requiero un préstamo para remodelación de mi casa",
    ],
    "reclamo": [
        "Presento reclamo por cobro indebido en mi tarjeta de crédito",
        "Me cobraron una comisión que no corresponde en mi cuenta",
        "Quiero reportar un cargo no reconocido en mi estado de cuenta",
        "Reclamo por débito automático no autorizado en mi cuenta corriente",
        "Presento queja por cobros duplicados en mi tarjeta",
        "Me descontaron un monto que no reconozco de mi cuenta de ahorros",
        "Solicito reversión de cargo no autorizado",
        "Reclamo porque me cobraron intereses de mora sin justificación",
        "Presento inconformidad por comisiones excesivas",
        "Quiero reportar un fraude en mi tarjeta de débito",
    ],
    "consulta_saldo": [
        "Necesito conocer el saldo actual de mi cuenta de ahorros",
        "Quiero consultar el estado de mi cuenta corriente",
        "Solicito información sobre los movimientos de mi cuenta",
        "Requiero un extracto bancario de los últimos tres meses",
        "Deseo saber cuánto tengo disponible en mi cuenta",
        "Necesito verificar mi saldo disponible",
        "Quiero revisar los últimos movimientos de mi tarjeta",
        "Solicito el estado actual de mis productos financieros",
        "Requiero información del saldo de mi CDT",
        "Necesito consultar mi cupo disponible de tarjeta de crédito",
    ],
    "cancelacion": [
        "Deseo cancelar mi tarjeta de crédito por no uso",
        "Solicito la cancelación de mi cuenta de ahorros",
        "Quiero dar de baja mi tarjeta débito",
        "Requiero cerrar mi cuenta corriente",
        "Necesito cancelar un producto financiero que ya no utilizo",
        "Solicito el cierre definitivo de mi cuenta",
        "Deseo dar de baja todos mis productos con el banco",
        "Quiero cancelar mi seguro asociado a la tarjeta",
        "Requiero la terminación de mi contrato de cuenta corriente",
        "Necesito cancelar mi línea de sobregiro",
    ],
    "actualizacion_datos": [
        "Necesito actualizar mi número de teléfono celular",
        "Solicito cambio de dirección de correspondencia",
        "Quiero actualizar mi correo electrónico en el sistema",
        "Requiero modificar mis datos de contacto",
        "Necesito cambiar el número de celular asociado a mi cuenta",
        "Solicito actualización de mi dirección de residencia",
        "Deseo cambiar mi correo para recibir extractos digitales",
        "Requiero actualizar mi información personal en el banco",
        "Necesito registrar un nuevo número telefónico",
        "Quiero modificar la dirección de envío de mi tarjeta",
    ],
}

CAMBIOS = (
    ("Solicito", "Pido"),
    ("Necesito", "Requiero"),
    ("Quiero", "Deseo"),
    ("Deseo", "Quisiera"),
    ("Presento", "Interpongo"),
    ("información", "info"),
)


def variar(texto):
    nuevo = texto
    opciones = [(a, b) for a, b in CAMBIOS if a in nuevo]
    if opciones:
        a, b = random.choice(opciones)
        nuevo = nuevo.replace(a, b, 1)
    if random.choice((True, False)):
        nuevo = "Hola, " + nuevo[0].lower() + nuevo[1:]
    else:
        nuevo = nuevo + " por favor"
    if nuevo == texto:
        nuevo = texto + " gracias"
    return nuevo


filas = []
numero = 0
grupo = 0
for categoria, textos in plantillas.items():
    for texto in textos:
        grupo += 1
        numero += 1
        filas.append({"id": numero, "solicitud": texto, "categoria": categoria, "grupo": grupo})
        ya_usados = {texto}
        for _ in range(random.randint(2, 4)):
            variante = variar(texto)
            while variante in ya_usados:
                variante += " gracias"
            ya_usados.add(variante)
            numero += 1
            filas.append({"id": numero, "solicitud": variante, "categoria": categoria, "grupo": grupo})

df = pd.DataFrame(filas).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
df.to_csv(os.path.join(CARPETA_DATOS, "solicitudes.csv"), index=False, encoding="utf-8-sig")

print(f"{len(df)} solicitudes en {df['grupo'].nunique()} grupos")
df.head(8)
```

### Qué debes ver

213 solicitudes y 50 grupos. La tabla muestra el texto, la categoría y el grupo. Dos filas con el mismo número de grupo son la misma petición.

---

## Paso 2. Mirar los datos

### Qué haces

Revisas cuántas solicitudes hay de cada tipo y qué tan largas son. Sirve para saber con qué va a aprender el modelo.

### Código

```python
%matplotlib inline

print(df["categoria"].value_counts())
print()
print("Solicitudes por grupo: mínimo", df.groupby("grupo").size().min(), ", máximo", df.groupby("grupo").size().max())

df["palabras"] = df["solicitud"].str.split().str.len()
print("Largo promedio:", round(df["palabras"].mean(), 1), "palabras")

df["categoria"].value_counts().plot(kind="bar", title="Solicitudes por categoría", color="steelblue")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_NOTEBOOKS, "categorias.png"), dpi=100)
plt.show()
```

### Qué debes ver

Cinco categorías, grupos de 3 a 5 solicitudes y un promedio de 9.5 palabras. El gráfico queda en `notebooks\categorias.png`.

---

## Paso 3. Limpiar el texto

### Qué haces

Dejas el texto en minúsculas, sin acentos ni signos, y quitas palabras que casi no aportan (`de`, `la`, `para`). Conservas números, `no`, `sin` y `mi`, porque en una solicitud bancaria sí cambian el sentido.

### Código

```python
from nltk.corpus import stopwords

nltk.data.path.append(CARPETA_NLTK)
nltk.download("stopwords", download_dir=CARPETA_NLTK, quiet=True)

vacias = set(stopwords.words("spanish")) - {"no", "sin", "contra", "sobre", "mi", "mis"}


def limpiar(texto):
    texto = unidecode(str(texto).lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return [p for p in texto.split() if p not in vacias and len(p) >= 2]


print(limpiar("Solicito información sobre créditos de libre inversión"))

df["tokens"] = df["solicitud"].apply(limpiar)
df["texto_limpio"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
df["n_tokens"] = df["tokens"].apply(len)

print("Palabras útiles por solicitud:", round(df["n_tokens"].mean(), 1))
df[["solicitud", "texto_limpio"]].head(5)
```

Guarda una copia que Excel puede abrir. En el notebook, `tokens` sigue siendo una lista.

```python
copia = df.copy()
copia["tokens"] = copia["texto_limpio"]
copia.to_csv(os.path.join(CARPETA_PROCESADOS, "solicitudes_limpias.csv"), index=False, encoding="utf-8-sig")
print("Archivo listo en data\\processed\\solicitudes_limpias.csv")
```

### Qué debes ver

`Solicito información sobre créditos...` queda como `solicito informacion sobre creditos libre inversion`. El promedio de palabras útiles queda en 6.9.

---

## Paso 4. Separar los grupos y representar las palabras

### Qué haces

Separas los grupos en tres cajas antes de enseñarle nada al modelo:

- **Entrenamiento:** ejemplos con los que aprende.
- **Validación:** solicitudes nuevas que se miran mientras aprende, para saber si está mejorando.
- **Prueba:** solicitudes que se reservan hasta el final. No entran al entrenamiento.

Después, cada palabra del entrenamiento se convierte en una lista de 100 números. Eso es Word2Vec. Una palabra que solo aparezca en validación o en prueba quedará como palabra desconocida.

### Código

```python
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

resumen = df.groupby("grupo")["categoria"].first().reset_index()
grupos_train, grupos_temp, _, categorias_temp = train_test_split(
    resumen["grupo"].to_numpy(),
    resumen["categoria"].to_numpy(),
    test_size=0.30,
    random_state=SEMILLA,
    stratify=resumen["categoria"].to_numpy(),
)
grupos_val, grupos_test = train_test_split(
    grupos_temp, test_size=0.50, random_state=SEMILLA, stratify=categorias_temp,
)

grupos_train = {int(g) for g in grupos_train}
grupos_val = {int(g) for g in grupos_val}
grupos_test = {int(g) for g in grupos_test}

df_train = df[df["grupo"].isin(grupos_train)].copy()
df_val = df[df["grupo"].isin(grupos_val)].copy()
df_test = df[df["grupo"].isin(grupos_test)].copy()

print(f"Grupos  entrenamiento / validación / prueba: {len(grupos_train)} / {len(grupos_val)} / {len(grupos_test)}")
print(f"Solicitudes entrenamiento / validación / prueba: {len(df_train)} / {len(df_val)} / {len(df_test)}")
```

```python
LARGO = max(8, int(df_train["n_tokens"].quantile(0.95)) + 2)
TAMANO_VECTOR = 100

w2v = Word2Vec(
    sentences=df_train["tokens"].tolist(),
    vector_size=TAMANO_VECTOR,
    window=5,
    min_count=1,
    workers=1,
    epochs=30,
    seed=SEMILLA,
    sg=1,
)
w2v.save(os.path.join(CARPETA_VECTORES, "word2vec.model"))
print(f"Cada frase se recorta a {LARGO} palabras")
print(f"El diccionario de entrenamiento tiene {len(w2v.wv)} palabras")

for palabra in ("tarjeta", "cancelar", "cuenta"):
    if palabra in w2v.wv:
        vecinas = ", ".join(p for p, _ in w2v.wv.most_similar(palabra, topn=3))
        print(f"  {palabra} se parece a: {vecinas}")
```

Con tan pocos textos, "se parece" significa que esas palabras aparecieron juntas en las plantillas. Aun así, dos redacciones de la misma solicitud comparten las mismas palabras y el modelo puede usarlas.

```python
vocabulario = sorted({p for tokens in df_train["tokens"] for p in tokens})
palabra_a_numero = {"<RELLENO>": 0, "<DESCONOCIDA>": 1}
numero_a_palabra = {0: "<RELLENO>", 1: "<DESCONOCIDA>"}
for i, palabra in enumerate(vocabulario, start=2):
    palabra_a_numero[palabra] = i
    numero_a_palabra[i] = palabra

TAMANO_VOCAB = len(palabra_a_numero)
matriz = np.zeros((TAMANO_VOCAB, TAMANO_VECTOR), dtype="float32")
filas_conocidas = []
for palabra, i in palabra_a_numero.items():
    if palabra in ("<RELLENO>", "<DESCONOCIDA>"):
        continue
    matriz[i] = w2v.wv[palabra]
    filas_conocidas.append(i)
matriz[1] = matriz[filas_conocidas].mean(axis=0)

np.save(os.path.join(CARPETA_VECTORES, "matriz.npy"), matriz)
with open(os.path.join(CARPETA_VECTORES, "vocabulario.json"), "w", encoding="utf-8") as archivo:
    json.dump(palabra_a_numero, archivo, ensure_ascii=False)

print(f"Matriz lista: {matriz.shape[0]} palabras, {matriz.shape[1]} números cada una")
print("La palabra desconocida usa el promedio de las palabras conocidas")
```

### Qué debes ver

35 grupos y 145 solicitudes para entrenar, 7 grupos para validación y 8 para prueba. Cada frase queda en 11 palabras. La matriz tiene 111 filas. La fila 0, el relleno, queda en ceros.

---

## Paso 5. Armar las parejas

### Qué haces

El modelo no lee una solicitud suelta. Lee dos y responde si son duplicadas.

- Pareja **positiva** (1): dos textos del mismo grupo.
- Pareja **negativa** (0): dos textos de grupos distintos, dentro de la misma caja.

Las parejas de prueba no comparten solicitudes con las de entrenamiento.

### Código

```python
from itertools import combinations

from tensorflow.keras.preprocessing.sequence import pad_sequences


def armar_parejas(datos, semilla=42):
    azar = np.random.RandomState(semilla)
    izquierda, derecha, etiquetas = [], [], []
    por_grupo = datos.groupby("grupo")["texto_limpio"].apply(list)

    for _, textos in por_grupo.items():
        for a, b in combinations(list(textos), 2):
            izquierda.append(a)
            derecha.append(b)
            etiquetas.append(1)

    positivos = len(etiquetas)
    ids = [int(g) for g in por_grupo.index]
    textos = {int(g): list(v) for g, v in por_grupo.items()}
    vistas = set()
    intentos = 0
    while etiquetas.count(0) < positivos and intentos < positivos * 50:
        intentos += 1
        g1, g2 = azar.choice(ids, size=2, replace=False)
        a = textos[int(g1)][azar.randint(0, len(textos[int(g1)]))]
        b = textos[int(g2)][azar.randint(0, len(textos[int(g2)]))]
        if (a, b) in vistas:
            continue
        vistas.add((a, b))
        izquierda.append(a)
        derecha.append(b)
        etiquetas.append(0)
    return izquierda, derecha, np.array(etiquetas)


def a_numeros(texto):
    desconocida = palabra_a_numero["<DESCONOCIDA>"]
    return [palabra_a_numero.get(p, desconocida) for p in str(texto).split()]


def a_matriz(frases):
    return pad_sequences([a_numeros(f) for f in frases], maxlen=LARGO, padding="post", value=0)


parejas = {}
for nombre, parte in (("train", df_train), ("val", df_val), ("test", df_test)):
    izq, der, y = armar_parejas(parte, SEMILLA)
    parejas[nombre] = {"izq": izq, "der": der, "y": y}
    print(f"{nombre}: {len(y)} parejas, mitad duplicadas y mitad no")

X_izq_train = a_matriz(parejas["train"]["izq"])
X_der_train = a_matriz(parejas["train"]["der"])
X_izq_val = a_matriz(parejas["val"]["izq"])
X_der_val = a_matriz(parejas["val"]["der"])
X_izq_test = a_matriz(parejas["test"]["izq"])
X_der_test = a_matriz(parejas["test"]["der"])
y_train = parejas["train"]["y"]
y_val = parejas["val"]["y"]
y_test = parejas["test"]["y"]

print("Forma de una caja de entrenamiento:", X_izq_train.shape)
```

### Qué debes ver

480 parejas de entrenamiento, 126 de validación y 124 de prueba. En cada caja hay tantos duplicados como no duplicados. La forma termina en 11.

---

## Paso 6. Armar la red

### Qué haces

La red lee las dos solicitudes con el **mismo** lector de texto. Compara los dos resultados y devuelve una probabilidad de 0 a 1. Cerca de 1 significa "parece duplicado".

Los números de las palabras quedan fijos. Con pocos ejemplos, dejar que el modelo los cambie borra lo que Word2Vec acaba de aprender.

### Código

```python
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model

tf.random.set_seed(SEMILLA)


@tf.keras.utils.register_keras_serializable(package="laboratorio")
class Distancia(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        return tf.abs(tf.subtract(inputs[0], inputs[1]))


def crear_modelo():
    entrada_1 = Input(shape=(LARGO,))
    entrada_2 = Input(shape=(LARGO,))
    palabras = Embedding(TAMANO_VOCAB, TAMANO_VECTOR, weights=[matriz], trainable=False, mask_zero=True)
    lector = LSTM(64)
    diferencia = Distancia()([lector(palabras(entrada_1)), lector(palabras(entrada_2))])
    capa = Dense(32, activation="relu")(diferencia)
    salida = Dense(1, activation="sigmoid")(capa)
    modelo = Model([entrada_1, entrada_2], salida)
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def cargar_modelo(ruta):
    return tf.keras.models.load_model(ruta, custom_objects={"Distancia": Distancia})


modelo = crear_modelo()
modelo.summary()
```

### Qué debes ver

Un resumen con dos entradas, una capa de palabras compartida, un LSTM compartido y una salida de un solo número. La capa de palabras figura como no entrenable.

---

## Paso 7. Entrenar

### Qué haces

Ejecutas el entrenamiento. El modelo solo estudia las parejas de entrenamiento. La validación sirve para detenerse cuando deja de mejorar. La prueba sigue guardada.

La primera vuelta tarda más. Las siguientes suelen durar un segundo. Si la validación deja de mejorar, el proceso se detiene solo. Si sigue mejorando, completa las 40 vueltas.

### Código

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

ruta_mejor = os.path.join(CARPETA_MODELOS, "mejor_modelo.keras")
history = modelo.fit(
    [X_izq_train, X_der_train],
    y_train,
    validation_data=([X_izq_val, X_der_val], y_val),
    epochs=40,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(ruta_mejor, monitor="val_loss", save_best_only=True),
    ],
)

modelo.save(os.path.join(CARPETA_MODELOS, "modelo_final.keras"))
pd.DataFrame(history.history).to_csv(
    os.path.join(CARPETA_MODELOS, "historial.csv"), index=False, encoding="utf-8-sig"
)

print("Vueltas realizadas:", len(history.history["loss"]))
print("Mejor acierto en validación:", round(max(history.history["val_accuracy"]), 3))
```

```python
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación (solicitudes nuevas)")
plt.title("¿Está aprendiendo o solo memoriza?")
plt.xlabel("Vuelta")
plt.ylabel("Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_NOTEBOOKS, "entrenamiento.png"), dpi=100)
plt.show()
```

### Cómo leer la gráfica

La línea de entrenamiento es el error con los ejemplos que ya vio. La de validación es el error con solicitudes de otros grupos. Si la primera baja y la segunda se queda quieta o sube, el modelo memorizó el taller.

### Qué debes ver

Con esta semilla el mejor acierto de validación queda cerca de 0.90 y el archivo `models\mejor_modelo.keras` queda guardado. Si el número baja de 0.60, avisa al facilitador antes de seguir.

---

## Paso 8. Probar con solicitudes que no vio

### Qué haces

Mides el modelo solo con la caja de prueba. Al lado calculas una comparación simple: si las dos frases comparten al menos la mitad de sus palabras, se marcan como duplicadas. Esa comparación dice cuánto se resuelve mirando las palabras, sin red neuronal.

Luego escribes solicitudes nuevas y cambias el umbral. El umbral es tu decisión: a partir de qué probabilidad el banco trata el par como duplicado.

### Código

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def palabras_compartidas(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0


base = np.array([
    1 if palabras_compartidas(a, b) >= 0.5 else 0
    for a, b in zip(parejas["test"]["izq"], parejas["test"]["der"])
])

mejor = cargar_modelo(ruta_mejor)
probabilidades = mejor.predict([X_izq_test, X_der_test], verbose=0).flatten()
UMBRAL = 0.5
predichos = (probabilidades >= UMBRAL).astype(int)

print("Comparando palabras (umbral 0.5 de palabras compartidas)")
print("  Acierto:", round(accuracy_score(y_test, base), 3))
print()
print("Red neuronal, en solicitudes de grupos nuevos")
print("  Acierto:   ", round(accuracy_score(y_test, predichos), 3))
print("  Precisión: ", round(precision_score(y_test, predichos, zero_division=0), 3))
print("  Cobertura: ", round(recall_score(y_test, predichos, zero_division=0), 3))
print("  F1:        ", round(f1_score(y_test, predichos, zero_division=0), 3))
print("  ROC-AUC:   ", round(roc_auc_score(y_test, probabilidades), 3))
print()
print(pd.crosstab(
    pd.Series(y_test, name="Real"),
    pd.Series(predichos, name="Predicho"),
))
```

Así se leen esas cifras, de cada 100 casos:

- **Acierto:** en cuántos la respuesta fue la correcta.
- **Precisión:** de los que marcó como duplicados, cuántos lo eran. Si es baja, el banco mezcla solicitudes distintas.
- **Cobertura:** de los duplicados reales, cuántos encontró. Si es baja, se cuela trabajo repetido.
- **ROC-AUC:** qué tan bien separa duplicados de no duplicados, sin depender del 0.5.

```python
def probabilidad(texto_1, texto_2, umbral=0.5):
    limpio_1 = " ".join(limpiar(texto_1))
    limpio_2 = " ".join(limpiar(texto_2))
    entrada = [a_matriz([limpio_1]), a_matriz([limpio_2])]
    valor = float(mejor.predict(entrada, verbose=0)[0][0])
    marca = "DUPLICADO" if valor >= umbral else "NO DUPLICADO"
    print(f"{valor:.2f}  {marca}   | {texto_1}")
    print(f"           | {texto_2}")
    print()
    return valor


print("Umbral 0.5")
probabilidad("Deseo cancelar mi tarjeta de crédito por no uso", "Quisiera cancelar mi tarjeta de crédito por no uso, por favor")
probabilidad("Necesito conocer el saldo actual de mi cuenta de ahorros", "Requiero conocer el saldo actual de mi cuenta de ahorros por favor")
probabilidad("Deseo cancelar mi tarjeta de crédito por no uso", "Necesito financiamiento para compra de vivienda")
probabilidad("Necesito conocer el saldo actual de mi cuenta de ahorros", "Solicito la cancelación de mi cuenta de ahorros")
```

La celda siguiente repite pares con el umbral en 0.9. Compara qué marca cambia.

```python
print("Mismos pares, umbral 0.9")
probabilidad("Deseo cancelar mi tarjeta de crédito por no uso", "Quisiera cancelar mi tarjeta de crédito por no uso, por favor", umbral=0.9)
probabilidad("Necesito conocer el saldo actual de mi cuenta de ahorros", "Requiero conocer el saldo actual de mi cuenta de ahorros por favor", umbral=0.9)
probabilidad("Necesito conocer el saldo actual de mi cuenta de ahorros", "Solicito la cancelación de mi cuenta de ahorros", umbral=0.9)
```

### Qué debes ver

Con esta semilla, comparar palabras acierta 1.00 y la red acierta 0.976 en la prueba. Anota los dos números. En este taller las variantes comparten casi todo el texto, por eso la comparación de palabras sale tan alta.

---

## Paso 9. Una palabra que el modelo no conoce

### Qué haces

Escribes una solicitud con una palabra que no estaba en los ejemplos de entrenamiento. El sistema no adivina un sinónimo: la guarda como `<DESCONOCIDA>` y sigue respondiendo una probabilidad.

### Código

```python
palabra_nueva = "refinanciamiento"
print("Número de la palabra nueva:", a_numeros(palabra_nueva))
print("1 significa desconocida")

probabilidad(
    "Solicito refinanciamiento hipotecario urgente",
    "Necesito reestructurar mi crédito de vivienda",
)
```

### Qué debes ver

La lista `[1]`. La probabilidad sale entre 0 y 1 aunque el modelo no haya visto esa palabra.

---

## Paso 10. Cierre

### Qué haces

Reúnes los archivos y respondes tres preguntas en una celda de texto del notebook (botón **+ Texto**, o celda Markdown).

### Código

```python
acierto = accuracy_score(y_test, predichos)
acierto_palabras = accuracy_score(y_test, base)
print("Solicitudes:", len(df))
print("Grupos entrenamiento / validación / prueba:", len(grupos_train), "/", len(grupos_val), "/", len(grupos_test))
print("Parejas entrenamiento / validación / prueba:", len(y_train), "/", len(y_val), "/", len(y_test))
print("Acierto comparando palabras:", round(acierto_palabras, 3))
print("Acierto de la red en la prueba:", round(acierto, 3))
print()
for ruta in (
    os.path.join(CARPETA_DATOS, "solicitudes.csv"),
    os.path.join(CARPETA_PROCESADOS, "solicitudes_limpias.csv"),
    os.path.join(CARPETA_VECTORES, "word2vec.model"),
    ruta_mejor,
):
    print("OK" if os.path.exists(ruta) else "FALTA", ruta)
```

Responde con tus números:

1. ¿La red acertó más, igual o menos que la comparación de palabras? ¿Qué parte del trabajo ya resolvía mirar las palabras repetidas?
2. Si mezclar dos solicitudes distintas le cuesta más al banco que revisar un duplicado a mano, ¿dejarías el umbral en 0.5 o lo subirías a 0.9? Mira si a 0.9 alguna pareja que sí era la misma solicitud dejó de marcarse.
3. ¿Qué pasó con la probabilidad cuando escribiste "refinanciamiento", una palabra que no estaba en el entrenamiento?

### Qué debes ver

Los cuatro archivos en `OK` y el acierto de la red por encima de 0.60.

---

## Archivos que quedan


| Archivo                | Dónde                                        | Para qué                                        |
| ---------------------- | -------------------------------------------- | ----------------------------------------------- |
| Solicitudes originales | `data\raw\solicitudes.csv`                   | Los textos tal como se generaron                |
| Solicitudes limpias    | `data\processed\solicitudes_limpias.csv`     | Texto ya normalizado                            |
| Word2Vec               | `embeddings\word2vec.model`                  | Números de cada palabra de entrenamiento        |
| Mejor modelo           | `models\mejor_modelo.keras`                  | La red con menor error de validación            |
| Historial              | `models\historial.csv`                       | Error de cada vuelta                            |
| Gráficos               | `notebooks\*.png`                            | Categorías y curvas de entrenamiento            |
| Notebook               | `notebooks\siamese_banking_duplicates.ipynb` | Todo lo que ejecutaste, más las tres respuestas |


## Cómo se evalúa


| Qué se revisa        | Peso | Listo cuando                                                           |
| -------------------- | ---- | ---------------------------------------------------------------------- |
| Datos y limpieza     | 20%  | Hay solicitudes, grupos y texto limpio                                 |
| Separación de grupos | 20%  | Entrenamiento, validación y prueba no comparten grupos                 |
| Entrenamiento        | 25%  | Corriste el entrenamiento y leíste la gráfica de las dos líneas        |
| Prueba               | 20%  | Reportaste acierto, precisión, cobertura y la comparación por palabras |
| Decisión de negocio  | 15%  | Respondiste el umbral y el caso de la palabra desconocida              |


## Si algo falla


| Qué ves                                      | Qué hacer                                                                         |
| -------------------------------------------- | --------------------------------------------------------------------------------- |
| El kernel no es **Python (nlp_banking_lab)** | No sigas. Avisa al facilitador                                                    |
| `NameError`                                  | Esa celda usa un resultado anterior. Ejecuta desde la celda de arranque, en orden |
| Falla la descarga de palabras vacías         | Revisa el internet de la máquina o pide al facilitador la carpeta `nltk_data`     |
| Una celda de gráficos no dibuja              | Vuelve a ejecutarla. El archivo `.png` igual se guarda en `notebooks`             |


